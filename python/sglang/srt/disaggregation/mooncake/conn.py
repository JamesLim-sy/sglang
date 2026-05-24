from __future__ import annotations

import concurrent.futures
import ctypes
import dataclasses
import logging
import os
import struct
import threading
import time
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import numpy.typing as npt
import requests
import zmq

from sglang.srt.disaggregation.base.conn import KVArgs, KVPoll
from sglang.srt.disaggregation.common.conn import (
    CommonKVBootstrapServer,
    CommonKVManager,
    CommonKVReceiver,
    CommonKVSender,
)
from sglang.srt.disaggregation.common.utils import (
    FastQueue,
    group_concurrent_contiguous,
)
from sglang.srt.disaggregation.mooncake.transfer_engine import MooncakeTransferEngine
from sglang.srt.disaggregation.mooncake.utils import (
    check_mooncake_custom_mem_pool_enabled,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import format_tcp_address, get_int_env_var, is_valid_ipv6_address

logger = logging.getLogger(__name__)


class KVTransferError(Exception):
    def __init__(self, bootstrap_room: int, failure_reason: str):
        super().__init__(failure_reason)
        self.bootstrap_room = bootstrap_room
        self.failure_reason = failure_reason

    def __str__(self):
        return f"KVTransferError(bootstrap_room={self.bootstrap_room}): {self.failure_reason}"


# Prefill internal
@dataclasses.dataclass
class TransferKVChunk:
    room: int
    prefill_kv_indices: npt.NDArray[np.int32]

    f"""
        当前 chunk 在完整 KV 序列中的索引范围（如 slice(0,50)），用于从
        TransferInfo.dst_kv_indices 中切出本 chunk 对应的目标槽位:
            chunked_dst_kv_indice = dst_kv_indices[index_slice].
        在 chunked prefill 下，整段 KV 分多块发送，每块的 index_slice
        指明「这块是整段里的哪一段」
    """
    index_slice: slice

    is_last: bool
    prefill_aux_index: Optional[int]
    state_indices: Optional[List[int]]


# transfer 传输时的快递单. 表达的 dst 地址
@dataclasses.dataclass
class TransferInfo:
    room: int  # 即 bootstrap room, mass level 的 request id.
    endpoint: str  # 代表 decode node 的 ip 地址.
    dst_port: int  # 代表 decode node 的 port 端口.
    mooncake_session_id: (
        str  # 含义: d_node 在 MC_Trans_Engine 层面的唯一连接标识符，近似 node_id
    )
    dst_kv_indices: npt.NDArray[np.int32]

    dst_aux_index: int
    dst_state_indices: List[int]
    required_dst_info_num: int
    is_dummy: bool

    @classmethod
    def from_zmq(cls, msg: List[bytes]):
        if msg[4] == b"" and msg[5] == b"":
            is_dummy = True
            dst_kv_indices = np.array([], dtype=np.int32)
            dst_aux_index = None
            dst_state_indices = []
        else:
            dst_kv_indices = np.frombuffer(msg[4], dtype=np.int32)
            dst_aux_index = int(msg[5].decode("ascii"))
            if msg[6] == b"":
                dst_state_indices = []
            else:
                dst_state_indices = list(np.frombuffer(msg[6], dtype=np.int32))
            is_dummy = False
        return cls(
            room=int(msg[0].decode("ascii")),
            endpoint=msg[1].decode("ascii"),
            dst_port=int(msg[2].decode("ascii")),
            mooncake_session_id=msg[3].decode("ascii"),
            dst_kv_indices=dst_kv_indices,
            dst_aux_index=dst_aux_index,
            dst_state_indices=dst_state_indices,
            required_dst_info_num=int(msg[7].decode("ascii")),
            is_dummy=is_dummy,
        )


"""
    decode: 启动时, decode node 向 prefill node 注册自己的 kvcache init addr 等信息.
    - KVArgsRegisterInfo 作为 decode node 发送给 prefill node 的初始化信息.
    - 使用 dataclasses 装饰器自动生成 __init__ 构造方法
"""


@dataclasses.dataclass
class KVArgsRegisterInfo:
    room: str
    endpoint: str  # 代表 decode node 的 ip 地址.
    dst_port: int  # 代表 decode node 的 port 端口.
    mooncake_session_id: (
        str  # 含义: d_node 在 MC_Trans_Engine 层面的唯一连接标识符，近似 node_id
    )
    dst_kv_ptrs: list[int]
    dst_aux_ptrs: list[int]
    dst_state_data_ptrs: list[int]
    dst_tp_rank: int  # 含义: decode node 在起 attention tp_group 中的 rank
    dst_attn_tp_size: int  # 含义: decode node 在其 attention tp_group 中的 size
    dst_kv_item_len: int

    @classmethod
    def from_zmq(cls, msg: List[bytes]):
        return cls(
            room=str(msg[0].decode("ascii")),
            endpoint=msg[1].decode("ascii"),
            dst_port=int(msg[2].decode("ascii")),
            mooncake_session_id=msg[3].decode("ascii"),
            dst_kv_ptrs=list(struct.unpack(f"{len(msg[4])//8}Q", msg[4])),
            dst_aux_ptrs=list(struct.unpack(f"{len(msg[5])//8}Q", msg[5])),
            dst_state_data_ptrs=list(struct.unpack(f"{len(msg[6])//8}Q", msg[6])),
            dst_tp_rank=int(msg[7].decode("ascii")),
            dst_attn_tp_size=int(msg[8].decode("ascii")),
            dst_kv_item_len=int(msg[9].decode("ascii")),
        )


class AuxDataCodec:
    """Handles serialization and deserialization of auxiliary data buffers"""

    @staticmethod
    def serialize_data_from_buffer(src_addr, data_length):
        """Serialize data from memory buffer to bytes"""
        buffer = (ctypes.c_byte * data_length).from_address(src_addr)
        return bytes(buffer)

    @staticmethod
    def deserialize_data_to_buffer(kv_args, buffer_index, aux_index, data):
        """Deserialize bytes into target memory buffer"""
        dst_aux_ptr = kv_args.aux_data_ptrs[buffer_index]
        item_len = kv_args.aux_item_lens[buffer_index]
        dst_addr = dst_aux_ptr + item_len * aux_index

        """
            创建一个指向目标内存地址的ctypes字节数组缓冲区
            用于将接收到的数据写入到指定的内存位置
        """
        buffer = (ctypes.c_byte * len(data)).from_address(dst_addr)
        buffer[:] = data
        return


class MooncakeKVManager(CommonKVManager):
    AUX_DATA_HEADER = b"AUX_DATA"

    def __init__(
        self,
        args: KVArgs,
        disaggregation_mode: DisaggregationMode,
        server_args: ServerArgs,
        is_mla_backend: Optional[bool] = False,
    ):
        super().__init__(args, disaggregation_mode, server_args, is_mla_backend)
        # 实例化 transfer engine 对象, 并构建 p2p 点对点传输的 "P2PHANDSHAKE".
        self.init_engine()
        # 将 KV cache 数据 buffer、auxiliary data buffer 和 state/extra pool data buffer 注册到 transfer engine 中, 供后续的 KV cache 和 auxiliary data 传输使用.
        self.register_buffer_to_engine()
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            self.start_prefill_thread()
            self.session_failures = defaultdict(int)
            self.failed_sessions = set()
            self.session_lock = threading.Lock()
            # Determine the number of threads to use for kv sender
            cpu_count = os.cpu_count()
            transfer_thread_pool_size = get_int_env_var(
                "SGLANG_DISAGGREGATION_THREAD_POOL_SIZE",
                min(max(4, int(0.5 * cpu_count) // 8), 12),
            )
            transfer_queue_size = get_int_env_var("SGLANG_DISAGGREGATION_QUEUE_SIZE", 4)
            self.transfer_queues: List[FastQueue] = [
                FastQueue() for _ in range(transfer_queue_size)
            ]
            assert transfer_thread_pool_size >= transfer_queue_size, (
                f"The environment variable SGLANG_DISAGGREGATION_THREAD_POOL_SIZE={transfer_thread_pool_size} must be "
                f"greater than or equal to SGLANG_DISAGGREGATION_QUEUE_SIZE={transfer_queue_size}."
            )
            self.executors = [
                concurrent.futures.ThreadPoolExecutor(
                    transfer_thread_pool_size // transfer_queue_size
                )
                for _ in range(transfer_queue_size)
            ]
            for queue, executor in zip(self.transfer_queues, self.executors):
                threading.Thread(
                    target=self.transfer_worker, args=(queue, executor), daemon=True
                ).start()
            # If a timeout happens on the prefill side, it means prefill instances
            # fail to receive the KV indices from the decode instance of this request.
            # These timeout requests should be aborted to release the tree cache.
            self.bootstrap_timeout = get_int_env_var(
                "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT", 300
            )

            self.enable_custom_mem_pool, self.custom_mem_pool_type = (
                check_mooncake_custom_mem_pool_enabled()
            )
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            self.heartbeat_failures = {}
            self.session_pool = defaultdict(requests.Session)
            self.session_pool_lock = threading.Lock()

            """
            Track the mapping between prefill node addresses and room IDs

            Context: In disaggregated inference, a "room" is a coordination unit that represents
            a specific request's KV cache transfer session between prefill and decode nodes.

            Example workflow:
            1. User sends request "What is AI?" -> assigned room_id=12345
            2. Prefill node processes prompt, generates KV cache for tokens ["What", "is", "AI", "?"]
            3. Room 12345 coordinates the transfer of this KV cache from prefill to decode node
            4. Decode node receives KV cache in room 12345, continues generation: "AI is..."
            5. Room 12345 tracks this entire request lifecycle until completion

            This tracker maps: prefill_node_address -> {room_id1, room_id2, ...}
            So we know which rooms (requests) each prefill node is handling
            """
            """
                - key: prefill_node_address, 可以粗暴理解成 prefill pod ip 地址.
                - val: set(room_id)
            """
            self.addr_to_rooms_tracker = defaultdict(set)

            """
                记录了每个 bootstrap_id 对应的 prefill_rank set, bootstrap_id 是 prefiill_node 与 decode node 之间
                针对 req 的链接号

                key : bootstrap_room
                val : set(prefill_rank)
            """
            self.prefill_response_tracker: Dict[int, Set[int]] = defaultdict(set)

            # Heartbeat interval should be at least 2 seconds
            self.heartbeat_interval = max(
                float(os.getenv("SGLANG_DISAGGREGATION_HEARTBEAT_INTERVAL", 5.0)), 2.0
            )
            # Heartbeat failure should be at least 1
            self.max_failures = max(
                get_int_env_var("SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE", 2), 1
            )
            self.start_decode_thread()

            # If a timeout happens on the decode side, it means decode instances
            # fail to receive the KV Cache transfer done signal after bootstrapping.
            # These timeout requests should be aborted to release the tree cache.
            self.waiting_timeout = get_int_env_var(
                "SGLANG_DISAGGREGATION_WAITING_TIMEOUT", 300
            )

        self.failure_records: Dict[int, str] = {}
        self.failure_lock = threading.Lock()

    def init_engine(self):
        self.engine = MooncakeTransferEngine(
            hostname=self.local_ip,  # gpu rank 进程自己找到的机器 ip <-> get_local_ip_auto()
            gpu_id=self.kv_args.gpu_id,
            ib_device=self.kv_args.ib_device,
        )

    """
    # for disagg
    def get_contiguous_buf_infos(self):
        # MLA has only one kv_buffer, so only the information of this buffer needs to be returned.
        #
        self.kv_buffer = [
            torch.zeros(
                    (size + page_size, 1, kv_lora_rank + qk_rope_head_dim),
                    dtype=self.store_dtype,
                    device=device)
                for _ in range(layer_num)
        ]

        self.data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.kv_buffer],
            dtype=torch.uint64, device=self.device)

        kv_data_ptrs = [
            self.kv_buffer[i].data_ptr() for i in range(self.layer_num)]

        kv_data_lens = [
            self.kv_buffer[i].nbytes for i in range(self.layer_num)]

        kv_item_lens = [
            self.kv_buffer[i][0].nbytes * self.page_size for i in range(self.layer_num)]
    """

    def register_buffer_to_engine(self):
        # Batch register KV data buffers
        # 注册 每层 kv_cache 的起始地址、长度 bytes 信息.
        """
        kv_data_ptrs = [self.k_buffer[i].data_ptr() for i in range(self.layer_num)] +
                       [self.v_buffer[i].data_ptr() for i in range(self.layer_num)]

        kv_data_lens = [self._get_key_buffer(i).nbytes for i in range(self.layer_num)] +
                       [self._get_value_buffer(i).nbytes for i in range(self.layer_num)]
        """
        # 注册每一层的 kvcache 指针和 tensor_nbytes 信息.
        if self.kv_args.kv_data_ptrs and self.kv_args.kv_data_lens:
            self.engine.batch_register(
                self.kv_args.kv_data_ptrs, self.kv_args.kv_data_lens
            )

        # Batch register auxiliary data buffers in metadata_buffers.
        # 注册 output_ids, output_hidden_states 等信息对应的指针和
        # tensor_nbytes 信息.
        if self.kv_args.aux_data_ptrs and self.kv_args.aux_data_lens:
            self.engine.batch_register(
                self.kv_args.aux_data_ptrs, self.kv_args.aux_data_lens
            )

        # Batch register state/extra pool data buffers
        if self.kv_args.state_data_ptrs and self.kv_args.state_data_lens:
            self.engine.batch_register(
                self.kv_args.state_data_ptrs, self.kv_args.state_data_lens
            )

    def _transfer_data(self, mooncake_session_id, transfer_blocks):
        if not transfer_blocks:
            return 0

        src_addrs, dst_addrs, lengths = zip(*transfer_blocks)
        return self.engine.batch_transfer_sync(
            mooncake_session_id, list(src_addrs), list(dst_addrs), list(lengths)
        )

    def _send_kvcache_generic(
        self,
        mooncake_session_id: str,
        src_data_ptrs: list[int],
        dst_data_ptrs: list[int],
        item_lens: list[int],
        prefill_data_indices: npt.NDArray[np.int32],
        dst_data_indices: npt.NDArray[np.int32],
        executor: concurrent.futures.ThreadPoolExecutor,
    ) -> int:
        """
        Generic KV cache transfer supporting both MHA and MLA architectures.
        This method is used by both send_kvcache (full pool) and maybe_send_extra.
        """
        # Group by indices for optimization
        prefill_kv_blocks, dst_kv_blocks = group_concurrent_contiguous(
            prefill_data_indices, dst_data_indices
        )

        layers_params = None

        # pp is not supported on the decode side yet
        if self.is_mla_backend:
            src_kv_ptrs, dst_kv_ptrs, layers_current_pp_stage = (
                self.get_mla_kv_ptrs_with_pp(src_data_ptrs, dst_data_ptrs)
            )
            kv_item_len = item_lens[0]
            layers_params = [
                (
                    src_kv_ptrs[layer_id],
                    dst_kv_ptrs[layer_id],
                    kv_item_len,
                )
                for layer_id in range(layers_current_pp_stage)
            ]
        else:
            src_k_ptrs, src_v_ptrs, dst_k_ptrs, dst_v_ptrs, layers_current_pp_stage = (
                self.get_mha_kv_ptrs_with_pp(src_data_ptrs, dst_data_ptrs)
            )
            kv_item_len = item_lens[0]

            # 2. 取得 layer level 的传输信息初始化
            layers_params = [
                (
                    src_k_ptrs[layer_id],
                    dst_k_ptrs[layer_id],
                    kv_item_len,
                )
                for layer_id in range(layers_current_pp_stage)
            ] + [
                (
                    src_v_ptrs[layer_id],
                    dst_v_ptrs[layer_id],
                    kv_item_len,
                )
                for layer_id in range(layers_current_pp_stage)
            ]
        assert layers_params is not None

        def set_transfer_blocks(
            src_ptr: int, dst_ptr: int, item_len: int
        ) -> List[Tuple[int, int, int]]:
            transfer_blocks = []
            for prefill_index, decode_index in zip(prefill_kv_blocks, dst_kv_blocks):
                src_addr = src_ptr + int(prefill_index[0]) * item_len
                dst_addr = dst_ptr + int(decode_index[0]) * item_len
                length = item_len * len(prefill_index)
                transfer_blocks.append((src_addr, dst_addr, length))
            return transfer_blocks

        # Worker function for processing a single layer
        def process_layer(src_ptr: int, dst_ptr: int, item_len: int) -> int:
            transfer_blocks = set_transfer_blocks(src_ptr, dst_ptr, item_len)
            return self._transfer_data(mooncake_session_id, transfer_blocks)

        # Worker function for processing all layers in a batch
        def process_layers(layers_params: List[Tuple[int, int, int]]) -> int:
            transfer_blocks = []
            for src_ptr, dst_ptr, item_len in layers_params:
                transfer_blocks.extend(set_transfer_blocks(src_ptr, dst_ptr, item_len))
            return self._transfer_data(mooncake_session_id, transfer_blocks)

        if self.enable_custom_mem_pool:
            futures = [
                executor.submit(
                    process_layer,
                    src_ptr,
                    dst_ptr,
                    item_len,
                )
                for (src_ptr, dst_ptr, item_len) in layers_params
            ]
            for future in concurrent.futures.as_completed(futures):
                status = future.result()
                if status != 0:
                    for f in futures:
                        f.cancel()
                    return status
        else:
            # Combining all layers' params in one batch transfer is more efficient
            # compared to using multiple threads
            return process_layers(layers_params)

        return 0

    def send_kvcache(
        self,
        mooncake_session_id: str,
        prefill_kv_indices: npt.NDArray[np.int32],
        dst_kv_ptrs: list[int],
        dst_kv_indices: npt.NDArray[np.int32],
        executor: concurrent.futures.ThreadPoolExecutor,
    ):
        return self._send_kvcache_generic(
            mooncake_session_id=mooncake_session_id,
            src_data_ptrs=self.kv_args.kv_data_ptrs,
            dst_data_ptrs=dst_kv_ptrs,
            item_lens=self.kv_args.kv_item_lens,
            prefill_data_indices=prefill_kv_indices,
            dst_data_indices=dst_kv_indices,
            executor=executor,
        )

    def send_kvcache_slice(
        self,
        mooncake_session_id: str,
        prefill_kv_indices: npt.NDArray[np.int64],  # src_indices
        dst_kv_ptrs: list[int],
        dst_kv_indices: npt.NDArray[np.int64],  # dst_indices
        dst_tp_rank: int,
        dst_attn_tp_size: int,
        dst_kv_item_len: int,
        executor: concurrent.futures.ThreadPoolExecutor,
    ):
        """
        Sends KV cache slices from this Prefill rank to a target Decode rank,
        supporting generic M-to-N TP size configurations.

        NOTE: This implementation calls the transfer engine for each token slot within
        each page to ensure correctness for any page_size and head-slicing configuration.
        This may introduce performance overhead (increased TTFT) for long sequences.
        """
        # Extract configuration
        """
            Engine_rank -> tp_rank
            Local_tp_rank_in_group -> tp_rank mod attn_tp_size
        """
        local_tp_rank_in_group = (
            self.kv_args.engine_rank % self.attn_tp_size
        )  # engine_rank -> tp_rank

        dst_tp_rank_in_group = dst_tp_rank % dst_attn_tp_size
        num_kv_heads = (
            self.kv_args.kv_head_num
        )  # max(1, total_num_kv_heads // tensor_parallel_size)
        num_layers = len(self.kv_args.kv_data_ptrs)
        page_size = self.kv_args.page_size

        # Calculate head distribution
        src_heads_per_rank = num_kv_heads
        dst_heads_per_rank = num_kv_heads * self.attn_tp_size // dst_attn_tp_size

        f"""
            - dst_kv_item_len 代表每个 dist page 的 nbytes 长度:
                (self.size + size.page_size, self.head_num, self.head_dim)
            - bytes_per_head_slice_to_send 代表每个 head 的 bytes 长度
        """
        bytes_per_head_slice_to_send = (
            dst_kv_item_len // page_size // dst_heads_per_rank
        )  # 即, head_dim * kv_bytes

        # Determine slicing parameters based on TP configuration
        if self.attn_tp_size > dst_attn_tp_size:
            # Send KVCache from multiple prefill instances to 1 decode instance
            f"""
                prefill_tp_8 -> decode_tp_4

                - src_rank_2:  dst_head_start_offset = (2 * 1) = 2
                - src_rank_3:  dst_head_start_offset = (3 * 1) = 3
            """
            src_head_start_offset = 0
            num_heads_to_send = src_heads_per_rank

            # NOTE(james): 有些 bug 存在
            dst_head_start_offset = (
                local_tp_rank_in_group * src_heads_per_rank
            ) % dst_heads_per_rank
        else:
            # Send KVCache from 1 prefill instance to multiple decode instances
            f"""
                prefill_tp_4 -> decode_tp_8
                - src_rank_1 -> head id 2,3

                dst_rank_2: dst_head_start_offset = (2 * 1) % 2 = 0
                dst_rank_3: dst_head_start_offset = (3 * 1) % 2 = 1
            """
            src_head_start_offset = (
                dst_tp_rank_in_group * dst_heads_per_rank
            ) % src_heads_per_rank
            num_heads_to_send = dst_heads_per_rank
            dst_head_start_offset = 0

        src_k_ptrs, src_v_ptrs, dst_k_ptrs, dst_v_ptrs, layers_current_pp_stage = (
            self.get_mha_kv_ptrs_with_pp(self.kv_args.kv_data_ptrs, dst_kv_ptrs)
        )

        # Calculate precise byte offset and length for the sub-slice within the token
        """
            Calculate precise byte offset and length for the sub-slice within the token
                从 prefill (src) 视角: src 为自身 slot 内取数起点的相对偏移;
            dst 为对方 slot 内按 head 序的写入口（第几个 head 的起始字节，对 decode slot 而言是「绝对」的 head 位置）
        """
        src_head_slice_offset = src_head_start_offset * bytes_per_head_slice_to_send
        dst_head_slice_offset = dst_head_start_offset * bytes_per_head_slice_to_send

        heads_bytes_per_token_to_send = num_heads_to_send * bytes_per_head_slice_to_send

        src_kv_item_len = self.kv_args.kv_item_lens[0]  # 每个 page 的 nbytes

        # Sanity check: The data sub-slice to be sent should fit into the dst buffer.
        # This means heads_bytes_per_token_to_send <= (dst_kv_item_len // page_size)
        if heads_bytes_per_token_to_send > (dst_kv_item_len // page_size):
            logger.error(
                f"[{mooncake_session_id}] slice size ({heads_bytes_per_token_to_send}) exceeds "
                f"target token slot size ({dst_kv_item_len // page_size})"
            )
            return -1

        layers_params = [
            (
                src_k_ptrs[layer_id],
                dst_k_ptrs[layer_id],
                src_kv_item_len,
                dst_kv_item_len,
                src_head_slice_offset,
                dst_head_slice_offset,
                heads_bytes_per_token_to_send,
            )
            for layer_id in range(layers_current_pp_stage)
        ] + [
            (
                src_v_ptrs[layer_id],
                dst_v_ptrs[layer_id],
                src_kv_item_len,
                dst_kv_item_len,
                src_head_slice_offset,
                dst_head_slice_offset,
                heads_bytes_per_token_to_send,
            )
            for layer_id in range(layers_current_pp_stage)
        ]

        def process_layer_tp_aware(layer_params):
            (
                src_ptr,
                dst_ptr,
                src_item_len,  # 含 page_size
                dst_item_len,
                src_head_slice_offset,
                dst_head_slice_offset,
                heads_bytes_per_token_to_send,
            ) = layer_params
            src_addr_list = []
            dst_addr_list = []
            length_list = []

            # Calculate strides for a single token slot
            # NOTE(james): prefill 和 decode node 的 page_size 必须相同.
            bytes_per_token_on_prefill = src_item_len // page_size
            bytes_per_token_on_decode = dst_item_len // page_size

            for i in range(len(prefill_kv_indices)):
                prefill_page_idx = int(prefill_kv_indices[i])
                decode_page_idx = int(dst_kv_indices[i])

                # Get the starting addresses for the current src and dst pages
                src_page_start_addr = src_ptr + prefill_page_idx * src_item_len
                dst_page_start_addr = dst_ptr + decode_page_idx * dst_item_len

                # Iterate through each valid token slot within the current page
                for token_slot_in_page in range(page_size):
                    # Calculate the start address of the current token slot
                    src_token_slot_start_addr = (
                        src_page_start_addr
                        + token_slot_in_page * bytes_per_token_on_prefill
                    )
                    dst_token_slot_start_addr = (
                        dst_page_start_addr
                        + token_slot_in_page * bytes_per_token_on_decode
                    )

                    # Calculate final src and dst addresses by applying head-slice offsets
                    src_slice_addr = src_token_slot_start_addr + src_head_slice_offset
                    dst_slice_addr = dst_token_slot_start_addr + dst_head_slice_offset

                    src_addr_list.append(src_slice_addr)
                    dst_addr_list.append(dst_slice_addr)
                    length_list.append(heads_bytes_per_token_to_send)

            # 调用的核心传输位置.
            return self.engine.batch_transfer_sync(
                mooncake_session_id, src_addr_list, dst_addr_list, length_list
            )

        futures = [
            executor.submit(
                process_layer_tp_aware,
                layer_params,
            )
            for layer_params in layers_params
        ]

        for future in concurrent.futures.as_completed(futures):
            status = future.result()
            if status != 0:
                for f in futures:
                    f.cancel()
                return status

        return 0

    def send_aux(
        self,
        req: TransferInfo,
        prefill_aux_index: int,
        dst_aux_ptrs: list[int],
    ):
        f"""
            ptrs = [
                self.output_ids.data_ptr(),
                self.cached_tokens.data_ptr(),
                self.output_token_logprobs_val.data_ptr(),
                self.output_token_logprobs_idx.data_ptr(),
                self.output_top_logprobs_val.data_ptr(),
                self.output_top_logprobs_idx.data_ptr(),
                self.output_hidden_states.data_ptr(),
            ]
            item_lens = [
                self.output_ids[0].nbytes,
                self.cached_tokens[0].nbytes,
                self.output_top_logprobs_val[0].nbytes,
                self.output_top_logprobs_idx[0].nbytes,
                self.output_topk_p[0].nbytes,
                self.output_topk_index[0].nbytes,
                self.output_hidden_states[0].nbytes,
            ]
        """
        # TODO(shangming): Fix me when nvlink_transport of Mooncake is bug-free
        if self.enable_custom_mem_pool and self.custom_mem_pool_type == "NVLINK":
            return self.send_aux_tcp(req, prefill_aux_index, dst_aux_ptrs)

        transfer_blocks = []
        prefill_aux_ptrs = self.kv_args.aux_data_ptrs
        prefill_aux_item_lens = self.kv_args.aux_item_lens

        # prefill_aux_idx 和 dst_aux_idx 对应的 meta_data_buffer 的 index
        for i, dst_aux_ptr in enumerate(dst_aux_ptrs):
            length = prefill_aux_item_lens[i]
            src_addr = prefill_aux_ptrs[i] + length * prefill_aux_index
            dst_addr = dst_aux_ptrs[i] + length * req.dst_aux_index
            transfer_blocks.append((src_addr, dst_addr, length))

        return self._transfer_data(req.mooncake_session_id, transfer_blocks)

    def send_aux_tcp(
        self,
        req: TransferInfo,
        prefill_aux_index: int,
        dst_aux_ptrs: list[int],
    ):
        prefill_aux_ptrs = self.kv_args.aux_data_ptrs
        prefill_aux_item_lens = self.kv_args.aux_item_lens

        for i in range(len(prefill_aux_ptrs)):
            length = prefill_aux_item_lens[i]
            src_addr = prefill_aux_ptrs[i] + length * prefill_aux_index
            data = AuxDataCodec.serialize_data_from_buffer(src_addr, length)

            self.send_aux_data_to_endpoint(
                remote=req.endpoint,
                dst_port=req.dst_port,
                room=req.room,
                buffer_index=i,
                aux_index=req.dst_aux_index,
                data=data,
            )

        return 0

    def send_aux_data_to_endpoint(
        self,
        remote: str,
        dst_port: int,
        room: int,
        buffer_index: int,
        aux_index: int,
        data: bytes,
    ):
        socket = self._connect(
            format_tcp_address(remote, dst_port), is_ipv6=is_valid_ipv6_address(remote)
        )

        socket.send_multipart(
            [
                MooncakeKVManager.AUX_DATA_HEADER,
                str(room).encode("ascii"),
                str(buffer_index).encode("ascii"),
                str(aux_index).encode("ascii"),
                struct.pack(">I", len(data)),
                data,
            ]
        )

    def _handle_aux_data(self, msg: List[bytes]):
        """Handle AUX_DATA messages received by the decode thread."""
        room = int(msg[1].decode("ascii"))
        buffer_index = int(msg[2].decode("ascii"))
        aux_index = int(msg[3].decode("ascii"))
        data_length = struct.unpack(">I", msg[4])[0]
        data = msg[5]

        if len(data) != data_length:
            logger.error(f"AUX_DATA length mismatch for bootstrap_room {room}")
            return

        AuxDataCodec.deserialize_data_to_buffer(
            self.kv_args, buffer_index, aux_index, data
        )

        logger.debug(
            f"Received AUX_DATA for bootstrap_room {room} with length:{len(data)}"
        )

    def maybe_send_extra(
        self,
        req: TransferInfo,
        prefill_state_indices: list[int],
        dst_state_data_ptrs: list[int],
        executor: concurrent.futures.ThreadPoolExecutor,
    ):
        """Send state or extra pool data with type-specific handling."""
        state_type = getattr(self.kv_args, "state_type", "none")

        if state_type == "mamba":
            return self._send_mamba_state(
                req,
                prefill_state_indices,
                dst_state_data_ptrs,
            )
        elif state_type in ["swa", "nsa"]:
            # Reuse _send_kvcache_generic interface to send extra pool data
            prefill_state_indices = np.array(prefill_state_indices, dtype=np.int32)
            dst_state_indices = np.array(req.dst_state_indices, dtype=np.int32)
            return self._send_kvcache_generic(
                mooncake_session_id=req.mooncake_session_id,
                src_data_ptrs=self.kv_args.state_data_ptrs,
                dst_data_ptrs=dst_state_data_ptrs,
                item_lens=self.kv_args.state_item_lens,
                prefill_data_indices=prefill_state_indices,
                dst_data_indices=dst_state_indices,
                executor=executor,
            )
        else:
            return 0

    def _send_mamba_state(
        self,
        req: TransferInfo,
        prefill_mamba_index: list[int],
        dst_state_data_ptrs: list[int],
    ):
        """Transfer Mamba states."""
        assert len(prefill_mamba_index) == 1, "Mamba should have single state index"

        transfer_blocks = []
        prefill_state_data_ptrs = self.kv_args.state_data_ptrs
        prefill_state_item_lens = self.kv_args.state_item_lens

        for i, dst_state_ptr in enumerate(dst_state_data_ptrs):
            length = prefill_state_item_lens[i]
            src_addr = prefill_state_data_ptrs[i] + length * int(prefill_mamba_index[0])
            dst_addr = dst_state_ptr + length * int(req.dst_state_indices[0])
            transfer_blocks.append((src_addr, dst_addr, length))

        return self._transfer_data(req.mooncake_session_id, transfer_blocks)

    def sync_status_to_decode_endpoint(
        self, remote: str, dst_port: int, room: int, status: int, prefill_rank: int
    ):
        # 让 socket 连接到对方的地址, 并发送消息.
        self._connect(
            format_tcp_address(remote, dst_port), is_ipv6=is_valid_ipv6_address(remote)
        ).send_multipart(
            [
                str(room).encode("ascii"),
                str(status).encode("ascii"),
                str(prefill_rank).encode("ascii"),
            ]
        )

    def transfer_worker(
        self, queue: FastQueue, executor: concurrent.futures.ThreadPoolExecutor
    ):
        while True:
            try:
                f"""
                    Queue 中的 TransferKVChunk 对象, 来自 Sender 阶段的 add_transfer_request 方法,
                    每次仅处理一条 Req 的 KVCache 传输.
                """
                kv_chunk: TransferKVChunk = (
                    queue.get()
                )  # FastQueue 带有 _cond_lock, 防止频繁 Query.

                f"""
                    self.transfer_infos[room][mooncake_session_id] = (
                            TransferInfo.from_zmq(waitting_req_bytes)
                        )

                    transfer_infos 的结构:
                        - key: int (room)
                        - val: dict[str, TransferInfo]
                            - key: mooncake_session_id (str), prefill pod ip (单卡 level)
                            - val: TransferInfo (包含了 decode node 的 ip 地址、port 端口、dst_kv_indices 等信息)
                        - transfer_infos: 拿到具体的 kvcache relative location.
                """
                reqs_to_be_processed = (
                    self.transfer_infos[kv_chunk.room].values()
                    if kv_chunk.room in self.transfer_infos
                    else []
                )
                polls = []
                dst_ranks_infos = []

                f"""
                    # prefill_rank:
                        pp_rank 0 + attn_tp_rank: 0 1 2 3 -> local_rank: 0 2 4 6
                        pp_rank 1 + attn_tp_rank: 0 1 2 3 -> local_rank: 1 3 5 7
                """
                local_rank = self.attn_tp_rank * self.pp_size + self.pp_rank

                f"""
                    Req 代表的是 Decode 握手后传输到 prefill pod 上的 TransferInfo,
                    对于 Prefill 要传给几个 Decode Node. 利用 len(mooncake_session_id) 来判断.
                """
                for req in reqs_to_be_processed:
                    if not req.is_dummy:
                        # Early exit if the request has failed

                        f"""
                            For mla attention.
                            - Dummy request means the decode instance is not used,
                            - so its status can be marked as success directly.
                        """
                        with self.session_lock:
                            # 处理 Decode 机器挂了的问题, 如果任务失败了, 那么这个 Request 记录失败信息.
                            if req.mooncake_session_id in self.failed_sessions:
                                self.record_failure(
                                    kv_chunk.room,
                                    f"Decode instance could be dead, remote mooncake session {req.mooncake_session_id} is not alive",
                                )

                                # 1. 更新本地 request_status 池子内 room 的 failure 状态.
                                self.update_status(kv_chunk.room, KVPoll.Failed)

                                # 2. 远程更新向 Decode 节点同步 room 对应的 failed 状态.
                                self.sync_status_to_decode_endpoint(
                                    req.endpoint,
                                    req.dst_port,
                                    req.room,
                                    KVPoll.Failed,
                                    local_rank,
                                )
                                f"""
                                    break out of the 'for-loop' at line starting with "for req in reqs_to_be_processed:"
                                    since this room has failed, no need to process the remain requests.
                                """
                                break

                        # req.dst_kv_indices  : 该 request 在 Decode 端整段 KV 的目标槽位索引 (一整段)；
                        # kv_chunk.index_slice: 当前 chunk 对应整段里的哪一段 (eg: slice(0, 50));
                        # chunked_dst_kv_indice: 本 chunk 在 decode 端目标槽位索引 (eg: dst_kv_indices[0:50])
                        # 与 kv_chunk.prefill_indices 一一对应, 用于 send_kvcache

                        # 1. Dst kvcache indices
                        chunked_dst_kv_indice = req.dst_kv_indices[kv_chunk.index_slice]

                        # NOTE: This is temporarily a workaround to deal with the case where the prefill_kv_indices
                        # is mismatched with the dst_kv_indices when page size > 1, this should never happen.
                        if len(chunked_dst_kv_indice) < len(
                            kv_chunk.prefill_kv_indices
                        ):
                            logger.warning(
                                f"len(chunked_dst_kv_indice) = {len(chunked_dst_kv_indice)}, len(kv_chunk.prefill_kv_indices) = {len(kv_chunk.prefill_kv_indices)}"
                            )

                            # 2. src kvcache indices
                            kv_chunk.prefill_kv_indices = kv_chunk.prefill_kv_indices[
                                : len(chunked_dst_kv_indice)
                            ]

                        target_rank_registration_info: KVArgsRegisterInfo = (
                            self.decode_kv_args_table[req.mooncake_session_id]
                        )
                        if self.is_mla_backend or (
                            self.attn_tp_size
                            == target_rank_registration_info.dst_attn_tp_size
                        ):
                            # 2. 边的 tp_size 信息匹配, 那么直接发送 kvcache 全量数据
                            ret = self.send_kvcache(
                                req.mooncake_session_id,
                                kv_chunk.prefill_kv_indices,
                                target_rank_registration_info.dst_kv_ptrs,
                                chunked_dst_kv_indice,
                                executor,
                            )
                        else:
                            ret = self.send_kvcache_slice(
                                req.mooncake_session_id,
                                kv_chunk.prefill_kv_indices,  # src_indices
                                target_rank_registration_info.dst_kv_ptrs,
                                chunked_dst_kv_indice,  # dst_indices
                                target_rank_registration_info.dst_tp_rank,
                                target_rank_registration_info.dst_attn_tp_size,
                                target_rank_registration_info.dst_kv_item_len,
                                executor,
                            )
                        if ret != 0:
                            with self.session_lock:
                                self.session_failures[req.mooncake_session_id] += 1
                                # Failures should never happen if the session is not dead, if the session fails once, mark it as failed
                                if self.session_failures[req.mooncake_session_id] >= 1:
                                    self.failed_sessions.add(req.mooncake_session_id)
                                    logger.error(
                                        f"Session {req.mooncake_session_id} failed."
                                    )

                            # 记录 failed room 及 failed 原因.
                            self.record_failure(
                                kv_chunk.room,
                                f"Failed to send kv chunk of {kv_chunk.room} to {req.endpoint}:{req.dst_port}",
                            )

                            # 向 decode node 同步失败信息
                            self.update_status(kv_chunk.room, KVPoll.Failed)
                            self.sync_status_to_decode_endpoint(
                                # endpoint_ip:port 通信地址
                                req.endpoint,
                                req.dst_port,
                                # 发送的信息
                                req.room,
                                KVPoll.Failed,
                                local_rank,
                            )
                            break
                        """
                            # kv_chunk.is_last: prefill stage ReqA 的 kvcache 要分成多个 chunk 然后发送,
                            # 这是最后一个 chunk, 也只有 last aux_data 发送完毕之后, 才认为该 Req 传输成功.
                        """
                        if kv_chunk.is_last:
                            if kv_chunk.state_indices is not None:
                                if not self.is_mla_backend and (
                                    self.attn_tp_size
                                    != target_rank_registration_info.dst_attn_tp_size
                                ):
                                    raise RuntimeError(
                                        f"PD Disaggregation does NOT support PD different TP sizes for non-MLA hybrid models yet."
                                    )

                                self.maybe_send_extra(
                                    req,
                                    kv_chunk.state_indices,
                                    target_rank_registration_info.dst_state_data_ptrs,
                                    executor,
                                )

                            if self.pp_group.is_last_rank:
                                # Only the last chunk we need to send the aux data
                                ret = self.send_aux(
                                    req,
                                    kv_chunk.prefill_aux_index,
                                    target_rank_registration_info.dst_aux_ptrs,
                                )
                            polls.append(True if ret == 0 else False)
                            dst_ranks_infos.append(
                                (req.endpoint, req.dst_port, req.room)
                            )

                            # Only sync status when all the dst ranks have received the kvcache
                            if len(polls) == req.required_dst_info_num:
                                status = KVPoll.Success if all(polls) else KVPoll.Failed
                                self.update_status(req.room, status)
                                for endpoint, dst_port, room in dst_ranks_infos:
                                    self.sync_status_to_decode_endpoint(
                                        endpoint, dst_port, room, status, local_rank
                                    )
                    else:
                        # Dummy request means the decode instance is not used, so its status can be marked as success directly
                        # Dummy request does not need to sync status to decode endpoint
                        if kv_chunk.is_last and req.room in self.request_status:
                            self.update_status(req.room, KVPoll.Success)

                # 收尾: 若该 room 已无需再追踪(或已成功). 则从 transfer_infos 中移除该 room 的记录, 避免泄露.
                # room not in request_status: schedule 侧已对该 request 调用了 clear() 方法, 清理记录;
                # 如 KVSender.poll() 返回 Success 后)
                if (
                    kv_chunk.room not in self.request_status
                    or self.check_status(kv_chunk.room) == KVPoll.Success
                ):
                    if kv_chunk.room in self.transfer_infos:
                        self.transfer_infos.pop(kv_chunk.room)

            except Exception as e:
                # NOTE(shangming): Remove this when we make sure the transfer thread is bug-free
                raise RuntimeError(
                    f"Transfer thread failed because of {e}. Prefill instance with bootstrap_port={self.bootstrap_port} is dead."
                )

    f"""
        控制任务的命令, 通过 socket 发送过来. 任务具体内容: 要搬运的 kvcache,
        则是通过 mooncake 利用 rdma 通道发送过来.

        2026-02-26 INFO conn.py 938 [TP0] Register KVArgs from 10.39.50.22:15845
        2026-02-26 INFO conn.py 938 [TP1] Register KVArgs from 10.39.50.22:16570
        2026-02-26 INFO conn.py 938 [TP2] Register KVArgs from 10.39.50.22:15361
        2026-02-26 INFO conn.py 938 [TP3] Register KVArgs from 10.39.50.22:16272
    """

    def start_prefill_thread(self):
        # self.server_socket = zmq.Context().socket(zmq.PULL), 绑定到 "单向收请求 socket"

        # 启动 prefill bootstrap thread
        def bootstrap_thread():
            """This thread recvs pre-alloc notification from the decode engine"""
            # KVPoll.Bootstrapping -> KVPoll.WaitingForInput
            while True:
                # 阻塞式接收一个多帧消息: recv_multipart() 会返回一个列表, 每个元素是一个对象.
                waiting_req_bytes = self.server_socket.recv_multipart()

                """
                    解析协议: 从多帧消息中提取 room id (等价于 request id 消息) 和
                            任务id mass level 的 req-id 信息.
                """
                room = waiting_req_bytes[0].decode("ascii")
                mooncake_session_id = waiting_req_bytes[3].decode("ascii")

                # bootstrap room=="None", 表示 decode 侧的 "首次握手" 注册该 session 的 kv 缓冲区信息:
                #   - 每个 mooncake_session_id (每个连接到本 prefill 的 decode rank) 会发送一次;
                #   - 故本分支在进程内 会执行多次 (== 连过来的 Decode 端数量);
                #   - 对同一 session_id 在稳定运行下只注册一次, 重新连时再次进入并覆盖.
                if room == "None":
                    """
                    初始化服务时, 将 decode pod 的 kvcache 初始化信息注册到;
                    decode_kv_args_table 中对应 decode receiver 中 _register_kv_args() 方法.
                    """
                    self.decode_kv_args_table[mooncake_session_id] = (
                        KVArgsRegisterInfo.from_zmq(waiting_req_bytes)
                    )

                    # 清理失败记录.
                    with self.session_lock:
                        """
                        任务链接被找到,
                        - 从 failed_sessions 集合 (set) 中移除;
                        - 从记录这个 session id 连接失败多少次 cnt 的字典 dict 中移除;
                        """
                        if mooncake_session_id in self.failed_sessions:
                            self.failed_sessions.remove(mooncake_session_id)
                        if mooncake_session_id in self.session_failures:
                            del self.session_failures[mooncake_session_id]
                    logger.debug(
                        f"Register KVArgs from {mooncake_session_id} successfully"
                    )
                    continue
                else:
                    f"""
                        代表 2th 阶段握手, 解析需要等待的 decode ranks 数量, 应该 runtime level.
                    """
                    required_dst_info_num = int(waiting_req_bytes[7].decode("ascii"))
                    room = int(room)
                    if room not in self.transfer_infos:
                        self.transfer_infos[room] = {}

                    f"""
                        - 这里的 MoonCake Session ID 可理解成, 这个 Prefill pod rank 需要接收几个 decode node rank 的请求.
                        - TransferInfo.from_zmq 拿到的是 具体某请求的 kvcache 位置信息.
                    """
                    self.transfer_infos[room][mooncake_session_id] = (
                        TransferInfo.from_zmq(waiting_req_bytes)
                    )
                    # NOTE: after bootstrapping we can mark the req as waiting for input,
                    # 哦, 我理解了这里的 room 中有多少个 session_id 已经就绪了.
                    if len(self.transfer_infos[room]) == required_dst_info_num:
                        self.update_status(room, KVPoll.WaitingForInput)

        # 整个生命周期, 持续运转.
        threading.Thread(target=bootstrap_thread).start()

    def start_decode_thread(self):
        # 请在本机的指定 ip 端口开始一个 "双向请求socket"
        #   - self.local_ip  是这面墙的地址 (比如: "xx小区 3 号楼201" )
        #   - self.rank_port 是插座的孔位号 (比如: "插座#8080" )
        def decode_thread():
            while True:
                # 阻塞式接收一个多帧消息: recv_multipart() 会返回一个列表, 每个元素是一个对象.
                #    - bootstrap_room: 房间 ID (请求ID)
                #    - status: KV 传输状态 (success/failed 等)
                #    - prefill_rank: 发送这个消息的 prefill rank ID
                msg = self.server_socket.recv_multipart()
                if msg[0] == MooncakeKVManager.AUX_DATA_HEADER:
                    self._handle_aux_data(msg)
                    continue

                """
                    NOTE(james): Decode 端并不想 Prefill 端会发送有形的 TransferKVChunk 而
                                 仅接收 Prefill 端发送过来的状态量.

                    sync_status_to_decode_endpoint
                        .send_multipart(
                            str(room).encode("ascii"),
                            str(status).encode("ascii"),
                            str(prefill_rank).encode("ascii"),
                        )
                """
                (bootstrap_room, status, prefill_rank) = msg
                status = int(status.decode("ascii"))
                bootstrap_room = int(bootstrap_room.decode("ascii"))

                # 来自第几个 prefill rank
                prefill_rank = int(prefill_rank.decode("ascii"))

                if status == KVPoll.Success:
                    if bootstrap_room in self.request_status:
                        self.prefill_response_tracker[bootstrap_room].add(prefill_rank)
                        expected_response_num = (
                            self.required_prefill_response_num_table[bootstrap_room]
                        )
                        arrived_response_num = len(
                            self.prefill_response_tracker[bootstrap_room]
                        )

                        # 所有的 prefill ranks 都发送过来请求了,
                        # 这个 bootstrap room is success and ready to be decoded.
                        if arrived_response_num == expected_response_num:
                            self.update_status(bootstrap_room, KVPoll.Success)
                elif status == KVPoll.Failed:
                    self.record_failure(
                        bootstrap_room,
                        f"Failed to get kvcache from prefill instance, it might be dead",
                    )
                    self.update_status(bootstrap_room, status)

        # 心跳检查: 检查 prefill pod 是否健康
        def heartbeat_checker():
            while True:
                time.sleep(self.heartbeat_interval)
                # connection_lock: 保护 connection_pool, prefill_*_table, addr_to_rooms_tracker 等 "拓扑/元数据"
                # 只持锁拷贝地址列表, 避免后续 HTTP 请求期间长期占用, 让 _handle_node_failure() 等能及时改拓扑.
                with self.connection_lock:
                    """
                    prefill_dp_size_table: 站在 decode node 的角度去看与之相连的 prefill nodes
                    addresses 是 bootstrap addr 的列表.
                    """
                    addresses = list(self.prefill_dp_size_table.keys())

                for bootstrap_addr in addresses:
                    session = None
                    try:
                        with self.session_pool_lock:
                            """
                            如果 bootstrap_addr 不在 session_pool 中，由于 session_pool 是 defaultdict(requests.Session),
                            它会自动为这个 key 创建一个新的 requests.Session() 实例， Session 是 HTTP 客户端会话对象，它可以:
                            1. 复用 TCP 连接 (Connection pooling), 避免每次请求都建立新连接
                            2. 保持 cookies 和认证信息
                            3. 设置默认的请求头和超时参数
                            4. 提供更好的性能，特别是对同一服务器的多次请求

                            新创建的 session 会被缓存起来，供同一个 bootstrap_addr 的后续请求使用。
                            # 只要尝试连接或连接未被判定为彻底失效, Session 对象就会保留在此，以复用 TCP 连接。
                            """
                            session = self.session_pool[bootstrap_addr]

                        """
                            发射 HTTP GET 请求到 prefill bootstrap server, to check if the prefill node is healthy
                            - URL: http://{prefill_node_address}/health (health check endpoint)
                            - timeout=(2, 3): connection timeout=2s, read timeout=3s
                            - Connection: keep-alive header maintains persistent HTTP connection
                        """
                        response = session.get(
                            f"http://{bootstrap_addr}/health",
                            timeout=(2, 3),
                            headers={"Connection": "keep-alive"},
                        )

                        if response.status_code == 200:
                            self.heartbeat_failures[bootstrap_addr] = 0

                            current_rooms = self.addr_to_rooms_tracker[
                                bootstrap_addr
                            ].copy()

                            for bootstrap_room in current_rooms:
                                # Remove KVPoll.Success requests from the tracker
                                if bootstrap_room not in self.request_status:
                                    """
                                    # addr_to_rooms_tracker 是 bootstrap_addr 到 rooms 的映射记录,

                                    - 用于表达已完成 (Finished/Success) 的 Request (room) 可以从心跳追踪列表
                                      (addr_to_rooms_tracker) 中移除了 (Evict)
                                    """
                                    self.addr_to_rooms_tracker[bootstrap_addr].discard(
                                        bootstrap_room
                                    )
                        else:
                            logger.info(
                                f"Attempting to reconnect to {bootstrap_addr}..."
                            )

                            self.heartbeat_failures[bootstrap_addr] = (
                                # self.heartbeat_failures.get(bootstrap_addr, 0) 含义,
                                #    - 从字典 self.heartbeat_failures 中获取 bootstrap_addr 的失败次数,
                                #    - 如果 bootstrap_addr 不在字典中, 则返回默认值
                                #    - 然后将失败次数 +1, 表示这次心跳检查失败了.
                                self.heartbeat_failures.get(bootstrap_addr, 0)
                                + 1
                            )

                            # key: 失败了的话一定要重建 session.
                            with self.session_pool_lock:
                                if bootstrap_addr in self.session_pool:
                                    del self.session_pool[bootstrap_addr]
                    except Exception:
                        logger.info(f"Attempting to reconnect to {bootstrap_addr}...")
                        self.heartbeat_failures[bootstrap_addr] = (
                            self.heartbeat_failures.get(bootstrap_addr, 0) + 1
                        )

                    if (
                        self.heartbeat_failures.get(bootstrap_addr, 0)
                        >= self.max_failures
                    ):
                        self._handle_node_failure(bootstrap_addr)
                        with self.session_pool_lock:
                            if bootstrap_addr in self.session_pool:
                                del self.session_pool[bootstrap_addr]

        threading.Thread(target=decode_thread).start()
        threading.Thread(target=heartbeat_checker).start()

    def check_status(self, bootstrap_room: int):
        return self.request_status[bootstrap_room]

    def update_status(self, bootstrap_room: int, status: KVPoll):
        if bootstrap_room not in self.request_status:
            self.request_status[bootstrap_room] = status
        else:
            # NOTE: status is only allowed to be incremented unless it is KVPoll.Failed
            if status == KVPoll.Failed:
                self.request_status[bootstrap_room] = KVPoll.Failed
            else:
                self.request_status[bootstrap_room] = max(
                    self.request_status[bootstrap_room], status
                )

    def record_failure(self, bootstrap_room: int, failure_reason: str):
        with self.failure_lock:
            self.failure_records[bootstrap_room] = failure_reason

    def get_session_id(self):
        return self.engine.get_session_id()

    def _handle_node_failure(self, failed_bootstrap_addr):
        with self.connection_lock:
            keys_to_remove = [
                k for k in self.connection_pool if k.startswith(failed_bootstrap_addr)
            ]
            for k in keys_to_remove:
                del self.connection_pool[k]
            if failed_bootstrap_addr in self.prefill_attn_tp_size_table:
                del self.prefill_attn_tp_size_table[failed_bootstrap_addr]
            if failed_bootstrap_addr in self.prefill_dp_size_table:
                del self.prefill_dp_size_table[failed_bootstrap_addr]
            if failed_bootstrap_addr in self.prefill_pp_size_table:
                del self.prefill_pp_size_table[failed_bootstrap_addr]

            possible_affected_rooms = self.addr_to_rooms_tracker.get(
                failed_bootstrap_addr, []
            )
            if failed_bootstrap_addr in self.addr_to_rooms_tracker:
                del self.addr_to_rooms_tracker[failed_bootstrap_addr]

        # Report the requests associated with the failed bootstrap addr and mark their status as KVPoll.Failed
        affected_rooms = []
        for room in possible_affected_rooms:
            if (
                room in self.request_status
                and self.check_status(room) != KVPoll.Success
            ):
                self.record_failure(
                    room,
                    f"Losing connection with prefill instance (bootstrap_addr: {failed_bootstrap_addr})",
                )
                self.update_status(room, KVPoll.Failed)
                affected_rooms.append(room)
        logger.error(
            f"Losing connection with prefill instance (bootstrap_addr: {failed_bootstrap_addr}), {len(affected_rooms)} requests affected"
        )

    # NOTE(james): 最后回头来清理这里的失败 cases.
    def add_transfer_request(
        self,
        bootstrap_room: int,
        kv_indices: npt.NDArray[np.int32],
        index_slice: slice,
        is_last: bool,
        aux_index: Optional[int] = None,
        state_indices: Optional[List[int]] = None,
    ):
        assert self.disaggregation_mode == DisaggregationMode.PREFILL
        assert not is_last or (is_last and aux_index is not None)

        if (
            bootstrap_room not in self.request_status
            or self.check_status(bootstrap_room) == KVPoll.Failed
        ):
            logger.debug(
                "Request with bootstrap_room=%s already failed", bootstrap_room
            )
            return

        """
            NOTE(james): 一定要首先收到了 decode 侧发送过来, 为这个请求开辟 quota,
                再由 prefill pod 发送过去.
        """
        if bootstrap_room not in self.transfer_infos:
            # This means that the current rank is a dummy rank for this request,
            # and it has already been marked as success, so there is no need to
            # add further chunks into the transfer queue.
            return

        # NOTE(shangming): sharding according to the dst_infos to make sure
        # requests with the same dst_sessions will be added into the same
        # queue, which enables early abort with failed sessions.

        """
            接收多个 d pod rank 的 mooncake_session_ids
            这个 transfer_info 来源还是在 start_prefill_thread() 内建立的.
        """
        dst_infos = self.transfer_infos[bootstrap_room].keys()
        session_port_sum = sum(int(session.rsplit(":", 1)[1]) for session in dst_infos)
        shard_idx = session_port_sum % len(self.transfer_queues)

        self.transfer_queues[shard_idx].put(
            TransferKVChunk(
                room=bootstrap_room,
                prefill_kv_indices=kv_indices,
                index_slice=index_slice,
                is_last=is_last,
                prefill_aux_index=aux_index,
                state_indices=state_indices,
            )
        )


################################################################################################
class MooncakeKVSender(CommonKVSender):

    def __init__(
        self,
        mgr: MooncakeKVManager,
        bootstrap_addr: str,  # bootstrap_addr ->  req.bootstrap_host : self.bootstrap_port, bootstrap_port 默认为 8998
        bootstrap_room: int,  # mass 层面代表着 req id
        dest_tp_ranks: List[int],
        pp_rank: int,
    ):
        super().__init__(mgr, bootstrap_addr, bootstrap_room, dest_tp_ranks, pp_rank)
        self.conclude_state = None
        self.init_time = time.time()

        # 这个值为 chunked_prefill 服务, 代表这个 req 在 meta_buffer 中的 slot_id
        # self.curr_idx = 0

    def send(
        self,
        kv_indices: npt.NDArray[
            np.int32
        ],  # 这个 kv_indices 代表 req_to_token 中的 page_ids
        state_indices: Optional[List[int]] = None,
    ):
        # index_slice 代表了 reqlen 的切片范围.
        index_slice = slice(self.curr_idx, self.curr_idx + len(kv_indices))

        self.curr_idx += len(kv_indices)
        is_last = self.curr_idx == self.num_kv_indices

        if not is_last:
            self.kv_mgr.add_transfer_request(
                self.bootstrap_room,
                kv_indices,
                index_slice,
                False,
            )
        else:
            self.kv_mgr.add_transfer_request(
                self.bootstrap_room,
                kv_indices,
                index_slice,
                True,
                aux_index=self.aux_index,
                state_indices=state_indices,
            )

    # 返回 kvcaches 的发送状态.
    def poll(self) -> KVPoll:
        if self.conclude_state is None:
            # 确认这个 Sender 任务的状态.
            status = self.kv_mgr.check_status(self.bootstrap_room)

            if status in (KVPoll.Success, KVPoll.Failed):
                # 成功与失败的结局
                self.conclude_state = status
            elif status == KVPoll.Bootstrapping:
                if self.init_time is not None:
                    now = time.time()
                    elapsed = now - self.init_time
                    if elapsed >= self.kv_mgr.bootstrap_timeout:
                        logger.warning_once(
                            "Some requests timed out when bootstrapping, "
                            "which means prefill instances fail to receive the KV indices from the decode instance of this request. "
                            "If a greater mean TTFT is acceptable, you can 'export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600' (10 minutes) to relax the timeout condition. "
                        )
                        self.kv_mgr.record_failure(
                            self.bootstrap_room,
                            f"Request {self.bootstrap_room} timed out after {elapsed:.1f}s in KVPoll.Bootstrapping",
                        )
                        self.conclude_state = KVPoll.Failed
                        return KVPoll.Failed

            return status
        else:
            return self.conclude_state

    # 清理这个 Req 在 Success/Failed 后的 status, 不再关心这个 Req 了.
    def clear(self) -> None:
        if self.bootstrap_room in self.kv_mgr.request_status:
            self.kv_mgr.request_status.pop(self.bootstrap_room)

    def failure_exception(self):
        # Explicitly set the status to failure since this request has failed in another rank
        if self.conclude_state is None:
            self.conclude_state = KVPoll.Failed

        self.clear()

        with self.kv_mgr.failure_lock:
            failure_reason = self.kv_mgr.failure_records.pop(
                self.bootstrap_room, "Failed due to an unknown reason from another rank"
            )
        raise KVTransferError(self.bootstrap_room, failure_reason)

    def abort(self):
        self.kv_mgr.record_failure(
            self.bootstrap_room,
            "Aborted by AbortReq.",
        )
        # Explicitly set the status to failure since this request has been aborted
        self.conclude_state = KVPoll.Failed


################################################################################################
class MooncakeKVReceiver(CommonKVReceiver):
    _ctx = zmq.Context()
    _socket_cache = {}
    _socket_locks = {}
    _global_lock = threading.Lock()

    def __init__(
        self,
        mgr: MooncakeKVManager,
        bootstrap_addr: str,  # bootstrap_addr -> req.bootstrap_host : self.bootstrap_port, bootstrap_port 默认为 8998
        bootstrap_room: Optional[int] = None,
        prefill_dp_rank: Optional[int] = None,
    ):
        self.session_id = mgr.get_session_id()  # session_id 代表 decode pod 的 local_ip

        # 最终总结状态
        self.conclude_state = None

        self.init_time = None
        super().__init__(mgr, bootstrap_addr, bootstrap_room, prefill_dp_rank)

        """
            设置对应的 Prefill Node 中应接受的 Req-id
            - addr_to_rooms_tracker: 追踪每个 bootstrap_addr 对应的 bootstrap_room (req id),
                                     用于心跳检查时判断哪些 reqs 可能受影响.
            - bootstrap_room: 代表 req id

            - CommonKVReceiver 内赋 room:KVPoll.status 的状态, 初始为 KVPoll.Bootstrapping, 一旦完成
                register Req 自身的 indices 等信息后, 转换为 KVPoll.WaitingForInput 状态.
        """
        self.kv_mgr.addr_to_rooms_tracker[self.bootstrap_addr].add(self.bootstrap_room)
        self.kv_mgr.update_status(self.bootstrap_room, KVPoll.WaitingForInput)

    def _get_bootstrap_info_from_server(
        self, engine_rank, target_dp_group, target_pp_rank
    ):
        """Fetch the bootstrap info from the bootstrap server."""
        try:
            url = f"http://{self.bootstrap_addr}/route?engine_rank={engine_rank}&target_dp_group={target_dp_group}&target_pp_rank={target_pp_rank}"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                bootstrap_info = response.json()
                return bootstrap_info
            else:
                logger.error(
                    f"Failed to get prefill server info: {response.status_code}, {response.text}"
                )
                return None
        except Exception as e:
            logger.error(f"Error fetching prefill info from bootstrap: {e}")
            return None

    def _register_kv_args(self):
        """
        # 这里 self.bootstrap_infos 是在 start_prefill_thread() 内通过 decode 侧发送过来的消息注册的,
        #    代表了这个 decode node 连接的 prefill nodes 的信息. 属于第一次 握手的内容. 打通 pd disagg 第一步.
        # self.bootstrap_infos 代表了 target prefill nodes 的 ip:port 信息.
        # 如果 decode attn tp size < prefill_attn_tp_size, 则需要向多个 prefill pods 发送 kvcache.
        """
        for bootstrap_info in self.bootstrap_infos:
            # struct.pack("Q", ptr) 将 ptr 转换为 8 字节 long long 形式二进制表达
            packed_kv_data_ptrs = b"".join(
                struct.pack("Q", ptr) for ptr in self.kv_mgr.kv_args.kv_data_ptrs
            )
            packed_aux_data_ptrs = b"".join(
                struct.pack("Q", ptr) for ptr in self.kv_mgr.kv_args.aux_data_ptrs
            )
            packed_state_data_ptrs = b"".join(
                struct.pack("Q", ptr) for ptr in self.kv_mgr.kv_args.state_data_ptrs
            )

            # Note(shangming): No need to add pp rank here since pp is not supported on the decode side yet
            tp_rank = self.kv_mgr.kv_args.engine_rank
            kv_item_len = self.kv_mgr.kv_args.kv_item_lens[0]

            # NOTE(james): 这里站在 prefill 视角做准备, dst 是 decode node
            dst_tp_rank = str(tp_rank).encode("ascii")
            dst_attn_tp_size = str(self.kv_mgr.attn_tp_size).encode("ascii")
            dst_kv_item_len = str(kv_item_len).encode("ascii")

            # NOTE(james):
            #   如果 scheduler 使用了线程池、协程或由多个 Callback 并触发了这两个 init 调用
            #   这是就会发生线程竞争, 导致 zmq 崩溃, 这才是 lock 发挥保障性的地方.
            sock, lock = self._connect_to_bootstrap_server(bootstrap_info)
            with lock:
                sock.send_multipart(
                    [
                        "None".encode("ascii"),
                        self.kv_mgr.local_ip.encode("ascii"),  # decode node 的 ip 地址
                        str(self.kv_mgr.rank_port).encode(
                            "ascii"
                        ),  # decode node 的 rank_port 端口号
                        self.session_id.encode(
                            "ascii"
                        ),  # decode node 的 mooncake_session_id
                        packed_kv_data_ptrs,
                        packed_aux_data_ptrs,
                        packed_state_data_ptrs,
                        dst_tp_rank,  # decode node 的 attn_tp_rank
                        dst_attn_tp_size,  # decode node 的 attn_tp_size
                        dst_kv_item_len,
                    ]
                )

    # NOTE(james): decode 阶段执行, 完成 decode node 向 prefill pod 发送具体的 kvcache 请求的信息.
    def init(
        self,
        kv_indices: npt.NDArray[np.int32],
        aux_index: Optional[int] = None,
        state_indices: Optional[List[int]] = None,
    ):
        if self.bootstrap_infos is None:
            self.kv_mgr.record_failure(
                self.bootstrap_room,
                f"Could not fetch prefill parallel info from bootstrap_addr: {self.bootstrap_addr}",
            )
            self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
            return

        for bootstrap_info in self.bootstrap_infos:
            sock, lock = self._connect_to_bootstrap_server(bootstrap_info)
            is_dummy = bootstrap_info["is_dummy"]

            with lock:
                sock.send_multipart(
                    [
                        str(self.bootstrap_room).encode("ascii"),
                        self.kv_mgr.local_ip.encode("ascii"),
                        str(self.kv_mgr.rank_port).encode("ascii"),
                        self.session_id.encode("ascii"),
                        kv_indices.tobytes() if not is_dummy else b"",
                        str(aux_index).encode("ascii") if not is_dummy else b"",
                        (
                            np.array(
                                state_indices,
                                dtype=np.int32,
                            ).tobytes()
                            if not is_dummy and state_indices is not None
                            else b""
                        ),
                        str(self.required_dst_info_num).encode("ascii"),
                    ]
                )
        self.init_time = time.time()

    def poll(self) -> KVPoll:
        if self.conclude_state is None:
            status = self.kv_mgr.check_status(self.bootstrap_room)
            if status in (KVPoll.Success, KVPoll.Failed):
                self.conclude_state = status
            elif status == KVPoll.WaitingForInput:
                if self.init_time is not None:
                    now = time.time()
                    elapsed = now - self.init_time
                    if elapsed >= self.kv_mgr.waiting_timeout:
                        logger.warning_once(
                            "Some requests fail to receive KV Cache transfer done signal after bootstrapping. "
                            "If a greater mean TTFT is acceptable, you can 'export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=600' (10 minutes) to relax the timeout condition. "
                        )
                        self.kv_mgr.record_failure(
                            self.bootstrap_room,
                            f"Request {self.bootstrap_room} timed out after {elapsed:.1f}s in KVPoll.WaitingForInput",
                        )
                        self.conclude_state = KVPoll.Failed
                        return KVPoll.Failed

            return status

        else:
            return self.conclude_state

    def clear(self) -> None:
        if self.bootstrap_room in self.kv_mgr.request_status:
            self.kv_mgr.request_status.pop(self.bootstrap_room)

        if self.bootstrap_room in self.kv_mgr.required_prefill_response_num_table:
            self.kv_mgr.required_prefill_response_num_table.pop(self.bootstrap_room)

        if self.bootstrap_room in self.kv_mgr.prefill_response_tracker:
            self.kv_mgr.prefill_response_tracker.pop(self.bootstrap_room)

    def failure_exception(self):
        # Explicitly set the status to failure since this request has failed in another rank
        if self.conclude_state is None:
            self.conclude_state = KVPoll.Failed

        self.clear()

        with self.kv_mgr.failure_lock:
            failure_reason = self.kv_mgr.failure_records.pop(
                self.bootstrap_room, "Failed due to an unknown reason from another rank"
            )
        raise KVTransferError(self.bootstrap_room, failure_reason)

    def abort(self):
        self.kv_mgr.record_failure(
            self.bootstrap_room,
            "Aborted by AbortReq.",
        )
        # Explicitly set the status to failure since this request has been aborted
        self.conclude_state = KVPoll.Failed


class MooncakeKVBootstrapServer(CommonKVBootstrapServer):
    pass
