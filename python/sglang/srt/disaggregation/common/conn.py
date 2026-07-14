from __future__ import annotations

import asyncio
import logging
import socket
import threading
from functools import cache
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import requests
import zmq
from aiohttp import web

from sglang.srt.disaggregation.base.conn import (
    BaseKVBootstrapServer,
    BaseKVManager,
    BaseKVReceiver,
    BaseKVSender,
    KVArgs,
    KVPoll,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.distributed import get_pp_group
from sglang.srt.layers.dp_attention import (
    get_attention_dp_rank,
    get_attention_dp_size,
    get_attention_tp_rank,
    get_attention_tp_size,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import (
    format_tcp_address,
    get_local_ip_auto,
    get_zmq_socket_on_host,
    is_valid_ipv6_address,
    maybe_wrap_ipv6_address,
)

logger = logging.getLogger(__name__)


########################################################################
class CommonKVManager(BaseKVManager):
    def __init__(
        self,
        args: KVArgs,
        disaggregation_mode: DisaggregationMode,
        server_args: ServerArgs,
        is_mla_backend: Optional[bool] = False,
    ):
        self.kv_args = args
        self.is_mla_backend = is_mla_backend
        self.disaggregation_mode = disaggregation_mode
        # for p/d multi node infer

        # 这 2 个成员的作用: 制成 bootstrap_server_url
        self.bootstrap_host = server_args.host  # default: 0.0.0.0
        self.bootstrap_port = server_args.disaggregation_bootstrap_port  # default: 8998
        self.dist_init_addr = server_args.dist_init_addr
        self.attn_tp_size = get_attention_tp_size()
        self.attn_tp_rank = get_attention_tp_rank()
        self.attn_dp_size = get_attention_dp_size()
        self.attn_dp_rank = get_attention_dp_rank()
        self.system_dp_size = (
            1 if server_args.enable_dp_attention else server_args.dp_size
        )
        self.system_dp_rank = (
            self.kv_args.system_dp_rank if self.kv_args.system_dp_rank else 0
        )
        self.pp_size = server_args.pp_size
        self.pp_rank = self.kv_args.pp_rank

        # NOTE(james): ip 理解成机器的地址, rank_port 理解成机器的临时端口. 用来区分机器上的不同 rank
        self.local_ip = get_local_ip_auto()

        # bind zmq socket
        context = zmq.Context()
        zmq_bind_host = maybe_wrap_ipv6_address(self.local_ip)

        f"""
            server_socket: ZMQ PULL socket, 可理解为「只收不发的信箱」.
            本节点 bind(local_ip, rank_port) 后, decode/prefill 节点用 PUSH 连过来「投递」数据,
            本节点只通过此 socket 接收, 不通过它发送。PUSH-PULL 是单向流水线: 对方推，我方接.
            若为 IPv6 地址，则显式开启 ZMQ 的 IPv6 支持.
        """
        self.rank_port, self.server_socket = get_zmq_socket_on_host(
            context, zmq.PULL, host=zmq_bind_host
        )
        logger.debug(f"kv manager bind to {zmq_bind_host}:{self.rank_port}")

        """
            PD 公有的内容, 创建 bootstrap room 和 poll.status 的映射关系.
            - key: bootstrap room, mass level 的 request id.
            - val: KVPoll status 状态
        """
        self.request_status: Dict[int, KVPoll] = {}

        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            self._register_to_bootstrap()

            """
                - key : int (bootstrap room),
                - val : dict[str, TransferInfo]
                    - key : str(mooncake_session_id), Prefill pod IP (单卡 level)
                    - val : TransferInfo
                - transfer_infos: 拿到具体的 kvcache relative location 信息.
            """
            self.transfer_infos = {}

            """
                首次握手时，拿到 decode node 的 kvcache buffer addr, stride 等信息
                - key : str(mooncake_session_id), decode pod IP (单卡 level)
                - val : KVArgsRegisterInfo
            """
            self.decode_kv_args_table = {}
            self.pp_group = get_pp_group()
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            """
            # 创建 connection pool, 用于存储 prefill 节点的信息.
            # 创建 connection lock, 用于保护 connection pool 的线程安全.
                - key: bootstrap_key
                - val: prefill_node -> ip:port

            注: connection_pool,
                prefill_attn_tp_size_table
                prefill_dp_size_table
                prefill_pp_size_table

              - 这 4 个 table 都为了快速查找 req 对应的 target prefill node 的 ip:port 信息.
            """
            self.connection_pool: Dict[str, Dict[str, Union[str, int]]] = {}
            self.connection_lock = threading.Lock()

            """
                用 bootstrap room 来区分不同请求对应的 prefill node 数量.
                - key: bootstrap room
                - val: required_prefill_response_num
            """
            self.required_prefill_response_num_table: Dict[int, int] = {}

            """
                用 bootstrap_addr 来区分不同
                    prefill_attn_tp_size_table, bootstrap_addr -> attn_tp_size
                    prefill_dp_size_table,      bootstrap_addr -> dp_size
                    prefill_pp_size_table,      bootstrap_addr -> pp_size
                - key : bootstrap_addr
                - val : int(prefill_attn_tp_size)  prefill node 的 attention 张量并行大小
            """
            self.prefill_attn_tp_size_table: Dict[str, int] = {}
            self.prefill_dp_size_table: Dict[str, int] = {}
            self.prefill_pp_size_table: Dict[str, int] = {}
        else:
            raise ValueError(
                f"Unsupported DisaggregationMode: {self.disaggregation_mode}"
            )

    """
        # Payload: 包含 Prefill 节点的完整信息，用于注册到 bootstrap server
        # 功能: 告知 bootstrap server 当前 Prefill 节点的并行配置和网络地址信息
        # - role: 节点角色，这里是 "Prefill"
        # - attn_tp_size/attn_tp_rank: 注意力层的张量并行大小和当前节点的张量并行排名
        # - attn_dp_size/attn_dp_rank: 注意力层的数据并行大小和当前节点的数据并行排名
        # - pp_size/pp_rank: 流水线并行大小和当前节点的流水线并行排名
        # - system_dp_size/system_dp_rank: 系统级数据并行大小和排名
    """

    def _register_to_bootstrap(self):
        """Register KVSender to bootstrap server via HTTP POST."""
        if self.dist_init_addr:
            # Multi-node case: bootstrap server's host is dist_init_addr
            if self.dist_init_addr.startswith("["):  # [ipv6]:port or [ipv6]
                if self.dist_init_addr.endswith("]"):
                    host = self.dist_init_addr
                else:
                    host, _ = self.dist_init_addr.rsplit(":", 1)
            else:
                host = socket.gethostbyname(self.dist_init_addr.rsplit(":", 1)[0])
        else:
            # Single-node case: bootstrap server's host is the same as http server's host
            f"""
                这个 bootstrap server 的功能: 用于 Prefill 节点向 bootstrap server
                注册自己的信息, 找到 bootstrap server 的地址, 并搭建 url 传输 Payload.

                NOTE(james): 理解这里, bootstrap_server_url 就是这个节点本身
                1. host: 线上默认 0.0.0.0, 许从任何网络接口访问该服务.
                2. bootstrap_port: 默认值 8998
                3. 构建 url 向 CommonBootstrapServer 发送 url 注册自身的 Prefill node 信息
            """
            host = self.bootstrap_host
            host = maybe_wrap_ipv6_address(host)

        bootstrap_server_url = f"{host}:{self.bootstrap_port}"  # bootstrap_port: 8998
        url = f"http://{bootstrap_server_url}/route"  # 创建 url 仅是手段.
        payload = {
            "role": "Prefill",
            "attn_tp_size": self.attn_tp_size,
            "attn_tp_rank": self.attn_tp_rank,
            "attn_dp_size": self.attn_dp_size,  # 1
            "attn_dp_rank": self.attn_dp_rank,  # 0
            "pp_size": self.pp_size,
            "pp_rank": self.pp_rank,
            "system_dp_size": self.system_dp_size,
            "system_dp_rank": self.system_dp_rank,
            "rank_ip": self.local_ip,
            "rank_port": self.rank_port,  # rank_ip/rank_port: 当前节点的 IP 地址和端口，供 Decode 节点连接使用
        }

        # 核心代码.
        try:
            """
            # 功能：将当前 Prefill 节点的并行配置、网络地址等信息注册到 bootstrap server
            - Payload:
                向 bootstrap server 发送 HTTP PUT 请求, 注册当前 Prefill 节点的信息,
                Decode 节点就可以通过 bootstrap server 发现并连接到对应的 Prefill 节点.
            """
            response = requests.put(url, json=payload, timeout=5)
            if response.status_code == 200:
                logger.debug("Prefill successfully registered to bootstrap server.")
            else:
                logger.error(
                    f"Prefill instance failed to connect to bootstrap server: {response.status_code}, {response.text}"
                )
        except Exception as e:
            logger.error(
                f"Prefill instance failed to register to bootstrap server: {e}"
            )

    @cache
    def _connect(self, endpoint: str, is_ipv6: bool = False):
        socket = zmq.Context().socket(zmq.PUSH)
        if is_ipv6:
            socket.setsockopt(zmq.IPV6, 1)
        socket.connect(endpoint)
        return socket

    def get_mha_kv_ptrs_with_pp(
        self, src_kv_ptrs: List[int], dst_kv_ptrs: List[int]
    ) -> Tuple[List[int], List[int], List[int], List[int], int]:
        # pp is not supported on the decode side yet
        start_layer = self.kv_args.prefill_start_layer
        num_kv_layers = len(src_kv_ptrs) // 2
        end_layer = start_layer + num_kv_layers
        dst_num_total_layers = len(dst_kv_ptrs) // 2

        src_k_ptrs = src_kv_ptrs[:num_kv_layers]  # 涵盖了 Eagle layer
        src_v_ptrs = src_kv_ptrs[num_kv_layers:]
        dst_k_ptrs = dst_kv_ptrs[start_layer:end_layer]
        dst_v_ptrs = dst_kv_ptrs[
            dst_num_total_layers + start_layer : dst_num_total_layers + end_layer
        ]
        layers_current_pp_stage = len(src_k_ptrs)
        return src_k_ptrs, src_v_ptrs, dst_k_ptrs, dst_v_ptrs, layers_current_pp_stage

    def get_mla_kv_ptrs_with_pp(
        self, src_kv_ptrs: List[int], dst_kv_ptrs: List[int]
    ) -> Tuple[List[int], List[int], int]:
        # pp is not supported on the decode side yet
        start_layer = self.kv_args.prefill_start_layer
        end_layer = start_layer + len(src_kv_ptrs)
        sliced_dst_kv_ptrs = dst_kv_ptrs[start_layer:end_layer]
        layers_current_pp_stage = len(src_kv_ptrs)
        return src_kv_ptrs, sliced_dst_kv_ptrs, layers_current_pp_stage


f"""
    bootstrap_host 来源于 mass 服务的配置,
    self.bootstrap_port 是预设的常量, 8998
"""


class CommonKVSender(BaseKVSender):
    def __init__(
        self,
        mgr: BaseKVManager,
        bootstrap_addr: str,  # bootstrap_addr 对应 req.bootstrap_host : self.bootstrap_port
        bootstrap_room: int,  # bootstrap_room 对应 req.id, mass level 的赋值, 全局唯一
        dest_tp_ranks: List[int],
        pp_rank: int,
    ):
        self.kv_mgr = mgr
        self.bootstrap_room = bootstrap_room
        self.aux_index = None
        self.bootstrap_server_url = bootstrap_addr

        # inner state
        self.curr_idx = 0

        """
            NOTE(james): Prefill 端
            1. 初始化设置这个 bootstrap_room 对应的 KVPoll::Bootstrapping
            2. 完整的接收到 decode 发送来的 Req level TransferInfo 后, 将状态更新为 KVPoll::WaitingForInput
        """
        self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Bootstrapping)

    def init(self, num_kv_indices: int, aux_index: Optional[int] = None):
        self.num_kv_indices = num_kv_indices  # 这个是传输 kvcache 的数量
        self.aux_index = aux_index  # meta_data_buffer idx

    def send(
        self,
        kv_indices: npt.NDArray[np.int32],
        state_indices: Optional[List[int]] = None,
    ):
        pass

    def poll(self) -> KVPoll:
        pass

    def failure_exception(self):
        raise Exception("Fake KVReceiver Exception")


########################################################################
class CommonKVReceiver(BaseKVReceiver):
    _ctx = zmq.Context()
    _socket_cache = {}
    _socket_locks = {}
    _global_lock = threading.Lock()

    def __init__(
        self,
        mgr: BaseKVManager,
        bootstrap_addr: str,
        bootstrap_room: Optional[int] = None,
        prefill_dp_rank: Optional[int] = None,
    ):
        self.bootstrap_room = bootstrap_room  # 由 mass 填充的 req.id
        self.bootstrap_addr = bootstrap_addr  # bootstrap_addr -> req.bootstrap_host:self.bootstrap_port, 但 bootstrap_port 没用.
        self.kv_mgr = mgr

        """
            NOTE(james):
            - target 就代表 Prefill pod
            - dst 就代表 Decode pod
        """

        """
            NOTE(james): Decode 端的初始化流程:
            1. 将当前 bootstrap_room 的状态设置为 KVPoll.Bootstrapping, 表示正在引导连接;
            2. runtime 阶段, 获取需要连接的 prefill node {ip:rank}, 确认 socket 连接信息, 服务于后续的 socket 的传输 PUSH;
                  完成 KVPoll.Bootstrapping -> KVPoll.WaitingForInput 的状态转变;
            3. 初始化阶段, 完成 prefill node 的 KVArgsRegisterInfo 注册;
        """
        self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Bootstrapping)

        # @cache
        if self.bootstrap_addr not in self.kv_mgr.prefill_dp_size_table:
            (
                self.prefill_attn_tp_size,
                self.prefill_dp_size,
                self.prefill_pp_size,
            ) = self._get_prefill_parallel_info_from_server()
            if (
                self.prefill_attn_tp_size is None
                or self.prefill_dp_size is None
                or self.prefill_pp_size is None
            ):
                self.kv_mgr.record_failure(
                    self.bootstrap_room,
                    f"Could not fetch prefill parallel info from bootstrap_addr: {self.bootstrap_addr}",
                )
                self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
                self.bootstrap_infos = None
                return
            else:
                logger.debug(
                    f"Fetch prefill parallel info from [{self.bootstrap_addr}]: DP size:{self.prefill_dp_size}, TP size:{self.prefill_attn_tp_size} PP size:{self.prefill_pp_size}"
                )
                self.kv_mgr.prefill_attn_tp_size_table[self.bootstrap_addr] = (
                    self.prefill_attn_tp_size
                )
                self.kv_mgr.prefill_dp_size_table[self.bootstrap_addr] = (
                    self.prefill_dp_size
                )
                self.kv_mgr.prefill_pp_size_table[self.bootstrap_addr] = (
                    self.prefill_pp_size
                )
        else:
            self.prefill_attn_tp_size = self.kv_mgr.prefill_attn_tp_size_table[
                self.bootstrap_addr
            ]
            self.prefill_dp_size = self.kv_mgr.prefill_dp_size_table[
                self.bootstrap_addr
            ]
            self.prefill_pp_size = self.kv_mgr.prefill_pp_size_table[
                self.bootstrap_addr
            ]

        # Currently, we don't allow prefill instance and decode instance to
        # have different TP sizes per DP rank, except for models using MLA.
        if self.kv_mgr.attn_tp_size == self.prefill_attn_tp_size:
            self.target_tp_rank = (
                self.kv_mgr.kv_args.engine_rank % self.kv_mgr.attn_tp_size
            )
            self.required_dst_info_num = 1
            self.required_prefill_response_num = 1 * (
                self.prefill_pp_size // self.kv_mgr.pp_size
            )

            # 哥们装的是 prefill node 对应的 ranks
            self.target_tp_ranks = [self.target_tp_rank]
        elif self.kv_mgr.attn_tp_size > self.prefill_attn_tp_size:
            """
            # 2. decode_attn_tp_size > prefill_attn_tp_size 的情况,
            #    需要从单个 prefill ranks 拉取 KVCache 数据.
            """
            if not self.kv_mgr.is_mla_backend:
                logger.warning_once(
                    "Performance is NOT guaranteed when using different TP sizes for non-MLA models. "
                )
            self.target_tp_rank = (
                self.kv_mgr.kv_args.engine_rank % self.kv_mgr.attn_tp_size
            ) // (self.kv_mgr.attn_tp_size // self.prefill_attn_tp_size)
            self.required_dst_info_num = (
                self.kv_mgr.attn_tp_size // self.prefill_attn_tp_size
            )
            self.required_prefill_response_num = 1 * (
                self.prefill_pp_size // self.kv_mgr.pp_size
            )

            # 哥们装的是 prefill node 对应的 ranks
            self.target_tp_ranks = [self.target_tp_rank]
        else:
            """
            # 3. decode_attn_tp_size < prefill_attn_tp_size 的情况,
            #    需要从多个 prefill ranks 拉取 KVCache 数据并进行合并.
            """
            if not self.kv_mgr.is_mla_backend:
                logger.warning_once(
                    "Performance is NOT guaranteed when using different TP sizes for non-MLA models. "
                )
            # For non-MLA models, one decode rank needs to retrieve KVCache from multiple prefill ranks for non MLA models;
            # 哥们装的是 prefill node 对应的 ranks
            self.target_tp_ranks = [
                rank
                for rank in range(
                    (self.kv_mgr.kv_args.engine_rank % self.kv_mgr.attn_tp_size)
                    * (self.prefill_attn_tp_size // self.kv_mgr.attn_tp_size),
                    (self.kv_mgr.kv_args.engine_rank % self.kv_mgr.attn_tp_size + 1)
                    * (self.prefill_attn_tp_size // self.kv_mgr.attn_tp_size),
                )
            ]

            # For MLA models, we can retrieve KVCache from only one prefill rank, but we still need to maintain
            # multiple connections in the connection pool and have to send dummy requests to other prefill ranks,
            # or the KVPoll will never be set correctly

            # target_tp_rank 这里就是完全为 mla 服务了. 设置 dummy
            self.target_tp_rank = self.target_tp_ranks[0]
            self.required_dst_info_num = 1
            if self.kv_mgr.is_mla_backend:
                self.required_prefill_response_num = (
                    self.prefill_pp_size // self.kv_mgr.pp_size
                )
            else:
                self.required_prefill_response_num = (
                    self.prefill_attn_tp_size // self.kv_mgr.attn_tp_size
                ) * (self.prefill_pp_size // self.kv_mgr.pp_size)

        if prefill_dp_rank is not None:
            logger.debug(f"Targeting DP rank: {prefill_dp_rank}")
            self.prefill_dp_rank = prefill_dp_rank
        else:
            self.prefill_dp_rank = bootstrap_room % self.prefill_dp_size

        # FIXME: alias here: target_dp_group -> prefill_dp_rank
        self.target_dp_group = self.prefill_dp_rank

        self.kv_mgr.required_prefill_response_num_table[self.bootstrap_room] = (
            self.required_prefill_response_num
        )
        # NOTE: key distinguished by bootstrap_addr, target_dp_group, and target_tp_rank

        """
            NOTE(james): key distinguished by bootstrap_addr, target_dp_group, and target_tp_rank.
                 connection_pool 构建 req 至 target prefill pod 的 ip:port 映射关系, 供后续发送 KVCache 数据使用.
        """
        bootstrap_key = (
            f"{self.bootstrap_addr}_{self.target_dp_group}_{self.target_tp_rank}"
        )

        if bootstrap_key not in self.kv_mgr.connection_pool:
            bootstrap_infos = []
            for target_tp_rank in self.target_tp_ranks:
                for target_pp_rank in range(self.prefill_pp_size):
                    bootstrap_info = self._get_bootstrap_info_from_server(
                        target_tp_rank, self.target_dp_group, target_pp_rank
                    )
                    if bootstrap_info is not None:
                        if self.kv_mgr.is_mla_backend:
                            # For MLA: target_tp_rank is the selected real rank, others are dummy ranks
                            bootstrap_info["is_dummy"] = not bool(
                                target_tp_rank == self.target_tp_rank
                                or self.target_tp_rank is None
                            )
                        else:
                            # For non-MLA: all target_tp_ranks are selected real ranks
                            bootstrap_info["is_dummy"] = False
                        logger.debug(
                            f"Fetched bootstrap info: {bootstrap_info} for DP {self.target_dp_group} TP {target_tp_rank} PP {target_pp_rank}"
                        )
                        bootstrap_infos.append(bootstrap_info)
                    else:
                        self.kv_mgr.record_failure(
                            self.bootstrap_room,
                            f"Could not fetch bootstrap info for engine rank: {self.kv_mgr.kv_args.engine_rank} and target_dp_group: {self.target_dp_group} and target_pp_rank {target_pp_rank}",
                        )
                        self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
                        return

            self.bootstrap_infos = bootstrap_infos
            self.kv_mgr.connection_pool[bootstrap_key] = self.bootstrap_infos

            # Register kv_args only once to prefill KVManager according to the info fetched from the bootstrap server
            """
            # 真正的首次握手, 注册当前 decode node 的并行配置到 kv_mgr 中, 供 prefill node 查询使用.
            """
            self._register_kv_args()
        else:
            self.bootstrap_infos = self.kv_mgr.connection_pool[bootstrap_key]

        assert len(self.bootstrap_infos) > 0

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

    # 初始化握手, 获取 prefill node 的并行配置
    def _get_prefill_parallel_info_from_server(
        self,
    ) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        """Fetch the prefill parallel info from the bootstrap server."""
        try:
            url = f"http://{self.bootstrap_addr}/route?engine_rank={-1}&target_dp_group={-1}&target_pp_rank={-1}"
            response = requests.get(url)
            if response.status_code == 200:
                prefill_parallel_info = response.json()
                return (
                    int(prefill_parallel_info["prefill_attn_tp_size"]),
                    int(prefill_parallel_info["prefill_dp_size"]),
                    int(prefill_parallel_info["prefill_pp_size"]),
                )
            else:
                logger.error(
                    f"Failed to get prefill parallel info: {response.status_code}, {response.text}"
                )
                return None, None, None
        except Exception as e:
            logger.error(f"Error fetching prefill parallel info from bootstrap: {e}")
            return None, None, None

    @classmethod
    def _connect(cls, endpoint: str, is_ipv6: bool = False):
        with cls._global_lock:
            if endpoint not in cls._socket_cache:
                sock = cls._ctx.socket(zmq.PUSH)
                if is_ipv6:
                    sock.setsockopt(zmq.IPV6, 1)

                # 连接到 prefill pod 的 ZMQ PULL socket, 供后续发送 KVCache 数据使用.
                # 这里的连接是持久化的, 连接成功后会被缓存到 _socket_cache 中.
                sock.connect(endpoint)
                cls._socket_cache[endpoint] = sock
                cls._socket_locks[endpoint] = (
                    threading.Lock()
                )  # 这里的锁更多是为了稳定性保障
            return cls._socket_cache[endpoint], cls._socket_locks[endpoint]

    @classmethod
    def _connect_to_bootstrap_server(cls, bootstrap_info: dict):
        ip_address = bootstrap_info["rank_ip"]  # prefill pod 的 ip 地址;
        port = bootstrap_info["rank_port"]  # prefill pod 的 rank 地址;
        is_ipv6_address = is_valid_ipv6_address(ip_address)
        sock, lock = cls._connect(
            format_tcp_address(ip_address, port), is_ipv6=is_ipv6_address
        )
        return sock, lock

    def _register_kv_args(self):
        pass

    def failure_exception(self):
        raise Exception("Fake KVReceiver Exception")


#########################################################################
"""
    功能:
    1. CommonKVBootstrapServer 作为一个独立的 HTTP 服务器, 负责协调 Prefill 和 Decode 实例之间的连接信息交换.
    2. prefill node 上启动这个 Server, 并向 Server 注册自己的 parallel setting,
    3. decode  node 上的 KVReceiver 启动时, 从这个 Server 获取对应 prefill node 的连接信息.
"""


class CommonKVBootstrapServer(BaseKVBootstrapServer):
    def __init__(self, host: str, port: int):
        """
        1. host: 0.0.0.0
        2. port: 8998
        """
        self.host = host
        self.port = port
        self.app = web.Application()
        self.store = dict()
        self.lock = asyncio.Lock()
        self._setup_routes()
        self.pp_size = None
        self.attn_tp_size = None
        self.dp_size = None

        f"""
            - key: dp_group
            - val:
                - key: attn_tp_rank
                - val:
                    - key: pp_rank
                    - val: rank_ip, rank_port

            注: [dp_group][attn_tp_rank][pp_rank] = {
                 "rank_ip": rank_ip,
                 "rank_port": rank_port
            }
        """
        self.prefill_port_table: Dict[
            int, Dict[int, Dict[int, Dict[str, Union[str, int]]]]
        ] = {}

        # Start bootstrap server
        self.thread = threading.Thread(target=self._run_server, daemon=True)
        self.run()

    def run(self):
        self.thread.start()

    def _setup_routes(self):
        """
        目的:
        1. 将 /route 和 /health 注册到 app 的路由表

        其他:
        1. * 代表 url 的通配符, 代表任意路径
        """
        self.app.router.add_route("*", "/route", self._handle_route)
        self.app.router.add_get("/health", self._handle_health_check)

    async def _handle_health_check(self, request):
        return web.Response(text="OK", status=200)

    async def _handle_route(self, request: web.Request):
        method = request.method
        if method == "PUT":
            return await self._handle_route_put(request)
        elif method == "GET":
            return await self._handle_route_get(request)
        else:
            return web.Response(
                text="Method not allowed", status=405, content_type="application/json"
            )

    """
        1. 现在 prefill node 上有个 8 个sche 进程, 外加一个 tokenizer_manager 进程;
        2. prefill port table 是主进程的 prefill port 信息, 记录了每个 sche 进程的 ip:port 信息.
    """

    async def _handle_route_put(self, request: web.Request):
        data = await request.json()
        role = data["role"]
        attn_tp_size = data["attn_tp_size"]
        attn_tp_rank = data["attn_tp_rank"]
        attn_dp_size = data["attn_dp_size"]
        attn_dp_rank = data["attn_dp_rank"]
        pp_size = data["pp_size"]
        pp_rank = data["pp_rank"]
        system_dp_size = data["system_dp_size"]
        system_dp_rank = data["system_dp_rank"]
        rank_ip = data["rank_ip"]
        rank_port = int(data["rank_port"])

        if self.attn_tp_size is None:
            self.attn_tp_size = attn_tp_size

        if self.dp_size is None:
            self.dp_size = attn_dp_size if system_dp_size == 1 else system_dp_size

        if self.pp_size is None:
            self.pp_size = pp_size

        if role == "Prefill":
            if system_dp_size == 1:
                dp_group = attn_dp_rank
            else:
                dp_group = system_dp_rank

            # Add lock to make sure thread-safe
            async with self.lock:
                if dp_group not in self.prefill_port_table:
                    self.prefill_port_table[dp_group] = {}
                if attn_tp_rank not in self.prefill_port_table[dp_group]:
                    self.prefill_port_table[dp_group][attn_tp_rank] = {}

            f"""
                prefill node 将自己的 rank_ip 和 rank_port 注册到 bootstrap server 上,
                以供 decode node 查询.
            """
            self.prefill_port_table[dp_group][attn_tp_rank][pp_rank] = {
                "rank_ip": rank_ip,  # 某个机器的 rank_ip, 非 0.0.0.0
                "rank_port": rank_port,  # 某个机器的 rank_port, 非 8998
            }
            logger.debug(
                f"Register prefill bootstrap: DP{dp_group} TP{attn_tp_rank} PP{pp_rank} with rank_ip: {rank_ip} and rank_port: {rank_port}"
            )

        return web.Response(text="OK", status=200)

    async def _handle_route_get(self, request: web.Request):
        engine_rank = request.query.get("engine_rank")
        target_dp_group = request.query.get("target_dp_group")
        target_pp_rank = request.query.get("target_pp_rank")
        if not engine_rank or not target_dp_group or not target_pp_rank:
            return web.Response(text="Missing inputs for bootstrap server.", status=400)

        # Currently we use engine_rank == -1 and target_dp_group == -1 to sync dp size
        if (
            int(engine_rank) == -1
            and int(target_dp_group) == -1
            and int(target_pp_rank) == -1
        ):
            prefill_parallel_info = {
                "prefill_attn_tp_size": self.attn_tp_size,
                "prefill_dp_size": self.dp_size,
                "prefill_pp_size": self.pp_size,
            }
            return web.json_response(prefill_parallel_info, status=200)

        # Find corresponding prefill info
        async with self.lock:
            bootstrap_info = self.prefill_port_table[int(target_dp_group)][
                int(engine_rank)
            ][int(target_pp_rank)]

        if bootstrap_info is not None:
            return web.json_response(bootstrap_info, status=200)
        else:
            return web.Response(text="Bootstrap info not Found", status=404)

    f"""
        处理高并发请求:
        1. Bootstrap Server 端, 需要处理来自大量 Decode Node 的查询请求 (GET /route),
            和 Prefill 节点的注册请求 (PUT /route).
        2. 使用 Asyncio + aiohttp 可以利用协程, 非阻塞 IO 高效地并发处理这些 HTTP 请求,
            而不需要为每个链接, 创建一个线程, 对于高吞吐的元数据服务非常关键.
    """

    def _run_server(self):
        f"""
            1. 启用 self.app 的 aiohttp 服务器, 监听 host:port 上的 HTTP 请求.
            2. 用 AppRunner 和 TCPSite 来管理服务器的生命周期, 包括启动和清理资源.
            3. 用 asyncio 的事件循环来运行服务器, 以支持异步处理请求.
            4. 用 site.start() 来启动服务器, 并用 loop.run_forever() 来保持服务器运行, 直到手动停止.
        """
        try:
            # Event Loop
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

            access_log = None
            if logging.getLogger(__name__).getEffectiveLevel() <= logging.DEBUG:
                access_log = self.app.logger

            self._runner = web.AppRunner(self.app, access_log=access_log)
            self._loop.run_until_complete(self._runner.setup())

            site = web.TCPSite(self._runner, host=self.host, port=self.port)
            self._loop.run_until_complete(site.start())
            self._loop.run_forever()
        except Exception as e:
            logger.error(f"Server error: {str(e)}")
        finally:
            # Cleanup
            self._loop.run_until_complete(self._runner.cleanup())
            self._loop.close()

    def close(self):
        """Shutdown"""
        if self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
            logger.info("Stopping server loop...")

        if self.thread.is_alive():
            self.thread.join(timeout=2)
            logger.info("Server thread stopped")

    def poll(self) -> KVPoll: ...
