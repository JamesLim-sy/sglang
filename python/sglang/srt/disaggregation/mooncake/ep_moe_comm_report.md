# EP-MoE 通信量理论建模报告

**场景**：EP8，H800×8 单机（不跨机），DeepEP intranode-only kernel 路径，均衡路由假设
**目的**：为 EP-MoE vs TP-MoE 的性能对比补齐 EP-MoE 侧的通信量理论模型（计算侧已有独立模型，本报告不涉及）

---

## 1. Workload 与拓扑参数

| 参数 | 符号 | 取值 | 说明 |
|---|---|---|---|
| 完整请求长度 | S | 16384 (16k) | prefill 单请求总 token 数 |
| EP 度 / 卡数 | R | 8 | 单机 H800×8，纯 NVLink，无跨机 |
| 每卡本地 token 数 | n_local | S/R = 2048 (2k) | 序列在 EP 维度上均分 |
| top-k | topk | 8 | 与 R 数值相同（巧合，非必然） |
| 总专家数 | E_total | 144 | |
| 每卡本地专家数 | E_local | E_total/R = 18 | 暂不考虑冗余专家 |
| hidden size | H(dim) | 6144 | |
| dispatch 精度 | — | FP8 (1 Byte/elem) | 供后续 grouped GEMM |
| combine 精度 | — | BF16 (2 Byte/elem) | topk 份输出加权求和，需更高精度 |
| NVLink 有效带宽 | BW_eff | 200 GB/s | H800 理论 NVLink 400GB/s，按 busbw 口径取 200GB/s；本报告直接作为"有效收发带宽"使用，不再叠加 AllReduce 的 (R-1)/R 折算 |
| 层级经验负载不均衡度 | balanceness | 0.70–0.75 | `avg_load/max_load` 口径，实测得到；对应最热 rank 比平均多收 33%–43%，见第 8 节 |

---

## 2. 建模范围声明

- 本报告**只建模通信侧**（dispatch + combine 的数据量与耗时），Grouped GEMM 计算侧已有独立模型，不在此重复。
- 第 3–7 节建立的是**理想化均衡路由**下的模型：边际概率均匀、无系统性偏斜，只用期望值，不引入 imbalance factor。第 8 节在此基础上引入实测的经验不均衡参数 balanceness，得到贴近真实系统的修正公式——**两者不是互斥关系，是"理想基准 → 真实修正"的递进关系**。
- 拓扑对应 DeepEP 的 **intranode-only kernel 路径**（`intranode.cu` / `test_intranode.py`），不涉及之前多机场景下 NVLink+RDMA 两级聚合的 hierarchical 模型。
- 不考虑 DeepSeek 式 group-limited / node-limited routing 的硬约束（如果实际路由策略有此约束，第 4 节的超几何模型需要替换为"先选 group 再选 expert"的两层模型）。

---

## 3. 朴素模型（第一版，存在系统性高估）

### 3.1 思路

假设每个被选中的专家都对应一次独立的跨 rank 传输，即每个 token 的 topk 份拷贝均匀分布到 (R-1) 个远端 rank 上：

```
N_net(朴素) = n_local × topk × (R-1)/R
```

### 3.2 代入数字

```
N_net = 2048 × 8 × 7/8 = 14336

V_dispatch = 14336 × 6144 × 1B  ≈ 88.08 MB
V_combine  = 14336 × 6144 × 2B  ≈ 176.16 MB

T_dispatch = 88.08MB / 200GB/s ≈ 440.4 μs
T_combine  = 176.16MB / 200GB/s ≈ 880.8 μs
T_comm_EP(朴素) ≈ 1321.2 μs
```

### 3.3 问题所在

该模型隐含假设"每个 rank 被命中的概率恒为 1"（即期望命中次数 = 1 ⟹ 必然命中）。但实际决定通信量的是 **(token, rank) 粒度上"是否命中"的 0/1 事件**，而非"命中次数"——若一个 token 的多个被选中专家恰好落在同一个 rank 上，该 rank 只需接收一份数据。由 Jensen 不等式，`P(count≥1) ≤ E[count]`，朴素模型必然高估通信量。

---

## 4. 修正模型：超几何 fanout

### 4.1 前置条件（两条必须同时成立）

**条件 A（结构性，top-k 机制自动满足，不是假设）**
标准 top-k 路由（对 E_total 个专家打分后取 top-k 索引）在数学上等价于从 E_total 个元素中**无放回**抽取 topk 个,任意两次选择必然指向不同专家。这是 top-k 算子的数学事实,不依赖打分函数的具体形式,没有例外情形。

**条件 B（分布性，对应"路由均衡"的建模选择）**
C(E_total, topk) 种组合在统计意义上等概率出现,等价于任意专家子集被命中的概率只取决于子集大小,不取决于具体是哪些专家。这一条对应题设中显式排除的"路由不均衡",是需要在报告中声明、并承担相应风险的简化假设——若路由存在语义相关性导致的联合分布偏斜,这里需要重新建模。

只有 A + B 同时成立，下述超几何公式才是精确解。A 是免费的，B 是需要声明的建模选择。

### 4.2 命中概率与 remote fanout

```
P_hit = 1 - C(E_total - E_local, topk) / C(E_total, topk)
      = 1 - C(126, 8) / C(144, 8)
      = 1 - ∏_{i=0}^{7} (126-i)/(144-i)
      ≈ 1 - 0.33384
      ≈ 0.66616

F_remote = (R-1) × P_hit = 7 × 0.66616 ≈ 4.6631
```

对比朴素模型隐含的 `F_remote_naive = 7`，真实 fanout 只有约 **4.66**，折扣系数 ≈ P_hit ≈ 0.666。

### 4.3 修正后的通信量与耗时

```
N_net(修正) = n_local × F_remote = 2048 × 4.6631 ≈ 9550

V_dispatch = 9550 × 6144 × 1B  ≈ 58.68 MB   (原 88.08 MB)
V_combine  = 9550 × 6144 × 2B  ≈ 117.35 MB  (原 176.16 MB)

T_dispatch = 58.68MB / 200GB/s ≈ 293.4 μs   (原 440.4 μs)
T_combine  = 117.35MB / 200GB/s ≈ 586.8 μs  (原 880.8 μs)
T_comm_EP(修正) ≈ 880.1 μs                  (原 1321.2 μs，高估约 50%)
```

**该修正生效的前提**：dispatch/combine 的实现在 (token, rank) 粒度做了去重——同一 rank 命中多个本地专家时只传一份数据/只回传一份聚合部分和。这是 DeepEP 的 permute + all-to-all + unpermute 模式的标准做法；若实际 kernel 是逐 (token, expert) pair 发送（不去重），应退回第 3 节的朴素公式。**这是决定用 880μs 还是 1321μs 的关键分岔点，需要对照实际 kernel 实现确认。**

### 4.4 条件 B 的性质：既非上界也非下界，是无相关性假设下的中性点

条件 B（`C(E_total,topk)` 组合等概率）只保证边际负载均衡，不约束联合分布的相关结构。固定边际均值 `μ = topk×E_local/E_total`（本例 =1）后，`P_hit` 的真实理论可行区间是：

```
P_hit_min = μ / min(E_local, topk) = 1/min(18,8) = 1/8 = 0.125   （全部命中挤在同一张卡，两点分布极值）
P_hit_max = min(1, μ) = 1                                          （命中完全打散到不同卡）
```

条件 B 给出的 `P_hit≈0.666` 落在 `[0.125, 1]` 区间中偏上位置，是"零额外相关性"的中性估计，不是数学意义上的上界或下界。

**但在工程判断上，它可以被当作保守上界使用**：`P_hit_min=1/R` 精确对应 **device-limited/group-limited routing**（每 token 最多只能激活 1 个 group，DeepSeek V2/V3 等系统的做法）这一具体约束。这类约束的设计目的就是主动把选择往"扎堆"方向压缩，只会降低 `P_hit`，不存在把它推高过条件 B 的机制（约束是单向的：只收窄不放宽）。因此：

- 若目标系统采用了 group/device-limited routing，条件 B 算出的通信量（880μs）大概率是一个**只会高估、不会低估**的保守上界，用于容量规划或延迟预算是安全的。
- 若目标系统无此类约束，纯靠 auxiliary loss 保证边际均衡，条件 B 既可能高估也可能低估（取决于 expert-id 到物理 rank 的映射是否与语义共激活簇偶然重合），此时需要用生产流量实测校准，不能只依赖理论点估计。

> **关于单次执行的统计涨落**：理想化均衡假设下，有限样本(n_local=2048)带来的随机噪声量级约 ±2%（推导见第 6 节），但这一效应比第 8 节引入的实测系统性不均衡（balanceness=0.70–0.75，对应 +33%~+43%）小一个数量级，被后者完全覆盖，故不再单独作为修正项保留。

---

## 5. 渐近极限：压缩为两参数闭式

固定 (R, topk)，令 E_total → ∞（专家切分越来越细）：

```
P_hit(E_total) = 1 - C(E_total-E_local, topk)/C(E_total, topk)
   ──E_total→∞──→  P_hit(∞) = 1 - (1 - 1/R)^topk
```

代入 R=8, topk=8：

```
P_hit(∞) = 1 - (7/8)^8 ≈ 1 - 0.34361 ≈ 0.65639
```

对比精确值 `P_hit(144) ≈ 0.66616`，两者相差约 1.5%——这是无放回抽样相对有放回抽样的**有限总体修正**（无放回抽样保证 8 次选择不重复，覆盖效率更高，因此命中概率弱高于二项分布近似）。

**理论意义**：这把一个三参数 `(E_total, E_local, topk)` 的组合数问题压缩为两参数 `(R, topk)` 的解析闭式：

```
F_remote(R, topk) ≈ (R-1) × [1 - (1-1/R)^topk]
```

该闭式解释了两种极限场景的本质差异：
- **粗粒度 EP（E_local 较大，如本报告 E_local=18）**：碰撞概率显著，折扣明显（本例中 F_remote=4.66 << topk=8）
- **细粒度 EP 极限（E_local=1，一卡一专家）**：`P_hit = topk/E_total` 精确成立，无组合修正项，朴素模型与精确模型重合

**结论**：折扣大小完全由 E_local（每卡放几个专家）决定，E_local 越大折扣越明显，E_local=1 时折扣消失。Fine-grained expert 切分不仅是计算侧的设计选择，也直接压低 dispatch/combine 的网络开销。

---

## 6. 期望值近似的有效性检验

n_local=2048 个 token 独立试验，对某个 remote rank 命中次数的标准差：

```
σ = sqrt(n_local × P_hit × (1-P_hit)) = sqrt(2048 × 0.666 × 0.334) ≈ 21.4
均值 ≈ 2048 × 0.666 ≈ 1364
相对波动 ≈ 21.4 / 1364 ≈ 1.6%
```

在 n_local=2048 这一 prefill batch 量级下，用期望值代替真实分布的误差 ~1.6%，在**理想化均衡路由**前提下可接受。**该近似依赖 n_local × P_hit >> 1**；若换到 decode 阶段（单次仅几十 token/rank），该近似会明显失效，需改用真实分布建模（这也是 DeepEP 为 decode 单独设计 low-latency kernel + 独立建模范式的原因之一）。

这里的 ~1.6% 只是**理想化假设成立前提下**的随机噪声；真实系统里专家热度不均带来的系统性偏斜（第 8 节 balanceness）比这个量级大一个数量级，是实际建模时更应该关注的部分。

---

## 7. 最终通用公式

```
T_comm_EP(S) = (S/R) × F_remote(R,topk) × H(dim) × 3B / BW_eff

            = (S/R) × (R-1) × [1-(1-1/R)^topk] × H(dim) × 3B / BW_eff
```

（3B = FP8 的 1B + BF16 的 2B；如需更精细，dispatch/combine 应分开写，各自乘各自字节数，并可加入 FP8 block-wise scale 的元数据开销，量级约 3%）

该公式的价值在于显式揭示通信量对各参数的敏感度：
- **S、topk、H(dim)**：一次方线性敏感
- **R**：出现在 `F_remote` 内部及外层 `1/R`，EP 度越大单卡分摊通信量下降得比线性更快——这是 EP 扩展性的来源
- **E_total（通过 E_local）**：只在有限尺寸修正项中出现（<2%量级），细粒度 EP 下可用渐近闭式替代精确超几何计算

---

## 8. 引入经验负载不均衡参数 balanceness：从理想模型到真实修正

第 3–7 节的模型全部建立在"边际概率均匀、无系统性偏斜"的理想化假设上。本节引入实测得到的层级经验参数 balanceness，把模型拉回真实系统。

### 8.1 定义与数值换算

采用最常见口径 `balanceness = avg_load / max_load`：

```
balanceness=0.75  →  max/avg = 1/0.75 ≈ 1.333  →  最热 rank 比平均多收 ≈33.3%
balanceness=0.70  →  max/avg = 1/0.70 ≈ 1.429  →  最热 rank 比平均多收 ≈42.9%
```

### 8.2 与第 4.4 节理论区间的关系

这本质上是第 4.4 节讨论的"`P_hit` 偏离条件 B 中性点"在真实系统里的具体体现——只不过 4.4 节是从路由约束的构造方式（group-limited routing）推理可能偏向哪个方向，这里是直接测出来的、逐 layer 的经验值，把理论区间 `[1/R, 1]` 收敛到了一个具体数字。

### 8.3 一阶修正：按最热 rank 主导重估时延

集合通信的完成时间由最慢一路决定，热 rank 是 dispatch 的 recv 侧瓶颈，也同步是 combine 的 send 侧瓶颈（两者同源：接了多少 dispatch 就要送出多少 combine）：

```
T_dispatch(热rank主导) ≈ T_dispatch(第4.3节理想估计) / balanceness
T_combine(热rank主导)  ≈ T_combine(第4.3节理想估计) / balanceness
T_comm_EP(热rank主导)  ≈ T_comm_EP(第4.3节理想估计) / balanceness
```

代入数字（基准：第 4.3 节 T_comm_EP=880.1μs）：

```
                    balanceness=0.75      balanceness=0.70
T_dispatch(热rank)   293.4/0.75≈391.2μs    293.4/0.70≈419.1μs
T_combine(热rank)    586.8/0.75≈782.4μs    586.8/0.70≈838.3μs
T_comm_EP(热rank)    880.1/0.75≈1173.5μs   880.1/0.70≈1257.3μs
```

对比理想估计，实际瓶颈时延高出 33%–43%，比第 6 节的统计噪声（~1.6%~2%）大一个数量级。

### 8.4 "直接除以 balanceness"隐含的三个假设（使用限制，需显式声明）

这一步除法只是一阶近似，成立与否取决于：

1. **热 rank 能否以同一个 BW_eff 吃满增大后的负载**：如果瓶颈发生在接收端总吞吐（NVSwitch 全交叉架构，每卡只受自身总注入/吐出带宽限制），除法基本成立；如果瓶颈发生在具体某几条物理链路（点对点拓扑），热 rank 能否吃满 BW_eff 取决于具体链路带宽，不必然成立。
2. **balanceness 的统计口径**：若为全局静态口径（整层/整 batch 跑完后统计 max/avg），除法直接对应"这张 rank 处理完它那份数据总共要多久"，成立；若为逐 chunk 的动态口径，真实瓶颈取决于流水线里最忙的瞬时窗口，直接套用全局平均的 balanceness 可能不精确。
3. **忽略了计算侧的联动效应**：热 rank 的 Grouped GEMM 计算侧同步变重（收到更多 token，本地专家要处理更多行）。若通信与计算做了 chunk 级 overlap，两侧同时变慢，除法只处理了通信侧，未联动调整计算侧估计，容易在"overlap 之后整层到底多慢"这一问题上口径不一致。

### 8.5 标量丢失的结构信息

balanceness 是一个标量，不区分 dispatch 的 recv 侧偏重和 combine 的 send 侧偏重——理论上两者应该联动相等，但实测中常常是分开 profile 的，两个方向的数值不一定完全相等；用同一个数字套两个阶段是又一层简化。

### 8.6 使用建议

- 作为一阶量级估计（back-of-envelope），这个除法是合理、有方向性价值的，能正确抓住"瓶颈由最重一路决定"的直觉。
- 若要写入精确建模：应分别测 dispatch/combine 各自的 max/avg；确认统计口径是全局静态还是逐 chunk 动态；确认是否按 layer 分别取值（不同 layer 路由模式不同，balanceness 通常逐层变化，不是全模型统一常数）；有条件时直接用每个 rank 的实际 token 分布直方图，而非压缩后的标量反推。

---

## 9. 对 EP-MoE vs TP-MoE 对比的结构性含义

这是本次分析中最值得写入最终对比结论的一条：

- **EP-MoE 的通信量正比于 topk**：每个 token 的 topk 份拷贝都要物理经过网络送到对应 expert 所在 rank，再送回来。
- **TP-MoE 的通信量与 topk 无关**：TP-MoE 沿 d_ff 切分专家权重，每个 rank 本地算出所有 topk 个被选中专家的部分和，本地加权求和后只对最终 `[n_local, H]` 结果做**一次** all-reduce，与 topk 取值无关。

**推论一**：本场景 topk=8 恰好把 EP-MoE 的通信成本推高到接近甚至超过 TP-MoE 的量级；若 topk 更小（如 top-1/top-2），EP-MoE 的通信优势会显著扩大。这条"通信量对 topk 的敏感度差异"建议作为对比报告的核心结论单独列出，而非只给两个总时间数字。

**推论二（第 8 节引入后新增）**：EP-MoE 的通信时延对真实路由不均衡高度敏感（880μs → 1173~1257μs），而 TP-MoE 的 all-reduce 通信量与 token 具体路由到哪个专家无关，天然不受 balanceness 影响，是一个确定性常数。**这意味着做 EP vs TP 对比时，EP-MoE 一侧应该用第 8 节的 balanceness 修正值而非第 4.3 节的理想值去对比，否则会系统性高估 EP-MoE 的相对优势**。

---

## 10. 待确认 / 待补齐事项

| 事项 | 影响 | 状态 |
|---|---|---|
| dispatch/combine 是否在 (token, rank) 粒度去重 | 决定用第 4 节修正模型（880μs）还是第 3 节朴素模型（1321μs） | **待确认实现细节** |
| 是否存在 group-limited/node-limited routing 约束 | 若有，第 4 节超几何模型需替换为两层模型，fanout 会更小且有确定上界 | 本报告假设无此约束 |
| NVLink 200GB/s busbw 是否为实测值 | 影响所有绝对时间估计的准确性 | 建议用 `test_intranode.py` 实测替换 |
| dispatch/combine 是否与 Grouped GEMM 做 chunk 级 overlap | 决定整层时延取"上界"（三段相加）还是"下界"（取 max + bubble） | 待确认推理引擎实现 |
| TP-MoE 侧完整公式 | 需对齐单位口径（是否同样用 200GB/s busbw）后才能做并排比较 | 待补充 |
| dispatch/combine 的 balanceness 是否分别测量 | 若两者数值不同，第 8.3 节需要分开代入而非共用一个数字 | **待确认实测数据** |
| balanceness 的统计口径（全局静态 / 逐 chunk 动态） | 决定第 8.3 节除法近似是否精确成立 | **待确认 profiling 方法** |
| balanceness 是否需要逐 layer 分别取值 | 不同 layer 路由模式不同，用单一常数可能不准确 | 建议逐层实测 |
