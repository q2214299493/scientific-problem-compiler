# A02 审计：四个 `_put_reuse` 的等值写入流程

基线 `d08ff7f`。全仓清单 `full_cleanup_inventory.csv` 的 A02 对应 F031–F034。按函数名和调用表达式在当前基线重新定位，共 4 个定义、17 个静态调用点。下表行号取自该基线。

**判定为 B。** 四处的读取、比较、缺失时写入以及异常传播顺序完全相同。可以共享的单位是对既有 Repository 执行一次等值写入。领域入口仍负责选择 Repository、生成 Key、验证业务对象，以及提供冲突消息。四处触发冲突的谓词相同，领域差异在错误文案和写入前后的工作。

## 四个入口与冲突错误

下列消息均由对应函数在 `get(key)` 成功且 `existing != record` 为真时抛出。异常类型均精确为 `FileExistsError`，`{key}` 为原始传入的 Key，无额外错误码。此时对 `put` 的调用次数为 0。

| 清单项与基线定义 | 直接调用者 | 调用点数 | 完整消息模板 |
| --- | --- | ---: | --- |
| F031 · [expert_knowledge/source.py](../src/spc/knowledge/expert_knowledge/source.py#L23) | `ExpertSourceIngestionService.ingest` | 2 | `conflicting immutable expert record: {key}` |
| F032 · [expert_knowledge/context.py](../src/spc/knowledge/expert_knowledge/context.py#L22) | `resolve_expert_knowledge_input`、`build_expert_knowledge_chunks` | 2 | `conflicting immutable expert compilation record: {key}` |
| F033 · [expert_knowledge/materializer.py](../src/spc/knowledge/expert_knowledge/materializer.py#L38) | `ExpertKnowledgeMaterializer.materialize` | 5 | `conflicting materialized expert record: {key}` |
| F034 · [literature_knowledge/materializer.py](../src/spc/knowledge/literature_knowledge/materializer.py#L97) | `LiteratureKnowledgeMaterializer.materialize` | 8 | `conflicting materialized record: {key}` |

三个专家函数都标注 `key: str` 和返回 `None`，对 Repository 和 record 采用鸭子类型。文献函数额外标注 `ModelRepository`、`BaseModel`。这些注解均未引入运行时类型检查。四者不生成 Key 或 ID，也不创建 Repository。

## Repository、Key 与业务验证逐项对照

下表按各入口实际调用顺序排列。表中的内容绑定 ID 通常采用领域前缀加 `content_hash(identity)[:24]`。完整记录是否另带 `content_hash` 或 `provenance_hash`，由其模型决定。进入四个函数时，Key 和记录均已由调用方给定。

| 基线调用点 | 默认 Repository 类型 | Key / ID 来源 | 写入前后的领域约束 |
| --- | --- | --- | --- |
| F031:267 | ExpertAttributionRepository | `attribution_id`，绑定专家、来源与版本、证据引用、媒介和归属依据/状态 | 已读取专家档案、解析来源、保存原始与规范文本，并登记归属证据。原始字节归档另有 `_immutable_copy`。 |
| F031:291 | ExpertSourceRepository | `expert_source_id`，绑定来源类型、原文/规范文本哈希、归属记录、路径及页范围等 | 归属记录先写入，来源记录随后引用其 ID 和哈希。 |
| F032:76 | ExpertKnowledgeCompilationInputRepository | `record.compilation_input_id`，绑定专家档案、来源、归属、领域及分块策略 | 校验来源，要求归属为 confirmed，且来源和归属的当前 Curation 均为 ACCEPTED。 |
| F032:125 | ExpertKnowledgeChunkRepository | `chunk.chunk_id`，由来源、归属、字符范围、文本、页码与章节等构造，排除 None | 先复验来源完整性，再按页与字符边界生成记录。整次分块完成后仍检查是否为空。 |
| F033:96 | ExpertKnowledgeProposalSetRepository | 直接使用提案集的 `proposal_set_id`；上游将编译输入、Provider 身份与配置、批次调用和响应绑定到 ID | 首先核对提案集的输入 ID 和哈希；来源完整性复验发生在这次写入之后。 |
| F033:170 | ExpertKnowledgeGroundingRepository | `grounding.grounding_id`，绑定来源、归属、Chunk、证据与页/章节 | 精确引文必须在目标 Chunk 唯一命中，先登记 EvidenceSpan。 |
| F033:207 | ExpertOpinionRepository | `opinion_id`，由专家、意见、范围、引用和归属等字段生成 | 检查专家身份及关联引文；写入后处理机器提取 Curation。 |
| F033:255 | ExpertCaseRepository | `case.case_id`，取 `expert-case-` 加 `provenance_hash[:24]` | 检查引用意见并汇总证据；写入后处理机器提取 Curation。此类经 CatalogRepository 继承按 ID 读写。 |
| F033:293 | ExpertKnowledgeCompilationRepository | `compilation.compilation_id`，绑定输入、Provider、提案、Chunk、产物、拒绝项及编译器版本 | 排在本次专家物化的末尾，生成汇总记录。 |
| F034:468 | IdentityBoundRepository[SourceQuote] | `quote.quote_id` 由 `source_quote_id` 绑定 Evidence ID、相对起止偏移和文本哈希 | 先校验输入、Chunk、唯一精确引文、区块边界、Locator 和来源引用。 |
| F034:476 | LiteratureKnowledgeGroundingRepository | `grounding.grounding_id`，绑定文献、表示与结构选择、Block、Evidence、Locator 和 Quote | 先写 Quote，再写 Grounding；通过 `for_operation` 取得本次操作的 Repository。 |
| F034:520 | IdentityBoundRepository[SourceClaim] | `claim.claim_id`，绑定文本、类型、来源角色、引用、强度及认识状态 | 检查 Quote 引用、状态、来源角色和 Claim 绑定；写后处理 Curation。 |
| F034:541 | IdentityBoundRepository[MethodFact] | `fact.fact_id`，前缀 `method-fact-`，绑定文本、属性、证据及状态 | 逐个解析本次物化的 Claim 引用，确认其中有明确的方法陈述；写后处理 Curation。 |
| F034:562 | IdentityBoundRepository[ModelFact] | `fact.fact_id`，前缀 `model-fact-`，绑定文本、属性、证据及状态 | 检查 Claim 引用及明确的模型陈述；写后处理 Curation。 |
| F034:638 | IdentityBoundRepository[ReportedResult] | `result.result_id`，绑定数量、值、单位、系统/方法/结果上下文、证据与状态 | 检查显式结果、单位、原文或表格支持及上下文引用；写后处理 Curation。 |
| F034:679 | KnowledgeRelationRepository | `relation.relation_id`，绑定两端类型及 ID、谓词、领域、证据和理由 | 解析两个已物化的端点并汇总证据；写后处理 Curation。 |
| F034:710 | LiteratureKnowledgeCompilationRepository | `record.compilation_id`，绑定输入、Provider、提案、表示与结构选择、产物、拒绝项及编译器 | 最终汇总；同样使用 `for_operation`。本流程开头的文献提案集直接调用自己的 Repository.put，不属于 A02。 |

## 共享代码、执行顺序与不可变写入

对四份函数体做 Python AST 对照，仅替换冲突消息的固定前缀后，所有语句结构一致。下表对 F031–F034 分别成立。

| 顺序及条件 | 实际行为 | 可观察结果 |
| --- | --- | --- |
| 1. 进入函数 | 先调用传入对象的 `get(key)`，只调用一次。默认实现按安全 Key 读取 JSON、验证模型，再核对持久化记录 ID 与 Key | 读取时的路径、解析、模型或绑定错误直接传播。 |
| 2a. `get` 抛出 `FileNotFoundError` | 调用同一对象的 `put(key, record)` 一次；不执行等值比较 | 写成功返回 `None`，忽略 Repository 返回的路径。写入抛出的任何异常原样传播，包括另一个 `FileNotFoundError`。 |
| 2b. `get` 返回记录 | 按 `existing != record` 调用对象本身的比较语义。没有 ID-only、哈希、字节或重新序列化比较 | 相等时直接结束并返回 `None`；不等时抛本模块的冲突错误。比较自身抛出的异常直接传播。 |
| 3. 仅缺失分支进入底层写入 | 默认 `IdentityBoundRepository.put` 核对传入记录 ID，再由 `ModelRepository.put` 验证路径并调用 `put_immutable_model` | 若此时已有相等模型则复用；若已有不同模型则抛 `FileExistsError("refusing to overwrite different record: {path}")`（实际消息代入路径）；缺失才写 JSON。 |

四个入口没有额外的写前校验代码。默认 Repository 的读、写校验与上述调用点的科研检查是三层不同的约束。ExpertCaseRepository 保留专属 Key/ID 错误；重复 ID 的 catalog 枚举检查属于 `list()`，A02 没有调用它。默认 JSON 写入是检查路径后调用 `dump_json`；此流程本身没有锁或原子“仅当不存在才创建”的并发保证。[Repository 读写实现](../src/spc/repository_components/base.py#L58) · [专家案例的专属错误](../src/spc/repository_components/expert.py#L16) · [JSON 写入](../src/spc/serialization.py#L60)

## 核验与合并风险

- **实际调用轨迹。** 对四个原函数分别运行 8 个内存探针：缺失、相等、不等、读权限失败、读绑定失败、比较抛错、写入冲突和写入抛出文件缺失。32/32 组的调用顺序、次数、返回值、异常类型与完整消息均符合上表。底层或比较器抛出的异常也核对了原异常对象身份。
- **真实 Repository。** 在临时目录用实际 `source_quotes` Repository 核验 9 种情况，包括普通缺失/相等/冲突、不安全 Key、存储 ID 错配、损坏 JSON、传入 ID 错配，以及模拟首次读取未命中而写入时已有相等/不等记录。原记录的 SHA-256 保持一致。最后两种是受控顺序验证，未进行并发压力测试。
- **直接调用 Repository.put 会改变错误优先级。** 额外使用两条各自具有有效内容绑定 ID 的 SourceQuote，将第二条传给第一条的已存在 Key。实测 A02 抛出 `FileExistsError("conflicting materialized record: quote-3e6577fda6bc3d31e13f9794")`，直接 `put` 则先报 `ValueError("repository key must equal record quote_id")`。两次拒绝后原文件 SHA-256 相同。先用 `exists()` 也会跳过 Repository 的读取契约。共享实现应保留对传入对象的 `get/put` 调用。
- **异常捕获的范围也是契约。** 只对 `get` 捕获 `FileNotFoundError`。自定义 Reader 若从读取内部抛出该异常，现行为同样进入写入分支。把比较或写入纳入这个捕获范围，会改变错误或导致额外写入。领域前置检查、ID 生成、Curation 和文件操作均应留在原调用顺序。
- 本次仅执行上述审计探针，未运行完整 pytest 或 Ruff。生产代码、测试及历史工件均未改动。

## 成本收益与结论

| 指标 | 当前基线 | 仅抽取共用流程的估计 |
| --- | --- | --- |
| 实现行数 | 每处 8 个非空行（含单行签名），共 32 行 | 一个约 8 行的函数、四个各约 2 行的领域转发函数、最多四行导入，约 20 行，净省约 12 行。按同一单行签名口径估算，实际取决于类型标注换行。 |
| 需维护的控制流 | 一处 `except` 加一处 `if`，四份共 8 个判断点 | 共用体 2 个判断点。四个原入口各多一层无分支转发，17 个调用点可保持原形式。 |
| 抽象和错误参数 | 四份函数各自持有固定文案 | 在现有模块增加一个无状态函数。除 repository、key、record 外，只需一个固定冲突前缀，在不等分支内拼接原 Key；无需错误策略对象、回调、类型分派或新模块。 |
| 第五种 Repository | 既有领域可复用当地入口，新领域可能再次复制控制流 | 遵循同一 `get/put` 及缺失契约即可复用。只有新冲突文案才需新前缀；若缺失以哨兵返回或要求事务，则须单独评估。 |

收益是把读取失败、等值复用和缺失写入的传播顺序集中维护。磁盘读取、写入和验证次数均应保持现状；这项整理没有性能收益目标。四种领域冲突仍可通过本地入口逐一追踪。

**建议：2. 只抽取公共底层实现。**
