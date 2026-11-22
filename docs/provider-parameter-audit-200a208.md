# SPC-CLEANUP-03G · Provider 参数校验审计

基线 `200a208`。检查范围为 `src/spc` 中的 Provider、Transport、Adapter、Invocation、CLI 与实际调用点。以下清单按可观察的调用顺序记录。A 表示判断、顺序和错误均相同；B 表示基础判断相同，须由调用方保留参数或错误文本；C 表示独立契约。除特别注明，显式参数错误均为 `ValueError`，没有额外的机器错误码。

## Python 调用入口

| 分类与入口（相对于 src/spc） | 输入、类型、默认值 | 原有判断顺序与错误 | 调用方和处置 |
| --- | --- | --- | --- |
| A/B · planning/structured_provider.py、approval/structured_provider.py 的两个结构化 Provider 构造函数 | 必传 `transport: LLMTransport`；仅关键字 `temperature: float = 0.0`、`max_attempts: int = 2` | 先 `math.isfinite(temperature)` 且非负，否则 `temperature must be a finite non-negative number`；再检查 `1 <= max_attempts <= 5`，否则 `max_attempts must be between one and five` | CLI plan/review、Workflow ProviderSettings、直接 Python 调用。共同校验一次，仍由各自构造函数存放原值。 |
| A/B · knowledge/{literature_knowledge,expert_knowledge}/provider.py 的两个结构化 Provider 构造函数 | 同上，但两者的 `max_attempts` 默认 **1** | 前两项的表达式和次序同上；温度错误是 `temperature must be finite and non-negative`，尝试次数错误相同 | 文献/专家提取 CLI、各自 Compiler 的调用方。共用同一纯函数，显式传入知识 Provider 的原文案。 |
| A · 文献 Provider 构造、partition_literature_knowledge_chunks | `max_chunks_per_batch: int = 40`、`max_batch_text_characters: int = 30_000` | 按块数、字符数依次检查小于 1，各报 `max_chunks_per_batch must be positive`、`max_batch_text_characters must be positive` | 构造函数在温度和次数之后调用；公开分批函数也可直接调用。两处共用文献私有规则，继续在两个入口执行。 |
| B/C · 文献 Provider 后续配置 | `max_batches: int 或 None = None`、`batch_failure_policy: str = "stop"`、`provider_output_dir: Path 或 None = None` | 两项批次上限之后，若已给批数且小于 1，报 `max_batches must be positive when provided`；再检查 stop/continue，报 `batch_failure_policy must be 'stop' or 'continue'`。输出路径在持久化工件时按原有规则检查 | 可选批数的判断可与专家共用。错误顺序、分批策略、诊断与不可变输出属于文献。 |
| B/C · 专家 Provider 与 partition_expert_knowledge_chunks | `max_chunks_per_batch: int = 20`、`max_batch_text_characters: int = 20_000`、`max_batches: int 或 None = None` | 构造时温度、次数后只检查已给批数是否小于 1，报 `max_batches must be positive`。块数与字符上限到分批时才检查，任一小于 1 报 `expert provider batch limits must be positive` | 专家提取、公开分批调用。保留延迟校验、短路次序、合并的错误文案。 |
| C · 知识 Provider 的 propose、分批及 Wire 输入 | 必传各自的 compilation_input 与有序 chunks；两个分批入口的 limits 如上 | 分批时还检查单块长度，分别报 `one literature knowledge chunk exceeds the batch character limit` / `one expert chunk exceeds the batch character limit`；空批分别报 `literature knowledge provider requires at least one chunk` / `expert provider requires at least one chunk`。文献另有每批 200000 字符硬上限和分批诊断 | 两个 Compiler、Materializer 和直接调用。独立保存各自 schema、绑定、输出约束、异常捕获与实际重试。 |
| C · planning/llm_transport.py 的 HTTPJSONLLMTransport、FakeLLMTransport、LLMTransport | HTTP 必传 endpoint/model_id 字符串，api_key 默认 None，timeout_seconds: float 默认 60.0；Fake 必传 responses，model_id 默认 fake-model；Protocol 只声明 generate_structured 签名 | HTTP 先 strip 判空，报 `LLM endpoint and model_id must not be blank`，再检查 http/https，报 `LLM endpoint must use http or https`。timeout 原值传给 urlopen，构造时没有范围限制；响应必须有字符串 output_text。Fake 消耗预设响应，耗尽报 RuntimeError | 所有结构化 Provider；HTTP、Fake、第三方 Transport 的契约独立。 |
| C · knowledge/literature_knowledge/codex_transport.py 的 CodexCLILLMTransport | executable: str/PathLike/序列 = codex；必传 model: str；timeout_seconds: float = 300.0；max_output_bytes: int = 1000000；environment: Mapping 或 None = None | 命令及分项非空 → `Codex CLI executable command must not be blank`；显式非空模型 → `an explicit Codex CLI model is required`；有限超时且 1..1800 含端点 → `Codex CLI timeout must be between 1 and 1800 seconds`；输出 1..1000000 → `Codex CLI output limit must be between 1 and 1000000 bytes`。generate_structured 在 runtime 探测前还要求温度等于 0 | 知识提取 CLI、check-codex-provider。独立保留认证探测、输入字节上限、超时异常、配置与输出字节哈希。 |
| C · providers.py、planning/provider.py、approval/provider.py、interpretation/provider.py 和两种 Knowledge Provider Protocol；Mock 实现 | 方法签名及 provider_id/version/config 各依领域。旧 MockProvider 必传 plans 序列。解释器只需 interpret(context)；Mock 规划、审批及知识 Provider 无通用温度/重试构造参数 | Protocol 不在构造时进行运行时类型校验；Mock 只执行本领域输入与输出规则。缺方法、缺属性仍在各实际使用点按原行为处理 | 旧 Compiler、Planning/Approval、解释/知识编译和用户自定义 Provider。通用参数检查只挂在四个显式需要它的构造函数上。 |

Python 本身继续处理缺失必传参数、未知关键字，以及 isfinite、比较、strip 或后续 range 遇到的不兼容类型。基线没有对数字做强制转换或严格整数断言。例如 True 作为尝试次数可以通过，范围内的浮点次数在构造时也可以通过；后续需要整数的操作仍按原契约处理。默认值、显式整数 0 和浮点 0.0 均按原输入参与配置绑定。

## CLI 与 Workflow 装配

| 分类、入口与调用方 | 选项、默认值与执行顺序 | 可见错误 / 处理 |
| --- | --- | --- |
| A · extract-literature-knowledge、plan、review 的 llm 分支 | endpoint、model 为可选 str，默认 None；仅选中 llm 才检查是否有任一为 None，再取 API key、构造 HTTP transport 和领域 Provider。温度默认 0.0，次数分别 1、2、2 | 三处同一 typer.BadParameter：`--provider llm requires --llm-endpoint and --llm-model`。空字符串留给 HTTP 处理。共用 CLI 专属入口。 |
| A · extract-expert-knowledge、extract-literature-knowledge 的 codex 分支 | codex_model: str 或 None = None，executable 默认 codex，timeout 默认 300；先判 None，再按原次序检查 schema（专家）、构造及探测 runtime、最后构造领域 Provider | 两处同一 typer.BadParameter：`--provider codex requires an explicit --codex-model`。专家默认只选 mock/codex；文献另支持 llm，带独立输出字节数和 batch_failure_policy。 |
| C · plan/review 的选择器、interpret、compile-successor-plan、check-codex-provider | plan/review 默认 mock，可选 llm；interpret 仅 mock；successor 只允许两项均为 mock；check-codex-provider 直接要求 --codex-model，用双 wire schema 及 runtime 做 preflight | 未知值仍按各命令报 BadParameter；分别见 `--provider must be 'mock' or 'llm'`、`Phase 2B only supports --provider mock`、`current successor CLI supports the existing offline mock providers only`。preflight 自己捕获并退出 1。 |
| C · compile-scientific-request、resume-scientific-run → _workflow_options → ProviderSettings / Orchestrator | interpretation/planning/approval 默认 mock/mock/none；两条 CLI 的温度默认 0.0，次数默认 1，Typer 在进入回调前限制 1..5。Workflow 的 Python options 有自己的默认值；ProviderSettings 本身接收显式参数。mock 提前返回，llm 检查 endpoint/model 是否为 None 后才构造 transport | Workflow 里使用 ValueError，按阶段保留 `planning provider llm requires --llm-endpoint and --llm-model` / `approval provider llm requires --llm-endpoint and --llm-model` 及各自 provider 枚举报错；none 在编排层处理。CLI 的类型转换、策略、预算和错误出口各自保留。 |

## Invocation、特殊能力与信任边界

| 入口及调用方 | 被验证的参数 / 记录、顺序与错误归属 |
| --- | --- |
| PlanningProvider.propose / revise / propose_evidence_requests / propose_directions / triage_directions / expand_directions → 各 Workflow Stage | Revision 和 Evidence Resolution 使用同一个 PlanningProvider，均无额外的通用构造器。各方法按输入模型、allowlist、响应 schema、修订内容、证据缺口或方向选择进行验证。阶段的能力检查与 Provider Claim 在真正调用前生效；恢复分别核对已有声明、输入及工件。各自的阻断原因、错误与重试循环归原领域。 |
| ApprovalProvider.review → IndependentApprovalService 与 authority 验证 | 必传普通或 revision review input，依其类型选择 schema 和 prompt；身份、评审输入哈希、审查策略、绑定、receipt、verdict 是独立审批条件。结构化输出最终失败仍是 ApprovalStructuredOutputError。 |
| workflow/providers.bind_provider；planning/hierarchical.planning_provider_config；planning/evidence_resolution._provider_config；Result Feedback Successor | 对现有 Provider 的 ID、版本、model_id、可选配置及原始参数建立绑定。ScientificRunProviderBinding 定义阶段配置哈希。SuccessorPlanningInvocation 按 submission/receipt 对齐及唯一性、legacy 审批字段成组、ID、内容哈希的顺序验证；选择、规划 Claim、审批 Claim 和完成记录分别验证。此处负责身份或恢复，没有重复校验温度或尝试次数。 |
| LiteratureKnowledgeProviderInvocation / LiteratureKnowledgeBatchInvocation / ExpertKnowledgeBatchInvocation → Transport、知识 Provider 及物化 | 输入输出与配置为 SHA-256，绑定 runtime、model 及批内有序 chunk IDs/hashes。先校验字段，再校验对齐、唯一、嵌套记录和相应的内容哈希；文献 Invocation 还检查 content-bound ID，专家与文献批次各有自己的错误文本。多批文献汇总另检查 runtime 可合并性。 |
| AgentAdapter / AgentExecutionAdapter、FTAgentAdapter；BackendAdapter 各 Protocol、Registry 与 backend 调用 | 导出/执行适配器使用能力 catalog 和领域输入；backend 注册检查 descriptor、重复 ID 和宣告能力。PaperQA 配置了 runner 才要求 runner ID/version；GROBID 还要求 service version（均为可选 str，默认 None）。Docling 单独检查本地 artifacts；BackendRun 校验 runtime/descriptor/output 绑定并记录失败。这些适配器没有结构化 LLM 参数契约。 |

直接 Python 调用还有以下独立契约。

- ScientificProblemCompiler.compile 对已绑定 domain 的模型和需要显式 domain 的旧请求分别报 TypeError，之后按阶段检查可调用的 propose 或 revise（`grounded compilation requires a PlanningProvider`、`bounded plan revision requires a revision-capable PlanningProvider`）。普通规划和方向生成需要 ScientificPlanningInput；triage 还需要 ResearchDirectionSet；expand 再需要 DirectionTriageRecord。revise 需要 PlanRevisionInput。propose_evidence_requests 的 triggering_review、directions、triage 和 triggering_review_input 均可省略，默认 None。
- IndependentApprovalService 的 approver_id 是必传 str，空白报 `approver_id must not be blank`，policy 默认 None。评审后依次检查候选未被修改、响应与审批输入相符，然后应用独立审批策略。
- 对空 chunks，MockLiteratureKnowledgeProvider 产生空提案；MockExpertKnowledgeProvider 报 `mock expert provider requires a chunk`。给这两个 Mock 提前增加结构化 Provider 的非空检查会改变行为。
- PaperQA 的配置错误为 `configured PaperQA runner requires explicit ID and version`，GROBID 为 `configured GROBID runner requires runner ID/version and service version`。可选 runner 默认为 None，两者均只在显式提供 runner 时才检查身份。

四个结构化 Provider 继续拥有各自的错误边界、输入输出和重试。CLI 选择器、transport、阶段恢复和历史模型不向第三方 Provider 强加新校验。A02 等值写入不涉及本轮参数校验。

## 实施后的调用关系

| 原来独立维护的位置 | 现在的权威实现 | 仍在调用点决定的行为 |
| --- | --- | --- |
| 规划、审批、文献、专家四个构造函数的前两项 | `planning.llm_transport.validate_generation_parameters` | 默认次数、温度错误原文、原值保存和各自的重试循环 |
| 两个知识 Provider 检查可选总批数 | `planning.llm_transport.validate_max_batches` | 错误原文、检查在各自构造函数中的位置 |
| 文献构造函数与公开分批入口各维护两项上限 | `knowledge.literature_knowledge.provider._validate_batch_limits` | 两个对外入口都能独立拒绝非法参数 |
| 文献/plan/review 的 llm 选项；文献/专家的 codex 选项 | CLI 本地的 `_require_llm_options` 和 `_require_codex_model` | 同一个 CLI 分支、Typer 错误、输入文件处理顺序及 runtime 探测顺序 |

Workflow 的两处 endpoint/model 判断仍附属于各自阶段的 ValueError；独立 HTTP transport 会检查空白与 URL scheme。Evidence Resolution 在缺少能力时仍报告 `EVIDENCE_REQUEST_PROVIDER_UNAVAILABLE`。文献分批还会拒绝类型错误的 transport.last_invocation（TypeError，`transport invocation provenance has an invalid type`），汇总时拒绝不同 runtime 的批次。这些调用均位于参数检查之后。

## 兼容性验证

- 从 `200a208` 的已有文献集成测试输出提取了固定记录 `tests/fixtures/provider-invocation-200a208.json`。该测试使用隔离的本地假 Codex 可执行文件，走真实的 Provider、transport、校验和存储路径。新实现既读取旧记录，也重放相同输入并比较完整 Invocation、Provider ID、版本和配置哈希。
- 使用同一模型名 `test-codex-model`，四个 Provider 的默认身份与配置哈希均与基线相同。规划/审批用 Workflow 的 `configuration_hash`，文献/专家用各自存储的 `provider_config_hash`。固定期望值见 `tests/test_provider_parameters.py`。
- 额外将基线源码和工作树源码导入两个独立 Python 进程。对 518 组输入逐项比较返回值、身份、原值类型表现或异常类型和完整消息，139 组成功、379 组抛出异常，前后差异 0。输入包括组合非法温度/次数、批次上限、Codex/HTTP 超时，以及 Workflow 不同 Provider、endpoint 和 model 组合。
- 固定 Invocation 的 ID 为 `literature-knowledge-provider-invocation-3593bfe365d0b1daa5bc0dd7`，内容哈希为 `7f8fabd07508fc5791227e292f47420433c6f3486a3619b5856c5e9bb5d8ab37`，配置、输入及输出哈希也逐字段相等。
- 38 项定向测试通过，包含直接 Python/CLI 调用、结构化输出重试、自定义规划/审批 Provider 的 Claim 去重，以及修订、证据请求和审批的中断恢复。
- 另以同一套有效科研上下文，经两个版本的实际 CLI 回调核对 plan、review、extract-literature-knowledge、extract-expert-knowledge 的 11 组缺参输入。退出码均为 2，去除终端样式后的诊断完全相同。
- 对比源码，四个结构化 Provider 的 11 个非构造方法（含实际调用、重试和配置读取）和 A02 的四个 `_put_reuse` 函数体逐字相同。

| 验收命令 | 基线 200a208 | 修改后 |
| --- | --- | --- |
| `python -m ruff check src tests` | 通过 | 通过 |
| `python -m pytest`（完整套件） | 1251 通过、1 跳过 | 1264 通过、1 跳过 |

两次均仅跳过同一项本地 Docling 集成测试。环境未配置 `SPC_DOCLING_ARTIFACTS_PATH`。
