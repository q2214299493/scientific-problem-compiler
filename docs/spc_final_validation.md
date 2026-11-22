# SPC 最终回归与架构冻结验收

**建议冻结代码基线 `ba6c04f7d185f7abad77acfddbc0fcb853cf8744`。** 2026-11-22 在 Windows 11 本机完成全仓回归。21 个指定里程碑均在该基线的第一父提交链上。完整 pytest 收集 1306 项，1305 项通过，失败和错误均为 0，1 项 Docling 真实模型集成因环境配置跳过。此次冻结结论适用于下述代码、固定契约和离线行为。线上负载、真实公网与未归档的生产数据仍属未验证范围。

验收分支为 `codex/spc-final-validation-20261122`。检查使用从最终基线创建的独立工作树。所有写入型业务探针都指向临时目录和历史样本副本。交付内容为本报告和 450 行可逐项筛选的证据清单，清单列出每个数据点的对照版本、检查方法、观察结果、判定及证据位置。[验收清单](spc_final_validation_inventory.csv)

## 1. Git 来源与真实父链

使用 `git log --first-parent --reverse 72bc0ac^..ba6c04f` 列出两端间的 35 个提交，再用 `git show -s --format="%H %P %s"` 检查每个对象的父提交。对相邻里程碑及各里程碑与最终基线分别执行 `git merge-base`。本地 `main` 指向 `72bc0ac`。35 个对象均为单父提交，除区间起点外，实际父提交均为上一行；起点父提交为 `b6e2393`。下表列的是实际直接父提交，14 个阶段内提交也按顺序列入 CSV。

| 里程碑 | 提交 | 实际父提交 |
|---|---|---|
| 原始 main | `72bc0ac` | `b6e2393` |
| Workflow | `6a9fb24` | `72bc0ac` |
| Workflow Fix | `68d0c34` | `6a9fb24` |
| Result Feedback | `f5cfdd7` | `31abf43` |
| Models | `ac25bd4` | `f5cfdd7` |
| Repositories | `94b4301` | `ac25bd4` |
| Validators | `3f53101` | `94b4301` |
| Knowledge 01A | `1b8dac8` | `3c7238a` |
| Knowledge 01B | `c5c5ec0` | `a02603d` |
| Successor Status | `4941d86` | `c5c5ec0` |
| Successor Status Fix | `f4ce42a` | `4941d86` |
| Cleanup 03A | `6acbaae` | `f4ce42a` |
| Cleanup 03B | `253ee76` | `d055db0` |
| Cleanup 03C | `79170b1` | `253ee76` |
| Cleanup 03C Fix | `9435a5f` | `79170b1` |
| Cleanup 03D | `b414df2` | `9435a5f` |
| Cleanup 03E | `778e009` | `b414df2` |
| Cleanup 03F | `200a208` | `778e009` |
| Cleanup 03G | `d08ff7f` | `200a208` |
| A02 Audit | `9d6665c` | `d08ff7f` |
| A02 Implementation | `ba6c04f` | `9d6665c` |

## 2. 完整自动化测试

运行环境为 Windows 11 build 26200、Python 3.13.9、pytest 8.4.2、Ruff 0.12.0、Pydantic 2.12.5、Typer 0.20.0、Click 8.3.1。Python 导入路径指向该工作树的 `src` 和 `tests`。pytest 使用默认测试目录，没有筛选表达式。额外启用了跳过原因和临时 JUnit XML 记录。

| 命令 | 退出码 | 实际结果 |
|---|---:|---|
| `python -m ruff check src tests` | 0 | All checks passed! |
| `python -m pytest` | 0 | 43 个测试模块，1306 项，1305 通过，0 失败，0 错误，1 跳过。终端耗时 454.30 秒（7 分 34 秒）。 |

唯一跳过项为 `test_real_docling_smoke_only_with_preconfigured_local_artifacts`，原因为 `SPC_DOCLING_ARTIFACTS_PATH is not configured`。其余结果与 JUnit 逐模块汇总一致。下文的测试子集均包含在这 1306 项中。独立的 CLI 对照、跨版本恢复与计数探针另外记录。[Docling 条件测试](../tests/test_backends.py)

## 3. CLI 与旧 Python 入口

从 Git 的 `72bc0ac` 导出源码，以两个隔离的 Python 进程分别加载原始 main 和最终版本的 `spc.cli:app`，核对加载路径并固定终端宽度、颜色和程序名。所有命令在临时工作目录中通过已注册的 Typer/Click 回调检查。同一命令同一输入按完整帮助或错误文本以及退出码比较。

| 对照项 | 本轮实际结果 |
|---|---|
| 命令目录、回调、参数 | 两版均为 43 个子命令。名称、回调名、参数名称与选项、必填标志、类型、默认值，以及已检查的选择范围和路径约束一致。 |
| 帮助 | 根命令和 43 个子命令的 `--help` 全文一致，全部退出 0。 |
| 参数解析 | 43 个未知选项用例均退出 2。38 个有必填参数的命令省略必填项均退出 2。另在 14 个有适用数值或枚举选项的命令上输入无效值，全部返回原错误全文及退出码 2。 |
| 文献接收与知识导入 | 完整测试通过本地 PDF 接收、专家来源导入、文献知识提取、查看与显式 Curation 的 CLI 路径。Collection 测试用记录型 Transport 验证限定范围内的发现和导入。 |
| 结果反馈与状态 | 两版分别对同一旧版父 Run 的副本执行接收、显式接受和状态查询。接收返回 `RESULT_REQUIRES_CURATION`，接受后有物化回执，状态为 1 条已接受结果、0 个完整周期，查询前后目录清单及文件 SHA-256 一致。缺失 submission 的业务错误退出 1，替换两份副本各自的绝对根路径后，错误全文相同。 |

10 个旧模块在最终版本均可导入。以原始 main 在这些模块内定义、且名称公开的函数和类为集合，对 389 个名称逐项验证它们与当前定义模块的对象具有 `is` 身份。另检查 Workflow 和 Result Feedback 的显式导出、旧 `ScientificProblemRunRepository`、Trust 包级入口及旧 `_create_locator` 别名。模型与 Repository 的专门清单还覆盖类型别名、辅助函数及归属模块。[模型契约测试](../tests/test_model_contracts.py) · [Repository 入口测试](../tests/test_repository_components.py) · [验证器入口测试](../tests/test_validation_compatibility.py)

| 入口 | 原公开函数/类 | 入口 | 原公开函数/类 |
|---|---:|---|---:|
| `spc.models` | 244 | `spc.knowledge.acquisition` | 16 |
| `spc.repositories` | 54 | `spc.knowledge.ingestion` | 15 |
| `spc.validators` | 21 | `spc.knowledge.structure` | 15 |
| `spc.workflow.service` | 6 | `spc.knowledge.collection` | 10 |
| `spc.result_feedback.service` | 5 | `spc.knowledge.trust` | 3 |

## 4. 模型与历史 Artifact

下表标明每组证据实际比对的东西。文件 SHA-256 针对原始字节，SPC 的 Content Hash 与 ID 则由项目自身的序列化、重建和验证规则计算。CSV 分别记录 240 个定义、7 个序列化样本的预期 Content Hash 和 6 个固定样本文件的原始 SHA-256。

| 历史对照 | 验证内容和结果 |
|---|---|
| `model-contracts-f5cfdd7.json.gz` | **191 个 Pydantic 模型 + 49 个 Enum = 240**。每个定义的默认 JSON Schema、validation Schema、serialization Schema 均与归档相等。模型字段、必填性、默认值、工厂、别名及枚举成员与值一致。旧模块的 268 个公开名称与已存清单一致，定义只出现一次。 |
| `model-history-f5cfdd7.zip` | 6 份 YAML、1 份 JSON 都能读回，各自 Content Hash 与归档匹配。覆盖 ScientificQuestionPlan、IndependentApprovalReceipt、ResultEvidenceSubmission、ResultEvidenceMaterializationReceipt、SuccessorPlanningCycle、PlanRevisionChain、ScientificProblemRun。 |
| `repositories-ac25bd4.zip` | 7 类 Repository 记录按历史预期哈希读取。集合页面工件、资源、专家档案、文献原件和规范文本经验证及等值复用后，12 个数据文件的路径与字节一致。 |
| `knowledge-01a-3f53101.zip` | PDF 17、HTML 39、DOCX 20，共 76 个文件。检查表示绑定、文档结构或补充来源。在 HTML 样本中重建表格和图注的 Evidence Span，核对 span ID、Content Hash 与定位器，检查归档再读取前后每个文件的哈希。 |
| `knowledge-chunks-b414df2.zip` | 专家 PDF/长文本和文献 HTML/PDF/长文本五组重建结果分别为 2、5、9、3、2 个 Chunk。21 个结果与有序历史模型逐字段相等，覆盖内容、偏移、页/章节/结构上下文、ID 和 Hash。 |
| `provider-invocation-200a208.json` | 使用本地 fake CLI 重建并逐字段比对旧调用记录。Invocation ID 为 `literature-knowledge-provider-invocation-3593bfe365d0b1daa5bc0dd7`，Content Hash 为 `7f8fabd07508fc5791227e292f47420433c6f3486a3619b5856c5e9bb5d8ab37`。输入、输出、配置哈希和 Provider 身份均匹配。 |
| Content-bound ID | 历史 Collection 页面、Result Submission 及其他结果记录可用原字段重建。默认值、显式空值、Unicode、字段顺序与各领域前缀由对应回归保护。 |

## 5. Workflow、恢复与审批

除读取固定归档外，本轮还在临时目录用 **原始 main 的代码和离线测试 Provider** 生成一份已批准的 Successor 及另一份实际经过修订的父 Run。换用最终代码打开这些 Run 的副本，规划、修订和审批回调上都设置了调用探针。

| 复验场景 | 观察结果 |
|---|---|
| 已批准的旧 Successor | 状态为 `APPROVED`，完整周期数为 1。以同一输入恢复后，cycle ID 和 Content Hash 与旧版生成值相同。 |
| 历史 Revision Chain | 两轮审批，实际使用一次修订。终态恢复返回 `APPROVED`。链的 Content Hash 保持为 `e2e1483c256eda4134d3f63e727b89504c5368c50048f34982b160f434b0d2ff`。 |
| 恢复副作用 | 上述两份 Run 副本合计 178 个文件。恢复前后路径、目录清单和原始 SHA-256 相等。三个受监测的 Provider 方法均调用 0 次。 |
| 缺少最终周期 | 另复制旧版的完整审批树，仅在复制时排除 `cycle.yaml`。最终版本报告 `SUCCESSOR_APPROVAL_OUTCOME_UNCERTAIN`，完整周期数为 0，状态查询前后清单及哈希相等。 |
| 空文件、孤立 Proposal / Outcome | 使用历史 run.json 的两个空规划文件、无必要 claim 的单独提案或结果均返回 `SUCCESSOR_PLANNING_OUTCOME_UNCERTAIN`，完整周期数为 0。只有 Invocation 时保留 `SUCCESSOR_INVOCATION_RESERVED`。有效产物待选时仍能显示待选或待审批状态。 |
| Lock、预算与 Provider Claim | 并发恢复同一 Run 被锁拒绝，修订只调用一次。全局修订预算在证据重规划和恢复后继续生效。持续证据缺口达到证据周期上限时停止。规划、证据请求或审批调用结果不确定时保留 claim，恢复不会重复调用。 |
| 审批、父计划与反馈 | 已改变的候选计划、伪造 Receipt、错 Run/错 Plan 的结果、未经声明的指纹差异及被撤回接受的结果均被拒绝。下一周期精确绑定已批准的父版本。只读查询、独立审批、导出信任策略分别有断言。 |

逐项用例和参数化执行数见 CSV 的“流程”部分。[跨循环不变量](../tests/test_cross_loop_invariants.py) · [修订与 Run Lock](../tests/test_plan_revision.py) · [证据预算](../tests/test_planning_evidence_resolution.py) · [独立审批](../tests/test_independent_approval.py) · [结果反馈](../tests/test_result_feedback.py)

## 6. Knowledge 安全

| 边界 | 本轮验证方式与结果 |
|---|---|
| SSRF、DNS、IP Pinning 与 TLS | 本地和私有地址被拒绝。重定向逐跳解析并绑定允许的 IP。连接端地址须与预检查一致，HTTPS 使用固定 IP 连接并传原主机名作 SNI。检查到默认 Transport 使用系统默认 SSL context。这些断言使用受控解析、Transport 或 socket/SSL 替身。 |
| Collection Scope | 域、路径前缀、连续重定向、DOI、元数据及关联资源均经 Scope 检查。记录型 Transport 的调用清单表明越界目标在发出请求前被拒绝，显式允许路径仍成功。 |
| Windows Junction 和 Safe Path | 在本机实际创建目录 Junction 和 symlink。外部空目录和含周期记录的外部目录均被拒绝；内部 Junction 可恢复；覆盖了目标消失及写前目录被替换。读写时仍验证 Run 根绑定。 |
| Evidence Provenance、Immutable Artifact | 复验来源字节、证据记录、计划绑定和结果归档。已存在的相同记录可复用，冲突按既有异常拒绝写入，原文件 SHA-256 保持一致。 |
| K1F Closing Verification 和自定义 Reader | 同一可信读取内复用已验证字节、文本和结构。关闭及后续独立读取重新验证，读取期间发生的来源或规范文本变更被拒绝。自定义来源钩子、规范文本读取器和结构归属读取器均有独立回退用例。 |

## 7. 性能与实现复杂度

| 项目 | 比较结果 | 证据与适用范围 |
|---|---|---|
| Knowledge 状态索引 | 每次构建 **2 → 1**；索引内各 Repository 的 `list/list_metadata` 实际调用 **48 → 24** | 本轮对 `f4ce42a` 与最终代码分别计数，空库与使用相同旧 Run 样本的有记录库均成立，两种状态结果在两个版本间分别相等。 |
| 同根 EvidenceStore | **2 → 1** | 本轮直接计数一次 `open_knowledge` 中的 `KnowledgeEvidenceStore` 构造。独立操作、根目录和读写模式隔离另由 pytest 验证。 |
| 普通周期目录 | **2 → 1** | 用同一个已批准的旧 Run 对照 `f4ce42a` 与最终代码。普通周期项原先各调用一次 `is_symlink`、`is_dir`，现在一次 `lstat`。计数限于枚举时对该项的直接检查；写入前和链接检查另行执行。 |
| 文献编译的 Repository | 额外构造 **26 → 0** | 历史审计在 9 个 Chunk、1 条 Grounding 的 HTML 样本上对五类原生 Repository 做构造 Spy。含依赖组装时总计 **31 → 5**。本轮完整测试重新验证额外构造为 0、结果哈希相等以及不同配置回退。 |
| Content-bound ID | 相同构造算法 **4 → 1** | 领域仍决定输入字段、前缀与默认值处理。 |
| 文本范围算法 | **2 → 1** | 专家和文献 Builder 共享字符范围算法，两者的分块策略、样本数量、定位和 ID 分别验证。 |
| Provider 公共参数 | 4 个结构化 Provider 共用温度和尝试次数校验；两种知识 Provider 共用批次上限校验 | 原错误优先级、各自的默认值、Transport 超时及历史配置哈希获得回归覆盖。 |
| A02 等值写入 | 控制流 **4 → 1** | 保留四个领域入口和冲突消息。32 组原行为探针及 9 项补充回归均通过。 |

**本机计时。** 两个版本分别在独立解释器内，对每类知识库预热 1 次后测量 7 次，计时期间关闭 Spy。空库中位数为 **15.05 → 16.05 ms**，有记录库为 **88.10 → 87.49 ms**。空库和有记录库的前后范围均重叠，具体范围见清单。测量与完整回归处于同一主机、同一时段，并非专用基准环境。这次可以确认调用次数减少，尚无证据证明稳定的端到端耗时或吞吐收益。后四项数字表示代码实现的复用程度。[Repository 构造审计](literature-repository-reuse-audit-778e009.md) · [ID 审计](content-bound-id-audit-9435a5f.md) · [分块审计](text-chunking-audit-b414df2.md) · [Provider 审计](provider-parameter-audit-200a208.md) · [A02 审计](a02-put-reuse-audit.md)

## 8. 剩余风险与冻结范围

| 尚未验证的范围 | 已有证据的边界 |
|---|---|
| 真实 Docling 模型推理 | 真实本地模型 smoke test 已跳过。离线配置、模型绑定与替身测试有独立结果。 |
| 第三方 Windows Reparse Point | 实测覆盖本机原生 Junction 和 symlink。其他厂商标签、驱动或文件系统没有实物样本。 |
| 外部生产环境调用量 | 现有证据覆盖功能、调用次数及局部并发。生产并发分布、P95/P99 和长时间吞吐未采样。 |
| 真实公网 HTTP、DNS 与 TLS | 离线探针验证策略与调用顺序。真实站点、证书链、代理、DNS 变化及服务可用性未在本轮联测。 |
| 全部生产历史数据 | 固定归档和原始 main 生成的离线 Run 可读、可验证、可恢复。仓库外私有历史记录和未归档版本未遍历。 |

**冻结决定。** 建议将 `ba6c04f` 作为本轮架构与数据契约的冻结点。本轮在已执行范围内未发现阻止冻结的回归。若要把真实 Docling 模型能力纳入验收，下一项验证是在已配置本地模型目录的环境中运行同一个 smoke test，并单独记录结果。
