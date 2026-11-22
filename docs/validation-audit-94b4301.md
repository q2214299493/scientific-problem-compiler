# SPC-CORE-03 验证审计

基线为 `94b4301`。基线的 `src/spc/validators.py` 同时承担计划、证据、审批和导出验证，当时没有 `src/spc/validation/` 包。

## 函数清单

[逐函数清单](validation-inventory-94b4301.csv) 收录 251 个函数和方法。范围覆盖旧入口、Approval、Planning、Result Feedback、Knowledge Trust 和 Downstream Import，并补入解释层验证器、导出写入端及两个完全相同的 Knowledge 合同校验函数。清单包含协议声明、内部辅助函数和调用验证器的服务方法，251 是已审计条目数。

- `baseline_module`、`line`、`function` 和 `input_signature` 标明原位置和被检对象。`authority_after` 给出实现位置，对提供者或编排方法则给出该方法的位置。
- `boundary` 区分公共基础 A、科研领域 B 和安全边界 C。`direct_rules_or_errors` 摘录原函数内的错误或判断条件。`local_rule_calls` 列出委托的校验调用。
- `direct_callers` 是在生产代码中按导入、接收对象或同类方法解析的调用者。`possible_receiver_modules` 单列静态分析无法确定接收对象的同名调用。协议的具体提供者由运行时决定。空白不表示没有外部 Python 调用者。
- `duplicate_or_boundary_note` 记录确认的重复或需要保留的复验。逐行的错误码和路径以原函数为准。下面的表汇总跨模块的调用关系。

## 权威实现与调用关系

| 层 | 对象及规则 | 实现位置（相对 `src/spc/`） | 主要调用者 |
|---|---|---|---|
| A | 报告有效性由有序 issue 序列决定，保留重复项、路径和错误文本 | `validation/common.py` 的 `validation_report`、`format_issue_codes`、`format_issue_messages` | 计划、解释、审批、导出和导入的验证报告或领域异常 |
| A | 从剔除 ID、content_hash 和空值后的 JSON 派生有前缀的 ID，再校验包含该 ID 的完整摘要 | `validation/common.py` 的 `content_binding_error`，模型合约使用 `require_content_binding`。规范化摘要仍由 `serialization.content_hash` 计算 | 编译回执、初审及修订的 review input、review record、独立审批回执、专家与文献合同中的同形校验 |
| B | EvidenceSpan、实际来源完整性、证据引用、指纹的证据或偏离声明 | `validation/evidence.py` 的 `validate_evidence_links`，来源字节委托 repository 的 `verify_evidence_integrity` | 科研计划验证，以及经它进入的编译、导出预检和父计划复验 |
| B | 计划、DAG、候选集合、意图和能力约束、只规划执行策略 | `validation/plans.py` 的 `validate_question_plan`、`validate_dag`、`validate_candidate_set`、`validate_phase1_boundary` | 编译器、CLI、Workflow、Result Feedback、导出 |
| B | 方法指纹差异是否已披露，结果是否与审批时的系统、方法及具体偏离值一致 | 计划比较由 `validation/plans.py` 的 `compare_method_fingerprints` 负责；结果比较由 `result_feedback/intake.py` 的 `_comparison` 和 `_same_difference` 负责 | 计划比较调用方、Result Intake |
| B | 编译回执来源，验证记录的计划 ID、版本、内容摘要、领域及领域包版本；有当前报告时还要比对 valid 和有序 issue codes | `validation/plans.py` 的 `validate_plan_compilation_receipt`、`validation_record_matches_plan`、`validate_plan_validation_record` | 计划审批、导出、Workflow、父计划验证 |
| B | 证据包的引用、引文与原文、结果单位和来源、比较约束、知识缺口 | `interpretation/validators.py` 的分项规则与 `validate_evidence_packet_integrity` | 解释层构建器、规划输入解析、恢复和 successor |
| B | Proposal 的允许引用，方向选择与扩展，每条修订反馈和声明的实际变更，证据请求的满足条件 | `planning/validators.py`、`planning/hierarchical.py`、`planning/revision.py`、`planning/evidence_resolution.py` | Planning Provider、编译器、Workflow、successor |
| C | 审批响应允许引用的对象、独立提供者身份、原候选计划未被改变、项目策略与审查决定 | `approval/validators.py` 的 `validate_approval_response`、`approval/service.py` 的 `review`、`approval/policy.py` 的 `apply` | Independent Approval Service 与 gate 绑定路径 |
| C | 批准计划的最低权威：精确版本、编译回执、独立审批链、条件与人工决定、信任策略、Gate 六组绑定 | `validation/approval.py` 的 `validate_approved_plan_authority`，同模块的 `validate_independent_approval_chain`、`validate_approval_state` 和 `gate_binds_*` | Workflow 的审批复用、修订和证据循环，Result Feedback 的父计划及 successor 复用 |
| C | 写前计划与 handoff，写后文件路径、必需文件、SHA-256、manifest 与磁盘内容的语义一致性 | `validation/export.py` 的 `validate_handoff_package` 和 `validate_export`，调用共享编译回执、独立审批链与 Gate 绑定规则 | GenericExportService、CLI、Downstream Import |
| C | 接收方的独立审批策略、目标 agent、唯一且可用的任务能力映射、接收包自身的内容绑定 | `downstream/validator.py` 的 `DownstreamImportValidator.import_package` | 下游 CLI、携带执行提案的结果接收 |
| C | 知识的当前人工审核、可信依赖、原始来源、引文、结构与文献表示、历史授权 | `knowledge/trust.py` 的 `TrustedKnowledgeValidator` 及其分项方法 | 知识视图、检索、快照和状态查询 |
| C | 结果的父 run、精确计划、任务和能力，产物字节与清单，科研准入、人工审核和物化 | `result_feedback/parent_authority.py` 的 `ParentAuthority`，`artifacts.py` 的 `ResultArtifacts`，`intake.py` 的 `ResultEvidenceIntakeService` | Intake、人工审核及 successor 输入选择 |
| C | Successor 的同一父计划、已接受结果、物化回执、调用内容、阶段产物和独立审批 | `result_feedback/successor.py` 的 `compile` 和 `_validate_reusable_cycle` 等复用检查 | Successor Planning 与恢复 |

兼容入口 `validators.py` 将原有名称直接指到对应实现。包内依赖顺序为 common → evidence → plans → approval → export（箭头指向使用者）。已有领域服务仍可以从旧路径导入。

## 合并的实现

1. 四个审批链产物的内容绑定由同一个函数计算，每类产物保留原来的 ID 前缀和错误码。专家和文献合同中两份逐行相同的校验器改为引用同一抛错入口，保留先报 ID、再报 Hash 的顺序。
2. 计划验证记录的五字段绑定用于有、无当前报告两条路径，各自的失败文本和额外检查保持原义。
3. 审批权威、导出预检和磁盘导出三处共用 Gate 对计划版本、verdict、验证记录、策略、编译回执和审批回执的绑定比较。verdict、验证记录和策略使用完整对象摘要，两个回执使用原有 content_hash。审批继续汇总为一条 Gate 失败，预检和磁盘验证继续分别报告各自的错误。
4. 原入口与 interpretation 的两份报告构造，以及审批、提案、层级规划和修订验证中的同形返回改用 `validation_report`。旧异常继续区分“有序代码”与“有序代码加消息”两种文本格式。

## 保留的复验和范围

- 结构化 Planning Provider 需要在交付 proposal 前决定是否重试，编译器也接受其他 provider 及直接传入的 proposal，因此在物化前复验。审批的结构化 provider 检查响应 schema，服务层另行检查引用是否被本次 review input 允许。
- 导出预检验证内存对象，staging 校验读取写出的实际文件；Downstream Import 在消费时重新读取持久化包。旧手工审批模式允许的导出仍受下游独立策略限制。导出沿用原有分项验证调用顺序，避免把缺件、gate 失败或文件问题统一改报为审批错误。
- Handoff 和完整独立审批链都检查候选计划与 verdict 的绑定。两个函数均可独立调用；组合调用保留历史诊断的顺序与重复次数。
- 知识缓存命中后仍复核可变的原始证据。Result Intake 与 Successor 在各自使用时重新确认父计划、已归档字节和物化记录。检索结果数、文件存在或上一阶段的成功标志不能替代这些输入。
- 模型及外部后端合同仍有各自的 ID 与摘要校验，部分构造器也计算同形 ID。它们属于反序列化或产物生成入口，具有各自的异常和输入契约，本轮保留。文件字节 SHA-256 与模型规范化 Content Hash 分别沿用原有实现。

## 兼容性核验

对照基线在独立 Python 进程中检查旧模块的 63 个公共名称、22 个函数签名，以及从模型、两个 Knowledge 合同和验证入口公开的 224 个不同 Pydantic 模型的 validation 与 serialization schema。逐函数比对修改范围内 39 个函数的 256 个 `ValidationIssue` 构造表达式（包含 200 个不同的字面量错误码）。对基线测试产出的磁盘导出包、下游接收及审批链变体回放，84 份有序报告完全一致，含 58 份拒绝报告。

| 检查 | 基线 | 最终版本 |
|---|---:|---:|
| Ruff（src、tests） | 通过 | 通过 |
| 完整 pytest | 1152 通过，1 跳过 | 1160 通过，1 跳过 |

两次全量测试均跳过需要本机 Docling 产物的可选集成用例。当前环境未配置 `SPC_DOCLING_ARTIFACTS_PATH`。
