# SPC-KNOWLEDGE-01B 审计

基线为 `1b8dac8`。本轮保留 `spc.knowledge.trust.TrustedKnowledgeValidator(repositories, evidence_store).validate()` 和 `spc.knowledge.collection` 的调用入口。Scope 修复影响会发出越界 HTTP 请求的 Collection 操作。

## Trust 的记录验证

原 `trust.py` 有 1470 行，`TrustedKnowledgeValidator` 有 35 个方法。迁移后 `trust.py` 有 850 行。下表覆盖这 35 个方法。领域函数收到当前记录及其所需的依赖。递归回调 `visit` 始终回到同一个验证器，并沿用当前 Curation 和正在访问的路径。

| 领域 | 基线方法 | 现有实现及检查内容 |
|---|---|---|
| 知识记录与读取入口 | `__init__`、`record_index`、`_require_record`、`_records_of_type` | `trust.py` 建立记录索引，保留重复和缺失记录检查，按稳定顺序返回已验证记录 |
| 当前 Curation | `resolve_current_curations`、`_record_hash`、`_assert_acyclic`、`_requires_curation` | `trust.py` 验证目标、哈希、替代链与唯一当前决定，区分需要人工接受的对象 |
| 遍历与关闭 | `validate`、`_validate_record`、`_k1f_authority`、`_revalidate_cached_evidence` | `trust.py` 选择根节点，按原顺序分派，检查科学权威，处理递归、已到达节点和关闭复验 |
| Evidence Reference 与来源 | `_verify_evidence_refs`、`_verify_evidence`、`_verify_source`、`_remember_evidence` | `trust.py` 检查 EvidenceSpan、源文件完整性、补充材料的接受状态、历史证据的授权，并登记可达证据 |
| Expert Source / Case | `_validate_expert_source`、`_validate_expert_case` | `trust_rules/expert.py` 调用原有专家来源验证，核对来源与归属哈希、案例与意见证据 |
| Expert Attribution / Opinion | `_validate_attribution`、`_validate_opinion` | `trust_rules/expert.py` 核对归属状态、来源版本、专家身份、证据片段覆盖与依赖 |
| Literature / Supplementary Source | `_validate_literature`、`_validate_supplementary_source` | `trust_rules/literature.py` 核对当前选择和接受状态、文献及补充材料到规范文本和来源的绑定 |
| Representation / Selection | `_validate_representation`、`_validate_representation_selection` | `trust_rules/literature.py` 调用既有表示解析与验证，检查选择、摄入记录和哈希 |
| Canonical Text / Ingestion | `_validate_canonical_text`、`_validate_html_canonical`、`_validate_ingestion`、`_validate_html_ingestion` | `trust_rules/literature.py` 保留 PDF 和 HTML 各自的原文、解析器、规范文本、来源及哈希检查 |
| Historical Authorization | `_require_historical_authorization`、`_validate_historical_authorization` | `trust.py` 要求恰好一条匹配授权，并把直接访问与证据访问交给同一个 `literature.validate_historical_authorization` 核对历史表示、哈希和来源 |
| Relation / Source Quote / Source Claim / Reported Result | `_validate_relation`、`_validate_source_quote`、`_validate_source_claim`、`_validate_reported_result`、`_raise_issues` | `trust_rules/interpretation.py` 核对关系两端和 Curation、引文证据、主张及结果上下文，沿用解释层规则并保留首条错误 |

`trust_rules/__init__.py` 提供共用的回调类型、Curation 类型集合、表示组件类型集合和原异常类。`trust.py` 继续导出原名称。记录验证函数和错误辅助函数共 18 个，原实现已迁走。核对依赖传参和导入位置之后，18 个函数的业务语句语法树与基线一致。`_ValidatedTrustedRead` 的五个方法和 `validate()` 的语法树也与基线一致。

## Collection 的发现和导入

原 `collection.py` 有 1377 行。原模块路径现在指向包入口 `collection/__init__.py`，继续导出原有类、函数及兼容名称。

| 模块 | 负责的操作 |
|---|---|
| `collection/urls.py` | URL、Origin 与路径规范化，按段匹配路径前缀，构造定义和 Scope Policy，判断实际请求是否获准 |
| `collection/discovery.py` | 解析页面、识别显式分页、抽取 DOI / PDF / 文章候选、判定成员依据并保留发现上下文 |
| `collection/connectors.py` | 连接器契约与通用 HTML 连接器，执行受限遍历，按资源标识去重，记录页面字节和每次发现 |
| `collection/snapshots.py` | 共用原有记录 ID 和哈希绑定函数，按原顺序写入资源与快照，计算完整性与两次快照之间的 Diff |
| `collection/importer.py` | 执行获取，隔离预期失败，写入导入事件、获取链接、逐资源结果、成员记录和可选 Diff |

**请求边界。** 基线会检查分页队列，但是重定向以及 DOI、元数据和关联资源的部分后续 HTTP 请求缺少 Collection Scope 检查。现在在调用连接器和获取服务时绑定各自的 URL 规则。`SafeHTTPFetcher` 在每一跳解析 DNS 之前检查当前 URL，随后执行原有的地址检查、IP Pinning、连接身份、TLS、媒体类型、大小和重定向次数限制。独立调用通用 HTML 连接器时同样启用页面规则。获准访问之后的网络失败仍由原有导入流程记录。

页面必须同时匹配允许的 Origin 和路径前缀。文献资源也必须匹配路径前缀，允许其他 Origin 需要显式启用 `allow_external_literature_links`。例如，需要访问 Crossref 时，实际元数据地址也须落在授权范围内。越界候选可以保留发现记录，请求会以 `COLLECTION_URL_OUTSIDE_SCOPE` 拒绝。该代码用于新增的范围拒绝。普通 Literature Acquisition 沿用原有网络行为。请求限制通过执行上下文传递，嵌套限制取交集，退出时恢复。

## K1F 可信读取的范围和所有者

| 数据 | 所有者及复用期限 | 必须重做的检查与回退 |
|---|---|---|
| 已到达记录、当前 Curation、已验证的 EvidenceSpan 与 SourceDocument | `TrustedKnowledgeValidator` 在每次 `validate()` 重建读取状态。构造时建立的记录索引与基线相同 | 关闭前按证据 ID 重新读取并比较 span，重新验证来源完整性并比较来源记录 |
| 已验证的来源字节、用于片段偏移的文本 | 文件证据存储可在同次完整性检查复用已验证字节。在一次 `list_evidence` 中，标准读取路径最多保留 8 MiB 来源内容用于文本复用 | 新读取重新打开文件。自定义完整性钩子、模型或记录读取器走各自原有路径 |
| 规范文本与解析后的规范记录 | `structure/validation.py` 在一次结构验证中保留同一份已校验规范记录与文本 | 仅在收集关闭指纹且使用受支持的标准读取器时启用。覆写 `get` 或 `read_text` 的读取器继续走这两个方法 |
| 编译权威结果、结构选择、初次验证时的结构文件指纹 | `K1FCompilationAuthorityIndex` 属于一次可信读取，结构缓存命中仍比较选择记录与结构哈希 | 关闭时重新检查绑定的表示、原始工件、规范文本、摄入与来源，并比较选择、结构、块、表、单元格、图的文件指纹。没有关闭指纹器时执行完整的结构选择验证 |
| 表格、单元格与其归属 | 每次编译验证临时持有，只在有关闭指纹的路径复用 | 自定义表格读取器或无关闭指纹时重新读取，命中仍检查归属、选中结构和文本区域 |
| 检索、图、快照共用的只读结果 | `_ValidatedTrustedRead` 从一次成功验证取得只读记录和结构指纹，消费者核对 Repository 与 EvidenceStore 对象身份 | 快照使用读结果时再次核对结构文件。读取的指纹与关闭时重算的指纹分别作为比较两端 |

`representation.py` 负责表示绑定和来源验证，`structure_selection.py` 负责把已有指纹收集传入结构检查。两者均未引入长期缓存。审计没有找到两套缓存实现可以安全合并。来源字节、规范文本、编译状态、原始指纹和关闭时的文件内容具有不同的寿命及验证用途。保留这些实现以及交接时使用的防御性副本。K1F、Structure、Representation 和 Repository 的实现文件与基线一致。

## 历史回归

- 用 `1b8dac8` 建立包含成功和失败获取、重复资源与两次快照的本地样本。迁移后通过 Repository 重开，比较了 57 个文件的 SHA-256、28 条 Collection 记录的序列化投影及重新计算的 Diff。逐项相同。
- 原测试使用相同页面字节重新生成两次快照。现在固定检查基线的两个快照哈希，以及旧快照到新快照的 Diff 哈希 `4824711d51524c90162cde3cd5cf66a7e97db13b313dccebee540125e1148d2e`。
- 用基线准备的专家和文献知识库重新运行统一入口，返回的 29 条可信记录与 10 条当前 Curation 逐字段相同。
- 9 个离线 HTTP 范围用例在基线上有 7 个失败、2 个通过。修复后 9 个全部通过。另有 42 个原 Collection 用例通过。测试使用记录实际调用的模拟传输，覆盖获准路径、跨域和路径越界的重定向、DOI 解析、元数据与关联资源。

## 回归结果

| 检查 | `1b8dac8` | KNOWLEDGE-01B |
|---|---|---|
| `python -m ruff check src tests` | 通过 | 通过 |
| 完整 pytest | 1171 通过，1 跳过，481.92 秒 | 1181 通过，1 跳过，439.75 秒 |
| 相同的 9 个新增 Scope 测试 | 2 通过，7 失败 | 9 通过 |
| 原导入名称 | Trust 57 个、Collection 81 个非双下划线名称 | 全部仍可访问，已核对主要兼容导出对象身份 |

两次完整测试均因未配置 `SPC_DOCLING_ARTIFACTS_PATH` 跳过同一个 Docling 集成用例。HTTP 边界用受控传输验证，历史兼容结论来自上述实际样本。
