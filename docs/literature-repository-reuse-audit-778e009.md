# SPC-CLEANUP-03F 文献编译 Repository 复用审计

基线 `778e009`。本轮跟踪从文献编译 CLI 组装依赖到 `LiteratureKnowledgeCompiler.compile` 完成物化的一次调用。清单 A02（F031–F034）的记录等值写入另列在末尾。03B 的 I02 已在 `253ee76` 处理。

## 重复构造与调用链

CLI 为当前操作创建 `KnowledgeRepositories(root)`。其 `LiteratureRepositories(root, evidence_store=...)` 已用同一个知识库根目录构造以下五个 Repository，两个组合入口的对应属性指向同一对象。这五类只持有目录、模型和身份字段。初始化本身不扫描磁盘、不创建目录，也不缓存验证结果。

在基线，编译函数和辅助函数随后从 `repositories.root` 再次创建相同的原生 Repository。使用固定历史 HTML（9 个有效 Chunk、1 条 Grounding），在基线和候选代码两个独立 Python 进程中分别用构造函数 Spy 计数。表中“编译追加”从组合入口返回后开始统计，仅计入列出的五类。

| Repository | 编译中使用此对象的调用链 | 组装，前 / 后 | 编译追加，前 / 后 |
| --- | --- | ---: | ---: |
| LiteratureKnowledgeCompilationInputRepository | compile 初次解析输入并写入；build_chunks、materialize 分别校验已存输入，并重新解析当前输入再写入 | 1 / 1 | 5 / 0 |
| LiteratureKnowledgeChunkRepository | build_chunks 逐块写入 9 次；materialize → validate_chunk 逐块读回 9 次 | 1 / 1 | 18 / 0 |
| LiteratureKnowledgeProposalSetRepository | materialize 校验提案集的输入绑定后写入 | 1 / 1 | 1 / 0 |
| LiteratureKnowledgeGroundingRepository | materialize 处理通过验证的引用后，将定位记录交给原有等值写入流程 | 1 / 1 | 1 / 0 |
| LiteratureKnowledgeCompilationRepository | materialize 将最终编译记录交给原有等值写入流程 | 1 / 1 | 1 / 0 |
| **合计** | **一次依赖组装加一次编译，共 31 → 5 次** | **5 / 5** | **26 / 0** |

范围计算、模型绑定、文件读写、缺失或冲突处理仍由原有实现执行。实测只衡量对象初始化次数。

## 等价条件与边界

编译从已传入的知识库组合取得对象。仅当两个组合入口持有同一原生实例，且具体类、目录及路径类型、模型和身份字段与标准构造一致，并且实例无额外状态或读写方法覆盖时才复用。根目录比较不调用 resolve，不将不同路径或符号链接主动归并。若任一条件不满足，编译沿用原有的按当前根目录打开原生 Repository 的路径。

- 两次分别组装的知识库即使指向相同目录也有独立实例。编译器自身不保存 Repository 或验证结果。
- 跨根目录、替换的模型、身份字段、子类、自定义 reader/writer 及来自另一次组装的替代对象均不满足复用条件。读取和写入仍使用当前编译原本绑定的磁盘记录。
- K1F 可信读取及关闭复验、独立审计/可信视图和 EvidenceStore 保留原路径。基线审计视图在这个样本中另构造 4 次，这些调用不计入编译的 26 次，也不在此次修改范围内。
- 兼容入口继续导出原有五个类，构造函数保留 `knowledge_root` 参数。原有 Repository 的 `get`、`put`、`list`、安全键检查和 Immutable Write 实现均未改动。

## 历史数据回放

在已提交的 `tests/fixtures/knowledge-01a-3f53101.zip` 的 HTML 临时副本上，用既有测试 Curation 和 Mock Literature Provider 执行正式编译。源文件及准入记录逐字节保持一致。两个版本各执行两次编译，读回的记录和写入结果一致。

| 对照项 | 基线和候选代码得到的相同值 |
| --- | --- |
| 有序 9 个 Chunk 的内容哈希 | `754a705e04acd2fd036b4c2a1eef8ad31ca18a5fe84c34914d54655ac3fba3db` |
| 最终编译记录的 content_hash | `a776329d547251e34fa2ba8fb00eae8e92dd3cbeb7653ff5dbf719b3750a4e11` |
| 五类 Repository 写出的 13 个 JSON，相对路径与原始文件 SHA-256 的清单哈希（Windows） | `f96e27d067f3a72a461c5b5244cdd809981309c77bea98e123435f5211c08ed8` |
| 本次编译新增的全部 18 个文件，按同一方式计算的清单哈希（Windows） | `271012366aaabc20da2c089cc37d1092817d66ba4fdba092cb1acbac664db1f3` |

## 留待独立任务的 A02

四份 `_put_reuse` 接收调用方已经持有的 Repository，自身构造次数均为 0。共同顺序是先 `get`，只有 `FileNotFoundError` 才调用 `put`，相等则返回，读到不等记录则抛出各自的 `FileExistsError`。其他读取错误沿原路径传播。四个函数的源码均已与 `778e009` 逐字比较，调用点的顺序保留。

| 位置（相对 src/spc/knowledge） | 原冲突消息 |
| --- | --- |
| expert_knowledge/source.py | `conflicting immutable expert record: {key}` |
| expert_knowledge/context.py | `conflicting immutable expert compilation record: {key}` |
| expert_knowledge/materializer.py | `conflicting materialized expert record: {key}` |
| literature_knowledge/materializer.py | `conflicting materialized record: {key}` |

## 验收结果

- `python -m ruff check src tests` 通过。
- 文献编译复用、Repository 兼容、EvidenceStore 及 K1F 可信读取的定向测试共 84 项通过。
- `python -m pytest` 全量执行，1251 项通过，1 项跳过。跳过的是 Docling 本地模型集成测试，环境未配置 `SPC_DOCLING_ARTIFACTS_PATH`。
