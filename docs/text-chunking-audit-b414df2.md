# SPC-CLEANUP-03E 文本分块审计

基线 `b414df2`。全仓审计 H03（F029、F030）指向 Expert 和 Literature 的 `context._bounded_ranges`。基线上的 AST 比对确认两个函数逐语句相同。输入相同的 8,100 个连续非空白字符时，两个实际调用入口的上限分别产生 5 段和 2 段。

| 类别 | 基线位置与调用者 | 输入、算法和输出 |
| --- | --- | --- |
| A，共享范围计算 | `knowledge/expert_knowledge/context.py:85`、`knowledge/literature_knowledge/context.py:140`，各自的 Chunk Builder 调用 | 对原 Python 字符串的指定半开区间从左向右取窗口。未到区间末尾时，查找最后一个换行或 ASCII 空格，只有位置严格大于上限的一半才提前断开。用 `isspace()` 去除两端空白，返回非空切片的绝对字符偏移。无编码转换或 Unicode 规范化；组合字符可以被窗口截开。 |
| B，Expert 策略 | `expert_knowledge/context.py` 的 `_page_segments` 和 `build_expert_knowledge_chunks`，由 ExpertKnowledgeCompiler 调用 | 校验来源完整性后读取 UTF-8 文本。PDF 按现有页范围切分，无页范围时按整篇切分。每窗最多 2,000 字符，按页和生成顺序返回。身份绑定来源、专家、归属、偏移、原文、页码和空章节路径。没有可用块时沿用原有异常。 |
| B，Literature 策略 | `literature_knowledge/context.py` 的 `build_literature_knowledge_chunks`，由 LiteratureKnowledgeCompiler 调用 | 验证当前编译输入，读取选定表示的规范文本。逐个处理准入的结构块，略过 PAGE 和未纳入的 Content Region。每窗最多 8,000 字符，按起始偏移、Chunk ID 排序返回。身份绑定文献、表示、结构、块引用及哈希、偏移、原文、页码、章节、区域及不确定性、表图上下文。 |
| C，专家来源预处理 | `expert_knowledge/source.py`，ExpertSourceIngestionService 调用 | 本地文本解码 UTF-8 并统一换行；PDF 借用 PDF 提取器，按页文长度和固定页间分隔符记录页范围；网页从规范 HTML 文本取得逻辑页。输出原始分块坐标使用的文本和页记录。 |
| C，规范文本与块 | `literature_text.canonicalize_page`、`canonical_block`，PDF、DOCX 或 HTML 提取器按格式调用 | 前者执行 NFKC、换行与空白规范化；后者用字符起点加文本长度求终点，用 UTF-8 文本哈希和页面、类型、章节、偏移生成 canonical-block 身份。 |
| C，PDF 导入 | `ingestion/pdf.PypdfLiteratureTextExtractor.extract`，LiteratureIngestionService 等调用 | 按页树提取并规范化文本，以固定页间分隔符连接，累计字符数，输出页文本及规范页块。 |
| C，HTML 导入 | `ingestion/html.HTMLLiteratureTextExtractor.extract` 和其 parser，Acquisition 等调用 | 从允许的标签抽取文本、折叠空白并以两个换行连接，输出规范文本和相应块范围。 |
| C，DOCX 导入 | `ingestion/docx.DocxSupplementaryTextExtractor.extract`，补充材料导入调用 | 读取正文段落和表格单元，规范化段落，记录段落或行列路径；输出逻辑页及块，段落和单元之间的连接符计入偏移。 |
| C，PDF/HTML 结构 | `structure/pdf.PDFDocumentStructureExtractor.extract`、`structure/html.HTMLDocumentStructureExtractor.extract`，DocumentStructureService 调用 | PDF 保留行结束符推进页内游标，定位标题、正文和表图说明；HTML 将 DOM 事件按标签及文本哈希与已存规范块匹配，保留区域、章节、表格拓扑和图注绑定。输出结构记录。 |
| C，表图上下文、证据定位 | Literature context 的 `_table_context`、`_figure_context` 和对应 ownership 函数；`ingestion/evidence.py`、`structure/evidence.py` 及结构验证 | 文献 Builder 与其读回验证器使用选定结构及绑定记录投影表格行列、跨度和图注。证据流程在已有坐标上绑定原文、来源和定位信息。 |
| C，Provider 分批 | 两个领域 Provider 的 `partition_*_knowledge_chunks` | 输入为已定身份的 Chunk 元组，根据数量和总文本长度分配调用批次。既不切原文也不生成 Chunk ID。 |

两域的 Chunk 构造继续使用既有内容绑定函数。Expert 策略为 `spc-expert-source-offset-chunks`，Literature 策略为 `spc-structure-block-bounded-chunks`，版本均为 `1.0.0`。两个 Builder 直接调用 `knowledge/text_chunking.bounded_text_ranges`，显式传入各自的长度上限。该函数只处理字符串坐标，依赖 Python 标准库。

**历史边界。** 游标规则为 `max(boundary, cursor + 1)`。一个满窗全为空白时，下一字符会被越过。例如恰好 2,000 个空格后跟 `XY`，Expert 仅返回 `Y`；8,000 个空格在 Literature 中有相同表现。本轮忠实保留该行为。

## 固定样本

在修改生产代码前，从 `b414df2` 的两个正式 Builder 按返回顺序收录 21 份仓库实际写出的 JSON 字节，保存为 `tests/fixtures/knowledge-chunks-b414df2.zip`。文献 HTML 和 PDF 来自已提交的 `knowledge-01a-3f53101.zip`；专家 PDF 来自已提交的双页样本；另以同一段 8,100 字符的中英文、组合字符及 emoji 文本走两条完整调用链。文献测试仅在历史知识库的临时副本中加入测试用 Curation，以经过正常的编译准入检查。

| 调用与输入 | 基线块数 | 基线返回序列的完整内容 SHA-256 |
| --- | ---: | --- |
| Expert，双页 PDF | 2 | `fc40048ede40e53d5edc6c6611e990e577618a05a20c1db9020fef3595c82e05` |
| Expert，8,100 字符 | 5 | `7dee8b915bb8bc880c7ee5cadefc05233c52c6d6bb8287de95b62243ffb6365e` |
| Literature，历史 HTML | 9 | `754a705e04acd2fd036b4c2a1eef8ad31ca18a5fe84c34914d54655ac3fba3db` |
| Literature，历史 PDF | 3 | `e5056aa83de82fae1164c7d6f137b0893b424d6e0e8e430e0f181fdf179ae2b1` |
| Literature，8,100 字符 | 2 | `613c7e3e825f97aff931e51e594af4445364d5343b98222842cca8aaac52e75b` |

摘要按 `spc.serialization.content_hash(list(chunks))` 计算。回归测试从压缩包读取原记录字节，经各自的 Chunk Repository 验证后，逐条与重建结果比较所有模型字段。历史 HTML 的 9 个块中有 5 个带表格上下文，1 个带图片上下文。

专家来源的入库时间参与其来源哈希。对同一份原文重新导入会产生另一份输入。夹具因此随 Chunk 保存基线来源记录、原始字节和 Curation；测试只在临时目录读取该副本，不替换时间戳或跳过来源校验。

## 前后对照结果

两份旧 Builder 在同一批固定来源上通过 5 项基线回放。提取公共函数后再次回放，21 条记录逐项相等；上表中的五组有序摘要也完全相同。定向测试共 29 项通过，覆盖范围计算及已存文献、证据和结构记录兼容性。

| 历史 HTML 块 | 前后相同的偏移 | 前后相同的 Chunk ID | 前后相同的记录哈希 |
| --- | --- | --- | --- |
| 表格单元 | `[109, 114)` | `literature-knowledge-chunk-6bcc3b208a58a27030112614` | `05dd870d10da5914f38d351bd3af90ca6e432f30feb040ceb8d14070a7af15ea` |
| 图注 | `[133, 161)` | `literature-knowledge-chunk-c22fb11a6ce0fc53dc5d275e` | `d21782f9a99324063b24e2dcfec3956c99df87ffe667835d27109bec4769328a` |

另外，对两个旧函数分别穷举 5 字符输入（字母、空格、换行、制表符、组合符），取 1–5 字符上限并检查完整区间、子区间和空区间。93,750 次新旧偏移序列比较全部相同。

`python -m ruff check src tests` 通过。完整 `python -m pytest` 用时 414.16 秒，1,242 项通过，1 项跳过。Docling 集成测试因未配置 `SPC_DOCLING_ARTIFACTS_PATH` 跳过。
