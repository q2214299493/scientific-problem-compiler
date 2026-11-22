# SPC-CLEANUP-03D 内容绑定 ID 审计

基线 `9435a5f`。全仓审计 H02、清单 F025–F028 指向四个同名构造器。本次从调用者实际交付的字段追踪到两次哈希及模型验证，同时核对 Collection、Result Feedback 和历史记录的独立约定。文末索引给出其他域在基线中的函数及行号。

## 一、构造算法完全相同

`knowledge/expert_knowledge/{context,provider,materializer}.py` 和 `knowledge/literature_knowledge/context.py` 中的四个 `_content_bound` 函数体完全相同。它们接收模型、前缀、ID 字段名和调用者准备好的字典。第一次用 `serialization.content_hash` 哈希身份字典，将前 24 位小写十六进制与前缀用连字符连接成 ID。随后先插入 ID，再展开原字典，对这份载荷计算完整的 64 位记录哈希。两次计算都完成后才构造并验证目标模型。

规范 JSON 递归按映射键排序，使用紧凑分隔符、UTF-8 和原 Unicode 字符。序列保持顺序。字符串不做 Unicode 规范化。字典中显式的空值参与哈希。嵌套 Pydantic 模型以 JSON 模式展开，带上默认值、排除空值。字符串枚举沿用其 JSON 字符串值。四个构造器都不会提前展开目标模型的默认值。生成的 ID 只参与第二次哈希，记录哈希本身不参与两次计算。

| 调用者 | ID 前缀 / 字段名 | 第一次哈希的输入及调用者执行的预处理 |
| --- | --- | --- |
| Expert context，`resolve_expert_knowledge_input` | expert-knowledge-input / compilation_input_id | 专家 ID、档案哈希、专家来源 ID/哈希、原来源 ID/版本/哈希、归属 ID/哈希、领域、分块策略 ID/版本。 |
| Expert context，`build_expert_knowledge_chunks` | expert-knowledge-chunk / chunk_id | 专家来源及专家 ID、归属 ID/哈希、原来源 ID/版本/类型、起止偏移、原文、页码和章节路径。过滤顶层空值。 |
| Expert provider，`build_expert_knowledge_proposal_set` | expert-knowledge-proposals / proposal_set_id | 编译输入 ID/哈希、Provider ID/版本/配置哈希、按原顺序保存的批次调用（默认空元组），以及响应以 JSON 模式展开的引用、意见、案例提案和默认值。 |
| Expert materializer，`materialize` | expert-knowledge-grounding / grounding_id | 专家来源、归属、分块、证据的 ID/哈希，页码和章节路径。过滤顶层空值。 |
| Expert materializer，`materialize` | expert-knowledge-compilation / compilation_id | 编译输入、专家来源及归属的 ID/哈希，原来源 ID/版本/哈希，Provider ID/版本/配置哈希，提案集 ID/哈希，分块和证据定位哈希表，批次调用，排序后的意见及案例 ID，依原序保存的拒绝记录，编译器 ID/版本。 |
| Literature context，`resolve_literature_knowledge_input` | literature-knowledge-input / compilation_input_id | 文献、表示选择、表示、规范文本、结构选择及结构的 ID/哈希，表示种类，来源 ID/版本，编译与分块策略 ID/版本，以及排序后的纳入内容区域。 |
| Literature context，`build_literature_knowledge_chunks` | literature-knowledge-chunk / chunk_id | 文献/表示/结构 ID，有序块引用及其哈希，页码、章节、内容区域及不确定性，规范文本起止偏移、原文、表格和图片上下文。过滤顶层空值。 |

这些 ID 写入编译输入、分块、提案、证据定位及编译记录，后续验证通过它们绑定引用。共享函数只接收已经准备好的身份字典。上述七处调用继续拥有各自的字段选择、过滤和排序规则。

## 二、底层哈希相同，业务身份不同

| 权威实现及调用方 | 字段、前缀、默认值及历史用途 | 处理决定 |
| --- | --- | --- |
| `result_feedback/submission.build_content_bound`，Intake 与 Successor 调用 | 先以占位 ID/哈希构造草稿。模型声明顺序中的首个缺失 `_id` 字段为输出 ID。JSON 模式展开时排除**所有缺失的 ID 字段**、哈希和空值，保留其余模型默认值。调用者提供比较、评估、物化回执、研究更新、调用、父计划绑定、选择及周期的前缀。 | 复用最后的两次哈希及记录构造，保留接口和全部预处理。 |
| `result_feedback/submission.build_result_evidence_submission`，外部结果声明 | 固定 `result-submission` / `submission_id`。提交模型展开默认值，排除该 ID、哈希和空值。声明顺序及父 Run/Plan 绑定属于提交本身。 | 在此入口完成原有预处理后调用共享构造器。 |
| `collection/snapshots._make_bound`，发现、快照、Diff 和导入调用 | 先过滤顶层空值，再算身份哈希，然后按前缀选择 ID 字段。九种 `collection-` 前缀后缀依次映射为 page→page_id、resource→discovered_resource_id、occurrence→occurrence_id、snapshot→snapshot_id、acquisition-link→link_id、resource-import→result_id、membership→membership_id、import→import_id、diff→diff_id。页面规范化 URL 并对发现及分页引用去重排序，Diff 绑定新旧快照和变化集合。 | 保留前缀分派及异常发生顺序。 |
| `collection/urls.make_collection_definition` | `literature-collection` 绑定来源类型、规范入口 URL、名称（默认入口 URL）、领域、连接器 ID/版本/完整性约定及实际生效的范围策略。全记录哈希另含集合 ID。 | 保留 URL 规范化与策略默认值。 |
| `collection/connectors.GenericHTMLCollectionConnector.discover` | `collection-resource` 的 ID 只绑定集合 ID、资源类型和规范化标识符。全记录哈希额外包含发现和成员资格信息。这些元数据可在资源 ID 不变时改变。 | 保留两套字段范围。 |
| `model_definitions/literature.literature_identity_id`，文献导入调用 | 前缀 `literature`。有 DOI 时使用规范化 DOI，否则使用规范化标题、原序作者和年份。这里会执行 NFKC、大小写折叠和空白折叠，并在 DOI 上移除指定前缀。完整文献哈希还绑定来源和元数据等字段。 | 保留文献身份算法。 |
| `knowledge/ingestion/evidence._store_canonical_span` 和 Expert 物化 | 两者都用 `evidence`。文献证据绑定规范文本 ID、起止偏移和准确文本的 JSON 哈希。专家证据绑定来源 ID/版本、起止偏移和准确文本的 JSON 哈希。EvidenceSpan 保存来源字节 SHA，没有独立的 `content_hash` 字段。 | 分别保留来源绑定。 |
| Acquisition、Structure、Expert/Literature 的物化、视图和 Provider | 输入按实体分别包含解析器或资源元数据、定位坐标、意见、提案集、编译输出及来源记录。有的现场构造也使用双哈希，有的只返回 ID，或以另一组字段生成来源哈希。 | 本次合并可互换的通用构造器及 Result Feedback 的等价末段。各实体的业务构造留在原处。 |
| Planning、Workflow、Approval、Downstream、Retrieval | 候选/任务键、父计划绑定、Run/Provider 身份、导出或检索范围及有序引用有各自的生命周期。部分只计算一次摘要。 | 保留领域构造、回执与读取验证。 |

### Collection 与 Result Feedback 的调用输入

下表列出基线中每处调用交给构造器的字段。空值过滤、默认值展开、ID 字段选择按上表执行。行号对应 `9435a5f`。

| 调用者 | 前缀 | 调用时提供的字段 |
| --- | --- | --- |
| `knowledge/collection/connectors.discover:267` | `collection-occurrence` | collection_id, discovered_resource_id, discovery_page_id, discovery_index, original_identifier, discovery_method, discovery_confidence, discovery_context, membership_eligible, membership_decision, membership_reason |
| `knowledge/collection/connectors._page_record:313` | `collection-page` | collection_id, requested_url, resolved_url, http_status, media_type, response_sha256, page_artifact_id, page_artifact_hash, connector_id, connector_version, discovered_resource_refs, pagination_refs |
| `knowledge/collection/importer.run:147` | `collection-acquisition-link` | collection_id, snapshot_id, discovered_resource_id, discovered_resource_hash, acquisition_id, acquisition_hash, resulting_literature_id, acquisition_status |
| `knowledge/collection/importer.run:163` | `collection-resource-import` | collection_id, snapshot_id, discovered_resource_id, discovered_resource_hash, acquisition_link_id, acquisition_link_hash, acquisition_status |
| `knowledge/collection/importer.run:212` | `collection-import` | executed_at, collection_id, snapshot_id, snapshot_hash, acquisition_link_hashes, resource_result_hashes, membership_hashes, total_resources, discovery_occurrence_count, discovered_resource_count, logical_literature_count, ingested_literature_count, ingested_count, metadata_only_count, auth_required_count, unavailable_count, failed_count, unverified_membership_count, status, warnings |
| `knowledge/collection/importer._failed_resource_result:281` | `collection-resource-import` | collection_id, snapshot_id, discovered_resource_id, discovered_resource_hash, acquisition_status, failure_code |
| `knowledge/collection/importer._build_memberships:330` | `collection-membership` | collection_id, snapshot_id, literature_id, discovered_resource_refs, acquisition_refs, occurrence_refs |
| `knowledge/collection/snapshots.diff_collection_snapshots:65` | `collection-diff` | old_snapshot_id, old_snapshot_hash, new_snapshot_id, new_snapshot_hash, added_resource_ids, removed_resource_ids, changed_resource_ids, unchanged_resource_ids |
| `knowledge/collection/snapshots.finish_discovery:135` | `collection-snapshot` | collection_id, connector_id, connector_version, connector_completeness_contract, scope_policy_hash, page_record_hashes, discovered_resource_hashes, resource_provenance_hashes, occurrence_hashes, visited_page_count, unique_resource_count, traversal_status, completeness_status, completeness_basis, completeness, completeness_reasons |
| `result_feedback/intake._comparison:176` | `result-context-comparison` | submission_id, submission_hash, parent_plan_id, parent_plan_hash, status, differences, approved_deviation_refs |
| `result_feedback/intake.intake:383` | `result-assessment` | submission_id, submission_hash, parent_run_id, parent_plan_id, parent_plan_hash, task_id, capability_id, parent_authority_origin, parent_authority_ref, content_bound, plan_bound, context_compatible, scientifically_admissible, context_comparison, issues, status |
| `result_feedback/intake._materialize:742` | `result-materialization` | submission_id, submission_hash, assessment_id, assessment_hash, curation_id, curation_hash, parent_run_id, parent_plan_id, parent_plan_hash, task_id, parent_authority_origin, parent_authority_ref, source_id, source_version, accepted_evidence_ids, reported_result_hashes, reported_observation_hashes, method_fact_hashes, model_fact_hashes, missing_observables |
| `result_feedback/successor.load_complete_planning_stage:495` | `successor-cycle` | invocation_id, invocation_hash, parent_binding_id, parent_binding_hash, research_update_id, research_update_hash, context_id, context_hash, evidence_packet_id, evidence_packet_hash, planning_input_id, planning_input_hash, planning_proposal_id, planning_proposal_hash, candidate_plan_ids, candidate_plan_hashes, status |
| `result_feedback/successor._rebuild_complete_approval_cycle:637` | `successor-cycle` | invocation_id, invocation_hash, parent_binding_id, parent_binding_hash, research_update_id, research_update_hash, context_id, context_hash, evidence_packet_id, evidence_packet_hash, planning_input_id, planning_input_hash, planning_proposal_id, planning_proposal_hash, candidate_plan_ids, candidate_plan_hashes, selection_id, selection_hash, selected_plan_id, selected_plan_hash, approval_review_id, approval_review_hash, approval_verdict_id, approval_verdict_hash, approval_receipt_id, approval_receipt_hash, gate_id, gate_hash, status |
| `result_feedback/successor._research_update:989` | `research-update` | parent_binding_id, parent_binding_hash, previous_questions, previous_hypotheses, new_reported_observations, affected_acceptance_criteria, affected_falsification_criteria, unresolved_criteria, new_conflicts, new_evidence_gaps, why_successor_plan_is_needed, recommended_status |
| `result_feedback/successor.compile:1347` | `successor-invocation` | parent_run_id, parent_plan_id, parent_plan_version, parent_plan_hash, result_submission_ids, result_submission_hashes, materialization_receipt_ids, materialization_receipt_hashes, follow_up_request, planning_provider_id, planning_provider_version, planning_provider_config_hash |
| `result_feedback/successor.compile:1475` | `successor-parent-binding` | parent_run_id, parent_plan_id, parent_plan_version, parent_plan_hash, result_submission_ids, result_submission_hashes, materialization_receipt_ids, materialization_receipt_hashes, accepted_evidence_ids, successor_cycle_index |
| `result_feedback/successor.compile:1636` | `successor-cycle` | invocation_id, invocation_hash, parent_binding_id, parent_binding_hash, research_update_id, research_update_hash, context_id, context_hash, evidence_packet_id, evidence_packet_hash, planning_input_id, planning_input_hash, status |
| `result_feedback/successor.compile:1819` | `successor-cycle` | invocation_id, invocation_hash, parent_binding_id, parent_binding_hash, research_update_id, research_update_hash, context_id, context_hash, evidence_packet_id, evidence_packet_hash, planning_input_id, planning_input_hash, planning_proposal_id, planning_proposal_hash, candidate_plan_ids, candidate_plan_hashes, status |
| `result_feedback/successor.compile:1853` | `successor-selection` | invocation_id, invocation_hash, proposal_id, proposal_hash, candidate_id, candidate_hash, selection_provenance |

### Knowledge 内联构造点的字段对照

下表记录各函数用于构造身份的字段来源。行号相对基线的 `src/spc/knowledge`。这些前缀都截取身份摘要的前 24 位；同名前缀可能绑定不同的字段。表中列出构造映射时提供的键，最终纳入哈希的键还取决于该函数的过滤和默认值处理。例如 `_make_resource` 排序候选、补三个空元组并过滤空值。`document-block` 在区域为 UNKNOWN 时排除该字段。`canonical-block` 的内层文本哈希使用原文 UTF-8 字节。各构造点继续使用原有的排序、过滤及内部摘要。

| 函数及行号 | 前缀 | 身份字段来源 |
| --- | --- | --- |
| `acquisition/metadata.py:77` _make_candidate | `fulltext-candidate` | 调用方传入的全部关键字参数 |
| `acquisition/metadata.py:100` _make_resource | `resolved-literature` | 原输入字段, metadata_retrieval_refs, metadata_retrieval_hashes, fulltext_candidates |
| `acquisition/metadata.py:128` _make_metadata_retrieval | `metadata-retrieval` | source_url, response_sha256, response_byte_size, media_type, resolver_id, resolver_version, response_payload_utf8 |
| `acquisition/metadata.py:145` _make_request | `literature-acquisition-request` | original_input, input_kind, requested_domain |
| `acquisition/metadata.py:310` _merge_metadata | `metadata-merge` | decisions |
| `acquisition/service.py:685` _make_attempt | `acquisition-attempt` | request_id, request_hash, candidate_id, candidate_hash, attempt_index, status, http_status, media_type, failure_code, raw_artifact_id, raw_artifact_hash, canonical_text_id, canonical_text_hash, ingestion_id, ingestion_hash, representation_id, representation_hash, metadata_merge_manifest_id, metadata_merge_manifest_hash |
| `acquisition/service.py:742` _finish | `literature-acquisition` | request_id, request_hash, resolver_id, resolver_version, resolved_resource_id, resolved_resource_hash, selected_candidate_id, selected_candidate_hash, metadata_merge_manifest_id, metadata_merge_manifest_hash, attempt_refs, attempt_hashes, resulting_literature_id, resulting_ingestion_id, resulting_canonical_text_id, resulting_representation_id, resulting_representation_hash, status, warnings |
| `collection/connectors.py:145` discover | `collection-resource` | collection_id, resource_kind, normalized_identifier |
| `collection/urls.py:139` make_collection_definition | `literature-collection` | source_kind, entry_url, name, domain, connector_id, connector_version, connector_completeness_contract, scope_policy |
| `expert_knowledge/materializer.py:144` materialize | `evidence` | source_id, source_version, start_offset, end_offset, text_hash |
| `expert_knowledge/materializer.py:209` materialize | `expert-opinion` | expert_id, domain, topic, statement, opinion_type, scope, rationale, conditions, evidence_refs, attribution_refs, related_claim_refs, related_workflow_refs, authority_status |
| `expert_knowledge/materializer.py:256` materialize | `expert-case` | domain, vague_request, translated_questions, positive, rationale, evidence_refs, opinion_refs, original_wording, latent_concern, atomic_questions, good_question_formulations, wrong_formulations, answerability_conditions, required_evidence_types, baseline_guidance, common_misinterpretations, resolution_pattern, applicability |
| `expert_knowledge/source.py:45` make_expert_profile | `expert` | display_name, organization, role |
| `expert_knowledge/source.py:92` _pdf_text | `literature-artifact` | literature_id, sha256 |
| `expert_knowledge/source.py:242` ingest | `evidence` | source_id, source_version, start_offset, end_offset, text_hash, purpose |
| `expert_knowledge/source.py:261` ingest | `expert-attribution` | expert_id, source_id, source_version, evidence_refs, medium, attribution_basis, attribution_status |
| `expert_knowledge/source.py:285` ingest | `expert-source` | expert_id, source_relationship, source_type, domain, input_kind, raw_sha256, raw_artifact_path, canonical_sha256, source_id, source_version, source_hash, attribution_id, attribution_hash, attribution_status, page_ranges |
| `graph.py:159` build | `knowledge-graph` | view_mode, domain_filter, topic_filter, nodes, edges |
| `graph.py:220` _node | `knowledge-node` | record_type, record_id |
| `ingestion/evidence.py:23` _store_canonical_span | `evidence` | canonical_text_id, start_offset, end_offset, text_hash |
| `ingestion/service.py:201` _make_ingestion | `literature-ingestion` | literature_id, raw_artifact_id, raw_artifact_hash, canonical_text_id, canonical_text_hash, source_id, source_version, parser_id, parser_version, parser_config_hash, ingestion_status, total_pages, pages_with_text, pages_without_text, text_coverage_ratio, warnings |
| `ingestion/service.py:256` _html_ingestion_record | `html-literature-ingestion` | literature_id, raw_artifact_id, raw_artifact_hash, canonical_text_id, canonical_text_hash, source_id, source_version, extractor_id, extractor_version, extractor_config_hash, ingestion_status |
| `ingestion/supplementary.py:126` _ingest_literature_supplementary | `literature-supplement` | literature_id, literature_hash, main_selection_id, main_selection_hash, raw_artifact_id, raw_artifact_hash, canonical_text_id, canonical_text_hash, source_id, source_version, source_sha256 |
| `literature_knowledge/codex_transport.py:440` generate_structured | `literature-knowledge-provider-invocation` | transport_id, transport_version, runtime_version, selected_model, invocation_config_hash, input_hash, output_hash |
| `literature_knowledge/materializer.py:123` _rebind_or_create_quote_evidence | `evidence` | canonical_text_id, start_offset, end_offset, text_hash |
| `literature_knowledge/materializer.py:370` _grounding_record | `literature-knowledge-grounding` | literature_id, literature_hash, representation_selection_id, representation_selection_hash, representation_id, representation_hash, structure_selection_id, structure_selection_hash, structure_id, structure_hash, block_id, block_hash, evidence_id, evidence_hash, locator_id, locator_hash, quote_id, content_region |
| `literature_knowledge/materializer.py:509` materialize | `claim` | text, claim_type, source_role, evidence_refs, source_quote_refs, claim_strength, epistemic_status |
| `literature_knowledge/materializer.py:540` materialize | `method-fact` | text, attributes, evidence_refs, epistemic_status |
| `literature_knowledge/materializer.py:561` materialize | `model-fact` | text, attributes, evidence_refs, epistemic_status |
| `literature_knowledge/materializer.py:611` materialize | `result-context` | system_context, method_context, method_fact_refs, model_fact_refs |
| `literature_knowledge/materializer.py:625` materialize | `result` | quantity, value, unit, system_context, method_context, result_context, evidence_refs, result_status |
| `literature_knowledge/materializer.py:676` materialize | `knowledge-relation` | subject_type, subject_id, predicate, object_type, object_id, domain, evidence_refs, rationale |
| `literature_knowledge/materializer.py:707` materialize | `literature-knowledge-compilation` | compilation_input_id, compilation_input_hash, provider_id, provider_version, provider_config_hash, proposal_set_id, proposal_set_hash, representation_selection_id, representation_selection_hash, structure_selection_id, structure_selection_hash, source_quote_ids, source_claim_ids, method_fact_ids, model_fact_ids, reported_result_ids, knowledge_relation_ids, grounding_hashes, rejected_proposals, compiler_id, compiler_version |
| `literature_knowledge/provider.py:137` build_literature_knowledge_proposal_set | `literature-knowledge-proposals` | compilation_input_id, compilation_input_hash, provider_id, provider_version, provider_config_hash, provider_invocation, batch_invocations, 响应的 JSON 字段（排除空值） |
| `literature_knowledge/provider.py:457` _aggregate_invocation | `literature-knowledge-provider-invocation` | transport_id, transport_version, runtime_version, selected_model, invocation_config_hash, input_hash, output_hash |
| `literature_knowledge/validation.py:94` curate_knowledge_record | `knowledge-curation` | target_type, target_id, target_hash, status, curator_id, rationale, evidence_refs, supersedes_curation_id |
| `literature_knowledge/view.py:219` build | `literature-knowledge-view` | literature_id, view_mode, records, compilation_ids, rejected_proposals |
| `literature_text.py:142` canonical_block | `canonical-block` | page_number, block_type, section_path, start_offset, end_offset, text_hash |
| `representation.py:164` build_representation_reference | `literature-representation` | representation_kind, literature_id, raw_artifact_id, raw_artifact_hash, canonical_text_id, canonical_text_hash, ingestion_id, ingestion_hash, source_id, source_version |
| `representation.py:284` select | `literature-selection` | literature_id, representation_id, representation_hash, selected_by, rationale, supersedes_selection_id |
| `structure/evidence.py:165` create_structured_evidence_locator | `structured-locator` | evidence_id, source_id, source_version, literature_id, representation_id, structure_id, block_id, page_number, section_path, content_region, table_id, table_cell_id, row_index, column_index, figure_id, canonical_start_offset, canonical_end_offset |
| `structure/records.py:104` _make_block | `document-block` | structure_id, block_type, ordinal, parent_block_id, page_number, heading_level, section_path, content_region, label, start_offset, end_offset, text_hash |
| `structure/records.py:133` _make_table | `table-structure` | structure_id, ordinal, label, caption_block_ref, page_number, section_path, extraction_status |
| `structure/records.py:166` _make_cell | `table-cell` | table_id, row_index, column_index, row_span, column_span, is_header, canonical_block_refs, start_offset, end_offset, text_hash |
| `structure/records.py:191` _make_figure | `figure-structure` | structure_id, ordinal, label, page_number, section_path, caption_block_ref, extraction_status |
| `structure/service.py:38` _structure_id | `document-structure` | literature_id, representation_id, representation_hash, representation_kind, canonical_text_id, canonical_text_hash, source_id, source_version, extractor_id, extractor_version, extractor_config_hash |
| `structure_selection.py:215` _create_selection | `document-structure-selection` | literature_id, representation_id, representation_hash, structure_id, structure_hash, selected_by_policy, selector_version, rationale, supersedes_selection_id |

## 三、算法和用途不同

原始 Artifact 和文件 SHA-256 直接对源字节求摘要。Provider 输入输出还可能绑定传输字节。来源校验和、修订指纹、目录名及诊断标记各有用途。`validation.common.content_binding_error` 在读取边界独立重算预期绑定。以上实现均保留。

## 历史基线与前后对照

修改前从三个已提交的固定归档取得下列数值。通过文献身份入口重算文献 ID，在临时副本中按历史偏移及原文重建两条证据，分别重建集合页面、结果提交、物化回执和 Successor 周期。两条 Evidence 的 SHA-256 是完整 EvidenceSpan 的规范内容哈希。

| 历史归档 / 记录 | 修改前后相同的 ID | 修改前后相同的 SHA-256 |
| --- | --- | --- |
| knowledge-01a-3f53101.zip, HTML 文献 | `literature-dd1abde275237ac8309698cd` | `f82eeb8f171f3b00ac68ed0f270393c22031426bff28df2ab52aea3129f6b616` |
| 同一归档， 第一条 Evidence Span | `evidence-4e53ef2dd414cfaeae594c11` | `ab99205b9c69af2b3645a78ef661114a9aa90deadea71ab742532bb9cd8c53d8` |
| 同一归档， 第二条 Evidence Span | `evidence-c5ac59b2432a1185835820f1` | `e3084046d7f77481b39130d6ef22eaa3a8de7691cdc3fb3a50f84f86487a31d2` |
| repositories-ac25bd4.zip, Collection 页面 | `collection-page-a528966052fce62888fb4eb4` | `65d75c627adaca94b009ecc25c55e2c7a468e9bf81a991d265f8c93feef96a38` |
| model-history-f5cfdd7.zip, ResultEvidenceSubmission.yaml | `result-submission-8c77318a76db82b48f16fcef` | `c89c9d912cd971dd8488929a04085faf9e480f04b89fcc9fca5ad81accf3c042` |
| 同一归档， ResultEvidenceMaterializationReceipt.yaml | `result-materialization-c314dd63cebe18f3fd9b2062` | `dbd703b8daee6d79514a87b42d46a3e4a9ef0347a70d09f41ee308180f15ff53` |
| 同一归档， SuccessorPlanningCycle.yaml | `successor-cycle-8c5f8b20a905563b929e10bd` | `d02d298f79d85f24e5aa3d702c0b3a4bd6f045972e97075100a788d82dbe432c` |

修改前，四份旧实现对七组固定输入都得出了相同的 ID 和记录哈希。回归测试固定了这些摘要，并覆盖键顺序、序列顺序、显式空值、模型默认值、Unicode 组合形式、字符串枚举及嵌套模型。

另用专家和文献编译测试将真实调用产生的同一份输入分别传给 Git 基线中的原函数和新函数。55 次调用的模型、ID 和哈希逐项相等，其中专家编译输入、分块、提案、证据定位和编译记录各 1 次，文献编译输入 5 次，文献分块 45 次。

`python -m ruff check src tests` 通过。完整 `python -m pytest` 运行 515.06 秒，1224 项通过，1 项跳过。跳过项为 Docling 集成测试，原因是本机未配置 `SPC_DOCLING_ARTIFACTS_PATH`。

## 基线源码索引

路径相对 `src/spc`。下表定位标识符相关语句中的哈希截取，包括内联构造和读取时的重算。括号中的数字为基线行号。前述表格列出通用构造器的间接调用点。

| 文件 | 函数及行号 |
| --- | --- |
| `approval/context_resolver.py` | `resolve` (92, 204), `resolve_revision` (277) |
| `approval/service.py` | `review` (69, 86, 122) |
| `backends/builders.py` | `build_external_document_element` (57), `build_external_document_parse_proposal` (86), `build_external_retrieval_hit` (116) |
| `backends/metadata/grobid.py` | `extract_metadata` (139), `extract` (177) |
| `backends/provenance.py` | `build_backend_runtime_artifact_manifest` (125), `build_backend_runtime_identity` (193), `create_backend_run_record` (364) |
| `backends/rebinding.py` | `_rebound_element` (212), `rebind` (269), `structure` (662) |
| `backends/retrieval/paperqa.py` | `_hit` (63), `retrieve` (160) |
| `backends/retrieval/resolver.py` | `_resolution` (70), `resolve_batch` (260) |
| `backends/retrieval/service.py` | `retrieve` (54) |
| `downstream/context.py` | `project` (103) |
| `downstream/proposal.py` | `build` (257) |
| `downstream/validator.py` | `import_package` (149) |
| `interpretation/claim_extractor.py` | `source_quote_id` (31), `extract_source_claims` (85) |
| `interpretation/comparison_analyzer.py` | `analyze_comparisons` (46) |
| `interpretation/conflict_detector.py` | `detect_conflicts` (72, 88) |
| `interpretation/evidence_packet_builder.py` | `build` (33, 80) |
| `interpretation/gap_analyzer.py` | `analyze_gaps` (73) |
| `interpretation/method_fact_extractor.py` | `extract_method_facts` (21) |
| `interpretation/mock_provider.py` | `interpret` (44, 90) |
| `interpretation/model_fact_extractor.py` | `extract_model_facts` (21) |
| `interpretation/result_extractor.py` | `extract_reported_results` (108, 119) |
| `knowledge/acquisition/metadata.py` | `_make_candidate` (77), `_make_resource` (100), `_make_metadata_retrieval` (128), `_make_request` (145), `_merge_metadata` (310) |
| `knowledge/acquisition/service.py` | `_make_attempt` (685), `_finish` (742) |
| `knowledge/collection/connectors.py` | `discover` (145) |
| `knowledge/collection/snapshots.py` | `_make_bound` (26) |
| `knowledge/collection/urls.py` | `make_collection_definition` (139) |
| `knowledge/expert_knowledge/context.py` | `_content_bound` (23) |
| `knowledge/expert_knowledge/materializer.py` | `_content_bound` (48), `materialize` (144, 209, 256) |
| `knowledge/expert_knowledge/provider.py` | `_content_bound` (56) |
| `knowledge/expert_knowledge/source.py` | `make_expert_profile` (45), `_pdf_text` (92), `ingest` (209, 242, 261, 285) |
| `knowledge/graph.py` | `build` (159), `_node` (220) |
| `knowledge/ingestion/evidence.py` | `_store_canonical_span` (23) |
| `knowledge/ingestion/service.py` | `_make_ingestion` (201), `_html_ingestion_record` (256) |
| `knowledge/ingestion/supplementary.py` | `_ingest_literature_supplementary` (126) |
| `knowledge/literature_knowledge/codex_transport.py` | `generate_structured` (440) |
| `knowledge/literature_knowledge/context.py` | `_content_bound` (48) |
| `knowledge/literature_knowledge/materializer.py` | `_rebind_or_create_quote_evidence` (123), `_grounding_record` (370), `materialize` (509, 540, 561, 611, 625, 676, 707) |
| `knowledge/literature_knowledge/provider.py` | `build_literature_knowledge_proposal_set` (137), `_aggregate_invocation` (457), `_persist_diagnostic` (486) |
| `knowledge/literature_knowledge/validation.py` | `curate_knowledge_record` (94) |
| `knowledge/literature_knowledge/view.py` | `build` (219) |
| `knowledge/literature_text.py` | `canonical_block` (142) |
| `knowledge/representation.py` | `build_representation_reference` (164), `select` (284) |
| `knowledge/structure/evidence.py` | `create_structured_evidence_locator` (165) |
| `knowledge/structure/records.py` | `_make_block` (104), `_make_table` (133), `_make_cell` (166), `_make_figure` (191) |
| `knowledge/structure/service.py` | `_structure_id` (38) |
| `knowledge/structure_selection.py` | `_create_selection` (215) |
| `model_definitions/literature.py` | `literature_identity_id` (169) |
| `planning/context_resolver.py` | `resolve` (169, 255) |
| `planning/evidence_resolution.py` | `_materialize_request` (371), `build_planning_evidence_request_set` (542), `_make_acquisition_proposal` (770), `_make_resolution` (832), `build_evidence_resolution_set` (1036), `augment_scientific_context` (1186) |
| `planning/hierarchical.py` | `build_hierarchical_stage_attempt` (116), `_materialize_direction` (128), `build_research_direction_set` (156), `build_direction_triage_record` (408), `build_hierarchical_expansion` (661) |
| `planning/identity.py` | `derive_candidate_task_id` (14) |
| `planning/materializer.py` | `_entity_id` (39, parameterized prefix and single-pass digest), `build_compilation_receipt` (89) |
| `planning/mock_provider.py` | `build_proposal_set` (68) |
| `planning/revision.py` | `_feedback_id` (108), `build_plan_revision_input` (242) |
| `repository_components/collection.py` | `put` (53) |
| `repository_components/literature.py` | `put` (93), `put` (249), `put` (320), `put` (369) |
| `repository_components/snapshot.py` | `create_snapshot` (272) |
| `result_feedback/intake.py` | `_comparison` (172), `_curation_record` (433), `curate` (493), `_materialize` (574, 600, 609, 633, 659, 673) |
| `result_feedback/submission.py` | `build_content_bound` (44), `build_result_evidence_submission` (61) |
| `result_feedback/successor.py` | `load_complete_planning_stage` (462), `_snapshot_with_results` (766), `_successor_context` (796, 832, 858, 873), `_successor_packet` (1014, 1090, 1154), `compile` (1732, 1925) |
| `retrieval/context_builder.py` | `build` (132) |
| `retrieval/knowledge_retriever.py` | `retrieve` (204) |
| `retrieval/query_builder.py` | `build_retrieval_query` (61) |
| `validation/common.py` | `content_binding_error` (54) |
| `workflow/orchestrator.py` | `start` (92) |
| `workflow/stages/approval.py` | `approve` (295) |
| `workflow/stages/evidence_resolution.py` | `_make_evidence_cycle_policy` (86), `_make_evidence_cycle_record` (102), `_make_evidence_request_attempt` (118), `_carry_unresolved_evidence_gaps` (166) |
| `workflow/stages/planning.py` | `record_candidates` (243) |
| `workflow/stages/revision.py` | `start_chain` (90), `revise` (486, 536, 558) |
