# Scientific Problem Compiler (SPC)

[![CI](https://github.com/q2214299493/scientific-problem-compiler/actions/workflows/ci.yml/badge.svg)](https://github.com/q2214299493/scientific-problem-compiler/actions/workflows/ci.yml)

SPC compiles vague scientific requests, reviewer comments, and source evidence into one to four evidence-grounded, falsifiable scientific question plans. Phase 1 is offline and planning-only: it never calls an external LLM or executes scientific software.

The compiler, independent approver, and exporter have separate data boundaries. A plan can be exported only after a hash-bound plan approval, a passed plan gate, and an explicit human selection. Every exported task is forced to `runnable: false`.

Before every export, SPC reloads the selected domain pack, checks its version, reruns `validate_question_plan` against the on-disk evidence store, and verifies a `PlanValidationRecord` bound to the plan ID, version, content hash, domain, and domain-pack version. Every `EvidenceSpan` is checked through its `SourceDocument` to the hash and exact offsets of the stored source file. `GateVerdict` is additionally hash-bound to both that validation record and the independent `ApprovalVerdict`. Hard red flags block export, and every blocking fix and required human decision must have a parsed, valid resolution.

## Quick start

```powershell
python -m pip install -e ".[dev]"
spc --help
pytest
```

`spc validate PLAN --state-dir .spc --record-output validation.yaml` creates the validation record required by `spc export --validation-record validation.yaml`. Export packages are checked for required files, checksums, and cross-file semantic consistency in a same-filesystem staging directory before one atomic rename into their final path.

The package ships `base` and `fischer_tropsch` domain packs. Domain-specific terminology and capabilities live in those packs; the core models contain no Fischer–Tropsch-specific fields.

## State and safety

Project planning state is stored under `.spc/`. Source content is copied into versioned, hash-addressed read-only records. Export packages are immutable by convention: SPC refuses to overwrite an existing export directory and writes a checksum manifest.

Core scientific text fields reject empty and whitespace-only values. Evidence validation explicitly matches each `EvidenceSpan` source ID and version to its `SourceDocument`, and export validation rejects a symlinked `checksums.json` before reading it.

Phase 1 explicitly rejects runnable tasks and command-bearing execution policies. It does not generate VASP, NEB, Dimer, MKM, KMC, HPC, or other execution inputs.

## Phase 2A retrieval

Phase 2A adds deterministic, offline lexical retrieval without changing the frozen Phase 1 plan, approval, gate, or export contracts. It searches verified `EvidenceSpan` records and domain-compatible expert cases, workflow patterns, and scientific capabilities. Exact phrases, Domain Pack aliases/synonyms, and token overlap are scored in that order; every hit records its score, matched terms, rationale, source record ID, and retriever version.

```powershell
spc retrieve request.txt `
  --domain fischer_tropsch `
  --state-dir .spc `
  --knowledge-dir knowledge `
  --output context.yaml
```

The resulting `ScientificContextPacket` binds the query hash, Domain Pack version, deterministic knowledge snapshot ID, record hashes, evidence source versions, ordered result IDs, and the corresponding ordered `RetrievalHit` hashes. `result_hashes` is mandatory; legacy manifests that omit it or use an identity computed without it are rejected. Its content hash is semantic: `KnowledgeSnapshot.created_at` and `RetrievalManifest.timestamp` remain audit metadata but do not change retrieval, context, or content identity. Retrieval remains planning-only and never invokes an LLM or scientific execution backend.

## Phase 2B interpretation

Phase 2B converts a `ScientificContextPacket` into a hash-bound `ScientificEvidencePacket`. The `MockInterpretationProvider` is deterministic and offline; the atomic-quote implementation reports provider version `mock-interpretation-2.0.0` so its provenance cannot be confused with the earlier algorithm. Phase 2B.2 separates atomic, exact `SourceQuote` text from normalized or paraphrased `SourceClaim` text. Every quote records relative offsets whose slice must exactly recover its text from one integrity-verified `EvidenceSpan`; its ID binds the evidence ID, offsets, and quote-text hash. A claim remains bound through explicit quote and evidence references rather than substring matching. Retrieved statements are never promoted to established facts.

```powershell
spc interpret context.yaml `
  --provider mock `
  --state-dir .spc `
  --output evidence-packet.yaml
```

Interpretation validation reopens every referenced `EvidenceSpan`, verifies its source file and Phase 2A snapshot hash, and rejects unretrieved, fabricated, out-of-range, or inexact quotes. Source roles and types use central, domain-neutral enums; provenance overrides punctuation heuristics, so a question mark alone never creates reviewer provenance. Numerical results retain units and the existing `ResultContext` links to applicable `MethodFact` and `ModelFact` records. Facet or method mismatches produce explicit comparison constraints, source conflicts remain unresolved, and predictive models are not relabeled as DFT results. Phase 2B does not generate a `ScientificQuestionPlan` and does not execute scientific software.

## Phase 2C grounded planning

Phase 2C resolves every retrieved knowledge record against the hash-bound `KnowledgeSnapshot`, revalidates the interpreted packet against `SourceEvidenceStore`, builds an immutable `ScientificPlanningInput`, and asks a separate `PlanningProvider` for one to four grounded `CandidatePlanDraft` objects. The deterministic `PlanMaterializer` assigns all authoritative IDs, fingerprints, DAG task IDs, and plan identities. Candidate distinctions use an axis/value pair, claim references remain in final plan provenance, and proposed deviations must bind an existing comparison baseline. Source hypotheses and reviewer requests remain non-factual, unresolved conflicts remain explicit, blocking evidence gaps must be addressed or propagated, and all tasks remain `runnable: false`.

```powershell
spc plan context.yaml evidence-packet.yaml `
  --domain fischer_tropsch `
  --state-dir .spc `
  --knowledge-dir knowledge `
  --provider mock `
  --output-dir .spc/candidates
```

`StructuredLLMPlanningProvider` accepts a replaceable `LLMTransport`, sends the non-authoritative `PlanningLLMResponse` JSON Schema, records the model and generation configuration, and retries malformed structured output within a fixed bound. SPC—not the model—binds proposal IDs, planning-input hashes, and provider identity. Normal CI uses only `FakeLLMTransport`; no API key or network access is required. The LLM path has no tool, shell, file, or scientific-execution access, and source text is passed only as untrusted evidence data.

## Phase 2D independent approval

Phase 2D reconstructs an immutable `ApprovalReviewInput` from the original request, primary evidence, interpreted evidence, planning context, candidate, and current deterministic validation. It revalidates the evidence store, Domain Pack, knowledge snapshot, candidate provenance, candidate hash, and `PlanValidationRecord` before review. A separate `ApprovalProvider` emits only a non-authoritative `ApprovalLLMResponse`; deterministic `ApprovalPolicy` prevents scores from overriding failed validation, blocking red flags, or unresolved human decisions. SPC then binds the exact candidate to an authoritative `ApprovalVerdict` without modifying the plan or passing the Plan Gate.

```powershell
spc review context.yaml evidence-packet.yaml planning-input.yaml `
  candidate-plan.yaml validation-record.yaml `
  --provider mock `
  --state-dir .spc `
  --knowledge-dir knowledge `
  --output approval-review.yaml
```

`StructuredLLMApprovalProvider` reuses the vendor-neutral transport but has a separate protocol, prompt, response schema, and provider identity from planning. Normal CI remains offline through `FakeLLMTransport`.

Phase 2D.1 closes the approval trust chain with a content-bound
`IndependentApprovalReceipt`. `spc review` now writes the review input, review
record, authoritative verdict, and receipt. A Phase 2C-materialized plan cannot
receive a passed Plan Gate or be exported unless those artifacts bind the same
candidate, provider, approver, review, and verdict hashes. The legacy
`spc approve` command remains available only for manual Phase 1 compatibility;
its verdict has no independent-review receipt and cannot satisfy the Phase 2D
boundary. Phase 2D export additionally requires `--review-input`,
`--review-record`, and `--approval-receipt`; all three artifacts are preserved
inside the immutable export package.

Phase 2D.1.1 makes `ProjectTrustPolicy` the external authority for that
boundary. Normal `spc plan` output uses `independent_required` and includes a
content-bound `PlanCompilationReceipt` for every candidate. Gate and export
bind the exact policy and compilation receipt, so clearing or rewriting
`source_proposal` cannot downgrade a grounded plan. Manual approval is accepted
only when the caller explicitly supplies a `legacy_manual_allowed` policy.
`source_proposal` remains lineage evidence, not an authorization control.

## Phase 3A trusted downstream handoff

Phase 3A adds a one-way trust boundary from an independently approved export
to a downstream `ExecutionProposal`. `DownstreamImportValidator` first runs the
complete export checksum and semantic validation, then binds the selected plan,
manifest, passed gate, external trust policy, compilation receipt, independent
approval receipt, and capability mappings into an immutable
`SPCExportPackage`. Legacy/manual exports are not accepted at this boundary.

`ExecutionProposalBuilder` resolves one exported task through the target
agent's `AgentCapabilityCatalog` and a replaceable `AgentExecutionAdapter`.
The adapter receives capability identifiers and catalog data, never the
`ScientificQuestionPlan`. The resulting proposal records inputs, expected
outputs, assumptions, resource metadata, validation requirements, and
provenance requirements, but it is always `authorized: false` and
`runnable: false`. SPC Core produces no shell command, submission request, or
scientific execution input; downstream authorization and execution remain
future phases.

Phase 3A.1 projects each selected `DAGTask` and its plan-level scientific
meaning into a content-bound `ScientificTaskExecutionContext`. Target input
requirements must resolve to fields in that context, and every scientific
output must have an explicit adapter reconciliation to the catalogued output
contract. Proposal dependencies exactly preserve the selected task DAG.
Capability catalogs reject duplicate or ambiguous mappings, and adapter
mapping results are checked for determinism. Recursive payload validation
rejects command-, script-, executable-, shell-, scheduler-, or submission-
bearing keys at any depth. These checks do not grant authorization:
`ExecutionProposal` remains both unauthorized and non-runnable.

## Knowledge Layer K1A

K1A adds immutable persistence contracts for curated literature metadata,
expert profiles, expert opinions, and domain-general scientific relations. It
reuses the existing evidence, claim, result, fact, expert-case, workflow, and
capability models rather than creating parallel scientific records. Repository
writes bind each safe record key to its model identity and reject conflicting
overwrites; relation IDs are unique and content-bound.

K1A.1 separates scientific identity from curation state. Literature IDs use a
normalized DOI when available, otherwise normalized title, authors, and year;
changing curation status never changes literature, opinion, or relation IDs.
Every transition is instead an immutable, hash-bound `KnowledgeCurationRecord`
that supersedes the prior decision without overwriting it.

`KnowledgeSnapshot` binds only accepted records whose current curation chain,
source document, evidence spans, expert/profile references, claims, workflows,
and relation endpoints pass `TrustedKnowledgeValidator`. Missing or tampered
provenance fails closed. `KnowledgeGraphBuilder` defaults to this trusted view;
its audit view may show all curation states and labels each node and edge status
in text. Mermaid output remains only a reproducible view: immutable repository
and curation records are authoritative.

K1A.2 closes provenance recursively for every scientific record reachable from
trusted relations. Quote offsets and source metadata, claim/quote evidence
equality and epistemic restrictions, result contexts, fact references, and
case/workflow evidence are validated with the Phase 2B binding rules. Expert
opinions require an immutable `ExpertAttributionRecord` whose named expert,
source version, and verified evidence cover the opinion. `trusted_record_hashes`
binds all reachable records into the snapshot while the earlier typed hash maps
remain available for compatibility.

The package includes an offline, non-FT fixture with independent literature and
expert-note sources, one literature claim relation, one expert-opinion
provenance relation, and one literature-workflow relation. K1A does not parse
PDFs, call external services or LLMs, build an embedding/vector store, or
authorize or execute scientific work.

## Knowledge Layer K1B

K1B stores each born-digital PDF as an immutable `RawLiteratureArtifact`, then
uses a replaceable, versioned `LiteratureTextExtractor` to produce a canonical
UTF-8 `CanonicalTextArtifact` with deterministic page/block offsets. A
`LiteratureIngestionRecord` binds both artifacts, parser configuration and the
canonical `SourceDocument`; PDF bytes are never an `EvidenceSpan` source.
Textless PDFs fail closed as `requires_ocr`, and K1B performs no OCR or claim
extraction. Reprocessing with another parser version retains the stable
DOI/bibliographic literature identity while adding a new immutable canonical
artifact, source version and ingestion record.

The `spc ingest-literature` command accepts a PDF plus bibliographic metadata.
Trusted snapshots recursively verify the raw bytes, canonical UTF-8 text,
block offsets, ingestion hashes and SourceDocument binding before use.

K1B.1 adds an immutable `LiteratureRepresentationSelection` chain. K1B.2 makes
promotion explicit: successful ingestion only creates a candidate, while
`spc select-literature-representation` validates and selects it without
extracting scientific claims. The current selection must then receive an
accepted curation record before trusted use. Trusted snapshots bind only that
selected ingestion/canonical/raw/source chain; unused successful ingestions
remain audit history. Evidence creation requires the current selection by
default, while historical evidence requires both explicit historical mode and
a record authorizing the exact evidence IDs before trusted scientific use.
Ingestion records retain page-level text coverage, and expected parser failures
persist a non-sensitive `failed` audit record after the raw PDF has been stored.

## Knowledge Layer K1C

K1C adds one acquisition gateway for DOI strings, HTTP(S) article resources,
and existing local PDFs. Immutable request, resolved-resource, full-text
candidate, and acquisition records preserve the resolver identity and the
complete handoff into K1B. DOI metadata resolution and article-page discovery
never fabricate unavailable full text. Local and downloaded PDFs use the same
K1B artifact, canonical-text, ingestion, and representation-promotion rules;
acquisition never curates or selects the new representation.

```powershell
spc add-literature 10.1234/example `
  --domain base `
  --knowledge-dir knowledge `
  --state-dir .spc
```

The replaceable network boundary accepts only HTTP(S), resolves and checks every
redirect target, rejects local/private/link-local destinations, enforces time
and size limits, and validates response media types. Remote PDFs are signature-
and hash-checked in a controlled staging directory before K1B ingestion. HTML
articles are canonicalized as deterministic UTF-8 blocks while retaining their
source URL and raw-artifact hash; they are not misrepresented as PDF ingestion.
CI uses injected offline transports and never requires live network access.

K1C.1 persists HTML full text as immutable raw bytes plus canonical UTF-8 text,
exact recoverable blocks, a `SourceDocument`, and a common PDF/HTML literature
representation reference. Each candidate attempt is recorded in deterministic
priority order, so an inaccessible or invalid first candidate can fall through
to the next usable PDF or HTML source. DOI acquisition retains the exact
Crossref response hash and payload and, when Crossref supplies no full-text
link, safely inspects the landing page for additional candidates. Metadata is
merged with explicit input taking precedence over resolved and embedded PDF
metadata while conflicts remain visible in an immutable merge manifest.

The network transport pins every request to the exact public IP set validated
before connection, while preserving the original Host header and TLS SNI, and
repeats DNS validation after every redirect. Acquisition still creates no
curation decision or scientific claim, fact, result, relation, or expert
opinion.

K1C.2 makes `LiteratureRepresentationReference` the common PDF/HTML trust
contract. The generic selector binds only the chosen representation ID and
hash; its accepted curation then controls the current raw/canonical/ingestion/
source chain in trusted snapshots. Older PDF ingestion-based selections and
historical authorizations are resolved through an explicit compatibility path.
Unselected representations remain audit history, and evidence from either a
historical PDF or HTML source requires the same immutable authorization record.

DOI acquisition now attempts Crossref candidates first and inspects the DOI
landing page exactly once only when none succeeds. Newly discovered candidates
are deduplicated against prior attempts and continue the same deterministic
attempt index sequence. Final status distinguishes authentication-only,
unsupported-media, metadata-only, unavailable-full-text, and hard integrity or
parser failures.

## Knowledge Layer K1D

K1D adds bounded project/collection discovery without turning SPC into a
general crawler. An immutable scope policy fixes allowed origins and path
prefixes, page/depth/resource limits, pagination rules, and accepted media
types. The generic connector follows only explicit static-HTML pagination,
recognizes DOI, article, citation-metadata, and direct-PDF resources, and uses
the K1C safe HTTP fetcher for every request. Authentication walls, unresolved
pagination, traversal limits, cycles, and JavaScript-only continuation remain
explicit incomplete states rather than being reported as complete.

```powershell
spc import-literature-collection https://example.org/project/publications `
  --domain base `
  --knowledge-dir knowledge `
  --state-dir .spc `
  --max-pages 100 `
  --max-resources 1000 `
  --max-depth 10 `
  --allowed-path-prefix /project/publications
```

Page records, logical discovered resources, every discovery occurrence,
content-bound snapshots, K1C acquisition links, batch import records, and
snapshot diffs remain immutable audit inventory. Repeated DOI/URL occurrences
are deduplicated without losing page provenance. Each unique resource is handed
to K1C independently, so one unavailable paper does not abort the batch. K1D
does not select or curate representations, generate scientific records, or add
crawl internals to the scientific knowledge graph.

K1D.1 separates traversal termination from proven collection completeness. A
generic page with no recognized next link is
`policy_exhausted_unverified`; only an explicit finite pagination chain can be
`proven_complete` under the generic connector contract. Snapshots bind the
scope-policy hash, connector completeness contract, and completeness basis.
Path prefixes use URL-segment boundaries, and optional `--allowed-origin`,
`--allowed-path-prefix`, `--allow-external-literature-links`, and `--max-depth`
arguments change scope only when explicitly supplied.

Every successful collection page is stored as exact immutable bytes and bound
to its page record. After K1C resolution, resources that converge on the same
`literature_id` become one `CollectionLiteratureMembership` while retaining all
resource, occurrence, and acquisition references. Import output distinguishes
occurrences, discovered resources, logical literature, and ingested literature;
low-confidence DOI text from arbitrary page bodies stays auditable but is not
silently imported as collection membership.

K1D.2 makes membership classification local to each discovered text fragment
or link. Lightweight ancestor semantics identify publication/member contexts
and exclude references, bibliographies, related content, navigation, and
footers even when the same page also contains a publication list. Neutral DOI,
article, and PDF links remain immutable audit occurrences with an explicit
unverified decision; citation metadata and locally verified publication cards
remain eligible. If the same resource later appears in an eligible context,
the logical resource is promoted while every original occurrence is retained.

## Knowledge Layer K1E0

K1E0 separates persistent knowledge evidence from project-local evidence.
`KnowledgeEvidenceStore(knowledge_dir)` stores literature and future expert
source material under `knowledge/evidence_store/sources` and
`knowledge/evidence_store/evidence`. `ProjectEvidenceStore(state_dir)` retains
reviewer comments, manuscripts, author responses, and other project evidence
under `.spc`. `SourceEvidenceStore` remains the backward-compatible project
store name.

`CompositeEvidenceStore(knowledge_store, project_store)` provides the read-only
view used by retrieval and Phase 2 validation. It detects source or EvidenceSpan
identity collisions and fails closed; writes require choosing one concrete
store explicitly. Literature ingestion, acquisition, collection import, and
representation selection now use the shared knowledge store by default.

Legacy literature evidence can be copied without deleting project records:

```powershell
spc migrate-knowledge-evidence `
  --knowledge-dir knowledge `
  --state-dir .spc
```

Migration follows existing literature ingestion/representation references,
verifies source bytes and EvidenceSpans before and after copying, rejects
conflicts, and is idempotent.

## Knowledge Layer K1E

K1E adds immutable, representation-bound document structure without changing
the evidence authority chain. PDF and HTML extractors map pages, headings,
paragraphs, list items, table captions/cells, and figure captions back to exact
canonical UTF-8 offsets. Ambiguous PDF table cells are never reconstructed;
the table remains explicitly partial instead.

```powershell
spc structure-literature `
  --literature-id literature-... `
  --representation-id literature-representation-... `
  --knowledge-dir knowledge

spc inspect-literature-structure `
  --structure-id document-structure-... `
  --knowledge-dir knowledge `
  --section Results `
  --section "CO dissociation"
```

`StructuredEvidenceLocator` adds page, section, table-cell, and figure-caption
navigation to an exact `EvidenceSpan`; it never replaces the span or creates a
scientific claim. Structure and locator repositories live only under the
shared knowledge root. Trusted snapshots include structures for the current
accepted representation and locators for evidence that independently passes
the existing curation and historical-evidence rules.

### K1E.1 structure authority

HTML tables use a deterministic rowspan/colspan occupancy grid. A table cell
binds one or more exact canonical blocks; callers must select a region when a
multi-paragraph cell cannot be represented by one honest `EvidenceSpan`.
Empty cells retain topology without inventing text.

The built-in extractor may create or reuse an immutable
`DocumentStructureSelection` only when its result does not reduce extraction
quality or return to an earlier structure. Custom extractor output is stored as
an audit artifact and is never authoritative by default. Explicit promotion,
including an intentional rollback, uses:

```powershell
spc select-document-structure `
  --representation-id literature-representation-... `
  --structure-id document-structure-... `
  --rationale "Reviewed corrected structure" `
  --allow-rollback `
  --knowledge-dir knowledge
```

Each promotion appends an immutable selection event; old selections and
structures remain available for audit. Trusted snapshots bind exactly the
current selected structure and reject locators attached to an older structure.

### K1E.2 content regions

HTML structure blocks and `StructuredEvidenceLocator` records preserve a local
`content_region`: main content, references, navigation, footer, related content,
supplementary context, or unknown. Reference/navigation headings cannot alter
the main scientific section path, and locally negative regions override a broad
article container. PDF regions remain `unknown` until a reliable layout-aware
extractor exists. Integer-numbered PDF lines are treated conservatively as
unresolved text rather than inferred headings or list items.

## Knowledge Layer K1F0 backend adapters

K1F0 adds a vendor-neutral optional adapter boundary for mature open-source
engines while retaining all scientific authority inside SPC:

```text
SPC Core and trusted evidence contracts
                 ^
      exact rebinding + validation
                 ^
       Backend Adapter Layer
                 |
 Docling / PaperQA2 / GROBID / MinerU
 LightRAG / KAG / RAGFlow-compatible services
```

External engines provide capabilities. SPC retains scientific authority.
Their document, retrieval, or metadata output is always a content-bound
`external_proposal`; it cannot directly create an `EvidenceSpan`, scientific
claim, knowledge relation, or plan. Document text must be found unambiguously
in the current SPC canonical text, and retrieval snippets must additionally
resolve through the current accepted representation and selected structure.

The base installation has no Docling, PaperQA2, MinerU, GROBID, graph service,
or model dependency. Optional Python integrations are lazy-loaded through
`spc[docling]` or `spc[paperqa]`; subprocess/service adapters require explicit
local configuration. Supported optional dependency major versions are bounded
so an untested upstream API fails closed. No upstream source is vendored.
License declarations and redistribution-review state are recorded in
`backend_licenses.yaml`.

```powershell
spc backends
spc backends --docling-artifacts-path C:\models\docling
spc backend-info docling --docling-artifacts-path C:\models\docling
spc inspect-backend-run backend-run-... --knowledge-dir knowledge

spc structure-literature `
  --literature-id literature-... `
  --representation-id literature-representation-... `
  --backend builtin `
  --knowledge-dir knowledge

spc structure-literature `
  --literature-id literature-... `
  --representation-id literature-representation-... `
  --backend docling `
  --docling-artifacts-path C:\models\docling `
  --knowledge-dir knowledge
```

`--backend docling` requires both the compatible optional package and an
explicit local model-artifact directory. The adapter disables remote services,
plugins, and implicit model downloads. Its typed document items and table-cell
topology are normalized into an external proposal; proposed content regions
remain untrusted until SPC performs exact canonical-text rebinding. The result
is audit-only by default. `--promote-external` explicitly submits an exactly
rebound result to the existing deterministic, anti-downgrade structure
selection policy. For HTML, external structures remain audit-only and cannot
replace the built-in deterministic content-region authority; automatic external
promotion is limited to PDF until a deterministic HTML reconciliation contract
exists.

K1F0.1 persists the exact `BackendDescriptor` and content-bound
`BackendRuntimeIdentity` used for each invocation under the knowledge root. A
runtime identity records the installed package or service version, adapter
version, integration mode, provider identity, and non-secret configuration
hash. `BackendRunRecord` binds both immutable records and distinguishes external
retrieval candidate count from SPC-resolved and unresolved counts. Resolution
batches additionally bind the current accepted representation and selected
structure; stale authority is rejected. Availability, invocation, and
downstream normalization failures are stored as sanitized audit records without
API keys, credentials, raw third-party payloads, or local model paths.

K1F0.2 closes the remaining backend trust bindings. Resolved retrieval records
bind the exact `EvidenceSpan` and `StructuredEvidenceLocator` hashes and reopen
both records during batch validation. Successful or partial backend runs carry
an explicit output type and ID, so inspection deterministically reopens and
validates the exact document proposal, retrieval result, or metadata proposal.
Docling compatibility matches the declared `>=2.65,<3` range. Its runtime
identity binds deterministic installed-component versions plus a content-bound
manifest of local model files; symlinks, escaping paths, incomplete scans, and
changed model bytes fail closed or produce a different runtime identity. The
configured absolute model path is operational metadata, not scientific
identity.

## Knowledge Layer K1F scientific knowledge compiler

K1F compiles the accepted current literature representation and current selected
document structure into evidence-grounded scientific proposals. Deterministic
materialization reopens canonical text, locates each exact quote within its allowed
structure block, and binds `EvidenceSpan`, `StructuredEvidenceLocator`,
`SourceQuote`, scientific records, grounding records, and immutable compilation
provenance. Provider output supplies proposal-local keys only; it cannot assign SPC
record IDs or offsets, access tools, or mark knowledge trusted.

New scientific records are curated as `machine_extracted`. They appear in the audit
view but enter the trusted-current view only after explicit acceptance and only while
their representation and structure remain current. HTML extraction defaults to main
content; supplementary content is opt-in, and reference/navigation/footer/related or
unknown regions are excluded. PDF unknown-region chunks remain explicitly uncertain.

```powershell
spc extract-literature-knowledge --literature-id literature-... --knowledge-dir knowledge --provider mock
spc inspect-literature-knowledge --literature-id literature-... --knowledge-dir knowledge --view audit
spc curate-knowledge --target-type source_claim --target-id claim-... --status accepted `
  --curator-id reviewer --rationale "Reviewed exact source grounding." --knowledge-dir knowledge
```

K1F.1 applies one shared trusted-current authority rule to the literature view,
global trusted graph, snapshots, and downstream retrieval contexts. Accepted
results cannot pull unaccepted method/model facts into trusted knowledge. Table
ownership is resolved only through the selected structure's table and cell
inventories, while numeric grounding preserves sign and scientific notation.
Inspection records include the scientific statement, exact quotes, formatted
locators, curation state, uncertain regions, and deterministic proposal
rejections.

Run the bounded two-document pipeline demonstration without network access:

```powershell
python examples/k1f_offline_demo.py
```

The demo ingests two in-memory HTML fixtures, selects and curates source
authority, structures both documents, extracts mock proposals, shows the audit
view, explicitly accepts one claim per document, then shows trusted-current
records. It also verifies that identical table offsets in different papers bind
different table identities. This demonstrates plumbing and trust gates, not
scientific extraction accuracy.

Real structured-model extraction is opt-in and uses the same vendor-neutral HTTP
transport as planning and approval:

```powershell
$env:SPC_LLM_API_KEY = "..."
spc extract-literature-knowledge `
  --literature-id literature-... `
  --knowledge-dir knowledge `
  --provider llm `
  --llm-endpoint https://your-endpoint.example/structured `
  --llm-model your-model-id `
  --llm-api-key-env SPC_LLM_API_KEY
```

No model call occurs unless `--provider llm` and both endpoint/model options are
supplied. API-key values are never written into K1F provenance. All LLM outputs
remain strict, untrusted proposals and require explicit scientific curation.
