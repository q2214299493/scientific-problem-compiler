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
