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
