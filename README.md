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

The resulting `ScientificContextPacket` binds the query hash, Domain Pack version, deterministic knowledge snapshot ID, record hashes, evidence source versions, ordered result IDs, and its own content hash. Retrieval remains planning-only and never invokes an LLM or scientific execution backend.

## Phase 2B interpretation

Phase 2B converts a `ScientificContextPacket` into a hash-bound `ScientificEvidencePacket`. The initial `MockInterpretationProvider` is deterministic and offline. Phase 2B.1 separates exact `SourceQuote` text from normalized or paraphrased `SourceClaim` text. Every quote must equal its `EvidenceSpan` exactly and carries the corresponding `SourceDocument` role and type; a claim remains bound through explicit quote and evidence references rather than substring matching. Retrieved statements are never promoted to established facts.

```powershell
spc interpret context.yaml `
  --provider mock `
  --state-dir .spc `
  --output evidence-packet.yaml
```

Interpretation validation reopens every referenced `EvidenceSpan`, verifies its source file and Phase 2A snapshot hash, and rejects unretrieved, fabricated, or inexact quotes. Numerical results retain units and a `ResultContext` that references applicable `MethodFact` and `ModelFact` records. Quantity extraction uses each value's local context so a temperature, pressure, and barrier in one sentence remain distinct. Facet or method mismatches produce explicit comparison constraints, source conflicts remain unresolved, and predictive models are not relabeled as DFT results. Phase 2B does not generate a `ScientificQuestionPlan` and does not execute scientific software.
