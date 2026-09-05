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
