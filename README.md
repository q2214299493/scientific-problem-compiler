# Scientific Problem Compiler (SPC)

SPC compiles vague scientific requests, reviewer comments, and source evidence into one to four evidence-grounded, falsifiable scientific question plans. Phase 1 is offline and planning-only: it never calls an external LLM or executes scientific software.

The compiler, independent approver, and exporter have separate data boundaries. A plan can be exported only after a hash-bound plan approval, a passed plan gate, and an explicit human selection. Every exported task is forced to `runnable: false`.

Before every export, SPC reloads the selected domain pack, checks its version, reruns `validate_question_plan` against the on-disk `EvidenceSpan` repository, and verifies a `PlanValidationRecord` bound to the plan ID, version, and content hash. `GateVerdict` is additionally hash-bound to both that validation record and the independent `ApprovalVerdict`. Conditional approvals cannot export until every blocking fix has a parsed, resolved `FixResolution`.

## Quick start

```powershell
python -m pip install -e ".[dev]"
spc --help
pytest
```

`spc validate PLAN --state-dir .spc --record-output validation.yaml` creates the validation record required by `spc export --validation-record validation.yaml`. Export packages are fully checksummed and validated in a same-filesystem staging directory before one atomic rename into their final path.

The package ships `base` and `fischer_tropsch` domain packs. Domain-specific terminology and capabilities live in those packs; the core models contain no Fischer–Tropsch-specific fields.

## State and safety

Project planning state is stored under `.spc/`. Source content is copied into versioned, hash-addressed read-only records. Export packages are immutable by convention: SPC refuses to overwrite an existing export directory and writes a checksum manifest.

Phase 1 explicitly rejects runnable tasks and command-bearing execution policies. It does not generate VASP, NEB, Dimer, MKM, KMC, HPC, or other execution inputs.
