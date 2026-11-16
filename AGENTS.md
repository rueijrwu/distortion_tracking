# Project rules (coordinator workflow)

## Roles
- The main Claude session is the **coordinator** and only coordinates: it delegates, tracks status and reports. It does not plan, implement or audit itself.
- **Planner and auditor:** gpt-6-astra, high effort.
- **Implementation:** gpt-6.1-sol, xhigh effort. All implementation must follow the current plan.
- **Audit:** every implementation must be audited by the auditor (gpt-6-astra high) before it is accepted.
- **Mechanical work** (reads, runs, checks): gpt-6-luna, low effort.
- **Sandbox failure fallback:** if a task fails because of the sandbox, use Haiku 5.5 (build/run only).

## Plans
- Do not create temporary or separate plan files. All plans must be consolidated into and updated in `CURRENT_PLAN.md`.
- Project cleanup follows `CURRENT_PLAN.md` Section 18; Section 17 remains the scientific/documentation handoff. MOVE and DELETE are complete, and the cleanup goal is closed. The harmless Windows-permission leftovers `.codex\AGENTS.md` (move denied) and empty `.tmp` (delete denied) are intentionally retained for optional user manual removal. See Section 18 for current status, static-only verification and provenance limits.

## P4 handoff (2026-11-16)
- The current model-selection goal is closed by user acceptance: FL is the default; F/quadratic is the fallback or future option. Approximately 1 arcmin FL gaze error is accepted.
- Current theory is `P4_MODEL.md`; procedures, constants and recorded evidence are in `P4_IMPLEMENTATION.md`; the documentation contract is `CURRENT_PLAN.md` Section 17. Section 16 is the historical experiment contract.
- Preserve the original strict result `L0=True`, `L1=True`, `L2=False`, `CLOSE: F`. User acceptance does not rewrite that verdict or relax numerical gates.
- The obsolete root `CURRENT_MODEL.md`, `Theory.md` and `P4_SPEC.md` are not current sources or prerequisites and must not be recreated for this handoff. Candidate validation/robustness work belongs to a separate new goal.

## Scripts and outputs
- Every script in `Script\` writes its output only to one folder in `runs\` with a shorter, meaningful name matching the script.
- Documents must have no path dependency on the `runs\` folder.
- Most experiment/script results can be written to `Record.md` as record/documentation.

## Python
- Invoke `C:\Users\rueijrwu\.pyenv-win-venv\envs\venv\Scripts\python.exe` by absolute path (see `AGENTS.md`).
