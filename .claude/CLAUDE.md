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

## Scripts and outputs
- Every script in `Script\` writes its output only to one folder in `runs\` with a shorter, meaningful name matching the script.
- Documents must have no path dependency on the `runs\` folder.
- Most experiment/script results can be written to `Record.md` as record/documentation.

## Python
- Invoke `C:\Users\rueijrwu\.pyenv-win-venv\envs\venv\Scripts\python.exe` by absolute path (see `AGENTS.md`).
