# Git Commit Convention

Git commit message convention for this project.

## PR Commit Rule

**Each PR must contain exactly one commit.** Squash all commits into one before submitting.

### How to Squash

```bash
# Squash the last 4 commits
git reset --soft HEAD~4
git commit -m "commit message"

# Or use interactive rebase
git rebase -i HEAD~4
# Change all but the first 'pick' to 'squash' or 's'
```

### Why Squash

- Keeps the main branch history clean, with each PR mapping to one complete feature
- The commit message auto-fills into the PR description
- Makes code review and issue tracing easier

## Commit Format

```
<type>: <subject>

<body>

Co-authored-by: <name> <email>
```

## Type

| Type | Description | Example |
|------|-------------|---------|
| `Add` | Add new feature or file | `Add: GitHub CI workflow for simulation example` |
| `Fix` | Fix a bug or issue | `Fix: Ensure timeout detection executes when all cores busy` |
| `Refactor` | Refactor code (no behavior change) | `Refactor: Extract shared PTO Runtime C API to common header` |
| `Rewrite` | Rewrite an implementation | `Rewrite kernels to use PTO tile-based operations` |
| `Rename` | Rename files/functions/variables | `Rename compile_kernel to compile_incore` |
| `Reorganize` | Reorganize project structure | `Reorganize project structure and update documentation` |
| `Migrate` | Migrate code or tech stack | `Migrate AICore kernel compilation from C++ to Python` |
| `Simplify` | Simplify code or API | `Simplify AicpuExecutor API and unify naming conventions` |
| `Support` | Support new platform/feature | `support extern func define in aicore` |

## Subject

- Written in English, capitalize the first letter
- Concise description of the change (50 characters or less)
- No trailing period
- Use imperative mood (Add, Fix, Update — not Added, Fixed, Updated)

## Body

- Separated from the subject by a blank line
- Explain **why** the change was made, not just **what** changed
- Use `-` to list specific changes
- May include the following sections:
  - **Problem/Root Cause**: Description of the issue and its root cause
  - **Solution**: How the issue was resolved
  - **Changes**: List of specific changes
  - **Impact**: Scope of impact

## Examples

### Simple Commit

```
Fix: Add task_status check to prevent duplicate task execution

Add task_status verification before executing tasks in aicore_executor.
This check was lost during a previous rebase and could cause tasks to
be executed multiple times.
```

### Detailed Commit

```
Fix: Ensure timeout detection executes when all cores busy

Problem:
- When cur_thread_tasks_in_flight >= core_num, continue statement skipped Phase 2
- Timeout mechanism completely failed, system hung with no diagnostic output

Root Cause:
- continue jumps to next iteration, bypassing all code after it

Solution:
- Change: if (busy) { continue } Phase2 Timeout
- To:     if (!busy) { Phase2 } Timeout (always runs)

Impact:
- Timeout can now detect hung tasks even when cores fully loaded
```

### Feature Addition

```
Add automated test framework with simulation platform support

- Add CodeRunner class for automated PTO runtime testing
  - Handles kernel compilation, runtime initialization, and result validation
  - Supports both a2a3 (hardware) and a2a3sim (simulation) platforms

- Add CLI interface (run_example.py) for running tests

- Add golden test examples for host_build_graph and host_build_graph_sim

- Reorganize scripts into examples/scripts/ directory

- Add comprehensive documentation in scripts/README.md
```

## PR Number

Merged PRs automatically append the PR number to the subject:

```
Fix: Add macOS compatibility for a2a3sim platform (#18)
```

## Co-authored-by

Add when collaborating with others:

```
Co-authored-by: Name <email@example.com>
Co-authored-by: Claude Opus 4.5 <noreply@anthropic.com>
```
