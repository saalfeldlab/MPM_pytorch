# Git Integration for Code Modification System

## Overview

The system now automatically version-controls all code modifications made by Claude during experimentation. Every code change is committed to git with descriptive messages extracted from Claude's reasoning logs.

## What Was Added

### 1. Git Tracking Module (`git_code_tracker.py`)

Complete git integration functionality:
- `is_git_repo()`: Check if directory is a git repository
- `get_modified_code_files()`: Detect which tracked files have changes
- `extract_code_modification_description()`: Parse description from logs
- `commit_code_modification()`: Create descriptive git commit
- `track_code_modifications()`: Main function - detects and commits changes
- `git_push()`: Optional push to remote (not auto-called)

### 2. Integration in `run_MPM.py`

After Claude's analysis (line ~425):
```python
# Git tracking: commit any code modifications made by Claude
if is_git_repo(root_dir):
    print(f"\n--- Checking for code modifications to commit ---")
    git_results = track_code_modifications(
        root_dir=root_dir,
        iteration=iteration,
        analysis_path=analysis_path,
        reasoning_path=reasoning_log_path
    )

    if git_results:
        for file_path, success, message in git_results:
            if success:
                print(f"✓ Git: {message}")
```

### 3. Updated Documentation

- **instruction_multimaterial_1_discs_3types.md**: Section on automatic git version control
- **CODE_MODIFICATION_SYSTEM.md**: Expanded safety mechanisms section
- **test_git_tracking.sh**: Test script for verification

## How It Works

### Automatic Workflow

```
┌─────────────────────────────────────────────────────────────┐
│ Iteration N                                                  │
├─────────────────────────────────────────────────────────────┤
│ 1. Training runs (subprocess)                                │
│ 2. Results saved to analysis.log                             │
│ 3. Claude analyzes & decides to modify code                  │
│ 4. Claude edits Siren_Network.py (adds LayerNorm)            │
│ 5. Claude logs: "CODE MODIFICATION: Added LayerNorm..."      │
│ 6. System saves reasoning to reasoning.log                   │
│                                                               │
│ ┌───────────────────────────────────────┐                   │
│ │ GIT TRACKING (NEW)                    │                   │
│ ├───────────────────────────────────────┤                   │
│ │ 1. Check if Siren_Network.py modified │                   │
│ │ 2. Extract description from logs      │                   │
│ │ 3. Stage file: git add                │                   │
│ │ 4. Create commit with:                │                   │
│ │    - Iter number                      │                   │
│ │    - Description                      │                   │
│ │    - Hypothesis                       │                   │
│ │    - File stats                       │                   │
│ │ 5. Print: "✓ Git: Committed as abc123"│                   │
│ └───────────────────────────────────────┘                   │
│                                                               │
│ 7. Save config snapshot                                      │
│ 8. Continue to next iteration                                │
└─────────────────────────────────────────────────────────────┘
```

### Commit Message Format

```
[Iter 12] Added CosineAnnealingLR scheduler to data_train_INR

File: src/MPM_pytorch/models/graph_trainer.py
Function: data_train_INR
Change: Added CosineAnnealingLR scheduler with T_max=total_steps
Hypothesis: Decaying learning rate may escape local minimum

Changes: 3 insertions(+), 0 deletions(-)

[Automated commit by Claude Code Modification System]
```

### Description Extraction

The system intelligently extracts descriptions from Claude's logs:

**From `reasoning.log`:**
```markdown
CODE MODIFICATION:
  File: src/MPM_pytorch/models/graph_trainer.py
  Function: data_train_INR
  Change: Added CosineAnnealingLR scheduler with T_max=total_steps
  Hypothesis: Decaying learning rate may escape local minimum
```

**From `analysis.md`:**
```markdown
## Iter 12: moderate
...
CODE MODIFICATION:
  ...
```

## Usage

### For Users

**No action required!** Git commits happen automatically.

```bash
# Run experiment normally
python run_MPM.py -o train_INR_Claude multimaterial_1_discs_3types iterations=100

# After running, view commit history
git log --oneline
git log --grep="Claude Code Modification"
```

### Viewing Modification History

```bash
# All code modification commits
git log --grep="Claude Code Modification" --oneline

# Full details of specific commit
git show abc123

# See what changed at iteration 15
git log --grep="Iter 15" -1 --stat

# View diff for a commit
git diff abc123^..abc123
```

### Rollback

**Option 1: Revert specific commit**
```bash
# Find the commit
git log --oneline | grep "Iter 15"

# Revert it (creates new commit undoing changes)
git revert abc123
```

**Option 2: Reset to before modification**
```bash
# Reset to commit before the change
git reset --hard abc123^

# WARNING: This discards the commit permanently
```

**Option 3: Create branch to test**
```bash
# Create branch without the modification
git checkout -b test-without-mod abc123^

# Test, then merge or discard
```

## Testing

### Quick Test

```bash
cd /workspace/MPM_pytorch

# Check if git integration works
./test_git_tracking.sh
```

### Manual Test

```bash
# 1. Make a small code change
echo "# Test comment" >> src/MPM_pytorch/models/Siren_Network.py

# 2. Run git tracker manually
python git_code_tracker.py /workspace/MPM_pytorch

# 3. Check result
git log -1 --stat
```

### Full Integration Test

```bash
# Run one iteration with code modification
python run_MPM.py -o train_INR_Claude multimaterial_1_discs_3types iterations=1

# Check if git commit was created
git log -1
```

## Configuration

### Enable/Disable Git Tracking

Currently: Always enabled if in a git repository

To disable (if needed):
```python
# In run_MPM.py, comment out the git tracking section:
# if is_git_repo(root_dir):
#     ... git tracking code ...
```

### Push to Remote

By default: **No automatic push** (commits stay local)

To enable auto-push after each commit, add to `run_MPM.py`:
```python
if git_results:
    # After committing, optionally push
    success, message = git_push(root_dir)
    if success:
        print(f"✓ Git: {message}")
```

### Push at Block Boundaries

Better approach - push only at block ends:
```python
if is_block_end:
    success, message = git_push(root_dir)
    if success:
        print(f"✓ Git: Pushed block {block_number} to remote")
```

## Benefits

1. **Full History**: Every code modification tracked
2. **Easy Rollback**: `git revert` any change
3. **Reproducibility**: Know exactly what code was used for each iteration
4. **Analysis**: Compare modifications across blocks
5. **Safety**: Never lose working code
6. **Collaboration**: Team can review modification history

## File Structure

```
MPM_pytorch/
├── git_code_tracker.py           # Git integration module (NEW)
├── test_git_tracking.sh          # Test script (NEW)
├── run_MPM.py                    # Modified to include git tracking
├── train_INR_subprocess.py       # Subprocess training script
├── instruction_*.md              # Updated with git info
├── CODE_MODIFICATION_SYSTEM.md   # Updated documentation
└── .git/                         # Git repository
    └── logs/
        └── HEAD                  # Contains commit history
```

## Troubleshooting

### "Not a git repository"

**Problem**: Warning message on iteration 1

**Solution**: Initialize git repo
```bash
cd /workspace/MPM_pytorch
git init
git add .
git commit -m "Initial commit"
```

### No commits being created

**Possible causes:**
1. No code modifications made by Claude
2. Files not tracked by git
3. Git command timeout

**Debug:**
```bash
# Check if files are tracked
git status

# Manually test git tracker
python git_code_tracker.py /workspace/MPM_pytorch

# Check git is working
git log
```

### Commit messages incomplete

**Issue**: Description not extracted from logs

**Cause**: LOG format doesn't match expected pattern

**Fix**: Ensure Claude logs use exact format:
```
CODE MODIFICATION:
  File: ...
  Function: ...
  Change: ...
  Hypothesis: ...
```

## Future Enhancements

Potential additions:
1. **Branching**: Create branch per experiment block
2. **Tags**: Tag successful configurations
3. **Diff analysis**: Automatic comparison between iterations
4. **Metrics in commits**: Include R² in commit message
5. **Remote sync**: Configurable auto-push strategy
6. **Rollback automation**: Auto-revert on training failure

## Summary

✅ **Automatic git commits** for all code modifications
✅ **Descriptive commit messages** extracted from Claude's logs
✅ **Easy rollback** via standard git commands
✅ **No manual intervention** required
✅ **Full version history** preserved
✅ **Safe by default** (local commits only, no auto-push)

The system is now **production-ready** with full version control!
