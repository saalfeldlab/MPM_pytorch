# Code Modification System for LLM-Guided Experimentation

## Overview

This system enables Claude to modify Python code between iterations while maintaining safety and reproducibility. Code modifications are automatically reloaded via subprocess execution.

## Architecture

### Files Structure

```
MPM_pytorch/
├── run_MPM.py                          # Main experiment loop
├── train_INR_subprocess.py             # Subprocess training script (NEW)
├── instruction_multimaterial_*.md      # Instructions with Step 5.2 for code mods
└── src/MPM_pytorch/models/
    ├── Siren_Network.py               # Network architecture (MODIFIABLE)
    └── graph_trainer.py               # Training loop (MODIFIABLE)
```

### How It Works

**Before (iterations 1-N):**
```
run_MPM.py imports modules once at startup
├─ Iteration 1: train_INR() called directly
├─ Iteration 2: train_INR() called directly (STALE CODE - no reload)
└─ Iteration N: train_INR() called directly (STALE CODE - no reload)
```

**After (with subprocess):**
```
run_MPM.py spawns subprocess for each iteration
├─ Iteration 1: subprocess → reloads modules → train_INR()
├─ Iteration 2: subprocess → reloads modules → train_INR() (FRESH CODE)
└─ Iteration N: subprocess → reloads modules → train_INR() (FRESH CODE)
```

### Code Flow

```python
# run_MPM.py (main loop)
for iteration in iteration_range:

    # Claude analyzes results, modifies code/config
    # ...

    # Training runs in subprocess (reloads modified code)
    if 'Claude' in task:
        subprocess.Popen([
            sys.executable,
            'train_INR_subprocess.py',
            '--config', config_path,
            '--field_name', field_name,
            '--device', device,
            '--log_file', analysis_log_path
        ])

    # Continue with UCB computation, Claude analysis, etc.
```

## Usage

### For Users

Run experiments normally:
```bash
python run_MPM.py -o train_INR_Claude multimaterial_1_discs_3types iterations=100
```

The system automatically:
1. Runs training in subprocess for each iteration
2. Reloads modified code
3. Captures errors if code modifications break

### For Claude (LLM)

**Step 5.2 in instructions provides:**
- When to modify code (only after config exhausted)
- What can be modified (Siren_Network.py, graph_trainer.py)
- Safety rules (minimal changes, document thoroughly)
- Example modifications (LR schedulers, loss functions, normalization)
- Logging format for code changes

**Example Claude workflow:**

1. **Iterations 1-4**: Config-only mutations
2. **Iteration 5**: All configs fail (R² < 0.75)
3. **Iteration 6**: Claude decides to modify code
   ```python
   # Claude edits graph_trainer.py
   # Adds: scheduler = CosineAnnealingLR(optim, T_max=total_steps)
   ```
4. **Iteration 6 runs**: Subprocess picks up modified code
5. **Results**: Claude compares to parent (was code change effective?)

## Safety Mechanisms

### 1. Subprocess Isolation
- Code errors don't crash main loop
- Returns exit code if training fails
- Error messages captured and displayed

### 2. Automatic Git Version Control
- After each iteration, modified code files are automatically committed
- Commit messages include: iteration number, description, hypothesis, file changes
- Easy rollback via `git revert <commit-hash>`
- Full modification history in git log
- Example commit message:
  ```
  [Iter 12] Added CosineAnnealingLR scheduler in data_train_INR (graph_trainer.py)
  Hypothesis: Decaying learning rate may escape local minimum

  File: src/MPM_pytorch/models/graph_trainer.py
  Changes: +3 lines, -0 lines

  [Automated commit by Claude Code Modification System]
  ```

### 3. Automatic Error Detection
```python
if process.returncode != 0:
    print("Training subprocess failed")
    print("This may indicate a code modification error")
    raise RuntimeError(f"Training failed at iteration {iteration}")
```

### 3. Rollback Procedure
If code change causes crash:
1. System reports error to Claude
2. Claude documents failure in log
3. Human manually reverts code change
4. Experiment continues with config-only mutations

### 4. Versioning
- Config snapshots: `log/Claude_exploration/{instruction}/configs/iter_{N:03d}_config.yaml`
- Code modifications logged in: `{config}_analysis.md` and `{config}_memory.md`
- Reasoning logged in: `{config}_reasoning.log`

## Examples of Allowed Modifications

### Network Architecture (Siren_Network.py)

**Add LayerNorm:**
```python
# In Siren class __init__, modify the loop:
for i in range(hidden_layers):
    self.net.append(SineLayer(hidden_features, hidden_features, ...))
    self.net.append(nn.LayerNorm(hidden_features))  # ADD
```

**Add Skip Connection:**
```python
# Create residual wrapper
class SineLayerWithSkip(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.layer = SineLayer(features, features, ...)
    def forward(self, x):
        return x + self.layer(x)  # residual
```

### Training Loop (graph_trainer.py)

**Add LR Scheduler:**
```python
# After optimizer creation (line ~600):
optim = torch.optim.Adam(lr=learning_rate, params=nnr_f.parameters())
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=total_steps)

# In training loop, after optim.step():
scheduler.step()
```

**Change Loss Function:**
```python
# Replace MSE with Huber loss:
# OLD: loss = ((model_output - ground_truth_batch) ** 2).mean()
# NEW:
loss = torch.nn.functional.huber_loss(model_output, ground_truth_batch, delta=0.1)
```

**Add Gradient Clipping:**
```python
# In training loop, after loss.backward():
loss.backward()
torch.nn.utils.clip_grad_norm_(nnr_f.parameters(), max_norm=1.0)  # ADD
optim.step()
```

## Testing

### Test subprocess script directly:
```bash
cd /workspace/MPM_pytorch
python train_INR_subprocess.py \
    --config config/multimaterial_1_discs_3types_Claude.yaml \
    --field_name Jp \
    --device cuda:0 \
    --log_file test_output.log
```

### Test with modified code:
1. Make a small change to `src/MPM_pytorch/models/graph_trainer.py`
2. Run subprocess script
3. Verify change is picked up (add print statement to confirm)

## Troubleshooting

### Import Errors
**Problem**: `ModuleNotFoundError: No module named 'torch'`

**Solution**: Subprocess uses wrong Python interpreter
- Check: `which python` should point to conda environment
- Fix: `sys.executable` in run_MPM.py uses current interpreter

### Code Modifications Not Applied
**Problem**: Changes to code don't affect training

**Solution**:
- Verify you're in Claude task mode (`train_INR_Claude`)
- Check subprocess is being called (look for "Running INR training in subprocess" message)
- Verify you modified the correct file path (`src/MPM_pytorch/models/...`)

### Subprocess Crashes
**Problem**: Training fails with non-zero exit code

**Solution**:
- Check error message in terminal output
- Look for syntax errors, indentation issues
- Verify imports are correct
- Test modified code in isolation before running full experiment

## Benefits of This Approach

1. **True Code Reload**: Each iteration gets fresh modules
2. **Safety**: Main loop doesn't crash from code errors
3. **Flexibility**: Claude can modify network architecture and training
4. **Reproducibility**: All changes logged and versioned
5. **Scientific Method**: Code changes as testable hypotheses

## Comparison to Alternatives

| Approach | Code Reload? | Safety | Flexibility |
|----------|-------------|--------|-------------|
| Direct import | ❌ No | ⚠️ Crashes main loop | ❌ Config only |
| importlib.reload() | ⚠️ Partial | ⚠️ Crashes main loop | ✅ Full |
| **Subprocess (ours)** | ✅ Full | ✅ Isolated | ✅ Full |

## Future Enhancements

### Potential Additions:
1. **Code versioning**: Git commit for each code modification
2. **Diff tracking**: Automatic diff generation for modified files
3. **Rollback automation**: Auto-revert if subprocess fails
4. **A/B testing**: Run both old and new code, compare results
5. **Code templates**: Pre-defined modification patterns

### Metrics to Track:
- Code modification success rate
- Average R² improvement from code vs config changes
- Most effective code modifications (frequency × impact)
- Code modification time cost vs benefit

## References

- Main paper: `paper/Understanding_an_experiment_LLM_memory_solution.pdf`
- Instructions: `instruction_multimaterial_1_discs_3types.md` (Step 5.2)
- Example runs: `log/Claude_exploration/instruction_*/`
