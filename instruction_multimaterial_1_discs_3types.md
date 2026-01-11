# MPM INR Training Landscape Study

## Goal

Map the **MPM INR training landscape**: understand which INR architectures and training configurations achieve best field reconstruction (R² > 0.95) for Material Point Method simulations. Other important evaluation are slope (~1.0) and training time. Get an intuition for the training_time. 

## Iteration Loop Structure

Each block = `n_iter_block` iterations exploring one INR configuration.
The prompt provides: `Block info: block {block_number}, iteration {iter_in_block}/{n_iter_block} within block`

## File Structure (CRITICAL)

You maintain TWO files:

### 1. Full Log (append-only record)

**File**: `{config}_analysis.md`

- Append every iteration's full log entry
- Append block summaries
- **Never read this file** — it's for human record only

### 2. Working Memory (active knowledge)

**File**: `{config}_memory.md`

- **READ at start of each iteration**
- **UPDATE at end of each iteration**
- Contains: knowledge base + previous block + current block only
- Fixed size (~500 lines max)

---

## Iteration Workflow (Steps 1-4, every iteration)

### Step 1: Read Working Memory

Read `{config}_memory.md` to recall:

- Established principles
- Previous block findings
- Current block progress

### Step 2: Analyze Current Results

**Metrics from `analysis.log`:**

- `final_mse`: Mean squared error on full dataset
- `final_r2`: R² score (correlation coefficient squared)
- `total_params`: Number of INR model parameters
- `training_time`: Wall clock time for training
- `field_name`: Which MPM field was trained (F, Jp, S, C)
- `inr_type`: Architecture type (siren_t, siren_txy, ngp)

**Classification:**

- **Excellent**: R² > 0.95
- **Good**: R² 0.90-0.95
- **Moderate**: R² 0.75-0.90
- **Poor**: R² < 0.75

**UCB scores from `ucb_scores.txt`:**

- Provides pre-computed UCB scores for all nodes including current iteration
- At block boundaries, the UCB file will be empty (erased). When empty, use `parent=root`

Example:

```
Node 2: UCB=2.175, parent=1, visits=1, R2=0.997
Node 1: UCB=2.110, parent=root, visits=2, R2=0.934
```

### Step 3: Write Outputs

Append to Full Log (`{config}_analysis.md`) and **Current Block** sections of `{config}_memory.md`:

- In memory.md: Insert iteration log in "Iterations This Block" section (BEFORE "Emerging Observations")
- Update "Emerging Observations" at the END of the file

**Log Form:**

```
## Iter N: [excellent/good/moderate/poor]
Node: id=N, parent=P
Mode/Strategy: [success-exploit/failure-probe]/[exploit/explore/boundary]
Config: lr_NNR_f=X, total_steps=Y, hidden_dim_nnr_f=H, n_layers_nnr_f=L, omega_f=W, batch_size=B
Metrics: final_r2=A, final_mse=B, total_params=C, compression_ratio=D, training_time=E
Field: field_name=F, inr_type=T
Mutation: [param]: [old] -> [new]
Parent rule: [one line]
Observation: [one line]
Next: parent=P
```


### Step 4: Select parent node in UCB tree

- Read `ucb_scores.txt`
- If empty → `parent=root`
- Otherwise → select node with **highest UCB**

Step B: Choose strategy

| Condition                            | Strategy            | Action                             |
| ------------------------------------ | ------------------- | ---------------------------------- |
| Default                              | **exploit**         | Highest UCB node, try mutation     |
| 3+ consecutive R² ≥ 0.95             | **failure-probe**   | Extreme parameter to find boundary |
| n_iter_block/4 consecutive successes | **explore**         | Select outside recent chain        |
| Good config found                    | **robustness-test** | Re-run same config                 |
| 2+ distant nodes with R² > 0.95      | **recombine**       | Merge params from both nodes       |
| 100% convergence, branching<10%      | **forced-branch**   | Select node in bottom 50% of tree  |
| Same param mutated 4+ times          | **switch-param**    | Mutate different parameter         |
| All R² > 0.94, branching <20%        | **random-branch**   | Select random unvisited parent     |

**Recombination details:**

Trigger: exists Node A and Node B where:
- Both R² > 0.95
- Not parent-child (distance ≥ 2 in tree)
- Different parameter strengths

Action:
- parent = higher R² node
- Mutation = adopt best param from other node

### Step 5: Edit Config File (default) or Modify Code (rare)

Choose ONE:
- **Step 5.1 (DEFAULT)**: Edit config file parameters only
- **Step 5.2 (RARE)**: Modify Python code (only when config insufficient)


## Step 5.1: Edit Config File (default approach)

Edit config file for next iteration according to Parent Selection Rule.
(The config path is provided in the prompt as "Current config")

**CRITICAL: Config Parameter Constraints**

**DO NOT add new parameters to the `claude:` section.** Only these fields are allowed:
- `field_name`: Jp, F, S, C
- `ucb_c`: float value (0.5-3.0)

Any other parameters (like `n_epochs`, `data_augmentation_loop`, `total_steps`, `n_iter_block`, etc.) belong in the `training:` or `graph_model:` sections, NOT in `claude:`.

Adding invalid parameters to `claude:` will cause a validation error and crash the experiment.

**Training Parameters (change within block):**

Mutate ONE parameter at a time for better causal understanding.
Does not apply to total_steps, as needed to constrain total training_time ~10 min

```yaml
training:
  learning_rate_NNR_f: 1.0E-5 # range: 1E-7 to 1E-3
  batch_size: 8 # values: 4, 8, 16, 32, never larger than 32
  total_steps: 50000 # range: 5000-200000 (SCALE inversely with hidden_dim for training_time ~10 min)
  n_training_frames: 48 # progression: 48 → 100 → 200 → 500 → 1000 → 2000 → 5000 → 10000
graph_model:
  hidden_dim_nnr_f: 512 # values: 128, 256, 512, 1024, 2048
  n_layers_nnr_f: 3 # range: 2-6
  omega_f: 30.0 # range: 1.0 to 100.0 (SIREN frequency)
  omega_f_learning: False # or True
  learning_rate_omega_f: 1.0E-6 # if omega_f_learning=True
  nnr_f_xy_period: 1.0 # spatial normalization for siren_txy
  # NGP-specific (if inr_type: ngp)
  # ngp_n_levels: 16 # range: 8-24
  # ngp_log2_hashmap_size: 19 # range: 15-22
claude:
  field_name: Jp # values: Jp, F, S, C (change between blocks)
  ucb_c: 1.414 # UCB exploration constant (0.5-3.0), adjust between blocks
```
## Step 5.2: Modify code (OPTIONAL - use sparingly)

**When to modify code:**
- When config-level parameters are insufficient OR when a failure mode indicates a fundamental limitation
- When you have a specific architectural hypothesis to test
- When 3+ iterations suggest a code-level change would help
- NEVER modify code in first 4 iterations of a block

**Files you can modify:**
1. `src/MPM_pytorch/models/Siren_Network.py` - Network architecture
2. `src/MPM_pytorch/models/graph_trainer.py` - Training loop (data_train_INR function)

**How code reloading works:**
- Training runs in a subprocess (`train_INR_subprocess.py`) for each iteration
- This subprocess reloads all Python modules, picking up any code modifications
- Code changes are immediately effective in the next iteration
- If the subprocess crashes due to syntax errors, the iteration fails and you'll see the error

**Automatic git version control:**
- After each iteration, the system checks if code files were modified
- Modified files are automatically committed to git with descriptive messages
- Commit messages include: iteration number, description extracted from logs, hypothesis
- This provides full version history and easy rollback capability
- If not in a git repository, modifications are still logged but not version-controlled

**Safety rules (CRITICAL):**
1. **Make minimal changes** - edit only what's necessary
2. **Test in isolation first** - don't combine code + config changes
3. **Document thoroughly** - explain WHY in mutation log
4. **One change at a time** - never modify multiple functions simultaneously
5. **Preserve interfaces** - don't change function signatures
6. **Add fallback** - wrap risky code in try/except if possible

**Example modifications allowed:**

### A. Network Architecture Changes (Siren_Network.py)

**Allowed modifications:**
- Add normalization layers (LayerNorm, BatchNorm) between SIREN layers
- Add skip connections / residual connections
- Modify initialization scheme (currently uses SIREN-specific init)
- Add dropout for regularization
- Change activation in final layer (currently linear or sin)

**Example: Add LayerNorm after each hidden layer**
```python
# In Siren class __init__, after each SineLayer:
self.net.append(SineLayer(hidden_features, hidden_features, ...))
self.net.append(nn.LayerNorm(hidden_features))  # ADD THIS
```

**Template for logging:**
```
Mutation: [code] Siren_Network.py: Added LayerNorm after each hidden layer
Hypothesis: Normalization may stabilize training for high omega_f
```

### B. Training Loop Changes (graph_trainer.py, data_train_INR function)

**Allowed modifications:**
- Change optimizer (Adam → AdamW, SGD, RMSprop)
- Add learning rate scheduler (CosineAnnealingLR, ReduceLROnPlateau)
- Add gradient clipping
- Modify loss function (add regularization terms, use different distance metrics)
- Change data sampling strategy (currently random batch sampling)
- Add early stopping logic

**Example: Add learning rate schedule**
```python
# After optimizer creation (around line 600):
optim = torch.optim.Adam(lr=learning_rate, params=nnr_f.parameters())
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=total_steps)  # ADD THIS

# In training loop (after optim.step()):
scheduler.step()  # ADD THIS
```

**Example: Add gradient clipping**
```python
# In training loop, after loss.backward():
loss.backward()
torch.nn.utils.clip_grad_norm_(nnr_f.parameters(), max_norm=1.0)  # ADD THIS
optim.step()
```

**Example: Change loss function**
```python
# Replace MSE with Huber loss (more robust to outliers):
# OLD: loss = ((model_output - ground_truth_batch) ** 2).mean()
# NEW:
loss = torch.nn.functional.huber_loss(model_output, ground_truth_batch, delta=0.1)
```

### C. Versioning and Rollback

**Automatic version control:**
1. Config snapshots: `log/Claude_exploration/{instruction}/configs/iter_{N:03d}_config.yaml`
2. Code modifications: Automatically committed to git after each iteration
3. Git commit format:
   ```
   [Iter N] Description from your log

   File: path/to/file.py
   Changes: +X lines, -Y lines
   Hypothesis: Your stated hypothesis

   [Automated commit by Claude Code Modification System]
   ```

**After modifying code:**
1. State clearly in mutation log: `"CODE MODIFIED: {file}:{function} - {one-line description}"`
2. Include hypothesis in the CODE MODIFICATION section (automatically extracted for git commit)
3. In memory.md "Emerging Observations", track: "Code mod iter X: {result}"
4. System will automatically commit to git with extracted description

**Rollback procedures:**

**Option A: Git revert (recommended)**
```bash
# Human can revert the specific commit
git log --oneline  # Find commit hash
git revert <commit-hash>
```

**Option B: Manual revert**
- If code change causes crash/error, state in observation: "Code modification failed, reverting"
- Human will manually revert the code change via git
- Continue with config-only mutations

**Viewing modification history:**
```bash
git log --grep="Claude Code Modification" --oneline
git show <commit-hash>  # See full diff
```

### D. Code Modification Decision Tree

```
Is R² consistently < 0.75 for 4+ iterations with good configs?
├─ NO → Use config mutations only (Step 5.1)
└─ YES → Consider code modification
           ├─ Is the issue architectural? (network capacity, representation)
           │  └─ YES → Modify Siren_Network.py
           └─ Is the issue optimization? (convergence, stability)
              └─ YES → Modify data_train_INR in graph_trainer.py
```

### E. Specific Hypotheses Worth Testing via Code

**Network architecture:**
- "Hypothesis: SIREN's periodic activation causes overfitting → test with ReLU in hidden layers"
- "Hypothesis: Deeper networks need residual connections → add skip connections"
- "Hypothesis: LayerNorm improves training stability for field S"

**Optimization:**
- "Hypothesis: Adam's momentum hurts sparse gradient signals → test SGD"
- "Hypothesis: Learning rate needs warmup → add linear warmup schedule"
- "Hypothesis: Loss landscape is non-convex → try curriculum learning (easy frames first)"

**Loss function:**
- "Hypothesis: MSE loss overweights outliers in field F → try Huber loss"
- "Hypothesis: Need physics-informed loss → add gradient penalty on spatial smoothness"

### F. Logging Code Modifications

**In iteration log, use this format:**
```
## Iter N: [excellent/good/moderate/poor]
Node: id=N, parent=P
Mode/Strategy: code-modification
Config: [unchanged from parent, or specify if also changed]
CODE MODIFICATION:
  File: src/MPM_pytorch/models/graph_trainer.py
  Function: data_train_INR
  Change: Added CosineAnnealingLR scheduler with T_max=total_steps
  Hypothesis: Decaying learning rate may escape local minimum
Metrics: final_r2=A, final_mse=B, total_params=C, compression_ratio=D, training_time=E
Field: field_name=F, inr_type=T
Mutation: [code] data_train_INR: Added LR scheduler
Parent rule: [one line]
Observation: [compare to parent - did code change help?]
Next: parent=P
```

### G. Constraints and Prohibitions

**NEVER:**
- Modify run_MPM.py (breaks the experiment loop)
- Change function signatures (breaks compatibility)
- Add dependencies requiring new pip packages (unless absolutely necessary)
- Make multiple simultaneous code changes (can't isolate causality)
- Modify code just to "try something" without hypothesis

**ALWAYS:**
- Explain the hypothesis motivating the code change
- Compare directly to parent iteration (same config, code-only diff)
- Document exactly what changed (file, line numbers, what was added/removed)
- Consider config-based solutions first

---

## Block Workflow (Steps 1-3, every end of block)

Triggered when `iter_in_block == n_iter_block`

### STEP 1: COMPULSORY — Edit Instructions (this file)

You MUST use the Edit tool to add/modify rules in this file.
Do NOT just write recommendations in the analysis log — actually edit the file.

After editing, state in analysis log: `"INSTRUCTIONS EDITED: added rule [X]"` or `"INSTRUCTIONS EDITED: modified [Y]"`

**Evaluate and modify rules based on:**

**Branching rate:**

- Branching rate < 20% → ADD exploration rule
- Branching rate 20-80% → No change needed
- **Branch rate** = parent ≠ (current_iter - 1), calculate for **entire block**

**Improvement rate:**

- If <30% improving → INCREASE exploitation (raise R² threshold)
- If >80% improving → INCREASE exploration (probe boundaries)

**Stuck detection:**

- Same R² plateau (±0.05) for 3+ iters? → ADD forced branching rule

**Dimension diversity:**

- Count consecutive iterations mutating **same parameter**
- If > 4 consecutive same-param → ADD switch-dimension rule

**UCB exploration constant (ucb_c):**

- `ucb_c` controls exploration vs exploitation: UCB(k) = R²_k + c × sqrt(ln(N) / n_k)
- Higher c (>1.5) → more exploration of under-visited branches
- Lower c (<1.0) → more exploitation of high-performing nodes
- Default: 1.414 (√2, standard UCB1)
- Adjust between blocks based on search behavior:
  - If stuck in local optimum (all R² similar, no improvement) → INCREASE ucb_c to 2.0
  - If too much random exploration (jumping between distant nodes) → DECREASE ucb_c to 1.0
  - Typical range: 0.5 to 3.0

### STEP 2: Choose Next Block Configuration

**Between-block changes (choose ONE per block boundary):**

Option A: **Change field_name**
- Rotate through fields: Jp → F → S → C → Jp...
- Allows testing same architecture across different fields

Option B: **Increase n_training_frames**
- Progression: 48 → 100 → 200 → 500 → 1000 → 2000 → 5000 → 10000
- Keep field_name constant to isolate effect of more training data
- IMPORTANT: Always ensure n_training_frames > batch_size

**Strategy guidelines:**
- Check Regime Comparison Table → choose untested combination
- **Do not replicate** previous block unless motivated (testing knowledge transfer)
- Alternate between Option A and B to build comprehensive understanding
- When stuck on one field, switch to another to test generalization

### STEP 3: Update Working Memory

Update `{config}_memory.md`:

- Update Knowledge Base with confirmed principles
- Add row to Regime Comparison Table
- Replace Previous Block Summary with **short summary** (2-3 lines, NOT individual iterations)
- Clear "Iterations This Block" section
- Write hypothesis for next block

---

## Working Memory Structure

```markdown
# Working Memory

## Knowledge Base (accumulated across all blocks)

### Regime Comparison Table

| Block | INR Type | Field | n_frames | Best R² | Best slope | Optimal lr_NNR_f | Optimal hidden_dim | Optimal n_layers | Optimal omega_f | Optimal total_steps | Training time (min) | Key finding |
| ----- | -------- | ----- | -------- | ------- | ---------- | ---------------- | ------------------ | ---------------- | --------------- | ------------------- | ------------------- | ----------- |
| 1     | siren_id | Jp    | 48       | 0.998   | 0.990      | 1E-5             | 512                | 3                | 30.0            | 50000               | 10.5                | ...         |

### Established Principles

[Confirmed patterns that apply across regimes]

### Open Questions

[Patterns needing more testing, contradictions]

---

## Previous Block Summary (Block N-1)

[Short summary only - NOT individual iterations]

---

## Current Block (Block N)

### Block Info

Field: field_name=X, inr_type=Y, ...
Iterations: M to M+n_iter_block

### Hypothesis

[Prediction for this block, stated before running]

### Iterations This Block

[Current block iterations only — cleared at block boundary]

### Emerging Observations

[Running notes on what's working/failing]
**CRITICAL: This section must ALWAYS be at the END of memory file. When adding new iterations, insert them BEFORE this section.**
```

---

## Knowledge Base Guidelines

### What to Add to Established Principles

Examples:

- ✓ "siren_txy needs lower lr_NNR_f than siren_id" (causal, generalizable)
- ✓ "omega_f > 50 causes training instability for F field" (boundary condition)
- ✓ "total_steps < 20k insufficient for R² > 0.95" (threshold)
- ✗ "lr_NNR_f=1E-5 worked in Block 4" (too specific)
- ✗ "Block 3 converged" (not a principle)

### Scientific Method (CRITICAL)

**Repeatability:**

- Each iteration is a single training run with stochastic initialization
- Results may vary between runs with identical config
- A "poor" run may succeed on retry; an "excellent" run may fail

**Evidence hierarchy:**

| Level            | Criterion                              | Action                 |
| ---------------- | -------------------------------------- | ---------------------- |
| **Established**  | Consistent across 3+ iterations/blocks | Add to Principles      |
| **Tentative**    | Observed 1-2 times                     | Add to Open Questions  |
| **Contradicted** | Conflicting evidence                   | Note in Open Questions |

---

## Theoretical Background

### INR Architectures for MPM

**SIREN (Sinusoidal Representation Networks)**:

```
f(t, id) → field_value
```

Three variants:
- `siren_t`: Input = time only (outputs all particles)
- `siren_id`: Input = (time, particle_id)
- `siren_txy`: Input = (time, x, y) - uses Lagrangian positions

**InstantNGP (Hash Encoding)**:
- Multi-resolution hash encoding + MLP
- Faster training but more memory
- `ngp`: Hash-encoded time representation

### MPM Fields

| Field | Description | Components | Typical Range |
|-------|-------------|------------|---------------|
| F | Deformation gradient | 4 | ~1.0-2.0 |
| Jp | Plastic deformation | 1 | ~0.8-1.2 |
| S | Stress tensor | 4 | ~0-0.01 |
| C | APIC matrix | 4 | ~-1 to 1 |

### Training Dynamics

**Learning rate sensitivity**:
- Too high → oscillation, NaN
- Too low → slow convergence, underfitting
- Optimal range: 1E-6 to 1E-4 for most fields

**Capacity vs Overfitting**:
- hidden_dim × n_layers determines capacity
- More capacity → better fit but slower
- Typical: 512×3 or 1024×3
- **IMPORTANT (Block 1 finding)**: Depth vs width tradeoff - 256×4 matches 512×3 accuracy (R²=0.908) at 2.5x faster training (7.7min vs 19.1min)
- **lr-depth relationship (Block 1 finding)**: Deeper networks require lower lr. Scaling: n_layers=3-4 tolerates lr=2E-5, n_layers=5 needs lr≤2E-5, n_layers=5 + lr=3E-5 fails catastrophically
- **5-layer ceiling (Block 2 finding)**: n_layers=5 degrades R² regardless of lr (tested 2E-5, 1.5E-5, 1E-5). 4 layers is the optimal depth for siren_txy.

**Training data scaling (Block 2 finding, updated Block 5)**:
- total_steps should scale with n_training_frames
- Approximate rule: steps_per_frame ≈ 1000 for R²>0.99 (Block 5 refinement)
- 48 frames → 50k steps OK; 100 frames → 100k steps sufficient (F field R²=0.999)
- **DATA SCALING BENEFIT (Block 5)**: More training frames IMPROVES accuracy (100 frames R²=0.9998 > 48 frames R²=0.9995 for F)

**SIREN frequency (omega_f)**:
- Low (1-10): smooth, low-frequency signals
- Medium (20-50): typical for MPM fields
- High (>50): high-frequency detail, unstable training
- **Field-specific optimal omega_f (Block 2 finding)**: F field optimal at omega_f=25, Jp optimal at 35. Different fields prefer different frequencies. F frequency mapping: 20(0.995)<25(0.999)>30(0.997)>35(0.996)>50(0.985)

**Compression ratio**:
- Data size / model parameters
- Higher is better (more compression)
- Typical target: >100x

---

## Metric Interpretation

### R² (Coefficient of Determination)

- R² = 1 - SS_res / SS_tot
- R² = 1.0: perfect prediction
- R² > 0.95: excellent (publication-quality)
- R² 0.90-0.95: good
- R² 0.75-0.90: moderate (needs improvement)
- R² < 0.75: poor (fundamental issue)

### MSE (Mean Squared Error)

- Lower is better
- Field-dependent scale
- Use R² for comparison across fields

### Training Time

- Linear with total_steps
- Quadratic with hidden_dim
- Consider cost vs accuracy tradeoff

---

## Common Failure Modes

1. **NaN loss**: lr_NNR_f too high or omega_f too high
2. **Underfitting**: total_steps too low, model too small
3. **Slow convergence**: lr_NNR_f too low
4. **Overfitting to noise**: batch_size too small
5. **Memory OOM**: hidden_dim or n_particles too large
6. **Training time explosion**: batch_size > 1 causes 7× slowdown (Block 1 finding) - use batch_size=1
7. **omega_f sensitivity**: For siren_txy, omega_f=30-35 is optimal. omega_f=20 underperforms, omega_f≥40 causes slope regression (Block 1 finding)
8. **hidden_dim ceiling**: hidden_dim=512 is optimal for siren_txy. hidden_dim=768 causes regression (R²=0.968→0.885) and 2× training time (Block 1 finding)
9. **lr upper bound**: lr_NNR_f=4E-5 causes training instability (R²=0.964→0.806). Optimal zone is 2E-5→3E-5 (Block 1 finding)
10. **F field efficiency (Block 2 finding)**: hidden_dim=256 sufficient for F field (R²=0.996), 256×4 achieves R²=0.9995 in 6.3min
11. **Depth > width (Block 2 finding)**: 512×2 (R²=0.977) < 256×3 (R²=0.996). Extra layer beats doubled width for SIREN.
12. **Field difficulty ranking (Block 2 finding)**: F field (4 components) achieves R²=0.9995 > Jp (1 component) at R²=0.968. F has more regular structure.
13. **S field (stress tensor) is HARD (Block 3 finding)**: Best R²=0.618 despite exhaustive optimization. omega_f=50 SHARP PEAK (both 45 and 60 regress). S values ~0-0.01 may need loss scaling or normalization code modification.
14. **Field-specific omega_f (Block 4 confirmed)**: Jp→35, F→25, C→30, S→50. Each field has distinct optimal frequency.
15. **Local optimum detection (Block 3 finding)**: When 5+ mutations from best node all regress, config-level is exhausted. Consider code modification or field change.
16. **C field optimal config (Block 4 finding)**: hidden_dim=384, n_layers=3, omega_f=30, lr=3E-5, 100k steps → R²=0.993. C behaves like F (easy), not S (hard).
17. **Overfitting via excess steps (Block 4 finding)**: 150k steps WORSE than 100k for C field (R²=0.979 vs 0.990). More training can hurt.
18. **Width ceiling field-dependent (Block 4 finding)**: hidden_dim=384 optimal for C field. 256→384 improves, 384→512 regresses. Optimal width varies by field.
19. **omega_f range for F (Block 5 refined)**: 15≤omega_f≤25 (plateau at 15-25, sharp dropoff at 30). Lower omega_f (15) slightly better than 25.
20. **LR-omega_f interaction (Block 5 finding)**: lr tolerance widens at lower omega_f. F field: lr=3E-5→4E-5 both work at omega_f=15, but lr=4E-5 fails at omega_f=30+ (Block 1).
21. **256×4 Pareto-optimal for F (Block 5 confirmed)**: 256×4 (R²=0.999, 6.4min) dominates 512×4 (R²=0.999, 12.5min). Same quality, 2× speed.
22. **Jp data scaling SUCCESS (Block 6 finding)**: 100 frames R²=0.982 > 48 frames R²=0.968 (+0.014). Requires 2000 steps/frame (vs F's 1000).
23. **Jp hidden_dim=384 optimal (Block 6 finding)**: 384 > 512 > 256 for Jp at 100 frames. Similar to C field, unlike F (256 optimal).
24. **Jp depth-sensitive (Block 6 finding)**: n_layers=4 causes major regression (R²=0.982→0.838) for Jp. 3 layers strictly optimal. Unlike F field (4 layers optimal).
25. **omega_f shifts with data scaling (Block 6 finding)**: Jp optimal omega_f: 35 (48 frames) → 30 (100 frames). More data → lower optimal frequency.

---

## Quick Reference

**Fast iteration (testing)**:
- total_steps=10000, hidden_dim=256, n_layers=2

**Production quality**:
- total_steps=50000-200000, hidden_dim=512-1024, n_layers=3-4

**Parameter mutation magnitude**:
- Learning rates: ×2 or ÷2
- Network size: double or half
- Training steps: ×1.5 or ×2
- omega_f: ±10 or ×1.5
