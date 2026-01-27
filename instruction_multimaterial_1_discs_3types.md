# MPM INR Training Landscape Study

## Goal

Map the **MPM INR training landscape**: understand which INR architectures and training configurations achieve best field reconstruction (R² > 0.995) for Material Point Method simulations. Other important evaluation are slope (~1.0) and training time. Get an intuition for the training_time.

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

- **Excellent**: R² > 0.995
- **Good**: R² 0.990 - 0.995
- **Moderate**: R² 0.90-0.99
- **Poor**: R² < 0.90

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

| Condition                            | Strategy             | Action                                    |
| ------------------------------------ | -------------------- | ----------------------------------------- |
| Default                              | **exploit**          | Highest UCB node, try mutation            |
| 3+ consecutive R² ≥ 0.95             | **failure-probe**    | Extreme parameter to find boundary        |
| n_iter_block/4 consecutive successes | **explore**          | Select outside recent chain               |
| Good config found                    | **robustness-test**  | Re-run same config                        |
| 2+ distant nodes with R² > 0.95      | **recombine**        | Merge params from both nodes              |
| 100% convergence, branching<10%      | **forced-branch**    | Select node in bottom 50% of tree         |
| Same param mutated 4+ times          | **switch-param**     | Mutate different parameter                |
| All R² > 0.94, branching <20%        | **random-branch**    | Select random unvisited parent            |
| Robustness test variance > 0.1       | **stochastic-field** | Switch to different field or try code mod |

**Stochastic variance detection (Block 9 finding):**

- If same config produces R² variance > 0.1 between runs, the field is **stochastically unstable**
- For S field: 1024×4 ranges 0.595-0.723, 1280×4 ranges 0.084-0.757
- When stochastic variance detected: (1) switch to different field, (2) try code modification (loss scaling, gradient clipping), or (3) accept unreliable ceiling

**Frame-dependent omega_f (Block 1 finding):**

- omega_f optimal values are FRAME-DEPENDENT:
  - Jp@48frames: omega_f=30 optimal (NOT 15-20)
  - Jp@200frames: omega_f=20-25 optimal (from prior)
- Rule: When changing n_training_frames, re-tune omega_f (expect higher omega_f for fewer frames)

**LR boundary mapping (Block 1 finding):**

- Jp@48frames LR boundary: 4E-5 < 5E-5 < 6E-5 (optimal) > 8E-5
- Rule: Probe LR upper boundary after finding good config. Optimal LR may be higher than prior knowledge suggests for fewer frames.

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
Total training_time may exceed 10 min if results plateau

```yaml
training:
  learning_rate_NNR_f: 1.0E-5 # range: 1E-7 to 1E-3
  batch_size: 8 # values: 4, 8, 16, 32, never larger than 32
  total_steps: 50000 # range: 5000-200000 (SCALE inversely with hidden_dim for training_time)
  n_training_frames: 48 # progression: 48 → 100 → 200 → 500 → 1000 → 2000 → 5000 → 10000
graph_model:
  hidden_dim_nnr_f: 512 # values: 128, 256, 512, 1024, 2048
  n_layers_nnr_f: 3 # range: 2-6
  omega_f: 30.0 # range: 1.0 to 100.0 (SIREN frequency)
  omega_f_learning: False # or True
  learning_rate_omega_f: 1.0E-6 # if omega_f_learning=True
  nnr_f_xy_period: 1.0 # spatial scaling (higher = expects slower spatial variation)
  nnr_f_T_period: 1.0 # time scaling (higher = expects slower temporal variation), prior: 10.0
  # NGP-specific - DO NOT USE (implementation incomplete)
  # ngp_n_levels: 16  # NOT READY
  # ngp_log2_hashmap_size: 19  # NOT READY
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
- **Use `inr_type: ngp` (InstantNGP/hash encoding) - NOT READY FOR USE** (implementation incomplete, will cause errors)

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

Add established principles

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

**SIREN Implementation Details**:

SIREN uses sinusoidal activations instead of ReLU:

$$\phi(x) = \sin(\omega_0 \cdot Wx + b)$$

Key implementation points:

| Component     | Formula                                           | Purpose                      |
| ------------- | ------------------------------------------------- | ---------------------------- |
| First layer   | $W \sim U(-1/n, 1/n)$                             | Input scaling                |
| Hidden layers | $W \sim U(-\sqrt{6/n}/\omega, \sqrt{6/n}/\omega)$ | Preserve gradient magnitude  |
| Activation    | $\sin(\omega_0 \cdot z)$                          | Periodic, smooth derivatives |

**Input normalization** (critical for `siren_txy`):

- Time: $t_{norm} = t / n\_frames / \text{nnr\_f\_T\_period}$
- Spatial: $(x, y)_{norm} = (x, y) / \text{nnr\_f\_xy\_period}$

The `omega_f` parameter controls frequency capacity - higher values capture finer spatial/temporal detail but risk training instability.

**Coordinate scaling parameters** (named "period" because they stretch the input range):
- `nnr_f_xy_period`: Divides spatial coordinates by this value. Larger period → inputs span smaller range → network sees slower spatial variation. Default: 1.0
- `nnr_f_T_period`: Divides time coordinate by this value. Larger period → inputs span smaller range → network sees slower temporal variation. Default: 1.0

**Intuition**: Think of it as "how many cycles fit in the data". Period=10 means the network treats the full time range as 1/10th of its natural period, so it expects 10× slower variation.

**Tuning guidelines:**
- **Physical prior**: Fields typically change less in time than in space → use `nnr_f_T_period > nnr_f_xy_period` (e.g., 10.0 vs 1.0)
- Smoother temporal dynamics → increase `nnr_f_T_period` (e.g., 2.0-10.0)
- Smoother spatial patterns → increase `nnr_f_xy_period`
- These allow independent tuning of spatial vs temporal sensitivity without changing `omega_f`

**InstantNGP (Hash Encoding)** ⚠️ **NOT READY - DO NOT USE**:

- Multi-resolution hash encoding + MLP
- Faster training but more memory
- `ngp`: Hash-encoded time representation
- **STATUS**: Implementation incomplete - will cause errors. Use SIREN variants only.

### MPM Fields

| Field | Description          | Components | Typical Range |
| ----- | -------------------- | ---------- | ------------- |
| F     | Deformation gradient | 4          | ~1.0-2.0      |
| Jp    | Plastic deformation  | 1          | ~0.8-1.2      |
| S     | Stress tensor        | 4          | ~0-0.01       |
| C     | APIC matrix          | 4          | ~-1 to 1      |

### Visualization Panels (Field Plot)

After each training iteration, a visualization is saved to `tmp_training/field/`. **You MUST analyze this plot qualitatively.**

**Panel Layout for 4-component fields (F, S, C):**

```
[Loss curve] [Traces]     [Info]      [Scatter]
[GT 00]      [GT 01]      [Pred 00]   [Pred 01]
[GT 10]      [GT 11]      [Pred 10]   [Pred 11]
```

**Panel Layout for 1-component fields (Jp):**

```
[Loss curve] [Traces]     [Info]
[GT field]   [Pred field] [Scatter]
```

**Panel descriptions:**

| Panel                                                          | What it shows                                                                              | What to look for                                                                                 |
| -------------------------------------------------------------- | ------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------ |
| **Loss curve** (top-left)                                      | MSE loss vs training steps. Gray=raw, Red=smoothed                                         | Convergence shape: steep initial drop is good. Oscillation = lr too high. Plateau = underfitting |
| **Traces** (top-middle)                                        | 10 particle time series. Green=GT, White=Pred. Centered by mean, spaced by 4×std           | Temporal dynamics match: white should follow green. Systematic offsets indicate bias             |
| **Info** (top-right for 4-comp, or right for 1-comp)           | Field name, step count, components, particles, frames, MSE                                 | Basic run metadata                                                                               |
| **Scatter** (top-right for 4-comp, or bottom-right for 1-comp) | Pred vs GT for ALL values. Red dashed=ideal, Green=regression fit                          | Cloud should be tight along diagonal. Slope<1 means underprediction. R² and slope shown          |
| **GT panels** (bottom rows)                                    | Ground truth field at frame n_frames/2. Each component shown separately. Yellow SSIM value | Spatial structure of the true field. SSIM measures structural similarity to prediction           |
| **Pred panels** (bottom rows)                                  | Slope-corrected prediction at same frame. White SSIM value                                 | Should match GT visually. Low SSIM despite high R² = wrong spatial patterns learned              |

**Qualitative analysis checklist:**

1. **Do Pred panels visually match GT panels?** Same spatial patterns, same contrast?
2. **Is SSIM consistent across components?** One low SSIM component = that component is harder
3. **Does the scatter show bias?** Slope ≠ 1 means systematic over/underprediction
4. **Are there spatial artifacts?** Blurry regions, wrong gradients, missing features?
5. **Do traces show temporal fidelity?** High-frequency oscillations captured?

**REQUIRED: Add qualitative observation to iteration log**

After viewing the plot, add a line to your iteration entry:

```
Visual: [brief qualitative observation about spatial match and any artifacts]
```

Examples:

- `Visual: GT/Pred match well, SSIM>0.95 all components, no visible artifacts`
- `Visual: F01 component blurry (SSIM=0.72), other components good`
- `Visual: Spatial patterns correct but contrast too low (slope=0.85)`
- `Visual: Complete mismatch - prediction shows uniform field, GT has structure`

### Prior Knowledge (from 17 blocks prior exploration)

**Field difficulty ranking**: F (R²=0.9999) > Jp (R²=0.997) > C (R²=0.989) >> S (R²=0.801)

**Optimal configurations per field** (use these as starting points):

| Field | hidden_dim | n_layers | omega_f | lr_NNR_f | steps/frame | Best R² |
| ----- | ---------- | -------- | ------- | -------- | ----------- | ------- |
| F     | 256        | 4        | 20      | 3E-5     | 800         | 0.9999  |
| Jp    | 384        | 3        | 15-20   | 4E-5     | 800         | 0.997   |
| C     | 640        | 3        | 20-25   | 3E-5     | 1000        | 0.989   |
| S     | 1280       | 4        | 50      | 2E-5     | 3000+       | 0.801   |

**Critical constraints discovered**:

- **n_layers**: Jp and C require EXACTLY 3 layers (both 2 and 4 degrade). F tolerates 2-5. S prefers 4.
- **hidden_dim ceilings**: S@1280 (1536 fails), Jp@384 (256 AND 512 both degrade at 50k steps, but 512 works with 200k steps - Block 1 multimaterial), C scales with frames
- **omega_f**: Field-specific and frame-dependent. More frames → lower optimal omega_f.
- **Data scaling**: F scales excellently (no diminishing returns to 1000f). Jp benefits. C hurts (mitigate with capacity). S fails entirely.
- **S field instability**: EXTREME stochastic variance (R² 0.08-0.80 same config). Use gradient clipping max_norm=1.0.
- **SIREN + normalization**: LayerNorm/BatchNorm DESTROY SIREN (R²=0.022). Never use.
- **batch_size**: Always use batch_size=1 to avoid training time explosion.

### Training Dynamics

**Learning rate sensitivity**:

- Too high → oscillation, NaN
- Too low → slow convergence, underfitting
- Optimal range: 1E-6 to 1E-4 for most fields
- **DATASET-SPECIFIC LR (Block 1 multimaterial finding)**: multimaterial_1_discs_3types tolerates 1.5x higher LR than prior knowledge. Jp optimal LR=6E-5 (not 4E-5). LR boundary: 6E-5 optimal, 7E-5 and 8E-5 both regress. Always probe LR upper boundary on new datasets.

**Capacity vs Overfitting**:

- hidden_dim × n_layers determines capacity
- More capacity → better fit but slower
- Typical: 512×3 or 1024×3
- **IMPORTANT (Block 1 finding)**: Depth vs width tradeoff - 256×4 matches 512×3 accuracy (R²=0.908) at 2.5x faster training (7.7min vs 19.1min)
- **lr-depth relationship (Block 1 finding)**: Deeper networks require lower lr. Scaling: n_layers=3-4 tolerates lr=2E-5, n_layers=5 needs lr≤2E-5, n_layers=5 + lr=3E-5 fails catastrophically
- **5-layer ceiling (Block 2 finding)**: n_layers=5 degrades R² regardless of lr (tested 2E-5, 1.5E-5, 1E-5). 4 layers is the optimal depth for siren_txy.
- **F field depth tolerance (Block 11 finding)**: F field TOLERATES 5 layers (R²=0.9999) unlike Jp which degrades. F: depth [2-5] all viable, Jp: depth ceiling at 3 layers.
- **Depth Pareto tradeoff (Block 11)**: F@200frames: 256×2 achieves R²=0.994 in 9.5min (speed Pareto), 256×4 achieves R²=0.9998 in 18min (accuracy Pareto).

**Training data scaling (Block 2 finding, updated Block 5, Block 10)**:

- total_steps should scale with n_training_frames
- Approximate rule: steps_per_frame ≈ 1000 for R²>0.99 (Block 5 refinement), Jp needs 2000 steps/frame (confirmed Block 1 multimaterial)
- 48 frames → 50k steps OK; 100 frames → 100k steps sufficient (F field R²=0.999)
- **DATA SCALING BENEFIT (Block 5)**: More training frames IMPROVES accuracy (100 frames R²=0.9998 > 48 frames R²=0.9995 for F)
- **OVERFITTING RISK (Block 10)**: Too many steps causes overfitting. Jp@200frames: 400k OK (0.989), 500k OVERFITS (0.939). Never exceed 2500 steps/frame.
- **DIMINISHING RETURNS (Block 10)**: Jp data scaling gains halve per doubling: +0.014 (48→100), +0.007 (100→200)
- **F NO DIMINISHING RETURNS (Block 11)**: F field shows NO diminishing returns at 200 frames (R²=0.9998 same as 100 frames). F more scalable than Jp.
- **F steps/frame scaling (Block 11, updated Block 14)**: F needs 1500 steps/frame at 200 frames (vs 1000 at 100 frames). BUT at 500 frames, only 800 steps/frame needed (efficiency gain).
- **F EXTREME DATA SCALING (Block 14)**: F field scales to 500 frames with NO diminishing returns (R²=0.9997). Speed Pareto: 400 steps/frame achieves R²=0.992 in 10min.
- **DATA SCALING FIELD CATEGORIZATION (Block 12)**: F benefits (no diminishing returns), Jp benefits (diminishing returns), C HURTS (0.994@200f < 0.996@100f), S HURTS (fails entirely). Half of fields penalized by more data.

**SIREN frequency (omega_f)**:

- Low (1-10): smooth, low-frequency signals
- Medium (20-50): typical for MPM fields
- High (>50): high-frequency detail, unstable training
- **Field-specific optimal omega_f (Block 2 finding, updated Block 10, Block 1 multimaterial)**: F field optimal at omega_f=15-25, Jp optimal at 20-25 (200 frames) or 30-35 (48 frames). More frames → lower optimal omega_f.
- **Jp@200frames omega_f map (Block 10)**: 35(0.786) << 30(0.982) < 25(0.985) ≈ 20(0.989) > 15(0.978). Optimal=20-25.
- **Jp@100frames omega_f map (Block 1 multimaterial, 9000 particles)**: 25(0.979) < 20(0.984) < 15(0.988). Optimal=15. More particles may favor lower omega_f.
- **F@200frames boundaries (Block 11)**: omega_f [15-40] viable (15=0.9996, 20=0.9998, 25=0.9997, 40=0.9991). lr [3E-5 to 1E-4] viable. lr=2E-4 FAILS.
- **F@100frames omega_f LOCAL MAXIMUM (Block 2 multimaterial)**: F has NARROW omega_f window at 100 frames: 15(0.957) < 18(0.989) < 25(0.985) < 20(0.998). omega_f=20 is LOCAL MAXIMUM. Even ±2 deviation causes significant degradation. More sensitive than Jp.
- **F@100frames optimal config (Block 2 multimaterial)**: 256×4-5, omega_f=20, lr=3E-5, 150k steps (1500 steps/frame). R²=0.998. F tolerates n_layers 4-5 equally (unlike Jp ceiling at 3).
- **C@200frames omega_f map (Block 12)**: 15(0.993) < 20(0.990) < 25(0.994) < 27(0.992) < 30(0.989). Peak at omega_f=25 (NOT 20 like F). Non-monotonic.
- **C depth ceiling (Block 12)**: C field is depth-sensitive like Jp. n_layers=4 regresses (0.987 vs 0.994 at n_layers=3). Optimal depth=3.
- **C@100frames COMPLETE PARAMETER MAPPING (Block 3 multimaterial)**: All parameters have LOCAL MAXIMA - no further tuning possible. Maps: omega_f: 22(0.985) < 25(0.989) > 28(0.979). hidden_dim: 512(0.984) < 640(0.989) > 768(0.977). n_layers: 2(0.968) < 3(0.989) > 4(0.982). lr: 2E-5(0.983) < 3E-5(0.989) > 4E-5(0.982). R²=0.989 is architectural ceiling.
- **C LR-DEPTH INTERACTION (Block 3 multimaterial)**: Shallow networks (3L) prefer lower lr (3E-5), deep networks (4L) prefer higher lr (4E-5). However 3L@optimal (0.989) always beats 4L@optimal (0.986). Depth penalty cannot be overcome via LR tuning.
- **C capacity ceiling at 640 (Block 3 multimaterial)**: hidden_dim=640 is LOCAL MAXIMUM - both 512 (0.984) and 768 (0.977) degrade. C field does NOT benefit from more capacity. Contrast with F (scales) and S (needs 1280).
- **S field capacity ceiling (Block 13)**: S requires MUCH more capacity than other fields. Optimal: 1280×4 (R²=0.801). Capacity scaling: 256→0.389, 512→0.638, 1024→0.646, 1280→0.801, 1536→0.145 (FAILS). S also requires omega_f=50, lr=2E-5 (NO tolerance for lr=3E-5).
- **S field gradient clipping (Block 17)**: CRITICAL - S field REQUIRES gradient clipping max_norm=1.0 for stable training. Clipping map: 0.25→0.810, 0.5→[0.828,0.181], 1.0→[0.785,0.787,0.732]. max_norm>1.5 causes catastrophic failure (R²<0.13). max_norm=1.0 is MOST STABLE (mean R²=0.768±0.031).
- **SIREN normalization incompatibility (Block 17)**: LayerNorm, BatchNorm INCOMPATIBLE with SIREN architecture. LayerNorm causes R²=0.022 (catastrophic). SIREN relies on omega-scaled initialization which normalization destroys.

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
- nnr_f_xy_period: ×2 or ÷2 (higher = slower spatial variation expected)
- nnr_f_T_period: ×2 or ÷2 (higher = slower temporal variation expected)
