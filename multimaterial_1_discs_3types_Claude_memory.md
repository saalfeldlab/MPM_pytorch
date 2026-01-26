# Working Memory: multimaterial_1_discs_3types_Claude

## Knowledge Base (accumulated across all blocks)

### Regime Comparison Table
| Block | INR Type | Field | n_frames | Best R² | Best slope | Optimal lr_NNR_f | Optimal hidden_dim | Optimal n_layers | Optimal omega_f | Optimal total_steps | Training time (min) | Key finding |
|-------|----------|-------|----------|---------|------------|------------------|--------------------|--------------------|-----------------|---------------------|---------------------|-------------|

### Established Principles

### Open Questions
- What is the minimum hidden_dim for Jp@100frames with 9000 particles?
- Is omega_f=80 ever appropriate for any field/frame combination?

---

## Previous Block Summary

(No previous block)

---

## Current Block (Block 1)

### Block Info
Field: Jp, INR Type: siren_txy, n_frames: 100
Iterations: 1-12
Dataset: multimaterial_1_discs_3types (9000 particles, 9 objects, 3 material types)

### Hypothesis
Prior knowledge suggests Jp@100frames needs hidden_dim≈384, omega_f≈25-30, lr≈4E-5. Starting config (hidden_dim=64, omega_f=80) is severely under-capacity and over-frequency. Expect dramatic improvement with capacity increase.

### Iterations This Block

**Iter 1: poor** (R²=0.399, slope=0.049)
Node: id=1, parent=root
Config: hidden_dim=64, n_layers=3, omega_f=80, lr=1E-5, 50k steps
Observation: Severe underfitting - model predicts near-constant. hidden_dim=64 is grossly inadequate.
Next: parent=1, hidden_dim 64→384, omega_f 80→25

**Iter 2: moderate** (R²=0.835, slope=0.257)
Node: id=2, parent=1
Config: hidden_dim=384, n_layers=3, omega_f=25, lr=1E-5, 50k steps
Mutation: hidden_dim 64→384, omega_f 80→25
Observation: Capacity increase improved R² (0.399→0.835), but slope=0.257 indicates severe underprediction.
Next: parent=2, lr 1E-5→4E-5

**Iter 3: moderate** (R²=0.913, slope=0.683)
Node: id=3, parent=2
Config: hidden_dim=384, n_layers=3, omega_f=25, lr=4E-5, 50k steps
Mutation: lr 1E-5→4E-5
Observation: LR increase improved R² (+0.078) and slope dramatically (+0.426). Model now predicting values, not constant.
Next: parent=3, probe LR upper boundary: lr 4E-5→6E-5

**Iter 4: moderate** (R²=0.942, slope=0.825)
Node: id=4, parent=3
Config: hidden_dim=384, n_layers=3, omega_f=25, lr=6E-5, 50k steps
Mutation: lr 4E-5→6E-5
Observation: LR increase improved R² (+0.029) and slope (+0.142). Scatter shows underprediction at high GT values. Per-frame MSE higher at early frames.
Visual: GT/Pred spatial patterns similar, some artifacts in Pred, slope<1 visible in scatter.
Next: parent=4, continue LR probe: lr 6E-5→8E-5

**Iter 5: moderate** (R²=0.892, slope=0.863)
Node: id=5, parent=4
Config: hidden_dim=384, n_layers=3, omega_f=25, lr=8E-5, 50k steps
Mutation: lr 6E-5→8E-5
Observation: **LR BOUNDARY FOUND.** R² regressed (0.942→0.892) while slope improved (0.825→0.863). lr=8E-5 overshoots. Optimal lr=6E-5.
Next: parent=4, increase hidden_dim 384→512 to improve capacity

**Iter 6: moderate** (R²=0.918, slope=0.822)
Node: id=6, parent=4
Config: hidden_dim=512, n_layers=3, omega_f=25, lr=6E-5, 50k steps
Mutation: hidden_dim 384→512
Observation: **CAPACITY INCREASE REGRESSED R²** (0.942→0.918). 512 may overfit or need more steps. hidden_dim=384 optimal confirmed.
Visual: GT/Pred similar but artifacts. Per-frame MSE higher at early frames.
Next: parent=6, total_steps 50k→100k (need 1000 steps/frame)

**Iter 7: moderate** (R²=0.961, slope=0.890)
Node: id=7, parent=6
Config: hidden_dim=512, n_layers=3, omega_f=25, lr=6E-5, 100k steps
Mutation: total_steps 50k→100k
Observation: **STEPS INCREASE HIGHLY EFFECTIVE.** R² improved +0.043, slope +0.068. hidden_dim=512 works well with 1000 steps/frame.
Next: parent=7, total_steps 100k→150k (test if more steps push R²>0.99)

**Iter 8: good** (R²=0.979, slope=0.930)
Node: id=8, parent=7
Mode/Strategy: exploit
Config: hidden_dim=512, n_layers=3, omega_f=25, lr=6E-5, 150k steps
Mutation: total_steps 100k→150k
Observation: **MORE STEPS CONTINUES TO IMPROVE.** R² +0.018 (0.961→0.979), slope +0.040 (0.890→0.930). 1500 steps/frame effective. Approaching R²>0.99 target.
Visual: GT/Pred spatial patterns match well, scatter tight along diagonal, some outliers at high GT values. Per-frame MSE high at early frames but converges to near-zero by frame 50. Loss curve still improving at 150k steps.
Next: parent=8, omega_f 25→20 (test if lower frequency helps remaining error)

### Emerging Observations
- Initial config (hidden_dim=64, omega_f=80) catastrophically wrong for Jp@100frames
- Capacity increase (64→384) + omega_f reduction (80→25) improved R² 2x (0.399→0.835)
- **LR BOUNDARY MAPPED**: 1E-5(0.835) < 4E-5(0.913) < 6E-5(0.942) > 8E-5(0.892). Optimal=6E-5.
- Prior knowledge confirmed: multimaterial tolerates 1.5x higher LR than prior 4E-5 baseline
- **HIDDEN_DIM CEILING REVISED**: At 50k steps: 384(0.942) > 512(0.918). At 100k steps: 512(0.961) - steps matter more than capacity!
- R² trajectory: 0.399 → 0.835 → 0.913 → 0.942 → 0.892 → 0.918 → 0.961 → **0.979** (new peak at Node 8)
- Slope trajectory: 0.049 → 0.257 → 0.683 → 0.825 → 0.863 → 0.822 → 0.890 → **0.930** (new peak at Node 8)
- **STEPS PER FRAME SCALING CONFIRMED**: 500 steps/frame→R²=0.918, 1000→0.961, **1500→0.979**. Approaching 0.99!
- Training time scaling: 8.6min (50k) → 10.4min (100k) → 16.8min (150k). Linear with steps.
- Next: test omega_f=20 to see if lower frequency helps with remaining error (currently slope=0.930 indicates slight underprediction)

