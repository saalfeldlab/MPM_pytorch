# Working Memory: multimaterial_1_discs_3types (parallel)

## Knowledge Base (accumulated across all blocks)

### Regime Comparison Table
| Block | INR Type | Field | n_training_frames | Best R² | Best slope | kino_R2 | kino_SSIM | Optimal lr_NNR_f | Optimal hidden_dim | Optimal n_layers | Optimal omega_f | Optimal total_steps | Training time (min) | Key finding |
|-------|----------|-------|-------------------|---------|------------|---------|-----------|------------------|--------------------|--------------------|-----------------|---------------------|---------------------|-------------|
| 1 (partial) | siren_txy | F | 400 | 0.9999 | 0.9999 | 0.9999 | 0.9918 | 8E-5 | 256 | 4 | 8.0 | 320000 | 18.5 | lr=8E-5 best; omega_f=8 confirmed; depth=4 required |

### Established Principles
- From appendix: F is most scalable field (no diminishing returns to 500f)
- F@200f optimal: 256×4, omega_f=9-10, lr=5E-5, 300k steps (R²=0.9997)
- F omega_f scales down ~2-3 per +100 frames (12→9-10 from 100→200)
- F capacity ceiling at 256 (384 HURTS)
- F depth ceiling at 4 layers (siren_txy)
- Period parameters must stay at 1.0 for F
- Data regularization allows higher lr at higher n_training_frames

### Open Questions
- F@400f lr upper boundary: 8E-5 works, does 1E-4 or 1.2E-4? (testing in batch 2)
- F@400f omega_f upper boundary: 8>6, does 10 help? (testing in batch 2)
- F@400f step optimization: 320k sufficient or does 400k help at lr=8E-5? (testing in batch 2)
- Steps/frame at 400f: 800/f (320k/400) achieves R²=0.9999 — already excellent

---

## Previous Block Summary

(No previous blocks — this is the initialization)

---

## Current Block (Block 1)

### Block Info
Field: field_name=F, inr_type=siren_txy, n_training_frames=400
Parallel mode: 4 slots exploring different parameter dimensions simultaneously

### Hypothesis
F field should scale to 400 frames with R²>0.999 based on extrapolation from 100→200→500 trend. Starting from appendix Section 6 baseline (omega_f=7-8, lr=5E-5, 256×4, 320k steps). The 4 initial slots test: (1) baseline omega_f=8, (2) lower omega_f=6, (3) higher lr=8E-5, (4) reduced depth n_layers=3 with more steps.

### Planned Initial Configurations (Batch 1)

| Slot | omega_f | lr_NNR_f | hidden_dim | n_layers | total_steps | Mutation dimension |
|------|---------|----------|------------|----------|-------------|--------------------|
| 00 | 8.0 | 5E-5 | 256 | 4 | 320000 | **Baseline** (appendix reference) |
| 01 | 6.0 | 5E-5 | 256 | 4 | 320000 | **omega_f** (lower, test continued scaling) |
| 02 | 8.0 | 8E-5 | 256 | 4 | 320000 | **lr** (higher, test data regularization at 400f) |
| 03 | 8.0 | 5E-5 | 256 | 3 | 400000 | **depth** (n_layers=3, more steps to compensate) |

All slots: siren_txy, batch_size=1, n_training_frames=400, nnr_f_xy_period=1.0, nnr_f_T_period=1.0

### Iterations This Block

## Iter 1: excellent — Baseline (omega_f=8, lr=5E-5, 256×4, 320k)
Node: id=1, parent=root. R²=0.9998, slope=0.9997, kino_R2=0.9997, kino_SSIM=0.9842, 18.9min
Observation: Appendix baseline works excellently at 400f. Loss still declining.

## Iter 2: excellent — omega_f=6 (lower than baseline)
Node: id=2, parent=root. R²=0.9996, slope=0.9995, kino_R2=0.9995, kino_SSIM=0.9744, 18.4min
Observation: omega_f=6 < omega_f=8. Linear scaling 12→9→6 does NOT continue. omega_f plateaus.

## Iter 3: excellent — lr=8E-5 (higher than baseline) **BEST**
Node: id=3, parent=root. R²=0.9999, slope=0.9999, kino_R2=0.9999, kino_SSIM=0.9918, 18.5min
Observation: lr=8E-5 BEST. Data regularization confirmed at 400f. 2× MSE improvement over baseline.

## Iter 4: good — n_layers=3 (shallower + more steps)
Node: id=4, parent=root. R²=0.9990, slope=0.9989, kino_R2=0.9987, kino_SSIM=0.9532, 18.5min
Observation: Depth=3 significantly worse even with +25% steps. F@400f REQUIRES depth=4.

### Emerging Observations

- **F scales to 400 frames**: All configs achieve R²≥0.999. Best R²=0.9999 (lr=8E-5).
- **omega_f scaling plateaus**: 12(100f) → 9(200f) → 8(400f). NOT linear — approaches asymptote.
- **lr ceiling rises with data**: 5E-5(200f) → 8E-5(400f). Consistent with Jp pattern. Need to probe higher.
- **Depth=4 mandatory**: n_layers=3 loses ~0.0008 R² even with more steps. No workaround.
- **Ranking**: lr=8E-5 (0.9999) > baseline (0.9998) > omega_f=6 (0.9996) > depth=3 (0.999)
- **Next batch strategy**: Exploit Node 3 (lr=8E-5). Test lr upper bound, omega_f=10, more steps.
