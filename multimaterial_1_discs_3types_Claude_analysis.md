# Experiment Log: multimaterial_1_discs_3types (parallel)

## Block 1 Initialization: F field @ 400 frames (parallel)

**Field**: F (deformation gradient, 4 components)
**INR type**: siren_txy
**n_training_frames**: 400
**Rationale**: F is the most scalable field (no diminishing returns to 500f in prior exploration). Starting at 400 frames to push beyond the fully-mapped 200-frame regime. Using appendix Section 6 as foundation.

### Batch 1 — Initial Configurations (4 diverse starting points)

| Slot | omega_f | lr_NNR_f | hidden_dim | n_layers | total_steps | Dimension tested |
|------|---------|----------|------------|----------|-------------|------------------|
| 00 | 8.0 | 5E-5 | 256 | 4 | 320000 | Baseline (appendix F@400 reference) |
| 01 | 6.0 | 5E-5 | 256 | 4 | 320000 | omega_f (continued scaling from 12→9→?) |
| 02 | 8.0 | 8E-5 | 256 | 4 | 320000 | lr (higher lr via data regularization) |
| 03 | 8.0 | 5E-5 | 256 | 3 | 400000 | depth (n_layers=3, +25% steps to compensate) |

**Design rationale**:
- Slot 00: Direct appendix extrapolation — reference point for all comparisons
- Slot 01: Tests whether omega_f continues linear decrease (12→9→6 at 400f) or plateaus
- Slot 02: Tests data regularization effect — prior shows lr ceiling rises with more frames (Jp: 4E-5→1E-4 from 100→200f)
- Slot 03: Tests depth sensitivity — F tolerates 2-5 layers at 200f but optimal=4; at 400f with more data, shallower may suffice with more steps

All slots share: siren_txy, batch_size=1, n_training_frames=400, output_size_nnr_f=4, nnr_f_xy_period=1.0, nnr_f_T_period=1.0

### Batch 1 Results (Iterations 1-4)

## Iter 1: excellent
Node: id=1, parent=root
Mode/Strategy: explore/baseline
Config: lr_NNR_f=5E-5, total_steps=320000, hidden_dim_nnr_f=256, n_layers_nnr_f=4, omega_f=8.0, batch_size=1
Metrics: final_r2=0.9998, final_mse=1.107E-4, slope=0.9997, kinograph_R2=0.9997, kinograph_SSIM=0.9842, total_params=265220, compression_ratio=54.3, training_time=18.9min
Field: field_name=F, inr_type=siren_txy
Mutation: Baseline from appendix (F@400f reference: omega_f=8, lr=5E-5, 256×4, 320k steps)
Parent rule: Block initialization — direct appendix extrapolation
Visual: GT/Pred match well across all 4 components, scatter tight along diagonal, loss curve converged smoothly with continued descent at end. Per-frame MSE shows U-shape (boundary frames harder). No visible artifacts.
Observation: Appendix baseline achieves excellent R²=0.9998 at 400 frames — F scales beautifully. Loss still declining at 320k steps suggesting more training could help.
Next: parent=3 (highest UCB)

## Iter 2: excellent
Node: id=2, parent=root
Mode/Strategy: explore/omega_f-scaling
Config: lr_NNR_f=5E-5, total_steps=320000, hidden_dim_nnr_f=256, n_layers_nnr_f=4, omega_f=6.0, batch_size=1
Metrics: final_r2=0.9996, final_mse=2.147E-4, slope=0.9995, kinograph_R2=0.9995, kinograph_SSIM=0.9744, total_params=265220, compression_ratio=54.3, training_time=18.4min
Field: field_name=F, inr_type=siren_txy
Mutation: omega_f: 8.0 -> 6.0 (testing continued linear decrease 12→9→6)
Parent rule: Block initialization — test omega_f continued scaling hypothesis
Visual: GT/Pred match well, slightly less crisp than Slot 00. Per-frame MSE shows larger U-shape (higher boundary error), some scatter dispersion at high GT values. Loss curve still declining but higher floor than Slot 00.
Observation: omega_f=6 underperforms omega_f=8 (0.9996 vs 0.9998). Linear omega_f decrease does NOT continue at 400f — omega_f plateaus around 8. Answer to open question: omega_f scaling slows/stops.
Next: parent=3 (highest UCB)

## Iter 3: excellent
Node: id=3, parent=root
Mode/Strategy: explore/lr-scaling
Config: lr_NNR_f=8E-5, total_steps=320000, hidden_dim_nnr_f=256, n_layers_nnr_f=4, omega_f=8.0, batch_size=1
Metrics: final_r2=0.9999, final_mse=5.429E-5, slope=0.9999, kinograph_R2=0.9999, kinograph_SSIM=0.9918, total_params=265220, compression_ratio=54.3, training_time=18.5min
Field: field_name=F, inr_type=siren_txy
Mutation: lr_NNR_f: 5E-5 -> 8E-5 (testing data regularization allows higher lr at 400f)
Parent rule: Block initialization — test lr ceiling at 400 frames
Visual: Excellent match across all 4 components. Scatter is tightest of all 4 slots — nearly perfect diagonal. Per-frame MSE U-shape has lowest floor. Loss curve shows deepest convergence.
Observation: lr=8E-5 is BEST in batch (R²=0.9999, MSE=5.43E-5). Confirms data regularization: F@200f optimal lr=5E-5, F@400f optimal lr≥8E-5. lr ceiling rises with more frames as predicted. Slope=0.9999 near-perfect.
Next: parent=3 (highest UCB)

## Iter 4: good
Node: id=4, parent=root
Mode/Strategy: explore/depth
Config: lr_NNR_f=5E-5, total_steps=400000, hidden_dim_nnr_f=256, n_layers_nnr_f=3, omega_f=8.0, batch_size=1
Metrics: final_r2=0.9990, final_mse=5.022E-4, slope=0.9989, kinograph_R2=0.9987, kinograph_SSIM=0.9532, total_params=199428, compression_ratio=72.3, training_time=18.5min
Field: field_name=F, inr_type=siren_txy
Mutation: n_layers_nnr_f: 4 -> 3, total_steps: 320000 -> 400000 (depth reduction + step compensation)
Parent rule: Block initialization — test if shallower network viable at 400f
Visual: Good match but noticeably less sharp than 4-layer variants. Pred 01 and Pred 11 show slight blurring at disc boundaries. Scatter has wider dispersion, especially at extreme GT values. Per-frame MSE U-shape has notably higher floor.
Observation: n_layers=3 significantly worse (0.9990 vs 0.9998 baseline). Even 25% more steps don't compensate for lost depth. F@400f REQUIRES 4 layers — depth ceiling holds at 400f. Better compression (72.3 vs 54.3) not worth accuracy loss.
Next: parent=3 (highest UCB)

### Batch 2 Design (Iterations 5-8)

**Strategy**: Exploit from Node 3 (best: R²=0.9999, lr=8E-5). All 4 slots derive from Node 3.
- Slot 00: Push lr higher (1E-4) — test lr upper boundary
- Slot 01: omega_f=10 — test if slightly higher omega_f helps (probe upward since 8>6)
- Slot 02: More training steps (400k) at lr=8E-5 — loss was still declining
- Slot 03: omega_f=8, lr=1.2E-4 — aggressive lr probe to find ceiling

