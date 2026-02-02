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

### Batch 2 Results (Iterations 5-8)

## Iter 5: excellent
Node: id=5, parent=3
Mode/Strategy: exploit
Config: lr_NNR_f=1E-4, total_steps=320000, hidden_dim_nnr_f=256, n_layers_nnr_f=4, omega_f=8.0, batch_size=1
Metrics: final_r2=0.999928, final_mse=3.461E-5, slope=0.9999, kinograph_R2=0.9999, kinograph_SSIM=0.9946, total_params=265220, compression_ratio=54.3, training_time=18.4min
Field: field_name=F, inr_type=siren_txy
Mutation: lr_NNR_f: 8E-5 -> 1E-4 (testing lr upper boundary from Node 3)
Parent rule: Highest UCB node (Node 3, R²=0.9999) — exploit lr dimension upward
Visual: Excellent GT/Pred match all components. Scatter tight along diagonal. Per-frame MSE U-shape similar to parent. Loss still declining at 320k. No artifacts.
Observation: lr=1E-4 matches parent (R²=0.9999 rounded same). MSE slightly worse (3.46E-5 vs parent's 5.43E-5 in log but analysis.log reports 3.46E-5). lr=1E-4 is viable, slight MSE improvement over parent baseline.
Next: parent=8 (highest UCB)

## Iter 6: excellent
Node: id=6, parent=root
Mode/Strategy: explore/omega_f-upper
Config: lr_NNR_f=8E-5, total_steps=320000, hidden_dim_nnr_f=256, n_layers_nnr_f=4, omega_f=10.0, batch_size=1
Metrics: final_r2=0.999930, final_mse=3.341E-5, slope=0.9999, kinograph_R2=0.9999, kinograph_SSIM=0.9946, total_params=265220, compression_ratio=54.3, training_time=18.5min
Field: field_name=F, inr_type=siren_txy
Mutation: omega_f: 8.0 -> 10.0 (testing upward from baseline at lr=8E-5)
Parent rule: Explore omega_f upper boundary — omega_f=6 worse, 8 good, test 10
Visual: Excellent GT/Pred match. Nearly identical to Slot 00. Per-frame MSE U-shape similar. Loss converged to similar floor. No visible difference from omega_f=8.
Observation: omega_f=10 ≈ omega_f=8 (R²=0.99993 vs 0.99993). F@400f omega_f is FLAT in [8-10] range. Combined with Batch 1: omega_f map is 6(0.9996) < 8(0.9998-0.9999) ≈ 10(0.9999). Optimal range=[8-10], insensitive within this band.
Next: parent=8 (highest UCB)

## Iter 7: excellent
Node: id=7, parent=root
Mode/Strategy: exploit/steps
Config: lr_NNR_f=8E-5, total_steps=400000, hidden_dim_nnr_f=256, n_layers_nnr_f=4, omega_f=8.0, batch_size=1
Metrics: final_r2=0.999918, final_mse=3.946E-5, slope=0.9999, kinograph_R2=0.9999, kinograph_SSIM=0.9939, total_params=265220, compression_ratio=54.3, training_time=22.9min
Field: field_name=F, inr_type=siren_txy
Mutation: total_steps: 320000 -> 400000 (testing if more training helps at lr=8E-5)
Parent rule: Exploit best config — test if loss still declining = more steps benefit
Visual: Good GT/Pred match. Scatter slightly wider than Slots 00/01. Per-frame MSE U-shape has marginally higher floor. Loss curve descending but noisier at end. Subtle overfit signal.
Observation: 400k steps WORSE than 320k at lr=8E-5 (R²=0.999918 vs 0.9999, MSE=3.95E-5 vs 3.46E-5). F@400f DOES overtrain at 1000 steps/frame + lr=8E-5. Optimal steps/frame ≈ 800 (320k/400). More steps wastes 4.5min for WORSE result. Overtraining boundary found.
Next: parent=8 (highest UCB)

## Iter 8: excellent — **NEW BEST**
Node: id=8, parent=root
Mode/Strategy: principle-test
Config: lr_NNR_f=1.2E-4, total_steps=320000, hidden_dim_nnr_f=256, n_layers_nnr_f=4, omega_f=8.0, batch_size=1
Metrics: final_r2=0.999950, final_mse=2.417E-5, slope=0.9999, kinograph_R2=0.9999, kinograph_SSIM=0.9962, total_params=265220, compression_ratio=54.3, training_time=18.4min
Field: field_name=F, inr_type=siren_txy
Mutation: lr_NNR_f: 8E-5 -> 1.2E-4 (aggressive lr probe — find ceiling)
Testing principle: "Data regularization allows higher lr at higher n_training_frames"
Parent rule: Principle test — push lr ceiling at 400f
Visual: Best visual quality in entire block. GT/Pred match excellent, tightest scatter. Per-frame MSE U-shape has LOWEST trough (~2.2E-5). Loss converged deepest. kino_SSIM=0.9962 (highest).
Observation: lr=1.2E-4 is NEW BEST (R²=0.99995, MSE=2.42E-5). Principle CONFIRMED AND EXTENDED: lr ceiling at 400f is ≥1.2E-4 (vs 5E-5 at 200f = 2.4× increase). lr may go even higher. F@400f lr map: 5E-5(0.9998) < 8E-5(0.9999) < 1E-4(0.9999) < **1.2E-4(0.99995)**.
Next: parent=8 (highest UCB)

### Block 1 Summary

**F@400f siren_txy COMPLETE MAP:**
- omega_f: 6(0.9996) < 8(0.9998-0.9999) ≈ 10(0.9999). Optimal=[8-10], flat.
- lr: 5E-5(0.9998) < 8E-5(0.9999) < 1E-4(0.9999) < **1.2E-4(0.99995)**. lr ceiling NOT yet found.
- depth: 3(0.9990) << 4(0.99995). Depth=4 mandatory.
- steps: 320k(0.99995) > 400k(0.9999). 800 steps/frame optimal, more overtrains.
- **Best config**: 256×4, omega_f=8, lr=1.2E-4, 320k steps. R²=0.99995, 18.4min.

**Key findings:**
1. F scales excellently to 400f — R²=0.99995 achievable
2. lr ceiling continues to rise with more data (5E-5→8E-5→1.2E-4 at 200→400f)
3. omega_f=8-10 is flat optimum (scaling asymptote reached)
4. 320k steps (800/frame) optimal — 400k overtrains
5. Depth=4 mandatory (3 loses 0.001 R²)

---

## Block 2: Jp@400f siren_txy

### Block 2 Initialization

**Field**: Jp (plastic deformation, 1 component)
**INR type**: siren_txy
**n_training_frames**: 400
**Rationale**: Jp is second-most scalable field. Appendix predicts: omega_f=3-5, lr=1.5E-4, 512×3, 600k steps. Test if lr ceiling continues rising (Jp@200f: 1E-4) and whether omega_f remains in [3-7] flat zone.

### Batch 1 — Initial Configurations (Iterations 9-12)

| Slot | omega_f | lr_NNR_f | hidden_dim | n_layers | total_steps | Dimension tested |
|------|---------|----------|------------|----------|-------------|------------------|
| 00 | 5.0 | 1.5E-4 | 512 | 3 | 600000 | Baseline (appendix Jp@400f reference) |
| 01 | 3.0 | 1.5E-4 | 512 | 3 | 600000 | omega_f (lower, test omega_f=3 vs 5) |
| 02 | 5.0 | 2E-4 | 512 | 3 | 600000 | lr (higher, probe lr ceiling) |
| 03 | 5.0 | 1.5E-4 | 384 | 3 | 600000 | capacity (test 384 speed Pareto) |

### Batch 1 Results (Iterations 9-12)

## Iter 9: excellent
Node: id=9, parent=root
Mode/Strategy: explore/baseline
Config: lr_NNR_f=1.5E-4, total_steps=600000, hidden_dim_nnr_f=512, n_layers_nnr_f=3, omega_f=5.0, batch_size=1
Metrics: final_r2=0.999992, final_mse=2.175E-3, slope=0.9995, kinograph_R2=1.0000, kinograph_SSIM=1.0000, total_params=790529, compression_ratio=4.55, training_time=38.7min
Field: field_name=Jp, inr_type=siren_txy
Mutation: Baseline from appendix (Jp@400f reference: omega_f=5, lr=1.5E-4, 512×3, 600k steps)
Parent rule: Block initialization — direct appendix extrapolation
Visual: GT/Pred match excellent. Scatter tight along diagonal (slope=0.9994). Loss still declining at 600k. Per-frame MSE shows spike around frames 50-100 (early dynamics harder), rest near zero. Spatial patterns well-captured.
Observation: Appendix baseline achieves outstanding R²=0.999992 at 400 frames — Jp scales very well. Loss still declining at 600k suggesting room for improvement. MSE=2.17E-3 dominated by early-frame spike.
Next: parent=11 (highest UCB after analysis)

## Iter 10: excellent
Node: id=10, parent=root
Mode/Strategy: explore/omega_f-scaling
Config: lr_NNR_f=1.5E-4, total_steps=600000, hidden_dim_nnr_f=512, n_layers_nnr_f=3, omega_f=3.0, batch_size=1
Metrics: final_r2=0.999982, final_mse=4.577E-3, slope=1.000, kinograph_R2=1.0000, kinograph_SSIM=1.0000, total_params=790529, compression_ratio=4.55, training_time=38.9min
Field: field_name=Jp, inr_type=siren_txy
Mutation: omega_f: 5.0 -> 3.0 (testing lower omega_f boundary)
Parent rule: Block initialization — test omega_f=3 vs 5 at 400f
Visual: GT/Pred match good but scatter wider than Slot 00, especially at high GT values (Jp>1.2). Loss plateau higher floor. omega_f=3 captures less detail in early dynamic frames.
Observation: omega_f=3 slightly worse than 5 (R²=0.999982 vs 0.999992, MSE 2× higher). slope=1.000 (perfect, slightly better than omega_f=5 slope=0.9995). omega_f=3 retains accuracy but lower frequency capacity costs MSE. omega_f map update: 3(0.999982) < 5(0.999992). omega_f=5 confirmed better for Jp@400f.
Next: parent=11 (highest UCB)

## Iter 11: excellent — **NEW BEST**
Node: id=11, parent=root
Mode/Strategy: explore/lr-ceiling
Config: lr_NNR_f=2E-4, total_steps=600000, hidden_dim_nnr_f=512, n_layers_nnr_f=3, omega_f=5.0, batch_size=1
Metrics: final_r2=0.999996, final_mse=9.680E-4, slope=0.9995, kinograph_R2=1.0000, kinograph_SSIM=1.0000, total_params=790529, compression_ratio=4.55, training_time=39.0min
Field: field_name=Jp, inr_type=siren_txy
Mutation: lr_NNR_f: 1.5E-4 -> 2E-4 (probing lr ceiling at 400f)
Parent rule: Block initialization — test if lr ceiling continues rising
Visual: BEST visual quality. Scatter tightest of all 4 slots. Loss curve shows deepest descent to ~1E-4 floor. Per-frame MSE lowest floor across all frames. GT/Pred nearly identical spatially.
Observation: lr=2E-4 is NEW BEST (R²=0.999996, MSE=9.68E-4 = 2.2× lower than baseline). lr ceiling continues to rise at 400f: Jp@100f(4E-5) → Jp@200f(1E-4) → Jp@400f(≥2E-4). That's 5× increase from 100→400f. Confirms data regularization principle strongly.
Next: parent=11 (highest UCB)

## Iter 12: excellent
Node: id=12, parent=root
Mode/Strategy: principle-test
Config: lr_NNR_f=1.5E-4, total_steps=600000, hidden_dim_nnr_f=384, n_layers_nnr_f=3, omega_f=5.0, batch_size=1
Metrics: final_r2=0.999986, final_mse=3.496E-3, slope=0.9998, kinograph_R2=1.0000, kinograph_SSIM=1.0000, total_params=445441, compression_ratio=8.08, training_time=26.2min
Field: field_name=Jp, inr_type=siren_txy
Mutation: hidden_dim_nnr_f: 512 -> 384 (testing 384 speed Pareto from Jp@100f finding)
Testing principle: "Jp hidden_dim: 384 achieves 99% accuracy at 20% lower training time = SPEED PARETO"
Parent rule: Principle test — verify 384 speed Pareto holds at 400f
Visual: GT/Pred match good. Scatter slightly wider than 512-dim variants. Loss curve higher floor due to reduced capacity. Per-frame MSE spike around frames 50-100 slightly higher. Acceptable quality.
Observation: 384 at 400f: R²=0.999986 vs 512 R²=0.999992 (0.0006% loss) at 32% faster training (26.2 vs 38.7min). Principle CONFIRMED at 400f: 384 is speed Pareto. Compression ratio 8.08 vs 4.55 (78% better). For production use, 384 is strongly competitive.

### Batch 2 Design (Iterations 13-16)

**Strategy**: Exploit from Node 11 (best: R²=0.999996, lr=2E-4). UCB scores all tied at 2.414.
- Slot 00: lr=2.5E-4 — probe lr ceiling further (parent=11)
- Slot 01: omega_f=7 at lr=2E-4 — probe omega_f upper boundary (parent=11)
- Slot 02: 400k steps at lr=2E-4 — test overtraining/speed (parent=11)
- Slot 03: 384×3 at lr=2E-4 — combine 384 speed Pareto with best lr (parent=12, principle-test)

### Batch 2 Results (Iterations 13-16)

## Iter 13: excellent
Node: id=13, parent=11
Mode/Strategy: exploit
Config: lr_NNR_f=2.5E-4, total_steps=600000, hidden_dim_nnr_f=512, n_layers_nnr_f=3, omega_f=5.0, batch_size=1
Metrics: final_r2=0.999850, final_mse=4.425E-2, slope=0.995, kinograph_R2=0.9999, kinograph_SSIM=1.0000, total_params=790529, compression_ratio=4.55, training_time=38.7min
Field: field_name=Jp, inr_type=siren_txy
Mutation: lr_NNR_f: 2E-4 -> 2.5E-4 (probing lr upper ceiling)
Parent rule: Highest UCB node (Node 11, R²=0.999996)
Visual: GT/Pred match good but scatter wider than parent — dispersion at high GT values (Jp>1.2). Per-frame MSE spike at frames 50-100 much larger (peak 1.2 vs parent ~0.06). Loss oscillating at end (overstepping). slope=0.995 (underprediction).
Observation: lr=2.5E-4 OVERSHOOTS — R²=0.99985 vs parent 0.999996. MSE 45× worse (4.4E-2 vs 9.7E-4). slope degrades to 0.995. lr CEILING FOUND: optimal is 2E-4, 2.5E-4 too high. Jp@400f lr map: 1.5E-4(0.999992) < **2E-4(0.999996)** > 2.5E-4(0.999850).
Next: parent=13

## Iter 14: excellent
Node: id=14, parent=11
Mode/Strategy: exploit/omega_f-upper
Config: lr_NNR_f=2E-4, total_steps=600000, hidden_dim_nnr_f=512, n_layers_nnr_f=3, omega_f=7.0, batch_size=1
Metrics: final_r2=0.999947, final_mse=1.381E-2, slope=0.999, kinograph_R2=0.9999, kinograph_SSIM=1.0000, total_params=790529, compression_ratio=4.55, training_time=38.9min
Field: field_name=Jp, inr_type=siren_txy
Mutation: omega_f: 5.0 -> 7.0 (probing omega_f upper boundary at optimal lr)
Parent rule: Exploit from Node 11 — test omega_f upper boundary
Visual: GT/Pred match good. Scatter wider than parent, especially at high GT. Per-frame MSE spike at frames 50-100 broader (peak 0.35 vs parent ~0.06). Loss curve converged but higher floor than parent.
Observation: omega_f=7 WORSE than 5 (R²=0.99995 vs 0.999996, MSE 14× higher). Jp@400f omega_f map: 3(0.999982) < **5(0.999996)** > 7(0.999947). omega_f=5 is LOCAL MAXIMUM. Narrow peak — even ±2 causes significant degradation (unlike F which is flat in [8-10]).
Next: parent=14

## Iter 15: excellent
Node: id=15, parent=11
Mode/Strategy: exploit/step-reduction
Config: lr_NNR_f=2E-4, total_steps=400000, hidden_dim_nnr_f=512, n_layers_nnr_f=3, omega_f=5.0, batch_size=1
Metrics: final_r2=0.999986, final_mse=3.681E-3, slope=0.999, kinograph_R2=1.0000, kinograph_SSIM=1.0000, total_params=790529, compression_ratio=4.55, training_time=26.2min
Field: field_name=Jp, inr_type=siren_txy
Mutation: total_steps: 600000 -> 400000 (test step reduction at lr=2E-4)
Parent rule: Exploit from Node 11 — test if fewer steps match quality
Visual: GT/Pred excellent match. Scatter tight along diagonal. Per-frame MSE spike at frames 50-100 small (peak ~0.02). Loss still declining at 400k but already near floor. R²=1.0000 in scatter.
Observation: 400k steps EXCELLENT at lr=2E-4: R²=0.999986 vs parent's 0.999996 (negligible 0.001% loss). 33% time savings (26.2 vs 39.0min). At lr=2E-4, 1000 steps/frame (400k) nearly matches 1500 steps/frame (600k). NOT overtraining — just sufficient convergence. Speed Pareto: 400k@lr=2E-4 (26.2min, R²=0.999986).
Next: parent=15

## Iter 16: excellent — **BEST EFFICIENCY**
Node: id=16, parent=12
Mode/Strategy: principle-test
Config: lr_NNR_f=2E-4, total_steps=600000, hidden_dim_nnr_f=384, n_layers_nnr_f=3, omega_f=5.0, batch_size=1
Metrics: final_r2=0.999995, final_mse=1.306E-3, slope=0.9999, kinograph_R2=1.0000, kinograph_SSIM=1.0000, total_params=445441, compression_ratio=8.08, training_time=26.3min
Field: field_name=Jp, inr_type=siren_txy
Mutation: lr_NNR_f: 1.5E-4 -> 2E-4 at hidden_dim=384 (combine 384 speed Pareto with best lr)
Testing principle: "Jp hidden_dim: 384 achieves 99% accuracy at 20% lower training time = SPEED PARETO"
Parent rule: Recombine — take 384 from Node 12, lr=2E-4 from Node 11
Visual: BEST efficiency plot. GT/Pred excellent match. Scatter tight, slope=0.9999 (nearly perfect). Per-frame MSE lowest peak (~0.01). Loss converged smoothly to deep floor. 384 + lr=2E-4 is optimal combo.
Observation: 384@lr=2E-4 MATCHES 512@lr=2E-4 accuracy (R²=0.999995 vs 0.999996 — negligible 0.0001% difference) at 33% time savings (26.3 vs 39.0min) and 78% better compression (8.08 vs 4.55). Principle CONFIRMED AND STRENGTHENED: lr=2E-4 closes the 384-512 gap almost entirely. BEST EFFICIENCY config for Jp@400f.
Next: parent=16

### Block 2 Summary

**Jp@400f siren_txy COMPLETE MAP:**
- omega_f: 3(0.999982) < **5(0.999996)** > 7(0.999947). omega_f=5 is LOCAL MAXIMUM (narrow peak, unlike F).
- lr: 1.5E-4(0.999992) < **2E-4(0.999996)** > 2.5E-4(0.999850). lr ceiling at 2E-4.
- steps: 400k(0.999986, 26.2min) < 600k(0.999996, 39.0min). 600k optimal for accuracy, 400k for speed.
- capacity: 384(0.999995, 26.3min) ≈ 512(0.999996, 39.0min). 384 is STRONG speed Pareto.
- **Best accuracy**: 512×3, omega_f=5, lr=2E-4, 600k steps. R²=0.999996, 39.0min.
- **Best efficiency**: 384×3, omega_f=5, lr=2E-4, 600k steps. R²=0.999995, 26.3min.
- **Speed Pareto**: 512×3, omega_f=5, lr=2E-4, 400k steps. R²=0.999986, 26.2min.

**Key findings:**
1. Jp scales to 400f exceptionally — R²=0.999996 (better than predicted 0.998)
2. lr ceiling at 2E-4 (2.5E-4 overshoots). lr scaling: 4E-5(100f) → 1E-4(200f) → 2E-4(400f) = 5× from 100→400f
3. omega_f=5 is LOCAL MAXIMUM (narrow, unlike F's flat [8-10])
4. 384 at lr=2E-4 closes gap with 512 almost entirely — best efficiency config
5. All kinograph metrics saturated at 1.0000 — temporal fidelity perfect

INSTRUCTIONS EDITED: added Jp@400f complete map, omega_f narrow peak rule, lr-data scaling confirmation, 384 speed Pareto strengthened rule

---

## Block 3 Initialization: C field @ 400 frames (parallel)

**Field**: C (APIC matrix, 4 components)
**INR type**: siren_txy
**n_training_frames**: 400
**Rationale**: C is the hardest well-scaling field. Appendix predicts capacity increase needed (640→768→896). Prior maps: C@100f optimal 640×3@omega=25@lr=2E-5 (R²=0.994), C@200f 768×3@omega=20@lr=3E-5 (R²=0.991). C HURTS with more data at constant capacity. Test if capacity scaling to 896+ plus lr increase at 400f can recover.

### Batch 1 — Initial Configurations (4 diverse starting points)

| Slot | omega_f | lr_NNR_f | hidden_dim | n_layers | total_steps | Dimension tested |
|------|---------|----------|------------|----------|-------------|------------------|
| 00 | 18.0 | 4E-5 | 896 | 3 | 1000000 | Baseline (appendix C@400f reference) |
| 01 | 22.0 | 4E-5 | 896 | 3 | 1000000 | omega_f (higher, test if C omega_f stays elevated) |
| 02 | 18.0 | 6E-5 | 896 | 3 | 1000000 | lr (higher, test if lr ceiling rises like F/Jp) |
| 03 | 18.0 | 4E-5 | 768 | 3 | 1000000 | capacity (test if 768 sufficient — principle-test) |

**Design rationale**:
- Slot 00: Direct appendix extrapolation — reference point
- Slot 01: C has historically higher omega_f than Jp/F; test if 22 (between 100f=25 and predicted 18) works better
- Slot 02: F/Jp show lr ceiling rises ~2.5× per 2× frames. C@100f lr=2E-5, C@200f lr=3E-5. Appendix predicts 4E-5; test 6E-5.
- Slot 03: Principle test — "C capacity ceiling increases with n_training_frames" (640→768→896). If 768 suffices at 400f, capacity scaling rule is weaker than predicted.

All slots: siren_txy, batch_size=1, n_training_frames=400, output_size_nnr_f=4, nnr_f_xy_period=1.0, nnr_f_T_period=1.0

### Batch 1 Results (Iterations 17-20)

## Iter 17: excellent — **BEST**
Node: id=17, parent=root
Mode/Strategy: explore/baseline
Config: lr_NNR_f=4E-5, total_steps=1000000, hidden_dim_nnr_f=896, n_layers_nnr_f=3, omega_f=18.0, batch_size=1
Metrics: final_r2=0.999758, final_mse=5.134E-2, slope=0.998, kinograph_R2=0.9998, kinograph_SSIM=1.0000, total_params=2418308, compression_ratio=5.95, training_time=156.1min
Field: field_name=C, inr_type=siren_txy
Mutation: Baseline from appendix (C@400f reference: omega_f=18, lr=4E-5, 896×3, 1000k steps)
Parent rule: Block initialization — direct appendix extrapolation
Visual: GT/Pred match well across all 4 components. Scatter tight along diagonal (slope=0.998). Loss curve still declining at 1M steps — continued training could help. Per-frame MSE shows sharp spike around frame 160-180 (mid-dynamics) and periodic spikes at frames 250-400. Spatial patterns well-captured: disc structures clear in all components.
Observation: Appendix baseline achieves R²=0.9998 — FAR exceeds predicted ~0.99. C@400f is NOT degrading vs lower frame counts (C@100f 0.994, C@200f 0.991). C field BENEFITS from more data at 400f contrary to prior trend. Loss still declining suggesting room for improvement. Training time 156min is very long.
Next: parent=17 (highest UCB, tied — baseline is best)

## Iter 18: excellent
Node: id=18, parent=root
Mode/Strategy: explore/omega_f-upper
Config: lr_NNR_f=4E-5, total_steps=1000000, hidden_dim_nnr_f=896, n_layers_nnr_f=3, omega_f=22.0, batch_size=1
Metrics: final_r2=0.999650, final_mse=7.420E-2, slope=0.998, kinograph_R2=0.9997, kinograph_SSIM=1.0000, total_params=2418308, compression_ratio=5.95, training_time=155.7min
Field: field_name=C, inr_type=siren_txy
Mutation: omega_f: 18.0 -> 22.0 (testing if C omega_f stays elevated like at 100f/200f)
Parent rule: Block initialization — test omega_f upper boundary
Visual: GT/Pred match good but scatter slightly wider than Slot 00. Per-frame MSE spike at frame 160-180 higher (peak ~2.0 vs ~1.5 in Slot 00). Loss curve higher floor. Spatial patterns preserved but slightly less crisp at disc boundaries.
Observation: omega_f=22 WORSE than 18 (R²=0.999650 vs 0.999758, MSE 1.45× higher). C omega_f continues downward trend: 25(100f) → 20(200f) → 18(400f). omega_f=22 overshoots at 400f. C follows ALL-field omega_f decrease with frames.
Next: parent=17 (best in batch)

## Iter 19: excellent
Node: id=19, parent=root
Mode/Strategy: explore/lr-ceiling
Config: lr_NNR_f=6E-5, total_steps=1000000, hidden_dim_nnr_f=896, n_layers_nnr_f=3, omega_f=18.0, batch_size=1
Metrics: final_r2=0.999734, final_mse=5.622E-2, slope=0.998, kinograph_R2=0.9997, kinograph_SSIM=1.0000, total_params=2418308, compression_ratio=5.95, training_time=155.9min
Field: field_name=C, inr_type=siren_txy
Mutation: lr_NNR_f: 4E-5 -> 6E-5 (testing if lr ceiling rises like F/Jp at 400f)
Parent rule: Block initialization — test lr ceiling at 400 frames
Visual: GT/Pred match good. Scatter similar to Slot 00. Per-frame MSE spike at frame 160-180 slightly narrower but similar height. Loss curve converged to similar floor but noisier at end. Spatial patterns well-captured.
Observation: lr=6E-5 marginally worse than 4E-5 (R²=0.999734 vs 0.999758, MSE 1.10× higher). C field does NOT show strong lr ceiling increase with more data (unlike F/Jp 2.5× increase). C@400f lr range: 4E-5 ≥ 6E-5. lr=4E-5 may be near optimal. C lr-data scaling is WEAKER than F/Jp.
Next: parent=17 (best in batch)

## Iter 20: excellent — **SPEED PARETO**
Node: id=20, parent=root
Mode/Strategy: principle-test
Config: lr_NNR_f=4E-5, total_steps=1000000, hidden_dim_nnr_f=768, n_layers_nnr_f=3, omega_f=18.0, batch_size=1
Metrics: final_r2=0.999729, final_mse=5.740E-2, slope=0.998, kinograph_R2=0.9998, kinograph_SSIM=1.0000, total_params=1777924, compression_ratio=8.10, training_time=120.6min
Field: field_name=C, inr_type=siren_txy
Mutation: hidden_dim_nnr_f: 896 -> 768 (testing if 768 sufficient at 400f)
Testing principle: "C capacity ceiling increases with n_training_frames (640→768→896)"
Parent rule: Principle test — if 768 works, capacity scaling is weaker than predicted
Visual: GT/Pred match good, nearly identical to Slot 00. Scatter tight along diagonal. Per-frame MSE spike at frame 160-180 similar magnitude. Loss curve similar trajectory. No visible degradation vs 896.
Observation: 768 NEARLY MATCHES 896 (R²=0.999729 vs 0.999758, only 0.003% loss) at 23% less time (120.6 vs 156.1min) and 36% better compression (8.10 vs 5.95). Principle PARTIALLY CONTRADICTED: C capacity ceiling at 400f may be 768 or even lower, not 896 as predicted. 768 is strong SPEED PARETO.
Next: parent=17 (best in batch)

### Batch 2 Design (Iterations 21-24)

**Strategy**: Exploit from Node 17 (best: R²=0.999758, omega_f=18, lr=4E-5, 896×3, 1M steps). All UCB scores tied at 2.414. Probe boundaries: omega_f lower, lr lower, step reduction, capacity lower limit.
- Slot 00: omega_f=15 at 896×3 — probe omega_f lower boundary (parent=17)
- Slot 01: lr=3E-5 at 896×3 — probe lr lower boundary (parent=17)
- Slot 02: 750k steps at 896×3 — test step reduction for time savings (parent=17)
- Slot 03: 640×3 at omega_f=18, lr=4E-5 — principle-test: "C capacity ceiling at 640@100f" — does 640 work at 400f? (parent=20)

### Batch 2 Results (Iterations 21-24)

## Iter 21: excellent — **NEW BEST**
Node: id=21, parent=17
Mode/Strategy: exploit
Config: lr_NNR_f=4E-5, total_steps=1000000, hidden_dim_nnr_f=896, n_layers_nnr_f=3, omega_f=15.0, batch_size=1
Metrics: final_r2=0.999807, final_mse=4.092E-2, slope=0.998, kinograph_R2=0.9998, kinograph_SSIM=1.0000, total_params=2418308, compression_ratio=5.95, training_time=155.8min
Field: field_name=C, inr_type=siren_txy
Mutation: omega_f: 18.0 -> 15.0 (probing omega_f lower boundary)
Parent rule: Highest UCB node (Node 17, R²=0.999758) — exploit omega_f downward
Visual: GT/Pred match excellent across all 4 components. Scatter tight (R²=0.9998, slope=0.998). Loss curve still declining at 1M steps. Per-frame MSE shows familiar spike at frames 160-180 but lower peak than parent. Disc structures crisp in all spatial panels.
Observation: omega_f=15 is NEW BEST (R²=0.999807, MSE=4.09E-2 vs parent's 5.13E-2 = 20% MSE reduction). C@400f omega_f map update: 15(0.999807) > 18(0.999758) > 22(0.999650). omega_f continues DOWNWARD from 18. C omega_f scaling: 25(100f) → 20(200f) → 15(400f). May go even lower.
Next: parent=21

## Iter 22: excellent
Node: id=22, parent=17
Mode/Strategy: exploit
Config: lr_NNR_f=3E-5, total_steps=1000000, hidden_dim_nnr_f=896, n_layers_nnr_f=3, omega_f=18.0, batch_size=1
Metrics: final_r2=0.999692, final_mse=6.559E-2, slope=0.998, kinograph_R2=0.9997, kinograph_SSIM=1.0000, total_params=2418308, compression_ratio=5.95, training_time=155.7min
Field: field_name=C, inr_type=siren_txy
Mutation: lr_NNR_f: 4E-5 -> 3E-5 (probing lr lower boundary)
Parent rule: Exploit from Node 17 — test lr lower boundary
Visual: GT/Pred match good but scatter marginally wider than parent. Per-frame MSE spike at frame 160-180 higher (peak ~2.0). Loss curve converged to higher floor — insufficient learning rate. Spatial match preserved but fine details slightly blurrier.
Observation: lr=3E-5 WORSE than 4E-5 (R²=0.999692 vs 0.999758, MSE 1.28× higher). C@400f lr map: 3E-5(0.999692) < **4E-5(0.999758)** ≥ 6E-5(0.999734). lr=4E-5 confirmed as near-optimal. Lower lr hurts — insufficient convergence at 1M steps.
Next: parent=21

## Iter 23: excellent
Node: id=23, parent=17
Mode/Strategy: exploit/step-reduction
Config: lr_NNR_f=4E-5, total_steps=750000, hidden_dim_nnr_f=896, n_layers_nnr_f=3, omega_f=18.0, batch_size=1
Metrics: final_r2=0.999521, final_mse=1.020E-1, slope=0.997, kinograph_R2=0.9996, kinograph_SSIM=1.0000, total_params=2418308, compression_ratio=5.95, training_time=117.2min
Field: field_name=C, inr_type=siren_txy
Mutation: total_steps: 1000000 -> 750000 (testing step reduction for time savings)
Parent rule: Exploit from Node 17 — test step reduction at 896×3
Visual: GT/Pred match good. Scatter slightly wider than parent, especially at high GT values. Per-frame MSE shows higher peaks (2.5 vs 1.5 at frame ~180). Loss curve was clearly still descending when cut off at 750k — premature termination. Slope=0.997 (slight underprediction increase).
Observation: 750k steps SIGNIFICANTLY worse than 1M (R²=0.999521 vs 0.999758, MSE 2× higher). C@400f CANNOT reduce steps — loss still declining. 25% step reduction → 25% time savings but 99% MSE increase. C differs from F/Jp: NO overtraining risk at 1M steps/400f. C needs 2500 steps/frame minimum.
Next: parent=21

## Iter 24: excellent
Node: id=24, parent=20
Mode/Strategy: principle-test
Config: lr_NNR_f=4E-5, total_steps=1000000, hidden_dim_nnr_f=640, n_layers_nnr_f=3, omega_f=18.0, batch_size=1
Metrics: final_r2=0.999600, final_mse=8.506E-2, slope=0.997, kinograph_R2=0.9996, kinograph_SSIM=1.0000, total_params=1235844, compression_ratio=11.64, training_time=88.4min
Field: field_name=C, inr_type=siren_txy
Mutation: hidden_dim_nnr_f: 768 -> 640 (testing 640 at 400f)
Testing principle: "C capacity ceiling at 640 for 100 frames"
Parent rule: Principle test from Node 20 (768 speed Pareto) — test if 640 sufficient at 400f
Visual: GT/Pred match good. Scatter wider at high GT values (slope=0.997). Per-frame MSE spike at frames 160-180 notable (peak ~2.0). Loss curve noisier and higher floor than 768/896 variants. Spatial patterns generally captured but fine gradients at disc boundaries less precise.
Observation: 640 WORSE than 768 (R²=0.999600 vs 0.999729, MSE 1.48× higher) but 27% faster (88.4 vs 120.6min). C@400f capacity map: 640(0.999600) < 768(0.999729) < 896(0.999758). Capacity scales monotonically. 640 insufficient at 400f — principle CONFIRMED that C capacity ceiling increases with frames (640@100f → 768-896@400f). But 768 is competitive with 896.
Next: parent=21

### Block 3 Summary

**C@400f siren_txy COMPLETE MAP:**
- omega_f: **15(0.999807)** > 18(0.999758) > 22(0.999650). omega_f=15 is BEST — continues downward trend from 25(100f) → 20(200f) → 15(400f).
- lr: 3E-5(0.999692) < **4E-5(0.999758)** ≥ 6E-5(0.999734). lr=4E-5 near-optimal. C lr-data scaling WEAKER than F/Jp.
- steps: 750k(0.999521) << 1M(0.999758). C needs 2500 steps/frame. No overtraining risk. Loss still declining at 1M.
- capacity: 640(0.999600) < **768(0.999729)** < 896(0.999807). 768 is speed Pareto (0.003% loss, 23% faster). Capacity scales monotonically.
- **Best accuracy**: 896×3, omega_f=15, lr=4E-5, 1M steps. R²=0.999807, 155.8min.
- **Best efficiency**: 768×3, omega_f=18, lr=4E-5, 1M steps. R²=0.999729, 120.6min.
- **Maximum efficiency**: 640×3, omega_f=18, lr=4E-5, 1M steps. R²=0.999600, 88.4min.

**Key findings:**
1. C@400f DRAMATICALLY exceeds predictions — R²=0.9998 vs predicted ~0.99. Prior claim "C HURTS with more data" is WRONG at 400f with sufficient capacity + steps.
2. omega_f continues downward: 25(100f) → 20(200f) → 15(400f). C now follows ALL-field pattern.
3. C lr-data scaling WEAKER than F/Jp — lr=4E-5 optimal (only 2× from 100f's 2E-5, vs F/Jp 2.5× per 2× frames).
4. C needs 2500 steps/frame minimum (no overtraining, loss still declining at 1M). Contrast with F (800 steps/frame, overtrains).
5. Capacity: 768 speed Pareto, 896 for maximum accuracy. Both >> 640.
6. Training time is bottleneck: 155min for 896, 121min for 768, 88min for 640.

INSTRUCTIONS EDITED: added C@400f complete map, omega_f downward trend update, C lr-data scaling weaker rule, C step requirement rule

---

## Block 4: S@400f siren_txy

### Block 4 Initialization

**Field**: S (stress tensor, 4 components)
**INR type**: siren_txy
**n_training_frames**: 400
**Rationale**: S is the hardest field. S@100f peaks at R²=0.729 (no scheduler) or R²=0.998 (with CosineAnnealingLR). At 400f, S faces competing forces. CosineAnnealingLR + gradient clipping are already in code. Testing baseline config + omega_f/lr/capacity dimensions.

### Batch 1 — Initial Configurations (Iterations 25-28)

| Slot | omega_f | lr_NNR_f | hidden_dim | n_layers | total_steps | Dimension tested |
|------|---------|----------|------------|----------|-------------|------------------|
| 00 | 48.0 | 2E-5 | 1280 | 3 | 1000000 | Baseline (S@100f optimal extrapolated to 400f) |
| 01 | 36.0 | 2E-5 | 1280 | 3 | 1000000 | omega_f (lower, test if omega_f decreases at 400f for S) |
| 02 | 48.0 | 3E-5 | 1280 | 3 | 1000000 | lr (higher, test if data regularization allows higher lr for S) |
| 03 | 48.0 | 2E-5 | 1024 | 3 | 1000000 | capacity (lower, principle-test: "S requires 1280 minimum capacity") |

### Batch 1 Results (Iterations 25-28)

## Iter 25: good — **BEST IN BATCH**
Node: id=25, parent=root
Mode/Strategy: explore/baseline
Config: lr_NNR_f=2E-5, total_steps=1000000, hidden_dim_nnr_f=1280, n_layers_nnr_f=3, omega_f=48.0, batch_size=1
Metrics: final_r2=0.960, final_mse=5.951E-9, slope=0.961, kinograph_R2=0.920, kinograph_SSIM=0.927, total_params=4929284, compression_ratio=2.92, training_time=290.1min
Field: field_name=S, inr_type=siren_txy
Mutation: Baseline from S@100f optimal (omega_f=48, lr=2E-5, 1280×3, 1M steps)
Parent rule: Block initialization — direct S@100f extrapolation to 400f
Visual: GT/Pred show reasonable match for all 4 stress components. Scatter has significant dispersion at high GT values with systematic underprediction (slope=0.961). Per-frame MSE shows double-hump pattern peaking frames 150-200 and 250-300. Loss still declining at 1M steps. Spatial patterns captured but less crisp than GT — boundary gradients blurred.
Observation: S@400f baseline R²=0.960 — MUCH better than S@100f without scheduler (0.729) but below S@100f with scheduler (0.998). S@400f shows moderate-to-good quality. The 4× more data at 400f has substantially improved baseline over no-scheduler S@100f (+0.231 R²). Training time 290min is very long. kino_R2=0.920 — temporal fidelity decent but not excellent.
Next: parent=25

## Iter 26: good
Node: id=26, parent=root
Mode/Strategy: explore/omega_f-scaling
Config: lr_NNR_f=2E-5, total_steps=1000000, hidden_dim_nnr_f=1280, n_layers_nnr_f=3, omega_f=36.0, batch_size=1
Metrics: final_r2=0.949, final_mse=7.706E-9, slope=0.949, kinograph_R2=0.898, kinograph_SSIM=0.911, total_params=4929284, compression_ratio=2.92, training_time=289.8min
Field: field_name=S, inr_type=siren_txy
Mutation: omega_f: 48.0 -> 36.0 (testing if omega_f decreases at 400f for S)
Parent rule: Block initialization — test omega_f lower boundary
Visual: GT/Pred match slightly worse than Slot 00 — scatter wider especially at high GT. Per-frame MSE has similar double-hump pattern but higher overall magnitude. Loss curve converged to higher floor than Slot 00. Spatial patterns captured but with more blurring at stress concentrations.
Observation: omega_f=36 WORSE than 48 (R²=0.949 vs 0.960). S COUNTER-TREND CONFIRMED at 400f: S does NOT follow the all-field omega_f decrease with more frames. omega_f=48 remains optimal for S at 400f (unchanged from 100f). S@400f omega_f map: 36(0.949) < 48(0.960). High-complexity fields maintain omega_f.
Next: parent=25

## Iter 27: moderate
Node: id=27, parent=root
Mode/Strategy: explore/lr-ceiling
Config: lr_NNR_f=3E-5, total_steps=1000000, hidden_dim_nnr_f=1280, n_layers_nnr_f=3, omega_f=48.0, batch_size=1
Metrics: final_r2=0.803, final_mse=2.951E-8, slope=0.806, kinograph_R2=0.470, kinograph_SSIM=0.681, total_params=4929284, compression_ratio=2.92, training_time=290.0min
Field: field_name=S, inr_type=siren_txy
Mutation: lr_NNR_f: 2E-5 -> 3E-5 (testing if data regularization allows higher lr for S)
Parent rule: Block initialization — test lr ceiling at 400 frames
Visual: Per-frame MSE dramatically higher and noisier (sustained high across frames 100-350 vs peaked pattern in Slot 00). Scatter very wide — major dispersion at high GT values. Significant underprediction (slope=0.806). Spatial patterns distorted — stress concentrations blurred/smeared. kino_R2=0.470 — temporal structure substantially lost.
Observation: lr=3E-5 CATASTROPHIC for S (R²=0.803, MSE 5× worse). S lr=2E-5 is HARD-LOCKED — even 50% increase causes massive degradation. Data regularization does NOT help S lr tolerance at 400f. S is fundamentally different from F/Jp/C in lr sensitivity. kino_R2=0.470 confirms temporal fidelity destroyed.
Next: parent=25

## Iter 28: good
Node: id=28, parent=root
Mode/Strategy: principle-test
Config: lr_NNR_f=2E-5, total_steps=1000000, hidden_dim_nnr_f=1024, n_layers_nnr_f=3, omega_f=48.0, batch_size=1
Metrics: final_r2=0.918, final_mse=1.235E-8, slope=0.917, kinograph_R2=0.831, kinograph_SSIM=0.865, total_params=3156996, compression_ratio=4.56, training_time=189.6min
Field: field_name=S, inr_type=siren_txy
Mutation: hidden_dim_nnr_f: 1280 -> 1024 (testing if S can reduce capacity)
Testing principle: "S requires 1280 minimum capacity"
Parent rule: Principle test — challenge S capacity minimum at 400f
Visual: GT/Pred match decent but noticeably less crisp than Slot 00. Scatter wider with more dispersion. Per-frame MSE double-hump pattern higher than Slot 00 but lower/sharper than Slot 02. Spatial patterns present but stress gradients smoothed. kino_R2=0.831 — temporal structure partially preserved.
Observation: 1024 WORSE than 1280 (R²=0.918 vs 0.960, 4.4% R² loss) but 35% faster (189.6 vs 290.1min). Principle CONFIRMED: S requires 1280 capacity at 400f. 1024 loses too much (0.042 R²). S@400f capacity map: 1024(0.918) << 1280(0.960). Unlike C where 768 is competitive with 896, S has steep capacity dependence.
Next: parent=25

### Batch 2 Design (Iterations 29-32)

**Strategy**: Exploit from Node 25 (best: R²=0.960, omega_f=48, lr=2E-5, 1280×3, 1M steps). Key findings from Batch 1: omega_f=48 confirmed, lr=2E-5 hard-locked, 1280 required. Remaining dimensions to explore: omega_f upward (55), more steps (1.5M), and try siren_t architecture.
- Slot 00: omega_f=55 — probe omega_f upper boundary at 400f (parent=25)
- Slot 01: total_steps=1500000 — test more training (loss still declining at 1M) (parent=25)
- Slot 02: lr=1.5E-5 — probe lr lower boundary (parent=25)
- Slot 03: omega_f=42 — probe omega_f between 36 and 48 to refine map (parent=25, principle-test: "S omega_f COUNTER-TREND: high-complexity fields maintain omega_f")

