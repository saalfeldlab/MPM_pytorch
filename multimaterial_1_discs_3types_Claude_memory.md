# Working Memory: multimaterial_1_discs_3types (parallel)

## Knowledge Base (accumulated across all blocks)

### Regime Comparison Table
| Block | INR Type | Field | n_training_frames | Best R² | Best slope | kino_R2 | kino_SSIM | Optimal lr_NNR_f | Optimal hidden_dim | Optimal n_layers | Optimal omega_f | Optimal total_steps | Training time (min) | Key finding |
|-------|----------|-------|-------------------|---------|------------|---------|-----------|------------------|--------------------|--------------------|-----------------|---------------------|---------------------|-------------|
| 1 | siren_txy | F | 400 | 0.99995 | 0.9999 | 0.9999 | 0.9962 | 1.2E-4 | 256 | 4 | 8.0 | 320000 | 18.4 | lr=1.2E-4 best; omega_f=[8-10] flat; depth=4 required; 400k overtrains |
| 2 | siren_txy | Jp | 400 | 0.999996 | 0.9995 | 1.0000 | 1.0000 | 2E-4 | 512 (384 speed) | 3 | 5.0 | 600000 (400k speed) | 39.0 (26.3 speed) | lr ceiling 2E-4; omega_f=5 narrow peak; 384 near-equal; all R²>0.99985 |
| 3 | siren_txy | C | 400 | 0.999807 | 0.998 | 0.9998 | 1.0000 | 4E-5 | 896 (768 speed) | 3 | 15.0 | 1000000 | 155.8 (120.6 speed) | omega_f=15 best; lr=4E-5 optimal; 768 speed Pareto; C REVERSES degradation trend at 400f |
| 4 | siren_txy | S | 400 | 0.970 | 0.970 | 0.940 | 0.942 | 2E-5 | 1280 | 3 | 55.0 | 1000000 | 290.0 | omega_f INCREASES for S (counter-trend); lr hard-locked [1.5-2]E-5; 1.5M overtrains; 1280 required |
| 5 | siren_txy | F | 600 | 0.999983 | 1.0000 | 1.0000 | 0.9987 | 1.8E-4 | 256 | 4 | 10.0 | 420000 | 24.1 | omega_f-lr INTERACTION (10>8 at high lr); lr ceiling NOT found at 2.2E-4; POSITIVE data scaling (exceeds 400f) |
| 6 | siren_txy | Jp | 600 | 0.999955 | 0.9986 | 1.0000 | 1.0000 | 2.5E-4 | 384 | 3 | 4.0 | 720000 | 31.7 | omega_f=[3-4] FLAT (peak BROADENED from 400f); 384 BEATS 512 unconditionally; recombine success; speed Pareto 23.9min |

### Established Principles
- From appendix: F is most scalable field (no diminishing returns to 500f) — **CONFIRMED to 600f** (R²=0.999983 > 0.99995@400f)
- F capacity ceiling at 256 (384 HURTS) — holds at 400f+
- F depth ceiling at 4 layers (siren_txy) — holds at 400f+ (depth=3 loses 0.001 R²)
- Period parameters must stay at 1.0 for F (and likely all fields)
- Data regularization allows higher lr at higher n_training_frames: F lr 5E-5(200f) → 1.2E-4(400f) → 1.8E-4(600f); Jp lr 4E-5(100f) → 1E-4(200f) → 2E-4(400f) → 2.5E-4(600f). ~2.5× per 2× frames. **Exception: C only ~2× per 4× frames. S does NOT benefit at all.**
- F overtrains at >700-800 steps/frame: 400f optimal 800/f, 600f optimal 700/f. Steps/frame DECREASES slightly with more frames.
- **omega_f-lr INTERACTION (Block 5 finding)**: F omega_f=10 beats 8 at high lr (1.8E-4), but 8≈10 at moderate lr (1.2E-4). Higher lr unlocks higher frequency capacity. omega_f "plateau" is lr-dependent.
- F@600f lr tolerance is VERY wide: [1.2-2.2]E-4 all viable. lr ceiling still NOT found at 2.2E-4.
- Jp 384 speed Pareto holds at 400f, 600f — **unconditionally beats 512** at every tested config at 600f. 384 is the default for Jp.
- Both F and Jp achieve R²>0.9999 at 400+ frames — confirming strong data scalability for low-complexity fields
- omega_f-to-frames scaling DIVERGES by field: F/Jp/C DECREASE omega_f with more frames. **S INCREASES omega_f** (48@100f → 55@400f). S is the ONLY upward-trending field.
- **omega_f peak BROADENS with more data (Block 6 finding)**: Jp@400f narrow peak at 5 becomes FLAT [3-4] at 600f. omega_f=3 at 600f matches omega=4 (0.99995 ≈ 0.99996). More data makes omega_f LESS sensitive.
- C@400f REVERSES prior degradation trend: 0.994(100f) → 0.991(200f) → 0.9998(400f). Prior claim "C HURTS with more data" is WRONG at 400f with sufficient capacity + steps.
- C lr-data scaling WEAKER than F/Jp: lr=2E-5(100f) → 4E-5(400f) = only 2× increase over 4× frames.
- C needs 2500 steps/frame minimum at 400f. No overtraining risk. Contrast with F (700-800 steps/frame, overtrains beyond).
- C capacity scales monotonically at 400f: 640(0.9996) < 768(0.9997) < 896(0.9998). 768 is speed Pareto.
- S@400f lr HARD-LOCKED at [1.5-2]E-5: lr=3E-5 catastrophic (R²=0.803). S does NOT benefit from data-regularized lr increase.
- S@400f 1M steps OPTIMAL: 1.5M overtrains due to CosineAnnealingLR reaching near-zero lr at T_max.
- S@400f capacity REQUIRED 1280: 1024 loses 4.4% R² (0.918 vs 0.960). Steep capacity dependence.
- S@400f omega_f map: 36(0.949) < 42(0.952) < 48(0.960) < 55(0.970). Nearly linear upward — NOT saturated.
- S@400f achieves R²=0.970 with scheduler+clipping — data scaling HELPS S enormously (+0.241 vs S@100f no-scheduler).
- **Jp@600f steps/frame = 1200/f optimal** (higher than F@600f at 700/f). Jp is more step-hungry than F at same frame count.
- **Recombination additive when dimensions are orthogonal** (Block 6 finding): omega_f + capacity = additive. omega_f + lr = NOT additive. Orthogonal dimensions (param-affecting vs arch-affecting) combine better.

### Open Questions
- S@400f omega_f: 55 best but trend not saturated — would 60 or 65 improve further?
- Does siren_t vs siren_txy matter for Jp@400f or F@400f? (siren_t dominates at 100f)
- siren_t for C@400f: siren_t gave 0.9999 at 100f. Would it help at 400f?
- C@600f: unexplored. Expect capacity needs increase further (896-1024?). omega_f~12? lr~5E-5?
- S@600f: unexplored. omega_f may rise further (60+?). Training time prohibitive (~400min?).
- F@600f lr ceiling: 2.2E-4 works — would 2.5E-4 or 3E-4 still hold?
- omega_f-lr interaction: Does this apply to other fields or only F?
- Jp@600f: lr NOT probed above 2.5E-4 at 384. Could 3E-4@384 beat 2.5E-4@384?
- Does omega_f broadening apply to F/C/S at 600f+, or only Jp?

---

## Previous Block Summary (Block 6)

Block 6: Jp@600f siren_txy, 8 iterations. Jp scales excellently to 600f — R²=0.999955 with 384×3@omega=4@lr=2.5E-4@720k, 31.7min. omega_f peak BROADENED: 400f narrow 5 → 600f flat [3-4]. 384 UNCONDITIONALLY beats 512. Recombine (omega=4+384) was additive. Speed Pareto: 540k@384, R²=0.99991, 23.9min.

---

## Current Block (Block 7)

### Block Info
Field: field_name=C, inr_type=siren_txy, n_training_frames=600
Parallel mode: 4 slots exploring different parameter dimensions simultaneously
Iterations: 49 to 56

### Hypothesis
C is the third field to test at 600f. C@400f achieved R²=0.9998 with 896×3@omega=15@lr=4E-5@1M steps (768 speed Pareto). At 600f, based on scaling rules:
- omega_f: C follows downward trend with more frames: 25(100f) → 20(200f) → 15(400f). At 600f, expect omega_f~12 (continued downward).
- lr: C lr-data scaling is WEAK (~2× per 4× frames): 2E-5(100f) → 4E-5(400f). At 600f, expect lr~5E-5 (slight increase, 1.25× for 1.5× frames).
- steps: C needs 2500 steps/frame minimum, no overtraining risk. At 600f, expect 1.5M steps (2500/f × 600f).
- capacity: C capacity scales monotonically at 400f: 640 < 768 < 896. At 600f, expect 896 baseline, possibly 1024 needed.
- output_size_nnr_f: 4 (C has 4 components)
Prediction: C@600f should achieve R²≥0.9998 with proper tuning. Training time ~160-200min for 896.

### Planned Initial Configurations (Batch 1)

| Slot | omega_f | lr_NNR_f | hidden_dim | n_layers | total_steps | Mutation dimension |
|------|---------|----------|------------|----------|-------------|--------------------|
| 00 | 12.0 | 5E-5 | 896 | 3 | 1500000 | **Baseline** (C@400f optimal, extrapolated to 600f) |
| 01 | 15.0 | 5E-5 | 896 | 3 | 1500000 | **omega_f** (keep 400f optimal, test if C follows downward trend less than F/Jp) |
| 02 | 12.0 | 4E-5 | 896 | 3 | 1500000 | **lr** (C@400f optimal lr, test if increase helps) |
| 03 | 12.0 | 5E-5 | 768 | 3 | 1500000 | **capacity** (principle-test: "C capacity scales monotonically at 400f") |

All slots: siren_txy, batch_size=1, n_training_frames=600, output_size_nnr_f=4, nnr_f_xy_period=1.0, nnr_f_T_period=1.0

### Iterations This Block

## Iter 49: excellent — **BEST OF BATCH** (baseline)
Node: id=49, parent=root
Mode/Strategy: exploit/baseline
Config: lr_NNR_f=5E-5, total_steps=1500000, hidden_dim_nnr_f=896, n_layers_nnr_f=3, omega_f=12.0, batch_size=1
Metrics: final_r2=0.999866, final_mse=2.727E-2, slope=0.999, kinograph_R2=0.9999, kinograph_SSIM=1.0000, total_params=2418308, compression_ratio=8.9, training_time=239.8min
Field: field_name=C, inr_type=siren_txy
Mutation: Baseline — C@400f optimal extrapolated to 600f
Visual: GT/Pred match excellent all components. Loss still declining at 1.5M. Per-frame MSE spikes at frames 250-400.
Observation: C@600f baseline R²=0.999866 — BEST of batch. Data scaling upward trend continues: 0.9998(400f) → 0.9999(600f).
Next: parent=49

## Iter 50: excellent
Node: id=50, parent=root
Mode/Strategy: explore/omega_f-comparison
Config: lr_NNR_f=5E-5, total_steps=1500000, hidden_dim_nnr_f=896, n_layers_nnr_f=3, omega_f=15.0, batch_size=1
Metrics: final_r2=0.999849, final_mse=3.076E-2, slope=0.999, kinograph_R2=0.9999, kinograph_SSIM=1.0000, total_params=2418308, compression_ratio=8.9, training_time=233.7min
Field: field_name=C, inr_type=siren_txy
Mutation: omega_f: 12.0 -> 15.0
Observation: omega_f=15 WORSE than 12. C omega_f downward trend CONFIRMED: 25→20→15→12. ~3-5 decrease per frame doubling.
Next: parent=49

## Iter 51: excellent
Node: id=51, parent=root
Mode/Strategy: explore/lr-comparison
Config: lr_NNR_f=4E-5, total_steps=1500000, hidden_dim_nnr_f=896, n_layers_nnr_f=3, omega_f=12.0, batch_size=1
Metrics: final_r2=0.999854, final_mse=2.978E-2, slope=0.999, kinograph_R2=0.9999, kinograph_SSIM=1.0000, total_params=2418308, compression_ratio=8.9, training_time=234.2min
Field: field_name=C, inr_type=siren_txy
Mutation: lr_NNR_f: 5E-5 -> 4E-5
Observation: lr=4E-5 marginally worse than 5E-5. C lr-data scaling WEAK but present: 4E-5(400f) → 5E-5(600f).
Next: parent=49

## Iter 52: excellent — speed Pareto candidate
Node: id=52, parent=root
Mode/Strategy: principle-test
Config: lr_NNR_f=5E-5, total_steps=1500000, hidden_dim_nnr_f=768, n_layers_nnr_f=3, omega_f=12.0, batch_size=1
Metrics: final_r2=0.999818, final_mse=3.716E-2, slope=0.999, kinograph_R2=0.9998, kinograph_SSIM=1.0000, total_params=1777924, compression_ratio=12.1, training_time=182.0min
Field: field_name=C, inr_type=siren_txy
Mutation: hidden_dim_nnr_f: 896 -> 768. Testing principle: "C capacity scales monotonically at 400f"
Observation: 768 loses 0.005% R² but saves 24% time. Capacity monotonicity CONFIRMED at 600f. 768=speed Pareto.
Next: parent=49

### Emerging Observations
- C@600f ALL 4 slots excellent (R²>0.9998). Baseline extrapolation was accurate.
- omega_f=12 CONFIRMED better than 15 — C follows downward trend: 25→20→15→12. Next probe: omega_f=10?
- lr=5E-5 marginally better than 4E-5 — C lr-data scaling is WEAK (1.25× from 400→600f). Next probe: lr=6E-5?
- Capacity monotonic: 896 > 768 confirmed at 600f. Next probe: 1024 for accuracy ceiling? Or accept 896 as optimal?
- Loss still declining at 1.5M steps (2500/f) — C may benefit from more steps. Test 2M (3333/f)?
- Training time ~240min for 896 — much longer than F(24min) or Jp(32min) at 600f. C is compute-expensive.
- All slopes ~0.999 — no underprediction bias. All kino_SSIM=1.0000 — perfect structural match.
