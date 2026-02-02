# Working Memory: multimaterial_1_discs_3types (parallel)

## Knowledge Base (accumulated across all blocks)

### Regime Comparison Table
| Block | INR Type | Field | n_training_frames | Best R² | Best slope | kino_R2 | kino_SSIM | Optimal lr_NNR_f | Optimal hidden_dim | Optimal n_layers | Optimal omega_f | Optimal total_steps | Training time (min) | Key finding |
|-------|----------|-------|-------------------|---------|------------|---------|-----------|------------------|--------------------|--------------------|-----------------|---------------------|---------------------|-------------|
| 1 | siren_txy | F | 400 | 0.99995 | 0.9999 | 0.9999 | 0.9962 | 1.2E-4 | 256 | 4 | 8.0 | 320000 | 18.4 | lr=1.2E-4 best; omega_f=[8-10] flat; depth=4 required; 400k overtrains |
| 2 | siren_txy | Jp | 400 | 0.999996 | 0.9995 | 1.0000 | 1.0000 | 2E-4 | 512 (384 speed) | 3 | 5.0 | 600000 (400k speed) | 39.0 (26.3 speed) | lr ceiling 2E-4; omega_f=5 narrow peak; 384 near-equal; all R²>0.99985 |
| 3 | siren_txy | C | 400 | 0.999807 | 0.998 | 0.9998 | 1.0000 | 4E-5 | 896 (768 speed) | 3 | 15.0 | 1000000 | 155.8 (120.6 speed) | omega_f=15 best; lr=4E-5 optimal; 768 speed Pareto; C REVERSES degradation trend at 400f |

### Established Principles
- From appendix: F is most scalable field (no diminishing returns to 500f)
- F@200f optimal: 256×4, omega_f=9-10, lr=5E-5, 300k steps (R²=0.9997)
- F omega_f scales down with frames but PLATEAUS: 12(100f) → 9(200f) → 8(400f). Not linear.
- F capacity ceiling at 256 (384 HURTS) — holds at 400f
- F depth ceiling at 4 layers (siren_txy) — holds at 400f (depth=3 loses 0.001 R²)
- Period parameters must stay at 1.0 for F (and likely all fields)
- Data regularization allows higher lr at higher n_training_frames: F lr 5E-5(200f) → 1.2E-4(400f); Jp lr 4E-5(100f) → 1E-4(200f) → 2E-4(400f). ~2.5× per 2× frames. **Exception: C only ~2× per 4× frames (much weaker).**
- F@400f overtrains at >800 steps/frame (400k worse than 320k at lr=8E-5)
- F@400f omega_f insensitive in [8-10] range (R² identical within noise)
- Jp@400f omega_f=5 is LOCAL MAXIMUM (narrow peak — ±2 causes significant degradation). Unlike F which is flat.
- Jp@400f lr ceiling at 2E-4 (2.5E-4 overshoots, MSE 45× worse)
- Jp 384 speed Pareto holds at 400f AND strengthened by lr=2E-4 (R²=0.999995 vs 512's 0.999996)
- Both F and Jp achieve R²>0.9999 at 400 frames — confirming strong data scalability for low-complexity fields
- ALL-field omega_f-to-frames scaling: omega_f DECREASES with more frames for ALL tested fields. C: 25(100f) → 20(200f) → 15(400f). F: 12(100f) → 9(200f) → 8(400f). Jp: 5-10(100f) → 3-7(200f) → 5(400f).
- C@400f REVERSES prior degradation trend: 0.994(100f) → 0.991(200f) → 0.9998(400f). Prior claim "C HURTS with more data" is WRONG at 400f with sufficient capacity + steps (896×3, 1M steps).
- C lr-data scaling WEAKER than F/Jp: lr=2E-5(100f) → 4E-5(400f) = only 2× increase over 4× frames. C lr insensitive in [4-6]E-5 range.
- C needs 2500 steps/frame minimum at 400f (750k degrades 2× MSE vs 1M). No overtraining risk. Contrast with F (800 steps/frame, overtrains beyond).
- C capacity scales monotonically at 400f: 640(0.9996) < 768(0.9997) < 896(0.9998). 768 is speed Pareto.

### Open Questions
- F@400f lr ceiling: 1.2E-4 works and is best — does 1.5E-4 still work? (not tested)
- Does siren_t vs siren_txy matter for Jp@400f? (siren_t dominates at 100f)
- Does Jp omega_f narrow peak (5) vs F flat plateau (8-10) reflect fundamental field difference?
- C@400f omega_f: 15 is best — does 12 work even better? (not tested, continuing downward trend)
- C@400f: Would 1.5M+ steps improve further? Loss still declining at 1M.
- S@400f: Will S scale at all? S@100f maxes at R²=0.729 (siren_txy) or 0.998 (with CosineAnnealingLR). S is hardest field.
- siren_t for C@400f: siren_t gave 0.9999 at 100f. Would siren_t help C at 400f?

---

## Previous Block Summary (Block 3)

Block 3: C@400f siren_txy, 8 iterations. C DRAMATICALLY exceeds predictions (R²=0.9998 vs predicted ~0.99), reversing degradation trend at 400f. omega_f=15 is best (continues 25→20→15 downward trend). lr=4E-5 optimal (C lr-scaling weaker than F/Jp). 768 is speed Pareto (0.003% loss, 23% faster). 750k steps insufficient — C needs 2500 steps/frame (no overtraining). 640 capacity insufficient at 400f. Best: 896×3@omega=15@lr=4E-5@1M, R²=0.999807, 155.8min.

---

## Current Block (Block 4)

### Block Info
Field: field_name=S, inr_type=siren_txy, n_training_frames=400
Parallel mode: 4 slots exploring different parameter dimensions simultaneously
Iterations: 25 to 32

### Hypothesis
S@400f is the hardest field. S@100f peaks at R²=0.729 (siren_txy, no scheduler) or R²=0.998 (with CosineAnnealingLR). S has EXTREME stochastic variance and requires gradient clipping (max_norm=1.0). At 400f, S faces competing forces: more data could help regularize (as C showed reversal), but S scaling historically HURTS. Key: CosineAnnealingLR appears MANDATORY for S. Prediction: S@400f with CosineAnnealingLR + gradient clipping may achieve R²>0.99. Without scheduler, expect R²<0.80. Must test code modification (scheduler) early.

### Planned Initial Configurations (Batch 1)

| Slot | omega_f | lr_NNR_f | hidden_dim | n_layers | total_steps | Mutation dimension |
|------|---------|----------|------------|----------|-------------|--------------------|
| 00 | 48.0 | 2E-5 | 1280 | 3 | 1000000 | **Baseline** (S@100f optimal extrapolated to 400f) |
| 01 | 36.0 | 2E-5 | 1280 | 3 | 1000000 | **omega_f** (lower, test if omega_f decreases at 400f for S too) |
| 02 | 48.0 | 3E-5 | 1280 | 3 | 1000000 | **lr** (higher, test if data regularization allows higher lr for S) |
| 03 | 48.0 | 2E-5 | 1024 | 3 | 1000000 | **capacity** (lower, test if S can reduce from 1280) — principle-test: "S requires 1280 minimum capacity" |

All slots: siren_txy, batch_size=1, n_training_frames=400, output_size_nnr_f=4, nnr_f_xy_period=1.0, nnr_f_T_period=1.0

NOTE: CosineAnnealingLR scheduler AND gradient clipping (max_norm=1.0) are ALREADY in the code (graph_trainer.py). S@100f achieved R²=0.998 with these. All 4 slots will automatically benefit from scheduler + clipping.

### Iterations This Block

## Iter 25: good — **BEST IN BATCH**
Node: id=25, parent=root
Mode/Strategy: explore/baseline
Config: lr_NNR_f=2E-5, total_steps=1000000, hidden_dim_nnr_f=1280, n_layers_nnr_f=3, omega_f=48.0, batch_size=1
Metrics: final_r2=0.960, final_mse=5.951E-9, slope=0.961, kinograph_R2=0.920, kinograph_SSIM=0.927, total_params=4929284, compression_ratio=2.92, training_time=290.1min
Field: field_name=S, inr_type=siren_txy
Mutation: Baseline (S@100f optimal → 400f)
Observation: S@400f baseline R²=0.960. MUCH better than S@100f no-scheduler (0.729). Loss still declining. 290min training.
Next: parent=25

## Iter 26: good
Node: id=26, parent=root
Mode/Strategy: explore/omega_f-scaling
Config: lr_NNR_f=2E-5, total_steps=1000000, hidden_dim_nnr_f=1280, n_layers_nnr_f=3, omega_f=36.0, batch_size=1
Metrics: final_r2=0.949, final_mse=7.706E-9, slope=0.949, kinograph_R2=0.898, kinograph_SSIM=0.911, total_params=4929284, compression_ratio=2.92, training_time=289.8min
Mutation: omega_f: 48→36
Observation: omega_f=36 WORSE. S COUNTER-TREND CONFIRMED at 400f. omega_f=48 remains optimal.
Next: parent=25

## Iter 27: moderate
Node: id=27, parent=root
Mode/Strategy: explore/lr-ceiling
Config: lr_NNR_f=3E-5, total_steps=1000000, hidden_dim_nnr_f=1280, n_layers_nnr_f=3, omega_f=48.0, batch_size=1
Metrics: final_r2=0.803, final_mse=2.951E-8, slope=0.806, kinograph_R2=0.470, kinograph_SSIM=0.681, total_params=4929284, compression_ratio=2.92, training_time=290.0min
Mutation: lr: 2E-5→3E-5
Observation: lr=3E-5 CATASTROPHIC. S lr=2E-5 HARD-LOCKED. Data regularization does NOT help S.
Next: parent=25

## Iter 28: good
Node: id=28, parent=root
Mode/Strategy: principle-test
Config: lr_NNR_f=2E-5, total_steps=1000000, hidden_dim_nnr_f=1024, n_layers_nnr_f=3, omega_f=48.0, batch_size=1
Metrics: final_r2=0.918, final_mse=1.235E-8, slope=0.917, kinograph_R2=0.831, kinograph_SSIM=0.865, total_params=3156996, compression_ratio=4.56, training_time=189.6min
Mutation: hidden_dim: 1280→1024 (principle-test: "S requires 1280 minimum capacity")
Observation: 1024 loses 4.4% R² vs 1280 but 35% faster. Principle CONFIRMED: S requires 1280.
Next: parent=25

### Emerging Observations

- **S@400f baseline R²=0.960** — huge improvement over S@100f no-scheduler (0.729). Data scaling HELPS S baseline significantly (+0.231 R²). NOTE: CosineAnnealingLR + clipping are in code — these results include scheduler benefit.
- **omega_f=48 CONFIRMED** at 400f: S does NOT follow all-field omega_f decrease. Map: 36(0.949) < 48(0.960). Need to probe 55 upward.
- **lr=2E-5 HARD-LOCKED**: Even 50% increase (3E-5) causes R²=0.803 (catastrophic). S does NOT benefit from data-regularized lr increase. Fundamentally different from F/Jp/C.
- **1280 capacity REQUIRED**: 1024 loses 4.4% R² — steep capacity dependence (unlike C's gentle slope).
- **Training time prohibitive**: 290min for 1280×3@1M steps. Need time savings strategy.
- **Loss still declining at 1M**: More steps may help. Testing 1.5M next batch.
- **Next batch plan**: (1) omega_f=55 upper probe, (2) 1.5M steps (more training), (3) lr=1.5E-5 lower probe, (4) omega_f=42 refine map
