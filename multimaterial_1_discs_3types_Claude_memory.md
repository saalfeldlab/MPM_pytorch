# Working Memory: multimaterial_1_discs_3types (parallel)

## Knowledge Base (accumulated across all blocks)

### Regime Comparison Table
| Block | INR Type | Field | n_training_frames | Best R² | Best slope | kino_R2 | kino_SSIM | Optimal lr_NNR_f | Optimal hidden_dim | Optimal n_layers | Optimal omega_f | Optimal total_steps | Training time (min) | Key finding |
|-------|----------|-------|-------------------|---------|------------|---------|-----------|------------------|--------------------|--------------------|-----------------|---------------------|---------------------|-------------|
| 1 | siren_txy | F | 400 | 0.99995 | 0.9999 | 0.9999 | 0.9962 | 1.2E-4 | 256 | 4 | 8.0 | 320000 | 18.4 | lr=1.2E-4 best; omega_f=[8-10] flat; depth=4 required; 400k overtrains |

### Established Principles
- From appendix: F is most scalable field (no diminishing returns to 500f)
- F@200f optimal: 256×4, omega_f=9-10, lr=5E-5, 300k steps (R²=0.9997)
- F omega_f scales down with frames but PLATEAUS: 12(100f) → 9(200f) → 8(400f). Not linear.
- F capacity ceiling at 256 (384 HURTS) — holds at 400f
- F depth ceiling at 4 layers (siren_txy) — holds at 400f (depth=3 loses 0.001 R²)
- Period parameters must stay at 1.0 for F
- Data regularization allows higher lr at higher n_training_frames: F lr 5E-5(200f) → 1.2E-4(400f) = 2.4× increase
- F@400f overtrains at >800 steps/frame (400k worse than 320k at lr=8E-5)
- F@400f omega_f insensitive in [8-10] range (R² identical within noise)

### Open Questions
- F@400f lr ceiling: 1.2E-4 works and is best — does 1.5E-4 still work? (not tested, leave for future)
- Jp@400f: appendix predicts omega_f=3-5, lr=1.5E-4, 512×3, 600k steps. Will test Block 2.
- Does Jp lr ceiling also rise dramatically at 400f? (Jp@200f: 1E-4 optimal)
- Does siren_t vs siren_txy matter for Jp@400f? (siren_t dominates at 100f)

---

## Previous Block Summary (Block 1)

Block 1: F@400f siren_txy, 8 iterations. F scales excellently to 400 frames (R²=0.99995). lr=1.2E-4 best (data regularization confirmed, 2.4× vs 200f). omega_f=[8-10] flat optimum (scaling plateau reached). Depth=4 mandatory, 320k steps optimal (400k overtrains).

---

## Current Block (Block 2)

### Block Info
Field: field_name=Jp, inr_type=siren_txy, n_training_frames=400
Parallel mode: 4 slots exploring different parameter dimensions simultaneously
Iterations: 9 to 16

### Hypothesis
Jp should scale to 400f with R²>0.997 based on extrapolation. Prior maps: Jp@100f optimal omega_f=[5-10], lr=4E-5 (siren_txy). Jp@200f optimal omega_f=[3-7] (flat), lr=1E-4. Appendix predicts 400f: omega_f=3-5, lr=1.5E-4, 512×3, 600k steps. Key test: does lr ceiling continue rising (1E-4→1.5E-4+)? Does omega_f remain flat at 3-5? Use siren_txy first (consistent with F block), test siren_t later if time permits.

### Planned Initial Configurations (Batch 1)

| Slot | omega_f | lr_NNR_f | hidden_dim | n_layers | total_steps | Mutation dimension |
|------|---------|----------|------------|----------|-------------|--------------------|
| 00 | 5.0 | 1.5E-4 | 512 | 3 | 600000 | **Baseline** (appendix Jp@400f reference) |
| 01 | 3.0 | 1.5E-4 | 512 | 3 | 600000 | **omega_f** (lower, test omega_f=3 vs 5) |
| 02 | 5.0 | 2E-4 | 512 | 3 | 600000 | **lr** (higher, probe lr ceiling) |
| 03 | 5.0 | 1.5E-4 | 384 | 3 | 600000 | **capacity** (test 384 speed Pareto from prior Jp@100f finding) |

All slots: siren_txy, batch_size=1, n_training_frames=400, nnr_f_xy_period=1.0, nnr_f_T_period=1.0

### Iterations This Block

(Starting fresh — Block 2)

### Emerging Observations

(No observations yet — Block 2 starting)
