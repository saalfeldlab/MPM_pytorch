# Working Memory: multimaterial_1_discs_3types_Claude

## Knowledge Base (accumulated across all blocks)

### Regime Comparison Table
| Block | INR Type | Field | n_frames | Best R² | Best slope | Optimal lr_NNR_f | Optimal hidden_dim | Optimal n_layers | Optimal omega_f | Optimal total_steps | Training time (min) | Key finding |
|-------|----------|-------|----------|---------|------------|------------------|--------------------|--------------------|-----------------|---------------------|---------------------|-------------|
| 1     | siren_txy | Jp   | 48       | 0.908   | 0.400      | 2E-5             | 256                | 4                  | 30.0            | 50000               | 7.7                 | 256×4 matches 512×3 at 2.5x faster |
| 2     | siren_txy | Jp   | 100      | 0.964   | 0.904      | 2E-5             | 512                | 4                  | 30.0            | 150000              | 72.1                | First R²>0.95; steps scale with frames |
| 3     | siren_txy | F    | 100      | 0.999   | 1.000      | 2E-5             | 512                | 4                  | 30.0            | 100000              | 48.5                | Field-agnostic architecture; F more efficient than Jp |

### Established Principles
1. **omega_f sensitivity**: omega_f=30 is strictly optimal for siren_txy. omega_f=35 degrades R² by 0.07, omega_f≥40 causes severe regression.
2. **Depth vs width tradeoff**: hidden_dim=256 with n_layers=4 achieves R²=0.908 for 48 frames, 512×4 achieves R²=0.964-0.999 for 100 frames.
3. **lr-depth relationship**: n_layers=4 tolerates lr=2E-5 optimally. n_layers=5 fails regardless of lr (tested 2E-5, 1.5E-5, 1E-5).
4. **5-layer ceiling**: n_layers=5 degrades R² at all learning rates. 4 layers is optimal depth for siren_txy.
5. **Training data scaling**: total_steps should scale with n_training_frames. Rule: ~1000-1500 steps/frame for R²>0.95.
6. **lr upper bound for 512×4**: lr=2E-5 is optimal. lr=1.5E-5 degrades R² (0.943→0.929).
7. **Field generalization**: Architecture (512×4, lr=2E-5, omega=30) is field-agnostic - works for both Jp and F fields.
8. **F field efficiency**: F achieves R²=0.999 vs Jp's R²=0.964 with identical config. F field is easier to fit.
9. **Steps boundary for F**: ~100 steps/frame ≈ R²=0.95 threshold (10k steps for 100 frames with 512×4).
10. **Capacity efficiency**: At 100 frames/10k steps, 128×4 (67k params) achieves R²=0.951, 256×4 (265k) achieves R²=0.965, 512×4 (1.05M) achieves R²=0.955. Smaller models viable for efficiency.

### Open Questions
1. Does the 512×4 configuration generalize to S and C fields?
2. What is the optimal steps/frame ratio for S field (stress tensor)?
3. How does capacity scaling change with more training frames (200, 500)?

---

## Previous Block Summary (Block 3)

Block 3 tested F field (4 components) with siren_txy at 100 frames. Optimal Jp config (512×4, lr=2E-5, omega=30) achieved exceptional R²=0.999 on F field - confirming architecture is field-agnostic. Systematic probing found steps boundary (~100 steps/frame for R²≥0.95) and capacity boundary (128×4 minimum viable). 11/12 iterations achieved R²≥0.95. Best efficiency: 128×4, 10k steps → R²=0.951 in 0.8min. Best quality: 512×4, 100k steps → R²=0.999 in 48.5min.

---

## Current Block (Block 4)

### Block Info
Field: field_name=S, inr_type=siren_txy
n_training_frames: 100, hidden_dim: 512, n_layers: 4
Iterations: 37 to 48

### Hypothesis
Testing if optimal config (512×4, lr=2E-5, omega=30, 100k steps) generalizes to S field (stress tensor). S field has 4 components like F, but different physical meaning and typical range (~0-0.01 vs ~1.0-2.0). May need lr adjustment due to smaller magnitude. Expect R²>0.90 if architecture transfers.

### Iterations This Block

## Iter 37: poor
Node: id=37, parent=root
Mode/Strategy: exploit (new block start)
Config: lr_NNR_f=2E-5, total_steps=100000, hidden_dim_nnr_f=512, n_layers_nnr_f=4, omega_f=30.0, batch_size=1
Metrics: final_r2=0.339, final_mse=9.76E-8, total_params=1054724, slope=0.393, training_time=49.7min
Field: field_name=S, inr_type=siren_txy, n_training_frames=100
Mutation: field_name: F -> S (new block)
Parent rule: New block start, testing if optimal config generalizes to S field
Observation: CRITICAL FAILURE! R²=0.339 despite optimal config achieving R²=0.999 on F and R²=0.964 on Jp. S field has fundamentally different characteristics requiring different approach.
Next: parent=37

## Iter 38: poor
Node: id=38, parent=37
Mode/Strategy: exploit
Config: lr_NNR_f=2E-5, total_steps=100000, hidden_dim_nnr_f=512, n_layers_nnr_f=4, omega_f=60.0, batch_size=1
Metrics: final_r2=0.516, final_mse=7.11E-8, total_params=1054724, slope=0.547, training_time=49.5min
Field: field_name=S, inr_type=siren_txy, n_training_frames=100
Mutation: omega_f: 30.0 -> 60.0
Parent rule: Highest UCB (node 37 at block start), testing omega_f increase hypothesis
Observation: SIGNIFICANT IMPROVEMENT! R² jumped from 0.339 → 0.516 (+52% relative). Higher omega_f helps S field. Still poor but clear positive direction.
Next: parent=38

### Emerging Observations
- **S field requires high omega_f**: omega_f=30 gave R²=0.339, omega_f=60 gave R²=0.516 (+52% relative improvement)
- **MSE is very low** (7-10E-8) suggesting S field values have small variance - model fits mean but not variations
- S field typical range (~0-0.01) is 2 orders smaller than F (~1.0-2.0), requires higher frequency to capture small-scale structure
- **Architecture is NOT fully field-agnostic** - S field needs omega_f tuning
- **Next hypothesis**: Continue increasing omega_f (try 90-100) to see if R² continues improving
- Alternative: If omega_f saturates, try increasing lr to help with small gradient magnitudes

