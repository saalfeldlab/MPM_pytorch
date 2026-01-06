# Working Memory: multimaterial_1_discs_3types_Claude

## Knowledge Base (accumulated across all blocks)

### Regime Comparison Table
| Block | INR Type | Field | n_frames | Best R² | Best slope | Optimal lr_NNR_f | Optimal hidden_dim | Optimal n_layers | Optimal omega_f | Optimal total_steps | Training time (min) | Key finding |
|-------|----------|-------|----------|---------|------------|------------------|--------------------|--------------------|-----------------|---------------------|---------------------|-------------|
| 1     | siren_txy | Jp   | 48       | 0.942   | 0.726      | 5E-5             | 256                | 4                  | 30.0            | 50000               | 8.0                 | Depth > width; omega_f=30 optimal |

### Established Principles
1. **omega_f sensitivity**: For siren_txy on Jp, omega_f=30 is optimal. omega_f=20 underperforms (-0.034 R²), omega_f=40 causes severe regression (-0.366 R²), omega_f=80 fails badly.
2. **Depth > width**: 256×4 (R²=0.942) outperforms both 512×3 (R²=0.900) and 512×4 (R²=0.861). Depth is more efficient than width for siren_txy.
3. **lr scales inversely with hidden_dim**: hidden_dim=256 works with lr=5E-5; hidden_dim=512 requires lr=2E-5. lr=1E-4 causes training collapse for 4-layer networks.
4. **n_layers upper bound**: 5 layers causes gradient issues (R² dropped 0.942→0.679). Optimal depth = 4 layers for siren_txy.
5. **hidden_dim lower bound**: hidden_dim=64 severely underfits (R²=0.472-0.549 regardless of omega_f).

### Open Questions
- Does omega_f=30 generalize to other fields (F, S, C)?
- Will the 256×4 architecture work for multi-component fields (F has 4 components)?
- Is R²>0.95 achievable with more total_steps, or is there a fundamental limit?
- Does slope ~0.73 indicate systematic bias that can be corrected?

---

## Previous Block Summary (Block 1)

Block 1 (siren_txy, Jp, n_frames=48): Best R²=0.942 at Node 6 (hidden_dim=256, n_layers=4, lr=5E-5, omega_f=30). Did NOT reach R²>0.95 target. Key findings: omega_f=30 optimal (bell curve confirmed), depth>width for efficiency, lr scales inversely with hidden_dim. Branching rate 42%, improvement rate 42%.

---

## Current Block (Block 2)

### Block Info
Field: field_name=F, inr_type=siren_txy
Config baseline: hidden_dim=256, n_layers=4, omega_f=30, lr=5E-5 (Block 1 best config)
Iterations: 13-24

### Hypothesis
The optimal config from Block 1 (256×4, lr=5E-5, omega_f=30) should transfer to field F. However, F has 4 components vs Jp's 1, which may require adjustments. Expect R² in 0.85-0.95 range initially. If lower, the network may need more capacity or different omega_f tuning.

### Iterations This Block

(Starting fresh for Block 2)

### Emerging Observations

(To be updated as Block 2 progresses)
