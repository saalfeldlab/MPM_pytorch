# Working Memory: multimaterial_1_discs_3types (parallel)

## Knowledge Base (accumulated across all blocks)

### Regime Comparison Table
| Block | INR Type | Field | n_frames | Best R² | Best slope | kino_R2 | kino_SSIM | Optimal lr_NNR_f | Optimal hidden_dim | Optimal n_layers | Optimal omega_f | Optimal total_steps | Training time (min) | Key finding |
|-------|----------|-------|----------|---------|------------|---------|-----------|------------------|--------------------|--------------------|-----------------|---------------------|---------------------|-------------|

### Established Principles
- Prior knowledge (from appendix): Jp@100f@9000p optimal omega_f=[5-10], lr=4E-5, hidden_dim=384-512, n_layers=3, 200k steps
- siren_txy is the architecture for this parallel exploration
- batch_size=1 mandatory to avoid training time explosion
- Depth ceiling: Jp requires exactly 3 layers

### Open Questions
- Does the parallel config reproduce prior sequential results?
- Is lr=6E-5 viable for Jp on this dataset (prior showed 6E-5 optimal at 48f)?
- Does omega_f=12 degrade Jp (prior showed 15 already suboptimal)?
- Does hidden_dim=512 improve over 384 or just add training time?

---

## Previous Block Summary

(No previous blocks — this is the first batch.)

---

## Current Block (Block 1)

### Block Info
Field: field_name=Jp, inr_type=siren_txy, n_training_frames=100
First batch: 4 parallel slots exploring parameter space

### Hypothesis
Jp@100f@9000p with siren_txy should reproduce prior results (R²~0.995-0.996). Slot 00 (baseline 384×3, omega_f=7, lr=4E-5) should match prior best. Slot 01 (512×3) may gain marginal accuracy at cost of training time. Slot 02 (lr=6E-5) probes whether higher lr improves on this dataset (prior showed 6E-5 optimal at 48f but 4E-5 at 100f). Slot 03 (omega_f=12) tests upper omega_f boundary (prior showed 10→0.995, 15→0.992).

### Planned Mutations (Batch 1)
| Slot | hidden_dim | omega_f | lr_NNR_f | n_layers | total_steps | Variation |
|------|-----------|---------|----------|----------|-------------|-----------|
| 00 | 384 | 7.0 | 4E-5 | 3 | 200k | Baseline known-optimal |
| 01 | 512 | 7.0 | 4E-5 | 3 | 200k | Capacity probe (+33%) |
| 02 | 384 | 7.0 | 6E-5 | 3 | 200k | LR probe (+50%) |
| 03 | 384 | 12.0 | 4E-5 | 3 | 200k | Higher omega_f probe |

All slots share: Jp, siren_txy, n_training_frames=100, batch_size=1, n_layers=3

### Iterations This Block

(Awaiting batch 1 results.)

### Emerging Observations

(No observations yet — first batch pending.)
