# Experiment Log: multimaterial_1_discs_3types (parallel)

## Block 1: Jp@100f, siren_txy, parallel initialization

### Batch 1 Plan (4 slots)
Field: Jp, INR: siren_txy, n_training_frames=100, batch_size=1, n_layers=3

| Slot | hidden_dim | omega_f | lr_NNR_f | total_steps | Variation |
|------|-----------|---------|----------|-------------|-----------|
| 00 | 384 | 7.0 | 4E-5 | 200k | Baseline known-optimal |
| 01 | 512 | 7.0 | 4E-5 | 200k | Capacity probe (+33%) |
| 02 | 384 | 7.0 | 6E-5 | 200k | LR probe (+50%) |
| 03 | 384 | 12.0 | 4E-5 | 200k | Higher omega_f probe |

Rationale: Prior sequential exploration established Jp@100f@9000p optimal at omega_f=[5-10], lr=4E-5, hidden_dim=384-512. This batch verifies baseline and probes three parameter dimensions simultaneously (capacity, lr, omega_f) for maximum information per batch.

