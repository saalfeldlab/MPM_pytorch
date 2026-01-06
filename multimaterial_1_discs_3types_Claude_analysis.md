# Experiment Log: multimaterial_1_discs_3types_Claude

## Block 1: siren_txy / Jp / n_frames=48

## Iter 1: poor
Node: id=1, parent=root
Mode/Strategy: baseline/exploit
Config: lr_NNR_f=1E-5, total_steps=50000, hidden_dim_nnr_f=64, n_layers_nnr_f=3, omega_f=80.0, batch_size=1
Metrics: final_r2=0.472, final_mse=107.24, total_params=12801, training_time=2.0min, slope=0.060
Field: field_name=Jp, inr_type=siren_txy
Mutation: baseline (no parent)
Parent rule: UCB file shows Node 1 as root - this is baseline
Observation: Severe underfitting. omega_f=80 is too high (>50 unstable per protocol). Small hidden_dim=64 compounds the issue.
Next: parent=1, mutate omega_f: 80 → 30

## Iter 2: poor
Node: id=2, parent=1
Mode/Strategy: exploit
Config: lr_NNR_f=1E-5, total_steps=50000, hidden_dim_nnr_f=64, n_layers_nnr_f=3, omega_f=30.0, batch_size=1
Metrics: final_r2=0.549, final_mse=107.35, total_params=12801, training_time=2.0min, slope=0.059
Field: field_name=Jp, inr_type=siren_txy
Mutation: omega_f: 80 → 30
Parent rule: Node 2 has highest UCB (1.549) - exploit
Observation: Slight improvement (+0.077 R²) from omega_f reduction, but still severely underfitting. hidden_dim=64 likely too small.
Next: parent=2, mutate hidden_dim_nnr_f: 64 → 256

## Iter 3: moderate
Node: id=3, parent=2
Mode/Strategy: exploit
Config: lr_NNR_f=1E-5, total_steps=50000, hidden_dim_nnr_f=256, n_layers_nnr_f=3, omega_f=30.0, batch_size=1
Metrics: final_r2=0.862, final_mse=74.61, total_params=198657, training_time=6.3min, slope=0.219
Field: field_name=Jp, inr_type=siren_txy
Mutation: hidden_dim_nnr_f: 64 → 256
Parent rule: Node 3 has highest UCB (2.086) - exploit
Observation: Significant improvement (+0.313 R²) from hidden_dim increase. Network capacity was the bottleneck. Slope still low (0.219 vs target 1.0) suggests systematic bias. Try increasing learning rate to improve convergence.
Next: parent=3, mutate lr_NNR_f: 1E-5 → 5E-5

## Iter 4: moderate
Node: id=4, parent=3
Mode/Strategy: exploit
Config: lr_NNR_f=5E-5, total_steps=50000, hidden_dim_nnr_f=256, n_layers_nnr_f=3, omega_f=30.0, batch_size=1
Metrics: final_r2=0.871, final_mse=20.38, total_params=198657, training_time=6.3min, slope=0.685
Field: field_name=Jp, inr_type=siren_txy
Mutation: lr_NNR_f: 1E-5 → 5E-5
Parent rule: Node 4 has highest UCB (2.285) - exploit
Observation: Marginal R² gain (+0.009) but dramatic slope improvement (0.219→0.685). Higher lr accelerates convergence quality. Pattern: lr controls slope, hidden_dim controls R² ceiling. MSE dropped significantly (74.61→20.38).
Next: parent=4, mutate hidden_dim_nnr_f: 256 → 512

## Iter 5: moderate
Node: id=5, parent=4
Mode/Strategy: exploit
Config: lr_NNR_f=5E-5, total_steps=50000, hidden_dim_nnr_f=512, n_layers_nnr_f=3, omega_f=30.0, batch_size=1
Metrics: final_r2=0.730, final_mse=33.59, total_params=790529, training_time=19.2min, slope=0.655
Field: field_name=Jp, inr_type=siren_txy
Mutation: hidden_dim_nnr_f: 256 → 512
Parent rule: Node 5 had highest UCB (2.311) but this was before Iter 5 ran - actual parent was Node 4
Observation: REGRESSION. R² dropped 0.871→0.730 (-0.141). Training time tripled (6.3→19.2 min). Larger network hurt performance. lr=5E-5 may be too high for larger network, or 50k steps insufficient. Return to Node 4 config and try different mutation.
Next: parent=4, mutate n_layers_nnr_f: 3 → 4

## Iter 6: good
Node: id=6, parent=4
Mode/Strategy: exploit
Config: lr_NNR_f=5E-5, total_steps=50000, hidden_dim_nnr_f=256, n_layers_nnr_f=4, omega_f=30.0, batch_size=1
Metrics: final_r2=0.942, final_mse=12.96, total_params=264449, training_time=8.0min, slope=0.726
Field: field_name=Jp, inr_type=siren_txy
Mutation: n_layers_nnr_f: 3 → 4
Parent rule: UCB selected Node 4 (highest before Iter 6) - exploit
Observation: Significant improvement (+0.071 R² over Node 4). Depth increase (3→4 layers) outperformed width increase (512 hidden). Training time reasonable (8.0 min vs 19.2 min for 512 hidden). Approaching target R²>0.95. Try lr increase to push past plateau.
Next: parent=6, mutate lr_NNR_f: 5E-5 → 1E-4

## Iter 7: poor
Node: id=7, parent=6
Mode/Strategy: exploit
Config: lr_NNR_f=1E-4, total_steps=50000, hidden_dim_nnr_f=256, n_layers_nnr_f=4, omega_f=30.0, batch_size=1
Metrics: final_r2=0.000, final_mse=121.11, total_params=264449, training_time=8.0min, slope=-0.00001
Field: field_name=Jp, inr_type=siren_txy
Mutation: lr_NNR_f: 5E-5 → 1E-4
Parent rule: UCB selected Node 6 (highest UCB=2.189) - exploit
Observation: FAILURE. lr=1E-4 too high for this architecture - training completely collapsed (R²=0.000). Confirms lr upper bound is ~5E-5 for siren_txy with 4 layers. Return to Node 6 and try different mutation.
Next: parent=6, mutate n_layers_nnr_f: 4 → 5

## Iter 8: moderate
Node: id=8, parent=6
Mode/Strategy: exploit
Config: lr_NNR_f=5E-5, total_steps=50000, hidden_dim_nnr_f=256, n_layers_nnr_f=5, omega_f=30.0, batch_size=1
Metrics: final_r2=0.679, final_mse=42.71, total_params=330241, training_time=9.7min, slope=0.531
Field: field_name=Jp, inr_type=siren_txy
Mutation: n_layers_nnr_f: 4 → 5
Parent rule: UCB selected Node 6 (highest UCB=2.189 at time of selection) - exploit
Observation: REGRESSION. R² dropped 0.942→0.679 (-0.263). 5 layers too deep - causing gradient issues. Optimal depth for siren_txy appears to be 4 layers (Node 6). Both depth increase (5 layers) and width increase (512 hidden) from Node 6 fail.
Next: parent=5, mutate total_steps: 50000 → 75000

## Iter 9: moderate
Node: id=9, parent=5
Mode/Strategy: exploit
Config: lr_NNR_f=2E-5, total_steps=50000, hidden_dim_nnr_f=512, n_layers_nnr_f=3, omega_f=30.0, batch_size=1
Metrics: final_r2=0.900, final_mse=28.43, total_params=790529, training_time=19.1min, slope=0.551
Field: field_name=Jp, inr_type=siren_txy
Mutation: lr_NNR_f: 5E-5 → 2E-5
Parent rule: UCB selected Node 5 (UCB=2.730, highest) - testing if lower lr helps hidden_dim=512
Observation: IMPROVEMENT. Lower lr (2E-5) helped hidden_dim=512 converge: R² 0.730→0.900 (+0.170). Still below Node 6 (0.942) but validates that larger network needs lower lr. Training time still high (19.1 min).
Next: parent=9, try n_layers_nnr_f: 3 → 4 to combine depth advantage

## Iter 10: moderate
Node: id=10, parent=9
Mode/Strategy: exploit
Config: lr_NNR_f=2E-5, total_steps=50000, hidden_dim_nnr_f=512, n_layers_nnr_f=4, omega_f=30.0, batch_size=1
Metrics: final_r2=0.861, final_mse=30.66, total_params=1053185, training_time=24.5min, slope=0.547
Field: field_name=Jp, inr_type=siren_txy
Mutation: n_layers_nnr_f: 3 → 4
Parent rule: UCB selected Node 9 (UCB=3.021, highest) - testing depth increase with larger hidden_dim
Observation: REGRESSION. R² dropped 0.900→0.861 (-0.039). Combining hidden_dim=512 + n_layers=4 does NOT help - worse than either Node 6 (256×4, R²=0.942) or Node 9 (512×3, R²=0.900). Training time excessive (24.5 min). Pattern: width and depth don't scale together well for siren_txy.
Next: parent=6, mutate omega_f: 30 → 40 (try frequency tuning on best config)

## Iter 11: poor
Node: id=11, parent=6
Mode/Strategy: exploit
Config: lr_NNR_f=5E-5, total_steps=50000, hidden_dim_nnr_f=256, n_layers_nnr_f=4, omega_f=40.0, batch_size=1
Metrics: final_r2=0.576, final_mse=54.26, total_params=264449, training_time=7.9min, slope=0.455
Field: field_name=Jp, inr_type=siren_txy
Mutation: omega_f: 30 → 40
Parent rule: UCB selected Node 6 (best R²=0.942) - exploring omega_f tuning
Observation: SEVERE REGRESSION. R² dropped 0.942→0.576 (-0.366). omega_f=40 too high. This confirms omega_f=30 is optimal for siren_txy on Jp field. Higher frequency causes network to attempt fitting oscillations it cannot represent. Both omega_f=40 (Iter 11) and omega_f=80 (Iter 1) fail badly.
Next: parent=6, mutate omega_f: 30 → 20 (try lower frequency)

## Iter 12: moderate (BLOCK END)
Node: id=12, parent=6
Mode/Strategy: exploit/boundary-test
Config: lr_NNR_f=5E-5, total_steps=50000, hidden_dim_nnr_f=256, n_layers_nnr_f=4, omega_f=20.0, batch_size=1
Metrics: final_r2=0.908, final_mse=17.26, total_params=264449, training_time=8.0min, slope=0.693
Field: field_name=Jp, inr_type=siren_txy
Mutation: omega_f: 30 → 20
Parent rule: UCB selected Node 6 - testing lower omega_f boundary
Observation: REGRESSION. R² dropped 0.942→0.908 (-0.034). omega_f=20 underperforms vs omega_f=30. omega_f boundary confirmed: 20 too low, 40 too high, 30 optimal. This completes the omega_f sweep: 20→0.908, 30→0.942, 40→0.576, 80→0.472. Clear bell curve with peak at 30.

=== BLOCK 1 SUMMARY ===
- **Best config**: Node 6 (hidden_dim=256, n_layers=4, lr=5E-5, omega_f=30) - R²=0.942, slope=0.726
- **Iterations**: 12 | **Excellent (R²>0.95)**: 0 | **Good (0.90-0.95)**: 1 | **Moderate (0.75-0.90)**: 6 | **Poor (<0.75)**: 5
- **Branching rate**: 5/12 (42%) - parent ≠ (current_iter - 1) for Iters 6, 8, 9, 11, 12
- **Improvement rate**: 5/12 (42%) - Iters 2, 3, 4, 6, 9 improved over parent
- **Key findings**:
  1. omega_f=30 is optimal for siren_txy on Jp field (20, 40, 80 all worse)
  2. Depth > width: 256×4 beats both 512×3 and 512×4
  3. Larger networks (hidden_dim=512) require lower lr (2E-5 vs 5E-5)
  4. lr=1E-4 causes training collapse
  5. 5 layers too deep - gradient issues
  6. hidden_dim=64 severely underfits
- **Did NOT reach R²>0.95 target** - best was 0.942. May need more total_steps or different field.

PROTOCOL EDITED: Added rule about omega_f=30 being optimal for siren_txy.
=== END BLOCK 1 ===

