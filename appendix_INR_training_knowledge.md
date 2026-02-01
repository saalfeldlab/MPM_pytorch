# Appendix: INR Training Knowledge (up to n_training_frames=200)

**Dataset**: multimaterial_1_discs_3types (9000 particles, 3 material types)
**Architecture**: SIREN-based INR (siren_txy, siren_t)
**Accumulated over**: 130+ exploration iterations across 17+ blocks

---

## 1. Optimal Configurations by Field and n_training_frames

### Jp (Plastic deformation, 1 component)

| n_training_frames | INR Type | omega_f | lr | hidden_dim | n_layers | steps | R² | Time (min) |
|--------|----------|---------|-----|------------|----------|-------|-----|-----------|
| 100 | siren_txy | 5-10 | 4E-5 | 384 | 3 | 200k | 0.995 | 12.2 |
| 100 | siren_txy | 5-10 | 4E-5 | 512 | 3 | 200k | 0.996 | 15.4 |
| 100 | siren_t | 7 | 8E-5 | 256 | 3 | 200k | 0.99995 | 9.7 |
| 200 | siren_txy | 3-7 | 1E-4 | 512 | 3 | 400k | 0.997 | 67.7 |
| 200 | siren_txy | 3 | 1.5E-4 | 512 | 3 | 300k | 0.997 | 40.5 |

### F (Deformation gradient, 4 components)

| n_training_frames | INR Type | omega_f | lr | hidden_dim | n_layers | steps | R² | Time (min) |
|--------|----------|---------|-----|------------|----------|-------|-----|-----------|
| 100 | siren_txy | 12 | 4-6E-5 | 256 | 4 | 150k | 0.998 | 8.1 |
| 100 | siren_t | 3 | 5E-5 | 256 | 3 | 150k | 1.000 | 8.9 |
| 100 | siren_t | 3 | 5E-5 | 256 | 2 | 100k | 1.000 | 5.5 |
| 200 | siren_txy | 9-10 | 5E-5 | 256 | 4 | 300k | 0.9997 | 32.4 |
| 200 | siren_txy | 9-10 | 5E-5 | 256 | 4 | 200k | 0.9988 | 27.6 |

### C (APIC matrix, 4 components)

| n_training_frames | INR Type | omega_f | lr | hidden_dim | n_layers | steps | R² | Time (min) |
|--------|----------|---------|-----|------------|----------|-------|-----|-----------|
| 100 | siren_txy | 25 | 2E-5 | 640 | 3 | 150k | 0.994 | 15.7 |
| 100 | siren_t | 3-5 | 5E-5 | 640 | 3 | 300k | 0.9999 | 26.0 |
| 100 | siren_t | 3 | 5E-5 | 640 | 2 | 150k | 0.9997 | 12.4 |
| 200 | siren_txy | 20 | 3E-5 | 768 | 3 | 500k | 0.991 | 68.7 |

### S (Stress tensor, 4 components)

| n_training_frames | INR Type | omega_f | lr | hidden_dim | n_layers | steps | R² | Time (min) | Notes |
|--------|----------|---------|-----|------------|----------|-------|-----|-----------|-------|
| 100 | siren_txy | 48 | 2E-5 | 1280 | 3 | 300k | 0.729 | 166 | Without scheduler |
| 100 | siren_txy | 48 | 3E-5 | 1280 | 3 | 500k | 0.998 | 151 | With CosineAnnealingLR |

---

## 2. Scaling Rules

### omega_f vs n_training_frames

| Field | 100 | 200 | Scaling rule |
|-------|-----|-----|-------------|
| Jp | 5-10 | 3-7 | ~20-30% reduction per +100 n_training_frames; becomes INSENSITIVE at 200 |
| F | 12 | 9-10 | ~2-3 decrease per +100 n_training_frames |
| C | 25 | 20 | ~20% reduction per +100 n_training_frames |
| S | 48 | (untested) | High-complexity fields do NOT shift down |

**Rule**: Low-complexity fields (Jp, F) shift omega_f down with more frames. High-complexity fields (C, S) maintain omega_f.

### Learning rate vs n_training_frames

| Field | 100 | 200 | Rule |
|-------|-----|-----|------|
| Jp | 4E-5 | 1E-4 | 2.5x INCREASE — more data regularizes |
| F | 4-6E-5 (wide) | 5E-5 (narrow) | Tolerance NARROWS with more frames |
| C | 2E-5 | 3E-5 | 50% increase |
| S | 2E-5 (3E-5 with scheduler) | (untested) | Hard-locked without scheduler |

**Rule**: Higher n_training_frames → higher lr ceiling (data regularizes), but for F the optimal window narrows.

### Steps per frame

| Field | steps/frame rule | Max before overfitting |
|-------|-----------------|----------------------|
| Jp | 2000 | 2500 (overfitting at 500k/n_training_frames=200) |
| F | 1500 at 200, 800 at 500 | Efficiency improves with higher n_training_frames |
| C | 2500 | Overtraining-resistant with siren_t |
| S | 3000-5000 | Requires scheduler |

(Column values = n_training_frames)

### Capacity vs n_training_frames

- **Jp**: 512×3 stable across n_training_frames
- **F**: 256×4 stable (capacity ceiling at 256, 384 HURTS)
- **C**: Increases 640→768 from n_training_frames=100→200
- **S**: 1280×3 required (high base capacity)

---

## 3. Architecture Comparison: siren_t vs siren_txy

| Property | siren_t | siren_txy |
|----------|---------|-----------|
| Input | t → all particles | (t, x, y) → per particle |
| omega_f | 50-88% LOWER | Baseline |
| lr | 2-2.5× HIGHER tolerated | Baseline |
| Depth | Can use 2 layers for speed | Needs 3-4 layers |
| Jp | R²=0.99995, 9.7min | R²=0.996, 15.4min |
| F | R²=1.000, 5.5min | R²=0.998, 8.1min |
| C | R²=0.9999, 12.4min | R²=0.994, 15.7min |
| S | **CATASTROPHIC** (R²=0.004) | R²=0.998 with scheduler |

**Recommendation**: Use siren_t for Jp/F/C. Use siren_txy for S (mandatory).

---

## 4. Known Failure Modes

### Critical failures
1. **siren_t + S field** → R²=0.001-0.004 (1D→36000D mapping underconstrained for stress)
2. **S without CosineAnnealingLR** → R²=0.729 ceiling
3. **siren_id architecture** → Fails for all fields (R²<0.10)
4. **F with T_period=2.0** → Catastrophic (R²=0.790), temporal smoothing 6× more damaging than spatial
5. **LayerNorm/BatchNorm + SIREN** → Destroys omega-scaled initialization (R²=0.022)

### Depth ceilings
- Jp: n_layers > 3 degrades
- F (siren_txy): n_layers > 4 degrades
- C: n_layers > 3 degrades
- S: n_layers > 3 degrades

### Capacity ceilings
- F: hidden_dim > 256 HURTS (384 → R²=0.989 vs 0.998)
- C: hidden_dim = 640 is LOCAL MAXIMUM (512 underfit, 768 overfit at n_training_frames=100)
- S: hidden_dim = 1280 optimal (1536 FAILS catastrophically)

### LR-steps interaction
- At high lr (≥1.5E-4), fewer steps can OUTPERFORM more steps (overshoot prevention)
- Example: Jp @ n_training_frames=200, lr=1.5E-4: 300k(0.997) > 400k(0.989)

---

## 5. Data Scaling Behavior per Field

| Field | 100→200 | Trend | Recommended action at n_training_frames≥400 |
|-------|-----------|-------|----------------------------|
| F | 0.998→0.9997 | NO diminishing returns | Scale confidently; reduce omega_f ~2-3, keep lr~5E-5 |
| Jp | 0.996→0.997 | Diminishing returns (gains halve per doubling) | Scale but expect smaller gains |
| C | 0.994→0.991 | **HURTS** without capacity increase | Must increase capacity (640→768→?) |
| S | 0.998→(untested) | Unknown (scheduler required) | Test cautiously |

### F at n_training_frames=500 (Block 14 finding)
- R²=0.9997 (NO diminishing returns even at n_training_frames=500)
- Only 800 steps/frame needed (vs 1500 at n_training_frames=200 — efficiency gain)
- Speed Pareto: 400 steps/frame achieves R²=0.992 in 10min

---

## 6. Starting Configurations for n_training_frames=400/600/1000

Based on scaling rules extrapolated from n_training_frames 100→200→500 data:

### F field (most scalable)
| n_training_frames | omega_f | lr | hidden_dim | n_layers | steps | Expected R² |
|--------|---------|-----|------------|----------|-------|------------|
| 400 | 7-8 | 5E-5 | 256 | 4 | 320k (800/f) | ~0.999 |
| 600 | 6-7 | 5E-5 | 256 | 4 | 420k (700/f) | ~0.999 |
| 1000 | 5-6 | 5E-5 | 256 | 4 | 600k (600/f) | ~0.999 |

### Jp field
| n_training_frames | omega_f | lr | hidden_dim | n_layers | steps | Expected R² |
|--------|---------|-----|------------|----------|-------|------------|
| 400 | 3-5 | 1.5E-4 | 512 | 3 | 600k (1500/f) | ~0.998 |
| 600 | 3-5 | 2E-4 | 512 | 3 | 750k (1250/f) | ~0.998 |
| 1000 | 3-5 | 2E-4 | 512 | 3 | 1000k (1000/f) | ~0.997 |

### C field (needs capacity scaling)
| n_training_frames | omega_f | lr | hidden_dim | n_layers | steps | Expected R² |
|--------|---------|-----|------------|----------|-------|------------|
| 400 | 18 | 4E-5 | 896 | 3 | 1000k (2500/f) | ~0.99 |
| 600 | 16 | 4E-5 | 1024 | 3 | 1500k (2500/f) | ~0.99 |
| 1000 | 14 | 5E-5 | 1024 | 3 | 2000k (2000/f) | ~0.99 |

### S field (requires scheduler, siren_txy only)
| n_training_frames | omega_f | lr | hidden_dim | n_layers | steps | scheduler | Expected R² |
|--------|---------|-----|------------|----------|-------|-----------|------------|
| 400 | 48 | 3E-5 | 1280 | 3 | 1000k | CosineAnnealingLR | ~0.99 |
| 600 | 45 | 3E-5 | 1280 | 3 | 1500k | CosineAnnealingLR | ~0.99 |
| 1000 | 42 | 3E-5 | 1280 | 3 | 2500k | CosineAnnealingLR | unknown |

**Note**: S field scaling is speculative — only tested at n_training_frames=100. Must proceed cautiously.

---

## 7. Key Principles for High-Frame Exploration

1. **Re-tune omega_f when changing n_training_frames** — never reuse omega_f from a different n_training_frames
2. **Probe HIGHER lr when increasing n_training_frames** — data regularizes and widens tolerance (Jp: 2.5× increase per 2× n_training_frames)
3. **Steps/frame can DECREASE with more n_training_frames** — F showed 1500→800 steps/frame from n_training_frames=200→500
4. **C field needs capacity increase** — plan for 640→768→896→1024 progression
5. **S field requires CosineAnnealingLR** — without it, R² ceiling is 0.73
6. **siren_t dominates for Jp/F/C** — but FAILS for S
7. **Never exceed depth ceilings** — Jp/C/S: 3 layers, F: 4 layers (siren_txy)
8. **Period parameters must stay at 1.0** — T_period=2.0 is catastrophic for F
9. **Overfitting risk increases with more steps** — never exceed 2500 steps/frame for Jp
10. **If slope < 1 (underprediction), increase lr** — monotonic relationship confirmed for Jp

---

## 8. Field Difficulty Ranking

F (easiest, R²≈1.0) > Jp (R²≈0.9999) > C (R²≈0.9999 with siren_t) >> S (R²=0.998 with scheduler)

Training time ranking: F (fastest) < Jp < C << S (slowest, 10-20× F)
