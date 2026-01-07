# Epistemic Analysis Instructions

## Purpose

This document describes how to conduct an epistemic analysis of LLM-guided scientific exploration, following the framework from "Understanding: an experiment-LLM-memory experiment" (Allier & Saalfeld, 2026).

## Background

The experiment-LLM-memory triad implements a closed-loop scientific reasoning process where:
- **External experiments** provide empirical validation
- **LLM** generates hypotheses and parameter mutations
- **Long-term memory** enables cumulative knowledge accumulation

Reasoning is **distributed** across these components. The goal of epistemic analysis is to quantify how the system acquires, tests, revises, and transfers knowledge.

---

## Reasoning Modes to Identify

### 1. Induction (Observations → Pattern)

**Definition**: Generalizing from specific observations to broader patterns.

**Identification criteria**:
- Multiple observations lead to a generalized rule
- Pattern spans across iterations or blocks
- Language markers: "scales with", "optimal for", "consistently", "pattern"

**Example**:
- Single observation: "lr=2E-5 works for this config"
- Cumulative induction: "lr scales inversely with depth: 4-layer needs lr=2E-5, 5-layer needs lr<2E-5"

**EXCLUDE**: Patterns that were provided as priors in the instruction file.

### 2. Abduction (Observation → Explanatory Hypothesis)

**Definition**: Inferring the best explanation for a surprising observation.

**Identification criteria**:
- Unexpected result triggers hypothesis formation
- Proposes causal mechanism
- Language markers: "likely because", "suggests", "explains", "caused by"

**Example**:
- Observation: "R² dropped 0.908→0.857 when frames increased"
- Abduction: "More data + same steps = underfitting"

### 3. Deduction (Hypothesis → Testable Prediction)

**Definition**: Deriving specific predictions from general hypotheses.

**Identification criteria**:
- Hypothesis leads to concrete prediction
- Prediction is testable in next iteration
- Language markers: "if...then", "should", "expect", "predict"

**Validation tracking**:
- Record whether prediction was confirmed or falsified
- Calculate validation rate = confirmed / total predictions

### 4. Falsification (Prediction vs Outcome → Reject/Refine)

**Definition**: Rejecting or refining hypotheses when predictions fail.

**Identification criteria**:
- Prediction contradicted by experimental result
- Hypothesis modified or rejected
- Language markers: "rejected", "falsified", "does NOT", "contrary to expectation"

**Example**:
- Prediction: "5-layer with lower lr will work"
- Outcome: "Tested lr=2E-5, 1.5E-5, 1E-5 - all failed"
- Falsification: "5-layer architecture definitively rejected"

### 5. Analogy/Transfer (Cross-Regime Knowledge Application)

**Definition**: Applying knowledge from one regime to another.

**Identification criteria**:
- Prior finding applied to new context
- Testing generalization
- Language markers: "generalizes", "transfers", "same config", "based on Block N"

### 6. Boundary Probing (Systematic Limit-Finding)

**Definition**: Systematically testing parameter limits.

**Identification criteria**:
- Sequential parameter changes in one direction
- Finding thresholds or failure points
- Language markers: "boundary", "minimum", "threshold", "limit"

---

## Excluding Priors

**Critical**: Do not count as discovered knowledge any patterns that were given as priors in the protocol/instruction file.

### Common priors to exclude:
- Parameter ranges provided in protocol
- Architecture types and their properties (from theoretical background)
- Classification thresholds (R² > 0.95 = excellent, etc.)
- General training dynamics (lr too high → oscillation, etc.)
- Common failure modes listed in protocol

### What to include:
- Specific parameter values discovered (e.g., "lr=2E-5 optimal" if not stated in protocol)
- Relationships discovered through exploration (e.g., "F field easier than Jp")
- Boundaries discovered through probing (e.g., "100 steps/frame minimum for R²≥0.95")
- Cross-block generalizations (e.g., "architecture is field-agnostic")

---

## Confidence Scoring for Emergent Principles

Each discovered principle should be assigned a confidence score based on evidence strength.

### Proposed Confidence Formula

```
confidence = min(100%, 50% + 10%×n_confirmations + 15%×n_alternatives_rejected + 10%×n_blocks)
```

**Origin**: This formula is a heuristic proposed for this analysis framework, not from the referenced paper. It is designed to reflect scientific epistemology:

| Component | Weight | Justification |
|-----------|--------|---------------|
| **Base** | 50% | A single observation provides moderate confidence |
| **n_confirmations** | +10% each | Replication increases confidence (classical statistics) |
| **n_alternatives_rejected** | +15% each | Rejecting competing hypotheses strengthens the principle (Popper) |
| **n_blocks** | +10% each | Cross-context replication indicates generalizability |

**Clarification on "alternatives rejected"**:
- This counts how many *competing hypotheses* were tested and failed
- Example: For "4-layer ceiling", alternatives like "5-layer works with lr=2E-5" were tested and rejected
- Each rejected alternative *increases* confidence in the main principle
- This is NOT counting times the principle itself failed (that would decrease confidence)

**Rationale**:
- Rejecting alternatives weighted higher (+15%) than simple confirmation (+10%) because eliminating competing explanations is stronger evidence (Popper's asymmetry)
- Cross-block evidence weighted because generalization across contexts is the hallmark of robust principles
- Formula saturates at 100% to avoid overconfidence

**Limitations**:
- Weights are heuristic, not empirically calibrated
- Does not account for quality of individual tests
- Assumes independence between tests (may overcount correlated evidence)

### Confidence Levels

| Level | Score | Criteria |
|-------|-------|----------|
| **Very High** | 90-100% | ≥5 confirming tests OR systematic boundary probing OR repeated falsification |
| **High** | 75-89% | 3-4 confirming tests across multiple contexts OR cross-block validation |
| **Medium** | 60-74% | 2 confirming tests OR single block evidence |
| **Low** | <60% | 1 test OR contradictory evidence exists |

### Evidence Types (by strength)

Ranking based on philosophy of science (Popper, Lakatos):

| Evidence Type | Weight | Description | Philosophical Basis |
|---------------|--------|-------------|---------------------|
| **Falsification** | Highest | Alternative hypothesis tested and rejected | Popper: falsification is asymmetrically stronger |
| **Boundary probing** | High | Systematic limit-finding with multiple data points | Identifies necessary conditions |
| **Cross-block replication** | High | Same finding in different regimes | Generalizability test |
| **Single confirmation** | Medium | One successful test | Necessary but not sufficient |
| **Indirect inference** | Low | Derived from other findings, not directly tested | Depends on validity of premises |

### Adjustments

- **Cap at 83-90%** if stochastic variance observed (results vary between runs)
- **Reduce by 10%** if only tested in one field/regime
- **Add "needs testing" note** if <3 confirming tests

---

## Analysis Procedure

### Step 1: Catalog all priors from instruction file
- List all pre-existing knowledge given to the LLM
- Note parameter ranges, theoretical background, guidelines

### Step 2: Parse logs chronologically
- Read analysis.md, memory.md, reasoning.log
- Extract each reasoning instance
- Tag with reasoning mode

### Step 3: Filter out prior-derived conclusions
- Remove any "discovery" that was already stated in priors
- Keep only genuinely emergent knowledge

### Step 4: Calculate metrics
- Count each reasoning mode
- Calculate deduction validation rate
- Calculate transfer success rate
- Identify cumulative vs single-shot reasoning

### Step 5: Assess emergent properties
- What understanding emerged that wasn't in priors?
- How did memory enable cumulative inference?
- Where did falsification lead to refinement?

---

## Output Format

### Summary Table: Reasoning Modes

| Reasoning Mode | Count | Validation Rate | Emergence Iteration |
|----------------|-------|-----------------|---------------------|
| Induction | N | N/A | First: X, Cumulative: Y |
| Abduction | N | N/A | First: X |
| Deduction | N | X% (Y/N validated) | First: X |
| Falsification | N | N/A | First: X |
| Analogy/Transfer | N | X% (Y/N successful) | First: X |
| Boundary Probing | N | N/A | First: X |

### Summary Table: Emergent Principles (Ordered by Confidence)

Order principles by confidence score (highest first: Very High → High → Medium → Low).

| # | Principle | Prior Given | What Discovered | Evidence | Confidence |
|---|-----------|-------------|-----------------|----------|------------|
| 1 | Name | "prior text" or None | Discovery description | N tests, M alternatives rejected | Level (X%) |

Include confidence calculation table:

| # | Principle | n_tests | n_alt_rejected | n_blocks | Calculated Score |
|---|-----------|---------|----------------|----------|------------------|
| 1 | Name | N | M | B | formula result |

### Detailed Analysis

For each mode, provide:
1. Representative examples
2. Single-shot vs cumulative instances
3. Contribution to knowledge accumulation

### Observed Reasoning Patterns Discussion

Discuss:
- What the system learned that wasn't provided as priors
- How memory stores findings across iterations
- What reasoning modes are observed in the logs

**Important caveat**: Do NOT claim "emergent reasoning" or that the system "transcends its components" without ablation studies. Claims about component contributions (e.g., "memory enables X") require comparative experiments (LLM-only, memory-ablated conditions). Describe observations, not causal claims.

### Reasoning Timeline

Track when each reasoning capability first appears in the logs:
- Iterations to first single-shot reasoning
- Iterations to cumulative induction (pattern from 3+ observations)
- Iterations to first falsification-driven principle
- Iterations to cross-domain transfer

Typical thresholds (from reference analysis):
| Capability | Typical First Appearance |
|------------|--------------------------|
| Single-shot reasoning | ~5 iterations |
| Cumulative induction | ~12 iterations |
| Principle via falsification | ~23 iterations |
| Cross-domain transfer | ~25 iterations |

---

## References

Allier, C., & Saalfeld, S. (2026). Understanding: an experiment-LLM-memory experiment. Janelia Research Campus, HHMI.
