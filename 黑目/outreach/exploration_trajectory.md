# Exploration trajectory: SAE feature decomposition non-uniqueness

## Project state

I am a CUHK-Shenzhen student. This is an independent theoretical exploration, not a paper I can fully defend alone. The current goal is to ask a researcher whether the direction is worth developing, not to present a finished contribution.

Local materials:

- Draft PDF: `SAE.pdf`
- Extended formalization: `SAE_extended/main.tex`
- Compiled extended PDF: `outreach/SAE_extended_draft.pdf`
- Version notes: `SAE_extended/README.md`
- Toy experiment report: `SAE_extended/experiments/results/report.md`

`xelatex` is not installed on this machine. The extended PDF was compiled with the smaller local Tectonic engine at `/home/jes/.cargo/bin/tectonic`.

## Core claim

SAE feature decomposition non-uniqueness may have a structural information-theoretic explanation.

1. Unconditional level:
   SAE training can be viewed as an unsupervised disentanglement problem. Under this framing, it falls into the Locatello 2019 non-identifiability regime. L1 and TopK sparsity priors do not obviously break the non-identifiability barrier.

2. Conditional level:
   Under an L-NL SAE architecture, three local constraints may combine into a Pareto-style lower bound on dictionary stability:
   O'Neill amortization gap, Cui interference matrix, and Bhalla incoherence.

## Known weak points

1. Proposition 3.1 / equivalence step:
   The move from SAE feature learning to unsupervised disentanglement has a logical jump. I need expert judgment on whether this is a legitimate reduction, only an analogy, or false.

2. Appendix A, step 3:
   The constants in the lower-bound derivation were not independently tracked from the original source proofs. The qualitative direction may survive, but the stated bound should not be treated as fully established.

3. Scaling-law predictions:
   A toy TopK-SAE experiment falsified two of the draft's main monotonic predictions:
   - predicted rho decreases with capacity L, but observed rho increased
   - predicted rho increases with observation dimension M, but observed rho decreased
   - predicted rho decreases with sparsity k, and this was weakly confirmed

## Current interpretation after toy experiments

The pure "zero-space random drift" explanation is probably too simple. The experiments suggest that SGD implicit bias and solution-space collapse may dominate in some regimes. This does not necessarily kill the identifiability framing, but it means the conditional scaling-law story needs revision.

## What I want feedback on

1. Is the Locatello / nonlinear-ICA framing for SAE feature decomposition a serious direction, or is the equivalence too weak to be useful?

2. Is there a defensible theorem-level path from sparse dictionary non-identifiability to cross-seed SAE feature instability?

3. Are the failed toy scaling laws a useful diagnostic showing a missing implicit-bias term, or do they indicate that the entire conditional lower-bound story is pointed in the wrong direction?

4. If the direction has value, what should be the next minimal technical step: fix the reduction, build a cleaner toy theorem, or run experiments on real LLM activations?
