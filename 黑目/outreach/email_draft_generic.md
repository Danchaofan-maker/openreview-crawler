Subject: Seeking brief advice on SAE feature non-uniqueness and identifiability

Dear Professor [Name],

I am a student at CUHK-Shenzhen. I recently wrote an independent exploratory draft on sparse autoencoders in mechanistic interpretability, focusing on why learned SAE feature dictionaries may be non-unique across random seeds.

The main hypothesis is that SAE feature decomposition has a structural identifiability issue:

1. unconditionally, SAE feature learning resembles unsupervised disentanglement / nonlinear ICA and may fall under the Locatello et al. non-identifiability barrier;
2. conditionally, under an L-NL SAE architecture, amortization gap, interference, and incoherence constraints may imply a lower-bound style tradeoff for feature stability.

I do not want to overstate the draft. The SAE-to-disentanglement equivalence step is currently the weakest part, and one lower-bound step still has constants I have not independently traced. A toy TopK-SAE experiment also falsified two of my proposed scaling-law predictions, so the current version likely misses an important optimization/implicit-bias mechanism.

I am writing to ask for a short expert judgment: is this identifiability framing a direction worth developing, or is the reduction too weak to support a serious theory project?

I attach the current draft and a one-page trajectory note listing the claims, weak points, and toy-experiment outcome. If this is outside your area, I would be grateful for any suggestion of a better person to ask.

Thank you for your time.

Best regards,

[Your Name]
CUHK-Shenzhen
