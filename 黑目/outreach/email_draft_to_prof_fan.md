Subject: Seeking brief advice on SAE feature non-uniqueness and identifiability framing

Dear Professor Fan,

I am a student at CUHK-Shenzhen. I recently wrote an independent exploratory draft on a question in mechanistic interpretability: why sparse autoencoders can learn substantially different feature dictionaries across random seeds.

The core idea is to frame SAE feature decomposition as an identifiability problem. At the unconditional level, I try to connect SAE feature learning to unsupervised disentanglement / nonlinear ICA and the Locatello et al. non-identifiability barrier. At the conditional level, I try to combine constraints from amortization gap, interference, and incoherence into a lower-bound style explanation for cross-seed feature instability.

I want to be upfront: this is not yet a paper I can fully defend. In particular, the SAE-to-disentanglement equivalence step may be too strong, and part of the lower-bound derivation still has constants I have not independently traced. I also ran a toy TopK-SAE experiment and two of the draft's monotonic scaling predictions were falsified, which suggests the current story misses something like SGD implicit bias or solution-space collapse.

Because your work is close to machine learning, matrix/tensor methods, clustering, graph representation learning, and explainable representation learning, I wanted to ask whether you think this direction is worth developing. I would be grateful for even a short judgment on one question:

Does treating SAE feature instability as a sparse representation / identifiability problem look mathematically meaningful, or is the framing too weak to be useful?

I attach a short trajectory note and the current draft. If you think another faculty member would be a better fit for this question, I would also appreciate a pointer.

Thank you for your time.

Best regards,

[Your Name]
CUHK-Shenzhen
