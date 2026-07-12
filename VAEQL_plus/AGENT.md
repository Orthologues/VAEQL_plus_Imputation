# Selected instructions from Open AI DeepResearch Plugin to what the Codex Agent shall execute during the next steps

## Following the commit `8b37a73`

### Issue 1 (4/4 SOLVED):
Summary title: Component-wise GMM posterior sampling and differentiable categorical reconstruction.

A crucial and indefensible caveat of the `Dis-$\beta$-VAEQL` algorithm: <br>
`The current encoder computes component-specific means and variances, but then appears to aggregate them into **a single posterior moment** before sampling. That is convenient computationally, but it is not the same thing as correctly sampling from a Gaussian mixture posterior, because mixture variance includes both within-component and between-component terms. This does not make the method unusable, but it is exactly the kind of issue a careful reviewer will notice. I would treat it as one of the first items to tighten before submission.`
#### My comments:
a. Address the first caveat in my current code, especially under `VAEQL_plus/beta_DVAE/torch_nn.py`. This is the foremost and an urgent issue before training the algorithm on PDS clinical datasets <b>(SOLVED)</b>.

b. Additionally, please use a Gumbel Softmax activation during decoding to enable reconstruction gradients to backpropagate through the categorical sample into `posterior_k_logits` <b>(SOLVED)</b>.

c. Additionally, ensure the monotonicity of the unary-encoded logits and activated probabilities for the values of each ordinal feature. Avoid using explicit ad-hoc cumulative function post-activation <b>(SOLVED)</b>.

d. Add SMOKE-TESTING for the aforementioned <b>a, b, c</b> fixes <b>(SOLVED)</b>.

### Issue 2 (1/4 SOLVED):
Summary title: Hybrid type-aware reconstruction objective and Q-learning interface.

Another crucial yet defensible caveat of the `Dis-$\beta$-VAEQL` algorithm: <br>
`The repository uses a hybrid type-aware reconstruction objective, but is not fully
likelihood-native in the same sense as HI-VAE. The numeric, positive-real, and count
features rely on transformed-space reconstruction surrogates, while binary, categorical,
and ordinal features use likelihood-equivalent grouped losses/heads. That makes the method scientifically defensible if described honestly as a hybrid type-aware reconstruction objective, but it should not be overstated as already implementing the full heterogeneous likelihood machinery of HI-VAE.`
#### My comments:
a. We should keep the current norm of my code since it is not supposed to be `HIVAE` plus Q-learning, and my current hybrid type-aware reconstruction objective for heterogeneous feature types is defensible. Modify the comments to clarify my method <b>(SOLVED)</b>.

b. Document and preserve the processed-space interface between beta-DVAE reconstruction and Q-learning actions <b>(TODO)</b>. The transformed common space is not merely a shortcut: it lets the Q-agent use comparable normalized numeric actions such as `a in {-delta, 0, +delta}` across feature types. After adjustment, continuous features remain continuous, counts can be inverse-transformed and rounded, binary features can be thresholded or sampled, nominal features can be selected by grouped argmax or sampling, and ordinal outputs can be decoded through ordered cumulative probabilities. Raw Poisson/log-normal heads would force the RL action space to modify rates, means, variances, or sampled values differently for every feature family, which would complicate the MDP.

c. However, add reverse-transformation during `Dis-$\beta$-VAEQL`-reconstruction and raw-scale evaluations to the code under `VAEQL_plus/beta_DVAE` if it is currently absent (less urgent, not necessary for evaluation of the algorithm) <b>(TODO)</b>.

d. Add a smaller HIVAE-style likelihood-native ablation if time permits <b>(TODO)</b>. Here, likelihood-native means that each decoder head outputs feature-family distribution parameters and trains with negative log-likelihood on the corresponding scale. For example, an explicit Poisson count head would output a positive rate `lambda(z)` and optimize `-log Poisson(x; lambda(z))`, while an explicit log-normal positive-continuous head would output `mu(z)` and positive `sigma(z)` and optimize `-log LogNormal(x; mu(z), sigma(z))`. This differs from the current transformed-space approach, where count and positive-continuous features are reconstructed as processed common-space values rather than explicit Poisson/log-normal decoder likelihoods.
