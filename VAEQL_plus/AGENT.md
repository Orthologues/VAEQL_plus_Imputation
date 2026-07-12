# Selected instructions from Open AI DeepResearch Plugin to what the Codex Agent shall execute during the next steps

## Following the commit `8b37a73`

### Issue 1 (4/4 SOLVED):
A crucial and indefensible caveat of the `Dis-$\beta$-VAEQL` algorithm: <br>
`The current encoder computes component-specific means and variances, but then appears to aggregate them into **a single posterior moment** before sampling. That is convenient computationally, but it is not the same thing as correctly sampling from a Gaussian mixture posterior, because mixture variance includes both within-component and between-component terms. This does not make the method unusable, but it is exactly the kind of issue a careful reviewer will notice. I would treat it as one of the first items to tighten before submission.`
#### My comments:
a. Address the first caveat in my current code, especially under `VAEQL_plus/beta_DVAE/torch_nn.py`. This is the foremost and an urgent issue before training the algorithm on PDS clinical datasets <b>(SOLVED)</b>.

b. Additionally, please use a Gumbel Softmax activation during decoding to enable reconstruction gradients to backpropagate through the categorical sample into `posterior_k_logits` <b>(SOLVED)</b>.

c. Additionally, ensure the monotonicity of the unary-encoded logits and activated probabilities for the values of each ordinal feature. Avoid using explicit ad-hoc cumulative function post-activation <b>(SOLVED)</b>.

d. Add SMOKE-TESTING for the aforementioned <b>a, b, c</b> fixes <b>(SOLVED)</b>.

### Issue 2 (1/2 SOLVED):
Another crucial yet defensible caveat of the `Dis-$\beta$-VAEQL` algorithm: <br>
`The repository uses a hybrid type-aware reconstruction objective, but is not fully
likelihood-native in the same sense as HI-VAE. The numeric, positive-real, and count
features rely on transformed-space reconstruction surrogates, while binary, categorical,
and ordinal features use likelihood-equivalent grouped losses/heads. That makes the method scientifically defensible if described honestly as a hybrid type-aware reconstruction objective, but it should not be overstated as already implementing the full heterogeneous likelihood machinery of HI-VAE.`
#### My comments:
a. We should keep the current norm of my code since it is not supposed to be `HIVAE` plus Q-learning, and my current hybrid type-aware reconstruction objective for heterogeneous feature types is defensible. Modify the comments to clarify my method <b>(SOLVED)</b>.

b. However, add reverse-transformation during `Dis-$\beta$-VAEQL`-reconstruction to the code under `VAEQL_plus/beta_DVAE` if it is currently absent (less urgent, not necessary for evaluation of the algorithm) <b>(TODO)</b>.
