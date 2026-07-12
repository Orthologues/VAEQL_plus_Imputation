# Selected instructions from Open AI DeepResearch Plugin to what the Codex Agent shall execute during the next steps

## Following the commit `8b37a73`

### Issue 1 (4/4 SOLVED):
Summary title: Component-wise GMM posterior sampling and differentiable categorical reconstruction.

A crucial and indefensible caveat of the `Dis-$\beta$-VAEQL` algorithm: <br>
`The current encoder computes component-specific means and variances, but then appears to aggregate them into **a single posterior moment** before sampling. That is convenient computationally, but it is not the same thing as correctly sampling from a Gaussian mixture posterior, because mixture variance includes both within-component and between-component terms. This does not make the method unusable, but it is exactly the kind of issue a careful reviewer will notice. I would treat it as one of the first items to tighten before submission.`

#### TODOs:
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

#### TODOs:
a. We should keep the current norm of my code since it is not supposed to be `HIVAE` plus Q-learning, and my current hybrid type-aware reconstruction objective for heterogeneous feature types is defensible. Modify the comments to clarify my method <b>(SOLVED)</b>.

b. Document and preserve the processed-space interface between beta-DVAE reconstruction and Q-learning actions <b>(TODO)</b>. The transformed common space is not merely a shortcut: it lets the Q-agent use comparable normalized numeric actions such as `a in {-delta, 0, +delta}` across feature types. After adjustment, continuous features remain continuous, counts can be inverse-transformed and rounded, binary features can be thresholded or sampled, nominal features can be selected by grouped argmax or sampling, and ordinal outputs can be decoded through ordered cumulative probabilities. Raw Poisson/log-normal heads would force the RL action space to modify rates, means, variances, or sampled values differently for every feature family, which would complicate the MDP.

c. Add a separate PDS/Ministral adapter metadata layer before compiling clinical source variables into the compact model-facing `FeatureTypeDict` <b>(TODO)</b>. This metadata should preserve `source_feature`, `canonical_feature`, `model_type`, `num_levels`, raw-to-canonical mappings, missing-value codes, evidence source, and confidence. For example, raw `ECOGPS` (ECOG scores for cancer treatments) can be represented as canonical ordinal `ecog` with six levels and then compiled into `{"ord_feats": {"ecog": 6}}`.

d. Add a smaller HIVAE-style likelihood-native ablation if time permits <b>(TODO)</b>. Here, likelihood-native means that each decoder head outputs feature-family distribution parameters and trains with negative log-likelihood on the corresponding scale. For example, an explicit Poisson count head would output a positive rate `lambda(z)` and optimize `-log Poisson(x; lambda(z))`, while an explicit log-normal positive-continuous head would output `mu(z)` and positive `sigma(z)` and optimize `-log LogNormal(x; mu(z), sigma(z))`. This differs from the current transformed-space approach, where count and positive-continuous features are reconstructed as processed common-space values rather than explicit Poisson/log-normal decoder likelihoods.

### Issue 3 (0/3 SOLVED):
Summary title: Primary transformed-space metrics with secondary raw-scale interpretability checks.

Research-mode recommendation: do not completely omit reverse transformation, but use it in a limited and clearly scoped way. The primary benchmark should remain in processed/transformed space because this is the modeling and Q-learning action space. Raw-scale reverse-transformed metrics should be reported only as secondary interpretability checks for selected clinically meaningful variables.

#### TODOs:
a. Add evaluation support for a two-tier metric report <b>(TODO)</b>.

| Family | Metric |
| --- | --- |
| continuous | transformed MAE/RMSE |
| positive continuous | transformed MAE/RMSE |
| count | log1p-space MAE/RMSE |
| binary | AUROC, F1 |
| categorical | macro-F1 |
| ordinal | ordinal MAE / weighted kappa |

This keeps the primary benchmark aligned with the transformed modeling/Q-learning space while using reverse transformation only for limited interpretability checks.

b. Add a small manuscript-facing secondary raw-scale table for clinically important variables only <b>(TODO)</b>. Candidate variables include `age, BMI, baseline lab values, ECOG, tumor burden, selected blood counts, and selected treatment-history counts`. This table should help readers interpret whether a transformed-space error such as standardized `RMSE = 0.42` is clinically meaningful, without requiring raw-scale metrics for every feature.

c. Add inverse transformation only for the selected clinically important variables used by TODO `b` <b>(TODO)</b>. This should support demonstration and manuscript interpretability, not become the default metric path for every feature. The implementation should recover raw-scale values for the chosen variables before computing the secondary raw-scale table.
