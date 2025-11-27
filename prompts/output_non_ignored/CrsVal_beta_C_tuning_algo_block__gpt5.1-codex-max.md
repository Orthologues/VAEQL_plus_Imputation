Updated LaTeX algorithm block reflecting the current cross-validation driver and successive-halving option in `codex_CV_beta_C_fine_tuning_draft/DisentangledBetaVAE.py` and the helper utilities:

```tex
\captionof{algorithm}{\textbf{Cross-Validated $\beta$–$C$ Search with Optional Successive Halving}}
\label{algo:betaC_successive_halving}
\begin{algorithmic}[10]
  \Require{ \
  \begin{tabular}[t]{@{}l@{}}
  \bfseries Input/config: \\
  $X_{\mathrm{full}}, X_{\mathrm{miss}}$ from CSV paths \\ \Comment{via \textsc{GetScaledData}, returns scaled arrays + scaler $\mathcal{S}$ with NaNs restored in $X_{\mathrm{miss}}$} \\
  $K$ \Comment{fold count} \\
  $\mathcal{B},\mathcal{C}$ \Comment{either explicit grids or linspace from $\beta_{\text{range}}, C_{\text{range}}$} \\
  $\text{epoch\_granularity}(\beta)$, $\text{max\_epochs}(\beta)$ \Comment{per-$\beta$ chunk + cap} \\
  $\eta$ (Adam LR), $B$ (batch size), $R$ (recycle count), $M$ (multiple imputations) \\
  $\text{budgets}$ \Comment{halving_epoch_budgets if using successive halving} \\
  $\kappa$ \Comment{halving keep ratio} \\
  $\mathfrak{m}$ \Comment{halving metric (e.g., MAE)} \\
  \end{tabular}
  }
  \Statex
  \Procedure{\textit{RunCrossValidatedSearch}}{$d_{\text{index}}, \text{config}$}
    \State $d \gets d_{\text{index}} - 1$; \quad load config JSON
    \State $\text{use\_halving} \gets (\beta_{\text{range}} \land C_{\text{range}} \text{ present})$
    \State $\mathcal{B} \gets$ explicit grid or $\mathrm{linspace}(\beta_{\text{range}})$; \quad $\mathcal{C} \gets$ likewise
    \State $(X_{\mathrm{full}}, X_{\mathrm{miss}}, \mathcal{S}) \gets \textsc{GetScaledData}(\cdot,\text{return\_scaler}=1,\text{put\_nans\_back}=1)$
    \State $k \gets d \bmod K$; \quad $(X_{\mathrm{train}}, X_{\mathrm{val}}^{\mathrm{miss}}, X_{\mathrm{val}}^{\mathrm{full}}, \mathrm{NA\_idx}) \gets \textsc{SplitTrainingAndValidation}(k)$
    \If{\text{use\_halving}}
      \State $\mathcal{H} \gets$ all $(\beta,C) \in \mathcal{B} \times \mathcal{C}$ with fresh models/optimizers
      \For{$E \in \text{sorted}(\text{budgets})$}
        \For{$h \in \mathcal{H}$}
          \State $\Delta \gets E - h.\text{trained\_epochs}$; \quad \textsc{TrainOneFold}$(h.\text{model}, X_{\mathrm{train}}, \beta_h, C_h, \Delta, B, \eta)$
          \State $m \gets \textsc{EvaluateModel}(h.\text{model}, X_{\mathrm{val}}^{\mathrm{miss}}, X_{\mathrm{val}}^{\mathrm{full}}, \mathrm{NA\_idx}, \mathcal{S}, R, M)$
          \State $m.k \gets k$; \quad $m.\text{epoch} \gets E$; \quad \textsc{SaveResults}$(m,\beta_h,C_h)$
          \State $h.\text{score} \gets \textsc{SelectMetric}(m,\mathfrak{m})$; \quad $h.\text{trained\_epochs} \gets E$
        \EndFor
        \State $\mathcal{H} \gets$ top $\lceil \kappa \cdot |\mathcal{H}| \rceil$ by $\text{score}$ (drop others)
        \If{$|\mathcal{H}| = 1$} \textbf{break} \EndIf
      \EndFor
      \State $(\hat{\beta}, \hat{C}, \widehat{VAE}) \gets$ best in $\mathcal{H}$; \quad $\hat{E} \gets$ its trained epochs
    \Else
      \State $\beta \gets \mathcal{B}[(d // K) \bmod |\mathcal{B}|]$; \quad $C \gets \mathcal{C}[(d // (K \cdot |\mathcal{B}|)) \bmod |\mathcal{C}|]$
      \State $\epsilon_{\text{chunk}} \gets \text{epoch\_granularity}[\beta]$ (default fallback); \quad $\epsilon_{\max} \gets \text{max\_epochs}[\beta]$
      \State $\rho \gets \lfloor \epsilon_{\max} / \epsilon_{\text{chunk}} \rfloor + 1$; \quad init $VAE$, Adam($\eta$)
      \For{$r = 1$ \textbf{to} $\rho$}
        \State \textsc{TrainOneFold}$(VAE, X_{\mathrm{train}}, \beta, C, \epsilon_{\text{chunk}}, B, \eta)$
        \State $m \gets \textsc{EvaluateModel}(VAE, X_{\mathrm{val}}^{\mathrm{miss}}, X_{\mathrm{val}}^{\mathrm{full}}, \mathrm{NA\_idx}, \mathcal{S}, R, M)$
        \State $m.k \gets k$; \quad $m.\text{epoch} \gets r \cdot \epsilon_{\text{chunk}}$; \quad \textsc{SaveResults}$(m,\beta,C)$
      \EndFor
      \State $(\hat{\beta}, \hat{C}, \widehat{VAE}, \hat{E}) \gets (\beta, C, VAE, \rho \cdot \epsilon_{\text{chunk}})$
    \EndIf
    \State \textsc{SaveModelCheckpoint}$(\widehat{VAE}, \hat{\beta}, \hat{C}, k, \hat{E})$
  \EndProcedure
\end{algorithmic}
```

Next step: drop this block into your manuscript or README to replace the prior grid-search-only description; adjust notation if you rename config keys.