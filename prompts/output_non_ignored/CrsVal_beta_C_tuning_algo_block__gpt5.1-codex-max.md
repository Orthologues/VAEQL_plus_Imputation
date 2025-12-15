Updated LaTeX algorithm block matching the current corner-halving cross-validation driver in `codex_CV_beta_C_fine_tuning_draft/DisentangledBetaVAE.py`:

```tex
\captionof{algorithm}{\textbf{Corner-Halving $\beta$–$C$ Search with $K$-Fold Evaluation}}
\label{algo:betaC_corner_halving}
\begin{algorithmic}[10]
  \Require{ \
  \begin{tabular}[t]{@{}l@{}}
  \bfseries Input/config: \\
  data paths $\to (X_{\mathrm{full}}, X_{\mathrm{miss}}, \mathcal{S})$ via \textsc{GetScaledData} \\ \Comment{$\mathcal{S}$ is the fitted scaler; $X_{\mathrm{miss}}$ keeps NaNs} \\
  $\beta_{\min}, \beta_{\max}, C_{\min}, C_{\max}$, accuracy $\alpha$ \\ \Comment{Default value $\alpha_0=0.01$} \\
  $K$ folds, batch size $B$, learning rate $\eta$, recycles $R$, multi-imputations $M$ \\
  $\text{budget variables:}$ \\ \Comment{halving\_epoch\_budgets; chunk = $\min$, cap = $\max$ (fallback to epoch\_chunk/max\_epochs)} \\
  tolerance $\tau$, patience $p$, optional $\text{min\_epochs}$, metric $\mathfrak{m}$ (default MAE) \\
  \end{tabular}
  }
  \Statex
  \Procedure{\textit{CornerHalvingSearch}}{$\text{config}$}
    \State load JSON config; $(X_{\mathrm{full}}, X_{\mathrm{miss}}, \mathcal{S}) \gets \textsc{GetScaledData}(\text{return\_scaler}=True,\text{put\_nans\_back}=True)$
    \State $(\beta_0, C_0) \gets (\beta_{\max}-\beta_{\min},\, C_{\max}-C_{\min})$; \quad $q^\star \gets \varnothing$
    \While{$(\beta_{\max}-\beta_{\min} > \alpha \beta_0) \lor (C_{\max}-C_{\min} > \alpha C_0)$}
      \State $\beta_m \gets (\beta_{\min}+\beta_{\max})/2$; \quad $C_m \gets (C_{\min}+C_{\max})/2$
      \State $\mathcal{Q} \gets \{(\beta_{\min},C_{\min}),(\beta_{\min},C_{\max}),(\beta_{\max},C_{\min}),(\beta_{\max},C_{\max})\}$
      \State $\mathcal{R} \gets$ \textsc{RunCandidateCVsInParallel}$(\mathcal{Q}, X_{\mathrm{full}}, X_{\mathrm{miss}}, \mathcal{S}, \text{config})$
      \State $r^\star \gets \arg\min_{r \in \mathcal{R}} r.\text{score}$ 
      \State $q^\star \gets r^\star.q$
      \State \textbf{update the grid towards the winning sub-grid}:
      \State \quad \textbf{if} $q^\star=(\beta_{\min},C_{\min})$ \textbf{then} $(\beta_{\max},C_{\max}) \gets (\beta_m,C_m)$
      \State \quad \textbf{elif} $q^\star=(\beta_{\min},C_{\max})$ \textbf{then} $(\beta_{\max},C_{\min}) \gets (\beta_m,C_m)$
      \State \quad \textbf{elif} $q^\star=(\beta_{\max},C_{\min})$ \textbf{then} $(\beta_{\min},C_{\max}) \gets (\beta_m,C_m)$
      \State \quad \textbf{else} $(\beta_{\min},C_{\min}) \gets (\beta_m,C_m)$
    \EndWhile
    \State \textsc{TrainVAEQLwithBestParams}$(q^\star.\beta, q^\star.C, X_{\mathrm{full}}, X_{\mathrm{miss}}, \mathcal{S}, \text{config})$
  \EndProcedure
  \Statex
  \Procedure{\textit{RunCandidateCV}}{$\beta, C, \text{config}$}
    \For{$k = 0$ \textbf{to} $K-1$}
      \State $(X_{\mathrm{train}}, X_{\mathrm{val}}^{\mathrm{miss}}, X_{\mathrm{val}}^{\mathrm{full}}, \mathrm{NA\_idx}) \gets \textsc{SplitTrainingAndValidation}(k)$
      \State $(\epsilon_{\text{chunk}}, \epsilon_{\max}) \gets \text{budgets}$; \quad init VAE + optimizer (Adam if configured)
      \State train in chunks until $\epsilon_{\max}$ or early-stop when metric $\mathfrak{m}$ stalls by $\tau$ for $p$ steps after $\text{min\_epochs}$
      \State after each chunk: $m \gets \textsc{EvaluateModel}(VAE, X_{\mathrm{val}}^{\mathrm{miss}}, X_{\mathrm{val}}^{\mathrm{full}}, \mathrm{NA\_idx}, \mathcal{S}, R, M)$
      \State annotate $m$ with fold $k$ and epoch; \textsc{SaveResults}$(m,\beta,C)$
      \State keep the best chunk metric for this fold
    \EndFor
    \State \Return mean fold metric as candidate score
  \EndProcedure
\end{algorithmic}
```
