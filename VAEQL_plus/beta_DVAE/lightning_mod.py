"""
Disentangled Beta-VAE PyTorch Lightning training implementation.

This module contains the LightningModule wrapper and cross-validation utilities
for beta-capacity training, fold-level evaluation, and corner-halving beta/C
fine tuning.

Mask-value semantics:
- 0: observed after preprocessing and pyampute; used for reconstruction-loss weighting.
- 1: originally missing in the raw/preprocessed data before pyampute.
- 2: missing introduced by pyampute before beta-C validation.
- 3: pre-validation type-0 cell re-amputated for beta-C validation metrics.
- 4: pre-validation type-1/type-2 cell selected again during beta-C validation;
     excluded from validation metrics.

Two masks are threaded through beta-C validation:
- `validation_mask`: 0/1/2/3/4 mask used for beta-C evaluation metrics.
- `imputation_mask`: original 0/1/2 mask passed to beta-DVAE imputation methods.

Author: Jiawei Zhao (jiz@imada.sdu.dk)
Date: 2026-04-01
"""

import os
import portalocker  # OS-level advisory lock with timeout; safer than sleep-polling lock files.
import math
import re
import io
import numpy as np
import pandas as pd
from typing import Set, Dict, Tuple, List, NamedTuple
from multiprocessing import Pool, cpu_count
from typing_extensions import override

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import lightning.pytorch as torch_lit

from VAEQL_plus.conf.config import DisentangledBetaVaeTuningConfig
from VAEQL_plus.conf.feat_types import FeaturesTypeDict
from .torch_nn import BetaGausMixedDVAE
from .utils import BetaGausMixedDVAEUtils, ArrayLike

class BetaGausMixedDVAETrainer(torch_lit.LightningModule):
    """
    PyTorch-Lightning training pipeline plus beta/C evaluation utilities.

    GMM-KL and beta-capacity loss are owned by `BetaGausMixedDVAE`.
    This class wraps that module as a LightningModule for fold-level optimization,
    while keeping static helper methods used for:
    - fold-level training and validation
    - candidate evaluation for beta/C tuning
    """
    
    ####################################################################################################
    # 1. Inherited Named Tuples for annotation of the return types of methods & Constants for this Class
    # `validation_input`: numeric validation fold input after beta-C validation amputation; pre-imputed after amputation, leaving no NaN values for beta-DVAE
    # `validation_mask`: 0/1/2/3/4 mask for beta-C validation semantics
    # `imputation_mask`: original 0/1/2 mask for beta-DVAE imputation semantics; unchanged by beta-C validation amputation
    ####################################################################################################
    
    # Candidate metric semantics:
    # - "mae": final loss from one `impute_single` refresh trajectory.
    # - "rmse": final RMSE from one `impute_single` refresh trajectory.
    # - "multi_mae": MAE of the mean prediction over multiple imputations.
    # - "multi_rmse": RMSE of the mean prediction over multiple imputations.
    # - "prop_*q": empirical quantile-interval coverage from multiple imputations.
    CAND_METRIC_KEYS: Set[str] = {
        "mae",
        "rmse",
        "multi_mae",
        "multi_rmse",
        "prop_80q",
        "prop_90q",
        "prop_95q",
        "prop_99q",
    }
    DEFAULT_CAND_METRIC_WEIGHTS: Dict[str, float] = {
        "mae": 0.15,
        "rmse": 0.15,
        "multi_mae": 0.25,
        "multi_rmse": 0.25,
        "prop_90q": 0.05,
        "prop_95q": 0.10,
        "prop_80q": 0.05,
        "prop_99q": 0.0,
    }
    
    class SplitTrainingValidationOutput(NamedTuple):
        """
        K-fold split output.
        `training_input` is the fold-updated numeric training matrix, with the same number of columns as `validation_input`.
        """
        training_input: ArrayLike
        validation_input: ArrayLike
        validation_mask: ArrayLike
        pre_val_amputation_mask: ArrayLike
    
    class TrainOneFoldOutput(NamedTuple):
        """
        Last mini-batch loss components observed by Lightning after one training chunk.
        `last_loss`: the most recent sum of the reconstruction loss `last_recon` and the beta-weighted and capacity-C-adjusted total KL-divergence `last_kl`.
        `last_kl`: the most recent total KL divergence of the Gaussian Mixture latent space, equal to the sum of `last_kl_disc` and `last_kl_cont`.
        `last_kl_disc`: the most recent KL divergence of the discrete latent variables in the Gaussian Mixture Model.
        `last_kl_cont`: the most recent KL divergence of the continuous latent variables in the Gaussian Mixture Model.
        """
        last_loss: float
        last_recon: float
        last_kl: float
        last_kl_disc: float
        last_kl_cont: float

    class TrainFoldEvalOutput(NamedTuple):
        """Best validation score, its metric name, and completed epochs for one fold."""
        best_metric_val: float
        selection_metric_name: str
        completed_epochs: int
    
    class EvaluateModelOutput(NamedTuple):
        """Validation metrics returned by `evaluate_model`."""
        mae: float
        rmse: float
        multi_mae: float
        multi_rmse: float
        prop_80q: float
        prop_90q: float
        prop_95q: float
        prop_99q: float
        

    ####################################################################################################
    # 2. Non-Static Methods
    ####################################################################################################
    def __init__(
        self,
        model: BetaGausMixedDVAE,
        beta: float,
        capacity_C: float,
        feat_type_dict: FeaturesTypeDict,
        lr: float,
        use_adam: bool,
        num_feat_loss_metric: str,
        optimizer: torch.optim.Optimizer,
        results_path: str = "beta_analysis.csv",
        log_imp_interval: int = 10,
    ):
        super().__init__()
        # config validation
        self.num_feat_loss_metric = self._normalize_num_feat_loss_metric(num_feat_loss_metric)
        # model and training params
        self.model = model
        self.beta = float(beta)
        self.capacity_C = float(capacity_C)
        self.feat_type_dict = feat_type_dict
        self.lr = float(lr)
        self.use_adam = bool(use_adam)
        self.external_optimizer = optimizer
        self.results_path = str(results_path)
        self.log_imp_interval = int(log_imp_interval)
        self.tau_epoch_idx = -1
        self.last_loss: float = float("nan")
        self.last_recon: float = float("nan")
        self.last_kl: float = float("nan")
        self.last_kl_disc: float = float("nan")
        self.last_kl_cont: float = float("nan")


    @override
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.model(x)


    @override
    def on_train_epoch_start(self) -> None:
        self.tau_epoch_idx += 1
        self.model.gumbel_softmax_tau = self.model.gumbel_softmax_tau_at_epoch(self.tau_epoch_idx)


    @override
    def training_step(
        self,
        batch: torch.Tensor | Tuple[torch.Tensor, ...] | List[torch.Tensor],
    ) -> torch.Tensor:
        input_t = batch[0] if isinstance(batch, (list, tuple)) else batch
        output_t = self.model(input_t)
        obs_mask = torch.zeros_like(input_t, dtype=torch.int8)
        loss_output = self.model.beta_capacity_loss(
            recon_logits=output_t["recon_logits"],
            x_obs_processed=input_t,
            posterior_z_component_mean=output_t["posterior_z_component_mean"],
            posterior_z_component_logvar=output_t["posterior_z_component_logvar"],
            posterior_k_probs=output_t["posterior_k_probs"],
            beta=self.beta,
            capacity_C=self.capacity_C,
            gmm_prior=self.model.get_gmm_prior_params(),
            feat_type_dict=self.feat_type_dict,
            obs_mask=obs_mask,
            num_feat_loss_metric=self.num_feat_loss_metric,
        )
        self.last_loss = float(loss_output.total.detach().item())
        self.last_recon = float(loss_output.recon_loss.detach().item())
        self.last_kl = float(loss_output.kl_mean.detach().item())
        self.last_kl_disc = float(loss_output.kl_disc_mean.detach().item())
        self.last_kl_cont = float(loss_output.kl_cont_mean.detach().item())
        BetaGausMixedDVAETrainer._log_training_losses(self, loss_output)
        return loss_output.total


    @override
    def configure_optimizers(self) -> torch.optim.Optimizer:
        if self.external_optimizer is not None:
            return self.external_optimizer
        return (
            torch.optim.Adam(self.model.parameters(), lr=self.lr)
            if self.use_adam
            else torch.optim.SGD(self.model.parameters(), lr=self.lr)
        )
        

    ####################################################################################################
    # 3. Static Helpers for training, validating and evaluating beta-DVAE
    ####################################################################################################
    @staticmethod
    def _log_training_losses(module: "BetaGausMixedDVAETrainer", loss_output) -> None:
        # `global_step` is a Lightning-managed counter populated during `trainer.fit(...)`.
        log_on_step = (int(module.global_step) % int(module.log_imp_interval)) == 0
        module.log("train_total_loss", loss_output.total, prog_bar=False, on_step=log_on_step, on_epoch=True)
        module.log("train_recon_loss", loss_output.recon_loss, prog_bar=False, on_step=log_on_step, on_epoch=True)
        module.log("train_kl_mean", loss_output.kl_mean, prog_bar=False, on_step=log_on_step, on_epoch=True)
        module.log("train_kl_disc_mean", loss_output.kl_disc_mean, prog_bar=False, on_step=log_on_step, on_epoch=True)
        module.log("train_kl_cont_mean", loss_output.kl_cont_mean, prog_bar=False, on_step=log_on_step, on_epoch=True)

    @staticmethod
    def _normalize_num_feat_loss_metric(num_feat_loss_metric: str) -> str:
        normalized = re.sub(r"[_-\.\s]+", "", str(num_feat_loss_metric).upper())
        if normalized in {"RMSE", "MAE"}:
            return normalized
        raise ValueError(
            "num_feat_loss_metric must resolve to 'RMSE' or 'MAE', "
            f"got {num_feat_loss_metric!r}"
        )

    @staticmethod
    def save_results(
        results: Dict,
        epoch: int,
        beta: float,
        capacity_C: float,
        results_path='beta_analysis.csv',
        lock_path='lock.txt'
    ):
        def _parse_s3_path(path: str) -> Tuple[str, str] | None:
            # Conventional S3 URL shape:
            #   s3://<bucket-name>/<object-key>
            # Example:
            #   s3://vaeql-pds-prod/beta_dvae/results/beta_analysis.csv
            if re.match(r"^\s*s3://", str(path)) is None:
                return None
            # S3 bucket naming rules cap bucket names at 63 chars.
            regex_m = re.match(
                r"^\s*s3://([a-z0-9][a-z0-9.-]{1,61}[a-z0-9])/(.+?)\s*$",
                str(path),
            )
            if regex_m is None:
                raise ValueError(f"Invalid S3 path: {path!r}. Expected format: s3://$bucket/$key")
            bucket, key = regex_m.group(1), regex_m.group(2)
            if not bucket or not key:
                raise ValueError(f"Invalid S3 path: {path!r}")
            return bucket, key

        def _new_imputation_info_dict() -> Dict[str, float | int | None]:
            return {
                "beta": beta,
                "C": capacity_C,
                "epoch": epoch,
                "selection_metric_name": results.get("selection_metric_name", "weighted_composite_metric"),
                "fold_idx": results.get("fold_idx", None),
                "mae": results.get("mae", None),
                "rmse": results.get("rmse", None),
                "multi_mae": results.get("multi_mae", None),
                "multi_rmse": results.get("multi_rmse", None),
                "kl_disc": results.get("kl_disc", None),
                "kl_cont": results.get("kl_cont", None),
                "prop_80q": results.get("prop_80q", None),
                "prop_90q": results.get("prop_90q", None),
                "prop_95q": results.get("prop_95q", None),
                "prop_99q": results.get("prop_99q", None),
            }

        # The branch of storing results on AWS-S3 if the path starts with "s3://"; 
        # otherwise, store results on local disk with file locking for safe concurrent appends
        s3_location = _parse_s3_path(results_path)
        if s3_location is not None:
            try:
                import boto3 # only for internet-based S3 access
                from botocore.exceptions import ClientError  # only for internet-based S3 access
            except Exception as e:
                raise RuntimeError(
                    "Saving results to s3:// requires `boto3` and AWS credentials."
                ) from e

            bucket, key = s3_location
            s3_client = boto3.client("s3")
            try:
                obj = s3_client.get_object(Bucket=bucket, Key=key)
                csv_text = obj["Body"].read().decode("utf-8")
                df_prev = pd.read_csv(io.StringIO(csv_text))
            except ClientError as e:
                err_code = e.response.get("Error", {}).get("Code")
                err_status = e.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
                if err_code == "NoSuchKey" or err_status == 404:
                    df_prev = pd.DataFrame()
                else:
                    raise RuntimeError(
                        "Failed to read existing S3 results CSV "
                        f"(bucket={bucket!r}, key={key!r}, "
                        f"error_code={err_code!r}, http_status={err_status!r})."
                    ) from e
            info_row = _new_imputation_info_dict()
            df_prev = pd.concat([df_prev, pd.DataFrame([info_row])], ignore_index=True)
            out_buffer = io.StringIO()
            df_prev.to_csv(out_buffer, index=False)
            s3_client.put_object(Bucket=bucket, Key=key, Body=out_buffer.getvalue().encode("utf-8"))
            return

        # The branch of storing result on the local filesystem with file locking to ensure safe concurrent appends by multi-processes;
        # holds an OS-level advisory lock to serialize read/append/write cycles.
        with portalocker.Lock(lock_path, mode="a+", timeout=2):
            if not os.path.exists(results_path):
                with open(results_path, 'w') as fh:
                    fh.write('beta,C,epoch,selection_metric_name,mae,rmse,multi_mae,multi_rmse,'
                             'kl_disc,kl_cont,'
                             'prop_80q,prop_90q,prop_95q,prop_99q,fold_idx\n')
        
            df_prev = pd.read_csv(results_path)
        
            info_row = _new_imputation_info_dict()
        
            df_prev = pd.concat([df_prev, pd.DataFrame([info_row])], ignore_index=True)
            df_prev.to_csv(results_path, index=False)
    
    
    @staticmethod
    def evaluate_model(
        model: BetaGausMixedDVAE,
        validation_input: ArrayLike,
        validation_ref: ArrayLike,
        validation_mask: ArrayLike,
        imputation_mask: ArrayLike,
        feat_type_dict: FeaturesTypeDict,
        num_imputations: int,
        single_impute_max_iter: int = 5,
    ) -> "BetaGausMixedDVAETrainer.EvaluateModelOutput":
        """
        Evaluate imputation quality on the validation fold.

        This method computes:
        - single-imputation endpoint metrics: `mae`, `rmse`
        - multiple-imputation aggregate metrics: `multi_mae`, `multi_rmse`
        - quantile-coverage metrics on validation-mask type-3 cells:
          `prop_80q`, `prop_90q`, `prop_95q`, `prop_99q`

        `num_imputations` controls how many stochastic imputation samples are drawn.
        `single_impute_max_iter` controls the iterative-refresh budget within each
        `impute_multiple` call (a single imputation) and is intentionally independent from
        `num_imputations`.

        `validation_input` must be pre-imputed (no NaN values).
        `validation_ref` is the reference matrix used for metric computation.
        `validation_mask` encodes beta-C validation semantics (including type-3 metric cells). 0/1/2/3/4 values are expected.
        `imputation_mask` is the original 0/1/2 mask used by beta-DVAE imputation routines.
        """
        
        validation_input_t = BetaGausMixedDVAEUtils.as_torch_tensor(
            validation_input,
            device=model.device,
            dtype=torch.float32,
        )
        validation_ref_t = BetaGausMixedDVAEUtils.as_torch_tensor(
            validation_ref,
            device=model.device,
            dtype=torch.float32,
        )
        validation_mask_t = BetaGausMixedDVAEUtils.as_torch_tensor(
            validation_mask,
            device=model.device,
            dtype=torch.int8,
        )
        imputation_mask_t = BetaGausMixedDVAEUtils.as_torch_tensor(
            imputation_mask,
            device=model.device,
            dtype=torch.int8,
        )

        if bool(torch.isnan(validation_input_t).any().item()):
            raise ValueError("`validation_input` must be pre-imputed and contain no NaN values.")
        if validation_ref_t.shape != validation_input_t.shape:
            raise ValueError(
                f"`validation_ref` must have shape {tuple(validation_input_t.shape)}, "
                f"got {tuple(validation_ref_t.shape)}"
            )
        if validation_mask_t.shape != validation_input_t.shape:
            raise ValueError(
                f"`validation_mask` must have shape {tuple(validation_input_t.shape)}, "
                f"got {tuple(validation_mask_t.shape)}"
            )
        if imputation_mask_t.shape != validation_input_t.shape:
            raise ValueError(
                f"`imputation_mask` must have shape {tuple(validation_input_t.shape)}, "
                f"got {tuple(imputation_mask_t.shape)}"
            )

        # multiple imputations for coverage
        multi_imputes_t = torch.empty(
            (int(num_imputations), *tuple(validation_input_t.shape)),
            device=model.device,
            dtype=torch.float32,
        )
        # populate the multiple imputation tensor
        for mi_idx in range(num_imputations):
            imputed_sample_t = model.impute_multiple(
                validation_input_t,
                max_iter=single_impute_max_iter,
                X_mask=imputation_mask_t,
                feat_type_dict=feat_type_dict,
            )
            multi_imputes_t[mi_idx] = imputed_sample_t.detach().to(
                device=model.device,
                dtype=torch.float32,
            )
    
        # evaluate multiple-imputation metrics
        coverage_q = BetaGausMixedDVAEUtils.evaluate_coverage_quantile(
            multi_imputes=multi_imputes_t,
            ref_data_arr=validation_ref_t,
            X_mask=validation_mask_t,
        )
        multi_impute_mean_t = torch.mean(multi_imputes_t, dim=0)
        # remember that type-3 cells are the only ones included in the beta-C validation metrics; 
        # type-4 cells are excluded from metrics since they don't have ground-truth reference values, 
        # and type-0/1/2 cells are not part of the beta-C validation semantics at all
        mask_type3_t = BetaGausMixedDVAEUtils.mask_validation_type3_indices(
            validation_mask_t,
            tuple(validation_ref_t.shape),
        ).to(device=model.device)
        multi_pred_type3_vals = multi_impute_mean_t[mask_type3_t]
        ref_type3_vals = validation_ref_t[mask_type3_t]
        multi_mae = float(torch.mean(torch.abs(multi_pred_type3_vals - ref_type3_vals)).item())
        multi_rmse = float(torch.sqrt(torch.mean((multi_pred_type3_vals - ref_type3_vals) ** 2)).item())
    
        impute_single_output = model.impute_single(
            validation_input_t,
            n_cycles=10,
            loss_option='BOTH',
            X_mask=imputation_mask_t,
            feat_type_dict=feat_type_dict,
        )
        final_mae = float(impute_single_output.losses_mae[-1])
        final_rmse = float(impute_single_output.losses_rmse[-1])
    
        return BetaGausMixedDVAETrainer.EvaluateModelOutput(
            mae=final_mae,
            rmse=final_rmse,
            multi_mae=multi_mae,
            multi_rmse=multi_rmse,
            prop_80q=float(coverage_q["prop_80q"]),
            prop_90q=float(coverage_q["prop_90q"]),
            prop_95q=float(coverage_q["prop_95q"]),
            prop_99q=float(coverage_q["prop_99q"]),
        )
    
    
    @staticmethod
    def split_training_and_validation(
        config: DisentangledBetaVaeTuningConfig,
        k_fold_idx: int,
        amputated_data_for_val: ArrayLike,
        preimputed_data_for_ref: ArrayLike,
        amputation_mask: ArrayLike,
    ) -> "BetaGausMixedDVAETrainer.SplitTrainingValidationOutput":
        """
        Split one fold for beta-C validation.

        `preimputed_data_for_ref` defines the total row count used for
        K-fold boundaries.
        `amputated_data_for_val` provides the base missing-data matrix from
        which the selected fold is extracted.
        `amputation_mask` provides the aligned 0/1/2 missingness mask for that fold.

        The selected fold is used to generate `validation_input`,
        `validation_mask`, and `imputation_mask`, while `training_input`
        remains the full base missing-data matrix without reinjecting
        validation-time amputation.
        """
        n_rows = len(preimputed_data_for_ref)
        if len(amputated_data_for_val) != n_rows:
            raise ValueError(
                f"`amputated_data_for_val` row count must match "
                f"`preimputed_data_for_ref`; "
                f"got {len(amputated_data_for_val)!r} and {n_rows!r}."
            )
        if len(amputation_mask) != n_rows:
            raise ValueError(
                f"`amputation_mask` row count must match `preimputed_data_for_ref`; "
                f"got {len(amputation_mask)!r} and {n_rows!r}."
            )
        k_folds = int(config["k_folds"])
        if k_folds < 5 or k_folds > 10:
            raise ValueError(f"`k_folds` must be >= 5 and <= 10, got {k_folds}")
        if not (0 <= int(k_fold_idx) < k_folds):
            raise ValueError(f"`k_fold_idx` must satisfy 0 <= idx < {k_folds}, got {k_fold_idx}")

        # Balanced fold boundaries without remainder drift.
        fold_bounds = np.linspace(0, n_rows, num=k_folds + 1, dtype=int)
        start_index = int(fold_bounds[k_fold_idx])
        end_index = int(fold_bounds[k_fold_idx + 1])
        if end_index <= start_index:
            raise ValueError(
                f"Selected fold must be non-empty, got start_index={start_index!r} "
                f"and end_index={end_index!r}."
            )
    
        current_fold = amputated_data_for_val[start_index:end_index]
        current_fold_mask = amputation_mask[start_index:end_index]
        if "features_type_dict" not in config:
            raise KeyError("Missing required config key: `features_type_dict`")
        feat_type_dict: FeaturesTypeDict = dict(config["features_type_dict"])

        current_fold_np = (
            current_fold.detach().cpu().numpy()
            if isinstance(current_fold, torch.Tensor)
            else np.asarray(current_fold)
        )
        current_fold_mask_np = (
            current_fold_mask.detach().cpu().numpy()
            if isinstance(current_fold_mask, torch.Tensor)
            else np.asarray(current_fold_mask)
        )
        complete_row_index = np.where(np.isfinite(current_fold_np).all(axis=1))[0]
        amputation_output = BetaGausMixedDVAEUtils.generate_validation_amputation(
            preimputed_data_arr=current_fold_np[complete_row_index],
            X_mask=current_fold_mask_np[complete_row_index],
            feat_type_dict=feat_type_dict,
            prop_miss_rows=1,
            prop_miss_col=0.1
        )
        validation_input = amputation_output.validation_input
        validation_mask = amputation_output.validation_mask
        pre_val_amputation_mask = amputation_output.val_fold_baseline_amputation_mask
        training_input = BetaGausMixedDVAEUtils.as_torch_tensor(
            amputated_data_for_val,
            dtype=torch.float32,
        ).clone()
    
        return BetaGausMixedDVAETrainer.SplitTrainingValidationOutput(
            training_input=BetaGausMixedDVAEUtils.as_torch_tensor(training_input, dtype=torch.float32),
            validation_input=BetaGausMixedDVAEUtils.as_torch_tensor(validation_input, dtype=torch.float32),
            validation_mask=BetaGausMixedDVAEUtils.as_torch_tensor(validation_mask, dtype=torch.int8),
            pre_val_amputation_mask=BetaGausMixedDVAEUtils.as_torch_tensor(pre_val_amputation_mask, dtype=torch.int8),
        )
    
    
    #########################################################
    # 3. Training loop for one fold
    #########################################################
    @staticmethod
    def train_one_fold(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        use_adam: bool,
        lr: float,
        train_tensor: torch.Tensor,
        epochs: int,
        beta: float,
        capacity_C: float,
        device: torch.device,
        feat_type_dict: FeaturesTypeDict,
        num_feat_loss_metric: str,
        # We expect the cohort sizes of the to-be-imputed clinical datasets to range between circa 3K-30K, therefore, 
        # a batch size of 128 should be reasonable for stable training without out of memory (OOM) on a single GPU; 
        # this can be tuned if needed
        batch_size: int = 128,
        use_check_point: bool = False 
    ) -> "BetaGausMixedDVAETrainer.TrainOneFoldOutput":
        """
        Train VAE for `epochs` on the given fold's training data with Lightning.
        Returns the latest mini-batch loss tuple:
        (last_loss, last_recon, last_kl, last_kl_disc, last_kl_cont).
        """
        idxed_train_t = TensorDataset(train_tensor)
        loader = DataLoader(idxed_train_t, batch_size=batch_size, shuffle=True, drop_last=False)
        lightning_mod = BetaGausMixedDVAETrainer(
            model=model,
            beta=beta,
            capacity_C=capacity_C,
            feat_type_dict=feat_type_dict,
            lr=float(lr),
            use_adam=bool(use_adam),
            num_feat_loss_metric=num_feat_loss_metric,
            optimizer=optimizer,
        )
        lightning_mod.to(device)
        accelerator = "gpu" if device.type == "cuda" else "cpu"
        trainer = torch_lit.Trainer(
            max_epochs=int(epochs),
            accelerator=accelerator,
            devices=1,
            logger=False,
            enable_checkpointing=use_check_point,
            enable_model_summary=True,
            enable_progress_bar=True,
        )
        trainer.fit(lightning_mod, loader)
    
        return BetaGausMixedDVAETrainer.TrainOneFoldOutput(
            last_loss=lightning_mod.last_loss,
            last_recon=lightning_mod.last_recon,
            last_kl=lightning_mod.last_kl,
            last_kl_disc=lightning_mod.last_kl_disc,
            last_kl_cont=lightning_mod.last_kl_cont,
        )
    
    
    #########################################################
    # 4. Optimizer / convergence helpers
    #########################################################
    
    @staticmethod
    def train_fold_with_eval(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        use_adam: bool,
        lr: float,
        train_tensor: torch.Tensor,
        beta: float,
        capacity_C: float,
        device: torch.device,
        feat_type_dict: FeaturesTypeDict,
        num_feat_loss_metric: str,
        batch_size: int,
        start_epoch: int,
        epochs_between_evals: int,
        max_epoch: int,
        metric_weights: Dict[str, float],
        validation_input: ArrayLike,
        validation_mask: ArrayLike,
        imputation_mask: ArrayLike,
        num_imputations: int,
        fold_idx: int,
        results_path: str,
        metric_improvement_tolerance: float,
        max_non_improved_rounds: int,
        min_epochs: int,
        lock_path: str = "lock.txt"
    ) -> "BetaGausMixedDVAETrainer.TrainFoldEvalOutput":
        """
        Train one fold with periodic evaluation and simple convergence check.
        Returns (best_metric_val, epochs_ran).
        """
        best_metric_val = math.inf
        non_improved_rounds_count = 0
        completed_epochs = 0
        selection_metric_name: str | None = None
        validation_input_t = BetaGausMixedDVAEUtils.as_torch_tensor(
            validation_input,
            device=device,
            dtype=torch.float32,
        )
        validation_mask_t = BetaGausMixedDVAEUtils.as_torch_tensor(
            validation_mask,
            device=device,
            dtype=torch.int8,
        )
        imputation_mask_t = BetaGausMixedDVAEUtils.as_torch_tensor(
            imputation_mask,
            device=device,
            dtype=torch.int8,
        )
        while completed_epochs < max_epoch:
            epochs_this_round = (
                start_epoch
                if completed_epochs == 0
                else epochs_between_evals
            )
            epochs_this_round = min(epochs_this_round, max_epoch - completed_epochs)
            train_output = BetaGausMixedDVAETrainer.train_one_fold(
                model=model,
                optimizer=optimizer,
                use_adam=use_adam,
                lr=lr,
                train_tensor=train_tensor,
                epochs=epochs_this_round,
                beta=beta,
                capacity_C=capacity_C,
                device=device,
                feat_type_dict=feat_type_dict,
                num_feat_loss_metric=num_feat_loss_metric,
                batch_size=batch_size,
            )
            completed_epochs += epochs_this_round
    
            eval_output = BetaGausMixedDVAETrainer.evaluate_model(
                model=model,
                validation_input=validation_input_t,
                validation_ref=validation_input_t,
                validation_mask=validation_mask_t,
                imputation_mask=imputation_mask_t,
                feat_type_dict=feat_type_dict,
                num_imputations=num_imputations,
            )
            results_eval = dict(eval_output._asdict())
            results_eval["selection_metric_name"] = "weighted_composite_metric"
            selection_metric_name = str(results_eval["selection_metric_name"])
            results_eval["fold_idx"] = fold_idx
            results_eval["kl_disc"] = train_output.last_kl_disc
            results_eval["kl_cont"] = train_output.last_kl_cont
            metric_value = BetaGausMixedDVAETrainer._composite_metric_score(results_eval, metric_weights)
    
            BetaGausMixedDVAETrainer.save_results(
                results_eval,
                epoch=completed_epochs,
                beta=beta,
                capacity_C=capacity_C,
                results_path=results_path,
                lock_path=lock_path
            )
    
            if best_metric_val - metric_value > metric_improvement_tolerance:
                best_metric_val = metric_value
                non_improved_rounds_count = 0
            else:
                non_improved_rounds_count += 1

            if completed_epochs >= min_epochs and non_improved_rounds_count >= max_non_improved_rounds:
                break

        if selection_metric_name is None:
            raise RuntimeError("`selection_metric_name` was not populated during fold evaluation.")

        return BetaGausMixedDVAETrainer.TrainFoldEvalOutput(
            best_metric_val=best_metric_val,
            selection_metric_name=selection_metric_name,
            completed_epochs=completed_epochs,
        )
    
    
    #########################################################
    # 5. Halving grid search utilities
    #########################################################
    
    @staticmethod
    def _resolve_metric_weights(config: Dict) -> Dict[str, float]:
        """
        Resolve and validate the weighted candidate-metric specification from config.

        Reads `halving_metric_weights` when provided; otherwise falls back to
        `DEFAULT_CAND_METRIC_WEIGHTS`. Only supported candidate-metric keys are
        kept, and only strictly positive weights are returned. When a custom
        weight dictionary is supplied, its positive weights must sum to 1.
        """
        metric_weights_config = config.get(
            "halving_metric_weights",
            BetaGausMixedDVAETrainer.DEFAULT_CAND_METRIC_WEIGHTS,
        )
        if not isinstance(metric_weights_config, dict):
            raise ValueError("`halving_metric_weights` must be a dictionary of metric->weight.")
        resolved_w: Dict[str, float] = {}
        for key, weight in metric_weights_config.items():
            metric_key = str(key)
            if metric_key not in BetaGausMixedDVAETrainer.CAND_METRIC_KEYS:
                raise ValueError(
                    f"Unsupported weighted metric '{metric_key}'. "
                    f"Permitted values: {sorted(BetaGausMixedDVAETrainer.CAND_METRIC_KEYS)}"
                )
            weight_f = float(weight)
            if not math.isfinite(weight_f) or weight_f < 0.0:
                raise ValueError(f"Weight for metric '{metric_key}' must be finite and >= 0, got {weight}")
            if weight_f > 0.0:
                resolved_w[metric_key] = weight_f
        if not resolved_w:
            raise ValueError("`halving_metric_weights` must include at least one positive weight.")
        if "halving_metric_weights" in config:
            weight_sum = float(sum(resolved_w.values()))
            # `rel_tol=0.0` means to ignore relative tolerance entirely and use only the absolute tolerance
            if not math.isclose(weight_sum, 1.0, rel_tol=0.0, abs_tol=1e-6):
                raise ValueError(
                    "`halving_metric_weights` positive weights must sum to 1.0, "
                    f"got {weight_sum!r}."
                )
        return resolved_w


    @staticmethod
    def _parse_coverage_target(metric_key: str) -> float | None:
        """
        Parse a nominal coverage target from metric keys like `prop_80q`.
        """
        coverage_target_match = re.search(r"(?<=prop_)([1-9][0-9]*)(?=q)", str(metric_key))
        if coverage_target_match is None:
            return None
        return float(coverage_target_match.group(1)) / 100.0

    @staticmethod
    def _composite_metric_score(results: Dict, metric_weights: Dict[str, float]) -> float:
        """
        Compute the weighted composite training-evaluation metric for candidate ranking.

        Lower is better. Coverage metrics are converted to absolute deviation from
        their nominal targets before weight-averaging into a single scalar score.
        """
        weighted_sum = 0.0
        for metric_key, weight in metric_weights.items():
            if not math.isfinite(float(weight)) or float(weight) <= 0.0:
                raise ValueError(
                    f"Weight for metric '{metric_key}' must be finite and > 0, got {weight!r}"
                )
            if metric_key not in results or results[metric_key] is None:
                raise KeyError(f"Weighted metric '{metric_key}' was not found in evaluation results.")
            value = float(results[metric_key])
            if not math.isfinite(value):
                raise ValueError(f"Weighted metric '{metric_key}' must be finite, got {results[metric_key]}")
            # `prop_*q` values are computed in `BetaGausMixedDVAEUtils.evaluate_coverage_quantile(...)`.
            coverage_target = BetaGausMixedDVAETrainer._parse_coverage_target(metric_key)
            if coverage_target is not None:
                component = abs(value - coverage_target)
            else:
                component = value
            weighted_sum += weight * component
        return weighted_sum
    
    
    @staticmethod
    def _infer_binary_ordinal_feat_indices(
        feat_type_dict: FeaturesTypeDict,
    ) -> Tuple[int, ...]:
        """
        Infer binary and ordinal feature-column indices from `features_type_dict`.
        """
        ord_feats = feat_type_dict.get("ord_feats", {})
        bi_feats = feat_type_dict.get("bi_feats", {})
        ordered_feat_names_raw = feat_type_dict["all_feats"]
        if isinstance(ordered_feat_names_raw, set):
            ordered_feat_names = sorted(str(name) for name in ordered_feat_names_raw)
        else:
            ordered_feat_names = [str(name) for name in ordered_feat_names_raw]
        name_to_index = {name: idx for idx, name in enumerate(ordered_feat_names)}
        binary_ordinal_feat_indices: List[int] = []
        for feat in sorted(ord_feats.keys()):
            n_orders = int(ord_feats[feat])
            for order in range(1, n_orders):
                col_name = f"{feat}-ge_{order}"
                if col_name not in name_to_index:
                    raise KeyError(f"Ordinal feature column was not found in `all_feats`: {col_name!r}")
                binary_ordinal_feat_indices.append(name_to_index[col_name])
        for feat in sorted(bi_feats.keys()):
            if feat not in name_to_index:
                raise KeyError(f"Binary feature column was not found in `all_feats`: {feat!r}")
            binary_ordinal_feat_indices.append(name_to_index[feat])
        return tuple(sorted(binary_ordinal_feat_indices))

    @staticmethod
    def _infer_bce_feat_mask(train_np: np.ndarray, config: Dict) -> torch.Tensor:
        """
        Build BCE feature mask (binary/ordinal columns) from feature metadata,
        with a [0, 1] observed-value heuristic fallback when metadata is absent.
        """
        n_features = int(train_np.shape[1])
        mask = np.zeros(n_features, dtype=bool)

        feat_type_dict_raw = config.get("features_type_dict", None)
        if feat_type_dict_raw is not None:
            feat_type_dict: FeaturesTypeDict = dict(feat_type_dict_raw)
            for idx in BetaGausMixedDVAETrainer._infer_binary_ordinal_feat_indices(
                feat_type_dict,
            ):
                i = int(idx)
                if not (0 <= i < n_features):
                    raise ValueError(f"BCE feature index out of bounds: {i} for `n_features`={n_features}")
                mask[i] = True
            return torch.tensor(mask, dtype=torch.bool)
    
        for j in range(n_features):
            col = train_np[:, j]
            obs = col[np.isfinite(col)]
            if obs.size == 0:
                continue
            if float(np.min(obs)) >= -1e-6 and float(np.max(obs)) <= 1.0 + 1e-6:
                mask[j] = True
    
        return torch.tensor(mask, dtype=torch.bool)
    

    @staticmethod
    def run_candidate_cv(job_kwargs: Dict) -> Dict:
        """
        Worker entry: train and evaluate one (beta, C) hyperparameter pair across all folds.
        """
        config = job_kwargs["config"]
        beta = job_kwargs["beta"]
        capacity_C = job_kwargs["C"]
        preimputed_data_for_ref = job_kwargs["preimputed_data_for_ref"]
        amputated_data_for_val = job_kwargs["amputated_data_for_val"]
        # `amputation_mask` uses: 0=observed, 1=originally missing, 2=pretraining amputation.
        amputation_mask = job_kwargs["amputation_mask"]
        use_adam = config.get("use_adam_optimizer", True)
        batch_size = config["batch_size"]
        lr = config["learning_rate"]
        num_imputations = config["m"]
        latent_dim = config["vae_cont_lat_dim"]
        n_gmm_components = int(config["n_gmm_components"])
        hidden_dim1 = config["hidden_size_1"]
        hidden_dim2 = config["hidden_size_2"]

        device = torch.device(job_kwargs.get("device", "cpu"))
        k_folds = config["k_folds"]
        metric_weights = BetaGausMixedDVAETrainer._resolve_metric_weights(config)
        metric_improvement_tolerance = config.get("convergence_tolerance", 1e-4)
        max_non_improved_rounds = config.get("convergence_patience", 10)
        num_feat_loss_metric = config.get("num_feat_loss_metric", "RMSE")
        min_epochs = config.get("min_epochs_before_convergence", 50)
        start_epoch, max_epoch, epochs_between_evals = (int(v) for v in config["halving_epoch_budgets"])
        min_epochs = start_epoch if min_epochs is None else int(min_epochs)
        fold_scores: Dict[str, Tuple[float]] = {}
        results_path = config["results_path"]
        
        for fold_idx in range(k_folds):
            split_output = BetaGausMixedDVAETrainer.split_training_and_validation(
                config,
                fold_idx,
                amputated_data_for_val,
                preimputed_data_for_ref,
                amputation_mask,
            )
            training_input_np = split_output.training_input
            validation_input = split_output.validation_input
            validation_mask = split_output.validation_mask
            pre_val_amputation_mask = split_output.pre_val_amputation_mask

            train_tensor = torch.tensor(training_input_np, dtype=torch.float32)
            if "features_type_dict" not in config:
                raise KeyError("Missing required config key: `features_type_dict`")
            feat_type_dict: FeaturesTypeDict = dict(config["features_type_dict"])
            model = BetaGausMixedDVAE(
                input_dim=train_tensor.shape[1],
                latent_dim=latent_dim,
                hidden_dim1=hidden_dim1,
                hidden_dim2=hidden_dim2,
                n_gmm_components=n_gmm_components,
            ).to(device)
            optimizer = (
                torch.optim.Adam(model.parameters(), lr=lr)
                if use_adam
                else torch.optim.SGD(model.parameters(), lr=lr)
            )
    
            fold_eval_output = BetaGausMixedDVAETrainer.train_fold_with_eval(
                model=model,
                optimizer=optimizer,
                use_adam=use_adam,
                lr=lr,
                train_tensor=train_tensor,
                beta=beta,
                capacity_C=capacity_C,
                device=device,
                feat_type_dict=feat_type_dict,
                num_feat_loss_metric=num_feat_loss_metric,
                batch_size=batch_size,
                start_epoch=start_epoch,
                epochs_between_evals=epochs_between_evals,
                max_epoch=max_epoch,
                metric_weights=metric_weights,
                validation_input=validation_input,
                validation_mask=validation_mask,
                imputation_mask=pre_val_amputation_mask,
                num_imputations=num_imputations,
                fold_idx=fold_idx,
                results_path=results_path,
                metric_improvement_tolerance=metric_improvement_tolerance,
                max_non_improved_rounds=max_non_improved_rounds,
                min_epochs=min_epochs
            )
            best_metric_val = fold_eval_output.best_metric_val
            selection_metric_name = fold_eval_output.selection_metric_name
            fold_scores.setdefault(selection_metric_name, []).append(best_metric_val)
    
        if len(fold_scores) != 1:
            raise ValueError(
                f"Expected exactly one fold-score metric key, got {tuple(fold_scores)!r}."
            )
        selection_metric_name, fold_score_values = next(iter(fold_scores.items()))
        avg_score = float(np.mean(fold_score_values))
        fold_scores[selection_metric_name] = tuple(fold_score_values)
        
        return {
            "beta": beta,
            "C": capacity_C,
            "score": avg_score,
            "fold_scores": fold_scores,
            "device": str(device)
        }
    
    
    @staticmethod
    def _available_devices() -> List[str]:
        """
        Prefers multi-GPU. If only one or none, fall back to CPU for multiprocessing.
        """
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            return [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        return ["cpu"]
    
    
    @staticmethod
    def _run_candidates_parallel(
        candidates: List[Tuple[float, float]], # a pair of `(beta, C)` hyperparams
        config: Dict,
        preimputed_data_for_ref: np.ndarray,
        amputated_data_for_val: np.ndarray,
        amputation_mask: np.ndarray,
    ) -> List[Dict]:
        """
        Execute Cross Validation for candidate `(beta, C)` pairs in parallel using GPUs when available,
        otherwise CPU multiprocessing.
        """
        devices = BetaGausMixedDVAETrainer._available_devices()
        jobs = []
        for idx, (beta_val, capacity_C) in enumerate(candidates):
            jobs.append({
                "beta": beta_val,
                "C": capacity_C,
                "config": config,
                "preimputed_data_for_ref": preimputed_data_for_ref,
                "amputated_data_for_val": amputated_data_for_val,
                "amputation_mask": amputation_mask,
                "device": devices[idx % len(devices)]
            })
    
        if len(jobs) == 1:
            return [BetaGausMixedDVAETrainer.run_candidate_cv(jobs[0])]
        
        if devices == ["cpu"]:
            # reserve at least one CPU for routine processes if there is more than one available
            workers = max(min(len(jobs), cpu_count()-1), 1) 
        else:
            # otherwise, utilize all GPUs/TPUs
            workers = min(len(jobs), len(devices)) 
    
        with Pool(processes=workers) as pool:
            results = list(pool.imap_unordered(BetaGausMixedDVAETrainer.run_candidate_cv, jobs))
        return results
    
    
    @staticmethod
    def iterative_halving_search(
        config: Dict,
        preimputed_data_for_ref: np.ndarray,
        amputated_data_for_val: np.ndarray,
        amputation_mask: np.ndarray,
    ) -> Dict:
        """
        Compatibility wrapper.
        Primary implementation lives in step2_beta_C_tuning/fine_tuner.py.
        """
        from VAEQL_plus.step2_beta_C_tuning.fine_tuner import iterative_halving_search as _impl
        return _impl(config, preimputed_data_for_ref, amputated_data_for_val, amputation_mask)
    
    
    @staticmethod
    def train_and_save_best_model(
        beta_val: float,
        capacity_C: float,
        config: Dict,
        preimputed_data_for_ref: np.ndarray,
        amputated_data_for_val: np.ndarray,
        amputation_mask: np.ndarray,
    ):
        """
        Compatibility wrapper.
        Primary implementation lives in step2_beta_C_tuning/fine_tuner.py.
        """
        from VAEQL_plus.step2_beta_C_tuning.fine_tuner import train_and_save_best_model as _impl
        return _impl(beta_val, capacity_C, config, preimputed_data_for_ref, amputated_data_for_val, amputation_mask)
