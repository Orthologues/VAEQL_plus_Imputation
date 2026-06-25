"""
Disentangled Beta-VAE utility helpers.

This module contains shared evaluation and synthetic-amputation helpers for
beta-DVAE tuning, including quantile coverage, normal-interval coverage, and
validation-set amputation.

Author: Jiawei Zhao (jiz@imada.sdu.dk)
Date: 2026-04-01
"""

import numpy as np
import torch
from typing import Dict, List, NamedTuple, Tuple, Union

ArrayLike = Union[np.ndarray, torch.Tensor]

class BetaGausMixedDVAEUtils:
    """
    Namespace for beta-DVAE tensor conversion, mask validation, and imputation-quality metrics.

    The methods in this class accept either NumPy arrays or torch tensors and normalize
    them to torch tensors before evaluating validation-amputation entries from a 0/1/2/3/4 mask.
    """

    NORMAL_CENTRAL_INTERVAL_Z = {
        "prop_80": 1.2815515655446004,
        "prop_90": 1.6448536269514722,
        "prop_95": 1.959963984540054,
        "prop_99": 2.5758293035489004,
    }

    class ValidationAmputationOutput(NamedTuple):
        """Validation-amputation tensors for beta-C tuning."""
        validation_input: torch.Tensor
        validation_mask: torch.Tensor
        val_fold_baseline_amputation_mask: torch.Tensor

    @staticmethod
    def as_torch_tensor(
        array_like: ArrayLike,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Convert a NumPy array or torch tensor to a torch tensor without CPU NumPy round-tripping."""
        if isinstance(array_like, torch.Tensor):
            return array_like.to(device=device, dtype=dtype) if device is not None or dtype is not None else array_like
        return torch.as_tensor(array_like, device=device, dtype=dtype)

    @staticmethod
    def stack_as_torch_tensor(
        array_like_list,
        MI_iterations: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Stack a sequence of tensors/arrays into a torch tensor."""
        if int(MI_iterations) < 2 or int(MI_iterations) > 100:
            raise ValueError(f"`MI_iterations` must be >= 2 and <= 100, got {MI_iterations}")
        if isinstance(array_like_list, torch.Tensor):
            stacked_t = array_like_list.to(device=device, dtype=dtype) if device is not None or dtype is not None else array_like_list
        elif isinstance(array_like_list, (list, tuple)) and array_like_list and isinstance(array_like_list[0], torch.Tensor):
            stacked_t = torch.stack([
                item.to(device=device, dtype=dtype) if device is not None or dtype is not None else item
                for item in array_like_list
            ])
        else:
            stacked_t = torch.as_tensor(np.asarray(array_like_list), device=device, dtype=dtype)
        if stacked_t.dim() < 3:
            raise ValueError(f"`multi_imputes` tensor must be at least 3D, got shape {tuple(stacked_t.shape)}")
        if int(stacked_t.shape[0]) != int(MI_iterations):
            raise ValueError(
                f"`MI_iterations` must match stacked tensor first dimension, got "
                f"MI_iterations={MI_iterations} and shape[0]={int(stacked_t.shape[0])}"
            )
        return stacked_t

    @staticmethod
    def mask_type2_indices(X_mask: ArrayLike, reference_shape: tuple[int, ...]) -> torch.Tensor:
        """Return a boolean mask of synthetically amputated cells from a 0/1/2 mask."""
        x_mask_t = BetaGausMixedDVAEUtils.as_torch_tensor(X_mask, dtype=torch.int8)
        if tuple(x_mask_t.shape) != reference_shape:
            raise ValueError(f"`X_mask` must have shape {reference_shape}, got {tuple(x_mask_t.shape)}")
        valid_values = (x_mask_t == 0) | (x_mask_t == 1) | (x_mask_t == 2)
        if not bool(valid_values.all().item()):
            invalid_values = torch.unique(x_mask_t[~valid_values]).detach().tolist()
            raise ValueError(f"`X_mask` can only contain values from {{0, 1, 2}}, got {sorted(int(v) for v in invalid_values)}")
        if not bool((x_mask_t == 0).any().item()):
            raise ValueError("`X_mask` must contain value 0 for observed cells.")
        if not bool((x_mask_t == 2).any().item()):
            raise ValueError("`X_mask` must contain value 2 for synthetically amputated cells.")
        return x_mask_t == 2

    @staticmethod
    def mask_validation_type3_indices(X_mask: ArrayLike, reference_shape: tuple[int, ...]) -> torch.Tensor:
        """Return a boolean metric mask: pre-validation type-0 cells re-amputated as type 3."""
        x_mask_t = BetaGausMixedDVAEUtils.as_torch_tensor(X_mask, dtype=torch.int8)
        if tuple(x_mask_t.shape) != reference_shape:
            raise ValueError(f"`X_mask` must have shape {reference_shape}, got {tuple(x_mask_t.shape)}")
        valid_values = (
            (x_mask_t == 0)
            | (x_mask_t == 1)
            | (x_mask_t == 2)
            | (x_mask_t == 3)
            | (x_mask_t == 4)
        )
        if not bool(valid_values.all().item()):
            invalid_values = torch.unique(x_mask_t[~valid_values]).detach().tolist()
            raise ValueError(f"`X_mask` can only contain values from {{0, 1, 2, 3, 4}}, got {sorted(int(v) for v in invalid_values)}")
        if not bool((x_mask_t == 0).any().item()):
            raise ValueError("`X_mask` must contain value 0 for reconstruction-loss weighting.")
        if not bool((x_mask_t == 2).any().item()):
            raise ValueError("`X_mask` must contain value 2 for imputation updates.")
        if not bool((x_mask_t == 3).any().item()):
            raise ValueError("`X_mask` must contain value 3 for beta-C validation metrics.")
        return x_mask_t == 3

    @staticmethod
    def evaluate_coverage_quantile(
        multi_imputes: ArrayLike,
        ref_data_arr: ArrayLike,
        X_mask: ArrayLike,
    ) -> Dict[str, float]:
        """Evaluate empirical quantile coverage over synthetically missing cells."""
        multi_imputes_t = BetaGausMixedDVAEUtils.stack_as_torch_tensor(
            multi_imputes,
            MI_iterations=int(len(multi_imputes)),
            dtype=torch.float32,
        )
        ref_data_t = BetaGausMixedDVAEUtils.as_torch_tensor(ref_data_arr, device=multi_imputes_t.device, dtype=torch.float32)
        mask_type3 = BetaGausMixedDVAEUtils.mask_validation_type3_indices(X_mask, tuple(ref_data_t.shape)).to(device=multi_imputes_t.device)
        ref_type3_vals = ref_data_t[mask_type3]
        quantiles = torch.quantile(
            multi_imputes_t,
            torch.tensor([0.10, 0.90, 0.05, 0.95, 0.025, 0.975, 0.005, 0.995], device=multi_imputes_t.device),
            dim=0,
        )
        if tuple(quantiles[0].shape) != tuple(ref_data_t.shape):
            raise ValueError(
                f"`multi_imputes` must reduce to shape {tuple(ref_data_t.shape)} after axis-0 quantiles, "
                f"got {tuple(quantiles[0].shape)}"
            )
        q10_type3_vals = quantiles[0][mask_type3]
        q90_type3_vals = quantiles[1][mask_type3]
        q05_type3_vals = quantiles[2][mask_type3]
        q95_type3_vals = quantiles[3][mask_type3]
        q025_type3_vals = quantiles[4][mask_type3]
        q975_type3_vals = quantiles[5][mask_type3]
        q005_type3_vals = quantiles[6][mask_type3]
        q995_type3_vals = quantiles[7][mask_type3]

        def prop_in_interval(lower_q_imputed_vals, upper_q_imputed_vals, ground_truth_vals) -> float:
            return float(((lower_q_imputed_vals < ground_truth_vals) & (ground_truth_vals < upper_q_imputed_vals)).float().mean().item())

        return {
            "prop_80q": prop_in_interval(q10_type3_vals, q90_type3_vals, ref_type3_vals),
            "prop_90q": prop_in_interval(q05_type3_vals, q95_type3_vals, ref_type3_vals),
            "prop_95q": prop_in_interval(q025_type3_vals, q975_type3_vals, ref_type3_vals),
            "prop_99q": prop_in_interval(q005_type3_vals, q995_type3_vals, ref_type3_vals),
        }

    @staticmethod
    def evaluate_coverage(
        multi_imputes: ArrayLike,
        ref_data_arr: ArrayLike,
        X_mask: ArrayLike,
    ) -> Dict[str, float]:
        """Evaluate normal-interval coverage and MAE over synthetically missing cells."""
        multi_imputes_t = BetaGausMixedDVAEUtils.stack_as_torch_tensor(
            multi_imputes,
            MI_iterations=int(len(multi_imputes)),
            dtype=torch.float32,
        )
        ref_data_t = BetaGausMixedDVAEUtils.as_torch_tensor(ref_data_arr, device=multi_imputes_t.device, dtype=torch.float32)
        mask_type3 = BetaGausMixedDVAEUtils.mask_validation_type3_indices(X_mask, tuple(ref_data_t.shape)).to(device=multi_imputes_t.device)
        ref_type3_vals = ref_data_t[mask_type3]
        imputed_mean_t = torch.mean(multi_imputes_t, dim=0)
        std_unbiased = multi_imputes_t.shape[0] > 1
        # The divisor used in calculations is N - ddof
        imputed_std_t = torch.std(multi_imputes_t, dim=0, unbiased=std_unbiased)
        if tuple(imputed_mean_t.shape) != tuple(ref_data_t.shape):
            raise ValueError(
                f"`multi_imputes` must reduce to shape {tuple(ref_data_t.shape)} after axis-0 mean, "
                f"got {tuple(imputed_mean_t.shape)}"
            )
        imputed_mean_type3_vals = imputed_mean_t[mask_type3]
        imputed_std_type3_vals = imputed_std_t[mask_type3]
        error_vals = ref_type3_vals - imputed_mean_type3_vals
        abs_error_vals = torch.abs(error_vals)
        abs_standardized_error_vals = abs_error_vals / (imputed_std_type3_vals + 1e-12)

        # Two-sided central normal-interval z-thresholds:
        # z_{0.90}=1.282, z_{0.95}=1.645, z_{0.975}=1.960, z_{0.995}=2.576.
        # Reference: standard normal quantile function, e.g. scipy.stats.norm.ppf.
        results = {
            key: float((abs_standardized_error_vals < z_threshold).float().mean().item())
            for key, z_threshold in BetaGausMixedDVAEUtils.NORMAL_CENTRAL_INTERVAL_Z.items()
        }
        results["multi_mae"] = float(torch.mean(abs_error_vals).item())
        results["multi_rmse"] = float(torch.sqrt(torch.mean(error_vals ** 2)).item())
        return results

    @staticmethod
    def build_amputation_units(
        feat_type_dict: Dict,
        n_col: int,
    ) -> tuple[Tuple[Tuple[int, ...], ...], Tuple[Tuple[int, ...], ...]]:
        """
        Build grouped amputation units for Ordinal and Non-binary Categorical Features from beta-DVAE-oriented feature metadata.

        Ordinal columns and categorical one-hot columns are grouped per source feature; all remaining columns are single-column units.
        """
        ordered_feat_names_raw = feat_type_dict["all_feats"]
        if not isinstance(ordered_feat_names_raw, set):
            raise TypeError(
                "`feat_type_dict['all_feats']` must be a set of preprocessed feature names."
            )
        ordered_feat_names = sorted(str(name) for name in ordered_feat_names_raw)
        if len(ordered_feat_names) != n_col:
            raise ValueError(
                f"Feature dimension mismatch: data has {n_col} columns, "
                f"but `len(feat_type_dict['all_feats'])`={len(ordered_feat_names)}."
            )

        name_to_index = {name: idx for idx, name in enumerate(ordered_feat_names)}
        grouped_indices: List[Tuple[int, ...]] = []

        for feat in sorted(feat_type_dict.get("ord_feats", {}).keys()):
            n_orders = int(feat_type_dict["ord_feats"][feat])
            ordinal_cols = [f"{feat}-ge_{order}" for order in range(1, n_orders)]
            ordinal_indices = tuple(name_to_index[col] for col in ordinal_cols if col in name_to_index)
            if ordinal_indices:
                grouped_indices.append(ordinal_indices)

        for feat in sorted(feat_type_dict.get("cat_feats", {}).keys()):
            one_hot_cols = [f"{feat}-is_{value}" for value in sorted(feat_type_dict["cat_feats"][feat])]
            categorical_indices = tuple(name_to_index[col] for col in one_hot_cols if col in name_to_index)
            if categorical_indices:
                grouped_indices.append(categorical_indices)

        grouped_col_indices = {idx for group in grouped_indices for idx in group}
        single_col_indices = [
            (idx,)
            for idx in range(n_col)
            if idx not in grouped_col_indices
        ]
        return tuple(grouped_indices), tuple(single_col_indices)

    @staticmethod
    def get_random_col_selection(
        grouped_amputation_units: Tuple[Tuple[int, ...], ...],
        single_col_amputation_units: Tuple[Tuple[int, ...], ...],
        prop_miss_col: float,
    ) -> np.ndarray:
        """Sample columns by amputation unit for one selected row."""
        selected_grouped_cols = [
            idx
            for group_unit in grouped_amputation_units
            if np.random.random() < prop_miss_col
            for idx in group_unit
        ]
        selected_single_cols = [
            unit[0]
            for unit in single_col_amputation_units
            if np.random.random() < prop_miss_col
        ]
        selected_cols = selected_grouped_cols + selected_single_cols
        if not selected_cols:
            return np.array([], dtype=np.int8)
        return np.array(selected_cols, dtype=np.int8)

    @staticmethod
    def generate_validation_amputation(
        preimputed_data_arr: ArrayLike,
        X_mask: ArrayLike,
        feat_type_dict: Dict,
        prop_miss_rows: float = 1.0,
        prop_miss_col: float = 0.1,
    ) -> "BetaGausMixedDVAEUtils.ValidationAmputationOutput":
        """
        Generate beta-C validation-amputation tensors.

        `preimputed_data_arr` is copied into `validation_input`; missingness is
        represented only through masks. `X_mask` is the fold-level baseline
        0/1/2 amputation mask used by imputation routines.

        The returned `validation_mask` extends the baseline mask with beta-C
        validation semantics:
        - 3 = pre-validation type-0 cell selected for beta-C metrics
        - 4 = pre-validation type-1/type-2 cell selected again, excluded from metrics

        Returns:
        - `validation_input` (float32)
        - `validation_mask` (int8, values 0/1/2/3/4)
        - `val_fold_baseline_amputation_mask` (int8, values 0/1/2)
        """
        if isinstance(preimputed_data_arr, torch.Tensor):
            preimputed_data_arr = preimputed_data_arr.detach().cpu().numpy()
        else:
            preimputed_data_arr = np.asarray(preimputed_data_arr)
        if isinstance(X_mask, torch.Tensor):
            X_mask = X_mask.detach().cpu().numpy()
        else:
            X_mask = np.asarray(X_mask)
        if preimputed_data_arr.shape != X_mask.shape:
            raise ValueError(
                f"`preimputed_data_arr` and `X_mask` must have the same shape, "
                f"got {preimputed_data_arr.shape} and {X_mask.shape}"
            )
        validation_input = np.copy(preimputed_data_arr)
        imputation_mask = X_mask.astype(np.int8, copy=True)
        validation_mask = np.copy(imputation_mask)
        n_rows_to_null = int(len(preimputed_data_arr) * prop_miss_rows)
        if n_rows_to_null == 0:
            raise ValueError(
                "`prop_miss_rows` selected zero rows for validation amputation; "
                f"got prop_miss_rows={prop_miss_rows} for {len(preimputed_data_arr)} rows."
            )

        grouped_amputation_units, single_col_amputation_units = BetaGausMixedDVAEUtils.build_amputation_units(
            feat_type_dict,
            preimputed_data_arr.shape[1],
        )
        random_rows = np.random.choice(range(len(preimputed_data_arr)), n_rows_to_null, replace=False)
        null_rows_list: List[int] = []
        null_cols_list: List[int] = []
        for row in random_rows:
            cols = BetaGausMixedDVAEUtils.get_random_col_selection(
                grouped_amputation_units,
                single_col_amputation_units,
                prop_miss_col,
            )
            if cols.size == 0:
                continue
            null_rows_list.extend([row] * len(cols))
            null_cols_list.extend(cols.tolist())

        if not null_cols_list:
            raise ValueError(
                "Validation amputation selected no columns; "
                f"check `prop_miss_col={prop_miss_col}` and feature amputation units."
            )

        null_rows = np.array(null_rows_list, dtype=np.int8)
        null_cols = np.array(null_cols_list, dtype=np.int8)
        pre_validation_vals = validation_mask[null_rows, null_cols]
        # Type-4 cells are explicitly excluded from validation metrics because
        # they were already non-observed before beta-C validation.
        validation_mask[null_rows, null_cols] = np.where(pre_validation_vals == 0, 3, 4)
        return BetaGausMixedDVAEUtils.ValidationAmputationOutput(
            validation_input=torch.as_tensor(validation_input, dtype=torch.float32),
            validation_mask=torch.as_tensor(validation_mask, dtype=torch.int8),
            val_fold_baseline_amputation_mask=torch.as_tensor(imputation_mask, dtype=torch.int8),
        )
