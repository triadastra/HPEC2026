"""Shared tabular feature extraction for tree-based forecasting models."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch


SUPPORTED_TABULAR_VARIANTS = {"embeddings", "onehot"}
TARGET_COLS = {"Value", "Weight"}
KEY_COLS = ["State", "Commodity", "Import/Export"]

# The window-end (month t-1) actuals are the single strongest predictor for a
# near-random-walk monthly series. The neural models see them (raw Value/Weight
# channels at the last timestep); the trees MUST too, or the comparison is
# unfair. We keep them as features under these names (they are the LAST OBSERVED
# value at t-1, NOT the target at t — so no leakage).
LAST_OBS_RENAME = {"Value": "value_last", "Weight": "weight_last"}
ID_FEATURE_NAMES = ["state_id", "comm_id", "flow_id"]


def to_feature_frame(
    X: np.ndarray,
    feature_names: list[str],
    variant: str,
    num_states: int,
    num_commodities: int,
    num_flows: int,
) -> pd.DataFrame:
    """Wrap the feature matrix in a DataFrame, marking the categorical ID columns
    as pandas `category` dtype (embeddings variant only) so XGBoost
    (enable_categorical) and LightGBM (auto-detect) split on them NATIVELY
    instead of treating the integer IDs as ordinal/continuous. Categories are
    pinned to the full id range so train/val/test share identical codes."""
    df = pd.DataFrame(X, columns=feature_names)
    if variant == "embeddings":
        ranges = {"state_id": num_states, "comm_id": num_commodities, "flow_id": num_flows}
        for col, n in ranges.items():
            if col in df.columns:
                codes = df[col].round().astype("int64")
                df[col] = pd.Categorical(codes, categories=range(n))
    return df


def validate_tabular_variant(model_name: str, variant: str) -> None:
    if variant not in SUPPORTED_TABULAR_VARIANTS:
        raise ValueError(
            f"{model_name} only supports 'embeddings' (integer IDs) and "
            "'onehot' variants in the tabular tree path."
        )


def category_feature_names(
    variant: str,
    num_states: int,
    num_commodities: int,
    num_flows: int,
) -> list[str]:
    if variant == "onehot":
        return (
            [f"state_{i}" for i in range(num_states)]
            + [f"commodity_{i}" for i in range(num_commodities)]
            + [f"flow_{i}" for i in range(num_flows)]
        )
    return ["state_id", "comm_id", "flow_id"]


def fallback_feature_names(
    variant: str,
    num_numeric_features: int,
    num_states: int,
    num_commodities: int,
    num_flows: int,
) -> list[str]:
    # num_numeric_features = value_last + weight_last + lag_count value-lags +
    # lag_count weight-lags + sin + cos  =>  lag_count = (n - 4) / 2.
    legacy_width = num_numeric_features - 4
    if legacy_width >= 0 and legacy_width % 2 == 0:
        lag_count = legacy_width // 2
        numeric_names = (
            ["value_last", "weight_last"]
            + [f"value_lag_{k}" for k in range(1, lag_count + 1)]
            + [f"weight_lag_{k}" for k in range(1, lag_count + 1)]
            + ["sin_month", "cos_month"]
        )
    else:
        # Multichannel datasets such as Census (9 base + 24 lags + calendar)
        # do not satisfy the predecessor dataset's two-channel formula. Keep
        # the frame width exact; categorical ID names below remain explicit so
        # native categorical splitting still works.
        numeric_names = [f"numeric_{index}" for index in range(num_numeric_features)]
    return numeric_names + category_feature_names(
        variant, num_states, num_commodities, num_flows
    )


def category_features(
    variant: str,
    state_ids: torch.Tensor,
    comm_ids: torch.Tensor,
    flow_ids: torch.Tensor,
    num_states: int,
    num_commodities: int,
    num_flows: int,
) -> np.ndarray:
    if variant == "onehot":
        parts = [
            torch.nn.functional.one_hot(state_ids, num_states),
            torch.nn.functional.one_hot(comm_ids, num_commodities),
            torch.nn.functional.one_hot(flow_ids, num_flows),
        ]
        return torch.cat(parts, dim=-1).cpu().numpy().astype(np.float32)

    return (
        torch.stack([state_ids, comm_ids, flow_ids], dim=-1)
        .cpu()
        .numpy()
        .astype(np.float32)
    )


def has_trade_dataset_contract(dataset: Any) -> bool:
    return all(
        hasattr(dataset, attr)
        for attr in (
            "df",
            "feat_cols",
            "state2id",
            "comm2id",
            "flow2id",
            "target_time_start",
            "target_time_end",
        )
    )


def extract_tabular_xy(
    loader: Any,
    *,
    variant: str,
    num_numeric_features: int,
    num_states: int,
    num_commodities: int,
    num_flows: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    dataset = getattr(loader, "dataset", None)
    if has_trade_dataset_contract(dataset):
        return extract_from_trade_dataset(
            dataset,
            variant=variant,
            num_states=num_states,
            num_commodities=num_commodities,
            num_flows=num_flows,
        )

    return extract_from_batches(
        loader,
        variant=variant,
        num_numeric_features=num_numeric_features,
        num_states=num_states,
        num_commodities=num_commodities,
        num_flows=num_flows,
    )


def extract_from_trade_dataset(
    dataset: Any,
    *,
    variant: str,
    num_states: int,
    num_commodities: int,
    num_flows: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    # Keep ALL per-step features incl. Value/Weight: on the feature row (t-1)
    # these are the LAST-OBSERVED actuals, not the target (which is row t). The
    # neural models see them; dropping them here was the unfair information gap.
    feat_cols_used = list(dataset.feat_cols)
    numeric_names = [LAST_OBS_RENAME.get(c, c) for c in feat_cols_used]
    feature_names = numeric_names + category_feature_names(
        variant, num_states, num_commodities, num_flows
    )

    X_list, y_list = [], []
    for (state, comm, flow), group in dataset.df.groupby(KEY_COLS, sort=False):
        if (
            state not in dataset.state2id
            or comm not in dataset.comm2id
            or flow not in dataset.flow2id
        ):
            continue

        group = group.sort_values("Time").reset_index(drop=True)
        if len(group) < dataset.input_len + 1:
            continue

        feat_prev = group.iloc[dataset.input_len - 1:-1].reset_index(drop=True)
        targ_next = group.iloc[dataset.input_len:].reset_index(drop=True)

        mask = np.ones(len(targ_next), dtype=bool)
        if dataset.target_time_start is not None:
            mask &= targ_next["Time"].to_numpy() >= np.datetime64(
                dataset.target_time_start
            )
        if dataset.target_time_end is not None:
            mask &= targ_next["Time"].to_numpy() <= np.datetime64(
                dataset.target_time_end
            )
        if not mask.any():
            continue

        x_numeric = feat_prev.loc[mask, feat_cols_used].to_numpy(np.float32)
        n = x_numeric.shape[0]
        state_ids = torch.full((n,), dataset.state2id[state], dtype=torch.long)
        comm_ids = torch.full((n,), dataset.comm2id[comm], dtype=torch.long)
        flow_ids = torch.full((n,), dataset.flow2id[flow], dtype=torch.long)
        x_cat = category_features(
            variant,
            state_ids,
            comm_ids,
            flow_ids,
            num_states,
            num_commodities,
            num_flows,
        )

        y = targ_next.loc[mask, ["Value", "Weight"]].to_numpy(np.float32)
        X_list.append(np.concatenate([x_numeric, x_cat], axis=1))
        y_list.append(y)

    if not X_list:
        raise ValueError("No tabular supervised rows could be built")

    return np.vstack(X_list), np.vstack(y_list), feature_names


def extract_from_batches(
    loader: Any,
    *,
    variant: str,
    num_numeric_features: int,
    num_states: int,
    num_commodities: int,
    num_flows: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    X_list, y_list = [], []
    with torch.no_grad():
        for batch in loader:
            x_numeric = batch["x_numeric"]
            state_ids = batch["state_ids"]
            comm_ids = batch["comm_ids"]
            flow_ids = batch["flow_ids"]

            # Include ALL channels at the last timestep (t-1), incl. Value/Weight
            # (index 0,1) — the last-observed actuals the neural models also see.
            x_last = x_numeric[:, -1, :].cpu().numpy().astype(np.float32)
            x_cat = category_features(
                variant,
                state_ids,
                comm_ids,
                flow_ids,
                num_states,
                num_commodities,
                num_flows,
            )
            X_list.append(np.concatenate([x_last, x_cat], axis=1))

            y = np.column_stack(
                [
                    batch["target_value"].cpu().numpy(),
                    batch["target_weight"].cpu().numpy(),
                ]
            )
            y_list.append(y)

    return (
        np.vstack(X_list),
        np.vstack(y_list),
        fallback_feature_names(
            variant,
            num_numeric_features,
            num_states,
            num_commodities,
            num_flows,
        ),
    )
