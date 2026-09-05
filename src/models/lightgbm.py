"""LightGBM tabular forecasting baseline."""

import json

import joblib
import lightgbm as lgb
import numpy as np
import torch.nn as nn

from .base import BaseModel, ModelFactory
from .tabular_common import (
    extract_tabular_xy,
    fallback_feature_names,
    to_feature_frame,
    validate_tabular_variant,
)


class LightGBMModel(BaseModel):
    """LightGBM peer baseline using the same rows as XGBoost."""

    def __init__(
        self,
        variant: str,
        num_numeric_features: int,
        num_states: int,
        num_commodities: int,
        num_flows: int = 2,
        **kwargs,
    ):
        validate_tabular_variant("LightGBM", variant)
        super().__init__(
            variant=variant,
            num_numeric_features=num_numeric_features,
            num_states=num_states,
            num_commodities=num_commodities,
            num_flows=num_flows,
            **kwargs,
        )
        self.encoder = nn.Identity()
        self.requires_training = False
        self.model_value = None
        self.model_weight = None
        self.feature_names = fallback_feature_names(
            variant,
            num_numeric_features,
            num_states,
            num_commodities,
            num_flows,
        )

    def _build_model(
        self,
        n_estimators: int = 2000,
        learning_rate: float = 0.05,
        num_leaves: int = 255,
        max_depth: int = -1,
        subsample: float = 0.9,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        min_child_samples: int = 20,
        n_jobs: int = -1,
        random_state: int = 42,
        **kwargs,
    ) -> None:
        self.lgbm_params = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "max_depth": max_depth,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "min_child_samples": min_child_samples,
            "n_jobs": n_jobs,
            # Seeded from the run's --seed (train.py threads it into the model
            # config), not pinned to 42. subsample/colsample make these models
            # genuinely stochastic, so a hardcoded seed meant every "multi-seed"
            # tree run repeated the same draw. (F13)
            "random_state": random_state,
            "objective": "regression",
            "metric": "rmse",
            "verbosity": -1,
        }

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "LightGBM model does not support forward(). Use fit() and predict()."
        )

    def _extract_numpy(self, loader) -> tuple[np.ndarray, np.ndarray]:
        X, y, feature_names = extract_tabular_xy(
            loader,
            variant=self.variant,
            num_numeric_features=self.num_numeric_features,
            num_states=self.num_states,
            num_commodities=self.num_commodities,
            num_flows=self.num_flows,
        )
        self.feature_names = feature_names
        return X, y

    def _to_frame(self, X: np.ndarray):
        return to_feature_frame(
            X, self.feature_names, self.variant,
            self.num_states, self.num_commodities, self.num_flows,
        )

    def fit(self, train_loader, val_loader=None, verbose: bool = True) -> None:
        X_train, y_train = self._extract_numpy(train_loader)
        X_val, y_val = (None, None)
        if val_loader is not None:
            X_val, y_val = self._extract_numpy(val_loader)

        if verbose:
            print(
                f"LightGBM tabular features: {X_train.shape[1]} "
                f"({self.variant}); rows={X_train.shape[0]}"
            )

        # DataFrames with category dtype => LightGBM splits natively on the ID
        # columns (embeddings variant) instead of as ordinal floats.
        Xtr = self._to_frame(X_train)
        Xva = self._to_frame(X_val) if X_val is not None else None

        self.model_value = lgb.LGBMRegressor(**self.lgbm_params)
        self.model_weight = lgb.LGBMRegressor(**self.lgbm_params)
        eval_set_value = [(Xva, y_val[:, 0])] if Xva is not None else None
        eval_set_weight = [(Xva, y_val[:, 1])] if Xva is not None else None
        # Real early stopping (was a no-op before: LightGBM 4.x needs the
        # callback, not the removed early_stopping_rounds fit-arg). best
        # iteration is used by predict() automatically.
        callbacks = (
            [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
            if Xva is not None else None
        )

        if verbose:
            print("Training value model...")
        self.model_value.fit(Xtr, y_train[:, 0], eval_set=eval_set_value, callbacks=callbacks)

        if verbose:
            print("Training weight model...")
        self.model_weight.fit(Xtr, y_train[:, 1], eval_set=eval_set_weight, callbacks=callbacks)

    def evaluate_loader(self, loader) -> dict[str, float]:
        preds, targets = self.predict_loader(loader)
        err = preds - targets
        return {
            "mse": float(np.mean(err ** 2)),
            "mae": float(np.mean(np.abs(err))),
            "value_mae": float(np.mean(np.abs(err[:, 0]))),
            "weight_mae": float(np.mean(np.abs(err[:, 1]))),
        }

    def predict_loader(self, loader) -> tuple[np.ndarray, np.ndarray]:
        X, y = self._extract_numpy(loader)
        return self.predict(X), y

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model_value is None or self.model_weight is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        Xf = self._to_frame(X)
        return np.column_stack(
            [self.model_value.predict(Xf), self.model_weight.predict(Xf)]
        )

    def save(self, path: str) -> None:
        joblib.dump(self.model_value, path + "_value.joblib")
        joblib.dump(self.model_weight, path + "_weight.joblib")
        metadata = {
            "variant": self.variant,
            "num_numeric_features": self.num_numeric_features,
            "num_states": self.num_states,
            "num_commodities": self.num_commodities,
            "num_flows": self.num_flows,
            "lgbm_params": self.lgbm_params,
            "feature_names": self.feature_names,
        }
        with open(path + "_metadata.json", "w") as f:
            json.dump(metadata, f)

    def load(self, path: str) -> None:
        self.model_value = joblib.load(path + "_value.joblib")
        self.model_weight = joblib.load(path + "_weight.joblib")
        with open(path + "_metadata.json", "r") as f:
            metadata = json.load(f)
        self.variant = metadata["variant"]
        self.num_numeric_features = metadata["num_numeric_features"]
        self.num_states = metadata["num_states"]
        self.num_commodities = metadata["num_commodities"]
        self.num_flows = metadata["num_flows"]
        self.lgbm_params = metadata["lgbm_params"]
        self.feature_names = metadata.get(
            "feature_names",
            fallback_feature_names(
                self.variant,
                self.num_numeric_features,
                self.num_states,
                self.num_commodities,
                self.num_flows,
            ),
        )


ModelFactory.register("lightgbm", LightGBMModel)
