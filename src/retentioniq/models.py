from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -35, 35)
    return 1.0 / (1.0 + np.exp(-values))


class StandardScaler:
    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "StandardScaler":
        x = np.asarray(x, dtype=float)
        self.mean_ = x.mean(axis=0)
        self.scale_ = x.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("Scaler has not been fit")
        return (np.asarray(x, dtype=float) - self.mean_) / self.scale_

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)


@dataclass
class LogisticRegressionGD:
    learning_rate: float = 0.08
    epochs: int = 900
    l2: float = 0.02
    tolerance: float = 1e-7
    random_seed: int = 42
    coef_: np.ndarray | None = None
    intercept_: float = 0.0
    loss_history_: list[float] = field(default_factory=list)

    def fit(self, x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> "LogisticRegressionGD":
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        rng = np.random.default_rng(self.random_seed)
        self.coef_ = rng.normal(0, 0.01, x.shape[1])
        self.intercept_ = 0.0
        if sample_weight is None:
            sample_weight = np.ones_like(y, dtype=float)
        else:
            sample_weight = np.asarray(sample_weight, dtype=float)
        sample_weight = sample_weight / np.mean(sample_weight)
        weight_sum = float(sample_weight.sum())

        previous_loss = np.inf
        for epoch in range(self.epochs):
            logits = x @ self.coef_ + self.intercept_
            probs = sigmoid(logits)
            errors = (probs - y) * sample_weight
            grad_w = (x.T @ errors) / weight_sum + self.l2 * self.coef_
            grad_b = errors.sum() / weight_sum
            self.coef_ -= self.learning_rate * grad_w
            self.intercept_ -= self.learning_rate * grad_b

            if epoch % 25 == 0 or epoch == self.epochs - 1:
                probs_clipped = np.clip(probs, 1e-12, 1 - 1e-12)
                loss = -np.average(
                    y * np.log(probs_clipped) + (1 - y) * np.log(1 - probs_clipped),
                    weights=sample_weight,
                )
                loss += 0.5 * self.l2 * float(np.dot(self.coef_, self.coef_))
                self.loss_history_.append(float(loss))
                if abs(previous_loss - loss) < self.tolerance:
                    break
                previous_loss = loss
        return self

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise ValueError("Model has not been fit")
        return np.asarray(x, dtype=float) @ self.coef_ + self.intercept_

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return sigmoid(self.decision_function(x))


@dataclass
class PlattCalibrator:
    learning_rate: float = 0.05
    epochs: int = 600
    slope_: float = 1.0
    intercept_: float = 0.0

    def fit(self, probabilities: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        p = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
        y = np.asarray(y, dtype=float)
        logits = np.log(p / (1 - p))
        slope = 1.0
        intercept = 0.0
        for _ in range(self.epochs):
            calibrated = sigmoid(slope * logits + intercept)
            error = calibrated - y
            slope -= self.learning_rate * float(np.mean(error * logits))
            intercept -= self.learning_rate * float(np.mean(error))
        self.slope_ = float(slope)
        self.intercept_ = float(intercept)
        return self

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
        logits = np.log(p / (1 - p))
        return sigmoid(self.slope_ * logits + self.intercept_)


def balanced_sample_weights(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=int)
    positives = max(1, int(y.sum()))
    negatives = max(1, int((y == 0).sum()))
    weights = np.where(y == 1, len(y) / (2 * positives), len(y) / (2 * negatives))
    return weights.astype(float)


@dataclass
class ProbabilityPipeline:
    feature_names: list[str]
    scaler: StandardScaler = field(default_factory=StandardScaler)
    model: LogisticRegressionGD = field(default_factory=LogisticRegressionGD)
    calibrator: PlattCalibrator | None = None

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_calibration: np.ndarray | None = None,
        y_calibration: np.ndarray | None = None,
    ) -> "ProbabilityPipeline":
        x_scaled = self.scaler.fit_transform(x_train)
        self.model.fit(x_scaled, y_train, sample_weight=balanced_sample_weights(y_train))
        if x_calibration is not None and y_calibration is not None:
            raw_probs = self.model.predict_proba(self.scaler.transform(x_calibration))
            self.calibrator = PlattCalibrator().fit(raw_probs, y_calibration)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        raw_probs = self.model.predict_proba(self.scaler.transform(x))
        if self.calibrator is not None:
            return self.calibrator.predict(raw_probs)
        return raw_probs

    def save(self, path: Path) -> None:
        if self.scaler.mean_ is None or self.scaler.scale_ is None or self.model.coef_ is None:
            raise ValueError("Pipeline must be fit before saving")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            feature_names=np.array(self.feature_names, dtype=object),
            scaler_mean=self.scaler.mean_,
            scaler_scale=self.scaler.scale_,
            coef=self.model.coef_,
            intercept=np.array([self.model.intercept_]),
            calibrator_slope=np.array([self.calibrator.slope_ if self.calibrator else 1.0]),
            calibrator_intercept=np.array([self.calibrator.intercept_ if self.calibrator else 0.0]),
            has_calibrator=np.array([self.calibrator is not None]),
        )

    @classmethod
    def load(cls, path: Path) -> "ProbabilityPipeline":
        data = np.load(path, allow_pickle=True)
        pipeline = cls(feature_names=data["feature_names"].astype(str).tolist())
        pipeline.scaler.mean_ = data["scaler_mean"]
        pipeline.scaler.scale_ = data["scaler_scale"]
        pipeline.model.coef_ = data["coef"]
        pipeline.model.intercept_ = float(data["intercept"][0])
        if bool(data["has_calibrator"][0]):
            pipeline.calibrator = PlattCalibrator(
                slope_=float(data["calibrator_slope"][0]),
                intercept_=float(data["calibrator_intercept"][0]),
            )
        return pipeline


@dataclass
class TwoModelUplift:
    control_model: ProbabilityPipeline
    treated_model: ProbabilityPipeline

    def predict_uplift(self, x: np.ndarray) -> np.ndarray:
        churn_if_control = self.control_model.predict_proba(x)
        churn_if_treated = self.treated_model.predict_proba(x)
        return churn_if_control - churn_if_treated


def fit_probability_pipeline(
    x_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    x_calibration: np.ndarray | None = None,
    y_calibration: np.ndarray | None = None,
    random_seed: int = 42,
) -> ProbabilityPipeline:
    model = LogisticRegressionGD(random_seed=random_seed)
    pipeline = ProbabilityPipeline(feature_names=feature_names, model=model)
    return pipeline.fit(x_train, y_train, x_calibration=x_calibration, y_calibration=y_calibration)


def fit_t_learner_uplift(
    x_train: np.ndarray,
    y_train: np.ndarray,
    treatment: np.ndarray,
    feature_names: list[str],
    random_seed: int = 42,
) -> TwoModelUplift:
    treatment = np.asarray(treatment, dtype=int)
    treated_idx = treatment == 1
    control_idx = treatment == 0
    if treated_idx.sum() < 50 or control_idx.sum() < 50:
        raise ValueError("Need at least 50 treated and 50 control rows to train uplift models")

    treated_model = fit_probability_pipeline(
        x_train[treated_idx],
        y_train[treated_idx],
        feature_names,
        random_seed=random_seed + 11,
    )
    control_model = fit_probability_pipeline(
        x_train[control_idx],
        y_train[control_idx],
        feature_names,
        random_seed=random_seed + 23,
    )
    return TwoModelUplift(control_model=control_model, treated_model=treated_model)
