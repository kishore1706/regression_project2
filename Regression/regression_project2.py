"""Production-ready refinance regression pipeline.

This module provides a single main class with helper methods for loading data,
training regression models, tuning, saving a persisted model, and predicting
missing refinance values.
"""

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


MODEL_CLASSES = [
    LinearRegression,
    Ridge,
    Lasso,
    KNeighborsRegressor,
    DecisionTreeRegressor,
    RandomForestRegressor,
    BaggingRegressor,
]


def validate_models(
    method: str,
    X_train: Any,
    X_test: Any,
    y_train: Any,
    y_test: Any,
) -> pd.DataFrame:
    """Evaluate a group of regression algorithms and return metrics."""
    logging.info("Validating models on %s set", method)
    rows: List[List[Any]] = []

    for model_class in MODEL_CLASSES:
        try:
            model = model_class()
            model.fit(X_train, y_train)
            predictions = model.predict(X_train if method == "train" else X_test)
            actuals = y_train if method == "train" else y_test

            mae = mean_absolute_error(actuals, predictions)
            mse = mean_squared_error(actuals, predictions)
            rmse = float(np.sqrt(mse))
            mape = mean_absolute_percentage_error(actuals, predictions)
            r2 = r2_score(actuals, predictions)

            rows.append([model_class.__name__, mae, mse, rmse, mape, r2])
        except Exception as exc:
            logging.exception("Failed validating %s", model_class.__name__)
            rows.append([model_class.__name__, np.nan, np.nan, np.nan, np.nan, np.nan])

    return pd.DataFrame(rows, columns=["method name", "mae", "mse", "rmse", "mape", "r2"])


def save_pickle(obj: Any, path: Path) -> None:
    """Persist a Python object to disk."""
    with path.open("wb") as file_handle:
        pickle.dump(obj, file_handle)
    logging.info("Saved pickle to %s", path)


def load_pickle(path: Path) -> Any:
    """Load a persisted Python object from disk."""
    with path.open("rb") as file_handle:
        return pickle.load(file_handle)


class RefinanceRegressionPipeline:
    """Main class for refinance regression training and prediction."""

    def __init__(
        self,
        data_path: str,
        model_path: str = "fmodel.sav",
        output_path: str = "final_predictions.xlsx",
        test_size: float = 0.1,
        random_state: int = 42,
    ):
        self.data_path = Path(data_path)
        self.model_path = Path(model_path)
        self.output_path = Path(output_path)
        self.test_size = test_size
        self.random_state = random_state

        self.df: Optional[pd.DataFrame] = None
        self.feature_data_df: Optional[pd.DataFrame] = None
        self.actual_df: Optional[pd.DataFrame] = None
        self.X: Optional[pd.DataFrame] = None
        self.y: Optional[pd.Series] = None
        self.X_train: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.y_test: Optional[pd.Series] = None
        self.X_train_scaled: Optional[np.ndarray] = None
        self.X_test_scaled: Optional[np.ndarray] = None
        self.scaler: Optional[StandardScaler] = None
        self.rf_model: Optional[RandomForestRegressor] = None

    def load_data(self) -> None:
        logging.info("Loading data from %s", self.data_path)
        self.df = pd.read_excel(self.data_path)
        self.feature_data_df = self.df[self.df["Refinance"].isna()].copy()
        self.actual_df = self.df[~self.df["Refinance"].isna()].copy()
        self.actual_df = self.actual_df.drop(columns=["Date"], errors="ignore")
        self.X = self.actual_df.drop(columns=["Refinance", "Date"], errors="ignore")
        self.y = self.actual_df["Refinance"]

        if self.X.empty or self.y.empty:
            raise ValueError("Dataset must contain features and target values.")

    def split_data(self) -> None:
        logging.info("Splitting training and test data")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=self.test_size,
            random_state=self.random_state,
        )

    def fit_linear_models(self) -> pd.DataFrame:
        logging.info("Fitting linear models")
        coefficient_data = []
        for model_class in [LinearRegression, Ridge, Lasso]:
            try:
                model = model_class()
                model.fit(self.X_train, self.y_train)
                coefficient_data.append(
                    {
                        "model": model_class.__name__,
                        "coefficients": model.coef_.tolist(),
                        "intercept": float(model.intercept_),
                    }
                )
            except Exception:
                logging.exception("Failed fitting %s", model_class.__name__)
        return pd.DataFrame(coefficient_data)

    def drop_highly_correlated_columns(self) -> None:
        logging.info("Dropping highly correlated columns")
        self.actual_df = self.actual_df.drop(columns=["Week No", "Year"], errors="ignore")
        self.actual_df = self.actual_df.drop(columns=["Date"], errors="ignore")
        self.X = self.actual_df.drop(columns=["Refinance", "Date"], errors="ignore")
        self.y = self.actual_df["Refinance"]
        self.split_data()

    def scale_data(self) -> None:
        logging.info("Scaling numeric features")
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

    def tune_ridge_lasso(self) -> Dict[str, Any]:
        logging.info("Tuning Ridge and Lasso models")
        results: Dict[str, Any] = {}

        ridge = Ridge()
        ridge_params = {
            "alpha": [
                1e-15,
                1e-10,
                1e-8,
                1e-6,
                0.01,
                0.1,
                1,
                5,
                10,
                20,
                30,
                35,
                40,
                50,
                100,
            ]
        }
        ridge_grid = GridSearchCV(ridge, ridge_params, scoring="neg_mean_squared_error", cv=5)
        ridge_grid.fit(self.X_train_scaled, self.y_train)
        results["ridge"] = {
            "best_params": ridge_grid.best_params_,
            "best_score": float(ridge_grid.best_score_),
        }

        lasso = Lasso(max_iter=5000)
        lasso_params = {"alpha": [0.01, 0.1, 1, 5, 10, 20, 30, 35, 40, 50, 100]}
        lasso_grid = GridSearchCV(lasso, lasso_params, scoring="neg_mean_squared_error", cv=5)
        lasso_grid.fit(self.X_train_scaled, self.y_train)
        results["lasso"] = {
            "best_params": lasso_grid.best_params_,
            "best_score": float(lasso_grid.best_score_),
        }

        return results

    def train_random_forest(self) -> None:
        logging.info("Training RandomForestRegressor")
        self.rf_model = RandomForestRegressor()
        self.rf_model.fit(self.X_train, self.y_train)

    def persist_model(self) -> None:
        if self.rf_model is None:
            raise RuntimeError("Random forest model is not trained yet.")
        save_pickle(self.rf_model, self.model_path)

    def predict_missing(self) -> pd.DataFrame:
        logging.info("Predicting missing refinance values")
        if self.feature_data_df is None or self.feature_data_df.empty:
            raise ValueError("No missing refinance data available for prediction.")

        feature_data = self.feature_data_df.drop(
            columns=["Date", "Week No", "Year", "Refinance"], errors="ignore"
        )
        rf_model = load_pickle(self.model_path)
        predictions = rf_model.predict(feature_data)

        output_df = self.feature_data_df.copy()
        output_df["predicted_Refinance"] = predictions
        return output_df

    def save_predictions(self, predictions: pd.DataFrame) -> None:
        predictions.to_excel(self.output_path, index=False)
        logging.info("Saved predictions to %s", self.output_path)

    def run(self) -> Dict[str, Any]:
        self.load_data()
        self.split_data()

        validation_train = validate_models(
            "train", self.X_train, self.X_test, self.y_train, self.y_test
        )
        validation_test = validate_models(
            "test", self.X_train, self.X_test, self.y_train, self.y_test
        )

        coefficients = self.fit_linear_models()

        self.drop_highly_correlated_columns()
        self.scale_data()

        validation_train_scaled = validate_models(
            "train",
            self.X_train_scaled,
            self.X_test_scaled,
            self.y_train,
            self.y_test,
        )
        validation_test_scaled = validate_models(
            "test",
            self.X_train_scaled,
            self.X_test_scaled,
            self.y_train,
            self.y_test,
        )

        tuning_results = self.tune_ridge_lasso()
        self.train_random_forest()
        self.persist_model()

        predictions = self.predict_missing()
        self.save_predictions(predictions)

        return {
            "validation_train": validation_train,
            "validation_test": validation_test,
            "validation_train_scaled": validation_train_scaled,
            "validation_test_scaled": validation_test_scaled,
            "coefficients": coefficients,
            "tuning_results": tuning_results,
            "predictions": predictions,
        }


def main() -> None:
    data_path = "C:\\Users\\chandini ch\\Downloads\\Weekly_Refinance_Volumes_Data.xlsx"
    pipeline = RefinanceRegressionPipeline(data_path=data_path)
    results = pipeline.run()
    logging.info("Pipeline completed successfully")
    logging.info("Validation summary:\n%s", results["validation_test"])


if __name__ == "__main__":
    main()
