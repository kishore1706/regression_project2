"""
Refinance prediction script with simple validation and persistence.

Adds docstrings and exception handling around data loading,
model training/tuning, and prediction export.
"""

import logging
import pickle
import pathlib
from typing import Any, Dict, Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
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

logging.basicConfig(level=logging.INFO)


class RefinancePredictor:
    """Encapsulates the refinance prediction workflow."""

    def __init__(
        self,
        data_path: str,
        model_path: str = "fmodel.sav",
        output_path: str = "final_predictions.xlsx",
        test_size: float = 0.1,
        random_state: int = 42,
    ):
        self.data_path = data_path
        self.model_path = pathlib.Path(model_path)
        if not self.model_path.is_absolute():
            self.model_path = pathlib.Path(__file__).resolve().parent / self.model_path
        self.scaler_path = self.model_path.with_name("scaler.sav")
        self.output_path = pathlib.Path(output_path)
        if not self.output_path.is_absolute():
            self.output_path = pathlib.Path(__file__).resolve().parent / self.output_path
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
        self.feature_data_df = self.df[self.df["Refinance"].isna()]
        self.actual_df = self.df[~self.df["Refinance"].isna()]
        self.actual_df = self.actual_df.drop(columns=["Date"], errors="ignore")
        self.X = self.actual_df.drop(columns=["Refinance", "Date"], errors="ignore")
        self.y = self.actual_df["Refinance"]

        if self.X.empty or self.y.empty:
            raise ValueError("The dataset does not contain valid features or target values.")

    def split_data(self) -> None:
        logging.info("Splitting data into train and test sets")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=self.test_size,
            random_state=self.random_state,
        )

    @staticmethod
    def model_validation(
        method: str,
        X_train: Any,
        X_test: Any,
        y_train: Any,
        y_test: Any,
    ) -> pd.DataFrame:
        logging.info("Validating models on %s data", method)
        validation_table = []
        algorithms = [
            LinearRegression,
            Ridge,
            Lasso,
            KNeighborsRegressor,
            DecisionTreeRegressor,
            RandomForestRegressor,
            BaggingRegressor,
        ]

        for algo_class in algorithms:
            try:
                algo = algo_class()
                algo.fit(X_train, y_train)
                predictions = algo.predict(X_train if method == "train" else X_test)
                ground_truth = y_train if method == "train" else y_test

                mae = mean_absolute_error(ground_truth, predictions)
                mse = mean_squared_error(ground_truth, predictions)
                rmse = float(np.sqrt(mse))
                mape = mean_absolute_percentage_error(ground_truth, predictions)
                r2 = r2_score(ground_truth, predictions)
                validation_table.append([algo_class.__name__, mae, mse, rmse, mape, r2])
            except Exception:
                logging.exception("Model %s failed during validation", algo_class.__name__)
                validation_table.append([algo_class.__name__, np.nan, np.nan, np.nan, np.nan, np.nan])

        return pd.DataFrame(
            validation_table,
            columns=["method name", "mae", "mse", "rmse", "mape", "r2"],
        )

    def fit_linear_models(self) -> None:
        logging.info("Fitting linear models for coefficient inspection")
        try:
            lr = LinearRegression()
            lr.fit(self.X_train, self.y_train)
            coeff_df = pd.DataFrame(
                {"column name": self.X_train.columns, "coeff_value_lr": pd.Series(lr.coef_)},
            )
            logging.info("Linear regression coefficients:\n%s", coeff_df)
            plt.rcParams["figure.figsize"] = (13, 6)
            plt.bar(coeff_df["column name"], coeff_df["coeff_value_lr"])
        except Exception:
            logging.exception("Linear regression inspection failed")

        try:
            rr = Ridge()
            rr.fit(self.X_train, self.y_train)
            _ = pd.Series(rr.coef_)
        except Exception:
            logging.exception("Ridge regression failed")

        try:
            lasso = Lasso()
            lasso.fit(self.X_train, self.y_train)
            _ = pd.Series(lasso.coef_)
        except Exception:
            logging.exception("Lasso regression failed")

    def plot_correlation_heatmap(self) -> None:
        logging.info("Plotting correlation heatmap")
        try:
            plt.figure(figsize=(12, 6))
            correlation_matrix = self.actual_df.corr()
            sns.heatmap(correlation_matrix, annot=True, cmap=plt.cm.CMRmap_r)
            plt.tight_layout()
            plt.savefig("correlation_heatmap.png")
            plt.close()
        except Exception:
            logging.exception("Failed to draw correlation heatmap")

    def drop_high_corr_columns(self) -> None:
        logging.info("Dropping highly correlated columns and re-splitting data")
        self.actual_df = self.actual_df.drop(columns=["Week No", "Year"], errors="ignore")
        self.actual_df = self.actual_df.drop(columns=["Date"], errors="ignore")
        self.X = self.actual_df.drop(columns=["Refinance", "Date"], errors="ignore")
        self.y = self.actual_df["Refinance"]
        self.split_data()

    def scale_data(self) -> None:
        logging.info("Scaling numeric features")
        try:
            self.scaler = StandardScaler()
            self.X_train_scaled = self.scaler.fit_transform(self.X_train)
            self.X_test_scaled = self.scaler.transform(self.X_test)
        except Exception:
            logging.exception("Scaling failed")
            raise

    def tune_ridge_lasso(self) -> None:
        logging.info("Tuning Ridge and Lasso hyperparameters")
        try:
            ridge = Ridge()
            ridge_params = {
                "alpha": [
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
            logging.info(
                "Ridge best params: %s, score: %s",
                ridge_grid.best_params_,
                ridge_grid.best_score_,
            )
        except Exception:
            logging.exception("Ridge GridSearchCV failed")

        try:
            lasso = Lasso(max_iter=5000)
            lasso_params = {"alpha": [0.01, 0.1, 1, 5, 10, 20, 30, 35, 40, 50, 100]}
            lasso_grid = GridSearchCV(lasso, lasso_params, scoring="neg_mean_squared_error", cv=5)
            lasso_grid.fit(self.X_train_scaled, self.y_train)
            logging.info(
                "Lasso best params: %s, score: %s",
                lasso_grid.best_params_,
                lasso_grid.best_score_,
            )
        except Exception:
            logging.exception("Lasso GridSearchCV failed")

    def train_random_forest(self) -> None:
        logging.info("Training RandomForestRegressor")
        try:
            self.rf_model = RandomForestRegressor()
            if self.X_train_scaled is None or self.X_test_scaled is None:
                raise ValueError("Scaled training and test sets are required for RandomForest training.")
            self.rf_model.fit(self.X_train_scaled, self.y_train)
            predictions = self.rf_model.predict(self.X_test_scaled)
            logging.info("RandomForest training score: %s", self.rf_model.score(self.X_train_scaled, self.y_train))
            logging.info("RandomForest testing score: %s", self.rf_model.score(self.X_test_scaled, self.y_test))
            logging.info("RandomForest test R2: %s", r2_score(self.y_test, predictions))
        except Exception:
            logging.exception("RandomForest training or evaluation failed")
            raise

    def persist_model(self) -> None:
        logging.info("Saving trained model to %s", self.model_path)
        if self.rf_model is None:
            raise ValueError("No trained model is available to persist.")
        with open(self.model_path, "wb") as handle:
            pickle.dump(self.rf_model, handle)

    def persist_scaler(self) -> None:
        logging.info("Saving scaler to %s", self.scaler_path)
        if self.scaler is None:
            raise ValueError("No scaler is available to persist.")
        joblib.dump(self.scaler, self.scaler_path)

    def predict_missing(self) -> pd.DataFrame:
        logging.info("Predicting missing refinance values")
        feature_data = self.feature_data_df.drop(columns=["Date", "Week No", "Year", "Refinance"], errors="ignore")
        with open(self.model_path, "rb") as handle:
            rf_model = pickle.load(handle)
        predictions = rf_model.predict(feature_data)
        feature_data_df = self.feature_data_df.copy()
        feature_data_df["predicted_Refinance"] = predictions
        return feature_data_df

    def save_predictions(self, predictions: pd.DataFrame) -> None:
        logging.info("Writing predictions to %s", self.output_path)
        predictions.to_excel(self.output_path, index=False)

    def run(self) -> Dict[str, Any]:
        self.load_data()
        self.split_data()

        validation_train = self.model_validation(
            "train", self.X_train, self.X_test, self.y_train, self.y_test
        )
        validation_test = self.model_validation(
            "test", self.X_train, self.X_test, self.y_train, self.y_test
        )

        self.fit_linear_models()
        self.plot_correlation_heatmap()
        self.drop_high_corr_columns()
        self.scale_data()

        validation_train_scaled = self.model_validation(
            "train",
            self.X_train_scaled,
            self.X_test_scaled,
            self.y_train,
            self.y_test,
        )
        validation_test_scaled = self.model_validation(
            "test",
            self.X_train_scaled,
            self.X_test_scaled,
            self.y_train,
            self.y_test,
        )

        self.tune_ridge_lasso()
        self.train_random_forest()
        self.persist_model()
        self.persist_scaler()

        predictions = self.predict_missing()
        self.save_predictions(predictions)

        return {
            "validation_train": validation_train,
            "validation_test": validation_test,
            "validation_train_scaled": validation_train_scaled,
            "validation_test_scaled": validation_test_scaled,
            "predictions": predictions,
        }


def main() -> None:
    data_path = "C:\\Users\\chandini ch\\Downloads\\Weekly_Refinance_Volumes_Data.xlsx"
    predictor = RefinancePredictor(data_path=data_path)
    try:
        results = predictor.run()
        logging.info("Pipeline finished successfully")
        logging.info("Validation results:\n%s", results["validation_test"])
    except Exception:
        logging.exception("Pipeline execution failed")
        raise


if __name__ == "__main__":
    main()
