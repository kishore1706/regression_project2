import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Make sure we can import the local predictor module in the same folder.
sys.path.append(str(Path(__file__).resolve().parent))
from regression_projectt2 import RefinancePredictor

MODEL_PATH = Path(__file__).resolve().parent / "fmodel.sav"


def load_input_data(uploaded_file: st.uploaded_file_manager.UploadedFile) -> pd.DataFrame:
    if uploaded_file.name.lower().endswith(".csv"):
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file)


def predict_missing_refinance(df: pd.DataFrame, model_path: Path) -> pd.DataFrame:
    predictor = RefinancePredictor(data_path="")
    predictor.model_path = str(model_path)
    predictor.feature_data_df = df[df["Refinance"].isna()].copy()

    if predictor.feature_data_df.empty:
        raise ValueError("No rows with missing Refinance values were found in the uploaded dataset.")

    return predictor.predict_missing()


def main() -> None:
    st.title("Refinance Prediction Frontend")
    st.write(
        "Upload a dataset containing both actual refinance rows and missing refinance rows. "
        "The app will use the saved model to predict values for rows missing `Refinance`."
    )

    st.markdown("**Required columns:** `Refinance`, `Date`, `Week No`, `Year`, plus all other feature columns.")

    uploaded_file = st.file_uploader("Upload Excel or CSV file", type=["csv", "xlsx"])
    model_exists = MODEL_PATH.exists()

    if not model_exists:
        st.error(f"Saved model not found at {MODEL_PATH}. Run the training pipeline first to create `fmodel.sav`.")
        return

    if uploaded_file is not None:
        try:
            data = load_input_data(uploaded_file)
            st.subheader("Uploaded data preview")
            st.dataframe(data.head())

            if st.button("Predict missing refinance values"):
                with st.spinner("Predicting..."):
                    predictions = predict_missing_refinance(data, MODEL_PATH)
                    st.success("Prediction complete")
                    st.subheader("Predicted refinance values")
                    st.dataframe(predictions)

                    output_file = Path("predicted_refinance.xlsx")
                    predictions.to_excel(output_file, index=False)
                    st.markdown(
                        f"Download the predictions: [{output_file.name}]({output_file.name})"
                    )
        except Exception as error:
            st.error(f"Prediction failed: {error}")

    else:
        st.info("Upload a dataset file to enable prediction.")


if __name__ == "__main__":
    main()
