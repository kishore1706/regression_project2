import pickle
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set Streamlit page config
st.set_page_config(
    page_title="Refinance Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        color: #1f77b4;
        text-align: center;
        font-size: 2.5em;
        margin-bottom: 10px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_data
def load_predictions(predictions_path: str) -> pd.DataFrame:
    """Load predictions from Excel file."""
    try:
        predictions_df = pd.read_excel(predictions_path)
        logger.info(f"Loaded {len(predictions_df)} predictions from {predictions_path}")
        return predictions_df
    except FileNotFoundError:
        st.error(f"Predictions file not found at {predictions_path}")
        return pd.DataFrame()
    except Exception as e:
        logger.exception("Error loading predictions")
        st.error(f"Error loading predictions: {str(e)}")
        return pd.DataFrame()


@st.cache_resource
def load_model(model_path: str):
    """Load the trained model."""
    try:
        with open(model_path, "rb") as handle:
            model = pickle.load(handle)
        logger.info(f"Model loaded successfully from {model_path}")
        return model
    except FileNotFoundError:
        st.error(f"Model file not found at {model_path}")
        return None
    except Exception as e:
        logger.exception("Error loading model")
        st.error(f"Error loading model: {str(e)}")
        return None


def display_summary_metrics(predictions_df: pd.DataFrame) -> None:
    """Display summary metrics in the main dashboard."""
    if "predicted_Refinance" not in predictions_df.columns:
        st.warning("predicted_Refinance column not found in predictions data.")
        return

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Total Predictions",
            value=len(predictions_df),
            delta="records",
        )

    with col2:
        avg_prediction = predictions_df["predicted_Refinance"].mean()
        st.metric(
            label="Average Predicted Refinance",
            value=f"{avg_prediction:.2f}",
        )

    with col3:
        max_prediction = predictions_df["predicted_Refinance"].max()
        st.metric(
            label="Max Predicted Value",
            value=f"{max_prediction:.2f}",
        )

    with col4:
        min_prediction = predictions_df["predicted_Refinance"].min()
        st.metric(
            label="Min Predicted Value",
            value=f"{min_prediction:.2f}",
        )


def display_predictions_table(predictions_df: pd.DataFrame) -> None:
    """Display predictions in an interactive table with sorting and filtering."""
    st.subheader("📋 Detailed Predictions")

    # Create columns for filtering
    col1, col2, col3 = st.columns(3)

    with col1:
        sort_column = st.selectbox(
            "Sort by:",
            ["predicted_Refinance"] + [col for col in predictions_df.columns if col != "predicted_Refinance"],
            index=0,
        )

    with col2:
        sort_order = st.radio("Order:", ["Ascending", "Descending"], index=0, horizontal=True)

    with col3:
        rows_to_display = st.slider(
            "Rows to display:",
            min_value=5,
            max_value=len(predictions_df),
            value=min(20, len(predictions_df)),
            step=5,
        )

    ascending = sort_order == "Ascending"
    sorted_df = predictions_df.sort_values(by=sort_column, ascending=ascending)
    
    # Display table
    st.dataframe(
        sorted_df.head(rows_to_display),
        use_container_width=True,
        height=400,
    )

    # Download button
    csv = sorted_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Predictions as CSV",
        data=csv,
        file_name="refinance_predictions.csv",
        mime="text/csv",
    )


def display_distribution_plot(predictions_df: pd.DataFrame) -> None:
    """Display distribution of predicted values."""
    if "predicted_Refinance" not in predictions_df.columns:
        return

    st.subheader("📈 Prediction Distribution")

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(predictions_df["predicted_Refinance"], bins=30, color="#1f77b4", edgecolor="black", alpha=0.7)
        ax.set_xlabel("Predicted Refinance Value")
        ax.set_ylabel("Frequency")
        ax.set_title("Distribution of Predicted Refinance Values")
        ax.grid(axis="y", alpha=0.3)
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(10, 5))
        predictions_df["predicted_Refinance"].plot(
            kind="box",
            ax=ax,
            color="#1f77b4",
        )
        ax.set_ylabel("Predicted Refinance Value")
        ax.set_title("Box Plot of Predicted Values")
        ax.grid(axis="y", alpha=0.3)
        st.pyplot(fig)


def display_statistics(predictions_df: pd.DataFrame) -> None:
    """Display detailed statistics about predictions."""
    if "predicted_Refinance" not in predictions_df.columns:
        return

    st.subheader("📊 Statistical Summary")

    stats_data = {
        "Metric": [
            "Count",
            "Mean",
            "Std Dev",
            "Min",
            "25%",
            "Median (50%)",
            "75%",
            "Max",
        ],
        "Value": [
            len(predictions_df),
            f"{predictions_df['predicted_Refinance'].mean():.2f}",
            f"{predictions_df['predicted_Refinance'].std():.2f}",
            f"{predictions_df['predicted_Refinance'].min():.2f}",
            f"{predictions_df['predicted_Refinance'].quantile(0.25):.2f}",
            f"{predictions_df['predicted_Refinance'].median():.2f}",
            f"{predictions_df['predicted_Refinance'].quantile(0.75):.2f}",
            f"{predictions_df['predicted_Refinance'].max():.2f}",
        ],
    }

    stats_df = pd.DataFrame(stats_data)
    st.dataframe(stats_df, use_container_width=True, hide_index=True)


def display_top_predictions(predictions_df: pd.DataFrame, n: int = 10) -> None:
    """Display top N and bottom N predictions."""
    if "predicted_Refinance" not in predictions_df.columns:
        return

    st.subheader("🔝 Top & Bottom Predictions")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Top Predictions:**")
        top_df = predictions_df.nlargest(n, "predicted_Refinance")
        st.dataframe(top_df, use_container_width=True, hide_index=True)

    with col2:
        st.write("**Bottom Predictions:**")
        bottom_df = predictions_df.nsmallest(n, "predicted_Refinance")
        st.dataframe(bottom_df, use_container_width=True, hide_index=True)


def display_prediction_ranges(predictions_df: pd.DataFrame) -> None:
    """Display predictions grouped by ranges."""
    if "predicted_Refinance" not in predictions_df.columns:
        return

    st.subheader("📊 Predictions by Range")

    min_val = predictions_df["predicted_Refinance"].min()
    max_val = predictions_df["predicted_Refinance"].max()

    # Create bins
    bins = np.linspace(min_val, max_val, 11)
    ranges = pd.cut(predictions_df["predicted_Refinance"], bins=bins)

    range_counts = ranges.value_counts().sort_index()

    col1, col2 = st.columns([2, 1])

    with col1:
        fig, ax = plt.subplots(figsize=(10, 5))
        range_counts.plot(kind="bar", ax=ax, color="#1f77b4", edgecolor="black")
        ax.set_xlabel("Prediction Range")
        ax.set_ylabel("Count")
        ax.set_title("Number of Predictions by Range")
        plt.xticks(rotation=45, ha="right")
        ax.grid(axis="y", alpha=0.3)
        st.pyplot(fig)

    with col2:
        range_data = pd.DataFrame({
            "Range": [str(interval) for interval in range_counts.index],
            "Count": range_counts.values,
        })
        st.dataframe(range_data, use_container_width=True, hide_index=True)


def main() -> None:
    """Main Streamlit application."""
    # Define paths
    current_dir = Path(__file__).parent
    model_path = current_dir / "fmodel.sav"
    predictions_path = current_dir / "final_predictions.xlsx"

    # Display header
    st.markdown('<h1 class="main-header">🎯 Refinance Prediction Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("---")

    # Sidebar
    st.sidebar.title("Dashboard Controls")
    st.sidebar.info(
        """
        This dashboard displays predictions for refinance volumes.
        
        **Model Path:** `fmodel.sav`
        **Predictions Path:** `final_predictions.xlsx`
        """
    )

    # Load data
    with st.spinner("Loading data..."):
        predictions_df = load_predictions(str(predictions_path))
        model = load_model(str(model_path))

    if predictions_df.empty:
        st.error("No predictions data available. Please run the training pipeline first.")
        st.stop()

    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Overview", "📋 Data Table", "📈 Statistics", "🔍 Analysis"]
    )

    with tab1:
        display_summary_metrics(predictions_df)
        st.markdown("---")
        display_distribution_plot(predictions_df)

    with tab2:
        display_predictions_table(predictions_df)

    with tab3:
        display_statistics(predictions_df)
        st.markdown("---")
        display_top_predictions(predictions_df, n=10)

    with tab4:
        display_prediction_ranges(predictions_df)

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #999; font-size: 0.9em;">
        <p>Refinance Prediction Dashboard | Powered by Streamlit</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
