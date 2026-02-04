import io
import textwrap
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans


st.set_page_config(page_title="DataSense", page_icon="📊", layout="wide")


TARGET_NAME_HINTS = {"target", "label", "y", "class", "outcome"}


@dataclass
class DetectionResult:
    learning_type: str
    reason: str
    candidate_targets: List[str]


def detect_learning_type(df: pd.DataFrame) -> DetectionResult:
    candidates = [
        col
        for col in df.columns
        if str(col).strip().lower() in TARGET_NAME_HINTS
    ]
    if candidates:
        return DetectionResult(
            learning_type="Supervised Learning",
            reason=(
                "This dataset contains a target-like column name, "
                "so Supervised Learning was applied."
            ),
            candidate_targets=candidates,
        )
    return DetectionResult(
        learning_type="Unsupervised Learning",
        reason=(
            "No obvious target/label column was detected, so Unsupervised "
            "Learning was applied."
        ),
        candidate_targets=[],
    )


def get_problem_type(series: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(series):
        unique_vals = series.dropna().nunique()
        if unique_vals <= 10:
            return "Classification"
        return "Regression"
    return "Classification"


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(axis=0, how="all").dropna(axis=1, how="all")


def encode_features(X: pd.DataFrame) -> pd.DataFrame:
    X_encoded = X.copy()
    for col in X_encoded.columns:
        if X_encoded[col].dtype == "object":
            X_encoded[col] = X_encoded[col].fillna("Unknown")
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
        else:
            X_encoded[col] = X_encoded[col].fillna(X_encoded[col].median())
    return X_encoded


def show_banner(learning_type: str, reason: str) -> None:
    st.markdown(
        f"""
        <div style="padding: 16px; background: #f0f6ff; border-left: 6px solid #4c8bf5; border-radius: 8px;">
            <h3 style="margin: 0;">📊 Detected Learning Type: {learning_type}</h3>
            <p style="margin: 8px 0 0;">{reason}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_dataset_overview(df: pd.DataFrame) -> None:
    st.subheader("Dataset Overview")
    st.write(df.head())
    st.write("Shape:", df.shape)
    st.write("Missing values:")
    st.write(df.isna().sum())


def render_visualizations(df: pd.DataFrame) -> None:
    st.subheader("Visualizations")
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if len(numeric_cols) >= 2:
        x_axis = st.selectbox("Scatter X-axis", numeric_cols, key="scatter_x")
        y_axis = st.selectbox("Scatter Y-axis", numeric_cols, key="scatter_y")
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x=x_axis, y=y_axis, ax=ax)
        st.pyplot(fig)
    else:
        st.info("Add at least two numeric columns for scatter plots.")

    if len(numeric_cols) >= 2:
        fig, ax = plt.subplots(figsize=(8, 5))
        corr = df[numeric_cols].corr()
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.info("Correlation heatmap requires at least two numeric columns.")

    if numeric_cols:
        method = st.selectbox("Outlier Detection Method", ["IQR", "Z-score"])
        col = st.selectbox("Outlier Feature", numeric_cols, key="outlier_feature")
        series = df[col].dropna()
        if method == "IQR":
            q1, q3 = series.quantile([0.25, 0.75])
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            outliers = df[(df[col] < lower) | (df[col] > upper)]
        else:
            z_scores = (series - series.mean()) / series.std()
            outliers = df.loc[z_scores.abs() > 3]
        fig, ax = plt.subplots()
        sns.boxplot(y=df[col], ax=ax)
        st.pyplot(fig)
        st.write(f"Outliers detected: {len(outliers)}")
    else:
        st.info("Outlier detection requires numeric columns.")


def train_supervised(df: pd.DataFrame, target: str) -> None:
    st.subheader("Supervised Learning")
    problem_type = get_problem_type(df[target])
    st.write(f"Detected problem type: **{problem_type}**")

    X = df.drop(columns=[target])
    y = df[target]
    X_encoded = encode_features(X)

    test_size = st.slider("Train-Test Split (Test Size)", 0.1, 0.5, 0.2, 0.05)
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=test_size, random_state=42
    )

    if problem_type == "Classification":
        model_options = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(),
            "K-Nearest Neighbors": KNeighborsClassifier(),
            "Random Forest": RandomForestClassifier(),
        }
    else:
        model_options = {
            "Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(),
            "Random Forest": RandomForestRegressor(),
        }

    model_name = st.selectbox("Choose a model", list(model_options.keys()))
    model = model_options[model_name]
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    st.write("Model Results")
    if problem_type == "Classification":
        preds = predictions
        st.write("Sample predictions:")
        st.write(preds[:5])
        if hasattr(model, "predict_proba"):
            st.write("Sample prediction probabilities:")
            st.write(model.predict_proba(X_test)[:5])
        st.write("Accuracy:", accuracy_score(y_test, preds))
        st.write("Confusion Matrix:")
        st.write(confusion_matrix(y_test, preds))
        st.text("Classification Report:")
        st.text(classification_report(y_test, preds))
    else:
        st.write("Sample predictions:")
        st.write(predictions[:5])
        st.write("MAE:", mean_absolute_error(y_test, predictions))
        st.write("MSE:", mean_squared_error(y_test, predictions))
        st.write("R2 Score:", r2_score(y_test, predictions))

    st.markdown(
        "**Model applicability check:** Logistic Regression, Decision Tree, "
        "Random Forest, and K-NN are available when appropriate. Choose the "
        "one that best fits your dataset and interpretability needs."
    )


def train_unsupervised(df: pd.DataFrame) -> None:
    st.subheader("Unsupervised Learning")
    numeric_df = df.select_dtypes(include=["number"]).dropna()
    if numeric_df.empty:
        st.warning("Unsupervised learning needs numeric columns.")
        return

    use_pca = st.checkbox("Apply PCA (Dimensionality Reduction)")
    features = numeric_df

    if use_pca:
        pca_components = st.slider("PCA Components", 2, min(5, features.shape[1]), 2)
        pca = PCA(n_components=pca_components)
        features = pd.DataFrame(pca.fit_transform(features))

    k = st.slider("Number of Clusters (K)", 2, 10, 3)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(features)
    df_clustered = features.copy()
    df_clustered["cluster"] = clusters

    st.write("Cluster Counts:")
    st.write(df_clustered["cluster"].value_counts())

    if features.shape[1] >= 2:
        fig, ax = plt.subplots()
        sns.scatterplot(
            x=features.iloc[:, 0],
            y=features.iloc[:, 1],
            hue=clusters,
            palette="tab10",
            ax=ax,
        )
        st.pyplot(fig)


def main() -> None:
    st.title("DataSense: Machine Learning Analytics")
    st.write(
        "Upload a CSV dataset and explore Supervised or Unsupervised learning "
        "with beginner-friendly, powerful analytics."
    )

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if not uploaded_file:
        st.info("Upload a dataset to get started.")
        return

    df = pd.read_csv(uploaded_file)
    df = clean_dataframe(df)

    detection = detect_learning_type(df)
    show_banner(detection.learning_type, detection.reason)

    render_dataset_overview(df)

    if detection.learning_type == "Supervised Learning":
        default_target = detection.candidate_targets[0] if detection.candidate_targets else None
        target = st.selectbox("Select target column", df.columns, index=(df.columns.get_loc(default_target) if default_target else 0))
        train_supervised(df, target)
    else:
        target_choice = st.checkbox("I actually have a target column")
        if target_choice:
            target = st.selectbox("Select target column", df.columns)
            train_supervised(df, target)
        else:
            train_unsupervised(df)

    render_visualizations(df)

    st.markdown(
        "---\n"
        "Need more advanced AI advice? Try experimenting with feature selection, "
        "model comparison, and hyperparameter tuning to maximize performance."
    )


if __name__ == "__main__":
    main()
