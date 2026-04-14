import io
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    silhouette_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.cluster import KMeans


st.set_page_config(page_title="DataSense", page_icon="📊", layout="wide")


TARGET_NAME_HINTS = {"target", "label", "y", "class", "outcome"}


@dataclass
class DetectionResult:
    learning_type: str
    reason: str
    candidate_targets: List[str]


@dataclass
class AppSettings:
    random_state: int
    drop_duplicates: bool
    max_rows: int
    preview_rows: int
    missing_strategy: str


def render_app_settings() -> AppSettings:
    st.sidebar.header("Experiment Settings")
    random_state = st.sidebar.number_input("Random seed", min_value=0, max_value=9999, value=42)
    drop_duplicates = st.sidebar.checkbox("Drop duplicate rows", value=False)
    max_rows = st.sidebar.slider("Max rows to use (0 = all)", min_value=0, max_value=50000, value=0, step=500)
    preview_rows = st.sidebar.slider("Preview rows", min_value=3, max_value=25, value=5)
    missing_strategy = st.sidebar.selectbox(
        "Missing-value strategy",
        ["keep", "drop_rows", "median_mode", "mean_mode"],
        index=2,
    )
    return AppSettings(
        random_state=int(random_state),
        drop_duplicates=drop_duplicates,
        max_rows=int(max_rows),
        preview_rows=int(preview_rows),
        missing_strategy=str(missing_strategy),
    )


def detect_learning_type(df: pd.DataFrame) -> DetectionResult:
    named_candidates = [
        col for col in df.columns if str(col).strip().lower() in TARGET_NAME_HINTS
    ]
    if named_candidates:
        return DetectionResult(
            learning_type="Supervised Learning",
            reason=(
                "A target-like column name was found, so the app defaulted "
                "to Supervised Learning."
            ),
            candidate_targets=named_candidates,
        )

    heuristic_candidates: List[str] = []
    for col in df.columns:
        series = df[col]
        nunique = series.nunique(dropna=True)
        if nunique <= 1:
            continue
        if series.dtype == "object" and nunique <= max(20, int(len(df) * 0.2)):
            heuristic_candidates.append(col)
        if pd.api.types.is_numeric_dtype(series) and nunique <= 15:
            heuristic_candidates.append(col)

    heuristic_candidates = list(dict.fromkeys(heuristic_candidates))
    if heuristic_candidates:
        return DetectionResult(
            learning_type="Supervised Learning",
            reason=(
                "No explicit target name was found, but one or more low-cardinality "
                "columns suggest a supervised task."
            ),
            candidate_targets=heuristic_candidates,
        )

    return DetectionResult(
        learning_type="Unsupervised Learning",
        reason=(
            "No reliable target signal was found, so the app defaulted to "
            "Unsupervised Learning."
        ),
        candidate_targets=[],
    )


def get_problem_type(series: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(series):
        unique_vals = series.dropna().nunique()
        if unique_vals <= 12:
            return "Classification"
        return "Regression"
    return "Classification"


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    cleaned.columns = [str(col).strip() for col in cleaned.columns]
    return cleaned


def impute_dataframe(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    if strategy == "keep":
        return df
    if strategy == "drop_rows":
        return df.dropna(axis=0)

    imputed = df.copy()
    numeric_cols = imputed.select_dtypes(include=["number"]).columns
    categorical_cols = imputed.select_dtypes(exclude=["number"]).columns

    if strategy in {"median_mode", "mean_mode"}:
        for col in numeric_cols:
            fill_value = imputed[col].median() if strategy == "median_mode" else imputed[col].mean()
            imputed[col] = imputed[col].fillna(fill_value)

    for col in categorical_cols:
        mode_values = imputed[col].mode(dropna=True)
        fill_value = mode_values.iloc[0] if not mode_values.empty else "Unknown"
        imputed[col] = imputed[col].fillna(fill_value)

    return imputed


def encode_features(X: pd.DataFrame) -> pd.DataFrame:
    X_encoded = X.copy()
    object_cols = X_encoded.select_dtypes(include=["object", "category"]).columns
    for col in object_cols:
        X_encoded[col] = X_encoded[col].fillna("Unknown").astype(str)

    numeric_cols = X_encoded.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        X_encoded[col] = X_encoded[col].fillna(X_encoded[col].median())

    X_encoded = pd.get_dummies(X_encoded, columns=object_cols, drop_first=True)
    return X_encoded


@st.cache_data(show_spinner=False)
def load_csv_from_bytes(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))


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


def render_dataset_overview(df: pd.DataFrame, preview_rows: int) -> None:
    st.subheader("Dataset Overview")
    st.write(df.head(preview_rows))
    st.write("Shape:", df.shape)
    st.write("Duplicate rows:", int(df.duplicated().sum()))
    st.write("Data types:")
    st.write(df.dtypes.astype(str))
    st.write("Missing values:")
    st.write(df.isna().sum())


def render_dataset_profile(df: pd.DataFrame) -> None:
    st.subheader("Data Quality Profile")
    numeric_cols = df.select_dtypes(include=["number"]).shape[1]
    categorical_cols = df.select_dtypes(exclude=["number"]).shape[1]
    duplicate_rows = int(df.duplicated().sum())
    missing_total = int(df.isna().sum().sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Numeric Columns", numeric_cols)
    c2.metric("Categorical Columns", categorical_cols)
    c3.metric("Duplicate Rows", duplicate_rows)
    c4.metric("Total Missing", missing_total)


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

    if len(numeric_cols) >= 2:
        threshold = st.slider("High correlation threshold", 0.5, 0.99, 0.85, 0.01)
        corr_matrix = df[numeric_cols].corr().abs()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        upper = corr_matrix.where(mask)
        pairs = (
            upper.stack()
            .reset_index()
            .rename(columns={"level_0": "Feature A", "level_1": "Feature B", 0: "Abs Correlation"})
        )
        strong_pairs = pairs[pairs["Abs Correlation"] >= threshold].sort_values(
            by="Abs Correlation", ascending=False
        )
        st.write("Highly correlated feature pairs:")
        st.dataframe(strong_pairs.head(20), use_container_width=True)

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
            std = series.std()
            if std == 0:
                outliers = df.iloc[0:0]
            else:
                z_scores = (series - series.mean()) / std
                outliers = df.loc[z_scores.abs() > 3]
        fig, ax = plt.subplots()
        sns.boxplot(y=df[col], ax=ax)
        st.pyplot(fig)
        st.write(f"Outliers detected: {len(outliers)}")
    else:
        st.info("Outlier detection requires numeric columns.")


def train_supervised(df: pd.DataFrame, target: str, random_state: int) -> None:
    st.subheader("Supervised Learning")
    problem_type = get_problem_type(df[target])
    st.write(f"Detected problem type: **{problem_type}**")

    if target not in df.columns:
        st.error("Selected target column was not found.")
        return

    X = df.drop(columns=[target])
    y = df[target]
    X_encoded = encode_features(X)

    if X_encoded.empty:
        st.error("No usable feature columns were found after encoding.")
        return

    if y.nunique(dropna=True) <= 1:
        st.error("Target column has only one class/value. Choose a different target.")
        return

    if problem_type == "Classification":
        class_dist = y.value_counts(normalize=True).rename("ratio").reset_index()
        class_dist.columns = ["class", "ratio"]
        st.write("Class distribution:")
        st.dataframe(class_dist, use_container_width=True)
        if not class_dist.empty and class_dist["ratio"].max() > 0.8:
            st.warning("Class imbalance detected (majority class > 80%). Consider resampling.")

    test_size = st.slider("Train-Test Split (Test Size)", 0.1, 0.5, 0.2, 0.05)
    run_cv = st.checkbox("Run 5-fold cross-validation on selected model", value=False)
    use_scaling = st.checkbox(
        "Standardize numeric features (recommended for Logistic Regression / K-NN)",
        value=True,
    )

    X_final = X_encoded.copy()
    if use_scaling:
        scaler = StandardScaler()
        X_final = pd.DataFrame(
            scaler.fit_transform(X_encoded),
            columns=X_encoded.columns,
            index=X_encoded.index,
        )

    stratify = y if problem_type == "Classification" and y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X_final,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    if problem_type == "Classification":
        model_options = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(random_state=random_state),
            "K-Nearest Neighbors": KNeighborsClassifier(),
            "Random Forest": RandomForestClassifier(random_state=random_state),
        }
    else:
        model_options = {
            "Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(random_state=random_state),
            "Random Forest": RandomForestRegressor(random_state=random_state),
        }

    model_name = st.selectbox("Choose a model", list(model_options.keys()))
    model = model_options[model_name]

    if run_cv:
        cv_scoring = "accuracy" if problem_type == "Classification" else "r2"
        cv_scores = cross_val_score(model, X_final, y, cv=5, scoring=cv_scoring)
        st.write(f"5-fold CV ({cv_scoring}) mean:", float(np.mean(cv_scores)))
        st.write(f"5-fold CV ({cv_scoring}) std:", float(np.std(cv_scores)))

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    st.markdown("### Baseline Benchmark")
    if problem_type == "Classification":
        baseline = DummyClassifier(strategy="most_frequent")
        baseline.fit(X_train, y_train)
        baseline_preds = baseline.predict(X_test)
        st.write("Baseline accuracy (most frequent class):", accuracy_score(y_test, baseline_preds))
    else:
        baseline = DummyRegressor(strategy="mean")
        baseline.fit(X_train, y_train)
        baseline_preds = baseline.predict(X_test)
        st.write("Baseline MAE (predict mean):", mean_absolute_error(y_test, baseline_preds))

    st.write("Model Results")
    if problem_type == "Classification":
        preds = predictions
        st.write("Sample predictions:")
        st.write(preds[:5])
        if hasattr(model, "predict_proba"):
            st.write("Sample prediction probabilities:")
            st.write(model.predict_proba(X_test)[:5])
        st.write("Accuracy:", accuracy_score(y_test, preds))
        st.write("Precision (weighted):", precision_score(y_test, preds, average="weighted", zero_division=0))
        st.write("Recall (weighted):", recall_score(y_test, preds, average="weighted", zero_division=0))
        st.write("F1 (weighted):", f1_score(y_test, preds, average="weighted", zero_division=0))
        st.write("Confusion Matrix:")
        cm = confusion_matrix(y_test, preds)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
        st.text("Classification Report:")
        st.text(classification_report(y_test, preds))

        if y_test.nunique() == 2 and hasattr(model, "predict_proba"):
            positive_class = sorted(pd.Series(y_test).dropna().unique())[-1]
            proba = model.predict_proba(X_test)
            class_list = list(model.classes_)
            if positive_class in class_list:
                pos_idx = class_list.index(positive_class)
                y_binary = (y_test == positive_class).astype(int)
                y_score = proba[:, pos_idx]
                fpr, tpr, _ = roc_curve(y_binary, y_score)
                auc_score = roc_auc_score(y_binary, y_score)
                fig, ax = plt.subplots()
                ax.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}")
                ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.set_title("ROC Curve")
                ax.legend(loc="lower right")
                st.pyplot(fig)
    else:
        st.write("Sample predictions:")
        st.write(predictions[:5])
        st.write("MAE:", mean_absolute_error(y_test, predictions))
        st.write("MSE:", mean_squared_error(y_test, predictions))
        st.write("R2 Score:", r2_score(y_test, predictions))
        residuals = y_test - predictions
        fig, ax = plt.subplots()
        sns.scatterplot(x=predictions, y=residuals, ax=ax)
        ax.axhline(0, color="red", linestyle="--")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Residual")
        ax.set_title("Residual Plot")
        st.pyplot(fig)

    st.markdown("### Quick Model Comparison")
    rows: List[Dict[str, float]] = []
    for compare_name, compare_model in model_options.items():
        compare_model.fit(X_train, y_train)
        compare_preds = compare_model.predict(X_test)
        if problem_type == "Classification":
            rows.append(
                {
                    "Model": compare_name,
                    "Accuracy": float(accuracy_score(y_test, compare_preds)),
                    "F1 (weighted)": float(
                        f1_score(y_test, compare_preds, average="weighted", zero_division=0)
                    ),
                }
            )
        else:
            rows.append(
                {
                    "Model": compare_name,
                    "MAE": float(mean_absolute_error(y_test, compare_preds)),
                    "R2": float(r2_score(y_test, compare_preds)),
                }
            )

    comparison_df = pd.DataFrame(rows)
    sort_col = "Accuracy" if problem_type == "Classification" else "R2"
    st.dataframe(comparison_df.sort_values(by=sort_col, ascending=False), use_container_width=True)

    if hasattr(model, "feature_importances_"):
        importance_df = pd.DataFrame(
            {"feature": X_train.columns, "importance": model.feature_importances_}
        ).sort_values(by="importance", ascending=False)
        st.markdown("### Feature Importance")
        st.dataframe(importance_df.head(20), use_container_width=True)
    elif hasattr(model, "coef_"):
        coef = model.coef_
        if isinstance(coef, np.ndarray):
            coef_flat = coef[0] if coef.ndim > 1 else coef
            coef_df = pd.DataFrame(
                {"feature": X_train.columns, "coefficient": coef_flat}
            ).sort_values(by="coefficient", key=np.abs, ascending=False)
            st.markdown("### Model Coefficients")
            st.dataframe(coef_df.head(20), use_container_width=True)

    results_df = pd.DataFrame({"actual": y_test, "predicted": predictions}, index=y_test.index)
    st.download_button(
        "Download predictions as CSV",
        data=results_df.to_csv(index=False).encode("utf-8"),
        file_name="datasense_predictions.csv",
        mime="text/csv",
    )

    st.markdown(
        "**Model applicability check:** Logistic Regression, Decision Tree, "
        "Random Forest, and K-NN are available when appropriate. Choose the "
        "one that best fits your dataset and interpretability needs."
    )


def train_unsupervised(df: pd.DataFrame, random_state: int) -> None:
    st.subheader("Unsupervised Learning")
    numeric_df = df.select_dtypes(include=["number"]).copy()
    numeric_df = numeric_df.dropna()
    if numeric_df.empty:
        st.warning("Unsupervised learning needs numeric columns.")
        return

    scaler = StandardScaler()
    scaled = pd.DataFrame(
        scaler.fit_transform(numeric_df),
        columns=numeric_df.columns,
        index=numeric_df.index,
    )

    use_pca = st.checkbox("Apply PCA (Dimensionality Reduction)")
    features = scaled

    if use_pca:
        if features.shape[1] < 2:
            st.warning("PCA requires at least two numeric features.")
            return
        pca_components = st.slider("PCA Components", 2, min(5, features.shape[1]), 2)
        pca = PCA(n_components=pca_components)
        features = pd.DataFrame(pca.fit_transform(features), index=features.index)
        st.write("Explained variance ratio:", pca.explained_variance_ratio_)

    show_elbow = st.checkbox("Show elbow analysis", value=False)
    if show_elbow:
        max_k = min(10, max(2, len(features) - 1))
        inertias = []
        ks = list(range(2, max_k + 1))
        for elbow_k in ks:
            elbow_model = KMeans(n_clusters=elbow_k, random_state=random_state, n_init=10)
            elbow_model.fit(features)
            inertias.append(float(elbow_model.inertia_))
        fig, ax = plt.subplots()
        ax.plot(ks, inertias, marker="o")
        ax.set_xlabel("K")
        ax.set_ylabel("Inertia")
        ax.set_title("Elbow Curve")
        st.pyplot(fig)

    k = st.slider("Number of Clusters (K)", 2, 10, 3)
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    clusters = kmeans.fit_predict(features)
    df_clustered = features.copy()
    df_clustered["cluster"] = clusters

    st.write("Cluster Counts:")
    st.write(df_clustered["cluster"].value_counts())

    centers_df = pd.DataFrame(kmeans.cluster_centers_, columns=features.columns)
    st.write("Cluster centers:")
    st.dataframe(centers_df, use_container_width=True)

    if len(np.unique(clusters)) > 1 and features.shape[0] > len(np.unique(clusters)):
        st.write("Silhouette Score:", float(silhouette_score(features, clusters)))

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

    download_clusters = features.copy()
    download_clusters["cluster"] = clusters
    st.download_button(
        "Download clustered dataset as CSV",
        data=download_clusters.to_csv(index=False).encode("utf-8"),
        file_name="datasense_clusters.csv",
        mime="text/csv",
    )


def main() -> None:
    st.title("DataSense: Machine Learning Analytics")
    st.write(
        "Upload a CSV dataset and explore Supervised or Unsupervised learning "
        "with beginner-friendly, powerful analytics."
    )

    settings = render_app_settings()

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if not uploaded_file:
        st.info("Upload a dataset to get started.")
        return

    try:
        df = load_csv_from_bytes(uploaded_file.getvalue())
    except Exception as exc:
        st.error(f"Could not read CSV file: {exc}")
        return

    df = clean_dataframe(df)
    df = impute_dataframe(df, settings.missing_strategy)
    if settings.drop_duplicates:
        before_rows = len(df)
        df = df.drop_duplicates()
        removed = before_rows - len(df)
        if removed > 0:
            st.info(f"Removed {removed} duplicate rows.")

    if settings.max_rows > 0 and len(df) > settings.max_rows:
        df = df.sample(n=settings.max_rows, random_state=settings.random_state)
        st.info(f"Sampled dataset down to {settings.max_rows} rows for faster analysis.")

    if df.empty:
        st.error("The uploaded file is empty after cleaning. Please upload a valid CSV.")
        return

    detection = detect_learning_type(df)
    show_banner(detection.learning_type, detection.reason)

    render_dataset_overview(df, settings.preview_rows)
    render_dataset_profile(df)

    if detection.learning_type == "Supervised Learning":
        default_target = (
            detection.candidate_targets[0] if detection.candidate_targets else None
        )
        target = st.selectbox(
            "Select target column",
            df.columns,
            index=(df.columns.get_loc(default_target) if default_target else 0),
        )
        train_supervised(df, target, settings.random_state)
    else:
        target_choice = st.checkbox("I actually have a target column")
        if target_choice:
            target = st.selectbox("Select target column", df.columns)
            train_supervised(df, target, settings.random_state)
        else:
            train_unsupervised(df, settings.random_state)

    render_visualizations(df)

    st.markdown(
        "---\n"
        "Need more advanced AI advice? Try experimenting with feature selection, "
        "model comparison, and hyperparameter tuning to maximize performance."
    )


if __name__ == "__main__":
    main()
