from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).parent
CSV_FILE = BASE_DIR / "titanic.csv"
FIGURES_DIR = BASE_DIR / "figures"
MODELS_DIR = BASE_DIR / "models"

FIGURES_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)


def main():

    print("=" * 70)
    print("TITANIC PREDICTIVE MODELING")
    print("=" * 70)

    # ---------------------------------------------------------------
    # 1. LOAD OFFLINE DATASET
    # ---------------------------------------------------------------

    df = pd.read_csv(CSV_FILE)

    print("\nDataset loaded from:", CSV_FILE)
    print("Shape:", df.shape)

    # ---------------------------------------------------------------
    # 2. SELECT FEATURES
    # ---------------------------------------------------------------

    target = "survived"

    # We deliberately exclude:
    # - alive: directly represents survival and causes leakage
    # - deck: too many missing values
    # - class: duplicate representation of pclass
    # - embark_town: duplicate representation of embarked
    # - who/adult_male: derived from sex/age and potentially redundant
    #
    # This keeps the model focused on useful passenger information.

    features = [
        "pclass",
        "sex",
        "age",
        "sibsp",
        "parch",
        "fare",
        "embarked",
        "alone",
    ]

    X = df[features]
    y = df[target]

    print("\nFeatures:")
    print(features)

    print("\nTarget:")
    print(target)

    print("\nClass distribution:")
    print(y.value_counts())
    print("\nClass percentages:")
    print(y.value_counts(normalize=True).mul(100))

    # ---------------------------------------------------------------
    # 3. STRATIFIED TRAIN / TEST SPLIT
    # ---------------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print("\n" + "=" * 70)
    print("TRAIN / TEST SPLIT")
    print("=" * 70)

    print("Training rows:", len(X_train))
    print("Testing rows:", len(X_test))

    print("\nTraining target distribution:")
    print(y_train.value_counts(normalize=True))

    print("\nTesting target distribution:")
    print(y_test.value_counts(normalize=True))

    # ---------------------------------------------------------------
    # 4. PREPROCESSING
    # ---------------------------------------------------------------

    numeric_features = [
        "pclass",
        "age",
        "sibsp",
        "parch",
        "fare",
    ]

    categorical_features = [
        "sex",
        "embarked",
        "alone",
    ]

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop="first",
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ]
    )

    # ---------------------------------------------------------------
    # 5. LOGISTIC REGRESSION
    # ---------------------------------------------------------------

    logistic_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    print("\nTraining Logistic Regression...")

    logistic_pipeline.fit(
        X_train,
        y_train,
    )

    logistic_pred = logistic_pipeline.predict(X_test)
    logistic_prob = logistic_pipeline.predict_proba(X_test)[:, 1]

    # ---------------------------------------------------------------
    # 6. RANDOM FOREST
    # ---------------------------------------------------------------

    random_forest_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=42,
                    class_weight="balanced",
                    min_samples_leaf=2,
                ),
            ),
        ]
    )

    print("Training Random Forest...")

    random_forest_pipeline.fit(
        X_train,
        y_train,
    )

    rf_pred = random_forest_pipeline.predict(X_test)
    rf_prob = random_forest_pipeline.predict_proba(X_test)[:, 1]

    # ---------------------------------------------------------------
    # 7. EVALUATION FUNCTION
    # ---------------------------------------------------------------

    def evaluate_model(
        name,
        y_true,
        predictions,
        probabilities,
    ):

        accuracy = accuracy_score(
            y_true,
            predictions,
        )

        precision = precision_score(
            y_true,
            predictions,
        )

        recall = recall_score(
            y_true,
            predictions,
        )

        f1 = f1_score(
            y_true,
            predictions,
        )

        roc_auc = roc_auc_score(
            y_true,
            probabilities,
        )

        print("\n" + "=" * 70)
        print(name)
        print("=" * 70)

        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1-score : {f1:.4f}")
        print(f"ROC-AUC  : {roc_auc:.4f}")

        print("\nClassification report:")
        print(
            classification_report(
                y_true,
                predictions,
            )
        )

        return {
            "model": name,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
        }

    logistic_results = evaluate_model(
        "LOGISTIC REGRESSION",
        y_test,
        logistic_pred,
        logistic_prob,
    )

    rf_results = evaluate_model(
        "RANDOM FOREST",
        y_test,
        rf_pred,
        rf_prob,
    )

    # ---------------------------------------------------------------
    # 8. CONFUSION MATRICES
    # ---------------------------------------------------------------

    print("\nCreating confusion matrices...")

    for name, predictions, filename in [
        (
            "Logistic Regression",
            logistic_pred,
            "confusion_matrix_logistic.png",
        ),
        (
            "Random Forest",
            rf_pred,
            "confusion_matrix_random_forest.png",
        ),
    ]:

        cm = confusion_matrix(
            y_test,
            predictions,
        )

        print(f"\n{name} confusion matrix:")
        print(cm)

        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Did not survive", "Survived"],
        )

        display.plot()

        plt.title(
            f"{name} Confusion Matrix"
        )

        plt.tight_layout()

        plt.savefig(
            FIGURES_DIR / filename,
            dpi=150,
        )

        plt.close()

    # ---------------------------------------------------------------
    # 9. ROC CURVES
    # ---------------------------------------------------------------

    print("\nCreating ROC curves...")

    logistic_fpr, logistic_tpr, _ = roc_curve(
        y_test,
        logistic_prob,
    )

    rf_fpr, rf_tpr, _ = roc_curve(
        y_test,
        rf_prob,
    )

    plt.figure(figsize=(8, 6))

    plt.plot(
        logistic_fpr,
        logistic_tpr,
        label=f"Logistic Regression AUC={logistic_results['roc_auc']:.3f}",
    )

    plt.plot(
        rf_fpr,
        rf_tpr,
        label=f"Random Forest AUC={rf_results['roc_auc']:.3f}",
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random classifier",
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "roc_curve_comparison.png",
        dpi=150,
    )

    plt.close()

    # ---------------------------------------------------------------
    # 10. RANDOM FOREST FEATURE IMPORTANCE
    # ---------------------------------------------------------------

    print("\nRandom Forest feature importance:")

    rf_preprocessor = (
        random_forest_pipeline
        .named_steps["preprocessor"]
    )

    rf_classifier = (
        random_forest_pipeline
        .named_steps["classifier"]
    )

    feature_names = (
        rf_preprocessor
        .get_feature_names_out()
    )

    importances = rf_classifier.feature_importances_

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    ).sort_values(
        "importance",
        ascending=False,
    )

    print(
        importance_df.head(15).to_string(
            index=False
        )
    )

    plt.figure(figsize=(10, 7))

    top_features = importance_df.head(10)

    plt.barh(
        top_features["feature"][::-1],
        top_features["importance"][::-1],
    )

    plt.xlabel("Importance")
    plt.title(
        "Random Forest Top 10 Feature Importances"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "random_forest_feature_importance.png",
        dpi=150,
    )

    plt.close()

    # ---------------------------------------------------------------
    # 11. MODEL COMPARISON
    # ---------------------------------------------------------------

    results = pd.DataFrame(
        [
            logistic_results,
            rf_results,
        ]
    )

    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    print(
        results.to_string(
            index=False
        )
    )

    results.to_csv(
        BASE_DIR / "model_comparison.csv",
        index=False,
    )

    # ---------------------------------------------------------------
    # 12. SAVE MODELS
    # ---------------------------------------------------------------

    joblib.dump(
        logistic_pipeline,
        MODELS_DIR / "logistic_regression.joblib",
    )

    joblib.dump(
        random_forest_pipeline,
        MODELS_DIR / "random_forest.joblib",
    )

    print("\nSaved models:")
    print(
        MODELS_DIR / "logistic_regression.joblib"
    )
    print(
        MODELS_DIR / "random_forest.joblib"
    )

    # ---------------------------------------------------------------
    # 13. MODEL SELECTION
    # ---------------------------------------------------------------

    best_model = results.sort_values(
        "roc_auc",
        ascending=False,
    ).iloc[0]

    print("\n" + "=" * 70)
    print("MODEL SELECTION")
    print("=" * 70)

    print(
        f"Best model by ROC-AUC: "
        f"{best_model['model']}"
    )

    print(
        f"ROC-AUC: "
        f"{best_model['roc_auc']:.4f}"
    )

    print(
        "\nBoth models use class_weight='balanced' "
        "to address class imbalance."
    )

    print(
        "\nPredictive modeling pipeline completed successfully."
    )


if __name__ == "__main__":
    main()
