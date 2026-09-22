from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from imblearn.over_sampling import SMOTE


BASE_DIR = Path(__file__).parent
CSV_FILE = BASE_DIR / "titanic.csv"
FIGURES_DIR = BASE_DIR / "figures"
MODELS_DIR = BASE_DIR / "models"

FIGURES_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)


def make_preprocessor(numeric_features, categorical_features):
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", drop="first")),
    ])

    return ColumnTransformer([
        ("num", numeric_pipe, numeric_features),
        ("cat", categorical_pipe, categorical_features),
    ])


def evaluate_model(name, pipeline, X_test, y_test):
    pred = pipeline.predict(X_test)
    prob = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, prob),
    }

    print(f"\n{name}")
    print("-" * 60)
    for key, value in metrics.items():
        if key != "model":
            print(f"{key}: {value:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, pred))

    return metrics, pred, prob


def main():
    print("=" * 70)
    print("TITANIC PREDICTIVE MODELING")
    print("=" * 70)

    df = pd.read_csv(CSV_FILE)
    print("\nDataset shape:", df.shape)

    target = "survived"

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

    X = df[features].copy()
    y = df[target].copy()

    numeric_features = [
        "pclass",
        "age",
        "sibsp",
        "parch",
        "fare",
        "alone",
    ]

    categorical_features = [
        "sex",
        "embarked",
    ]

    # ------------------------------------------------------------
    # 1. CLASS BALANCE
    # ------------------------------------------------------------
    print("\nCLASS BALANCE")
    print(y.value_counts())
    print(y.value_counts(normalize=True))

    # ------------------------------------------------------------
    # 2. STRATIFIED TRAIN / TEST SPLIT
    # ------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print("\nTrain shape:", X_train.shape)
    print("Test shape:", X_test.shape)
    print("Stratification preserves the survivor/non-survivor class ratio.")

    preprocessor = make_preprocessor(
        numeric_features,
        categorical_features,
    )

    # ------------------------------------------------------------
    # 3. LOGISTIC REGRESSION
    # ------------------------------------------------------------
    logistic_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])

    logistic_pipeline.fit(X_train, y_train)

    logistic_metrics, logistic_pred, logistic_prob = evaluate_model(
        "Logistic Regression",
        logistic_pipeline,
        X_test,
        y_test,
    )

    ConfusionMatrixDisplay.from_predictions(y_test, logistic_pred)
    plt.title("Logistic Regression Confusion Matrix")
    plt.savefig(FIGURES_DIR / "logistic_regression_confusion_matrix.png")
    plt.close()

    joblib.dump(
        logistic_pipeline,
        MODELS_DIR / "logistic_regression.joblib",
    )

    # ------------------------------------------------------------
    # 4. DECISION TREE
    # ------------------------------------------------------------
    tree_pipeline = Pipeline([
        ("preprocessor", make_preprocessor(
            numeric_features,
            categorical_features,
        )),
        ("model", DecisionTreeClassifier(
            random_state=42,
            class_weight="balanced",
            max_depth=5,
        )),
    ])

    tree_pipeline.fit(X_train, y_train)

    tree_metrics, tree_pred, tree_prob = evaluate_model(
        "Decision Tree",
        tree_pipeline,
        X_test,
        y_test,
    )

    ConfusionMatrixDisplay.from_predictions(y_test, tree_pred)
    plt.title("Decision Tree Confusion Matrix")
    plt.savefig(FIGURES_DIR / "decision_tree_confusion_matrix.png")
    plt.close()

    tree_model = tree_pipeline.named_steps["model"]
    tree_preprocessor = tree_pipeline.named_steps["preprocessor"]
    tree_features = tree_preprocessor.get_feature_names_out()

    plt.figure(figsize=(20, 10))
    plot_tree(
        tree_model,
        feature_names=tree_features,
        class_names=["Not Survived", "Survived"],
        filled=True,
        max_depth=3,
        fontsize=7,
    )
    plt.title("Decision Tree")
    plt.savefig(FIGURES_DIR / "decision_tree.png", bbox_inches="tight")
    plt.close()

    # ------------------------------------------------------------
    # 5. RANDOM FOREST
    # ------------------------------------------------------------
    rf_pipeline = Pipeline([
        ("preprocessor", make_preprocessor(
            numeric_features,
            categorical_features,
        )),
        ("model", RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
            min_samples_leaf=2,
            oob_score=True,
        )),
    ])

    rf_pipeline.fit(X_train, y_train)

    rf_metrics, rf_pred, rf_prob = evaluate_model(
        "Random Forest",
        rf_pipeline,
        X_test,
        y_test,
    )

    rf_model = rf_pipeline.named_steps["model"]

    print("\nRandom Forest OOB Score:",
          f"{rf_model.oob_score_:.4f}")

    ConfusionMatrixDisplay.from_predictions(y_test, rf_pred)
    plt.title("Random Forest Confusion Matrix")
    plt.savefig(FIGURES_DIR / "random_forest_confusion_matrix.png")
    plt.close()

    joblib.dump(
        rf_pipeline,
        MODELS_DIR / "random_forest.joblib",
    )

    # ------------------------------------------------------------
    # 6. MODEL COMPARISON
    # ------------------------------------------------------------
    comparison = pd.DataFrame([
        logistic_metrics,
        tree_metrics,
        rf_metrics,
    ])

    print("\nMODEL COMPARISON")
    print(comparison.to_string(index=False))

    comparison.to_csv(
        BASE_DIR / "model_comparison.csv",
        index=False,
    )

    # ------------------------------------------------------------
    # 7. GRID SEARCH RANDOM FOREST
    # ------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RANDOM FOREST GRID SEARCH")
    print("=" * 70)

    grid_pipeline = Pipeline([
        ("preprocessor", make_preprocessor(
            numeric_features,
            categorical_features,
        )),
        ("model", RandomForestClassifier(
            random_state=42,
            oob_score=True,
        )),
    ])

    param_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 5, 10],
        "model__max_features": ["sqrt", "log2"],
    }

    grid = GridSearchCV(
        grid_pipeline,
        param_grid,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1,
    )

    grid.fit(X_train, y_train)

    print("\nBest Parameters:")
    print(grid.best_params_)
    print(f"Best CV ROC-AUC: {grid.best_score_:.4f}")

    tuned_rf = grid.best_estimator_
    tuned_metrics, tuned_pred, tuned_prob = evaluate_model(
        "Tuned Random Forest",
        tuned_rf,
        X_test,
        y_test,
    )

    tuned_rf_model = tuned_rf.named_steps["model"]
    print(f"Tuned RF OOB Score: {tuned_rf_model.oob_score_:.4f}")

    # ------------------------------------------------------------
    # 8. TUNED RF FEATURE IMPORTANCE
    # ------------------------------------------------------------
    tuned_preprocessor = tuned_rf.named_steps["preprocessor"]
    tuned_features = tuned_preprocessor.get_feature_names_out()
    importances = tuned_rf_model.feature_importances_

    feature_importance = pd.DataFrame({
        "feature": tuned_features,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    print("\nTop Random Forest Features:")
    print(feature_importance.head(10).to_string(index=False))

    feature_importance.to_csv(
        BASE_DIR / "random_forest_feature_importance.csv",
        index=False,
    )

    plt.figure(figsize=(10, 6))
    top = feature_importance.head(10).sort_values("importance")
    plt.barh(top["feature"], top["importance"])
    plt.title("Top Random Forest Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "random_forest_feature_importance.png")
    plt.close()

    # ------------------------------------------------------------
    # 9. ROC CURVES
    # ------------------------------------------------------------
    plt.figure(figsize=(8, 6))

    for name, probability in [
        ("Logistic Regression", logistic_prob),
        ("Decision Tree", tree_prob),
        ("Random Forest", rf_prob),
        ("Tuned Random Forest", tuned_prob),
    ]:
        fpr, tpr, _ = roc_curve(y_test, probability)
        auc = roc_auc_score(y_test, probability)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "roc_curve_comparison.png")
    plt.close()

    # ------------------------------------------------------------
    # 10. CLASS IMBALANCE EXPERIMENT
    # ------------------------------------------------------------
    print("\n" + "=" * 70)
    print("CLASS IMBALANCE EXPERIMENT")
    print("=" * 70)

    imbalance_preprocessor = make_preprocessor(
        numeric_features,
        categorical_features,
    )

    X_train_encoded = imbalance_preprocessor.fit_transform(X_train)
    X_test_encoded = imbalance_preprocessor.transform(X_test)

    # Baseline: no imbalance handling
    baseline = LogisticRegression(max_iter=1000)
    baseline.fit(X_train_encoded, y_train)
    baseline_pred = baseline.predict(X_test_encoded)

    # Balanced class weights
    balanced = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
    )
    balanced.fit(X_train_encoded, y_train)
    balanced_pred = balanced.predict(X_test_encoded)

    # SMOTE only on training data
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(
        X_train_encoded,
        y_train,
    )

    smote_model = LogisticRegression(max_iter=1000)
    smote_model.fit(X_train_smote, y_train_smote)
    smote_pred = smote_model.predict(X_test_encoded)

    imbalance_results = pd.DataFrame([
        {
            "method": "Baseline",
            "precision": precision_score(
                y_test, baseline_pred, zero_division=0
            ),
            "recall": recall_score(
                y_test, baseline_pred, zero_division=0
            ),
            "f1": f1_score(
                y_test, baseline_pred, zero_division=0
            ),
        },
        {
            "method": "Class Weight Balanced",
            "precision": precision_score(
                y_test, balanced_pred, zero_division=0
            ),
            "recall": recall_score(
                y_test, balanced_pred, zero_division=0
            ),
            "f1": f1_score(
                y_test, balanced_pred, zero_division=0
            ),
        },
        {
            "method": "SMOTE Training Fold",
            "precision": precision_score(
                y_test, smote_pred, zero_division=0
            ),
            "recall": recall_score(
                y_test, smote_pred, zero_division=0
            ),
            "f1": f1_score(
                y_test, smote_pred, zero_division=0
            ),
        },
    ])

    print(imbalance_results.to_string(index=False))

    imbalance_results.to_csv(
        BASE_DIR / "class_imbalance_comparison.csv",
        index=False,
    )

    print(
        "\nSMOTE was applied only to the training fold; "
        "the test set remained untouched."
    )

    # ------------------------------------------------------------
    # 11. SELECT AND SAVE BEST COMPLETE PIPELINE
    # ------------------------------------------------------------
    all_models = [
        (logistic_metrics, logistic_pipeline),
        (tree_metrics, tree_pipeline),
        (rf_metrics, rf_pipeline),
        (tuned_metrics, tuned_rf),
    ]

    best_metrics, best_pipeline = max(
        all_models,
        key=lambda item: item[0]["roc_auc"],
    )

    print("\n" + "=" * 70)
    print("BEST CLASSIFIER")
    print("=" * 70)
    print("Model:", best_metrics["model"])
    print(f"Accuracy:  {best_metrics['accuracy']:.4f}")
    print(f"Precision: {best_metrics['precision']:.4f}")
    print(f"Recall:    {best_metrics['recall']:.4f}")
    print(f"F1:        {best_metrics['f1']:.4f}")
    print(f"ROC-AUC:   {best_metrics['roc_auc']:.4f}")

    best_path = MODELS_DIR / "best_pipeline.joblib"
    joblib.dump(best_pipeline, best_path)
    print("\nSaved complete pipeline:", best_path)

    # Reload and test using RAW input.
    loaded_pipeline = joblib.load(best_path)

    # 1. The reloaded pipeline reproduces the in-memory predictions exactly.
    reloaded_matches = (
        loaded_pipeline.predict(X_test) == best_pipeline.predict(X_test)
    ).all()
    print("Reloaded predictions identical on all test rows:",
          bool(reloaded_matches))

    # 2. It accepts brand-new raw passengers: text categories, unscaled
    #    numbers, and a missing age, with no manual preprocessing.
    new_passengers = pd.DataFrame([
        {"pclass": 1, "sex": "female", "age": 29.0, "sibsp": 0,
         "parch": 0, "fare": 100.0, "embarked": "C", "alone": True},
        {"pclass": 3, "sex": "male", "age": None, "sibsp": 0,
         "parch": 0, "fare": 7.25, "embarked": "S", "alone": True},
    ])

    new_predictions = loaded_pipeline.predict(new_passengers)
    new_probabilities = loaded_pipeline.predict_proba(new_passengers)[:, 1]

    print("\nRaw new-passenger predictions from the reloaded pipeline:")
    for (_, passenger), prediction, probability in zip(
        new_passengers.iterrows(),
        new_predictions,
        new_probabilities,
    ):
        print(
            f"  class {passenger['pclass']} {passenger['sex']:<6} "
            f"age={passenger['age']} fare={passenger['fare']:.2f} -> "
            f"survived={int(prediction)} (p={probability:.3f})"
        )

    # ------------------------------------------------------------
    # 12. REGRESSION: PREDICT FARE
    # ------------------------------------------------------------
    print("\n" + "=" * 70)
    print("MULTIVARIATE LINEAR REGRESSION — FARE")
    print("=" * 70)

    regression_target = "fare"

    regression_features = [
        "survived",
        "pclass",
        "age",
        "sibsp",
        "parch",
        "sex",
        "embarked",
        "alone",
    ]

    Xr = df[regression_features].copy()
    yr = df[regression_target].copy()

    regression_numeric = [
        "survived",
        "pclass",
        "age",
        "sibsp",
        "parch",
        "alone",
    ]

    regression_categorical = [
        "sex",
        "embarked",
    ]

    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        Xr,
        yr,
        test_size=0.20,
        random_state=42,
    )

    regression_preprocessor = make_preprocessor(
        regression_numeric,
        regression_categorical,
    )

    regression_pipeline = Pipeline([
        ("preprocessor", regression_preprocessor),
        ("model", LinearRegression()),
    ])

    regression_pipeline.fit(Xr_train, yr_train)

    yr_pred = regression_pipeline.predict(Xr_test)

    mae = mean_absolute_error(yr_test, yr_pred)
    rmse = mean_squared_error(
        yr_test,
        yr_pred,
    ) ** 0.5
    r2 = r2_score(yr_test, yr_pred)

    transformed_features = (
        regression_pipeline
        .named_steps["preprocessor"]
        .get_feature_names_out()
    )

    p = len(transformed_features)
    n = len(yr_test)

    adjusted_r2 = (
        1 - (1 - r2) * (n - 1) / (n - p - 1)
        if n - p - 1 > 0
        else float("nan")
    )

    regression_results = pd.DataFrame([{
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Adjusted_R2": adjusted_r2,
    }])

    print("\nRegression Metrics:")
    print(regression_results.to_string(index=False))

    regression_results.to_csv(
        BASE_DIR / "regression_metrics.csv",
        index=False,
    )

    # Residual plot
    residuals = yr_test - yr_pred

    plt.figure(figsize=(8, 6))
    plt.scatter(yr_pred, residuals)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted Fare")
    plt.ylabel("Residual")
    plt.title("Fare Regression Residual Plot")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fare_regression_residuals.png")
    plt.close()

    # Heteroscedasticity check: does the residual spread change with the
    # predicted value? Use plain arrays so pandas does not align the test
    # set's original index against the 0..n-1 prediction index.
    residual_values = residuals.to_numpy()

    residual_prediction_corr = pd.Series(abs(residual_values)).corr(
        pd.Series(yr_pred)
    )

    print(
        f"\nCorrelation between absolute residuals and predictions: "
        f"{residual_prediction_corr:.4f}"
    )

    spread = (
        pd.DataFrame({
            "prediction_band": pd.qcut(
                yr_pred,
                3,
                labels=["low", "mid", "high"],
            ),
            "residual": residual_values,
        })
        .groupby("prediction_band", observed=True)["residual"]
        .agg(["count", "std"])
    )

    print("\nResidual spread by predicted-fare band:")
    print(spread.to_string())

    spread_ratio = spread.loc["high", "std"] / spread.loc["low", "std"]
    print(f"High/low residual std ratio: {spread_ratio:.2f}")

    if spread_ratio > 2:
        print(
            "Residual spread grows with predicted fare -> "
            "heteroscedasticity is present."
        )
    else:
        print(
            "Residual spread is roughly constant -> "
            "no clear heteroscedasticity."
        )

    # ------------------------------------------------------------
    # 13. FINAL WRITTEN CONCLUSION
    # ------------------------------------------------------------
    print("\n" + "=" * 70)
    print("FINAL MODELING CONCLUSION")
    print("=" * 70)

    print(
        f"The evaluated classifiers achieved ROC-AUC values of "
        f"{logistic_metrics['roc_auc']:.4f} for Logistic Regression, "
        f"{tree_metrics['roc_auc']:.4f} for Decision Tree, "
        f"{rf_metrics['roc_auc']:.4f} for Random Forest, and "
        f"{tuned_metrics['roc_auc']:.4f} for Tuned Random Forest."
    )

    print(
        f"The selected classifier was {best_metrics['model']} with "
        f"accuracy {best_metrics['accuracy']:.4f}, "
        f"precision {best_metrics['precision']:.4f}, "
        f"recall {best_metrics['recall']:.4f}, "
        f"F1 {best_metrics['f1']:.4f}, and "
        f"ROC-AUC {best_metrics['roc_auc']:.4f}."
    )

    print(
        "The imbalance experiment compared baseline Logistic Regression, "
        "class-weight balancing, and SMOTE applied only to training data."
    )

    print(
        f"The fare regression achieved MAE {mae:.4f}, "
        f"RMSE {rmse:.4f}, R2 {r2:.4f}, and "
        f"Adjusted R2 {adjusted_r2:.4f}."
    )

    print("\nAll modeling outputs have been saved under analytics/.")


if __name__ == "__main__":
    main()
