import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path


BASE_DIR = Path(__file__).parent
CSV_FILE = BASE_DIR / "titanic.csv"
FIGURES_DIR = BASE_DIR / "figures"

FIGURES_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")


def main():

    print("=" * 70)
    print("TITANIC ANALYTICS — MODULE 2")
    print("=" * 70)

    # Required: load Seaborn Titanic dataset exactly once.
    df = sns.load_dataset("titanic")

    # Save offline copy.
    df.to_csv(CSV_FILE, index=False)

    print("\nOffline dataset saved to:", CSV_FILE)

    # ---------------------------------------------------------------
    # 1. BASIC DATASET INSPECTION
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)
    print("DATASET SHAPE")
    print("=" * 70)
    print(df.shape)

    print("\n" + "=" * 70)
    print("DATASET INFO")
    print("=" * 70)
    df.info()

    print("\n" + "=" * 70)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 70)
    print(df.describe(include="all"))

    # ---------------------------------------------------------------
    # 2. MISSING VALUES
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)
    print("MISSING VALUES")
    print("=" * 70)

    missing = pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_percent": df.isna().mean() * 100
    })

    missing = missing[missing["missing_count"] > 0]

    print(missing.sort_values("missing_percent", ascending=False))

    # Threshold-based handling.
    cleaned = df.copy()

    print("\nMissing-value handling:")

    for column in df.columns:

        percentage = df[column].isna().mean() * 100

        if percentage == 0:
            continue

        if percentage < 5:

            cleaned = cleaned.dropna(subset=[column])

            print(
                f"{column}: {percentage:.2f}% -> "
                "dropped rows"
            )

        elif percentage <= 30:

            if pd.api.types.is_numeric_dtype(cleaned[column]):

                value = cleaned[column].median()

                cleaned[column] = cleaned[column].fillna(value)

                print(
                    f"{column}: {percentage:.2f}% -> "
                    f"median imputation ({value:.2f})"
                )

            else:

                value = cleaned[column].mode()[0]

                cleaned[column] = cleaned[column].fillna(value)

                print(
                    f"{column}: {percentage:.2f}% -> "
                    f"mode imputation ({value})"
                )

        else:

            if cleaned[column].dtype == "object":

                cleaned[column] = cleaned[column].fillna("Missing")

                print(
                    f"{column}: {percentage:.2f}% -> "
                    "filled with 'Missing'"
                )

            else:

                cleaned = cleaned.drop(columns=[column])

                print(
                    f"{column}: {percentage:.2f}% -> "
                    "column dropped"
                )

    print("\nShape after missing-value handling:")
    print(cleaned.shape)

    # ---------------------------------------------------------------
    # 3. UNIVARIATE ANALYSIS
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)
    print("UNIVARIATE ANALYSIS")
    print("=" * 70)

    for column in ["age", "fare"]:

        q1 = cleaned[column].quantile(0.25)
        q3 = cleaned[column].quantile(0.75)

        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outlier_count = (
            (cleaned[column] < lower) |
            (cleaned[column] > upper)
        ).sum()

        print(
            f"\n{column.upper()}"
        )

        print(f"Q1: {q1:.2f}")
        print(f"Q3: {q3:.2f}")
        print(f"IQR: {iqr:.2f}")
        print(f"Lower bound: {lower:.2f}")
        print(f"Upper bound: {upper:.2f}")
        print(f"Outlier count: {outlier_count}")

        # Histogram.
        plt.figure(figsize=(8, 5))
        sns.histplot(cleaned[column], kde=True)

        plt.title(f"{column.title()} Distribution")
        plt.xlabel(column.title())
        plt.ylabel("Frequency")

        plt.tight_layout()

        plt.savefig(
            FIGURES_DIR / f"{column}_histogram.png",
            dpi=150
        )

        plt.close()

        # Boxplot.
        plt.figure(figsize=(8, 5))
        sns.boxplot(x=cleaned[column])

        plt.title(f"{column.title()} Boxplot")
        plt.xlabel(column.title())

        plt.tight_layout()

        plt.savefig(
            FIGURES_DIR / f"{column}_boxplot.png",
            dpi=150
        )

        plt.close()

    # Fare statistics.

    fare_mean = cleaned["fare"].mean()
    fare_median = cleaned["fare"].median()
    fare_mode = cleaned["fare"].mode()[0]

    print("\nFare statistics:")
    print(f"Mean: {fare_mean:.2f}")
    print(f"Median: {fare_median:.2f}")
    print(f"Mode: {fare_mode:.2f}")

    if fare_mean > fare_median:
        print(
            "Interpretation: Fare is positively/right skewed."
        )

    elif fare_mean < fare_median:
        print(
            "Interpretation: Fare is negatively/left skewed."
        )

    else:
        print(
            "Interpretation: Mean and median are approximately equal."
        )

    # ---------------------------------------------------------------
    # 4. BIVARIATE ANALYSIS
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)
    print("BIVARIATE ANALYSIS")
    print("=" * 70)

    survival_sex = (
        cleaned.groupby("sex")["survived"]
        .mean()
        .mul(100)
    )

    print("\nSurvival by sex (%):")
    print(survival_sex)

    survival_class = (
        cleaned.groupby("pclass")["survived"]
        .mean()
        .mul(100)
    )

    print("\nSurvival by passenger class (%):")
    print(survival_class)

    survival_sex_class = (
        cleaned.groupby(["sex", "pclass"])["survived"]
        .mean()
        .mul(100)
    )

    print("\nSurvival by sex and passenger class (%):")
    print(survival_sex_class)

    # ---------------------------------------------------------------
    # 5. BOOLEAN MASK
    # ---------------------------------------------------------------

    female_mask = cleaned["sex"] == "female"
    first_class_mask = cleaned["pclass"] == 1

    female_first_class = cleaned[
        female_mask & first_class_mask
    ]

    print("\nBoolean-mask example:")
    print(
        "Female first-class passengers:",
        len(female_first_class)
    )

    # ---------------------------------------------------------------
    # 6. CORRELATION MATRIX
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)
    print("CORRELATION MATRIX")
    print("=" * 70)

    correlation_columns = [
        "survived",
        "pclass",
        "age",
        "sibsp",
        "parch",
        "fare"
    ]

    correlation = cleaned[
        correlation_columns
    ].corr()

    print(correlation)

    plt.figure(figsize=(9, 7))

    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        square=True,
        cmap="coolwarm"
    )

    plt.title(
        "Titanic Numeric Feature Correlation"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "correlation_heatmap.png",
        dpi=150
    )

    plt.close()

    # Strongest two correlations.

    pairs = []

    for i in range(len(correlation_columns)):

        for j in range(i + 1, len(correlation_columns)):

            a = correlation_columns[i]
            b = correlation_columns[j]

            value = correlation.loc[a, b]

            pairs.append(
                (a, b, value, abs(value))
            )

    pairs.sort(
        key=lambda x: x[3],
        reverse=True
    )

    print("\nTwo strongest correlations:")

    for a, b, value, absolute_value in pairs[:2]:

        print(
            f"{a} vs {b}: "
            f"{value:.4f}"
        )

    # ---------------------------------------------------------------
    # 7. MULTIVARIATE VISUALIZATIONS
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)
    print("CREATING VISUALIZATIONS")
    print("=" * 70)

    # Chart 1: survival by sex.

    plt.figure(figsize=(8, 5))

    sns.barplot(
        data=cleaned,
        x="sex",
        y="survived",
        hue="sex",
        legend=False
    )

    plt.title("Survival Rate by Sex")
    plt.ylabel("Survival Rate")

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "survival_by_sex.png",
        dpi=150
    )

    plt.close()

    # Chart 2: survival by class.

    plt.figure(figsize=(8, 5))

    sns.barplot(
        data=cleaned,
        x="pclass",
        y="survived",
        hue="pclass",
        legend=False
    )

    plt.title("Survival Rate by Passenger Class")
    plt.ylabel("Survival Rate")

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "survival_by_class.png",
        dpi=150
    )

    plt.close()

    # Chart 3: sex + class.

    plt.figure(figsize=(9, 6))

    sns.barplot(
        data=cleaned,
        x="pclass",
        y="survived",
        hue="sex"
    )

    plt.title(
        "Survival Rate by Passenger Class and Sex"
    )

    plt.ylabel("Survival Rate")

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "survival_by_class_and_sex.png",
        dpi=150
    )

    plt.close()

    # Chart 4: fare vs survival.

    plt.figure(figsize=(8, 6))

    sns.boxplot(
        data=cleaned,
        x="survived",
        y="fare",
        hue="survived",
        legend=False
    )

    plt.title("Fare Distribution by Survival")

    plt.xlabel("Survived")

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "fare_by_survival.png",
        dpi=150
    )

    plt.close()

    # Chart 5: age distribution by survival.

    plt.figure(figsize=(9, 6))

    sns.histplot(
        data=cleaned,
        x="age",
        hue="survived",
        kde=True,
        element="step"
    )

    plt.title(
        "Age Distribution by Survival"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "age_by_survival.png",
        dpi=150
    )

    plt.close()

    # ---------------------------------------------------------------
    # 8. ADDITIONAL MULTIVARIATE VISUALIZATIONS
    # ---------------------------------------------------------------

    # Multivariate Chart 1: Age vs Fare, with survival and sex.
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=cleaned,
        x="age",
        y="fare",
        hue="survived",
        style="sex",
        alpha=0.75
    )
    plt.title("Age vs Fare by Survival and Sex")
    plt.xlabel("Age")
    plt.ylabel("Fare")
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "age_fare_survival_sex.png",
        dpi=150
    )
    plt.close()

    print("\nMultivariate Chart 1 Interpretation:")
    print(
        "The scatter plot examines age and fare while distinguishing passengers "
        "by survival outcome and sex. Higher fares are spread across a wide age "
        "range, while survival patterns differ between the two survival groups."
    )

    # Multivariate Chart 2: Age by passenger class and survival.
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=cleaned,
        x="pclass",
        y="age",
        hue="survived"
    )
    plt.title("Age Distribution by Passenger Class and Survival")
    plt.xlabel("Passenger Class")
    plt.ylabel("Age")
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "age_by_class_and_survival.png",
        dpi=150
    )
    plt.close()

    print("\nMultivariate Chart 2 Interpretation:")
    print(
        "This chart compares age distributions across passenger classes while "
        "also separating survivors from non-survivors. Passenger classes show "
        "different age distributions, and survival status provides an additional "
        "dimension for comparing those distributions."
    )

    # Multivariate Chart 3: Fare by passenger class and survival.
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=cleaned,
        x="pclass",
        y="fare",
        hue="survived"
    )
    plt.title("Fare Distribution by Passenger Class and Survival")
    plt.xlabel("Passenger Class")
    plt.ylabel("Fare")
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "fare_by_class_and_survival.png",
        dpi=150
    )
    plt.close()

    print("\nMultivariate Chart 3 Interpretation:")
    print(
        "Fare distributions vary substantially across passenger classes, with "
        "higher-class passengers generally having higher fares. Separating the "
        "groups by survival allows the relationship between fare level, class, "
        "and survival to be examined together."
    )

    # Multivariate Chart 4: Survival rate by class, sex, and embarkation port.
    survival_by_three = (
        cleaned.groupby(
            ["pclass", "sex", "embarked"],
            observed=True
        )["survived"]
        .mean()
        .reset_index()
    )

    survival_by_three["sex_embarked"] = (
        survival_by_three["sex"]
        + " / "
        + survival_by_three["embarked"]
    )

    plt.figure(figsize=(12, 7))
    sns.barplot(
        data=survival_by_three,
        x="pclass",
        y="survived",
        hue="sex_embarked",
        errorbar=None
    )
    plt.title("Survival Rate by Class, Sex, and Embarkation Port")
    plt.xlabel("Passenger Class")
    plt.ylabel("Survival Rate")
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "survival_class_sex_embarked.png",
        dpi=150
    )
    plt.close()

    print("\nMultivariate Chart 4 Interpretation:")
    print(
        "This grouped analysis combines passenger class, sex, and embarkation "
        "port while examining survival rate. The chart highlights how survival "
        "rates vary across class and sex groups, while the underlying grouping "
        "also accounts for embarkation port."
    )

    # ---------------------------------------------------------------
    # 9. STANDARDIZATION SANITY CHECK
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)
    print("STANDARDIZATION SANITY CHECK")
    print("=" * 70)

    for column in ["age", "fare"]:

        mean = cleaned[column].mean()
        std = cleaned[column].std()

        standardized = (
            cleaned[column] - mean
        ) / std

        print(f"\n{column}:")

        print(
            f"Before -> "
            f"mean={mean:.6f}, "
            f"std={std:.6f}"
        )

        print(
            f"After  -> "
            f"mean={standardized.mean():.6f}, "
            f"std={standardized.std():.6f}"
        )

    # ---------------------------------------------------------------
    # 9. INTERPRETATIONS
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)
    print("KEY DATA STORY")
    print("=" * 70)

    print(
        "1. Female passengers had a substantially higher "
        "survival rate than male passengers."
    )

    print(
        "2. First-class passengers had the highest survival "
        "rate, followed by second and third class."
    )

    print(
        "3. Fare is right-skewed, with expensive tickets "
        "creating high-value outliers."
    )

    print(
        "4. Passenger class and fare show a meaningful "
        "relationship, while survival is negatively related "
        "to passenger class coding."
    )

    print(
        "5. The sex and passenger-class interaction provides "
        "a stronger survival story than either variable alone."
    )

    print("\nEDA pipeline completed successfully.")
    print("Figures saved in:", FIGURES_DIR)


if __name__ == "__main__":
    main()
