# Module 2 — Titanic Analytics and Machine Learning

A single pipeline over the Titanic dataset: profile it, clean it, tell a visual story about who survived, then build, tune and evaluate predictive models on the same data.

Every number below comes from running the two scripts. Charts are saved in [`figures/`](figures/) as supporting files; the written interpretation for each is in this README.

## Install and run

From the repository root, with the virtual environment active:

```bash
pip install -r analytics/requirements.txt

python analytics/titanic_analysis.py   # Part A: load, profile, clean, EDA
python analytics/modeling.py           # Part B: models, tuning, regression, saved pipeline
```

| File | Role |
|---|---|
| `titanic_analysis.py` | The **only** `sns.load_dataset("titanic")` call. Saves `titanic.csv`, then profiling, cleaning and every EDA chart |
| `titanic.csv` | Committed offline copy of the loaded dataset, saved right after loading |
| `modeling.py` | Reads the same `titanic.csv` (no second network load); all of Part B |
| `models/best_pipeline.joblib` | The complete fitted pipeline (preprocessing + classifier) |
| `figures/` | Every generated chart |

**One load, one dataset.** `titanic_analysis.py` calls `sns.load_dataset("titanic")` once and immediately runs `df.to_csv("titanic.csv", index=False)`. `modeling.py` never calls Seaborn's loader: it runs `pd.read_csv("titanic.csv")` and continues from that file. So the dataset is fetched from the network/cache exactly once. `titanic.csv` holds the loaded data before cleaning, because Task 8 requires the modeling pipeline to do its own imputation, fitted on the training split only (see [Task 8](#task-8--preprocessing-fit-on-train-only)).

---

## Part A — Profiling, cleaning and the data story

### Task 1 — Profile

- `df.shape` → **(891, 15)**
- `df.info()` → 891 rows. Numeric: `survived`, `pclass`, `sibsp`, `parch` (int), `age`, `fare` (float). Text/categorical: `sex`, `embarked`, `class`, `who`, `deck`, `embark_town`, `alive`. Boolean: `adult_male`, `alone`.
- `df.describe()` highlights: mean age 29.70 (min 0.42, max 80); mean fare 32.20 with a max of 512.33; survival rate 38.38%; 577 of 891 passengers are male.

**Missing values** (every column with any missing data):

| Column | Missing | % missing |
|---|---:|---:|
| `deck` | 688 | **77.22%** |
| `age` | 177 | **19.87%** |
| `embarked` | 2 | **0.22%** |
| `embark_town` | 2 | **0.22%** |

All other columns have 0% missing.

### Task 2 — Missing-value handling (threshold rule)

Rule: **under 5% → drop rows; 5%–30% → impute; above 30% → drop the column or encode "missing" as a category.**

| Column | Measured | Rule band | Action |
|---|---:|---|---|
| `embarked` | 0.22% | < 5% | Dropped the 2 rows |
| `embark_town` | 0.22% | < 5% | Dropped rows (the same 2 passengers) |
| `age` | 19.87% | 5%–30% | Median imputation (median = 28.0). The median, not the mean, because age is slightly right-skewed and the median is robust to the older-passenger tail |
| `deck` | 77.22% | > 30% | **Dropped the column** |

**Why `deck` is dropped rather than encoded as "Missing":** only 203 of 891 passengers have a deck recorded, and 175 of those 203 are first-class passengers. A "Missing" category would mostly mean "not first class", which `pclass` already captures without gaps. Imputing 77% of a column from 23% of observations would be guesswork. Dropping it loses almost no independent information.

Cleaned shape: **(889, 14)**.

### Task 3 — Univariate analysis

Charts: `figures/age_histogram.png`, `figures/age_boxplot.png`, `figures/fare_histogram.png`, `figures/fare_boxplot.png`.

**IQR outliers** (outside `[Q1 − 1.5×IQR, Q3 + 1.5×IQR]`, on the cleaned data):

| Column | Q1 | Q3 | IQR | Lower bound | Upper bound | **Outliers** |
|---|---:|---:|---:|---:|---:|---:|
| `age` | 22.00 | 35.00 | 13.00 | 2.50 | 54.50 | **65** |
| `fare` | 7.90 | 31.00 | 23.10 | −26.76 | 65.66 | **114** |

Age has outliers at both ends (infants under 2.5 and passengers over 54.5). The count is inflated slightly because the 177 imputed ages all sit at 28, which narrows the IQR. Every fare outlier is on the high side, since the lower bound is negative.

**Fare central tendency:** mean **32.10**, median **14.45**, mode **8.05**.

**Fare is right-skewed.** The ordering is **mode (8.05) < median (14.45) < mean (32.10)**, the classic right-skew pattern. Most passengers paid low third-class fares (the mode), while a long tail of expensive first-class tickets, up to 512.33, pulls the mean to more than double the median.

### Task 4 — Bivariate analysis (boolean masks)

All rates are computed with boolean masks, e.g. `cleaned[(cleaned["sex"] == "female") & (cleaned["pclass"] == 1)]["survived"].mean()`.

**(a) By sex**

| Sex | Survival rate | n |
|---|---:|---:|
| female | **74.04%** | 312 |
| male | **18.89%** | 577 |

**(b) By passenger class**

| Class | Survival rate | n |
|---|---:|---:|
| 1 | **62.62%** | 214 |
| 2 | **47.28%** | 184 |
| 3 | **24.24%** | 491 |

**(c) By sex AND class** (`sex_mask & class_mask`)

| | Class 1 | Class 2 | Class 3 |
|---|---:|---:|---:|
| female | **96.74%** (n=92) | **92.11%** (n=76) | **50.00%** (n=144) |
| male | **36.89%** (n=122) | **15.74%** (n=108) | **13.54%** (n=347) |

Combined masks with `|`: `female | class 1` → 63.59% (n=434); `male & (class 2 | class 3)` → 14.07% (n=455).

**Correlation matrix.** Computed on exactly these six columns: `survived`, `pclass`, `age`, `sibsp`, `parch`, `fare`. The boolean flags `adult_male` and `alone` are excluded because they are derived from `sex`/`age` and from `sibsp + parch` respectively. Heatmap: `figures/correlation_heatmap.png`.

| | survived | pclass | age | sibsp | parch | fare |
|---|---:|---:|---:|---:|---:|---:|
| **survived** | 1.000 | −0.336 | −0.070 | −0.034 | 0.083 | 0.255 |
| **pclass** | −0.336 | 1.000 | −0.337 | 0.082 | 0.017 | −0.548 |
| **age** | −0.070 | −0.337 | 1.000 | −0.233 | −0.171 | 0.094 |
| **sibsp** | −0.034 | 0.082 | −0.233 | 1.000 | 0.415 | 0.161 |
| **parch** | 0.083 | 0.017 | −0.171 | 0.415 | 1.000 | 0.218 |
| **fare** | 0.255 | −0.548 | 0.094 | 0.161 | 0.218 | 1.000 |

**Two strongest correlations** (the largest `abs(r)` among off-diagonal pairs):

1. **`pclass` vs `fare`: r = −0.548.** A higher class number (a cheaper class) goes with a lower fare. Fare is largely the price of the class, so the two features carry overlapping information. This is why class and fare tell a similar survival story below, and it matters for multicollinearity in the regression model.
2. **`sibsp` vs `parch`: r = +0.415.** Both count family members aboard: passengers travelling with a spouse or siblings also tended to travel with parents or children, because families boarded together. The two are partly redundant family-size measures.

Against the target, the strongest links are `pclass` (−0.336) and `fare` (+0.255). Age alone correlates only weakly with survival (−0.070), and the multivariate charts below show why.

### Task 5 — Multivariate data story

The argument the charts build: **sex was the main driver of survival, class set how much that advantage was worth, and fare and port mostly stand in for class.**

**Chart 1 — Survival rate by class and sex** (`figures/survival_by_class_and_sex.png`)
Women survived far more often than men in every class, but class decided how large that advantage was: 96.7% of first-class women survived against 50.0% of third-class women. A first-class man (36.9%) still did worse than a third-class woman, so sex outweighs class. Class, in turn, separates men: first-class men survived at more than twice the rate of second- or third-class men (15.7% and 13.5%).

**Chart 2 — Fare by class and survival** (`figures/fare_by_class_and_survival.png`)
Within first class, survivors paid a much higher median fare than non-survivors (77.34 vs 44.75). This is consistent with pricier first-class tickets being tied to better cabins and faster access to the boat deck. In third class the medians are almost identical (8.52 vs 8.05), so fare adds little there. Fare's overall link to survival (r = 0.255) is therefore mostly a class effect plus a premium inside first class.

**Chart 3 — Age by class and survival** (`figures/age_by_class_and_survival.png`)
Survivors were slightly younger than non-survivors in every class, with the widest gap in first class (median 33 vs 38.5). First-class passengers were also older overall (median 35 vs 28 in third class), which is the age–class correlation of −0.337. These two effects pull in opposite directions: older passengers were more often first class (which helped) but were older (which hurt). That explains why age on its own barely correlates with survival (−0.070).

**Chart 4 — Age vs fare by survival and sex** (`figures/age_fare_survival_sex.png`)
Survivors cluster among women at every age and among passengers paying high fares: those above the fare outlier bound of 65.66 survived at 67.5%, against 25.2% for those at or below the median fare. Young children are the exception to the male pattern: among passengers with a recorded age under 13, boys survived at 56.8%, triple the 18.9% rate of males overall. The vertical band at age 28 is the 177 median-imputed ages, which carry no real age information.

**Chart 5 — Survival by class, sex and embarkation port** (`figures/survival_class_sex_embarked.png`)
Cherbourg passengers survived most overall (55.4% vs 39.0% Queenstown and 33.7% Southampton), but 50.6% of Cherbourg's passengers were first class, compared with 19.7% of Southampton's. Within the same class and sex the port gap mostly shrinks: first-class women survived at 97.7% (Cherbourg) and 95.8% (Southampton). Port is therefore mainly a proxy for class mix. Some Queenstown bars rest on only 1–2 passengers (e.g. first-class Queenstown men, n=1) and should not be read as real rates.

### Task 6 — Standardization sanity check (EDA only)

Z-score `z = (x − mean) / std` applied manually to the full cleaned DataFrame:

| Column | Before: mean | Before: std | After: mean | After: std |
|---|---:|---:|---:|---:|
| `age` | 29.315 | 12.985 | 0.000000 | 1.000000 |
| `fare` | 32.097 | 49.698 | 0.000000 | 1.000000 |

Both columns come out with mean 0 and standard deviation 1, confirming the transform. This is only a sanity check. The modeling pipeline does its own scaling, fitted on the training split only.

---

## Part B — Predictive modeling

### Task 7 — Stratified split

Class balance: **549 did not survive (61.6%)**, **342 survived (38.4%)**.

`train_test_split(..., test_size=0.20, stratify=y, random_state=42)` → 712 training rows, 179 test rows.

**Why stratify:** with a 62/38 imbalance, a plain random split could, by chance, put noticeably more or fewer survivors in the 179-row test set. Precision, recall and F1 would then partly reflect the luck of the split rather than the model. Stratifying keeps the survivor share the same in both parts: **38.34% in train, 38.55% in test**. The split happens **before** any preprocessing.

### Task 8 — Preprocessing (fit on train only)

Features: `pclass`, `sex`, `age`, `sibsp`, `parch`, `fare`, `embarked`, `alone`.

A `ColumnTransformer` inside a scikit-learn `Pipeline`:

| Columns | Steps |
|---|---|
| numeric: `pclass`, `age`, `sibsp`, `parch`, `fare`, `alone` | `SimpleImputer(strategy="median")` → `StandardScaler()` |
| categorical: `sex`, `embarked` | `SimpleImputer(strategy="most_frequent")` → `OneHotEncoder(drop="first", handle_unknown="ignore")` |

Because the preprocessor is a step in the same `Pipeline` as the model, `pipeline.fit(X_train, y_train)` fits the imputers, encoder and scaler on the training split only. `pipeline.predict(X_test)` then applies them in transform-only mode, so the test data is never used for fitting. This is the same median/mode strategy as Task 2, but the medians are learned from the 712 training rows, not the full dataset.

### Task 9 — Three classifiers

All three use the identical split and preprocessing:

- **Logistic Regression**: `LogisticRegression(max_iter=1000, class_weight="balanced")`
- **Decision Tree**: `DecisionTreeClassifier(max_depth=5, class_weight="balanced")`. Depth is capped at 5 to limit overfitting.
- **Random Forest**: `RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced", oob_score=True)`

The decision tree is rendered with `plot_tree` in `figures/decision_tree.png`, labeled with feature names (e.g. `cat__sex_male`, `num__pclass`) and class names (`Not Survived`, `Survived`). For readability the figure shows the top 3 of its 5 levels. The root split is **sex**. Women are then split by **class** and **fare**, and men by **age**, which separates out young boys. This matches the EDA story.

### Task 10 — Evaluation of all three classifiers

Test set: 179 passengers, 69 of whom survived.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7989 | 0.7260 | **0.7681** | 0.7465 | **0.8501** |
| Decision Tree | 0.7709 | 0.7258 | 0.6522 | 0.6870 | 0.8060 |
| Random Forest | **0.8101** | **0.7536** | 0.7536 | **0.7536** | 0.8361 |

**Confusion matrices** (rows = actual `[not survived, survived]`, columns = predicted):

| Model | TN | FP | FN | TP |
|---|---:|---:|---:|---:|
| Logistic Regression | 90 | 20 | 16 | 53 |
| Decision Tree | 93 | 17 | 24 | 45 |
| Random Forest | 93 | 17 | 17 | 52 |

Plots: `figures/logistic_regression_confusion_matrix.png`, `figures/decision_tree_confusion_matrix.png`, `figures/random_forest_confusion_matrix.png`. ROC curves for all models: `figures/roc_curve_comparison.png`.

### Task 11 — Class-imbalance comparison

Balance: 61.6% not survived / 38.4% survived, a moderate imbalance. Model: Logistic Regression, retrained three ways on the same preprocessed training split.

| Variant | Precision | Recall | F1 |
|---|---:|---:|---:|
| (a) Baseline, no handling | **0.7833** | 0.6812 | 0.7287 |
| (b) `class_weight="balanced"` | 0.7260 | 0.7681 | 0.7465 |
| (c) SMOTE on the training fold only | 0.7397 | **0.7826** | **0.7606** |

SMOTE is applied with `SMOTE().fit_resample(X_train_encoded, y_train)` **after** the split and only to the training data. The test set keeps its real class ratio, so no synthetic rows leak into evaluation.

**Conclusion:** **SMOTE worked best**, with the highest F1 (0.761) and recall (0.783). Both imbalance strategies traded about 4–6 points of precision for 9–10 points of recall, raising F1 over the baseline's 0.729. The baseline favours the majority class: its high precision comes from predicting "survived" cautiously, and it misses more real survivors. SMOTE edges out class weighting because it adds new synthetic survivor examples between existing ones, while class weighting only up-weights the survivors already there. The gap is small, though: one extra correctly found survivor out of 69. `class_weight="balanced"` is a simpler alternative that gets nearly the same result.

### Task 12 — Hyperparameter tuning

`GridSearchCV` (5-fold, `scoring="roc_auc"`) over a `Pipeline` whose estimator is `RandomForestClassifier(oob_score=True, random_state=42)`:

| Parameter | Grid | **Best** |
|---|---|---|
| `n_estimators` | 100, 200 | **200** |
| `max_depth` | None, 5, 10 | **5** |
| `max_features` | `"sqrt"`, `"log2"` | **`"sqrt"`** |

- Best cross-validated ROC-AUC: **0.8755**
- **OOB score of the best forest: 0.8244**
- Test set: accuracy 0.8045, precision 0.8542, recall 0.5942, F1 0.7009, ROC-AUC 0.8399 (confusion matrix TN 103, FP 7, FN 28, TP 41)

The tuned forest is shallower (`max_depth=5`) and becomes very precise but conservative: it rarely predicts "survived" wrongly (7 false positives) but misses 28 of 69 survivors. Top feature importances: `sex_male` 0.443, `fare` 0.188, `pclass` 0.141, `age` 0.113.

### Task 13 — Regression side-task: predicting `fare`

`LinearRegression` in the same kind of `Pipeline`, predicting `fare` from `survived`, `pclass`, `age`, `sibsp`, `parch`, `sex`, `embarked`, `alone` (80/20 split, preprocessing fit on train only). Adjusted R² uses p = 9 features after one-hot encoding and n = 179 test rows.

| MAE | RMSE | R² | Adjusted R² |
|---:|---:|---:|---:|
| 20.89 | 30.58 | 0.3956 | 0.3635 |

**Residual plot:** `figures/fare_regression_residuals.png`.

**Heteroscedasticity: yes, it is present.** The residuals fan out as the predicted fare grows. Their standard deviation is 7.39 for the lowest third of predictions, 13.39 for the middle third and **45.71 for the highest third**, a 6.2× increase. |residual| correlates with the prediction at r = 0.549. The spread is not random: the model is fairly accurate for cheap third-class fares and very unreliable for expensive first-class ones, where fares range up to 512. The plot also curves, and some predictions fall below zero. A linear model on raw `fare` is a poor fit for such a right-skewed target; modeling `log(fare)` would be the natural next step.

### Task 14 — Model comparison and recommendation

Classification and regression metrics are on different scales and are **not comparable**. They are shown as two separate column groups, with a dash where a metric does not apply.

| Model | **Classification →** Accuracy | Precision | Recall | F1 | ROC-AUC | **Regression →** MAE | RMSE | R² | Adjusted R² |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7989 | 0.7260 | 0.7681 | 0.7465 | **0.8501** | — | — | — | — |
| Decision Tree | 0.7709 | 0.7258 | 0.6522 | 0.6870 | 0.8060 | — | — | — | — |
| Random Forest | **0.8101** | 0.7536 | 0.7536 | **0.7536** | 0.8361 | — | — | — | — |
| Tuned Random Forest | 0.8045 | **0.8542** | 0.5942 | 0.7009 | 0.8399 | — | — | — | — |
| Linear Regression (`fare`) | — | — | — | — | — | 20.89 | 30.58 | 0.3956 | 0.3635 |

Classification metrics are fractions from 0 to 1 (higher is better). MAE and RMSE are in fare units (lower is better), and R² is the share of fare variance explained.

**Recommendation: deploy Logistic Regression.** It has the best ROC-AUC on the test set (**0.850**), meaning it ranks survivors above non-survivors most reliably across all thresholds, and it finds the most survivors (recall **0.768**, 53 of 69). The untuned Random Forest has slightly higher accuracy (0.810 vs 0.799) and F1 (0.754 vs 0.747), but that 1-point edge is about 2 passengers out of 179, and it comes with lower AUC and much less interpretability. The tuned forest's high precision (0.854) costs too much recall (0.594), missing 28 survivors. Logistic Regression is also the simplest model to explain, since each coefficient shows a feature's direction and strength. That makes it the right default, and its decision threshold can be tuned later if precision matters more.

### Task 15 — Saved complete pipeline

The best model by test ROC-AUC, **Logistic Regression**, is saved as one fitted `Pipeline` object containing the `ColumnTransformer` (imputers, one-hot encoder, scaler) and the classifier:

```python
joblib.dump(best_pipeline, "analytics/models/best_pipeline.joblib")
```

`modeling.py` then reloads it with `joblib.load` and checks two things:

1. Its predictions on all 179 test rows are **identical** to the in-memory pipeline's.
2. It works on brand-new **raw** input (text categories, unscaled numbers and a missing age) with no manual preprocessing:

```text
class 1 female age=29.0 fare=100.00 -> survived=1 (p=0.957)
class 3 male   age=nan  fare=7.25   -> survived=0 (p=0.138)
```

`models/logistic_regression.joblib` and `models/random_forest.joblib` are also saved as complete pipelines.
