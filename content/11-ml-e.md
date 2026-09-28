=== POST ===
slug: feature-scaling-normalization
title: Feature Scaling and Normalisation: Standardisation, Min–Max and Robust Scaling
category: machine-learning
level: Beginner
tags: preprocessing, scaling, standardization, normalization
summary: Many algorithms silently assume features share a scale. We explain which models need scaling and why, compare standardisation, min–max, robust and quantile scaling, and show how to apply them without leakage.
---
Suppose you predict loan default from *annual income* (values around 500,000) and *number of dependants* (values 0–8). To a distance-based or gradient-based algorithm, income looks 100,000 times more important — purely because of its units. **Feature scaling** puts features on comparable scales. It takes one line of code, yet forgetting it is among the most common reasons a model underperforms.

## Which algorithms need scaling?

| Needs scaling | Why |
|---|---|
| k-NN, k-means, SVM (RBF), DBSCAN | Distances are dominated by large-scale features |
| Linear/logistic regression with regularisation | Penalties treat all coefficients equally |
| PCA | Variance is dominated by large-scale features |
| Neural networks, any gradient descent | Poor conditioning slows or destabilises optimisation |

| Does not need scaling | Why |
|---|---|
| Decision trees, random forests, gradient boosting | Splits depend only on the ordering of values |
| Naive Bayes (per-feature distributions) | Each feature is modelled separately |

## Why gradient descent cares

If features have very different scales, the loss surface becomes a long narrow valley: the Hessian is ill-conditioned, with condition number $\kappa$ roughly proportional to the ratio of feature variances. A learning rate small enough for the steep direction crawls along the flat one. Standardising makes the valley rounder, allowing larger steps and faster convergence.

## Standardisation (z-score)

$$
x' = \frac{x - \mu}{\sigma}
$$

Each feature gets mean 0 and standard deviation 1. It is the default for most models, preserves the shape of the distribution and does not bound values. It is sensitive to outliers because they inflate $\sigma$.

## Min–max scaling

$$
x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}
$$

Maps values to $[0, 1]$. Useful when an algorithm expects bounded inputs (image pixels are often scaled this way). Very sensitive to outliers: one extreme value squashes everything else into a tiny interval.

## Robust scaling

$$
x' = \frac{x - \text{median}}{\text{IQR}}
$$

Uses the median and interquartile range, which ignore extreme values. Choose it when features contain outliers you do not want to remove.

## Other transforms

- **Max-abs scaling** divides by the maximum absolute value — keeps zeros as zeros, suitable for **sparse** data such as TF-IDF matrices.
- **Power transforms** (Box–Cox for positive data, Yeo–Johnson for any data) reduce skewness towards a Gaussian shape.
- **Quantile transform** maps each feature to a uniform or normal distribution by ranks — very robust, but distorts distances and linear relationships.
- **Unit-norm (row) normalisation** scales each *sample* to length 1 — used for text vectors and embeddings when only direction matters (cosine similarity).

```python
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer

rng = np.random.default_rng(0)
income = np.r_[rng.lognormal(10, 0.5, 995), [5e6, 7e6, 8e6, 9e6, 1e7]].reshape(-1, 1)  # with outliers

for name, sc in [("standard", StandardScaler()), ("min-max", MinMaxScaler()),
                 ("robust", RobustScaler()), ("yeo-johnson", PowerTransformer())]:
    z = sc.fit_transform(income).ravel()
    print(f"{name:<12} median={np.median(z):8.3f}  IQR={np.subtract(*np.percentile(z, [75, 25])):8.3f}  max={z.max():9.2f}")
```

Notice how min–max scaling compresses the typical incomes into a tiny range because of five extreme values, while robust scaling and the power transform keep them spread out.

## Scaling without leakage

:::warning
Fit the scaler on the **training data only**, then apply the same fitted transformation to validation and test data. Fitting on the full dataset lets test statistics leak into training. The safest pattern is a `Pipeline`, which does this automatically inside cross-validation.
:::

```python
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(return_X_y=True)
print("SVM without scaling:", cross_val_score(SVC(), X, y, cv=5).mean().round(3))
print("SVM with scaling:   ", cross_val_score(make_pipeline(StandardScaler(), SVC()), X, y, cv=5).mean().round(3))
```

## Scaling targets

For regression with neural networks, scaling the **target** (e.g. standardising, or log-transforming skewed targets) often stabilises training. Remember to invert the transformation for predictions — scikit-learn's `TransformedTargetRegressor` handles this.

## Normalisation inside neural networks

Deep networks extend the idea internally with **batch normalisation**, **layer normalisation** and related techniques that keep activations well scaled layer by layer — covered in the deep learning track.

:::exercise
1. Train k-NN on the wine dataset with each scaler and compare cross-validated accuracy.
2. Fit logistic regression with gradient descent on unscaled and standardised features; count the iterations to converge.
3. When would you prefer the quantile transform over standardisation? Give a concrete example.
:::

:::takeaway
- Distance-based, regularised, PCA and gradient-trained models need scaled features; trees do not.
- Standardise by default; use robust scaling with outliers, min–max for bounded inputs, max-abs for sparse data.
- Power and quantile transforms reduce skew.
- Always fit scalers on training data only — use pipelines.
:::

=== POST ===
slug: handling-missing-data
title: Handling Missing Data: Mechanisms, Imputation and Indicators
category: machine-learning
level: Intermediate
tags: missing data, imputation, mcar, mar, mnar, preprocessing
summary: Missing values are rarely random. We classify missingness as MCAR, MAR or MNAR, compare deletion and imputation strategies from simple to iterative, and show why a missingness indicator is often a feature in itself.
---
Real datasets have holes. A survey respondent skips the income question; a sensor fails during a storm; a clinic did not record a test. How you handle these gaps can bias your conclusions, leak information or discard valuable signal. Before choosing a technique, you must ask **why** the data is missing.

## Missingness mechanisms (Rubin's taxonomy)

1. **MCAR — Missing Completely At Random.** The probability of missingness is unrelated to any data, observed or not. Example: a random subset of forms was lost. Deleting incomplete rows loses efficiency but does not bias results.
2. **MAR — Missing At Random.** Missingness depends only on **observed** variables. Example: younger respondents skip the income question more often, and age is recorded. Imputation methods that condition on observed variables can correct for this.
3. **MNAR — Missing Not At Random.** Missingness depends on the **missing value itself**. Example: people with very high or very low incomes decline to report income. No method can fully correct this without assumptions or extra data; it requires domain reasoning and sensitivity analysis.

:::note
You usually cannot prove which mechanism holds from the data alone. But you can investigate: compare the distributions of other variables for rows with and without missing values. If they differ, the data is not MCAR — and simply dropping rows may bias your model.
:::

## Strategy 1: deletion

- **Listwise deletion** — drop rows with any missing value. Simple; acceptable if missingness is rare and MCAR. Dangerous otherwise: with 20 features each missing 5% independently, you lose about 64% of rows.
- **Column deletion** — drop features that are mostly missing (e.g. > 60–80%), unless the missingness itself is informative.

## Strategy 2: simple imputation

Replace missing values with a statistic computed on the **training set**:

- **Mean** — for roughly symmetric numeric features.
- **Median** — robust for skewed features (income).
- **Most frequent / constant "Missing" category** — for categorical features.

Simple imputation shrinks variance and weakens correlations, but combined with an indicator (below) it is often surprisingly effective for predictive modelling.

## Strategy 3: missingness indicators

Add a binary feature `x_is_missing`. This lets the model learn that **missingness itself is informative** — a missing lab test may mean the doctor did not consider it necessary, which says something about the patient. In predictive tasks, indicator + simple imputation frequently beats sophisticated imputation.

## Strategy 4: model-based imputation

- **k-NN imputation** — fill a value from similar rows.
- **Iterative imputation (MICE-style)** — model each feature with missing values as a function of the others, cycling until convergence.
- **Multiple imputation** — create several plausible completed datasets, analyse each, and combine the results (Rubin's rules). This is the gold standard for **statistical inference**, because it propagates uncertainty about the missing values into standard errors.

## Strategy 5: models that handle missingness natively

Gradient boosting libraries (XGBoost, LightGBM, scikit-learn's `HistGradientBoosting`) learn a default direction for missing values at each split — often the simplest and best option for tabular prediction.

```python
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
rng = np.random.default_rng(0)
X_miss = X.copy()
# MAR-style missingness: feature 0 missing more often when feature 1 is high
p = 0.1 + 0.5 * (X[:, 1] > np.median(X[:, 1]))
X_miss[rng.random(len(X)) < p, 0] = np.nan
X_miss[rng.random(X.shape) < 0.05] = np.nan            # plus some random holes

strategies = {
    "mean":            SimpleImputer(strategy="mean"),
    "median+indicator": SimpleImputer(strategy="median", add_indicator=True),
    "kNN":             KNNImputer(n_neighbors=5),
    "iterative":       IterativeImputer(random_state=0, max_iter=10),
}
for name, imp in strategies.items():
    pipe = make_pipeline(imp, StandardScaler(), LogisticRegression(max_iter=3000))
    print(f"{name:<17} acc = {cross_val_score(pipe, X_miss, y, cv=5).mean():.3f}")
print(f"{'native (HGB)':<17} acc = {cross_val_score(HistGradientBoostingClassifier(), X_miss, y, cv=5).mean():.3f}")
```

## Rules for doing it right

:::warning
1. **Fit imputers on training data only** — inside a pipeline — to avoid leakage.
2. **Never impute the target** for training; drop rows with missing labels (or treat them as unlabelled for semi-supervised learning).
3. **Beware encoded missingness**: values like −999, 0, "N/A" or blank strings often mean "missing". Find them during EDA.
4. **Report** how much data was missing and how you handled it; decisions can change conclusions.
:::

## Choosing an approach

| Goal | Recommended |
|---|---|
| Quick, strong predictive baseline | Gradient boosting with native handling, or median imputation + indicators |
| Linear models / neural networks | Imputation + indicators inside a pipeline |
| Statistical inference (effects, confidence intervals) | Multiple imputation |
| Suspected MNAR | Domain reasoning, sensitivity analysis, collect more data |

:::exercise
1. For a household survey, give one realistic example each of MCAR, MAR and MNAR missingness.
2. Using the code above, make feature 0's missingness depend on its own value (MNAR). Which strategy suffers most?
3. Show numerically that mean imputation reduces the variance of a feature and its correlation with other features.
:::

:::takeaway
- Classify missingness: MCAR, MAR, MNAR — the mechanism determines what is safe.
- Deletion is only safe for rare MCAR gaps.
- Simple imputation + missingness indicators is a strong predictive baseline; multiple imputation suits inference.
- Fit imputers within pipelines; look for disguised missing values.
:::

=== POST ===
slug: encoding-categorical-variables
title: Encoding Categorical Variables: One-Hot, Ordinal, Target and Beyond
category: machine-learning
level: Beginner
tags: preprocessing, categorical, one-hot encoding, target encoding, embeddings
summary: Models need numbers, but many features are categories. We compare one-hot, ordinal, frequency, target and hashing encoders, handle high cardinality and unseen categories, and avoid target-encoding leakage.
---
District, occupation, language, product type, blood group — tabular data is full of **categorical variables**. Most algorithms require numeric input, so we must encode categories as numbers. The naive approach — assigning integers 1, 2, 3 — can quietly damage a model. Choosing the right encoding depends on the variable's nature, its **cardinality** (number of distinct values) and the model.

## Nominal vs ordinal

- **Nominal** categories have no order: colours, countries, languages.
- **Ordinal** categories have a natural order: education level, severity (mild < moderate < severe), income bracket.

## Ordinal (integer) encoding

Map categories to integers. Appropriate for **ordinal** variables in the correct order.

:::warning
Integer-encoding a **nominal** variable (Dhaka = 1, Chattogram = 2, Sylhet = 3) tells a linear model that Sylhet is "three times" Dhaka and that Chattogram lies between them. Linear models, k-NN and neural networks will misuse this fake ordering. Tree models tolerate it better, because they can isolate categories with several splits — but even they benefit from a sensible encoding.
:::

## One-hot encoding

Create one binary column per category:

| district | is_Dhaka | is_Chattogram | is_Sylhet |
|---|---|---|---|
| Dhaka | 1 | 0 | 0 |
| Sylhet | 0 | 0 | 1 |

- The default for nominal variables with **low cardinality**.
- For linear models with an intercept, drop one column (or rely on regularisation) to avoid perfect collinearity — the "dummy variable trap".
- Handle **unseen categories** at prediction time (`handle_unknown="ignore"` produces an all-zero row).
- With high cardinality (thousands of values), one-hot creates huge sparse matrices and rare columns with little data. Group rare categories into "Other" (`min_frequency` in scikit-learn).

## Frequency / count encoding

Replace each category with how often it appears. Simple, one column, and often useful for trees (frequent vs rare categories behave differently). Different categories with equal counts become indistinguishable.

## Target (mean) encoding

Replace each category with the **mean target** for that category — e.g. the default rate of each occupation. Compact and powerful for high-cardinality features, but dangerous:

- **Leakage**: if a row's own label contributes to its encoding, the model sees the answer. Rare categories become near-perfect predictors on training data and fail on test data.
- **Noise**: a category seen twice has an unreliable mean.

Remedies:

1. **Out-of-fold encoding** — compute each row's encoding from other folds only (scikit-learn's `TargetEncoder` does this with cross-fitting).
2. **Smoothing** — shrink towards the global mean:

$$
\text{enc}(c) = \frac{n_c\,\bar{y}_c + m\,\bar{y}}{n_c + m}
$$

where $n_c$ is the category count and $m$ controls the strength of shrinkage — a Bayesian estimate with a prior centred on the global mean.

3. **Ordered target statistics** (CatBoost) — encode each row using only earlier rows in a random permutation.

## Hashing encoding

Map categories to a fixed number of columns with a hash function. Memory is fixed regardless of cardinality, and new categories need no dictionary — useful for streaming data and huge vocabularies. The cost is **collisions** (different categories sharing a column) and loss of interpretability.

## Learned embeddings

Neural networks can learn a dense vector for each category via an **embedding layer**, as they do for words. Similar categories (e.g. products bought together) end up close in embedding space. Entity embeddings are especially effective for high-cardinality features in deep tabular models, and the learned vectors can be reused in other models.

## Comparison

| Encoding | Columns | Best for | Risks |
|---|---|---|---|
| Ordinal | 1 | True ordinal variables; trees | Fake order for nominal data |
| One-hot | $K$ | Low-cardinality nominal | Dimensionality explosion |
| Frequency | 1 | Trees; high cardinality | Collisions of equal counts |
| Target | 1 (per class) | High cardinality | Leakage — use cross-fitting + smoothing |
| Hashing | fixed $m$ | Streaming, huge cardinality | Collisions |
| Embedding | $d$ | Neural networks | Needs data; less interpretable |

```python
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, TargetEncoder
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

rng = np.random.default_rng(0)
n = 5000
occupations = [f"occ_{i}" for i in range(300)]              # high cardinality
occ_risk = dict(zip(occupations, rng.normal(0, 1, 300)))
df = pd.DataFrame({
    "education": rng.choice(["primary", "secondary", "tertiary"], n),
    "district": rng.choice(["Dhaka", "Chattogram", "Sylhet", "Khulna"], n),
    "occupation": rng.choice(occupations, n),
})
logit = df["occupation"].map(occ_risk) + (df["education"] == "tertiary") * 0.8
y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

prep = ColumnTransformer([
    ("edu", OrdinalEncoder(categories=[["primary", "secondary", "tertiary"]]), ["education"]),
    ("dist", OneHotEncoder(handle_unknown="ignore"), ["district"]),
    ("occ", TargetEncoder(target_type="binary", random_state=0), ["occupation"]),  # cross-fitted
])
pipe = make_pipeline(prep, LogisticRegression(max_iter=1000))
print("CV AUC:", cross_val_score(pipe, df, y, cv=5, scoring="roc_auc").mean().round(3))
```

:::tip
A practical default: one-hot for low-cardinality nominal features (< ~15 values), ordinal for truly ordered features, cross-fitted smoothed target encoding (or native categorical support in LightGBM/CatBoost) for high-cardinality features.
:::

:::exercise
1. Encode a nominal feature as integers and as one-hot for a logistic regression. Compare performance.
2. Implement naive (leaky) target encoding on the example data and compare training vs cross-validated AUC with the cross-fitted version.
3. Derive the smoothed target-encoding formula as the posterior mean of a Beta–Bernoulli model.
:::

:::takeaway
- Distinguish nominal from ordinal categories; never give nominal data a fake order in linear models.
- One-hot for low cardinality; target, frequency, hashing or embeddings for high cardinality.
- Target encoding leaks unless cross-fitted and smoothed.
- Plan for unseen and rare categories at prediction time.
:::

=== POST ===
slug: imbalanced-data-techniques
title: Learning from Imbalanced Data
category: machine-learning
level: Intermediate
tags: imbalanced data, smote, class weights, resampling, rare events
summary: When one class is rare, naive models ignore it. We cover the right metrics, class weighting, over- and under-sampling, SMOTE, threshold moving and calibration — and when each is appropriate.
---
Fraud is rare. Serious disease is rare. Equipment failure is rare. Yet these rare events are usually exactly what we want to detect. In **imbalanced** datasets, a standard classifier minimising average error can achieve excellent accuracy by ignoring the minority class entirely. Today we study how to learn and evaluate properly when classes are imbalanced.

## Step 1: fix the evaluation first

Before changing the model, change the metric. Accuracy is meaningless when 99% of examples are negative. Use:

- **Precision, recall and F1** for the minority class;
- the **precision–recall curve and average precision** (baseline = prevalence);
- **balanced accuracy** or **MCC**;
- **cost-based metrics** when error costs are known;
- **stratified** splits so every fold contains minority examples.

Often, a model trained normally on imbalanced data already *ranks* examples well (good PR-AUC) and only needs a **better threshold**. Check that before resampling anything.

## Step 2: threshold moving

A classifier outputs $P(y = 1 \mid \mathbf{x})$. The default threshold 0.5 is appropriate only for equal error costs. With false-negative cost $c_{FN}$ and false-positive cost $c_{FP}$, a well-calibrated model should predict positive when

$$
P(y = 1 \mid \mathbf{x}) \ge \frac{c_{FP}}{c_{FP} + c_{FN}}
$$

Alternatively, choose the threshold on validation data to hit a required recall or a review capacity. Threshold moving is simple, preserves calibration and is often all you need.

## Step 3: cost-sensitive learning (class weights)

Weight the loss so that minority errors count more:

$$
L = -\frac{1}{n}\sum_i w_{y_i}\big[y_i\ln p_i + (1 - y_i)\ln(1 - p_i)\big]
$$

A common choice is $w_c \propto 1/n_c$ (`class_weight="balanced"` in scikit-learn; `scale_pos_weight` in XGBoost/LightGBM). Weighting changes the decision boundary without discarding or duplicating data.

:::note
Class weighting and resampling **distort predicted probabilities**: a model trained as if positives were common will over-predict them. If you need calibrated probabilities (for risk communication or expected-cost decisions), recalibrate on untouched validation data, or prefer threshold moving.
:::

## Step 4: resampling

- **Random under-sampling** of the majority class — fast and effective with huge datasets, but discards information.
- **Random over-sampling** of the minority class — duplicates examples; risks overfitting to those exact points.
- **SMOTE** (Synthetic Minority Over-sampling Technique, Chawla et al., 2002) — creates synthetic minority examples by interpolating between a minority point and one of its $k$ minority nearest neighbours:

$$
\mathbf{x}_{\text{new}} = \mathbf{x}_i + \lambda(\mathbf{x}_{\text{nn}} - \mathbf{x}_i), \qquad \lambda \sim U(0, 1)
$$

Variants such as Borderline-SMOTE and ADASYN focus on hard examples near the boundary; **Tomek links** and **ENN** clean overlapping majority examples.

:::warning
**Resample only the training data, and only inside cross-validation folds.** If you apply SMOTE before splitting, synthetic points derived from test examples leak into training, and your evaluation becomes wildly optimistic. Use `imblearn.pipeline.Pipeline`, which applies samplers only during `fit`.
:::

## Step 5: algorithmic approaches

- **Ensembles with balanced sampling** — e.g. Balanced Random Forest, EasyEnsemble (bagging on balanced subsamples).
- **Focal loss** — down-weights easy examples so training focuses on hard ones; popular in object detection where background dominates.
- **Anomaly detection** — when positives are extremely rare or diverse, model the normal class only.

## Putting it together

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, recall_score, precision_score
from imblearn.pipeline import make_pipeline as imb_pipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

X, y = make_classification(n_samples=20000, n_features=15, n_informative=6,
                           weights=[0.99, 0.01], class_sep=1.0, random_state=0)
cv = StratifiedKFold(5, shuffle=True, random_state=0)

setups = {
    "plain":        imb_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    "class weight": imb_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
    "SMOTE":        imb_pipeline(StandardScaler(), SMOTE(random_state=0), LogisticRegression(max_iter=2000)),
}
for name, pipe in setups.items():
    p = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
    pred = (p >= 0.5).astype(int)
    print(f"{name:<13} AP={average_precision_score(y, p):.3f}  "
          f"P={precision_score(y, pred):.3f}  R={recall_score(y, pred):.3f}  F1={f1_score(y, pred):.3f}")

# Threshold moving on the plain model: pick the threshold maximising F1 (on out-of-fold predictions)
p = cross_val_predict(setups["plain"], X, y, cv=cv, method="predict_proba")[:, 1]
ths = np.linspace(0.02, 0.9, 89)
best = max(ths, key=lambda t: f1_score(y, (p >= t).astype(int)))
print(f"plain + threshold {best:.2f}: F1={f1_score(y, (p >= best).astype(int)):.3f}")
```

Typically all approaches reach similar **average precision** (ranking quality), while their F1 at threshold 0.5 differs greatly — confirming that much of the "imbalance problem" is really a **threshold problem**. (In a real project, choose the threshold on a validation set separate from the final test set.)

## Practical recommendations

1. Use appropriate metrics and stratified splits.
2. Start with a strong model (e.g. gradient boosting) and **tune the threshold**.
3. Try class weights next; they are cheap and preserve all data.
4. Try resampling (SMOTE, under-sampling) if the above is insufficient — inside pipelines only.
5. Recalibrate probabilities if they will be used as risks.
6. Above all, **collect more minority examples** if at all possible — nothing beats real data.

:::exercise
1. Derive the cost-optimal threshold $c_{FP}/(c_{FP} + c_{FN})$ from expected costs.
2. Apply SMOTE **before** splitting in the example and compare the (leaky) score with the correct pipeline.
3. Train LightGBM with and without `scale_pos_weight` and compare calibration curves.
:::

:::takeaway
- Fix evaluation first: PR-AUC, recall/precision, MCC, stratified splits.
- Threshold moving is often the simplest and best remedy.
- Class weights and resampling (SMOTE) shift the boundary but distort probabilities.
- Resample only inside training folds; recalibrate if probabilities matter.
:::

=== POST ===
slug: feature-selection-methods
title: Feature Selection: Filter, Wrapper and Embedded Methods
category: machine-learning
level: Intermediate
tags: feature selection, mutual information, rfe, lasso, dimensionality
summary: More features are not always better. We compare filter methods (correlation, mutual information), wrappers (RFE, sequential selection) and embedded methods (lasso, tree importance), and learn to select without leaking.
---
Adding irrelevant features makes models slower, harder to interpret, more expensive to maintain and — especially with limited data — less accurate. **Feature selection** chooses a subset of informative features. Unlike PCA, which creates new combined features, selection keeps original features, preserving interpretability: "the model uses these twelve survey questions" is easier to explain than "the model uses principal component 7".

## Why select features?

- **Generalisation** — fewer irrelevant inputs reduce variance and the curse of dimensionality.
- **Interpretability** — simpler models are easier to explain and audit.
- **Cost** — each feature may cost money to collect (a lab test, a survey question, a sensor).
- **Speed** — training and inference get faster.
- **Robustness** — fewer features means fewer things to break when data pipelines change.

## Three families of methods

### 1. Filter methods
Score each feature independently of any model, then keep the top ones.

- **Variance threshold** — remove near-constant features.
- **Correlation** with the target (linear relationships only).
- **Statistical tests** — ANOVA F-test for numeric features vs class; chi-squared for count features vs class.
- **Mutual information** — captures non-linear dependence.

Fast and model-agnostic, but they ignore **interactions** (a feature useless alone may be powerful in combination) and **redundancy** (ten copies of the same informative feature all score highly).

### 2. Wrapper methods
Search over feature subsets using a model's validated performance as the score.

- **Sequential forward selection** — start empty; repeatedly add the feature that improves CV score most.
- **Sequential backward elimination** — start with all; repeatedly remove the least useful.
- **Recursive Feature Elimination (RFE)** — fit a model, drop the least important features by coefficient or importance, repeat. **RFECV** chooses the number of features by cross-validation.

Wrappers account for interactions and the specific model, but are computationally expensive ($O(d^2)$ model fits for sequential search) and can overfit the validation data when many subsets are compared.

### 3. Embedded methods
Selection happens during training.

- **Lasso / L1-regularised models** drive irrelevant coefficients to exactly zero.
- **Tree-based importance** from random forests or gradient boosting.
- **Elastic net** for correlated feature groups.

Efficient and usually a good compromise.

## Comparing methods

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.feature_selection import (SelectKBest, mutual_info_classif, f_classif, RFECV,
                                       SelectFromModel)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold

# 8 informative, 4 redundant, 88 pure-noise features
X, y = make_classification(n_samples=600, n_features=100, n_informative=8, n_redundant=4,
                           n_repeated=0, shuffle=False, random_state=0)
cv = StratifiedKFold(5, shuffle=True, random_state=0)
base = LogisticRegression(max_iter=3000)

pipes = {
    "all features":   make_pipeline(StandardScaler(), base),
    "filter: F-test": make_pipeline(StandardScaler(), SelectKBest(f_classif, k=12), base),
    "filter: MI":     make_pipeline(StandardScaler(), SelectKBest(mutual_info_classif, k=12), base),
    "embedded: L1":   make_pipeline(StandardScaler(),
                                    SelectFromModel(LogisticRegression(penalty="l1", C=0.1, solver="liblinear")),
                                    base),
    "wrapper: RFECV": make_pipeline(StandardScaler(), RFECV(LogisticRegression(max_iter=3000), step=5, cv=3), base),
}
for name, p in pipes.items():
    print(f"{name:<16} CV accuracy = {cross_val_score(p, X, y, cv=cv).mean():.3f}")
```

The first 12 columns are the useful ones (because `shuffle=False`); inspecting which features each method keeps shows how well it recovers them.

## The selection-bias trap

:::warning
Feature selection is part of model fitting. If you select features using the **entire dataset** and then cross-validate, you leak information — with enough noise features, you can achieve impressive "accuracy" on pure noise (we demonstrated this in the cross-validation lecture). Always place selection **inside** the pipeline so it is repeated within each training fold.
:::

## Stability

Selected subsets can change dramatically with small data perturbations, especially with correlated features. **Stability selection** (Meinshausen & Bühlmann) runs lasso on many bootstrap subsamples and keeps features selected in a large fraction of runs — a more trustworthy basis for scientific claims about which variables matter.

## Practical guidance

| Situation | Suggested approach |
|---|---|
| Thousands of features, quick reduction | Variance threshold + univariate filter (MI) |
| Linear model, want sparsity | Lasso / elastic net |
| Tree models | Usually no selection needed; prune with permutation importance if desired |
| Few features, compute available | RFECV or sequential selection |
| Scientific claims about variables | Stability selection; report uncertainty |
| Features are costly to collect | Wrapper with a cost-aware objective |

:::tip
Modern gradient boosting and regularised models cope well with many irrelevant features. Feature selection is most valuable when data is scarce, features are costly, interpretability is required, or inference speed matters.
:::

:::exercise
1. Explain why univariate filters can miss XOR-type interactions. Construct an example.
2. Run the example, extract which feature indices each method selects, and compute the fraction of truly informative features recovered.
3. Implement stability selection with lasso over 100 bootstrap samples and plot selection frequencies.
:::

:::takeaway
- Filters are fast but ignore interactions and redundancy; wrappers are accurate but expensive; embedded methods balance both.
- Lasso and tree importance are practical embedded selectors.
- Selection must happen inside cross-validation to avoid leakage.
- Check stability before making claims about which features matter.
:::

=== POST ===
slug: hyperparameter-tuning
title: Hyperparameter Tuning: Grid, Random, Bayesian and Early-Stopping Methods
category: machine-learning
level: Intermediate
tags: hyperparameters, grid search, random search, bayesian optimization, optuna
summary: Hyperparameters control how models learn. We compare grid search, random search, Bayesian optimisation and successive halving, explain why random search beats grid search, and tune efficiently with Optuna.
---
**Parameters** are learned from data (weights, split thresholds). **Hyperparameters** are chosen before training and control the learning process: learning rate, tree depth, regularisation strength, number of neighbours. Good hyperparameters can be the difference between a mediocre and an excellent model. Tuning them efficiently — without overfitting the validation set — is a core skill.

## The tuning problem

We want

$$
\boldsymbol{\lambda}^* = \arg\min_{\boldsymbol{\lambda} \in \Lambda}\; \text{ValError}\big(\mathcal{A}_{\boldsymbol{\lambda}}(\mathcal{D}_{\text{train}}),\, \mathcal{D}_{\text{val}}\big)
$$

This objective is expensive (each evaluation trains a model), noisy, non-differentiable, and often mixes continuous, integer and categorical dimensions. It is a **black-box optimisation** problem.

## Grid search

Evaluate every combination on a predefined grid. Exhaustive and easy to parallelise, but the cost grows **exponentially** with the number of hyperparameters: 5 values for each of 6 hyperparameters = 15,625 runs.

## Random search

Sample configurations at random from specified distributions. Bergstra and Bengio (2012) showed that random search is usually **far more efficient** than grid search. The reason: typically only a few hyperparameters really matter. With a 3×3 grid over two hyperparameters, you test only **3 distinct values** of the important one. With 9 random samples, you test **9 distinct values** of each.

:::tip
Sample scale-type hyperparameters (learning rate, regularisation strength, $C$, $\gamma$) **log-uniformly**, e.g. between $10^{-5}$ and $10^{-1}$. Sampling uniformly would spend 90% of trials between 0.01 and 0.1.
:::

## Bayesian optimisation

Random search ignores what it has learned. **Bayesian optimisation** builds a **surrogate model** of the objective from past trials and chooses the next configuration by maximising an **acquisition function** that balances exploring uncertain regions and exploiting promising ones.

- **Gaussian-process** surrogates with **Expected Improvement**:
$$
\text{EI}(\boldsymbol{\lambda}) = \mathbb{E}\big[\max(0,\; f^* - f(\boldsymbol{\lambda}))\big]
$$
where $f^*$ is the best value so far.
- **Tree-structured Parzen Estimators (TPE)**, used by Optuna and Hyperopt, model the densities of good and bad configurations and handle conditional, mixed-type spaces well.

Bayesian methods typically find good configurations in fewer trials — valuable when each trial takes hours.

## Multi-fidelity methods: stop bad trials early

Most configurations are clearly bad after a small fraction of the budget. **Successive halving** trains many configurations with a small budget (few epochs, a data subset), keeps the best fraction (e.g. top third), gives them more budget, and repeats. **Hyperband** runs successive halving with several trade-offs between breadth and depth. **ASHA** is an asynchronous version for parallel clusters. Combining Bayesian sampling with early stopping (e.g. BOHB, or Optuna with pruners) is the modern standard.

## Practical example with Optuna

```python
import optuna
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier

X, y = load_breast_cancer(return_X_y=True)
X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
cv = StratifiedKFold(5, shuffle=True, random_state=0)

def objective(trial):
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 4, 64, log=True),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 100, log=True),
        "l2_regularization": trial.suggest_float("l2_regularization", 1e-6, 10, log=True),
        "max_iter": 400,
    }
    model = HistGradientBoostingClassifier(**params, random_state=0)
    return cross_val_score(model, X_dev, y_dev, cv=cv, scoring="roc_auc").mean()

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
study.optimize(objective, n_trials=60, show_progress_bar=False)
print("best CV AUC:", round(study.best_value, 4), "\nbest params:", study.best_params)

final = HistGradientBoostingClassifier(**study.best_params, max_iter=400, random_state=0).fit(X_dev, y_dev)
print("held-out test accuracy:", round(final.score(X_test, y_test), 4))
```

## Avoiding validation overfitting

:::warning
Every trial "peeks" at the validation data. After hundreds of trials, the best validation score is optimistically biased — partly skill, partly luck. Protect yourself:

1. Keep a **final test set** that tuning never sees (as above), or use **nested cross-validation**.
2. Prefer cross-validation over a single small validation split.
3. Be sceptical of tiny improvements found after many trials.
:::

## What to tune (and what not to)

| Model | High-impact hyperparameters |
|---|---|
| Logistic / linear | Regularisation strength ($C$ or $\alpha$), penalty type |
| SVM (RBF) | $C$, $\gamma$ |
| Random forest | `max_features`, `min_samples_leaf` (trees: just "enough") |
| Gradient boosting | learning rate + number of trees (early stopping), depth/leaves, min child samples, subsampling |
| Neural networks | learning rate (by far), batch size, weight decay, schedule, architecture width/depth |

Tune the few that matter; fix the rest at sensible defaults. And remember: hyperparameter tuning usually gives **smaller** gains than better data and features.

## Reproducibility

Log every trial's configuration, seed, code version and score (Optuna, MLflow or Weights & Biases). Report the search space and budget in papers — a method tuned with 500 trials compared against a baseline tuned with 10 is not a fair comparison.

:::exercise
1. Compare grid search (5×5) and random search (25 trials) for an RBF SVM's $C$ and $\gamma$. Which finds a better configuration?
2. Add a `MedianPruner` to an Optuna study for a neural network that reports validation loss each epoch. How much time is saved?
3. Explain why the best validation score after many trials is biased upwards.
:::

:::takeaway
- Hyperparameters are chosen outside training; tuning is noisy black-box optimisation.
- Random search beats grid search because few hyperparameters matter; sample scales log-uniformly.
- Bayesian optimisation (GP, TPE) and multi-fidelity methods (Hyperband, ASHA) are more efficient.
- Keep a final untouched test set; report search budgets for fair comparisons.
:::

=== POST ===
slug: anomaly-detection
title: Anomaly Detection: Finding the Unusual
category: machine-learning
level: Intermediate
tags: anomaly detection, outliers, isolation forest, one-class svm, autoencoders
summary: Fraud, faults and intrusions are rare and varied. We cover statistical, distance-, density- and tree-based detectors, autoencoders for complex data, and how to evaluate detectors when labels are scarce.
---
Anomaly detection asks: which observations do not fit the pattern of the rest? A sudden spike in network traffic, a transaction far from a customer's habits, a sensor reading drifting before a machine fails, a registration record with impossible values. Anomalies are **rare**, often **unlabelled**, and **diverse** — tomorrow's fraud may look nothing like yesterday's. This makes anomaly detection one of the most practically important and conceptually subtle areas of ML.

## Types of anomalies

- **Point anomalies** — a single observation is unusual (a withdrawal of 50× the usual amount).
- **Contextual anomalies** — unusual in context (30 °C is normal in summer, anomalous in winter).
- **Collective anomalies** — a sequence or group is unusual even if each point is not (a slow, steady data exfiltration).

## Settings

- **Supervised**: labelled anomalies exist → treat as imbalanced classification.
- **Semi-supervised (novelty detection)**: train only on normal data; flag anything different.
- **Unsupervised (outlier detection)**: unlabelled data containing some anomalies; assume they are few and different.

## Statistical methods

For a roughly Gaussian feature, flag points with $|z| = |x - \mu|/\sigma > 3$. More robust: the **modified z-score** using the median and median absolute deviation (MAD), since outliers inflate the mean and standard deviation. For multivariate data, the **Mahalanobis distance**

$$
D_M(\mathbf{x}) = \sqrt{(\mathbf{x} - \boldsymbol{\mu})^\top\boldsymbol{\Sigma}^{-1}(\mathbf{x} - \boldsymbol{\mu})}
$$

accounts for correlations — a point can be normal in each feature separately yet anomalous jointly (tall *and* very light). Use a robust covariance estimate (Minimum Covariance Determinant, `EllipticEnvelope`).

## Distance- and density-based methods

- **k-NN distance** — the distance to the $k$-th nearest neighbour; large = isolated.
- **Local Outlier Factor (LOF)** — compares a point's local density to its neighbours' densities. A point in a sparse region *next to* a dense cluster is flagged, even if globally the sparse region is normal. LOF handles clusters of varying density.

## Isolation Forest

**Isolation Forest** (Liu, Ting & Zhou, 2008) inverts the usual logic: instead of modelling normal data, it isolates points. Build random trees by choosing a random feature and a random split value. Anomalies — few and different — get isolated in **few splits** (short paths); normal points require many. The anomaly score is based on the average path length $E[h(\mathbf{x})]$ across trees:

$$
s(\mathbf{x}) = 2^{-E[h(\mathbf{x})]/c(n)}
$$

where $c(n)$ normalises by the average path length of an unsuccessful search in a binary tree. Scores near 1 indicate anomalies. It is fast, scales to large data and works well in moderately high dimensions — an excellent default.

## One-Class SVM

Learns a boundary around normal data in a kernel feature space; points outside are anomalies. Powerful but sensitive to kernel parameters and slow on large datasets.

## Reconstruction-based methods: autoencoders

For images, sequences and other complex data, train an **autoencoder** on normal data. It learns to compress and reconstruct normal patterns; anomalous inputs reconstruct poorly. The **reconstruction error** $\|\mathbf{x} - \hat{\mathbf{x}}\|^2$ is the anomaly score. Variants use VAEs, predictive models for time series (forecast error as the score), or embeddings from pretrained networks combined with k-NN.

## Comparing detectors

```python
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(0)
normal = np.vstack([rng.normal([0, 0], 1, (900, 2)), rng.normal([6, 6], 0.5, (300, 2))])
anomalies = rng.uniform(-6, 12, (40, 2))
X = StandardScaler().fit_transform(np.vstack([normal, anomalies]))
y = np.r_[np.zeros(len(normal)), np.ones(len(anomalies))]      # 1 = anomaly (used only to evaluate)

detectors = {
    "Isolation Forest": IsolationForest(n_estimators=300, random_state=0).fit(X),
    "LOF": LocalOutlierFactor(n_neighbors=30, novelty=True).fit(X),
    "One-Class SVM": OneClassSVM(gamma=0.5, nu=0.05).fit(X),
    "Robust covariance": EllipticEnvelope(contamination=0.05, random_state=0).fit(X),
}
for name, d in detectors.items():
    score = -d.score_samples(X)                       # higher = more anomalous
    print(f"{name:<18} ROC-AUC={roc_auc_score(y, score):.3f}  AP={average_precision_score(y, score):.3f}")
```

The robust-covariance detector assumes a single elliptical cluster and struggles with two normal clusters; Isolation Forest and LOF adapt to the structure.

## Evaluating without many labels

- If some labelled anomalies exist, use **precision at k**, **PR-AUC** and recall at a fixed alert budget.
- Otherwise, have domain experts review the **top-ranked** alerts and measure the fraction that are genuine (precision at k) — this is how most real systems are evaluated.
- The `contamination` parameter sets the decision threshold, not the ranking; choose it from the alert capacity your team can review.

:::warning
"Anomalous" is not the same as "interesting" or "bad". A detector flags *rare* things, which may include data-entry errors, new legitimate behaviour, or under-represented groups. In humanitarian or social-protection contexts, automatically treating anomalies as fraud can harm vulnerable people whose circumstances are simply uncommon. Keep a human in the loop and review flagged cases for fairness.
:::

## Practical tips

- Scale features; engineer features that express "normal behaviour" (e.g. deviation from a user's own history).
- Handle time: use rolling baselines and seasonality for time series.
- Combine detectors (average normalised ranks) for robustness.
- Monitor and retrain — normal behaviour drifts.

:::exercise
1. Show why LOF flags a point near a dense cluster that a global k-NN distance would miss. Build a 2-D example.
2. Train an autoencoder on MNIST digits 0–8 and use reconstruction error to detect digit 9. Report ROC-AUC.
3. Design an alerting policy for a team that can review 50 cases per day from 100,000 daily transactions.
:::

:::takeaway
- Anomalies are rare, diverse and often unlabelled; distinguish point, contextual and collective anomalies.
- Use robust statistics, Mahalanobis distance, LOF, Isolation Forest, One-Class SVM or autoencoders.
- Isolation Forest is a fast, strong default; LOF handles varying densities.
- Evaluate with precision at k and expert review; anomalies are not automatically wrongdoing.
:::

=== POST ===
slug: ensemble-stacking-and-blending
title: Ensemble Learning III: Voting, Stacking and Blending
category: machine-learning
level: Intermediate
tags: ensembles, stacking, blending, voting, meta-learning
summary: Different models make different mistakes. We combine them with hard and soft voting, weighted averaging, and stacking with a meta-learner trained on out-of-fold predictions — and discuss when the extra complexity pays off.
---
We have met two ensemble families: **bagging** averages many copies of one high-variance model, and **boosting** builds a sequence of weak learners. A third family combines **different kinds of models**: a gradient-boosted tree ensemble, a regularised logistic regression, a k-NN model and a neural network each capture different aspects of the data. Combining them can outperform any single model — the approach behind many competition-winning solutions.

## Why diverse models help

Recall the ensemble variance formula: averaging reduces error most when models' errors are **weakly correlated**. Models from different families make different kinds of mistakes: trees capture interactions but produce step-shaped predictions; linear models extrapolate smoothly but miss interactions; k-NN captures local structure. Diversity is the key ingredient.

## Voting and averaging

- **Hard voting** — majority vote of predicted classes.
- **Soft voting** — average predicted probabilities, then take the argmax. Usually better, because confident models get more say. Requires reasonably calibrated probabilities.
- **Weighted averaging** — weights chosen on validation data (e.g. by a simple search or by minimising validation log-loss).

For regression, average the predictions (possibly weighted).

## Stacking

**Stacked generalisation** (Wolpert, 1992) learns *how* to combine models. A **meta-learner** is trained on the base models' predictions.

The critical detail is how to generate training data for the meta-learner:

1. Split the training data into $K$ folds.
2. For each base model and each fold, train on the other $K - 1$ folds and predict the held-out fold. This yields **out-of-fold (OOF)** predictions for every training example — predictions made by models that never saw that example.
3. Train the meta-learner on the OOF predictions (optionally plus original features).
4. Refit each base model on the full training data to produce test-time inputs for the meta-learner.

:::warning
If you train the meta-learner on predictions from base models that were fitted on the **same** examples, those predictions are overconfident (base models have memorised the training labels). The meta-learner learns to trust the most overfit model, and the stack performs badly on new data. Out-of-fold predictions are essential.
:::

The meta-learner should usually be **simple** — logistic or ridge regression with non-negative weights — to avoid overfitting the meta-level.

## Blending

**Blending** is a simpler variant: hold out a validation set, train base models on the rest, and train the meta-learner on predictions for the holdout. It is easier to implement and avoids leakage, but uses less data for each stage.

## Implementation

```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import (StackingClassifier, VotingClassifier, RandomForestClassifier,
                              HistGradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
cv = StratifiedKFold(5, shuffle=True, random_state=0)

base = [
    ("lr",  make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))),
    ("svm", make_pipeline(StandardScaler(), SVC(probability=True, random_state=0))),
    ("knn", make_pipeline(StandardScaler(), KNeighborsClassifier(15))),
    ("rf",  RandomForestClassifier(n_estimators=300, random_state=0)),
    ("gb",  HistGradientBoostingClassifier(random_state=0)),
]
for name, m in base:
    print(f"{name:<6} {cross_val_score(m, X, y, cv=cv, scoring='roc_auc').mean():.4f}")

voting = VotingClassifier(base, voting="soft")
stack = StackingClassifier(base, final_estimator=LogisticRegression(max_iter=2000),
                           cv=5, stack_method="predict_proba", passthrough=False)
print(f"soft voting {cross_val_score(voting, X, y, cv=cv, scoring='roc_auc').mean():.4f}")
print(f"stacking    {cross_val_score(stack, X, y, cv=cv, scoring='roc_auc').mean():.4f}")
```

Scikit-learn's `StackingClassifier` generates out-of-fold predictions internally using its `cv` argument. Note that we evaluate the whole stack with an **outer** cross-validation — the same nested principle as for hyperparameter tuning.

## Checking diversity

Before stacking, inspect the **correlation of OOF predictions** between base models. If two models' predictions correlate at 0.99, keeping both adds cost but little value. Aim for strong individual models that disagree on different examples.

## Costs and trade-offs

| Benefit | Cost |
|---|---|
| Often 1–3% improvement in competition-style metrics | Many models to train, serve and monitor |
| Robustness: one model's failure mode is diluted | Higher latency and memory at inference |
| Uses complementary strengths | Harder to explain and debug |

:::tip
In production, the extra complexity of a large stack is often not worth a small gain. A pragmatic alternative is to average two or three strong, diverse models, or to **distil** the ensemble into a single model trained to mimic its predictions. Choose based on the real value of the accuracy gain versus the operational burden.
:::

:::exercise
1. Generate OOF predictions manually with `cross_val_predict` for three base models and fit a logistic regression meta-learner. Compare with `StackingClassifier`.
2. Demonstrate the leakage problem: train the meta-learner on in-sample predictions and compare test performance.
3. Compute the correlation matrix of base-model OOF probabilities. Remove the most redundant model and re-evaluate the stack.
:::

:::takeaway
- Combining diverse model families reduces error when their mistakes are weakly correlated.
- Soft voting averages probabilities; weighted averaging tunes contributions.
- Stacking trains a simple meta-learner on out-of-fold predictions — never in-sample predictions.
- Weigh accuracy gains against operational complexity; consider distillation.
:::
