=== POST ===
slug: bias-variance-tradeoff
title: The Bias–Variance Trade-off: Derivation and Intuition
category: machine-learning
level: Intermediate
tags: bias, variance, generalization, model complexity, theory
summary: Why do simple models underfit and complex models overfit? We derive the bias–variance decomposition of expected squared error, visualise it, and discuss how modern deep learning complicates the classical picture.
---
Every modelling decision — the degree of a polynomial, the depth of a tree, the strength of regularisation, the number of neighbours in k-NN — moves you along a single axis of trade-offs. On one side lies **bias**: systematic error from a model too simple to capture the truth. On the other lies **variance**: error from a model so flexible that it chases noise in the particular training set. Today we derive this decomposition precisely.

## The setting

Assume data is generated as $y = f(\mathbf{x}) + \epsilon$, where $f$ is the true function and $\epsilon$ is noise with $\mathbb{E}[\epsilon] = 0$ and $\text{Var}(\epsilon) = \sigma^2$. We draw a training set $\mathcal{D}$, fit a model $\hat{f}_{\mathcal{D}}$, and evaluate at a fixed test point $\mathbf{x}$.

The key conceptual step: **the training set is random**. Draw a different sample and you get a different fitted model. So $\hat{f}_{\mathcal{D}}(\mathbf{x})$ is a random variable. Let $\bar{f}(\mathbf{x}) = \mathbb{E}_{\mathcal{D}}[\hat{f}_{\mathcal{D}}(\mathbf{x})]$ be the average prediction over all possible training sets.

## The decomposition

The expected squared error at $\mathbf{x}$, averaging over training sets and noise, is

$$
\mathbb{E}\big[(y - \hat{f}_{\mathcal{D}}(\mathbf{x}))^2\big] = \underbrace{\big(f(\mathbf{x}) - \bar{f}(\mathbf{x})\big)^2}_{\text{Bias}^2} + \underbrace{\mathbb{E}_{\mathcal{D}}\big[(\hat{f}_{\mathcal{D}}(\mathbf{x}) - \bar{f}(\mathbf{x}))^2\big]}_{\text{Variance}} + \underbrace{\sigma^2}_{\text{Irreducible noise}}
$$

*Derivation sketch.* Write $y - \hat{f} = (f - \bar{f}) + (\bar{f} - \hat{f}) + \epsilon$. Square and take expectations. The cross terms vanish because $\mathbb{E}[\epsilon] = 0$, $\epsilon$ is independent of $\mathcal{D}$, and $\mathbb{E}_{\mathcal{D}}[\bar{f} - \hat{f}] = 0$. $\blacksquare$

- **Bias** measures how far the *average* model is from the truth — a property of the model family's assumptions.
- **Variance** measures how much the model *fluctuates* across training sets — sensitivity to the particular sample.
- **Noise** is the floor that no model can beat.

## The archery analogy

Imagine many archers, each trained on a different dataset, shooting at a target:

| | Low variance | High variance |
|---|---|---|
| **Low bias** | Tight cluster on the bullseye (ideal) | Scattered around the bullseye |
| **High bias** | Tight cluster, off-centre | Scattered and off-centre |

## How complexity moves the balance

| Model / knob | Increase complexity by… | Effect |
|---|---|---|
| Polynomial regression | higher degree | bias ↓, variance ↑ |
| k-NN | smaller $k$ | bias ↓, variance ↑ |
| Decision tree | greater depth | bias ↓, variance ↑ |
| Ridge / lasso | smaller $\lambda$ | bias ↓, variance ↑ |
| Neural network | more parameters, fewer regularisers | (usually) bias ↓, variance ↑ |

Total error is minimised where the sum of bias² and variance is smallest — the bottom of the familiar U-shaped test-error curve.

## Estimating bias and variance by simulation

```python
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

rng = np.random.default_rng(0)
f = lambda x: np.sin(2 * np.pi * x)
x_test = np.linspace(0.05, 0.95, 50)
sigma = 0.3

for degree in [1, 3, 9]:
    preds = []
    for trial in range(300):                              # 300 independent training sets
        x = rng.random(20); y = f(x) + rng.normal(0, sigma, 20)
        m = make_pipeline(PolynomialFeatures(degree), LinearRegression()).fit(x[:, None], y)
        preds.append(m.predict(x_test[:, None]))
    preds = np.array(preds)
    bias2 = ((preds.mean(0) - f(x_test)) ** 2).mean()
    var = preds.var(0).mean()
    print(f"degree {degree}: bias^2={bias2:.3f}  variance={var:.3f}  "
          f"total={bias2 + var + sigma**2:.3f}")
```

Degree 1 shows high bias and low variance; degree 9 shows the reverse; degree 3 balances them.

## Reducing each component

**To reduce bias:** use a more flexible model, add informative features, reduce regularisation, train longer.

**To reduce variance:** collect more data, regularise, simplify the model, use feature selection, apply **bagging/ensembling** (averaging many high-variance models — the principle behind random forests), use data augmentation, stop early.

:::note
**More data reduces variance but not bias.** If a linear model cannot represent a curved relationship, a million examples will not help. Plotting learning curves (next lecture) tells you which problem you have — and therefore whether collecting more data is worth the cost.
:::

## The modern twist: double descent

Classical theory predicts that once a model is complex enough to fit the training data perfectly (the **interpolation threshold**), test error should explode. Yet very large neural networks interpolate their training data and still generalise well. Researchers observed **double descent**: as complexity increases past the interpolation threshold, test error first peaks and then *decreases again*. Explanations involve implicit regularisation by the optimiser — among the many solutions that fit the data, gradient descent tends to find smooth, low-norm ones. The bias–variance decomposition is still mathematically true; what changes is how variance behaves for heavily over-parameterised models. We examine this in the deep-learning track.

:::exercise
1. Complete the derivation of the decomposition, showing each cross term is zero.
2. Modify the simulation for k-NN regression with $k = 1, 5, 20$. Plot bias² and variance against $k$.
3. Explain why bagging reduces variance but not bias, using the variance of an average of correlated variables.
:::

:::takeaway
- Expected error = bias² + variance + irreducible noise.
- Bias comes from restrictive assumptions; variance from sensitivity to the training sample.
- Complexity trades one for the other; choose it by validation.
- More data cures variance, not bias; over-parameterised models exhibit double descent.
:::

=== POST ===
slug: overfitting-underfitting-learning-curves
title: Overfitting and Underfitting: Diagnosis with Learning Curves
category: machine-learning
level: Beginner
tags: overfitting, underfitting, learning curves, validation curves, diagnosis
summary: Before fixing a model you must diagnose it. We learn to read learning curves and validation curves, recognise high bias and high variance, and choose the right remedy instead of guessing.
---
Andrew Ng often tells students that the most valuable ML skill is knowing **what to try next**. When a model disappoints, should you collect more data, add features, use a bigger model, or regularise more? Each option costs time and money, and choosing wrongly can waste months. Learning curves are the diagnostic instrument that tells you which option will help.

## Symptoms

| Diagnosis | Training error | Validation error | Gap |
|---|---|---|---|
| **Underfitting** (high bias) | High | High | Small |
| **Overfitting** (high variance) | Low | High | Large |
| Good fit | Low | Low (close to target) | Small |

You need a **target** to judge "high": human-level performance, a published result, or the business requirement. The gap between training error and the target is **avoidable bias**; the gap between training and validation error is **variance**.

## Learning curves

A **learning curve** plots training and validation error against the **number of training examples**.

- **High bias**: both curves converge quickly to a *high* error plateau. Adding data does not help — the curves are already flat and close together.
- **High variance**: training error is low, validation error is much higher, and the gap narrows slowly as data increases. More data *will* help.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.datasets import load_digits
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

X, y = load_digits(return_X_y=True)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, (name, model) in zip(axes, [("Gaussian Naive Bayes (high bias)", GaussianNB()),
                                    ("RBF SVM, gamma=0.01 (high variance)", SVC(gamma=0.01))]):
    sizes, tr, va = learning_curve(model, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 8))
    ax.plot(sizes, 1 - tr.mean(1), "o-", label="training error")
    ax.plot(sizes, 1 - va.mean(1), "o-", label="validation error")
    ax.set_title(name); ax.set_xlabel("training examples"); ax.legend()
plt.tight_layout(); plt.show()
```

## Validation curves

A **validation curve** plots training and validation error against a **hyperparameter** that controls complexity (tree depth, regularisation strength, polynomial degree).

- On the low-complexity side, both errors are high → underfitting.
- On the high-complexity side, training error keeps falling while validation error rises → overfitting.
- Choose the value at the minimum of validation error.

```python
from sklearn.model_selection import validation_curve
from sklearn.tree import DecisionTreeClassifier

depths = range(1, 21)
tr, va = validation_curve(DecisionTreeClassifier(random_state=0), X, y,
                          param_name="max_depth", param_range=depths, cv=5)
best = list(depths)[va.mean(1).argmax()]
print("best depth by validation:", best)
```

## Choosing the remedy

| If you have… | Try | Do NOT waste time on |
|---|---|---|
| High bias | Bigger/more flexible model; more or better features; less regularisation; train longer; better architecture | Collecting more data |
| High variance | More data; data augmentation; regularisation; dropout; simpler model; feature selection; ensembling; early stopping | Bigger model (usually) |
| Both | Address bias first, then variance | — |
| Train/validation mismatch (different distributions) | Make training data more like deployment data; domain adaptation | Tuning on the wrong distribution |

:::note
In deep learning, the classical trade-off is looser. You can often reduce bias with a bigger network **and** reduce variance with more data and regularisation simultaneously, which is why "bigger model + more data + good regularisation" has been such a successful recipe. The diagnostic principle still holds: measure first, then act.
:::

## Loss curves during training

For iterative learners (neural networks, boosting), plot training and validation loss against **epochs**:

- Both decreasing → keep training.
- Validation loss starts rising while training loss keeps falling → overfitting begins; **early stopping** saves the best checkpoint.
- Training loss not decreasing → optimisation problem (learning rate, bugs, bad initialisation), not a generalisation problem.
- Validation loss *lower* than training loss → possible if training uses dropout/augmentation, or a sign of leakage or an unrepresentative split.

## A diagnostic checklist

1. Can the model **overfit a tiny subset** (e.g. 50 examples) to near-zero training error? If not, there is a bug or the model is too weak.
2. Compare training error with the target → avoidable bias.
3. Compare validation with training error → variance.
4. Plot learning curves → will more data help?
5. Perform **error analysis** on validation mistakes → which categories of errors dominate?

:::exercise
1. Generate learning curves for logistic regression and for a random forest on a dataset of your choice. Diagnose each.
2. Train a decision tree with depths 1–25 and plot the validation curve. Where does overfitting begin?
3. A model has 2% training error, 15% validation error and human-level error of 1%. What is your next step and why?
:::

:::takeaway
- Underfitting: high training and validation error. Overfitting: low training error, large gap.
- Learning curves reveal whether more data will help; validation curves reveal the right complexity.
- Match the remedy to the diagnosis — don't collect data to fix bias.
- Always verify the model can overfit a small subset before diagnosing generalisation.
:::

=== POST ===
slug: train-validation-test-cross-validation
title: Train, Validation and Test Splits — and Cross-Validation Done Right
category: machine-learning
level: Beginner
tags: cross-validation, validation, data splitting, leakage, nested cv
summary: Trustworthy evaluation starts with correct data splits. We cover hold-out validation, k-fold, stratified, group and time-series cross-validation, nested CV for tuning, and the leakage traps that invalidate results.
---
A model is only as trustworthy as its evaluation. I have reviewed student projects with 99% accuracy that turned out to be worthless because the test data had leaked into training. Today we learn how to split data so that the numbers you report actually predict real-world performance.

## Three sets, three purposes

| Set | Used for | Touched how often |
|---|---|---|
| Training | Fitting model parameters | Constantly |
| Validation (development) | Choosing hyperparameters, features, models | Many times |
| Test | Final unbiased estimate of performance | **Once** |

Every time you make a decision based on a dataset, you fit to it a little. The validation set absorbs that bias, protecting the test set. If you tune on the test set, your reported score becomes optimistically biased.

Typical splits: 60/20/20 or 80/10/10 for moderate data; for very large datasets (millions of examples), validation and test sets of a few thousand to tens of thousands may be enough.

## k-fold cross-validation

With limited data, a single validation split is noisy and wasteful. **k-fold cross-validation**:

1. Split the training data into $k$ equal folds (commonly 5 or 10).
2. For each fold $i$: train on the other $k - 1$ folds, validate on fold $i$.
3. Report the mean (and standard deviation) of the $k$ validation scores.

Every example is used for validation exactly once. The cost is $k$ training runs. **Leave-one-out** CV ($k = n$) has low bias but high variance and high cost; 5 or 10 folds is the usual compromise.

## Choosing the right splitter

- **Stratified k-fold** — preserves class proportions in each fold. Essential for imbalanced classification.
- **Group k-fold** — keeps all records of a group (patient, user, household, camp) in the same fold. Without it, the model can "recognise" individuals and look better than it is.
- **Time-series split** — train on the past, validate on the future, with an expanding or sliding window. Random shuffling of time series leaks future information.
- **Repeated k-fold** — repeat with different shuffles to reduce the variance of the estimate.

```python
import numpy as np
from sklearn.model_selection import (KFold, StratifiedKFold, GroupKFold,
                                     TimeSeriesSplit, cross_val_score)
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=600, weights=[0.9, 0.1], random_state=0)
groups = np.repeat(np.arange(100), 6)            # e.g. 6 records per patient

model = LogisticRegression(max_iter=1000)
for name, cv in [("KFold", KFold(5, shuffle=True, random_state=0)),
                 ("Stratified", StratifiedKFold(5, shuffle=True, random_state=0)),
                 ("Group", GroupKFold(5)),
                 ("TimeSeries", TimeSeriesSplit(5))]:
    s = cross_val_score(model, X, y, cv=cv, groups=groups if name == "Group" else None, scoring="f1")
    print(f"{name:<11} F1 = {s.mean():.3f} ± {s.std():.3f}")
```

## Data leakage: the silent killer

**Leakage** occurs when information unavailable at prediction time influences training or evaluation. Common forms:

1. **Preprocessing leakage** — fitting a scaler, imputer, PCA or feature selector on the full dataset before splitting. The validation folds influence the transformation.
2. **Target leakage** — a feature derived from the outcome ("discharge date" when predicting hospital admission; "account closed flag" when predicting churn).
3. **Temporal leakage** — using future data to predict the past.
4. **Group leakage** — the same entity in train and test.
5. **Duplicate leakage** — near-identical examples across splits (common in scraped image and text datasets).

:::warning
The cure for preprocessing leakage is simple: put **every** data-dependent transformation inside a `Pipeline`, and pass the pipeline to cross-validation. Then each fold fits the preprocessing on its own training portion only.
:::

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

# Leakage demo: pure noise features, random labels -> true accuracy is 50%
rng = np.random.default_rng(0)
Xn, yn = rng.normal(size=(100, 10_000)), rng.integers(0, 2, 100)

# WRONG: select features on all data, then cross-validate
X_sel = SelectKBest(f_classif, k=20).fit_transform(Xn, yn)
print("leaky CV accuracy:", cross_val_score(LogisticRegression(), X_sel, yn, cv=5).mean())

# RIGHT: selection inside the pipeline
pipe = make_pipeline(SelectKBest(f_classif, k=20), LogisticRegression())
print("honest CV accuracy:", cross_val_score(pipe, Xn, yn, cv=5).mean())
```

The leaky version reports impressive accuracy on pure noise; the honest version reports about 50%. This exact mistake has appeared in published research.

## Nested cross-validation

If you tune hyperparameters with cross-validation and then report the best CV score, that score is optimistically biased — you selected the maximum of noisy estimates. **Nested CV** fixes this:

- The **inner loop** tunes hyperparameters (e.g. `GridSearchCV`).
- The **outer loop** evaluates the entire tuning procedure on held-out folds.

```python
from sklearn.model_selection import GridSearchCV
inner = GridSearchCV(make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
                     {"logisticregression__C": [0.01, 0.1, 1, 10]}, cv=3)
outer_scores = cross_val_score(inner, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=1))
print("nested CV estimate:", outer_scores.mean().round(3))
```

## Practical guidance

- Decide the split strategy by asking: **how will the model be used?** If it will predict for new patients, split by patient. If it will predict next month, split by time.
- Keep a final test set locked away; use CV within the training portion.
- Report mean ± standard deviation across folds.
- After choosing everything, retrain on all training data before the final test.

:::exercise
1. Run the leakage demo with `k = 5`, `20`, `200` selected features. How does the leaky score change?
2. For a dataset of hospital visits with multiple visits per patient, which splitter would you use? Why?
3. Implement a time-series split for daily sales data and compare it with a random k-fold. Which gives the more realistic estimate?
:::

:::takeaway
- Train fits, validation selects, test reports — once.
- Use stratified, group or time-series splits to mirror deployment.
- Put all preprocessing inside pipelines to prevent leakage.
- Use nested CV to evaluate a tuning procedure honestly.
:::

=== POST ===
slug: classification-metrics
title: Evaluation Metrics for Classification: Accuracy, Precision, Recall and F1
category: machine-learning
level: Beginner
tags: metrics, precision, recall, f1, confusion matrix, evaluation
summary: Accuracy can be dangerously misleading. We build the confusion matrix, define precision, recall, specificity, F-scores and balanced accuracy, and learn to choose metrics from the costs of errors.
---
A disease affects 1% of patients. A model that always predicts "healthy" achieves **99% accuracy** — and is completely useless. This is why professionals never report accuracy alone. Today we learn the vocabulary of classification metrics and, more importantly, how to choose the metric that matches the real costs of mistakes.

## The confusion matrix

For binary classification with a "positive" class (the thing we are trying to detect):

| | Predicted positive | Predicted negative |
|---|---|---|
| **Actually positive** | True Positive (TP) | False Negative (FN) |
| **Actually negative** | False Positive (FP) | True Negative (TN) |

Every metric is a function of these four numbers.

## The core metrics

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

$$
\text{Precision} = \frac{TP}{TP + FP} \qquad \text{"When the model says positive, how often is it right?"}
$$

$$
\text{Recall (Sensitivity, TPR)} = \frac{TP}{TP + FN} \qquad \text{"Of all real positives, how many did we find?"}
$$

$$
\text{Specificity (TNR)} = \frac{TN}{TN + FP} \qquad \text{False Positive Rate} = 1 - \text{Specificity}
$$

$$
F_1 = 2\cdot\frac{\text{Precision}\cdot\text{Recall}}{\text{Precision} + \text{Recall}}
$$

$F_1$ is the **harmonic mean** of precision and recall. Unlike the arithmetic mean, it is low if *either* component is low, so a model cannot hide terrible recall behind high precision.

The generalised $F_\beta$ weights recall $\beta$ times as important as precision:

$$
F_\beta = (1 + \beta^2)\frac{\text{Precision}\cdot\text{Recall}}{\beta^2\,\text{Precision} + \text{Recall}}
$$

Use $F_2$ when missing positives is worse; $F_{0.5}$ when false alarms are worse.

## Precision versus recall: choosing by consequences

| Application | Costlier error | Emphasise |
|---|---|---|
| Cancer screening | Missing a cancer (FN) | Recall |
| Spam filtering | Losing a real email (FP) | Precision |
| Fraud detection | Depends: investigation cost vs fraud loss | Precision at a fixed recall, or cost-based |
| Identifying vulnerable families for urgent follow-up | Missing a family in need (FN) | Recall, with capacity constraints on FP |
| Automated content removal | Removing legitimate speech (FP) | Precision |

:::note
Precision and recall trade off through the **decision threshold**. Lowering the threshold catches more positives (recall ↑) but raises false alarms (precision ↓). A single metric value describes *one* threshold; the curves in the next lecture describe *all* thresholds.
:::

## Metrics for imbalanced data

- **Balanced accuracy** = (Recall + Specificity)/2 — the average recall over classes. The "always healthy" model scores 0.5.
- **Matthews Correlation Coefficient (MCC)**:
$$
\text{MCC} = \frac{TP\cdot TN - FP\cdot FN}{\sqrt{(TP + FP)(TP + FN)(TN + FP)(TN + FN)}}
$$
ranges from −1 to +1, uses all four cells and is informative even with severe imbalance.
- **Cohen's kappa** — agreement corrected for chance.

## Multiclass averaging

For $K$ classes, compute per-class precision/recall/F1 (each class vs the rest), then average:

- **Macro**: unweighted mean over classes — every class matters equally; exposes poor minority-class performance.
- **Weighted**: mean weighted by class frequency.
- **Micro**: pool all TP/FP/FN globally — for single-label multiclass it equals accuracy.

## Computing everything

```python
import numpy as np
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, recall_score,
                             f1_score, balanced_accuracy_score, matthews_corrcoef,
                             classification_report)

rng = np.random.default_rng(0)
y_true = (rng.random(1000) < 0.05).astype(int)            # 5% positives
scores = np.where(y_true == 1, rng.beta(5, 2, 1000), rng.beta(2, 5, 1000))
y_pred = (scores >= 0.5).astype(int)

print(confusion_matrix(y_true, y_pred))
print("accuracy          ", round(accuracy_score(y_true, y_pred), 3))
print("precision         ", round(precision_score(y_true, y_pred), 3))
print("recall            ", round(recall_score(y_true, y_pred), 3))
print("F1                ", round(f1_score(y_true, y_pred), 3))
print("balanced accuracy ", round(balanced_accuracy_score(y_true, y_pred), 3))
print("MCC               ", round(matthews_corrcoef(y_true, y_pred), 3))
print("always-negative accuracy:", round(accuracy_score(y_true, np.zeros_like(y_true)), 3))
print(classification_report(y_true, y_pred, digits=3))
```

Notice that the trivial always-negative model has accuracy of about 95% here — higher than you might expect from a real model — while its recall, F1 and MCC are zero.

## Cost-sensitive evaluation

When you know the costs, use them directly. If a false negative costs $c_{FN}$ and a false positive $c_{FP}$:

$$
\text{Expected cost} = c_{FN}\cdot FN + c_{FP}\cdot FP
$$

For a well-calibrated probabilistic classifier, the cost-minimising threshold is

$$
t^* = \frac{c_{FP}}{c_{FP} + c_{FN}}
$$

If missing a positive is nine times worse than a false alarm, predict positive whenever $p \ge 0.1$.

## Beyond aggregate metrics

- **Slice metrics** by subgroup (region, gender, language) to detect unequal performance.
- **Error analysis**: read the actual false positives and negatives.
- **Confidence intervals**: metrics from small test sets are noisy (see the hypothesis-testing lecture).

:::exercise
1. A model has TP = 40, FN = 10, FP = 60, TN = 890. Compute accuracy, precision, recall, F1, specificity and MCC.
2. For the same model, would you deploy it for cancer screening? For automatically blocking financial transactions? Justify.
3. Derive the cost-minimising threshold $t^*$ from expected-cost reasoning.
:::

:::takeaway
- Accuracy misleads on imbalanced data; always inspect the confusion matrix.
- Precision = trust in positive predictions; recall = coverage of real positives; F1 balances them.
- Balanced accuracy and MCC are robust summaries for imbalance; use macro averaging for multiclass.
- Choose metrics and thresholds from the real costs of errors.
:::

=== POST ===
slug: roc-auc-and-precision-recall-curves
title: ROC Curves, AUC and Precision–Recall Curves
category: machine-learning
level: Intermediate
tags: roc, auc, precision-recall, thresholds, calibration
summary: A classifier's score supports many thresholds. ROC and precision–recall curves summarise all of them. We construct both, interpret AUC probabilistically, and learn when each curve tells the truth.
---
Most classifiers output a **score** or probability, and we choose a threshold to turn it into a decision. Different thresholds give different confusion matrices. Rather than evaluating one arbitrary threshold, we can evaluate the **ranking quality** of the scores across all thresholds. That is what ROC and precision–recall curves do.

## The ROC curve

The **Receiver Operating Characteristic** curve (originally from radar signal detection in World War II) plots, for every threshold:

- the **True Positive Rate** (recall) $TPR = \frac{TP}{TP + FN}$ on the y-axis,
- against the **False Positive Rate** $FPR = \frac{FP}{FP + TN}$ on the x-axis.

As the threshold decreases from $+\infty$ to $-\infty$, the curve moves from $(0, 0)$ (predict nothing positive) to $(1, 1)$ (predict everything positive).

- A **perfect** classifier passes through $(0, 1)$.
- A **random** classifier follows the diagonal $TPR = FPR$.
- A curve below the diagonal indicates a classifier whose scores are inverted.

## AUC: area under the ROC curve

The **AUC** (or ROC-AUC) summarises the curve with one number between 0 and 1. It has a beautiful probabilistic interpretation:

$$
\text{AUC} = P\big(s(\mathbf{x}^+) > s(\mathbf{x}^-)\big)
$$

— the probability that a randomly chosen positive example receives a higher score than a randomly chosen negative one. It is equivalent to the normalised **Mann–Whitney U** statistic. AUC = 0.5 means random ranking; AUC = 1.0 means perfect ranking.

Properties:

- **Threshold-independent** — evaluates ranking, not a particular decision.
- **Invariant to class imbalance** in the sense that TPR and FPR are each computed within one class.
- **Invariant to monotone transformations of scores** — it ignores calibration entirely.

## Why ROC can mislead under heavy imbalance

Suppose 10 positives and 100,000 negatives. An FPR of 1% sounds tiny — but it means 1,000 false positives against at most 10 true positives. Precision is below 1%. The ROC curve looks excellent while the model is operationally useless, because FPR's denominator (all negatives) is huge.

## The precision–recall curve

The **PR curve** plots **precision** against **recall** for every threshold. Because precision's denominator includes false positives directly, the PR curve is sensitive to imbalance and shows the practical cost of catching more positives.

- The baseline for a random classifier is a **horizontal line at the positive prevalence** $\pi = P(y = 1)$ — not 0.5.
- The area under the PR curve is summarised as **Average Precision (AP)**.

:::tip
**Rule of thumb:** use ROC-AUC when classes are reasonably balanced or when you care about ranking across both classes equally. Use the **PR curve and AP** when positives are rare and you care about the quality of positive predictions (fraud, rare disease, information retrieval). Report the prevalence alongside AP so readers know the baseline.
:::

## Computing and plotting

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score

X, y = make_classification(n_samples=20_000, n_features=20, weights=[0.98, 0.02],
                           class_sep=0.8, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y, test_size=0.3, random_state=0)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
for name, m in [("Logistic", LogisticRegression(max_iter=1000)),
                ("Random forest", RandomForestClassifier(n_estimators=300, random_state=0))]:
    s = m.fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
    fpr, tpr, _ = roc_curve(y_te, s)
    prec, rec, _ = precision_recall_curve(y_te, s)
    ax1.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_te, s):.3f})")
    ax2.plot(rec, prec, label=f"{name} (AP={average_precision_score(y_te, s):.3f})")
ax1.plot([0, 1], [0, 1], "k--", lw=1); ax1.set_xlabel("FPR"); ax1.set_ylabel("TPR"); ax1.legend()
ax2.axhline(y_te.mean(), color="k", ls="--", lw=1, label="prevalence")
ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision"); ax2.legend()
plt.tight_layout(); plt.show()
```

Run it: both models may have high ROC-AUC, yet their AP values are far lower and more clearly separated — the PR curve reveals the difficulty of the rare class.

## Choosing an operating point

A curve is not a decision. To deploy, pick a threshold using:

- **A constraint**: "recall must be at least 90%" → choose the threshold achieving that with the highest precision.
- **Capacity**: "our team can review 200 cases per day" → choose the threshold that flags 200 cases (precision at top-k).
- **Costs**: minimise expected cost as shown in the previous lecture.
- **Youden's J** statistic $\max(TPR - FPR)$ — a common default in medicine.

Always choose the threshold on the **validation** set, then report test performance at that fixed threshold.

## Calibration is a separate question

A model can rank perfectly (AUC = 1) while its probabilities are badly miscalibrated (predicting 0.6 for every positive and 0.4 for every negative). If decisions depend on the *value* of the probability — expected-cost thresholds, risk communication — check calibration with reliability diagrams and the **Brier score** $\frac{1}{n}\sum_i(p_i - y_i)^2$, and recalibrate with Platt scaling or isotonic regression if needed.

## Multiclass extensions

For $K$ classes, compute one-vs-rest curves per class and average (macro or weighted), or report per-class curves. Scikit-learn's `roc_auc_score(..., multi_class="ovr")` implements this.

:::exercise
1. Prove the probabilistic interpretation of AUC for a finite dataset by counting correctly ordered positive–negative pairs.
2. Using the example, find the threshold achieving 80% recall and report precision there.
3. Construct scores with AUC ≈ 0.95 on a dataset with 0.1% prevalence and show how low precision can be at 90% recall.
:::

:::takeaway
- ROC plots TPR vs FPR across thresholds; AUC is the probability a positive outranks a negative.
- Under heavy imbalance, ROC can look excellent while precision is poor — use PR curves and AP.
- The PR baseline equals the prevalence.
- Curves evaluate ranking; choose an operating threshold by constraints or costs, and check calibration separately.
:::

=== POST ===
slug: regression-metrics
title: Regression Metrics: MSE, RMSE, MAE, R² and Beyond
category: machine-learning
level: Beginner
tags: metrics, regression, rmse, mae, r-squared, mape
summary: How good is a numeric prediction? We compare MSE, RMSE, MAE, R², MAPE and quantile loss, explain how each responds to outliers and scale, and match metrics to real decisions.
---
When predicting numbers — prices, temperatures, travel times, food-aid quantities — the question "how accurate is the model?" has several answers. Different metrics punish different kinds of error. Choosing the wrong one can make a model look good while it fails exactly where it matters.

## Mean Squared Error and RMSE

$$
\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2, \qquad \text{RMSE} = \sqrt{\text{MSE}}
$$

- Squaring punishes large errors heavily: one error of 10 costs as much as 100 errors of 1.
- RMSE is in the **same units** as the target, making it interpretable ("typically off by about 3.2 °C").
- The prediction minimising expected squared error is the **conditional mean** $\mathbb{E}[y \mid \mathbf{x}]$.
- Sensitive to outliers — a few extreme values can dominate.

## Mean Absolute Error

$$
\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|
$$

- Treats all errors linearly; more **robust to outliers**.
- Minimised by the conditional **median**.
- Also in target units; often easier to explain to non-technical stakeholders.

:::note
RMSE ≥ MAE always. The ratio RMSE/MAE tells you about the error distribution: if RMSE is much larger than MAE, a few large errors dominate. That is a prompt to inspect those cases — they may be outliers, data errors or an important subpopulation the model handles badly.
:::

## Coefficient of determination R²

$$
R^2 = 1 - \frac{\sum_i(y_i - \hat{y}_i)^2}{\sum_i(y_i - \bar{y})^2}
$$

$R^2$ compares the model with the trivial predictor that always outputs the mean.

- $R^2 = 1$: perfect predictions.
- $R^2 = 0$: no better than the mean.
- $R^2 < 0$: **worse** than the mean — possible on test data.

Caveats: $R^2$ depends on the variance of the target in the evaluation set (a narrow test range lowers $R^2$ even for a good model), it never decreases when features are added on the training set (use adjusted $R^2$ or validation data), and a high $R^2$ does not mean predictions are accurate enough for the decision at hand.

## Percentage errors

**MAPE** (Mean Absolute Percentage Error):

$$
\text{MAPE} = \frac{100\%}{n}\sum_{i=1}^{n}\left|\frac{y_i - \hat{y}_i}{y_i}\right|
$$

Scale-free and popular in business forecasting, but it explodes when $y_i$ is near zero and penalises over-prediction more than under-prediction. Alternatives: **sMAPE** (symmetric), **WAPE** $\frac{\sum|y_i - \hat{y}_i|}{\sum|y_i|}$ (weighted — robust for intermittent demand), and **MASE** (scaled by a naive forecast's error — recommended for comparing forecasts across series).

## Log-scale errors

For targets spanning orders of magnitude (population, income, counts), use **RMSLE**:

$$
\text{RMSLE} = \sqrt{\frac{1}{n}\sum_i\big(\ln(1 + \hat{y}_i) - \ln(1 + y_i)\big)^2}
$$

It measures *relative* error: predicting 110 for 100 costs about the same as predicting 1,100 for 1,000.

## Quantile (pinball) loss and prediction intervals

Often a single number is not enough — a planner needs to know "how much stock covers 90% of scenarios?". The **quantile loss** for quantile $\tau$ is

$$
L_\tau(y, \hat{y}) = \max\big(\tau(y - \hat{y}),\; (\tau - 1)(y - \hat{y})\big)
$$

Minimising it yields the $\tau$-th conditional quantile. Training models for $\tau = 0.05$ and $0.95$ gives a 90% **prediction interval**, which you evaluate by **coverage** (does 90% of the truth fall inside?) and **width**.

## Comparing metrics on data with outliers

```python
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

rng = np.random.default_rng(0)
y = rng.normal(100, 15, 500)
pred_a = y + rng.normal(0, 5, 500)            # consistently a little off
pred_b = y.copy(); pred_b[:5] += 120          # perfect except five huge errors

for name, p in [("A: small errors everywhere", pred_a), ("B: few huge errors", pred_b)]:
    rmse = mean_squared_error(y, p) ** 0.5
    mae = mean_absolute_error(y, p)
    print(f"{name:<28} RMSE={rmse:6.2f}  MAE={mae:5.2f}  R2={r2_score(y, p):.3f}")
```

RMSE prefers model A; MAE prefers model B. Which is better depends on the application: for aid delivery, five catastrophic under-estimates might mean five communities without supplies — the RMSE view matters. For a recommendation estimate, occasional large misses may be tolerable.

## Choosing a metric

| Situation | Recommended metric |
|---|---|
| Large errors are disproportionately harmful | RMSE / MSE |
| Outliers in target; want typical error | MAE |
| Communicate improvement over the mean baseline | $R^2$ (alongside an absolute metric) |
| Targets span orders of magnitude | RMSLE, or MAE on log-scale |
| Forecasts across many series of different scale | MASE, WAPE |
| Decisions need uncertainty | Quantile loss, interval coverage and width |

:::tip
The **training loss** and the **evaluation metric** need not be the same, but they should be aligned. If you evaluate with MAE, consider training with MAE or Huber loss. If the business cares about asymmetric costs (under-stocking vs over-stocking), use an asymmetric (quantile) loss.
:::

:::exercise
1. Show that the constant $c$ minimising $\sum_i(y_i - c)^2$ is the mean and that minimising $\sum_i|y_i - c|$ gives the median.
2. Construct a test set where a model has $R^2 < 0$.
3. Train `GradientBoostingRegressor(loss="quantile")` for $\tau = 0.1, 0.5, 0.9$ on a dataset and evaluate interval coverage.
:::

:::takeaway
- MSE/RMSE punish large errors (mean-optimal); MAE is robust (median-optimal).
- $R^2$ compares with the mean baseline and can be negative on test data.
- Percentage and log errors handle scale; beware MAPE near zero.
- Quantile loss gives prediction intervals — often more useful than a point estimate.
:::

=== POST ===
slug: k-nearest-neighbors
title: k-Nearest Neighbours: Learning by Similarity
category: machine-learning
level: Beginner
tags: knn, instance-based learning, distance, classification, regression
summary: The simplest learning algorithm stores the data and asks the neighbours. We analyse k-NN's bias–variance behaviour, distance choices, scaling, efficient search structures and its surprising theoretical guarantees.
---
Suppose you move to a new neighbourhood and want to guess whether a house is expensive. You look at the houses nearby. That is **k-nearest neighbours (k-NN)** — perhaps the most intuitive learning algorithm ever devised. It has no training phase at all: it simply remembers the data and makes predictions by consulting similar examples. Despite its simplicity it teaches deep lessons about distance, scale and dimensionality, and it is the ancestor of modern vector-search systems.

## The algorithm

To predict for a query $\mathbf{x}$:

1. Compute the distance from $\mathbf{x}$ to every training point.
2. Select the $k$ closest points $\mathcal{N}_k(\mathbf{x})$.
3. **Classification**: majority vote among their labels. **Regression**: average their targets.

$$
\hat{y}(\mathbf{x}) = \frac{1}{k}\sum_{i \in \mathcal{N}_k(\mathbf{x})} y_i
$$

**Weighted k-NN** gives closer neighbours more influence, e.g. weights $w_i = 1/d(\mathbf{x}, \mathbf{x}_i)$.

k-NN is **non-parametric** (the model grows with the data) and **instance-based** or **lazy** (all computation happens at prediction time).

## The role of k

- $k = 1$: the decision boundary follows every training point, forming a **Voronoi tessellation**. Training error is zero; variance is very high; noisy labels create islands of misclassification.
- Large $k$: smoother boundaries, lower variance, higher bias. With $k = n$ every prediction is the majority class.

Choose $k$ by cross-validation; odd values avoid ties in binary classification. A common starting point is $k \approx \sqrt{n}$, but always validate.

## A surprising guarantee

Cover and Hart (1967) proved that as the number of training examples grows to infinity, the error rate of the **1-NN** classifier is at most **twice the Bayes error** (the best possible error). With $k \to \infty$ and $k/n \to 0$, k-NN is **consistent**: its error converges to the Bayes error. So this trivially simple method is, asymptotically, nearly optimal — given enough data. The catch is the phrase "enough data", which in high dimensions can mean astronomically much.

## Distance and scaling

k-NN is only as good as its notion of similarity.

- **Euclidean** distance is the default for continuous features.
- **Manhattan** distance is more robust to single large coordinate differences.
- **Cosine** distance suits text and embeddings.
- **Hamming** or **Gower** distances handle categorical or mixed data.

:::warning
**Always scale features.** If one feature ranges from 0 to 100,000 (income) and another from 0 to 1 (a ratio), the distance is determined almost entirely by income. Standardise or min–max scale features, inside a pipeline, before using k-NN.
:::

Irrelevant features also hurt: each adds noise to every distance. Feature selection or learned embeddings help. **Metric learning** methods (such as Large Margin Nearest Neighbours or Siamese networks) learn a distance in which same-class points are close.

## Implementation from scratch

```python
import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k=5):
        self.k = k
    def fit(self, X, y):
        self.X, self.y = np.asarray(X, float), np.asarray(y)
        return self
    def predict(self, Xq):
        Xq = np.asarray(Xq, float)
        # squared Euclidean distances via broadcasting: (q, n)
        d2 = ((Xq[:, None, :] - self.X[None, :, :]) ** 2).sum(-1)
        idx = np.argpartition(d2, self.k, axis=1)[:, :self.k]
        return np.array([Counter(self.y[row]).most_common(1)[0][0] for row in idx])

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X, y = load_wine(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
for scaled in [False, True]:
    A, B = (X_tr, X_te)
    if scaled:
        sc = StandardScaler().fit(X_tr); A, B = sc.transform(X_tr), sc.transform(X_te)
    acc = (KNN(5).fit(A, y_tr).predict(B) == y_te).mean()
    print(f"scaled={scaled}: accuracy={acc:.3f}")
```

On the wine dataset, scaling typically raises accuracy from around 70% to over 95% — a dramatic demonstration of why scaling matters.

## Computational cost and fast search

Naive prediction costs $O(nd)$ per query — fine for thousands of points, slow for millions. Solutions:

- **KD-trees** and **ball trees** partition space for exact search; efficient in low dimensions (roughly $d < 20$) but degrade in high dimensions.
- **Approximate Nearest Neighbour (ANN)** methods trade a little accuracy for massive speed: locality-sensitive hashing, product quantisation, and graph-based indexes such as **HNSW**. Libraries like FAISS and vector databases implement them.

:::note
Modern semantic search and retrieval-augmented generation are k-NN at heart: text is embedded into vectors by a neural network, and the system retrieves the nearest neighbours of the query embedding with an ANN index. The 1950s algorithm lives on at web scale.
:::

## Strengths and weaknesses

**Strengths:** no training; naturally multiclass; adapts to complex boundaries; easy to update (just add points); predictions are explainable by showing the neighbours.

**Weaknesses:** slow and memory-hungry at prediction time; sensitive to scaling and irrelevant features; suffers badly from the curse of dimensionality; needs a meaningful distance.

:::exercise
1. Plot k-NN decision boundaries on a 2-D dataset (e.g. `make_moons`) for $k = 1, 15, 75$.
2. Use cross-validation to choose $k$ and the weighting scheme (uniform vs distance) on the wine dataset.
3. Measure prediction time for brute-force search and a KD-tree as dataset size grows from $10^3$ to $10^5$ in 3 and in 50 dimensions.
:::

:::takeaway
- k-NN predicts by majority vote or averaging over the $k$ nearest training points.
- Small $k$ → low bias, high variance; large $k$ → the reverse; choose $k$ by CV.
- Asymptotically, 1-NN error is at most twice the Bayes error.
- Scale features and choose distances carefully; use trees or ANN indexes for speed.
:::

=== POST ===
slug: naive-bayes-classifiers
title: Naive Bayes Classifiers: Simple, Fast and Surprisingly Strong
category: machine-learning
level: Beginner
tags: naive bayes, generative models, text classification, probability
summary: Naive Bayes applies Bayes' theorem with a bold independence assumption. We derive Gaussian, multinomial and Bernoulli variants, explain why it works despite being "naive", and build a text classifier.
---
Naive Bayes is one of the oldest machine-learning classifiers still in daily use. Early spam filters were built on it, it remains a strong baseline for text classification, and it trains in a single pass over the data. It is also our first **generative** classifier — it models how the data is generated in each class, rather than modelling the decision boundary directly.

## From Bayes' theorem to a classifier

For a class $c$ and features $\mathbf{x} = (x_1, \dots, x_d)$:

$$
P(c \mid \mathbf{x}) = \frac{P(\mathbf{x} \mid c)\,P(c)}{P(\mathbf{x})} \propto P(c)\,P(\mathbf{x} \mid c)
$$

Estimating the full joint likelihood $P(\mathbf{x} \mid c)$ is hopeless in high dimensions — with 10,000 binary word features there are $2^{10{,}000}$ combinations. The **naive assumption** is that features are **conditionally independent given the class**:

$$
P(\mathbf{x} \mid c) = \prod_{j=1}^{d} P(x_j \mid c)
$$

The classifier predicts

$$
\hat{c} = \arg\max_c \left[\ln P(c) + \sum_{j=1}^{d}\ln P(x_j \mid c)\right]
$$

We sum logarithms rather than multiply probabilities to avoid numerical underflow.

## Three common variants

| Variant | Feature type | $P(x_j \mid c)$ | Typical use |
|---|---|---|---|
| Gaussian NB | Continuous | $\mathcal{N}(x_j \mid \mu_{jc}, \sigma_{jc}^2)$ | Sensor data, simple numeric features |
| Multinomial NB | Counts | $\propto \theta_{jc}^{x_j}$ (word frequencies) | Document classification |
| Bernoulli NB | Binary | $\theta_{jc}^{x_j}(1 - \theta_{jc})^{1 - x_j}$ | Presence/absence of words; short texts |

Training is just counting (or computing means and variances) per class — one pass over the data, $O(nd)$.

## Laplace smoothing

If the word "lottery" never appeared in a legitimate email during training, $P(\text{lottery} \mid \text{ham}) = 0$, and a single occurrence would force the ham probability to zero regardless of all other evidence. **Additive (Laplace) smoothing** fixes this:

$$
\hat{\theta}_{jc} = \frac{\text{count}(j, c) + \alpha}{\sum_{j'}\text{count}(j', c) + \alpha|V|}
$$

with $\alpha = 1$ (Laplace) or smaller values (Lidstone). In Bayesian terms this is the posterior mean under a symmetric Dirichlet prior — the pseudo-count idea from the MAP lecture.

## Why does it work when the assumption is false?

Words in a document are obviously not independent ("New" and "York" co-occur). Yet Naive Bayes classifies well. The reason: classification needs only the **correct ranking** of classes, not correct probabilities. Violations of independence distort the probability *estimates* — typically pushing them towards 0 or 1 — but often preserve which class scores highest. Domingos and Pazzani (1997) analysed conditions under which Naive Bayes remains optimal for classification despite dependence.

:::warning
Because it double-counts correlated evidence, Naive Bayes produces **poorly calibrated, overconfident probabilities**. Use its predicted class freely, but do not treat its probabilities as reliable risk estimates without recalibration.
:::

## A text classifier in practice

```python
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

cats = ["sci.med", "sci.space", "rec.sport.hockey", "talk.politics.misc"]
train = fetch_20newsgroups(subset="train", categories=cats, remove=("headers", "footers", "quotes"))
test = fetch_20newsgroups(subset="test", categories=cats, remove=("headers", "footers", "quotes"))

models = {
    "Multinomial NB": make_pipeline(CountVectorizer(stop_words="english"), MultinomialNB(alpha=0.1)),
    "Complement NB":  make_pipeline(TfidfVectorizer(stop_words="english"), ComplementNB(alpha=0.3)),
    "Logistic reg.":  make_pipeline(TfidfVectorizer(stop_words="english"), LogisticRegression(max_iter=2000)),
}
for name, m in models.items():
    m.fit(train.data, train.target)
    print(f"{name:<15} accuracy = {accuracy_score(test.target, m.predict(test.data)):.3f}")

# Most indicative words per class for Multinomial NB
nb = models["Multinomial NB"]
vocab = nb[0].get_feature_names_out()
for i, c in enumerate(train.target_names):
    top = nb[1].feature_log_prob_[i].argsort()[-8:][::-1]
    print(c, "->", ", ".join(vocab[top]))
```

Naive Bayes is typically competitive with logistic regression here, trains in a fraction of a second, and its top words per class make it easy to inspect. **Complement Naive Bayes** often improves results on imbalanced text data.

## Generative versus discriminative

Naive Bayes is **generative** — it models $P(\mathbf{x}, c)$. Logistic regression is **discriminative** — it models $P(c \mid \mathbf{x})$ directly. In fact, Gaussian Naive Bayes with shared variances produces a decision rule of the same *form* as logistic regression. Ng and Jordan (2002) showed a classic trade-off:

- Naive Bayes approaches its (higher) asymptotic error **faster** — it wins with **little data**.
- Logistic regression has **lower asymptotic error** — it wins with **more data**.

Generative models can also handle missing features naturally (just omit the term) and can generate synthetic examples.

## When to use Naive Bayes

- A fast, strong **baseline** for text classification.
- Very **high-dimensional sparse** data with limited training examples.
- **Streaming/online** settings — counts update incrementally.
- Resource-constrained devices.

:::exercise
1. Implement Multinomial Naive Bayes from scratch with Laplace smoothing and compare with scikit-learn.
2. Train Gaussian Naive Bayes on the Iris dataset and plot a reliability diagram. Is it calibrated?
3. Show that Gaussian Naive Bayes with class-independent variances yields a linear decision boundary.
:::

:::takeaway
- Naive Bayes assumes conditional independence of features given the class.
- Training is counting; smoothing prevents zero probabilities.
- It ranks classes well even when the assumption fails, but its probabilities are overconfident.
- As a generative model it excels with little data and sparse, high-dimensional features.
:::
