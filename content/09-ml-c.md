=== POST ===
slug: decision-trees
title: Decision Trees: Splitting Criteria, Pruning and Interpretability
category: machine-learning
level: Beginner
tags: decision trees, gini, entropy, pruning, cart, interpretability
summary: Decision trees learn a flowchart of if–then questions. We derive Gini impurity and information gain, build a tree greedily, control overfitting with pruning, and see why trees are the building blocks of the best tabular models.
---
A doctor triaging patients asks a sequence of questions: *Is the temperature above 38.5 °C? Is breathing laboured? Is the patient under five?* Each answer narrows the possibilities. A **decision tree** learns such a sequence of questions automatically from data. Trees are interpretable, handle mixed feature types with little preprocessing, and — combined into ensembles — power the most successful models for tabular data.

## Structure

A decision tree consists of:

- **Internal nodes** that test a feature (e.g. $x_j \le t$);
- **Branches** for the outcomes of the test;
- **Leaves** that output a prediction — the majority class (classification) or the mean target (regression) of training examples reaching that leaf.

A tree partitions the feature space into axis-aligned rectangles, each with a constant prediction.

## Learning a tree: greedy recursive splitting

Finding the optimal tree is NP-hard, so algorithms like **CART** (Classification and Regression Trees) grow trees greedily:

1. At the current node, consider every feature $j$ and every candidate threshold $t$.
2. Choose the split that most reduces **impurity**.
3. Recurse on each child until a stopping condition is met.

## Impurity measures

For a node with class proportions $p_1, \dots, p_K$:

**Gini impurity** — the probability of misclassifying a random example if labelled randomly according to the node's class distribution:

$$
G = 1 - \sum_{k=1}^{K} p_k^2
$$

**Entropy**:

$$
H = -\sum_{k=1}^{K} p_k\log_2 p_k
$$

Both are zero for a pure node and maximal for a uniform mix. The quality of a split into children $L$ and $R$ is the **weighted impurity decrease**:

$$
\Delta = I(\text{parent}) - \frac{n_L}{n}I(L) - \frac{n_R}{n}I(R)
$$

With entropy, $\Delta$ is the **information gain**. In practice Gini and entropy produce very similar trees; Gini is slightly faster to compute. For regression trees, impurity is the **variance** (MSE) within the node.

:::example
A node has 10 examples: 5 "pass" and 5 "fail". Gini = $1 - 0.5^2 - 0.5^2 = 0.5$. Splitting on "attendance > 80%" gives a left child with 4 pass / 1 fail (Gini = $1 - 0.64 - 0.04 = 0.32$) and a right child with 1 pass / 4 fail (Gini = 0.32). Weighted impurity = 0.32, so $\Delta = 0.18$. The algorithm compares this with every other candidate split.
:::

## Overfitting and how to control it

A tree grown until every leaf is pure memorises the training data — high variance. Controls:

**Pre-pruning (early stopping):** limit `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_leaf_nodes`, or require a minimum impurity decrease.

**Post-pruning:** grow a large tree, then prune back. CART's **cost-complexity pruning** minimises

$$
R_\alpha(T) = R(T) + \alpha|T|
$$

where $R(T)$ is the training error and $|T|$ the number of leaves. Increasing $\alpha$ yields a nested sequence of smaller trees; choose $\alpha$ by cross-validation.

```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_text

X, y = load_breast_cancer(return_X_y=True, as_frame=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)

full = DecisionTreeClassifier(random_state=0).fit(X_tr, y_tr)
print("unpruned: leaves =", full.get_n_leaves(), " test acc =", round(full.score(X_te, y_te), 3))

path = full.cost_complexity_pruning_path(X_tr, y_tr)
best_alpha = max(path.ccp_alphas[:-1], key=lambda a: cross_val_score(
    DecisionTreeClassifier(random_state=0, ccp_alpha=a), X_tr, y_tr, cv=5).mean())
pruned = DecisionTreeClassifier(random_state=0, ccp_alpha=best_alpha).fit(X_tr, y_tr)
print("pruned:   leaves =", pruned.get_n_leaves(), " test acc =", round(pruned.score(X_te, y_te), 3))
print(export_text(pruned, feature_names=list(X.columns), max_depth=3))
```

The pruned tree is much smaller, usually at least as accurate, and readable as a set of rules.

## Strengths

- **Interpretable** (when small): a doctor or manager can follow the logic.
- **No scaling needed**: splits depend only on the ordering of values.
- **Handles non-linearity and interactions** automatically.
- **Mixed data types** and (in some implementations) missing values.
- **Fast prediction**: $O(\text{depth})$.

## Weaknesses

- **High variance**: small data changes can produce a completely different tree.
- **Axis-aligned boundaries**: a diagonal boundary requires a staircase of many splits.
- **Greedy**: a split that looks poor now may enable excellent splits later (XOR-type patterns).
- **Biased impurity**: features with many distinct values offer more candidate splits and can be over-selected.
- **Poor extrapolation** in regression: predictions are constant outside the training range.

## Feature importance

Summing the impurity decrease contributed by each feature over all splits gives **mean decrease in impurity (MDI)** importance. It is fast but biased towards high-cardinality features. **Permutation importance** on validation data — shuffle a feature and measure the drop in performance — is more reliable.

:::note
The high variance of trees, their weakness, becomes a strength in ensembles. Averaging many different deep trees (random forests) cancels variance; adding many shallow trees sequentially (gradient boosting) reduces bias. The next lectures build on exactly this.
:::

:::exercise
1. Compute the Gini impurity and entropy of a node with class counts (8, 2) and (5, 5).
2. Build a tree on the XOR problem: `X = [[0,0],[0,1],[1,0],[1,1]]`, `y = [0,1,1,0]`. Why does a greedy depth-1 split fail?
3. Train unpruned trees on two bootstrap samples of the same dataset and compare their top splits. What does this tell you about variance?
:::

:::takeaway
- Trees split greedily to maximise impurity decrease (Gini, entropy, or variance).
- Unrestricted trees overfit; use pre-pruning or cost-complexity pruning chosen by CV.
- Trees are interpretable, scale-invariant and capture interactions, but have high variance.
- Prefer permutation importance over impurity-based importance.
:::

=== POST ===
slug: bagging-and-bootstrap
title: Bagging and the Bootstrap: Variance Reduction by Averaging
category: machine-learning
level: Intermediate
tags: ensembles, bagging, bootstrap, variance reduction, out-of-bag
summary: Averaging many noisy models trained on resampled data produces one stable model. We study the bootstrap, derive why bagging reduces variance, and use out-of-bag error as a free validation estimate.
---
Ask one student to estimate the number of beans in a jar and you get a noisy guess. Ask a hundred students and average their guesses, and you usually get remarkably close. This "wisdom of crowds" works when the guesses are **diverse** and their errors **partially cancel**. **Bagging** — bootstrap aggregating, introduced by Leo Breiman in 1996 — applies this idea to machine-learning models.

## The bootstrap

The **bootstrap** (Efron, 1979) creates new datasets by sampling $n$ examples **with replacement** from the original $n$. Each bootstrap sample:

- has the same size as the original;
- contains some examples multiple times and omits others.

The probability that a given example is *not* chosen in $n$ draws is

$$
\left(1 - \frac{1}{n}\right)^n \xrightarrow{n \to \infty} e^{-1} \approx 0.368
$$

So each bootstrap sample contains about **63.2%** of the unique original examples; the remaining ~36.8% are **out-of-bag (OOB)**.

Statisticians use the bootstrap to estimate the sampling variability of any statistic — confidence intervals for a median, an F1 score, or a regression coefficient — without distributional formulas.

## Bagging

1. Draw $B$ bootstrap samples.
2. Train a model $\hat{f}_b$ on each.
3. Aggregate: average for regression, majority vote (or averaged probabilities) for classification:

$$
\hat{f}_{\text{bag}}(\mathbf{x}) = \frac{1}{B}\sum_{b=1}^{B}\hat{f}_b(\mathbf{x})
$$

## Why bagging reduces variance

Suppose each model's prediction at $\mathbf{x}$ has variance $\sigma^2$, and any two have correlation $\rho$. The variance of their average is

$$
\text{Var}\left(\frac{1}{B}\sum_b\hat{f}_b\right) = \rho\,\sigma^2 + \frac{1 - \rho}{B}\,\sigma^2
$$

- As $B \to \infty$, the second term vanishes, leaving $\rho\sigma^2$.
- If models were independent ($\rho = 0$), variance would fall to zero.
- Because all models see overlapping data, $\rho > 0$, so the benefit is limited by **correlation**.

Bagging leaves bias roughly unchanged (each model has similar bias). Therefore it helps most for **low-bias, high-variance** learners such as deep decision trees, and helps little for stable learners like linear regression or k-NN with large $k$.

:::note
The formula above is the single most important equation in ensemble learning. It tells you the two levers: build **many** models (reduce the second term) and make them **decorrelated** (reduce $\rho$). Random forests pull the second lever by randomising features at each split.
:::

## Out-of-bag evaluation

Each example is OOB for roughly 37% of the models. Predict each example using only the models that did not see it, and compare with its label. The resulting **OOB error** is an almost unbiased estimate of test error — obtained for free, without a separate validation set or cross-validation.

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

X, y = make_classification(n_samples=2000, n_features=20, n_informative=8,
                           flip_y=0.05, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

tree = DecisionTreeClassifier(random_state=0).fit(X_tr, y_tr)
print("single deep tree test acc:", round(tree.score(X_te, y_te), 3))

for B in [1, 10, 50, 200]:
    bag = BaggingClassifier(DecisionTreeClassifier(), n_estimators=B, oob_score=B > 1,
                            random_state=0, n_jobs=-1).fit(X_tr, y_tr)
    oob = f"  OOB acc: {bag.oob_score_:.3f}" if B > 1 else ""
    print(f"bagging B={B:>3}: test acc {bag.score(X_te, y_te):.3f}{oob}")
```

Test accuracy climbs as $B$ increases and then plateaus; the OOB estimate tracks it closely.

## Bagging from scratch

```python
from collections import Counter

def bagging_fit(X, y, B=50, seed=0):
    rng = np.random.default_rng(seed)
    models = []
    for _ in range(B):
        idx = rng.integers(0, len(X), len(X))          # sample with replacement
        models.append(DecisionTreeClassifier().fit(X[idx], y[idx]))
    return models

def bagging_predict(models, X):
    votes = np.stack([m.predict(X) for m in models])   # (B, n)
    return np.array([Counter(col).most_common(1)[0][0] for col in votes.T])

models = bagging_fit(X_tr, y_tr)
print("from-scratch bagging acc:", (bagging_predict(models, X_te) == y_te).mean().round(3))
```

## Practical notes

- More estimators never hurt accuracy (unlike boosting, bagging does not overfit by adding models); they only cost computation. Stop when OOB error plateaus.
- Training is **embarrassingly parallel** — each model is independent.
- Variants: **pasting** (sampling without replacement), **random subspaces** (sampling features), and **random patches** (both).
- Bagging reduces the interpretability of a single tree; use permutation importance or SHAP to explain the ensemble.

## Bagging in deep learning

Training several networks with different random seeds and averaging them — a **deep ensemble** — is one of the most reliable ways to improve accuracy and uncertainty estimates in deep learning. Even without bootstrap resampling, random initialisation and data order provide enough diversity.

:::exercise
1. Verify by simulation that about 63.2% of unique examples appear in a bootstrap sample of size $n = 10{,}000$.
2. Apply bagging to linear regression and to deep decision trees on the same regression dataset. Which benefits more, and why?
3. Using the variance formula, compute the ensemble variance for $\sigma^2 = 1$, $\rho = 0.3$ and $B = 1, 10, 100, \infty$.
:::

:::takeaway
- Bootstrap samples draw $n$ examples with replacement; ~36.8% are out-of-bag.
- Bagging averages models trained on bootstrap samples, reducing variance but not bias.
- Ensemble variance $= \rho\sigma^2 + (1 - \rho)\sigma^2/B$: decorrelation is the limiting factor.
- OOB error provides a free estimate of generalisation error.
:::

=== POST ===
slug: random-forests
title: Random Forests: Decorrelated Trees and Robust Predictions
category: machine-learning
level: Intermediate
tags: random forests, ensembles, feature importance, bagging, trees
summary: Random forests add feature randomness to bagged trees, breaking their correlation. We explain the algorithm, its hyperparameters, OOB estimates, feature importance pitfalls and when forests are the right tool.
---
If you can use only one algorithm for a new tabular dataset with no time to tune, the random forest is an excellent choice. Introduced by Leo Breiman in 2001, it combines bagging with an extra dose of randomness, producing models that are accurate, robust to noise and outliers, and nearly impossible to badly misconfigure.

## The key idea: decorrelate the trees

Bagged trees are correlated because they all tend to pick the same strong features near the root. Recall the ensemble variance formula:

$$
\text{Var} = \rho\,\sigma^2 + \frac{1 - \rho}{B}\,\sigma^2
$$

To reduce $\rho$, random forests restrict each split to a **random subset of $m$ features** (out of $d$). Strong features are unavailable at some splits, forcing trees to explore other structure. Each tree becomes slightly worse individually, but the ensemble improves because errors are less correlated.

## The algorithm

For $b = 1, \dots, B$:

1. Draw a bootstrap sample of the training data.
2. Grow a tree on it. At **each node**:
   - select $m$ features at random;
   - find the best split among only those $m$ features;
   - split, and recurse — typically growing trees deep, with no pruning.

Predict by averaging (regression) or majority vote/averaged probabilities (classification).

Typical defaults: $m = \sqrt{d}$ for classification, $m = d/3$ (or all features, in some libraries) for regression.

## Hyperparameters that matter

| Hyperparameter | Effect | Guidance |
|---|---|---|
| `n_estimators` ($B$) | More trees → lower variance; never overfits | 200–1000; stop when OOB error plateaus |
| `max_features` ($m$) | Smaller → more decorrelation, weaker trees | Tune among $\sqrt{d}$, $\log_2 d$, 0.3–0.5·$d$ |
| `min_samples_leaf` | Larger → smoother, less variance | 1–10; larger for noisy regression |
| `max_depth` | Limit tree depth | Usually unlimited; limit for speed/memory |
| `class_weight` | Reweight classes | `"balanced"` for imbalanced data |

Random forests are famous for working well with defaults — a big practical advantage.

```python
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

X, y = fetch_california_housing(return_X_y=True, as_frame=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)

tree = DecisionTreeRegressor(random_state=0).fit(X_tr, y_tr)
print("single tree MAE:", round(mean_absolute_error(y_te, tree.predict(X_te)), 3))

for m in [1.0, 0.5, 0.33]:
    rf = RandomForestRegressor(n_estimators=300, max_features=m, oob_score=True,
                               n_jobs=-1, random_state=0).fit(X_tr, y_tr)
    print(f"RF max_features={m}: test MAE {mean_absolute_error(y_te, rf.predict(X_te)):.3f}"
          f"  OOB R^2 {rf.oob_score_:.3f}")
```

(With `max_features=1.0`, the forest is plain bagging.)

## Feature importance — handle with care

Random forests offer two importance measures:

1. **Mean decrease in impurity (MDI)** — fast, computed during training, but **biased towards continuous and high-cardinality features** and computed on training data.
2. **Permutation importance** — shuffle one feature in validation data and measure the drop in performance. More reliable, but can understate the importance of correlated features (the model uses the correlated partner instead).

```python
from sklearn.inspection import permutation_importance
rf = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=0).fit(X_tr, y_tr)
pi = permutation_importance(rf, X_te, y_te, n_repeats=10, random_state=0, n_jobs=-1)
for i in pi.importances_mean.argsort()[::-1]:
    print(f"{X.columns[i]:<12} {pi.importances_mean[i]:.3f} ± {pi.importances_std[i]:.3f}")
```

:::warning
Feature importance tells you what the **model** relies on, not what **causes** the outcome. A forest predicting hospital mortality might rely heavily on "number of tests ordered" — a proxy for how sick doctors already believed the patient was, not a cause of death.
:::

## Other useful by-products

- **OOB error** — free validation.
- **Proximity matrix** — how often two examples land in the same leaf; useful for clustering, outlier detection and imputing missing values.
- **Quantile regression forests** — keep all leaf targets to estimate prediction intervals.
- **Isolation Forest** — a related randomised-tree method for anomaly detection.

## Strengths and weaknesses

**Strengths:** strong accuracy with little tuning; robust to outliers and irrelevant features; handles non-linearity and interactions; parallelisable; no feature scaling; OOB estimates.

**Weaknesses:** less interpretable than a single tree; large memory footprint and slower prediction than a linear model; cannot extrapolate beyond the training target range; usually slightly less accurate than well-tuned gradient boosting on tabular data.

## Random forest or gradient boosting?

| Consideration | Random forest | Gradient boosting |
|---|---|---|
| Tuning effort | Low | Moderate to high |
| Peak accuracy on tabular data | Very good | Often best |
| Overfitting by adding trees | No | Yes (needs early stopping) |
| Training parallelism | Across trees | Within trees |
| Noisy labels | Very robust | Can chase noise |

:::exercise
1. Plot OOB error against `n_estimators` from 10 to 1,000. Where does it plateau?
2. Add five columns of random noise and one random ID-like high-cardinality column to a dataset. Compare MDI and permutation importance.
3. Train a random forest regressor on $y = x$ for $x \in [0, 10]$ and predict at $x = 15$. Explain the result.
:::

:::takeaway
- Random forests = bagging + random feature subsets at each split, decorrelating trees.
- They work well with defaults; more trees never overfit.
- Use permutation importance, and remember importance is not causation.
- Forests cannot extrapolate; gradient boosting often edges them out when tuned.
:::

=== POST ===
slug: adaboost-explained
title: "Boosting I: AdaBoost and the Power of Weak Learners"
category: machine-learning
level: Intermediate
tags: boosting, adaboost, ensembles, weak learners, exponential loss
summary: Can many weak rules of thumb combine into a strong classifier? AdaBoost answered yes. We walk through the reweighting algorithm, derive it as exponential-loss minimisation, and discuss its margins and sensitivity to noise.
---
In 1988 Michael Kearns asked a theoretical question: if we can learn classifiers that are only *slightly* better than random guessing — **weak learners** — can we combine them into an arbitrarily accurate **strong learner**? Robert Schapire proved the answer is yes, and in 1995 Yoav Freund and Schapire created **AdaBoost** (Adaptive Boosting), a practical algorithm that won them the Gödel Prize. AdaBoost with decision stumps powered the Viola–Jones face detector that ran in early digital cameras.

## Boosting versus bagging

- **Bagging** trains models **independently in parallel** on resampled data and averages them — reducing **variance**.
- **Boosting** trains models **sequentially**, each focusing on the mistakes of the ensemble so far — reducing **bias** (and often variance too).

## The AdaBoost algorithm

Binary labels $y_i \in \{-1, +1\}$. Start with uniform example weights $w_i = 1/n$. For $t = 1, \dots, T$:

1. Train a weak learner $h_t$ on the **weighted** data (typically a **decision stump** — a one-split tree).
2. Compute its weighted error:
$$
\epsilon_t = \sum_{i:\, h_t(\mathbf{x}_i) \ne y_i} w_i
$$
3. Compute its vote weight:
$$
\alpha_t = \frac{1}{2}\ln\frac{1 - \epsilon_t}{\epsilon_t}
$$
4. Update and renormalise example weights:
$$
w_i \leftarrow \frac{w_i\,\exp\big(-\alpha_t\,y_i\,h_t(\mathbf{x}_i)\big)}{Z_t}
$$

Final classifier:

$$
H(\mathbf{x}) = \text{sign}\left(\sum_{t=1}^{T}\alpha_t\,h_t(\mathbf{x})\right)
$$

**Reading the update:** misclassified examples ($y_ih_t(\mathbf{x}_i) = -1$) are multiplied by $e^{\alpha_t} > 1$ — they become more important; correctly classified examples are down-weighted. The next weak learner is forced to concentrate on the hard cases. Learners with low error receive large votes $\alpha_t$; a learner at chance ($\epsilon_t = 0.5$) receives $\alpha_t = 0$.

## Training error falls exponentially

If each weak learner has edge $\gamma_t = 0.5 - \epsilon_t > 0$, the training error of $H$ is bounded by

$$
\prod_t 2\sqrt{\epsilon_t(1 - \epsilon_t)} \le \exp\left(-2\sum_{t=1}^{T}\gamma_t^2\right)
$$

Even a small consistent edge drives training error to zero exponentially fast. This is the theoretical heart of boosting.

## AdaBoost as exponential-loss minimisation

Friedman, Hastie and Tibshirani (2000) showed that AdaBoost performs **forward stagewise additive modelling** with the **exponential loss**:

$$
L(y, F(\mathbf{x})) = e^{-yF(\mathbf{x})}, \qquad F(\mathbf{x}) = \sum_t\alpha_t h_t(\mathbf{x})
$$

At each step it greedily adds the term $\alpha_th_t$ that most reduces this loss; the formula for $\alpha_t$ is exactly the optimal step size. This statistical view opened the door to **gradient boosting**, which works with any differentiable loss (next lecture).

The quantity $yF(\mathbf{x})$ is the **margin**. The exponential loss keeps pushing margins larger even after all training points are correctly classified — which helps explain AdaBoost's often surprising resistance to overfitting as $T$ grows.

## Implementation from scratch

```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y01 = make_classification(n_samples=1500, n_features=10, n_informative=5, random_state=1)
y = 2 * y01 - 1                                        # labels in {-1, +1}
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

def adaboost(X, y, T=200):
    n = len(y); w = np.full(n, 1 / n)
    learners, alphas = [], []
    for t in range(T):
        stump = DecisionTreeClassifier(max_depth=1).fit(X, y, sample_weight=w)
        pred = stump.predict(X)
        eps = np.clip(w[pred != y].sum(), 1e-10, 1 - 1e-10)
        alpha = 0.5 * np.log((1 - eps) / eps)
        w *= np.exp(-alpha * y * pred); w /= w.sum()
        learners.append(stump); alphas.append(alpha)
    return learners, np.array(alphas)

def predict(learners, alphas, X, upto=None):
    F = sum(a * h.predict(X) for h, a in zip(learners[:upto], alphas[:upto]))
    return np.sign(F)

L, A = adaboost(X_tr, y_tr)
for T in [1, 10, 50, 200]:
    print(f"T={T:>3}: train acc {np.mean(predict(L, A, X_tr, T) == y_tr):.3f}  "
          f"test acc {np.mean(predict(L, A, X_te, T) == y_te):.3f}")
```

A single stump is weak; two hundred of them form a strong classifier.

## Weaknesses

:::warning
The exponential loss punishes misclassified points exponentially. **Mislabelled examples and outliers** receive ever-growing weights, and AdaBoost may contort itself to fit them. On noisy data, use more robust losses (logistic loss via LogitBoost or gradient boosting), limit the number of rounds, or clean labels.
:::

- Sequential training cannot be parallelised across rounds.
- Performance depends on weak learners being better than chance but not too complex.

## Variants

- **SAMME** — multiclass AdaBoost (used by scikit-learn).
- **Real AdaBoost** — uses class probability estimates.
- **LogitBoost** — minimises logistic loss; more robust to noise.
- **Gradient boosting** — the general framework (next lecture).

:::exercise
1. Compute $\alpha_t$ for weak learners with $\epsilon_t = 0.1, 0.3, 0.45, 0.5$. Interpret the values.
2. Flip 10% of the training labels and compare AdaBoost with a random forest as $T$ grows.
3. Track the weights of the ten examples with the highest final weight. What do they have in common?
:::

:::takeaway
- Boosting trains learners sequentially, reweighting examples the ensemble gets wrong.
- AdaBoost's vote weights $\alpha_t = \frac{1}{2}\ln\frac{1-\epsilon_t}{\epsilon_t}$; training error falls exponentially with the edge.
- AdaBoost is forward stagewise minimisation of exponential loss — the bridge to gradient boosting.
- It is sensitive to label noise and outliers.
:::

=== POST ===
slug: gradient-boosting-machines
title: "Boosting II: Gradient Boosting Machines"
category: machine-learning
level: Advanced
tags: gradient boosting, boosting, trees, loss functions, shrinkage
summary: Gradient boosting performs gradient descent in function space, fitting each new tree to the negative gradient of the loss. We derive the algorithm, explain shrinkage and subsampling, and tune it properly.
---
Gradient boosting is the workhorse behind countless winning solutions on tabular data — in credit risk, click prediction, demand forecasting and data-science competitions. Jerome Friedman's 1999 insight was to view boosting as **gradient descent, not in parameter space, but in function space**. Once you see it that way, you can boost with any differentiable loss.

## The additive model

We build a model as a sum of $M$ simple functions (usually small regression trees):

$$
F_M(\mathbf{x}) = F_0(\mathbf{x}) + \sum_{m=1}^{M}\nu\,h_m(\mathbf{x})
$$

where $\nu \in (0, 1]$ is the **learning rate** (shrinkage). Each new tree $h_m$ should move $F$ in the direction that decreases the loss most.

## Functional gradient descent

Our objective is $\sum_i L(y_i, F(\mathbf{x}_i))$. Treat the predictions $F(\mathbf{x}_i)$ as the variables. The steepest-descent direction for each training point is the **negative gradient**, called the **pseudo-residual**:

$$
r_{im} = -\left[\frac{\partial L(y_i, F(\mathbf{x}_i))}{\partial F(\mathbf{x}_i)}\right]_{F = F_{m-1}}
$$

But a gradient defined only at training points does not generalise. So we **fit a regression tree $h_m$ to the pseudo-residuals** — it approximates the gradient direction as a function that can be evaluated anywhere.

## The algorithm

1. Initialise with the best constant: $F_0 = \arg\min_c\sum_iL(y_i, c)$ (the mean for squared loss; the log-odds for logistic loss).
2. For $m = 1, \dots, M$:
   1. Compute pseudo-residuals $r_{im}$.
   2. Fit a regression tree $h_m$ to $\{(\mathbf{x}_i, r_{im})\}$ with $J$ leaves.
   3. For each leaf $j$, compute the optimal output $\gamma_{jm} = \arg\min_\gamma\sum_{\mathbf{x}_i \in R_{jm}}L(y_i, F_{m-1}(\mathbf{x}_i) + \gamma)$ (a line search per leaf).
   4. Update $F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \nu\sum_j\gamma_{jm}\mathbb{1}[\mathbf{x} \in R_{jm}]$.

## Pseudo-residuals for common losses

| Loss | $L(y, F)$ | Pseudo-residual $r$ |
|---|---|---|
| Squared error | $\frac{1}{2}(y - F)^2$ | $y - F$ (the ordinary residual) |
| Absolute error | $\lvert y - F \rvert$ | $\text{sign}(y - F)$ |
| Huber | quadratic near 0, linear beyond $\delta$ | clipped residual |
| Logistic (binary) | $\ln(1 + e^{-yF})$, $y \in \{\pm1\}$ | $\frac{y}{1 + e^{yF}}$, equivalently $y_{01} - p$ |
| Quantile $\tau$ | pinball loss | $\tau$ or $\tau - 1$ |

For squared loss, gradient boosting is intuitive: **each tree fits the residual errors of the current model**. For other losses, "residual" generalises to "negative gradient".

## From scratch (squared loss)

```python
import numpy as np
from sklearn.tree import DecisionTreeRegressor

class GBRegressor:
    def __init__(self, n_trees=300, lr=0.05, depth=3):
        self.n_trees, self.lr, self.depth = n_trees, lr, depth
    def fit(self, X, y):
        self.f0 = y.mean()
        F = np.full(len(y), self.f0)
        self.trees = []
        for _ in range(self.n_trees):
            residual = y - F                               # negative gradient of 0.5*(y-F)^2
            t = DecisionTreeRegressor(max_depth=self.depth).fit(X, residual)
            F += self.lr * t.predict(X)
            self.trees.append(t)
        return self
    def predict(self, X):
        return self.f0 + self.lr * sum(t.predict(X) for t in self.trees)

rng = np.random.default_rng(0)
X = rng.uniform(-3, 3, (800, 2))
y = np.sin(X[:, 0]) * np.cos(X[:, 1]) + 0.1 * rng.normal(size=800)
m = GBRegressor().fit(X[:600], y[:600])
print("test RMSE:", np.sqrt(np.mean((m.predict(X[600:]) - y[600:]) ** 2)).round(4))
```

## Regularisation: the key to good generalisation

Unlike random forests, gradient boosting **will overfit** if you keep adding trees. Controls:

1. **Shrinkage** ($\nu$): small learning rates (0.01–0.1) need more trees but generalise better. Friedman found shrinkage to be the single most effective regulariser.
2. **Number of trees** $M$: choose by **early stopping** on validation data.
3. **Tree size**: depth 3–8 (or 8–64 leaves). Depth controls the order of interactions the model can capture — depth 1 (stumps) gives an additive model.
4. **Stochastic gradient boosting**: fit each tree on a random subsample (e.g. 50–80%) of rows — reduces variance and speeds training.
5. **Column subsampling**, **minimum samples per leaf**, and **L1/L2 penalties on leaf values** (in modern libraries).

:::tip
A robust tuning recipe: fix a learning rate of 0.05–0.1, use early stopping to find the number of trees, then tune depth/leaves and `min_samples_leaf`, then subsampling. Finally, lower the learning rate (e.g. halve it) and double the trees for a small last gain.
:::

```python
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_absolute_error

X, y = fetch_california_housing(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=0)
gb = HistGradientBoostingRegressor(learning_rate=0.05, max_iter=2000, max_leaf_nodes=31,
                                   early_stopping=True, validation_fraction=0.1,
                                   n_iter_no_change=50, random_state=0).fit(X_tr, y_tr)
print("trees used:", gb.n_iter_, " test MAE:", round(mean_absolute_error(y_te, gb.predict(X_te)), 3))
```

## Why gradient boosting dominates tabular data

- Trees handle heterogeneous features, missing values, monotonic and non-linear relationships and interactions.
- Boosting reduces bias progressively while regularisation controls variance.
- Flexible losses fit many business objectives (ranking, quantiles, Poisson counts).
- Benchmarks comparing tree ensembles and deep networks on typical medium-sized tabular datasets have repeatedly found tree-based boosting to be highly competitive or superior, especially with limited tuning time.

:::exercise
1. Derive the pseudo-residual for the logistic loss $\ln(1 + e^{-yF})$.
2. Using the from-scratch class, plot training and test RMSE against the number of trees for learning rates 1.0, 0.1 and 0.01.
3. Train with depth 1 and depth 6. Which captures the interaction in $\sin(x_1)\cos(x_2)$ better and why?
:::

:::takeaway
- Gradient boosting fits each new tree to the negative gradient (pseudo-residuals) of the loss.
- For squared loss, trees fit residuals; any differentiable loss works.
- Shrinkage, early stopping, tree size and subsampling are the essential regularisers.
- It is the default choice for high-accuracy models on tabular data.
:::

=== POST ===
slug: xgboost-lightgbm-catboost
title: XGBoost, LightGBM and CatBoost: Modern Gradient Boosting Libraries
category: machine-learning
level: Intermediate
tags: xgboost, lightgbm, catboost, gradient boosting, tabular data
summary: Three libraries turned gradient boosting into an industrial tool. We compare their key innovations — second-order optimisation, histogram splits, leaf-wise growth, GOSS, ordered target statistics — and show how to use each well.
---
Friedman's gradient boosting is elegant but, in its original form, slow on large datasets. Starting in 2014, three open-source libraries transformed it into the most successful method for tabular data: **XGBoost**, **LightGBM** and **CatBoost**. Knowing what each innovated — and how to use them without overfitting — is essential for any practising data scientist.

## XGBoost: regularised, second-order boosting

XGBoost (Chen & Guestrin, 2016) introduced several ideas:

**1. A regularised objective.** For $M$ trees $f_m$ with $T$ leaves and leaf weights $\mathbf{w}$:

$$
\mathcal{L} = \sum_i L(y_i, \hat{y}_i) + \sum_m\Omega(f_m), \qquad \Omega(f) = \gamma T + \frac{1}{2}\lambda\|\mathbf{w}\|^2
$$

**2. Second-order (Newton) approximation.** Expanding the loss to second order around the current prediction with gradients $g_i$ and Hessians $h_i$, the optimal weight of leaf $j$ containing examples $I_j$ is

$$
w_j^* = -\frac{\sum_{i \in I_j}g_i}{\sum_{i \in I_j}h_i + \lambda}
$$

and the **gain** from a split into left and right children is

$$
\text{Gain} = \frac{1}{2}\left[\frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda}\right] - \gamma
$$

Splits with negative gain are pruned — $\gamma$ acts as a minimum gain requirement. Using curvature (Hessians) makes each step more accurate than first-order gradient boosting.

**3. Engineering:** sparsity-aware splits that learn a default direction for missing values, weighted quantile sketches for candidate splits, cache-aware blocks, out-of-core computation and distributed training.

## LightGBM: speed at scale

LightGBM (Microsoft, 2017) focused on speed and memory:

- **Histogram-based splitting** — bucket continuous features into (e.g.) 255 bins, so finding splits costs $O(\text{bins})$ instead of $O(n)$.
- **Leaf-wise (best-first) growth** — grow the leaf with the largest gain anywhere in the tree, instead of level by level. It reaches lower loss with fewer leaves, but can overfit on small data — control it with `num_leaves` and `min_data_in_leaf`.
- **GOSS** (Gradient-based One-Side Sampling) — keep all examples with large gradients and randomly sample those with small gradients, reweighting to stay unbiased.
- **EFB** (Exclusive Feature Bundling) — merge sparse features that are rarely non-zero together.
- Native categorical splits.

## CatBoost: categorical features done right

CatBoost (Yandex, 2017–2018) targeted two problems:

- **Categorical features.** Target encoding (replacing a category with the mean target) leaks the label if computed on the same data. CatBoost uses **ordered target statistics**: process examples in a random permutation, and encode each example using only the targets of examples *before* it.
- **Prediction shift.** Standard boosting computes residuals using a model trained on the same examples, biasing gradients. **Ordered boosting** uses separate models so each example's residual comes from a model that did not see it.
- **Symmetric (oblivious) trees** — the same split is used across an entire level. They are fast at inference and act as regularisation.

CatBoost often performs well with default settings, especially on data with many categorical columns.

## Comparison

| | XGBoost | LightGBM | CatBoost |
|---|---|---|---|
| Split finding | Exact or histogram | Histogram | Histogram |
| Tree growth | Level-wise (default) | Leaf-wise | Symmetric (oblivious) |
| Categorical handling | Native (recent versions) or encode | Native | Ordered target statistics (best-in-class) |
| Speed on large data | Fast | Fastest (typically) | Fast; slower to train on some data |
| Defaults | Good | Need care on small data | Very good |
| GPU support | Yes | Yes | Yes |

## A practical template

```python
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

df = fetch_openml("adult", version=2, as_frame=True).frame
y = (df.pop("class") == ">50K").astype(int)
cat_cols = df.select_dtypes("category").columns.tolist()
X_tr, X_va, y_tr, y_va = train_test_split(df, y, test_size=0.2, stratify=y, random_state=0)

# LightGBM
lgbm = lgb.LGBMClassifier(n_estimators=5000, learning_rate=0.03, num_leaves=31,
                          min_child_samples=40, subsample=0.8, subsample_freq=1,
                          colsample_bytree=0.8, reg_lambda=1.0, verbose=-1)
lgbm.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(200, verbose=False)])
print("LightGBM AUC:", round(roc_auc_score(y_va, lgbm.predict_proba(X_va)[:, 1]), 4))

# XGBoost (native categorical support)
xgbm = xgb.XGBClassifier(n_estimators=5000, learning_rate=0.03, max_depth=6, subsample=0.8,
                         colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist",
                         enable_categorical=True, early_stopping_rounds=200, eval_metric="auc")
xgbm.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
print("XGBoost AUC:", round(roc_auc_score(y_va, xgbm.predict_proba(X_va)[:, 1]), 4))

# CatBoost
cb = CatBoostClassifier(iterations=5000, learning_rate=0.05, depth=6, eval_metric="AUC",
                        od_type="Iter", od_wait=200, verbose=False)
cb.fit(X_tr.astype({c: str for c in cat_cols}), y_tr, cat_features=cat_cols,
       eval_set=(X_va.astype({c: str for c in cat_cols}), y_va))
print("CatBoost AUC:", round(roc_auc_score(y_va, cb.predict_proba(X_va.astype({c: str for c in cat_cols}))[:, 1]), 4))
```

:::warning
Early stopping on a validation set uses that set for a decision, so it is no longer an unbiased test. Keep a separate **test** set (or use nested cross-validation) for the final number you report.
:::

## Tuning priorities

1. `learning_rate` + number of trees via early stopping.
2. Tree complexity: `max_depth` (XGBoost/CatBoost) or `num_leaves` (LightGBM), plus `min_child_samples`/`min_child_weight`.
3. Row and column subsampling (0.6–0.9).
4. L1/L2 regularisation on leaf weights.
5. For imbalanced data: `scale_pos_weight` or class weights, and evaluate with PR-AUC.

Use a Bayesian optimiser such as Optuna for efficient search. Beyond that, gains usually come from **features**, not hyperparameters.

## Useful extras

- **Monotonic constraints** — force predictions to increase with a feature (e.g. risk with age) for sensible, auditable models.
- **Custom objectives** — supply gradient and Hessian functions.
- **SHAP values** — fast exact explanations for tree ensembles (TreeSHAP), built into all three libraries.

:::exercise
1. Derive the optimal leaf weight $w_j^*$ from the second-order Taylor expansion of the loss.
2. Train LightGBM on a small dataset (2,000 rows) with `num_leaves=255` and with `num_leaves=15`. Compare validation performance.
3. Add a monotonic constraint to one feature and plot a partial-dependence curve before and after.
:::

:::takeaway
- XGBoost: regularised objective, second-order gain formula, sparsity-aware splits.
- LightGBM: histograms, leaf-wise growth, GOSS and EFB for speed.
- CatBoost: ordered target statistics and ordered boosting for categorical data; symmetric trees.
- Always use early stopping and a separate test set; features matter more than hyperparameters.
:::

=== POST ===
slug: support-vector-machines
title: Support Vector Machines: Maximum-Margin Classification
category: machine-learning
level: Intermediate
tags: svm, margin, hinge loss, classification, convex optimization
summary: Among all separating hyperplanes, SVMs pick the one with the widest margin. We derive the hard- and soft-margin formulations, the hinge loss, support vectors and the role of the C parameter.
---
Before deep learning's rise, **Support Vector Machines** were the state of the art for many classification problems — handwriting, text categorisation, bioinformatics. They combine an elegant geometric idea, a convex optimisation problem with a unique solution, and strong theoretical generalisation guarantees. Vladimir Vapnik and colleagues developed them from statistical learning theory, and they remain a valuable tool for medium-sized, high-dimensional problems.

## The geometric idea: maximise the margin

For linearly separable data there are infinitely many separating hyperplanes $\mathbf{w}^\top\mathbf{x} + b = 0$. Which is best? SVMs choose the one that maximises the **margin** — the distance to the nearest training points of either class. Intuitively, a wide margin leaves room for noise in future data: small perturbations of test points will not cross the boundary.

The distance from point $\mathbf{x}_i$ to the hyperplane is $\frac{|\mathbf{w}^\top\mathbf{x}_i + b|}{\|\mathbf{w}\|}$. Because $(\mathbf{w}, b)$ can be rescaled freely, we fix the scale so that the closest points satisfy $y_i(\mathbf{w}^\top\mathbf{x}_i + b) = 1$. The margin width is then $\frac{2}{\|\mathbf{w}\|}$.

## Hard-margin SVM

Maximising $2/\|\mathbf{w}\|$ is equivalent to minimising $\frac{1}{2}\|\mathbf{w}\|^2$:

$$
\min_{\mathbf{w}, b}\;\frac{1}{2}\|\mathbf{w}\|^2 \quad \text{subject to} \quad y_i(\mathbf{w}^\top\mathbf{x}_i + b) \ge 1 \quad \forall i
$$

This is a **convex quadratic program** with a unique global solution.

## Soft-margin SVM

Real data is rarely perfectly separable, and forcing separation makes the model a hostage to outliers. Introduce **slack variables** $\xi_i \ge 0$ that allow violations:

$$
\min_{\mathbf{w}, b, \boldsymbol{\xi}}\;\frac{1}{2}\|\mathbf{w}\|^2 + C\sum_{i=1}^{n}\xi_i \quad \text{s.t.} \quad y_i(\mathbf{w}^\top\mathbf{x}_i + b) \ge 1 - \xi_i,\;\; \xi_i \ge 0
$$

- $\xi_i = 0$: correctly classified, outside the margin.
- $0 < \xi_i \le 1$: inside the margin but correct.
- $\xi_i > 1$: misclassified.

The hyperparameter **$C$** controls the trade-off:

- **Large $C$**: violations are expensive → narrow margin, fits training data closely → low bias, high variance.
- **Small $C$**: violations are cheap → wide margin, more regularised → higher bias, lower variance.

## The hinge-loss view

At the optimum $\xi_i = \max(0, 1 - y_if(\mathbf{x}_i))$, so the soft-margin SVM is equivalent to

$$
\min_{\mathbf{w}, b}\;\sum_{i=1}^{n}\max\big(0,\,1 - y_i(\mathbf{w}^\top\mathbf{x}_i + b)\big) + \frac{\lambda}{2}\|\mathbf{w}\|^2, \qquad \lambda = \frac{1}{C}
$$

— **hinge loss plus L2 regularisation**. Compare with logistic regression, which uses the logistic loss $\ln(1 + e^{-yf})$. Both are convex surrogates for the 0–1 loss. The hinge loss is exactly zero for points beyond the margin, which is why SVM solutions depend only on a subset of points.

| Loss | Formula (margin $m = yf$) | Behaviour |
|---|---|---|
| 0–1 | $\mathbb{1}[m \le 0]$ | What we care about; not convex |
| Hinge (SVM) | $\max(0, 1 - m)$ | Zero beyond margin → sparse solution |
| Logistic | $\ln(1 + e^{-m})$ | Never exactly zero → probabilities |
| Exponential (AdaBoost) | $e^{-m}$ | Very sensitive to outliers |

## Support vectors

From the dual (see the Lagrange multipliers lecture), the solution is

$$
\mathbf{w} = \sum_{i=1}^{n}\alpha_iy_i\mathbf{x}_i, \qquad 0 \le \alpha_i \le C
$$

Only points with $\alpha_i > 0$ — those on or inside the margin, or misclassified — contribute. These are the **support vectors**. Removing any other training point leaves the model unchanged. The number of support vectors also bounds the leave-one-out error, one of several theoretical results linking sparsity to generalisation.

```python
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

X, y = make_blobs(n_samples=400, centers=2, cluster_std=2.2, random_state=4)
for C in [0.01, 1, 100]:
    clf = SVC(kernel="linear", C=C).fit(X, y)
    margin = 2 / np.linalg.norm(clf.coef_)
    print(f"C={C:>6}: support vectors={clf.n_support_.sum():>3}  margin width={margin:.3f}  "
          f"train acc={clf.score(X, y):.3f}")

grid = GridSearchCV(make_pipeline(StandardScaler(), SVC(kernel="linear")),
                    {"svc__C": np.logspace(-3, 3, 13)}, cv=5).fit(X, y)
print("best C:", grid.best_params_)
```

Small $C$ gives a wide margin with many support vectors; large $C$ a narrow margin with fewer.

## Practical considerations

- **Scale features** — SVMs are distance-based.
- **Probabilities**: SVMs output signed distances, not probabilities. Platt scaling (`probability=True`) fits a sigmoid on top, at extra cost.
- **Multiclass**: typically one-vs-one (scikit-learn's `SVC`) or one-vs-rest (`LinearSVC`).
- **Scaling to large data**: kernel SVM training scales between $O(n^2)$ and $O(n^3)$; for large linear problems, use `LinearSVC` or SGD with hinge loss, which scale linearly.
- **Imbalanced data**: use `class_weight="balanced"`.

:::note
SVMs remain strong when the number of features is large relative to the number of examples — text classification with TF-IDF features, genomics — and when you want a well-understood convex model. Their real magic appears with kernels, our next lecture.
:::

:::exercise
1. Show that the margin width is $2/\|\mathbf{w}\|$ under the canonical scaling.
2. Plot the decision boundary, margins and support vectors for $C = 0.01$ and $C = 100$ on a 2-D dataset.
3. Train logistic regression and a linear SVM on the same text classification task. Compare accuracy, training time and the number of non-zero dual coefficients.
:::

:::takeaway
- SVMs choose the separating hyperplane with maximum margin $2/\|\mathbf{w}\|$.
- Soft margins with slack variables handle overlap; $C$ trades margin width against violations.
- Equivalent to hinge loss + L2 regularisation; the solution depends only on support vectors.
- Scale features, tune $C$, and use linear solvers for large datasets.
:::

=== POST ===
slug: kernel-methods-kernel-trick
title: Kernel Methods and the Kernel Trick
category: machine-learning
level: Advanced
tags: kernels, svm, rbf, feature maps, mercer
summary: Kernels let linear algorithms learn non-linear functions by computing inner products in high- or infinite-dimensional feature spaces — without ever visiting them. We study feature maps, Mercer's theorem, common kernels and their limits.
---
A linear classifier cannot separate points inside a circle from points outside it. But map each point $(x_1, x_2)$ to $(x_1, x_2, x_1^2 + x_2^2)$ and the classes become separable by a plane. Mapping to a higher-dimensional feature space can make non-linear problems linear. The problem: good feature spaces may have millions or infinitely many dimensions. The **kernel trick** lets us work in such spaces **without ever computing the features**.

## Feature maps

Let $\boldsymbol{\phi}: \mathcal{X} \to \mathcal{F}$ map inputs into a feature space. A linear model there, $f(\mathbf{x}) = \mathbf{w}^\top\boldsymbol{\phi}(\mathbf{x}) + b$, is non-linear in $\mathbf{x}$. For degree-2 polynomial features in $d$ dimensions, $\boldsymbol{\phi}$ has $O(d^2)$ components; for degree $p$, $O(d^p)$ — computing them explicitly quickly becomes infeasible.

## The trick

Many algorithms — the SVM dual, ridge regression, PCA, k-means, Gaussian processes — access data **only through inner products** $\mathbf{x}_i^\top\mathbf{x}_j$. Replace every inner product with a **kernel function**

$$
k(\mathbf{x}, \mathbf{x}') = \boldsymbol{\phi}(\mathbf{x})^\top\boldsymbol{\phi}(\mathbf{x}')
$$

that computes the feature-space inner product directly.

:::example
For $\mathbf{x} \in \mathbb{R}^2$, let $k(\mathbf{x}, \mathbf{z}) = (\mathbf{x}^\top\mathbf{z})^2$. Expanding:
$(x_1z_1 + x_2z_2)^2 = x_1^2z_1^2 + 2x_1x_2z_1z_2 + x_2^2z_2^2 = \boldsymbol{\phi}(\mathbf{x})^\top\boldsymbol{\phi}(\mathbf{z})$
with $\boldsymbol{\phi}(\mathbf{x}) = (x_1^2, \sqrt{2}x_1x_2, x_2^2)$. The kernel costs one dot product and a square; the explicit map costs more — and the gap grows enormously with degree and dimension.
:::

The kernelised SVM decision function becomes

$$
f(\mathbf{x}) = \sum_{i \in \text{SV}}\alpha_iy_i\,k(\mathbf{x}_i, \mathbf{x}) + b
$$

— a weighted sum of similarities to the support vectors.

## Which functions are valid kernels?

**Mercer's theorem**: a symmetric function $k$ corresponds to an inner product in some feature space if and only if, for every finite set of points, the **Gram matrix** $\mathbf{K}_{ij} = k(\mathbf{x}_i, \mathbf{x}_j)$ is **positive semi-definite**. Valid kernels can be combined: sums, products, positive scalings and $k(f(\mathbf{x}), f(\mathbf{x}'))$ are all kernels. This lets you design kernels for strings, graphs, trees and molecules.

## Common kernels

| Kernel | Formula | Notes |
|---|---|---|
| Linear | $\mathbf{x}^\top\mathbf{x}'$ | No mapping; fast |
| Polynomial | $(\gamma\,\mathbf{x}^\top\mathbf{x}' + r)^p$ | Interactions up to degree $p$ |
| RBF / Gaussian | $\exp(-\gamma\|\mathbf{x} - \mathbf{x}'\|^2)$ | Infinite-dimensional feature space; the default |
| Laplacian | $\exp(-\gamma\|\mathbf{x} - \mathbf{x}'\|_1)$ | Less smooth |
| Sigmoid | $\tanh(\gamma\,\mathbf{x}^\top\mathbf{x}' + r)$ | Not always PSD |
| String / graph kernels | Count shared substructures | Text, proteins, molecules |

### Understanding the RBF kernel

The RBF kernel measures similarity that decays with squared distance. Its feature space is infinite-dimensional (its Taylor expansion contains polynomials of every degree). The parameter $\gamma$ sets the reach of each training example:

- **Large $\gamma$**: narrow bumps — each point influences only its immediate neighbourhood → wiggly boundary, overfitting risk (in the limit, like 1-NN).
- **Small $\gamma$**: wide bumps → smooth, nearly linear boundary, underfitting risk.

$C$ and $\gamma$ interact strongly; tune them **jointly** on a logarithmic grid.

```python
import numpy as np
from sklearn.datasets import make_circles
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = make_circles(n_samples=600, factor=0.4, noise=0.12, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

print("linear kernel acc:", SVC(kernel="linear").fit(X_tr, y_tr).score(X_te, y_te).round(3))

grid = GridSearchCV(make_pipeline(StandardScaler(), SVC(kernel="rbf")),
                    {"svc__C": np.logspace(-2, 3, 6), "svc__gamma": np.logspace(-3, 2, 6)},
                    cv=5).fit(X_tr, y_tr)
print("best RBF params:", grid.best_params_, " test acc:", round(grid.score(X_te, y_te), 3))
```

The linear SVM performs near chance on concentric circles; the RBF SVM separates them almost perfectly.

## Other kernel methods

- **Kernel ridge regression**: $\hat{f}(\mathbf{x}) = \mathbf{k}(\mathbf{x})^\top(\mathbf{K} + \lambda\mathbf{I})^{-1}\mathbf{y}$ — closed form.
- **Gaussian processes**: a Bayesian view where the kernel is the covariance function; they give predictions with uncertainty and are the engine of Bayesian optimisation.
- **Kernel PCA**: non-linear dimensionality reduction.
- **Support vector regression (SVR)**: uses an $\epsilon$-insensitive loss.

The **representer theorem** explains why these all work: for a broad class of regularised problems, the optimal function is a linear combination of kernel evaluations at the training points, $f(\cdot) = \sum_i\alpha_ik(\mathbf{x}_i, \cdot)$.

## Limitations and remedies

- **Scalability**: the Gram matrix is $n \times n$ — memory $O(n^2)$, training up to $O(n^3)$. Beyond roughly 50,000–100,000 examples, exact kernel methods become impractical.
- **Remedies**: **Random Fourier features** (Rahimi & Recht) approximate shift-invariant kernels with an explicit random feature map of modest dimension, so a linear model can be used; the **Nyström** method approximates the Gram matrix with a subset of points.
- **Kernel choice** encodes prior knowledge but must be made by hand, whereas neural networks *learn* their features. Interestingly, theory shows that infinitely wide neural networks behave like kernel methods (the **Neural Tangent Kernel**), a bridge between the two worlds.

```python
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
approx = make_pipeline(StandardScaler(), RBFSampler(gamma=1.0, n_components=500, random_state=0),
                       SGDClassifier(loss="hinge", alpha=1e-4, random_state=0))
print("random Fourier features acc:", approx.fit(X_tr, y_tr).score(X_te, y_te).round(3))
```

:::exercise
1. Show that $k(\mathbf{x}, \mathbf{z}) = (\mathbf{x}^\top\mathbf{z} + 1)^2$ corresponds to a feature map with constant, linear and quadratic terms, and write it out for $d = 2$.
2. Prove that the sum of two valid kernels is a valid kernel using the PSD definition.
3. Plot RBF SVM decision boundaries on `make_moons` for $\gamma \in \{0.1, 1, 10, 100\}$ with fixed $C$.
:::

:::takeaway
- Kernels compute feature-space inner products without explicit feature maps.
- Valid kernels produce PSD Gram matrices (Mercer); kernels can be combined and designed for structured data.
- The RBF kernel is the default; tune $C$ and $\gamma$ jointly.
- Exact kernel methods scale poorly; use random features or Nyström for large data.
:::
