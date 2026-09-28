=== POST ===
slug: what-is-machine-learning
title: What Is Machine Learning? The Learning Problem Formalised
category: machine-learning
level: Beginner
tags: machine learning, generalization, risk, hypothesis space
summary: Tom Mitchell's definition, the formal learning problem, empirical risk minimisation and the central goal of generalisation — the conceptual foundation for every model in this track.
---
Welcome to the Machine Learning track. We now move from programming behaviour by hand to **learning** it from data. Before meeting any algorithm, we must formalise what "learning" means, because the formal statement tells us what can go wrong — and almost everything that goes wrong in practice is a failure to generalise.

## Mitchell's definition

Tom Mitchell's 1997 definition remains the clearest:

> A computer program is said to learn from experience **E** with respect to some class of tasks **T** and performance measure **P**, if its performance at tasks in T, as measured by P, improves with experience E.

For a spam filter: T = classify emails; P = fraction classified correctly; E = a corpus of labelled emails. Always identify T, P and E before building anything. Many failed projects never clearly defined P.

## Traditional programming versus machine learning

In traditional programming, humans write **rules**; the computer applies them to **data** to produce **answers**. In machine learning, humans provide **data and answers**; the computer produces the **rules** (a model), which are then applied to new data. ML is the right tool when rules are too complex to write (recognising faces), change over time (fraud patterns), or must be personalised (recommendations).

## The formal supervised learning problem

We assume:

- An unknown **data distribution** $\mathcal{P}$ over input–output pairs $(\mathbf{x}, y)$.
- A **training set** $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^{n}$ drawn **i.i.d.** (independently and identically distributed) from $\mathcal{P}$.
- A **hypothesis space** $\mathcal{H}$ of candidate functions $h: \mathcal{X} \to \mathcal{Y}$ (e.g. all linear functions, all trees of depth 5, all networks of a given architecture).
- A **loss function** $\ell(h(\mathbf{x}), y)$ measuring the cost of a prediction.

Our true goal is to minimise the **expected risk** (generalisation error):

$$
R(h) = \mathbb{E}_{(\mathbf{x}, y) \sim \mathcal{P}}\big[\ell(h(\mathbf{x}), y)\big]
$$

But we cannot compute it — $\mathcal{P}$ is unknown. We can only compute the **empirical risk** on the training set:

$$
\hat{R}_n(h) = \frac{1}{n}\sum_{i=1}^{n}\ell(h(\mathbf{x}_i), y_i)
$$

**Empirical Risk Minimisation (ERM)** picks $\hat{h} = \arg\min_{h \in \mathcal{H}}\hat{R}_n(h)$.

## The central problem: generalisation

A model that memorises the training set achieves zero empirical risk, yet may perform terribly on new data. The difference

$$
R(\hat{h}) - \hat{R}_n(\hat{h})
$$

is the **generalisation gap**. Machine learning is, at its core, the science of keeping this gap small while also keeping the training error small.

:::note
Think of students preparing for an exam. One memorises past papers word by word (zero "training error"); another understands the concepts. On a new exam, the second student wins. Overfitting is memorisation; generalisation is understanding. Every technique you will learn — regularisation, validation, cross-validation, data augmentation, early stopping — exists to reward understanding over memorisation.
:::

## Decomposing the error

The excess risk of our learned model over the best possible predictor (the **Bayes optimal** predictor $h^*$) splits into two parts:

$$
R(\hat{h}) - R(h^*) = \underbrace{\Big[R(\hat{h}) - \min_{h \in \mathcal{H}} R(h)\Big]}_{\text{estimation error}} + \underbrace{\Big[\min_{h \in \mathcal{H}} R(h) - R(h^*)\Big]}_{\text{approximation error}}
$$

- A **small** hypothesis space has large approximation error (it cannot represent the truth) but small estimation error.
- A **large** hypothesis space has small approximation error but large estimation error (many functions fit the data by chance).

This trade-off — the bias–variance trade-off in another form — governs model selection throughout the course. Even the Bayes optimal predictor has non-zero error $R(h^*)$ if labels are noisy: the **irreducible error**.

## The i.i.d. assumption and its failure

ERM's guarantees assume training and test data come from the same distribution. In the real world this often fails: a model trained on hospital A's scanners is deployed at hospital B; a spam filter meets new spam tactics; a model trained before a crisis is used during one. This **distribution shift** is one of the main reasons deployed models degrade, and we will study it in the MLOps track.

## A first learning example

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

for k in [1, 5, 15]:
    model = KNeighborsClassifier(n_neighbors=k).fit(X_tr, y_tr)
    print(f"k={k:>2}: train acc={model.score(X_tr, y_tr):.3f}  test acc={model.score(X_te, y_te):.3f}")
```

With $k = 1$ the training accuracy is perfect — the model memorises — but the test accuracy is what matters. The held-out test set is our estimate of $R(\hat{h})$.

:::exercise
1. Identify T, P and E for (a) a crop-yield predictor, (b) a machine translation system, (c) a model that prioritises humanitarian cases for review.
2. Explain the difference between empirical risk and expected risk in your own words.
3. Give two real examples of distribution shift that could affect a model deployed in your country.
:::

:::takeaway
- ML = improving performance P at task T with experience E.
- We minimise empirical risk as a proxy for the unknowable expected risk.
- The goal is generalisation; the generalisation gap measures overfitting.
- Error = approximation error + estimation error (+ irreducible noise); the i.i.d. assumption often fails in deployment.
:::

=== POST ===
slug: types-of-machine-learning
title: Types of Machine Learning: Supervised, Unsupervised, Self-Supervised and Reinforcement
category: machine-learning
level: Beginner
tags: supervised, unsupervised, self-supervised, reinforcement learning, taxonomy
summary: Learning problems differ by the kind of feedback available. We map the major paradigms, their typical tasks and algorithms, and the hybrid settings — semi-supervised, weak and transfer learning — that dominate practice.
---
The most important question when facing a new ML problem is not "which algorithm?" but "**what kind of feedback does my data provide?**" The answer determines the family of methods available. Today we build a clear taxonomy.

## Supervised learning

Each training example has an input $\mathbf{x}$ and a desired output $y$ (the **label**). The model learns the mapping $\mathbf{x} \mapsto y$.

- **Classification**: $y$ is a category (spam/ham, disease A/B/C, digit 0–9).
- **Regression**: $y$ is a real number (house price, temperature, travel time).
- **Structured prediction**: $y$ is a structure (a sequence of tags, a parse tree, a segmentation mask, a translated sentence).

Typical algorithms: linear and logistic regression, decision trees, random forests, gradient boosting, SVMs, neural networks.

The bottleneck is **labels** — they cost human time and expertise. Labelling medical images may require specialists; labelling speech in low-resource languages may require rare language skills.

## Unsupervised learning

Only inputs $\mathbf{x}$ are available. The goal is to discover structure:

- **Clustering** — group similar items (customer segments, document topics).
- **Dimensionality reduction** — compress data while preserving structure (PCA, UMAP).
- **Density estimation** — model $p(\mathbf{x})$ (Gaussian mixtures, normalising flows).
- **Anomaly detection** — find unusual points (fraud, equipment failure).

Unsupervised results are harder to evaluate because there is no ground truth to compare against. Always validate clusters with domain experts or downstream tasks.

## Self-supervised learning

A powerful modern middle ground: create labels **from the data itself** by hiding part of the input and predicting it.

- **Masked language modelling** (BERT): hide 15% of words and predict them.
- **Next-token prediction** (GPT): predict each word from the previous ones.
- **Contrastive learning** (SimCLR, CLIP): learn that two augmented views of an image — or an image and its caption — belong together.
- **Masked image modelling** (MAE): reconstruct hidden image patches.

Self-supervision lets models learn from billions of unlabelled examples. The resulting **representations** are then adapted to downstream tasks with few labels. This recipe — **pretrain, then fine-tune** — is the foundation of modern foundation models.

## Reinforcement learning

An **agent** interacts with an **environment**, taking actions and receiving **rewards**. There are no correct labels for each action; feedback is evaluative ("that was good/bad") and often **delayed** (a chess move's value is revealed only at the end of the game). The agent must balance **exploration** and **exploitation**.

Applications: game playing, robotics, resource management, recommendation, and fine-tuning language models from human feedback (RLHF).

## Comparison table

| Paradigm | Feedback | Typical question | Example |
|---|---|---|---|
| Supervised | Correct answer per example | What is $y$ for this $\mathbf{x}$? | Predict loan default |
| Unsupervised | None | What structure exists? | Segment customers |
| Self-supervised | Derived from data | What is missing/next? | Pretrain a language model |
| Reinforcement | Scalar reward, often delayed | Which action maximises long-term reward? | Control a robot arm |

## Hybrid settings you will meet in practice

- **Semi-supervised learning** — few labels, many unlabelled examples. Techniques: pseudo-labelling, consistency regularisation.
- **Weak supervision** — noisy labels from heuristics, rules or crowdsourcing, combined statistically.
- **Active learning** — the model chooses which examples humans should label next to maximise information.
- **Transfer learning** — reuse a model trained on one task or domain for another.
- **Few-shot and zero-shot learning** — generalise from a handful of examples, or from a task description alone (as large language models do with prompts).
- **Online learning** — update the model continuously as data arrives.
- **Multi-task learning** — train one model on several related tasks to share knowledge.

## Choosing a paradigm: a decision guide

```text
Do you have labelled outputs for your task?
├── Yes, plenty ................................ supervised learning
├── A few, plus lots of unlabelled data ........ semi-supervised / transfer / active learning
├── No, but you want structure ................. unsupervised learning
├── No, but raw data is abundant ............... self-supervised pretraining, then adapt
└── Feedback comes as rewards from actions ..... reinforcement learning
```

```python
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression

X, y = make_blobs(n_samples=300, centers=3, random_state=42)

# Supervised: we use the labels y
clf = LogisticRegression(max_iter=500).fit(X, y)
print("supervised accuracy:", clf.score(X, y))

# Unsupervised: labels are ignored; we discover 3 groups
km = KMeans(n_clusters=3, n_init=10, random_state=0).fit(X)
print("cluster sizes:", [int((km.labels_ == k).sum()) for k in range(3)])
```

:::note
The boundaries are blurring. A single large language model is pretrained with self-supervision, fine-tuned with supervised instruction data, aligned with reinforcement learning from human feedback, and then used zero-shot. Understanding each paradigm lets you understand how these systems are built.
:::

:::exercise
1. Classify each task: (a) predicting tomorrow's rainfall, (b) grouping news articles by topic without labels, (c) training a robot to walk, (d) learning image features by predicting image rotations.
2. You have 200 labelled and 50,000 unlabelled satellite images of settlements. Propose a learning strategy.
3. Why is evaluation harder for unsupervised learning? Propose two evaluation approaches for a clustering of survey responses.
:::

:::takeaway
- Supervised learning uses labels; unsupervised finds structure; self-supervised creates labels from data; RL learns from rewards.
- Labels are expensive — hence semi-supervised, weak, active and transfer learning.
- Modern foundation models combine several paradigms in sequence.
:::

=== POST ===
slug: machine-learning-workflow
title: The Machine Learning Workflow: From Problem to Deployed Model
category: machine-learning
level: Beginner
tags: workflow, data, evaluation, baseline, crisp-dm
summary: Successful ML projects follow a disciplined process — problem framing, data collection, exploration, baselines, iteration, evaluation and deployment. We walk through it with a complete scikit-learn example.
---
Beginners think machine learning is about choosing algorithms. Practitioners know that algorithm choice is a small part of the job. The rest is framing the problem, understanding the data, building evaluation you can trust, and iterating. Today we walk through the complete workflow, the same one I recommend for your final-year projects.

## Step 1: Frame the problem

- **What decision will the model support?** A model is useful only if it changes a decision.
- **What is the target?** Define it precisely. "Customer churn" — within 30 days? 90 days?
- **What is the success metric**, both technical (F1, RMSE) and operational (time saved, errors avoided)?
- **What is the baseline?** How is the task done today, and how well?
- **What are the constraints?** Latency, interpretability, privacy, fairness, cost.

:::tip
Write a one-page **problem statement** before touching data. If you cannot state what a false positive and a false negative cost, you cannot choose a metric, and you cannot know when you are done.
:::

## Step 2: Collect and understand the data

Ask where the data comes from, how it was labelled, what time period it covers, and who is missing from it. Then perform **Exploratory Data Analysis (EDA)**: distributions, missing values, outliers, correlations, class balance, duplicates, and **leakage** — features that would not be available at prediction time.

## Step 3: Split the data correctly

Split **before** any fitting (including scaling or feature selection) to avoid leakage:

- **Training set** — fit parameters.
- **Validation set** — choose hyperparameters and models.
- **Test set** — final, one-time estimate of performance.

For time-dependent data, split **by time** (train on the past, test on the future). For data with groups (multiple records per patient), split **by group** so the same patient never appears in both train and test.

## Step 4: Build a simple baseline

Start with the simplest reasonable model — predict the majority class, predict the mean, or a logistic regression. A baseline tells you whether the problem is learnable and gives a reference point. Surprisingly often, a well-tuned simple model is hard to beat on tabular data.

## Step 5: Iterate

Improve in order of expected payoff: fix data quality issues → engineer features → try stronger model families → tune hyperparameters → ensemble. Change one thing at a time and track every experiment.

## Step 6: Evaluate thoroughly

Beyond one headline metric:

- Confusion matrix and per-class metrics.
- Performance on important **subgroups** (region, gender, language) — fairness checks.
- **Error analysis**: read 50–100 misclassified examples and categorise the causes.
- Calibration of predicted probabilities.
- Robustness to realistic shifts.

## Step 7: Deploy and monitor

Package the model and its preprocessing together, serve predictions, and monitor input distributions, prediction distributions and — when labels arrive — real-world performance. Plan for retraining. (The MLOps track covers this in depth.)

## A complete example

```python
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report

# 1-2. Data: the Titanic survival dataset
df = fetch_openml("titanic", version=1, as_frame=True).frame
X = df[["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]]
y = (df["survived"].astype(int))

# 3. Split first
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)

num = ["age", "sibsp", "parch", "fare"]
cat = ["pclass", "sex", "embarked"]
prep = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num),
    ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat),
])

# 4-5. Baseline and candidates, compared by cross-validation on the training set only
candidates = {
    "majority baseline": DummyClassifier(strategy="most_frequent"),
    "logistic regression": LogisticRegression(max_iter=1000),
    "gradient boosting": HistGradientBoostingClassifier(random_state=0),
}
for name, model in candidates.items():
    pipe = Pipeline([("prep", prep), ("model", model)])
    scores = cross_val_score(pipe, X_tr, y_tr, cv=5, scoring="f1")
    print(f"{name:<20} F1 = {scores.mean():.3f} ± {scores.std():.3f}")

# 6. Final evaluation of the chosen model on the untouched test set
best = Pipeline([("prep", prep), ("model", HistGradientBoostingClassifier(random_state=0))]).fit(X_tr, y_tr)
print(classification_report(y_te, best.predict(X_te)))
```

Notice three professional habits: preprocessing lives **inside** the pipeline (so it is fitted only on training folds), we compare against a **baseline**, and the test set is used **once**.

## Common workflow failures

| Failure | Symptom | Prevention |
|---|---|---|
| Data leakage | Suspiciously high validation scores | Split first; pipelines; check feature timing |
| Wrong metric | Great metric, useless model | Frame costs of errors up front |
| Test-set reuse | Reported results do not hold in production | Validation set for decisions; test once |
| Ignoring subgroups | Harm to under-represented users | Sliced evaluation |
| No monitoring | Silent degradation | Monitor drift and outcomes |

:::exercise
1. Write a one-page problem statement for an ML system that predicts which registered families need urgent follow-up.
2. Run the example and add a random forest candidate. Does it beat gradient boosting?
3. Deliberately introduce leakage (e.g. scale using the whole dataset before splitting, or include a feature derived from the target) and measure the inflated score.
:::

:::takeaway
- Frame the decision, target, metric, baseline and constraints before modelling.
- Split early and correctly (time and group aware); keep preprocessing in pipelines.
- Start with a baseline, iterate systematically, and evaluate beyond one metric.
- Use the test set once; deploy with monitoring.
:::

=== POST ===
slug: linear-regression-from-first-principles
title: Linear Regression from First Principles
category: machine-learning
level: Beginner
tags: regression, linear regression, least squares, normal equations, gradient descent
summary: The most important model in statistics and ML. We derive least squares geometrically and analytically, solve it with the normal equations and gradient descent, and interpret coefficients carefully.
---
Linear regression is over two hundred years old — Legendre and Gauss used least squares to predict the orbits of comets — and it remains one of the most useful tools in data science. It is also the perfect laboratory: nearly every concept in ML (loss functions, optimisation, regularisation, overfitting, probabilistic interpretation) appears in its simplest form here.

## The model

Given features $\mathbf{x} \in \mathbb{R}^d$, linear regression predicts

$$
\hat{y} = w_0 + w_1x_1 + \dots + w_dx_d = \mathbf{w}^\top\mathbf{x}
$$

where we append a constant feature $x_0 = 1$ so the intercept $w_0$ is absorbed into $\mathbf{w}$. For $n$ examples stacked as rows of the **design matrix** $\mathbf{X} \in \mathbb{R}^{n \times (d+1)}$, predictions are $\hat{\mathbf{y}} = \mathbf{X}\mathbf{w}$.

"Linear" means linear **in the parameters** — the features themselves can be non-linear transformations of raw inputs ($x^2$, $\log x$), as we will see next lecture.

## The loss: mean squared error

$$
J(\mathbf{w}) = \frac{1}{n}\sum_{i=1}^{n}(y_i - \mathbf{w}^\top\mathbf{x}_i)^2 = \frac{1}{n}\|\mathbf{y} - \mathbf{X}\mathbf{w}\|^2
$$

Why squares? Three reasons: (1) it is smooth and convex, giving a unique closed-form solution; (2) it corresponds to maximum likelihood under **Gaussian noise**; (3) it penalises large errors heavily. The last reason is also its weakness — outliers dominate the fit.

## The closed-form solution

Setting the gradient to zero:

$$
\nabla_{\mathbf{w}}J = -\frac{2}{n}\mathbf{X}^\top(\mathbf{y} - \mathbf{X}\mathbf{w}) = \mathbf{0} \;\Longrightarrow\; \mathbf{X}^\top\mathbf{X}\,\mathbf{w} = \mathbf{X}^\top\mathbf{y}
$$

These are the **normal equations**. If $\mathbf{X}^\top\mathbf{X}$ is invertible:

$$
\hat{\mathbf{w}} = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}
$$

**Geometric view.** $\hat{\mathbf{y}} = \mathbf{X}\hat{\mathbf{w}}$ is the orthogonal projection of $\mathbf{y}$ onto the column space of $\mathbf{X}$; the residual vector is perpendicular to every feature column.

:::warning
$\mathbf{X}^\top\mathbf{X}$ is singular when features are perfectly collinear or when $d > n$. Even when invertible, it may be **ill-conditioned**, making coefficients wildly unstable. In code use `np.linalg.lstsq` (based on QR/SVD) rather than an explicit inverse, and consider ridge regularisation.
:::

## Gradient descent solution

The closed form costs $O(nd^2 + d^3)$. For very large $d$ or streaming data, use gradient descent:

$$
\mathbf{w} \leftarrow \mathbf{w} + \frac{2\eta}{n}\mathbf{X}^\top(\mathbf{y} - \mathbf{X}\mathbf{w})
$$

```python
import numpy as np

rng = np.random.default_rng(0)
n = 200
area = rng.uniform(40, 200, n)                    # square metres
rooms = rng.integers(1, 6, n)
price = 15 + 0.9 * area + 8 * rooms + rng.normal(0, 10, n)   # in lakh taka (synthetic)

X = np.column_stack([np.ones(n), area, rooms])
y = price

# 1) Least squares (stable)
w_ls, *_ = np.linalg.lstsq(X, y, rcond=None)

# 2) Gradient descent on standardised features (for good conditioning)
mu, sd = X[:, 1:].mean(0), X[:, 1:].std(0)
Xs = np.column_stack([np.ones(n), (X[:, 1:] - mu) / sd])
w = np.zeros(3)
for _ in range(2000):
    w += 0.1 * 2 / n * Xs.T @ (y - Xs @ w)
# convert back to original units
w_gd = np.r_[w[0] - (w[1:] * mu / sd).sum(), w[1:] / sd]

print("least squares:", w_ls.round(3))
print("grad descent :", w_gd.round(3))
```

Both recover coefficients close to the true values (15, 0.9, 8). Note the **standardisation** before gradient descent — without it, the very different feature scales create an ill-conditioned problem and gradient descent crawls.

## Evaluating a regression model

- **MSE / RMSE** — in squared / original units.
- **MAE** — robust to outliers.
- **$R^2$** — fraction of variance explained: $R^2 = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$. $R^2 = 0$ means no better than predicting the mean; it can be negative on test data.

Always plot **residuals** against predictions and against each feature. Patterns in residuals (curves, funnels) reveal missing non-linearity or non-constant variance (heteroscedasticity).

## Interpreting coefficients

$w_j$ is the expected change in $y$ for a one-unit increase in $x_j$, **holding all other features fixed**. Cautions:

- Coefficients depend on units; standardise to compare importance.
- With correlated features, individual coefficients can be unstable or have counter-intuitive signs.
- **Association is not causation.** A coefficient describes a pattern in observational data, not the effect of intervening.

## Statistical assumptions

For valid confidence intervals and hypothesis tests on coefficients, classical theory assumes: linearity, independent errors, constant error variance, and (for small samples) normally distributed errors. The **Gauss–Markov theorem** states that under the first three, OLS is the best linear unbiased estimator (BLUE). For prediction alone, these assumptions matter less — but checking residuals remains good practice.

:::exercise
1. Derive the normal equations for simple regression $y = w_0 + w_1x$ and show $w_1 = \text{Cov}(x, y)/\text{Var}(x)$.
2. Run the code without standardisation. How many gradient-descent iterations are needed now?
3. Add an extreme outlier to the data and compare OLS with `sklearn.linear_model.HuberRegressor`.
:::

:::takeaway
- Linear regression minimises squared error; the solution satisfies the normal equations $\mathbf{X}^\top\mathbf{X}\mathbf{w} = \mathbf{X}^\top\mathbf{y}$.
- Geometrically it projects $\mathbf{y}$ onto the feature column space.
- Use `lstsq` or gradient descent (with standardised features), never explicit inverses.
- Evaluate with RMSE/MAE/$R^2$ and residual plots; interpret coefficients cautiously.
:::

=== POST ===
slug: polynomial-regression-basis-functions
title: Polynomial Regression and Basis Functions: Non-Linearity with Linear Models
category: machine-learning
level: Beginner
tags: regression, polynomial, basis functions, overfitting, feature engineering
summary: Linear models can fit curves if we transform the inputs. We study polynomial, spline and radial basis features, watch overfitting happen as degree grows, and connect it to model selection.
---
Real relationships are rarely straight lines. Crop yield rises with fertiliser and then falls; demand follows seasonal cycles. Does that mean we must abandon linear regression? No. The trick is to transform the inputs into new features — **basis functions** — and then fit a linear model on those. This simple idea is also our first clear view of overfitting.

## Basis function expansion

Replace the raw input $x$ with a vector of features $\boldsymbol{\phi}(x) = [\phi_0(x), \phi_1(x), \dots, \phi_M(x)]$ and fit

$$
\hat{y} = \mathbf{w}^\top\boldsymbol{\phi}(x) = \sum_{j=0}^{M} w_j\phi_j(x)
$$

The model is still **linear in $\mathbf{w}$**, so all the machinery of least squares — closed form, convexity, fast solvers — still applies. Only the design matrix changes: $\boldsymbol{\Phi}_{ij} = \phi_j(x_i)$.

## Common basis functions

| Basis | Form | Character |
|---|---|---|
| Polynomial | $\phi_j(x) = x^j$ | Global; unstable at high degree and at edges |
| Radial basis (Gaussian) | $\phi_j(x) = \exp\left(-\frac{(x - c_j)^2}{2s^2}\right)$ | Local bumps centred at $c_j$ |
| Splines | Piecewise polynomials joined smoothly at knots | Flexible, stable, widely used in statistics |
| Fourier | $\sin(kx), \cos(kx)$ | Periodic patterns (seasonality) |
| Step / binning | $\mathbb{1}[x \in \text{bin}_j]$ | Piecewise constant; simple, interpretable |

With several inputs, polynomial features include **interaction terms** such as $x_1x_2$. The number of features grows combinatorially: degree-$p$ polynomials in $d$ variables have $\binom{d + p}{p}$ terms.

## Watching overfitting happen

Fit polynomials of increasing degree to 15 noisy samples of a sine wave:

```python
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

rng = np.random.default_rng(1)
f = lambda x: np.sin(2 * np.pi * x)
x_tr = np.sort(rng.random(15)); y_tr = f(x_tr) + rng.normal(0, 0.2, 15)
x_te = np.linspace(0, 1, 200);  y_te = f(x_te) + rng.normal(0, 0.2, 200)

for degree in [1, 3, 5, 9, 14]:
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(x_tr[:, None], y_tr)
    tr = mean_squared_error(y_tr, model.predict(x_tr[:, None]))
    te = mean_squared_error(y_te, model.predict(x_te[:, None]))
    print(f"degree {degree:>2}: train MSE {tr:.4f}   test MSE {te:.4f}")
```

A typical result:

| Degree | Training MSE | Test MSE | Diagnosis |
|---|---|---|---|
| 1 | high | high | **Underfitting** — a line cannot bend |
| 3 | low | low | Good fit |
| 9 | very low | higher | Starting to overfit |
| 14 | ≈ 0 | enormous | **Overfitting** — passes through every point, wild oscillations |

With 15 points and 15 coefficients (degree 14), the polynomial interpolates the data exactly, including its noise. Plot it and you will see huge swings between training points — especially near the edges (**Runge's phenomenon**). The coefficients also become enormous, a sign we will exploit with regularisation.

:::note
This is the canonical picture of the bias–variance trade-off: training error falls monotonically with complexity, while test error forms a U-shape. Your job is to find the bottom of the U — using a validation set or cross-validation, **never** the training error.
:::

## Choosing the degree

```python
from sklearn.model_selection import cross_val_score
for degree in range(1, 12):
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    score = -cross_val_score(model, x_tr[:, None], y_tr, cv=5,
                             scoring="neg_mean_squared_error").mean()
    print(degree, round(score, 4))
```

Pick the degree with the lowest cross-validated error — or, better, keep a flexible basis and control complexity with **regularisation** (next lecture), which is usually more stable than choosing a discrete degree.

## Splines: the practical choice

High-degree global polynomials are unstable. **Splines** fit low-degree polynomials (typically cubic) on intervals between **knots**, constrained to join smoothly. They give flexible, well-behaved curves and underlie **Generalised Additive Models (GAMs)**:

$$
\hat{y} = w_0 + f_1(x_1) + f_2(x_2) + \dots + f_d(x_d)
$$

where each $f_j$ is a smooth spline. GAMs capture non-linear effects while remaining interpretable — you can plot each $f_j$ — which makes them popular in medicine and public policy.

```python
from sklearn.preprocessing import SplineTransformer
spline_model = make_pipeline(SplineTransformer(n_knots=6, degree=3), LinearRegression())
spline_model.fit(x_tr[:, None], y_tr)
print("spline test MSE:", mean_squared_error(y_te, spline_model.predict(x_te[:, None])))
```

## From basis functions to kernels and neural networks

Basis expansion raises a question: which basis should we choose? Two powerful answers come later in the course:

- **Kernel methods** implicitly use infinitely many basis functions while computing only inner products.
- **Neural networks** *learn* the basis functions from data: each hidden unit is an adaptive basis function, and the output layer is a linear model on top of them.

In this sense, a neural network is "linear regression on learned features".

:::exercise
1. Plot the fitted curves for degrees 1, 3 and 14 from the example, along with the true sine function.
2. Print the coefficient magnitudes for degree 14. What do you observe?
3. Model monthly temperature data with Fourier features $\sin(2\pi k t/12), \cos(2\pi k t/12)$ for $k = 1, 2$.
:::

:::takeaway
- Basis functions let linear models fit non-linear relationships; the model stays linear in parameters.
- Increasing flexibility lowers training error but eventually raises test error (overfitting).
- Choose complexity by validation/cross-validation — or regularise.
- Splines and GAMs offer stable, interpretable non-linearity; neural networks learn their own bases.
:::

=== POST ===
slug: regularization-ridge-lasso-elastic-net
title: Regularisation: Ridge, Lasso and Elastic Net
category: machine-learning
level: Intermediate
tags: regularization, ridge, lasso, elastic net, overfitting, feature selection
summary: Penalising large weights tames overfitting. We derive ridge regression's closed form, explain why lasso yields sparse models, combine them in elastic net, and tune the penalty by cross-validation.
---
In the last lecture a degree-14 polynomial fit the training points perfectly and produced absurd predictions — with gigantic coefficients. **Regularisation** attacks the problem directly: we add a penalty that discourages large or numerous weights. It is the most widely used defence against overfitting in all of machine learning, from linear models to billion-parameter networks (where it appears as weight decay).

## The regularised objective

$$
J(\mathbf{w}) = \underbrace{\frac{1}{n}\|\mathbf{y} - \mathbf{X}\mathbf{w}\|^2}_{\text{fit the data}} + \underbrace{\lambda\,\Omega(\mathbf{w})}_{\text{keep it simple}}
$$

The hyperparameter $\lambda \ge 0$ controls the trade-off. $\lambda = 0$ gives ordinary least squares; $\lambda \to \infty$ forces the weights to zero. By convention, the **intercept is not penalised**.

## Ridge regression (L2)

$\Omega(\mathbf{w}) = \|\mathbf{w}\|_2^2 = \sum_j w_j^2$. The closed-form solution is

$$
\hat{\mathbf{w}}_{\text{ridge}} = (\mathbf{X}^\top\mathbf{X} + \lambda'\mathbf{I})^{-1}\mathbf{X}^\top\mathbf{y}
$$

(with $\lambda' = n\lambda$ for our scaling). Adding $\lambda'\mathbf{I}$ makes the matrix **always invertible** and well-conditioned — ridge was originally invented (Hoerl & Kennard, 1970) precisely to stabilise regression with correlated features.

**SVD view.** With $\mathbf{X} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$, ridge shrinks the component along each singular direction by the factor

$$
\frac{\sigma_j^2}{\sigma_j^2 + \lambda'}
$$

Directions with large singular values (strongly supported by data) are barely shrunk; directions with small singular values (poorly determined, noise-prone) are shrunk heavily. That is exactly the right behaviour.

Ridge shrinks all coefficients towards zero but rarely makes any **exactly** zero.

## Lasso (L1)

$\Omega(\mathbf{w}) = \|\mathbf{w}\|_1 = \sum_j |w_j|$. Lasso (Tibshirani, 1996 — "least absolute shrinkage and selection operator") has no closed form but is convex; it is solved by coordinate descent.

Its defining property is **sparsity**: many coefficients become exactly zero, performing automatic **feature selection**. For a single coefficient with orthonormal features, the lasso solution is the **soft-thresholding** operator:

$$
\hat{w}_j = \text{sign}(w_j^{\text{OLS}})\,\max\big(|w_j^{\text{OLS}}| - \tfrac{\lambda'}{2},\; 0\big)
$$

Any OLS coefficient smaller than the threshold is set to zero; larger ones are shrunk by a constant. Compare ridge, which scales every coefficient by the same factor $1/(1 + \lambda')$ and never reaches zero.

:::note
Geometrically, lasso's constraint region $\|\mathbf{w}\|_1 \le t$ is a diamond (a cross-polytope in high dimensions) with corners on the axes. The elliptical contours of the squared loss usually first touch it at a corner, where some coordinates are zero. Ridge's constraint region is a round ball with no corners.
:::

## Elastic net

Lasso has two weaknesses: with groups of highly correlated features it tends to pick one arbitrarily, and when $d > n$ it selects at most $n$ features. **Elastic net** (Zou & Hastie, 2005) mixes both penalties:

$$
\Omega(\mathbf{w}) = \alpha\|\mathbf{w}\|_1 + \frac{1 - \alpha}{2}\|\mathbf{w}\|_2^2
$$

It keeps lasso's sparsity while selecting correlated features together, and is a robust default for high-dimensional data such as genomics or text.

## Comparison

| | Ridge | Lasso | Elastic net |
|---|---|---|---|
| Penalty | $\|\mathbf{w}\|_2^2$ | $\|\mathbf{w}\|_1$ | mix |
| Closed form | Yes | No | No |
| Sparse solution | No | Yes | Yes |
| Correlated features | Shares weight among them | Picks one | Groups them |
| Bayesian prior | Gaussian | Laplace | — |

## Practical rules

1. **Standardise features** before regularising. Penalties treat all coefficients equally, so scale matters.
2. **Tune $\lambda$ by cross-validation** over a logarithmic grid (e.g. $10^{-4}$ to $10^{2}$).
3. Consider the **one-standard-error rule**: choose the simplest model whose CV error is within one standard error of the minimum.

```python
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV, LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

# 100 samples, 200 features, only 10 truly informative
X, y, coef = make_regression(n_samples=100, n_features=200, n_informative=10,
                             noise=10, coef=True, random_state=0)
alphas = np.logspace(-3, 3, 50)
models = {
    "OLS": LinearRegression(),
    "Ridge": RidgeCV(alphas=alphas),
    "Lasso": LassoCV(alphas=alphas, max_iter=20000, cv=5),
    "ElasticNet": ElasticNetCV(l1_ratio=[.2, .5, .8], alphas=alphas, max_iter=20000, cv=5),
}
for name, m in models.items():
    pipe = make_pipeline(StandardScaler(), m)
    r2 = cross_val_score(pipe, X, y, cv=5, scoring="r2").mean()
    pipe.fit(X, y)
    nz = int((np.abs(pipe[-1].coef_) > 1e-6).sum())
    print(f"{name:<10} CV R^2 = {r2:6.3f}   non-zero coefficients = {nz}")
```

With 200 features and only 100 samples, OLS overfits badly, ridge helps, and lasso/elastic net do best while keeping a small number of features — close to the 10 that truly matter.

## Regularisation beyond linear models

The same idea appears everywhere: **weight decay** in neural networks, the $C$ parameter in SVMs and logistic regression (inverse regularisation strength), tree depth limits, dropout, early stopping, and data augmentation. All express a preference for simpler explanations — a mathematical Occam's razor.

:::exercise
1. Derive the ridge closed form by setting the gradient of the regularised objective to zero.
2. Plot the **regularisation path** — each coefficient as a function of $\lambda$ — for ridge and lasso on the example data (`sklearn.linear_model.lasso_path`).
3. Create two perfectly correlated copies of an informative feature. Compare how ridge, lasso and elastic net distribute weight between them.
:::

:::takeaway
- Regularisation adds a complexity penalty weighted by $\lambda$, chosen by cross-validation.
- Ridge (L2) shrinks smoothly and stabilises ill-conditioned problems.
- Lasso (L1) produces sparse models by soft-thresholding; elastic net handles correlated groups.
- Standardise features first; do not penalise the intercept.
:::

=== POST ===
slug: logistic-regression-explained
title: Logistic Regression: Probabilistic Classification Done Right
category: machine-learning
level: Beginner
tags: classification, logistic regression, sigmoid, cross-entropy, odds
summary: Despite its name, logistic regression is a classifier — and one of the most reliable. We derive it from log-odds, train it by maximum likelihood, interpret its coefficients and understand its decision boundary.
---
Logistic regression is often the first classifier students learn and the last one experts abandon. It is fast, interpretable, well-calibrated and hard to beat as a baseline. It is also a single neuron with a sigmoid activation — so understanding it deeply prepares you for neural networks.

## Why not linear regression for classification?

For a binary label $y \in \{0, 1\}$, linear regression predicts values outside $[0, 1]$ that cannot be interpreted as probabilities, and it is badly distorted by points far from the boundary. We want a model that outputs a valid probability $P(y = 1 \mid \mathbf{x})$.

## From log-odds to the sigmoid

The **odds** of an event with probability $p$ are $p/(1 - p)$, ranging over $(0, \infty)$. The **log-odds** (logit) range over all real numbers — so we can model them linearly:

$$
\ln\frac{p}{1 - p} = \mathbf{w}^\top\mathbf{x} = z
$$

Solving for $p$ gives the **sigmoid** (logistic) function:

$$
p = \sigma(z) = \frac{1}{1 + e^{-z}}
$$

The sigmoid maps any real score to $(0, 1)$, with $\sigma(0) = 0.5$, $\sigma(z) \to 1$ as $z \to \infty$, and the symmetry $\sigma(-z) = 1 - \sigma(z)$.

## Training by maximum likelihood

Each label is Bernoulli with probability $p_i = \sigma(\mathbf{w}^\top\mathbf{x}_i)$. The negative log-likelihood is the **binary cross-entropy**:

$$
J(\mathbf{w}) = -\frac{1}{n}\sum_{i=1}^{n}\Big[y_i\ln p_i + (1 - y_i)\ln(1 - p_i)\Big]
$$

Its gradient has a beautiful form:

$$
\nabla_{\mathbf{w}}J = \frac{1}{n}\sum_{i=1}^{n}(p_i - y_i)\,\mathbf{x}_i = \frac{1}{n}\mathbf{X}^\top(\mathbf{p} - \mathbf{y})
$$

— identical in form to linear regression's gradient, with predictions passed through a sigmoid. There is no closed-form solution, but the loss is **convex**, so gradient descent, Newton's method (known here as *iteratively reweighted least squares*) or L-BFGS find the global optimum.

:::warning
If the classes are **perfectly separable**, the unregularised MLE does not exist: the weights grow without bound to push probabilities to exactly 0 and 1. Regularisation (the default in scikit-learn, controlled by $C = 1/\lambda$) fixes this and generally improves generalisation.
:::

## The decision boundary

We predict class 1 when $p \ge 0.5$, i.e. when $\mathbf{w}^\top\mathbf{x} \ge 0$. The boundary $\mathbf{w}^\top\mathbf{x} = 0$ is a **hyperplane** — logistic regression is a linear classifier. To obtain curved boundaries, add polynomial or other basis features.

The threshold 0.5 is not sacred. If false negatives are costly (missing a disease), lower the threshold; if false positives are costly, raise it. Choose it using the costs of errors and the validation set.

## Interpreting coefficients

Each coefficient is a change in **log-odds** per unit of the feature. Exponentiating gives an **odds ratio**:

$$
e^{w_j} = \text{factor by which the odds multiply when } x_j \text{ increases by 1}
$$

If $w_{\text{smoker}} = 0.9$, smokers have $e^{0.9} \approx 2.46$ times the odds of the outcome, holding other features fixed. This interpretability is why logistic regression dominates in medicine, epidemiology and credit scoring.

## Implementation from scratch

```python
import numpy as np

def sigmoid(z):
    return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))  # stable

def train_logreg(X, y, lr=0.1, epochs=3000, lam=1e-3):
    Xb = np.column_stack([np.ones(len(X)), X])
    w = np.zeros(Xb.shape[1])
    for _ in range(epochs):
        p = sigmoid(Xb @ w)
        grad = Xb.T @ (p - y) / len(y)
        grad[1:] += lam * w[1:]                      # L2, intercept not penalised
        w -= lr * grad
    return w

rng = np.random.default_rng(0)
n = 500
hours = rng.uniform(0, 10, n)
attendance = rng.uniform(0.3, 1.0, n)
logit = -6 + 0.8 * hours + 4 * attendance
passed = (rng.random(n) < sigmoid(logit)).astype(float)

X = np.column_stack([hours, attendance])
w = train_logreg((X - X.mean(0)) / X.std(0), passed)
print("weights (standardised features):", w.round(3))
print("odds ratio per 1 SD of study hours:", np.exp(w[1]).round(2))
```

## Calibration

Because it is trained by maximum likelihood, logistic regression tends to produce **well-calibrated** probabilities: among cases predicted at 0.7, roughly 70% are positive. Many more powerful models (boosted trees, deep networks) are less well calibrated. Check with a reliability diagram (`sklearn.calibration.calibration_curve`).

## Strengths and limitations

**Strengths:** fast, convex, interpretable, calibrated, works well with many sparse features (text), strong baseline.

**Limitations:** linear decision boundary unless features are engineered; sensitive to strongly correlated features (use regularisation); cannot capture complex interactions automatically.

:::exercise
1. Show that $\sigma'(z) = \sigma(z)(1 - \sigma(z))$ and use it to derive the gradient of the cross-entropy loss.
2. Train `sklearn.linear_model.LogisticRegression` on the synthetic data above and compare coefficients with the from-scratch version.
3. Plot precision and recall as the decision threshold varies from 0.1 to 0.9. Which threshold would you choose for a screening application?
:::

:::takeaway
- Logistic regression models log-odds linearly; the sigmoid converts them to probabilities.
- It is trained by minimising binary cross-entropy (convex); gradient $= \mathbf{X}^\top(\mathbf{p} - \mathbf{y})/n$.
- The decision boundary is linear; the threshold should reflect error costs.
- Coefficients are log-odds ratios; probabilities are usually well calibrated.
:::

=== POST ===
slug: softmax-regression-multiclass
title: Softmax Regression and Multiclass Classification Strategies
category: machine-learning
level: Beginner
tags: classification, softmax, multiclass, one-vs-rest, cross-entropy
summary: How do we classify into more than two classes? We generalise logistic regression to softmax regression, derive its gradient, and compare it with one-vs-rest and one-vs-one strategies.
---
Most real classification problems have more than two classes: ten digits, dozens of document categories, thousands of product types, fifty thousand vocabulary words for next-token prediction. Today we extend logistic regression to $K$ classes. The result — **softmax regression** (multinomial logistic regression) — is also exactly the final layer of almost every neural network classifier.

## The softmax function

Give each class $k$ its own weight vector $\mathbf{w}_k$ and compute a score (**logit**) $z_k = \mathbf{w}_k^\top\mathbf{x}$. Convert scores to probabilities with the **softmax**:

$$
P(y = k \mid \mathbf{x}) = \text{softmax}(\mathbf{z})_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}
$$

Properties:

- Outputs are positive and sum to one — a valid categorical distribution.
- It preserves order: the largest logit gets the largest probability.
- It is invariant to adding a constant to all logits (so one weight vector is redundant; regularisation resolves this).
- For $K = 2$, softmax reduces to the sigmoid of the difference of logits.

The name comes from being a smooth ("soft") version of the argmax. Scaling logits by a **temperature** $T$, $\text{softmax}(\mathbf{z}/T)$, makes the distribution sharper ($T < 1$) or flatter ($T > 1$) — a knob we will meet again in knowledge distillation and language-model sampling.

## Loss: categorical cross-entropy

With a one-hot label $\mathbf{y}$ and predicted probabilities $\mathbf{p}$:

$$
J = -\frac{1}{n}\sum_{i=1}^{n}\sum_{k=1}^{K} y_{ik}\ln p_{ik} = -\frac{1}{n}\sum_{i=1}^{n}\ln p_{i, y_i}
$$

— the negative log-probability of the correct class, i.e. the negative log-likelihood under a categorical model.

## The gradient

Remarkably, the gradient with respect to the logits is again "prediction minus target":

$$
\frac{\partial J_i}{\partial \mathbf{z}_i} = \mathbf{p}_i - \mathbf{y}_i
$$

so for the weight matrix $\mathbf{W} \in \mathbb{R}^{d \times K}$ (columns $\mathbf{w}_k$):

$$
\nabla_{\mathbf{W}}J = \frac{1}{n}\mathbf{X}^\top(\mathbf{P} - \mathbf{Y})
$$

The loss is convex in $\mathbf{W}$, so training finds the global optimum.

```python
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

X, y = load_digits(return_X_y=True)                  # 8x8 digit images, 10 classes
X = X / 16.0
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)

def softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True)             # numerical stability
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)

K, d = 10, X.shape[1]
W, b = np.zeros((d, K)), np.zeros(K)
Y = np.eye(K)[y_tr]                                  # one-hot labels
for epoch in range(500):
    P = softmax(X_tr @ W + b)
    G = (P - Y) / len(X_tr)                          # dJ/dZ
    W -= 0.5 * (X_tr.T @ G + 1e-4 * W)
    b -= 0.5 * G.sum(axis=0)
    if epoch % 100 == 0:
        loss = -np.log(P[np.arange(len(y_tr)), y_tr] + 1e-12).mean()
        print(epoch, round(loss, 4))

acc = (softmax(X_te @ W + b).argmax(1) == y_te).mean()
print("test accuracy:", round(acc, 3))
```

A simple linear softmax model reaches roughly 95% accuracy on this small digits dataset.

## Alternative strategies: reducing multiclass to binary

Any binary classifier can be extended to $K$ classes:

### One-vs-Rest (OvR)
Train $K$ binary classifiers, each separating one class from all others; predict the class with the highest score. Simple and scalable, but each classifier faces an imbalanced problem, and scores from separately trained classifiers may not be comparable.

### One-vs-One (OvO)
Train a classifier for every pair of classes — $K(K-1)/2$ of them — and predict by majority vote. Each classifier trains on less data, which suits algorithms that scale poorly with dataset size (such as kernel SVMs), but the number of models grows quadratically.

| Strategy | Models trained | Probabilities | Typical use |
|---|---|---|---|
| Softmax (multinomial) | 1 | Coherent, sum to 1 | Default for linear models and neural nets |
| One-vs-Rest | $K$ | Need normalisation | Linear SVMs, very many classes |
| One-vs-One | $K(K-1)/2$ | From votes | Kernel SVMs |

## Multi-label is different

In **multi-label** classification, each example can belong to several classes at once (a photo containing both "beach" and "dog"). Softmax is wrong here because it forces classes to compete. Use **independent sigmoids** — one binary logistic output per label — with binary cross-entropy summed over labels.

## Evaluating multiclass models

- **Confusion matrix** — which classes are confused with which.
- **Per-class precision/recall/F1**, combined with **macro** averaging (treats classes equally — good for imbalanced data) or **micro/weighted** averaging.
- **Top-k accuracy** — whether the true class is among the $k$ most probable (standard for ImageNet with 1,000 classes).

:::tip
When classes are imbalanced, report **macro-F1** alongside accuracy. A model that ignores a rare class can still have high accuracy; macro-F1 will expose it.
:::

:::exercise
1. Show that softmax with $K = 2$ equals the sigmoid of $z_1 - z_2$.
2. Derive $\partial J/\partial z_k = p_k - y_k$ using the softmax Jacobian.
3. Train scikit-learn's `LogisticRegression` in multinomial mode and with `OneVsRestClassifier` on the digits data. Compare accuracy and calibration.
:::

:::takeaway
- Softmax converts $K$ logits into a probability distribution; temperature controls sharpness.
- Categorical cross-entropy is the NLL; its gradient w.r.t. logits is $\mathbf{p} - \mathbf{y}$.
- OvR and OvO reduce multiclass problems to binary ones; softmax is the principled default.
- Multi-label problems need independent sigmoids, not softmax.
:::
