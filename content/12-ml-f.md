=== POST ===
slug: recommender-systems-collaborative-filtering
title: Recommender Systems I: Content-Based and Collaborative Filtering
category: machine-learning
level: Intermediate
tags: recommender systems, collaborative filtering, similarity, cold start
summary: Recommendation engines drive what billions of people watch, read and buy. We compare content-based and collaborative approaches, build user- and item-based neighbourhood models, and confront cold start and popularity bias.
---
Recommender systems are arguably the most economically influential machine-learning systems ever built. They decide which videos you see next, which products appear first, and which courses a learning platform suggests. Today we learn the two classical families of recommenders and their core difficulties.

## The problem

We have $m$ users, $n$ items and a sparse **interaction matrix** $\mathbf{R} \in \mathbb{R}^{m \times n}$, where $r_{ui}$ is user $u$'s rating of item $i$ (explicit feedback) or an indicator of a click, purchase or view (implicit feedback). Most entries are unknown — typical matrices are more than 99% empty. The task is to predict missing entries, or more usefully, to **rank** unseen items for each user.

## Content-based filtering

Describe each item by features (genre, keywords, text embedding, price) and build a **profile** for each user from the items they liked. Recommend items similar to the profile.

- **Strengths:** works for new items with features; recommendations are explainable ("because you liked X"); no need for other users' data.
- **Weaknesses:** limited by the quality of item features; tends to recommend "more of the same" (low serendipity); new users have no profile.

## Collaborative filtering (CF)

CF ignores item content and relies on the **wisdom of other users**: people who agreed in the past tend to agree in the future.

### User-based CF

Predict $u$'s rating of item $i$ from similar users who rated $i$:

$$
\hat{r}_{ui} = \bar{r}_u + \frac{\sum_{v \in N_i(u)}\text{sim}(u, v)\,(r_{vi} - \bar{r}_v)}{\sum_{v \in N_i(u)}|\text{sim}(u, v)|}
$$

Subtracting each user's mean rating $\bar{r}$ removes individual rating habits (generous vs strict raters).

### Item-based CF

Predict from items similar to $i$ that $u$ has rated. Item–item similarities are more stable than user–user similarities (items change less than people's tastes) and can be precomputed. Amazon's classic "customers who bought this also bought" system was item-based.

### Similarity measures

- **Cosine similarity** between rating vectors.
- **Pearson correlation** (cosine on mean-centred ratings) — adjusts for rating scales.
- **Jaccard similarity** for binary implicit data.
- **Shrinkage** $\text{sim} \times \frac{n_{\text{common}}}{n_{\text{common}} + \beta}$ — down-weights similarities computed from few co-rated items.

```python
import numpy as np

R = np.array([            # rows: users, cols: items, 0 = unknown
    [5, 4, 0, 1, 0],
    [4, 5, 1, 0, 1],
    [1, 0, 5, 4, 5],
    [0, 1, 4, 5, 4],
    [5, 5, 0, 0, 1],
], dtype=float)
mask = R > 0

def item_similarity(R, mask):
    means = np.where(mask, R, np.nan)
    C = np.where(mask, R - np.nanmean(means, axis=0, keepdims=True), 0)   # item-centred
    norms = np.linalg.norm(C, axis=0) + 1e-9
    return (C.T @ C) / np.outer(norms, norms)

S = item_similarity(R, mask)

def predict(u, i, k=2):
    rated = np.where(mask[u])[0]
    nbrs = rated[np.argsort(-S[i, rated])[:k]]
    w = S[i, nbrs]
    return float((w @ R[u, nbrs]) / (np.abs(w).sum() + 1e-9))

print("user 0, item 2:", round(predict(0, 2), 2))   # low: user 0 likes items 0-1, dislikes 3
print("user 3, item 0:", round(predict(3, 0), 2))
```

## Implicit feedback

Most real data is implicit: clicks, views, purchases, listening time. It has no negative signal — an unclicked item may be disliked **or simply unseen**. Implicit methods treat interactions as positive confidence and sample unobserved items as weak negatives. Evaluation uses **ranking metrics**:

- **Precision@k / Recall@k** — relevant items in the top $k$;
- **NDCG@k** — rewards relevant items ranked higher;
- **MAP** and **hit rate**.

Evaluate by holding out each user's **most recent** interactions (temporal split), not random ones, to mimic real use.

## The hard problems

1. **Cold start** — new users and new items have no interactions. Remedies: content features, popularity-based defaults, onboarding questions, hybrid models.
2. **Sparsity** — few co-rated items make similarities unreliable.
3. **Popularity bias** — popular items get recommended more, get more interactions, and become even more popular — a feedback loop that starves niche items.
4. **Filter bubbles** — optimising short-term engagement can narrow what users see.
5. **Scalability** — millions of users × items; neighbourhood methods need approximate nearest-neighbour search.

:::note
A recommender is not only a predictor; it **changes the data it will later be trained on**. This feedback loop makes offline accuracy a weak proxy for real-world value. Industry relies heavily on online A/B tests, and responsible designers add goals such as diversity, novelty and fairness to creators — not just clicks.
:::

## Hybrid systems

Modern recommenders combine collaborative signals, content features and context (time, device, location) — often in two stages: a fast **candidate generation** step retrieves hundreds of items (e.g. via embedding nearest neighbours), and a heavier **ranking model** (gradient boosting or a neural network) orders them. The next lecture covers matrix factorisation, the method that learns the embeddings.

:::exercise
1. Implement user-based CF on the matrix above and compare predictions with item-based CF.
2. Explain why Pearson correlation is preferable to cosine for explicit ratings.
3. Design an onboarding flow to solve cold start for a course-recommendation system for new students.
:::

:::takeaway
- Content-based filtering uses item features; collaborative filtering uses patterns across users.
- Item-based CF with mean-centred similarities is simple, stable and scalable.
- Implicit feedback needs ranking metrics and temporal evaluation.
- Cold start, sparsity, popularity bias and feedback loops are the central challenges.
:::

=== POST ===
slug: matrix-factorization-recommendations
title: Recommender Systems II: Matrix Factorisation and Latent Factors
category: machine-learning
level: Advanced
tags: recommender systems, matrix factorization, als, embeddings, netflix prize
summary: Matrix factorisation represents users and items as vectors in a shared latent space. We derive the regularised objective, train it with SGD and ALS, add biases and implicit feedback, and connect it to modern embedding models.
---
The Netflix Prize (2006–2009) offered one million dollars to anyone who could improve Netflix's rating predictions by 10%. The decisive technique was **matrix factorisation**: represent every user and every item as a short vector of latent factors so that their dot product predicts the rating. The idea is simple, scales to huge data and underlies the embedding-based retrieval used by today's recommendation systems.

## The model

Choose a latent dimension $k$ (e.g. 20–200). Learn user vectors $\mathbf{p}_u \in \mathbb{R}^k$ and item vectors $\mathbf{q}_i \in \mathbb{R}^k$ such that

$$
\hat{r}_{ui} = \mu + b_u + b_i + \mathbf{p}_u^\top\mathbf{q}_i
$$

where $\mu$ is the global mean rating, $b_u$ a user bias (some people rate everything high) and $b_i$ an item bias (some films are universally liked). Stacking vectors, $\mathbf{R} \approx \mathbf{P}\mathbf{Q}^\top$ — a low-rank approximation.

The latent dimensions often become interpretable after training: one might separate serious dramas from light comedies; another, films for children from films for adults. Nobody labels these dimensions; they emerge from the data.

## Why not just use the SVD?

Classical SVD requires a **complete** matrix. Filling missing entries with zeros or means would treat "unseen" as "disliked" and swamp the model with fake data. Instead we fit only the **observed** entries $\mathcal{K}$:

$$
\min_{\mathbf{P}, \mathbf{Q}, \mathbf{b}}\sum_{(u,i) \in \mathcal{K}}\big(r_{ui} - \hat{r}_{ui}\big)^2 + \lambda\left(\|\mathbf{p}_u\|^2 + \|\mathbf{q}_i\|^2 + b_u^2 + b_i^2\right)
$$

Regularisation is essential: with many parameters and sparse data, the model would otherwise memorise the observed ratings.

## Training method 1: stochastic gradient descent

For each observed rating, compute the error $e_{ui} = r_{ui} - \hat{r}_{ui}$ and update:

$$
\mathbf{p}_u \leftarrow \mathbf{p}_u + \eta(e_{ui}\mathbf{q}_i - \lambda\mathbf{p}_u), \qquad \mathbf{q}_i \leftarrow \mathbf{q}_i + \eta(e_{ui}\mathbf{p}_u - \lambda\mathbf{q}_i)
$$

(and similarly for biases). This is Simon Funk's famous approach from the Netflix Prize.

```python
import numpy as np

def train_mf(ratings, n_users, n_items, k=20, lr=0.01, reg=0.05, epochs=30, seed=0):
    rng = np.random.default_rng(seed)
    P = 0.1 * rng.normal(size=(n_users, k)); Q = 0.1 * rng.normal(size=(n_items, k))
    bu, bi = np.zeros(n_users), np.zeros(n_items)
    mu = np.mean([r for _, _, r in ratings])
    for epoch in range(epochs):
        rng.shuffle(ratings)
        for u, i, r in ratings:
            e = r - (mu + bu[u] + bi[i] + P[u] @ Q[i])
            bu[u] += lr * (e - reg * bu[u]); bi[i] += lr * (e - reg * bi[i])
            P[u], Q[i] = P[u] + lr * (e * Q[i] - reg * P[u]), Q[i] + lr * (e * P[u] - reg * Q[i])
    return mu, bu, bi, P, Q

# Synthetic data with a true rank-3 structure
rng = np.random.default_rng(1)
U, I, K = 300, 200, 3
true = rng.normal(size=(U, K)) @ rng.normal(size=(I, K)).T
obs = [(u, i, float(np.clip(3 + true[u, i] + rng.normal(0, .3), 1, 5)))
       for u in range(U) for i in range(I) if rng.random() < 0.05]
split = int(0.8 * len(obs)); train, test = obs[:split], obs[split:]
mu, bu, bi, P, Q = train_mf(list(train), U, I)
rmse = np.sqrt(np.mean([(r - (mu + bu[u] + bi[i] + P[u] @ Q[i])) ** 2 for u, i, r in test]))
base = np.sqrt(np.mean([(r - mu) ** 2 for _, _, r in test]))
print(f"test RMSE: MF={rmse:.3f}  global-mean baseline={base:.3f}")
```

## Training method 2: alternating least squares (ALS)

The objective is not jointly convex in $\mathbf{P}$ and $\mathbf{Q}$, but it **is** convex in each with the other fixed. ALS alternates: fix $\mathbf{Q}$ and solve a ridge regression for every user vector in closed form,

$$
\mathbf{p}_u = \left(\mathbf{Q}_{\mathcal{I}_u}^\top\mathbf{Q}_{\mathcal{I}_u} + \lambda\mathbf{I}\right)^{-1}\mathbf{Q}_{\mathcal{I}_u}^\top\mathbf{r}_u
$$

then fix $\mathbf{P}$ and solve for every item. Each half-step is embarrassingly parallel, which is why ALS is the standard in distributed systems such as Spark MLlib.

## Implicit feedback

Hu, Koren and Volinsky (2008) adapted ALS to implicit data: treat every user–item pair as a **preference** $p_{ui} \in \{0, 1\}$ (interacted or not) with a **confidence** $c_{ui} = 1 + \alpha\,(\text{interaction count})$. All pairs enter the loss, weighted by confidence, and clever algebra keeps ALS efficient despite the dense matrix. Alternatives train with **ranking losses** such as **Bayesian Personalised Ranking (BPR)**, which learns that observed items should score higher than sampled unobserved ones.

## From factorisation to deep recommenders

- **Factorisation machines** generalise MF to arbitrary features (user attributes, context) by modelling pairwise feature interactions with latent vectors.
- **Two-tower models** use neural networks to map user features and item features into the same embedding space — MF with learned encoders. They handle cold start (new items have features) and power large-scale candidate retrieval via approximate nearest-neighbour search.
- **Sequential recommenders** (e.g. transformer-based) model the order of a user's interactions.

:::note
The dot product $\mathbf{p}_u^\top\mathbf{q}_i$ is the same operation used for retrieval in semantic search and for attention in transformers. Learning to place related things close together in a vector space is one of the unifying ideas of modern ML.
:::

:::exercise
1. Derive the SGD update for $\mathbf{p}_u$ from the regularised squared error.
2. Implement one ALS iteration for the synthetic data and compare convergence with SGD.
3. After training, find the five nearest item vectors to a chosen item. Do they share a true latent structure?
:::

:::takeaway
- MF predicts $\hat{r}_{ui} = \mu + b_u + b_i + \mathbf{p}_u^\top\mathbf{q}_i$ from latent user and item vectors.
- Fit only observed entries with regularisation — not a plain SVD.
- Train with SGD or ALS; ALS parallelises well; implicit data needs confidence weighting or ranking losses.
- Two-tower neural models extend MF with features and power embedding retrieval.
:::

=== POST ===
slug: time-series-forecasting-arima
title: Time Series Forecasting: Stationarity, ARIMA and Evaluation
category: machine-learning
level: Intermediate
tags: time series, arima, forecasting, stationarity, seasonality
summary: Forecasting demand, rainfall or arrivals needs models that respect time. We decompose series into trend and seasonality, test for stationarity, build ARIMA and SARIMA models, and evaluate with rolling-origin backtesting.
---
Forecasting is everywhere: electricity demand, crop prices, hospital admissions, the number of new arrivals a reception centre should prepare for next month. Time series differ from ordinary tabular data in one crucial way: **observations are ordered and dependent**. Shuffling them destroys information, and random train–test splits leak the future. Today we learn the classical statistical approach — still a strong baseline that every forecaster must know.

## Components of a time series

A series $y_t$ is often described as a combination of:

- **Trend** — long-term increase or decrease;
- **Seasonality** — regular patterns with a fixed period (weekly, yearly);
- **Cycles** — longer, irregular rises and falls (economic cycles);
- **Noise** — the unpredictable remainder.

**Decomposition** (e.g. STL: Seasonal–Trend decomposition using Loess) separates these components and is the first plot to make for any new series.

## Stationarity

A series is (weakly) **stationary** if its mean, variance and autocovariance do not change over time. Most classical models assume stationarity. Trends and seasonality violate it.

Tools:

- **Plots** of the series and its rolling mean/variance.
- **Augmented Dickey–Fuller (ADF) test** — null hypothesis: a unit root (non-stationary).
- **Differencing** — $\nabla y_t = y_t - y_{t-1}$ removes trends; seasonal differencing $y_t - y_{t-s}$ removes seasonality of period $s$.
- **Log or Box–Cox transform** — stabilises growing variance.

## Autocorrelation

The **autocorrelation function (ACF)** measures correlation between $y_t$ and $y_{t-k}$ for each lag $k$. The **partial autocorrelation function (PACF)** measures correlation at lag $k$ after removing the effect of shorter lags. Their patterns guide model choice.

## The ARIMA family

**AR($p$) — autoregressive:** today depends linearly on the last $p$ values:

$$
y_t = c + \phi_1y_{t-1} + \dots + \phi_py_{t-p} + \varepsilon_t
$$

**MA($q$) — moving average:** today depends on the last $q$ forecast errors:

$$
y_t = c + \varepsilon_t + \theta_1\varepsilon_{t-1} + \dots + \theta_q\varepsilon_{t-q}
$$

**ARIMA($p, d, q$)** applies an ARMA($p, q$) model to the series differenced $d$ times. **SARIMA($p,d,q$)($P,D,Q$)$_s$** adds seasonal AR, differencing and MA terms at lag $s$. Adding external regressors (holidays, rainfall, prices) gives **SARIMAX**.

Identification heuristics:

| Pattern | Suggests |
|---|---|
| PACF cuts off after lag $p$, ACF decays | AR($p$) |
| ACF cuts off after lag $q$, PACF decays | MA($q$) |
| Both decay | ARMA; select by AIC |

In practice, fit a few candidates and choose by **AIC/BIC**, or use automatic search (e.g. `pmdarima.auto_arima`). Then check that **residuals look like white noise** (Ljung–Box test, ACF of residuals).

```python
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

rng = np.random.default_rng(0)
t = np.arange(120)                                  # 10 years of monthly data
y = 200 + 1.5 * t + 25 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 8, 120)
series = pd.Series(y, index=pd.date_range("2016-01-01", periods=120, freq="MS"))

print("ADF p-value (raw):", round(adfuller(series)[1], 3))
print("ADF p-value (diff):", round(adfuller(series.diff().dropna())[1], 3))

train, test = series[:-12], series[-12:]
model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit(disp=False)
fc = model.get_forecast(12)
pred, ci = fc.predicted_mean, fc.conf_int(alpha=0.05)

naive = train.iloc[-12:].values                     # seasonal naive: same month last year
mae = lambda a, b: np.mean(np.abs(np.asarray(a) - np.asarray(b)))
print(f"SARIMA MAE: {mae(test, pred):.2f}   seasonal-naive MAE: {mae(test, naive):.2f}")
print("95% interval for first forecast month:", ci.iloc[0].round(1).tolist())
```

## Evaluating forecasts properly

:::warning
Never evaluate a forecaster with random cross-validation. Use **rolling-origin (walk-forward) backtesting**: train on data up to time $T$, forecast $T+1 \dots T+h$, move $T$ forward, repeat. Report errors per forecast horizon, because accuracy degrades as you look further ahead.
:::

Always compare against **simple baselines**: the naive forecast (last value), the seasonal naive forecast (same period last season) and a moving average. Surprisingly often, sophisticated models barely beat them. Metrics: MAE, RMSE, and scale-free MASE (error relative to the naive forecast) for comparing across series.

## Beyond ARIMA

- **Exponential smoothing (ETS / Holt–Winters)** — weighted averages with trend and seasonality; excellent for many business series.
- **Prophet** — additive model with trend changepoints and holidays; easy to use.
- **Machine learning on lag features** — gradient boosting with lags, rolling statistics and calendar features; strong when many related series are available (**global models**).
- **Deep learning** — RNNs, temporal convolution, transformer forecasters and pretrained time-series foundation models.

Large forecasting competitions (the M-competitions) have repeatedly shown that **combinations** of methods and well-tuned global ML models perform very well, while simple statistical methods remain hard to beat for individual short series.

:::exercise
1. Decompose a real monthly series (e.g. rainfall or airline passengers) with STL and describe its components.
2. Implement rolling-origin evaluation for SARIMA and the seasonal naive baseline over the last 36 months.
3. Build a gradient-boosting forecaster with lags 1, 2, 12 and month-of-year features and compare it with SARIMA.
:::

:::takeaway
- Time series have order and dependence; decompose into trend, seasonality and noise.
- Make series stationary with transforms and differencing; use ACF/PACF to guide ARIMA orders.
- SARIMA(X) handles seasonality and external regressors; check residuals are white noise.
- Evaluate with walk-forward backtesting against naive baselines.
:::

=== POST ===
slug: semi-supervised-learning
title: Semi-Supervised Learning: Learning from Few Labels and Many Unlabelled Examples
category: machine-learning
level: Intermediate
tags: semi-supervised, pseudo-labelling, consistency regularization, label propagation
summary: Labels are expensive; unlabelled data is cheap. We study the assumptions that make unlabelled data useful and the main techniques — self-training, label propagation, consistency regularisation and FixMatch.
---
Labelling 100,000 medical images might require months of specialists' time; collecting the images themselves is easy. **Semi-supervised learning (SSL)** uses a small labelled set together with a large unlabelled set. When it works, it can approach fully supervised performance with a fraction of the labels. But it does not always work, and understanding **why** is the key.

## When can unlabelled data help?

Unlabelled data tells us about $p(\mathbf{x})$, the distribution of inputs. This helps predict $p(y \mid \mathbf{x})$ only if the two are linked. Three standard assumptions express such links:

1. **Smoothness assumption** — points close in a high-density region should share labels.
2. **Cluster assumption** — data forms clusters, and points in the same cluster tend to share a class. Equivalently, the decision boundary should pass through **low-density** regions.
3. **Manifold assumption** — data lies near a low-dimensional manifold, and labels vary smoothly along it.

:::warning
If these assumptions fail — for example, if classes overlap heavily in input space — unlabelled data can **hurt**, pulling the boundary to a wrong low-density region. Always compare against a supervised baseline trained on the labelled data alone.
:::

## Self-training (pseudo-labelling)

The simplest method:

1. Train a model on labelled data.
2. Predict the unlabelled data; take the most **confident** predictions as pseudo-labels.
3. Add them to the training set and retrain. Repeat.

Risk: **confirmation bias** — early mistakes get reinforced. Mitigate with high confidence thresholds, class-balanced selection and re-evaluation of pseudo-labels each round.

## Graph-based methods: label propagation

Build a similarity graph over all points (labelled and unlabelled). Labels **spread** along edges: each unlabelled node iteratively takes a weighted average of its neighbours' label distributions, while labelled nodes stay fixed. This directly implements the smoothness and cluster assumptions. It works well for moderate dataset sizes with a meaningful distance.

```python
import numpy as np
from sklearn.datasets import make_moons
from sklearn.semi_supervised import LabelSpreading, SelfTrainingClassifier
from sklearn.svm import SVC

X, y = make_moons(n_samples=1000, noise=0.08, random_state=0)
rng = np.random.default_rng(0)
y_partial = np.full_like(y, -1)                         # -1 means unlabelled
labelled = np.r_[rng.choice(np.where(y == 0)[0], 5, replace=False),
                 rng.choice(np.where(y == 1)[0], 5, replace=False)]
y_partial[labelled] = y[labelled]                       # only 10 labels!

sup = SVC(probability=True, gamma=2).fit(X[labelled], y[labelled])
print("supervised (10 labels):", round((sup.predict(X) == y).mean(), 3))

ls = LabelSpreading(kernel="knn", n_neighbors=10).fit(X, y_partial)
print("label spreading:       ", round((ls.transduction_ == y).mean(), 3))

st = SelfTrainingClassifier(SVC(probability=True, gamma=2), threshold=0.9).fit(X, y_partial)
print("self-training:         ", round((st.predict(X) == y).mean(), 3))
```

With only ten labels, label spreading follows the two crescent shapes and labels nearly everything correctly — the cluster assumption holds perfectly here.

## Consistency regularisation

Modern deep SSL relies on a powerful idea: a model's prediction should **not change** under realistic perturbations of the input. For an unlabelled example $\mathbf{x}$ and augmentations $\alpha(\cdot)$, add a loss

$$
\mathcal{L}_{\text{cons}} = \big\|f(\alpha_1(\mathbf{x})) - f(\alpha_2(\mathbf{x}))\big\|^2
$$

This pushes the decision boundary away from dense regions (where small perturbations would flip predictions). Methods include the Π-model, Mean Teacher (the target comes from an exponential moving average of the model's weights) and Virtual Adversarial Training.

## FixMatch: a simple, strong recipe

**FixMatch** (Sohn et al., 2020) combines pseudo-labelling and consistency:

1. For an unlabelled image, predict on a **weakly** augmented version (flip, crop).
2. If the maximum probability exceeds a threshold (e.g. 0.95), use its argmax as a pseudo-label.
3. Train the model to predict that pseudo-label on a **strongly** augmented version (colour distortion, cutout).

The loss is

$$
\mathcal{L} = \mathcal{L}_s + \lambda_u\frac{1}{\mu B}\sum_{b}\mathbb{1}\big[\max q_b \ge \tau\big]\;\text{CE}\big(\hat{q}_b,\, f(\mathcal{A}_{\text{strong}}(\mathbf{x}_b))\big)
$$

On CIFAR-10, FixMatch reported strong accuracy using only a few labels per class — a dramatic demonstration of SSL's potential.

## Self-supervised pretraining: the other route

Today, a very common way to exploit unlabelled data is **self-supervised pretraining** (contrastive learning, masked modelling) followed by fine-tuning on the few labels. SSL and self-supervision are complementary: pretrain on everything, then apply semi-supervised fine-tuning.

## Practical checklist

- Ensure labelled and unlabelled data come from the **same distribution**; class-distribution mismatch or out-of-distribution unlabelled data harms SSL.
- Keep a **labelled validation set** — small but trustworthy — to verify SSL actually helps.
- Start simple: pseudo-labelling with a strong base model, or pretrained embeddings + label propagation.

:::exercise
1. Increase the noise in `make_moons` until the clusters overlap. When does label spreading stop helping?
2. Implement self-training manually with a confidence threshold and track pseudo-label accuracy each round.
3. Explain why weak/strong augmentation in FixMatch reduces confirmation bias.
:::

:::takeaway
- Unlabelled data helps only when $p(\mathbf{x})$ carries information about $p(y \mid \mathbf{x})$ (smoothness, cluster, manifold assumptions).
- Self-training, label propagation and consistency regularisation are the core techniques.
- FixMatch unites pseudo-labelling with weak/strong augmentation consistency.
- Always compare with a supervised baseline; SSL can hurt when assumptions fail.
:::

=== POST ===
slug: active-learning
title: Active Learning: Letting the Model Choose What to Label
category: machine-learning
level: Intermediate
tags: active learning, uncertainty sampling, labeling, human in the loop
summary: If labelling is expensive, label the most informative examples first. We cover pool-based active learning, uncertainty and diversity sampling, query-by-committee, and the practical pitfalls of real annotation loops.
---
You have 200,000 unlabelled text messages and a budget to label 2,000. Which 2,000? Picking at random wastes effort on easy, redundant examples. **Active learning** lets the model choose the examples whose labels would teach it the most. In many applications it reaches a target accuracy with a fraction of the labels needed by random sampling.

## The pool-based loop

1. Start with a small labelled seed set $\mathcal{L}$ and a large unlabelled pool $\mathcal{U}$.
2. Train a model on $\mathcal{L}$.
3. Use a **query strategy** to score examples in $\mathcal{U}$ by informativeness.
4. Send the top examples (a **batch**) to human annotators.
5. Move them from $\mathcal{U}$ to $\mathcal{L}$; retrain; repeat until the budget is spent or performance plateaus.

## Uncertainty sampling

Query the examples the model is least sure about. With predicted class probabilities $p(y \mid \mathbf{x})$:

- **Least confidence:** $1 - \max_y p(y \mid \mathbf{x})$.
- **Margin:** $p(y_1 \mid \mathbf{x}) - p(y_2 \mid \mathbf{x})$ between the top two classes (smaller = more uncertain).
- **Entropy:** $H = -\sum_y p(y \mid \mathbf{x})\log p(y \mid \mathbf{x})$.

For a linear classifier, uncertainty sampling picks points near the decision boundary — exactly where labels refine the boundary most.

## Query-by-committee and Bayesian approaches

Train a **committee** of models (bootstrap samples, different seeds or architectures) and query examples on which they **disagree** most (vote entropy or KL divergence from the consensus). For neural networks, **BALD** (Bayesian Active Learning by Disagreement) selects points maximising mutual information between the prediction and the model parameters:

$$
I(y; \boldsymbol{\theta} \mid \mathbf{x}) = H\big[\mathbb{E}_{\boldsymbol{\theta}}\,p(y \mid \mathbf{x}, \boldsymbol{\theta})\big] - \mathbb{E}_{\boldsymbol{\theta}}\,H\big[p(y \mid \mathbf{x}, \boldsymbol{\theta})\big]
$$

It favours points where the model's *parameters* are uncertain (reducible, **epistemic** uncertainty) rather than points that are inherently ambiguous (irreducible, **aleatoric** noise) — an important distinction, since labelling a truly ambiguous example teaches little.

## Diversity and representativeness

Pure uncertainty sampling in batches picks many near-duplicate examples from the same confusing region, and it may favour outliers. Better batch strategies combine uncertainty with **diversity**:

- cluster the uncertain candidates and pick one per cluster;
- **core-set** selection — choose points that best cover the data in embedding space;
- **BADGE** — sample diverse points in the space of gradient embeddings, balancing uncertainty and diversity.

## A simulation

```python
import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X, y = load_digits(return_X_y=True); X = X / 16
X_pool, X_test, y_pool, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)

def run(strategy, budget=300, batch=10, seed=0):
    rng = np.random.default_rng(seed)
    labelled = list(rng.choice(len(X_pool), 20, replace=False))
    curve = []
    while len(labelled) <= budget:
        m = LogisticRegression(max_iter=2000).fit(X_pool[labelled], y_pool[labelled])
        curve.append((len(labelled), m.score(X_test, y_test)))
        rest = np.setdiff1d(np.arange(len(X_pool)), labelled)
        if strategy == "random":
            pick = rng.choice(rest, batch, replace=False)
        else:                                       # margin uncertainty
            P = np.sort(m.predict_proba(X_pool[rest]), axis=1)
            pick = rest[np.argsort(P[:, -1] - P[:, -2])[:batch]]
        labelled += list(pick)
    return curve

for s in ["random", "margin"]:
    c = run(s)
    print(s, [f"{n}:{a:.3f}" for n, a in c[::7]])
```

Margin sampling typically reaches a given accuracy with noticeably fewer labels than random sampling.

## Real-world pitfalls

:::warning
1. **Sampling bias** — actively selected data is not a random sample. Evaluate on a **separately drawn random test set**, never on the actively collected data.
2. **Cold start** — with very few labels, the model's uncertainty is unreliable; begin with random or diversity-based sampling.
3. **Annotator cost is not uniform** — uncertain examples are often harder and slower (or less accurate) to label. Consider cost-aware selection and annotator agreement.
4. **Model dependence** — data selected for one model may be less useful for another model you train later.
5. **Class imbalance** — active learning is excellent at finding rare-class examples, but monitor class balance.
:::

## Where active learning shines

- Medical imaging and document review, where expert labels are costly.
- Low-resource language tasks, where annotators are scarce.
- Building classifiers for new categories of humanitarian feedback or incident reports, where the label set evolves.
- Combined with **pretrained embeddings** or **semi-supervised learning** — use unlabelled data for representations and active learning for label efficiency.

:::exercise
1. Add an entropy strategy to the simulation and compare it with margin sampling.
2. Implement a diversity-aware batch strategy: take the 100 most uncertain examples, cluster them into 10 groups with k-means, and pick the most uncertain point in each.
3. Explain why the actively labelled set should not be used as the test set.
:::

:::takeaway
- Active learning queries the most informative examples to save labelling cost.
- Uncertainty (least confidence, margin, entropy), committee disagreement and BALD measure informativeness.
- Combine uncertainty with diversity for batch selection.
- Evaluate on an independent random sample; beware cold start and annotation costs.
:::

=== POST ===
slug: learning-theory-pac-vc-dimension
title: "Learning Theory: PAC Learning, VC Dimension and Generalisation Bounds"
category: machine-learning
level: Advanced
tags: learning theory, pac, vc dimension, generalization bounds, sample complexity
summary: Why should a model that fits training data work on new data? Learning theory answers precisely. We develop PAC learning, finite-class bounds, the VC dimension, and discuss what these bounds do and do not explain about deep learning.
---
Every practical technique in this track — validation, regularisation, simpler models — rests on an implicit promise: that performance on a sample predicts performance on the population. **Statistical learning theory** makes this promise precise. It tells us how many examples are needed, how model complexity affects generalisation, and why learning is possible at all. This is the most mathematical lecture of the track; take it slowly.

## The PAC framework

Leslie Valiant (1984) introduced **Probably Approximately Correct (PAC)** learning. Let $R(h)$ be the true error of hypothesis $h$ and $\hat{R}(h)$ its training error on $n$ i.i.d. examples. A hypothesis class $\mathcal{H}$ is **PAC-learnable** if there is an algorithm that, for any $\epsilon, \delta \in (0, 1)$ and any data distribution, given

$$
n \ge n(\epsilon, \delta)
$$

examples, outputs $h$ with

$$
P\big(R(h) \le \epsilon\big) \ge 1 - \delta
$$

"Approximately correct" = error at most $\epsilon$; "probably" = with confidence $1 - \delta$. The function $n(\epsilon, \delta)$ is the **sample complexity**.

## Finite hypothesis classes

**Realisable case** (some $h \in \mathcal{H}$ has zero error). Any consistent hypothesis (zero training error) has true error at most $\epsilon$ with probability $1 - \delta$ if

$$
n \ge \frac{1}{\epsilon}\left(\ln|\mathcal{H}| + \ln\frac{1}{\delta}\right)
$$

*Proof sketch.* A "bad" hypothesis with error $> \epsilon$ survives $n$ independent examples with probability $< (1 - \epsilon)^n \le e^{-\epsilon n}$. By the union bound over at most $|\mathcal{H}|$ bad hypotheses, the probability that any survives is $\le |\mathcal{H}|e^{-\epsilon n}$. Set this $\le \delta$ and solve for $n$. $\blacksquare$

**Agnostic case** (no perfect hypothesis). Using Hoeffding's inequality and a union bound, with probability $1 - \delta$, **for all** $h \in \mathcal{H}$ simultaneously:

$$
R(h) \le \hat{R}(h) + \sqrt{\frac{\ln|\mathcal{H}| + \ln(2/\delta)}{2n}}
$$

This is the first **generalisation bound**: true error ≤ training error + a complexity term that grows with $\ln|\mathcal{H}|$ and shrinks as $1/\sqrt{n}$. It formalises the bias–variance trade-off: a richer class lowers training error but raises the complexity penalty.

## Infinite classes: the VC dimension

Linear classifiers form an infinite class, so $\ln|\mathcal{H}|$ is useless. Vapnik and Chervonenkis measured capacity differently.

A set of points is **shattered** by $\mathcal{H}$ if, for **every** possible labelling of those points, some $h \in \mathcal{H}$ realises it. The **VC dimension** $d_{VC}(\mathcal{H})$ is the size of the largest set that can be shattered.

| Hypothesis class | VC dimension |
|---|---|
| Thresholds on the real line | 1 |
| Intervals on the real line | 2 |
| Linear classifiers (half-planes) in $\mathbb{R}^2$ | 3 |
| Linear classifiers in $\mathbb{R}^d$ | $d + 1$ |
| Axis-aligned rectangles in $\mathbb{R}^2$ | 4 |
| $\sin(\omega x)$ thresholds | $\infty$ (despite one parameter!) |

:::example
Half-planes in 2-D can shatter 3 points in general position (all $2^3 = 8$ labellings are achievable), but no set of 4 points: for 4 points in convex position, labelling opposite corners the same (an XOR pattern) cannot be separated by a line. So $d_{VC} = 3$.
:::

The last row is a warning: **the number of parameters is not the same as capacity**.

## The VC bound

With probability $1 - \delta$, for all $h \in \mathcal{H}$:

$$
R(h) \le \hat{R}(h) + O\left(\sqrt{\frac{d_{VC}\ln(n/d_{VC}) + \ln(1/\delta)}{n}}\right)
$$

The **fundamental theorem of statistical learning** states that a class is PAC-learnable **if and only if** its VC dimension is finite, with sample complexity $\Theta\left(\frac{d_{VC} + \ln(1/\delta)}{\epsilon^2}\right)$ in the agnostic case. A rough rule of thumb from this theory: you need a number of examples that is a multiple of the VC dimension.

## Structural Risk Minimisation

Vapnik's **SRM** principle chooses among nested classes $\mathcal{H}_1 \subset \mathcal{H}_2 \subset \dots$ by minimising the **bound** (training error + complexity term), not the training error alone. Regularisation, SVM margin maximisation (large margins imply lower effective capacity) and model selection by penalised criteria are practical descendants of SRM.

```python
import numpy as np
# How the complexity term shrinks with n for a finite class (|H| = 1e6, delta = 0.05)
H, delta = 1e6, 0.05
for n in [100, 1_000, 10_000, 100_000]:
    gap = np.sqrt((np.log(H) + np.log(2 / delta)) / (2 * n))
    print(f"n={n:>7}: training error + {gap:.3f} bounds the true error")
```

## Other complexity measures

- **Rademacher complexity** — how well $\mathcal{H}$ can fit random ±1 labels on the actual data; data-dependent and often tighter.
- **Margin bounds** — generalisation depends on the margin relative to the data's scale, not dimension (explains SVMs in high dimensions).
- **PAC-Bayes bounds** — depend on the KL divergence between a posterior over hypotheses and a prior; among the few bounds that give non-vacuous numbers for some deep networks.
- **Algorithmic stability** — algorithms whose output changes little when one example changes generalise well.

## The deep learning puzzle

Modern networks have far more parameters than training examples and can fit **randomly labelled** data perfectly (Zhang et al., 2017) — so their capacity by VC or Rademacher standards is enormous, and classical bounds are **vacuous** (they bound the error by more than 100%). Yet the same networks generalise well on real labels. Explanations under active research include the implicit bias of SGD towards simple solutions, flat minima, norm-based and compression-based bounds, and the structure of natural data. Classical theory is not wrong — its bounds are valid — but they are too loose to explain deep learning's success.

:::note
Learning theory's lasting practical lessons are qualitative and robust: error on a sample converges to the true error at rate about $1/\sqrt{n}$; capacity must be controlled relative to data; and evaluation on held-out data is the reliable guide when bounds are loose.
:::

:::exercise
1. How many examples guarantee error ≤ 0.05 with probability 0.99 for a consistent learner over $|\mathcal{H}| = 2^{20}$ hypotheses?
2. Prove that intervals on the real line have VC dimension 2.
3. Train a small neural network on MNIST with randomly shuffled labels. Does training accuracy reach 100%? What does this imply about capacity?
:::

:::takeaway
- PAC learning formalises "probably approximately correct" with sample complexity $n(\epsilon, \delta)$.
- For finite classes, the generalisation gap scales as $\sqrt{\ln|\mathcal{H}|/n}$.
- The VC dimension measures capacity for infinite classes; finite VC dimension ⇔ PAC-learnable.
- Classical bounds are vacuous for deep networks, which remains an open research question.
:::

=== POST ===
slug: association-rule-mining-apriori
title: Association Rule Mining: Apriori, FP-Growth and Market-Basket Analysis
category: machine-learning
level: Beginner
tags: association rules, apriori, fp-growth, market basket, unsupervised
summary: Which items appear together? Association rule mining discovers patterns like "bread and butter imply milk". We define support, confidence and lift, derive the Apriori algorithm, and interpret rules responsibly.
---
A supermarket notices that customers who buy lentils and rice often buy cooking oil. A pharmacy sees certain medicines prescribed together. A learning platform finds that students who complete the statistics module usually attempt the machine learning module next. **Association rule mining** finds such co-occurrence patterns in large collections of transactions. It is an unsupervised, highly interpretable technique and one of the classic methods of data mining.

## Terminology

- A **transaction** is a set of items (one shopping basket, one patient's prescriptions).
- An **itemset** is any set of items, e.g. {rice, lentils}.
- An **association rule** $X \Rightarrow Y$ says: transactions containing $X$ tend to contain $Y$ ($X \cap Y = \emptyset$).

## Measuring rules

For $N$ transactions:

$$
\text{support}(X) = \frac{\#\{\text{transactions containing } X\}}{N}
$$

$$
\text{confidence}(X \Rightarrow Y) = \frac{\text{support}(X \cup Y)}{\text{support}(X)} = \hat{P}(Y \mid X)
$$

$$
\text{lift}(X \Rightarrow Y) = \frac{\text{confidence}(X \Rightarrow Y)}{\text{support}(Y)} = \frac{\hat{P}(X, Y)}{\hat{P}(X)\,\hat{P}(Y)}
$$

- **Support** — how common the pattern is.
- **Confidence** — how often the rule is right when it applies.
- **Lift** — how much more likely $Y$ is given $X$ than in general. Lift > 1 indicates positive association; = 1 independence; < 1 negative association.

:::warning
High confidence alone is misleading. If 90% of all baskets contain tea, then *any* rule "$X \Rightarrow$ tea" has confidence around 90% even when $X$ has nothing to do with tea. Always check **lift** (or leverage/conviction), which compares against the baseline popularity of $Y$.
:::

## The mining problem

Find all rules with support ≥ `min_support` and confidence ≥ `min_confidence`. The difficulty: with $d$ items there are $2^d$ possible itemsets. For a shop with 10,000 products, brute force is impossible.

The solution splits into two steps:

1. Find all **frequent itemsets** (support ≥ threshold).
2. Generate high-confidence rules from them.

## The Apriori principle

**Every subset of a frequent itemset is frequent.** Equivalently: if an itemset is infrequent, all its supersets are infrequent. This **anti-monotonicity** of support lets us prune the search massively.

**Apriori algorithm** (Agrawal & Srikant, 1994):

1. Count single items; keep frequent ones ($L_1$).
2. For $k = 2, 3, \dots$: generate candidate $k$-itemsets by joining frequent $(k-1)$-itemsets; **prune** any candidate with an infrequent $(k-1)$-subset; scan the data to count support; keep the frequent ones ($L_k$).
3. Stop when no new frequent itemsets appear.

Its weakness is repeated database scans and potentially huge candidate sets.

## FP-Growth

**FP-Growth** (Han et al., 2000) compresses the database into a prefix tree (the **FP-tree**), where transactions sharing frequent items share paths. It then mines frequent itemsets recursively from conditional trees **without candidate generation**, needing only two scans of the data. It is usually much faster than Apriori on large datasets.

## Example

```python
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules

transactions = [
    ["rice", "lentils", "oil", "salt"],
    ["rice", "lentils", "oil"],
    ["rice", "oil", "sugar"],
    ["lentils", "oil", "onion"],
    ["rice", "lentils", "onion", "oil"],
    ["tea", "sugar", "milk"],
    ["tea", "sugar", "biscuits"],
    ["tea", "milk", "biscuits", "sugar"],
    ["rice", "lentils", "salt"],
    ["tea", "sugar"],
]
te = TransactionEncoder()
df = pd.DataFrame(te.fit_transform(transactions), columns=te.columns_)

itemsets = fpgrowth(df, min_support=0.3, use_colnames=True)
rules = association_rules(itemsets, metric="confidence", min_threshold=0.7)
cols = ["antecedents", "consequents", "support", "confidence", "lift"]
print(rules[cols].sort_values("lift", ascending=False).round(2).to_string(index=False))
```

Rules such as {rice, lentils} ⇒ {oil} and {tea} ⇒ {sugar} emerge with lift well above 1.

## Applications

- **Retail**: store layout, cross-selling, promotions, bundle design.
- **Healthcare**: co-occurring diagnoses or adverse drug combinations (for further study, not conclusions).
- **Web usage mining**: pages visited together; navigation design.
- **Education**: modules or resources that students use together.
- **Supply planning**: items requested together at distribution points, helping pack kits efficiently.

## Interpreting rules responsibly

- **Association is not causation.** Buying a phone case does not cause buying a phone; they co-occur.
- **Multiple testing**: mining thousands of rules guarantees some spurious ones. Validate rules on held-out transactions or over time.
- **Thresholds** strongly shape the output: too high misses interesting niche patterns; too low floods analysts with trivial rules.
- **Redundancy**: many rules restate each other; summarise with closed or maximal itemsets.

:::tip
Present only a small number of rules ranked by lift (with minimum support), and discuss them with domain experts. The value of association mining is in the conversations it starts, not in the raw rule list.
:::

:::exercise
1. Compute support, confidence and lift for {tea} ⇒ {sugar} by hand from the example transactions.
2. Show why the Apriori principle holds: prove that support is anti-monotone.
3. Run FP-Growth on a public groceries dataset and report the five rules with the highest lift among those with support ≥ 1%.
:::

:::takeaway
- Association rules $X \Rightarrow Y$ are evaluated by support, confidence and lift.
- Confidence misleads for popular consequents — check lift.
- Apriori prunes candidates using anti-monotonicity; FP-Growth avoids candidate generation with an FP-tree.
- Rules describe co-occurrence, not causation; validate and interpret with domain experts.
:::
