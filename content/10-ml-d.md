=== POST ===
slug: k-means-clustering
title: k-Means Clustering: Algorithm, Objective and Pitfalls
category: machine-learning
level: Beginner
tags: clustering, k-means, unsupervised learning, elbow method, silhouette
summary: k-means partitions data into k groups by alternating assignment and update steps. We derive it as coordinate descent, cover k-means++ initialisation, choosing k, and the assumptions that make it fail.
---
A humanitarian organisation wants to place a limited number of distribution points so that every household is as close as possible to one. A marketing team wants to group customers by behaviour. Both are **clustering** problems, and the first algorithm everyone reaches for is **k-means**. It is simple, fast and often effective — but its assumptions are strong, and you must know when they break.

## The objective

Given points $\mathbf{x}_1, \dots, \mathbf{x}_n$ and a number of clusters $k$, find centroids $\boldsymbol{\mu}_1, \dots, \boldsymbol{\mu}_k$ and assignments $c_i \in \{1, \dots, k\}$ minimising the **within-cluster sum of squares (inertia)**:

$$
J = \sum_{i=1}^{n}\|\mathbf{x}_i - \boldsymbol{\mu}_{c_i}\|^2
$$

Minimising $J$ exactly is NP-hard. **Lloyd's algorithm** finds a local minimum.

## Lloyd's algorithm

1. **Initialise** $k$ centroids.
2. **Assignment step**: assign each point to its nearest centroid: $c_i = \arg\min_j\|\mathbf{x}_i - \boldsymbol{\mu}_j\|^2$.
3. **Update step**: move each centroid to the mean of its assigned points: $\boldsymbol{\mu}_j = \frac{1}{|C_j|}\sum_{i \in C_j}\mathbf{x}_i$.
4. Repeat until assignments stop changing.

**Why it converges:** it is **coordinate descent** on $J$. The assignment step minimises $J$ over assignments with centroids fixed; the update step minimises $J$ over centroids with assignments fixed (the mean minimises squared distances). $J$ never increases, and there are finitely many partitions, so the algorithm terminates — at a local, not necessarily global, minimum. Each iteration costs $O(nkd)$.

```python
import numpy as np

def kmeans(X, k, iters=100, seed=0):
    rng = np.random.default_rng(seed)
    centroids = X[rng.choice(len(X), k, replace=False)]
    for _ in range(iters):
        d2 = ((X[:, None, :] - centroids[None]) ** 2).sum(-1)     # (n, k)
        labels = d2.argmin(1)
        new = np.array([X[labels == j].mean(0) if np.any(labels == j) else centroids[j]
                        for j in range(k)])
        if np.allclose(new, centroids):
            break
        centroids = new
    inertia = ((X - centroids[labels]) ** 2).sum()
    return labels, centroids, inertia
```

## Initialisation matters: k-means++

Random initial centroids can land in the same cluster, producing poor local optima. **k-means++** (Arthur & Vassilvitskii, 2007) spreads them out:

1. Choose the first centroid uniformly at random from the data.
2. Choose each next centroid with probability proportional to $D(\mathbf{x})^2$, the squared distance to the nearest centroid already chosen.

This gives an expected objective within a factor $O(\log k)$ of optimal, and in practice converges faster and better. Scikit-learn uses k-means++ by default and runs several initialisations (`n_init`), keeping the best.

## Choosing k

There is no universally correct $k$. Tools:

- **Elbow method** — plot inertia against $k$ and look for a bend. Inertia always decreases with $k$, and the elbow is often ambiguous.
- **Silhouette score** — for each point, $s = \frac{b - a}{\max(a, b)}$, where $a$ is the mean distance to its own cluster and $b$ the mean distance to the nearest other cluster. Ranges from −1 to 1; higher is better.
- **Gap statistic** — compares inertia with that expected under a null reference distribution.
- **Domain constraints** — often decisive: "we can open five centres", "the marketing team can handle four segments".

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=1500, centers=4, cluster_std=1.0, random_state=7)
for k in range(2, 8):
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
    print(f"k={k}: inertia={km.inertia_:9.1f}  silhouette={silhouette_score(X, km.labels_):.3f}")
```

## The hidden assumptions

k-means implicitly assumes clusters are:

1. **Spherical (isotropic)** — it uses Euclidean distance to a centre.
2. **Similar in size and density** — boundaries lie halfway between centroids.
3. **Convex and well separated.**

:::warning
k-means will happily return $k$ clusters for **any** data, even uniform noise. It fails on elongated clusters, nested rings, clusters of very different sizes or densities, and in the presence of outliers (which drag centroids). Always visualise (e.g. with PCA or UMAP) and sanity-check clusters with domain knowledge.
:::

Also remember to **scale features** — k-means is distance-based — and to handle categorical variables appropriately (k-modes / k-prototypes).

## Variants

- **Mini-batch k-means** — updates centroids from small random batches; scales to millions of points.
- **k-medoids (PAM)** — centres must be actual data points; works with any distance and is more robust to outliers.
- **Spherical k-means** — cosine similarity for text embeddings.
- **Soft k-means / Gaussian mixtures** — probabilistic assignments (a later lecture).

## Applications

Customer segmentation, image colour quantisation (cluster pixel colours into a palette), document grouping, initialising other algorithms, **vector quantisation** (compressing embeddings — the idea behind product quantisation in vector search), and facility location as in our opening example.

:::exercise
1. Prove that the mean minimises $\sum_i\|\mathbf{x}_i - \boldsymbol{\mu}\|^2$ over $\boldsymbol{\mu}$.
2. Run k-means on `make_moons` data. Why does it fail? Which algorithm from the next lectures would succeed?
3. Compress an image by clustering its pixel colours into 8 and 16 colours with k-means.
:::

:::takeaway
- k-means minimises within-cluster squared distances by alternating assignment and mean updates.
- It converges to a local minimum; k-means++ and multiple restarts improve results.
- Choose $k$ with silhouette/elbow plus domain constraints.
- It assumes spherical, similar-sized clusters and is sensitive to scale and outliers.
:::

=== POST ===
slug: hierarchical-clustering
title: Hierarchical Clustering and Dendrograms
category: machine-learning
level: Beginner
tags: clustering, hierarchical clustering, dendrogram, linkage, ward
summary: Hierarchical clustering builds a whole tree of nested clusters instead of a single partition. We compare linkage criteria, read dendrograms, and learn when a hierarchy is more useful than k flat clusters.
---
Biologists organise living things into a hierarchy: species, genus, family, order. Libraries organise books into nested categories. Sometimes the natural structure of data is not a flat set of $k$ groups but a **hierarchy** of groups within groups. **Hierarchical clustering** discovers such structure and presents it as a tree called a **dendrogram**, letting you choose the level of granularity after seeing the data.

## Two strategies

- **Agglomerative (bottom-up)**: start with every point as its own cluster; repeatedly merge the two closest clusters until one remains. This is by far the most common approach.
- **Divisive (top-down)**: start with one cluster and recursively split it (e.g. with k-means for $k = 2$). Less common and more expensive.

## The agglomerative algorithm

1. Compute all pairwise distances between points.
2. Treat each point as a cluster.
3. Repeat $n - 1$ times: find the two closest clusters, merge them, and update the distances from the merged cluster to all others.

The result is a binary tree recording every merge and the distance at which it occurred.

## Linkage: how far apart are two clusters?

The crucial design choice is how to measure the distance between **clusters** $A$ and $B$:

| Linkage | Distance between clusters | Behaviour |
|---|---|---|
| Single | $\min_{a \in A, b \in B}d(a, b)$ | Finds elongated, chain-like clusters; prone to "chaining" through noise |
| Complete | $\max_{a \in A, b \in B}d(a, b)$ | Compact clusters of similar diameter; sensitive to outliers |
| Average (UPGMA) | mean of all pairwise distances | A compromise; widely used in biology |
| Ward | increase in total within-cluster variance caused by merging | Compact, similar-sized clusters; similar spirit to k-means |

**Ward's method** merges the pair whose union increases the within-cluster sum of squares least:

$$
\Delta(A, B) = \frac{|A||B|}{|A| + |B|}\|\boldsymbol{\mu}_A - \boldsymbol{\mu}_B\|^2
$$

It is a good default for numeric data with Euclidean distance.

## Reading a dendrogram

The x-axis lists data points (leaves); the y-axis shows merge distance. Each horizontal join is a merge; its height is the distance at which the merge happened.

- **Cutting** the dendrogram horizontally at height $h$ yields a flat clustering: the number of vertical lines crossed equals the number of clusters.
- **Long vertical gaps** indicate natural, well-separated clusters — a good place to cut.
- The **cophenetic correlation** measures how faithfully the dendrogram preserves the original pairwise distances.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, cophenet
from scipy.spatial.distance import pdist
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=60, centers=[[0, 0], [5, 5], [0, 6]], cluster_std=0.9, random_state=3)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, method in zip(axes, ["single", "average", "ward"]):
    Z = linkage(X, method=method)
    c, _ = cophenet(Z, pdist(X))
    dendrogram(Z, ax=ax, no_labels=True, color_threshold=None)
    ax.set_title(f"{method} linkage (cophenetic r = {c:.2f})")
plt.tight_layout(); plt.show()

Z = linkage(X, method="ward")
labels = fcluster(Z, t=3, criterion="maxclust")      # cut to obtain 3 clusters
print(np.bincount(labels))
```

## Complexity

The naive algorithm needs $O(n^2)$ memory for the distance matrix and $O(n^3)$ time; optimised implementations reach $O(n^2)$ time for several linkages. In practice hierarchical clustering is comfortable for up to tens of thousands of points. For larger data, cluster a sample, use connectivity constraints, or use approaches like BIRCH that build a compact summary first.

## Strengths and weaknesses

**Strengths**
- No need to fix $k$ in advance — explore granularity visually.
- Works with **any distance or similarity** (e.g. edit distance between strings, correlation between gene expression profiles).
- The dendrogram is an interpretable summary of the data's structure.
- Deterministic (no random initialisation).

**Weaknesses**
- Merges are **greedy and irreversible** — an early mistake propagates.
- Quadratic memory limits scale.
- Results depend heavily on linkage and distance choices.

:::note
In bioinformatics, hierarchical clustering of genes and samples, displayed as a heatmap with dendrograms on both axes, is one of the most common figures in the literature. In NLP and social science it is used to build taxonomies of topics. In a humanitarian setting, it can organise free-text needs assessments into a hierarchy of themes and sub-themes that analysts can inspect at any level of detail.
:::

## Connectivity-constrained clustering

Sometimes clusters must be spatially contiguous — for example, grouping neighbouring districts into regions. Scikit-learn's `AgglomerativeClustering` accepts a **connectivity** graph so that only neighbouring clusters can merge.

:::exercise
1. Construct a small dataset where single linkage "chains" two obvious clusters together through a bridge of points, while Ward separates them.
2. Cluster the Iris dataset with Ward linkage, cut at 3 clusters, and compare with the true species using the adjusted Rand index.
3. Compute a hierarchical clustering of 30 short sentences using cosine distance between TF-IDF vectors and inspect the dendrogram.
:::

:::takeaway
- Agglomerative clustering merges the closest clusters repeatedly, producing a dendrogram.
- Linkage (single, complete, average, Ward) defines cluster distance and shapes the results.
- Cut the dendrogram where there is a large gap to obtain flat clusters.
- It works with any distance but needs $O(n^2)$ memory.
:::

=== POST ===
slug: dbscan-density-clustering
title: "DBSCAN and Density-Based Clustering"
category: machine-learning
level: Intermediate
tags: clustering, dbscan, hdbscan, density, outliers
summary: DBSCAN defines clusters as dense regions separated by sparse ones. It finds arbitrarily shaped clusters, labels outliers as noise and needs no k. We study core points, parameter selection and HDBSCAN.
---
Imagine plotting GPS locations of mobile-phone users in a city. People concentrate in markets, campuses and neighbourhoods of irregular shape, with scattered points everywhere in between. k-means would force every point into one of $k$ round clusters. What we really want is to find **dense regions of any shape** and to label isolated points as **noise**. That is exactly what **DBSCAN** (Density-Based Spatial Clustering of Applications with Noise, Ester et al., 1996) does.

## Core idea

A cluster is a maximal set of **density-connected** points. Density is measured with two parameters:

- $\varepsilon$ (`eps`): the radius of a neighbourhood;
- `min_samples`: the minimum number of points (including the point itself) required in that neighbourhood.

Every point becomes one of three types:

1. **Core point** — has at least `min_samples` points within distance $\varepsilon$.
2. **Border point** — not a core point, but within $\varepsilon$ of a core point.
3. **Noise point** — neither; labelled as an outlier (label −1).

## The algorithm

1. For each unvisited point, find its $\varepsilon$-neighbourhood.
2. If it is a core point, start a new cluster and **expand** it: add all density-reachable points — neighbours of core points, their core neighbours' neighbours, and so on.
3. Border points join the cluster of a neighbouring core point; points reachable from no core point are noise.

Formally, $q$ is **directly density-reachable** from core point $p$ if $q$ lies within $\varepsilon$ of $p$. **Density-reachability** is the transitive closure through chains of core points, and two points are **density-connected** if both are reachable from a common core point.

## Why DBSCAN is useful

- **Arbitrary shapes** — rings, crescents, winding streets.
- **No need to choose $k$** — the number of clusters emerges from the data.
- **Explicit noise detection** — outliers are not forced into clusters.
- Deterministic for core points (border points may depend on processing order).

```python
import numpy as np
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score

X, y = make_moons(n_samples=600, noise=0.07, random_state=0)
X = np.vstack([X, np.random.default_rng(0).uniform(-1.5, 2.5, (30, 2))])   # add scattered noise
y = np.r_[y, [-1] * 30]
Xs = StandardScaler().fit_transform(X)

km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(Xs)
db = DBSCAN(eps=0.25, min_samples=8).fit(Xs)
print("k-means ARI:", round(adjusted_rand_score(y, km.labels_), 3))
print("DBSCAN  ARI:", round(adjusted_rand_score(y, db.labels_), 3),
      "| clusters:", len(set(db.labels_) - {-1}), "| noise points:", int((db.labels_ == -1).sum()))
```

DBSCAN recovers both crescents and flags most of the scattered points as noise; k-means slices the moons in half.

## Choosing the parameters

**`min_samples`**: a common heuristic is $2d$ for $d$-dimensional data (at least 3–5 even in 2-D). Larger values make clustering more conservative and more robust to noise.

**`eps`**: use the **k-distance plot**. For each point, compute the distance to its $k$-th nearest neighbour ($k$ = `min_samples` − 1), sort these distances, and plot them. Choose $\varepsilon$ at the "knee" where the curve bends sharply upward — points beyond the knee are in sparse regions.

```python
from sklearn.neighbors import NearestNeighbors
k = 7
dist, _ = NearestNeighbors(n_neighbors=k + 1).fit(Xs).kneighbors(Xs)
kdist = np.sort(dist[:, -1])
print("suggested eps range (80th–95th percentile):", np.percentile(kdist, [80, 95]).round(3))
```

:::warning
DBSCAN uses a **single global density threshold**. If your data contains clusters of very different densities — a dense city centre and a sparse suburb — no single $\varepsilon$ works: a small value fragments the sparse cluster into noise, a large value merges dense clusters together. Also, like all distance-based methods, it needs scaled features and degrades in high dimensions.
:::

## HDBSCAN: the modern upgrade

**HDBSCAN** (Campello, Moulavi & Sander, 2013) removes the need to pick $\varepsilon$ and handles varying densities:

1. It builds a hierarchy of DBSCAN clusterings over **all** values of $\varepsilon$ (using a "mutual reachability" distance that smooths density estimates).
2. It condenses the hierarchy, keeping clusters that persist over a wide range of densities (**stability**).
3. It returns a flat clustering, noise labels, and a membership probability for each point.

Its main parameter, `min_cluster_size`, is intuitive. HDBSCAN is available in scikit-learn (`sklearn.cluster.HDBSCAN`) and is widely used to cluster text embeddings — for example, in topic-modelling pipelines that combine sentence embeddings, UMAP and HDBSCAN.

```python
from sklearn.cluster import HDBSCAN
hdb = HDBSCAN(min_cluster_size=20).fit(Xs)
print("HDBSCAN ARI:", round(adjusted_rand_score(y, hdb.labels_), 3))
```

## Complexity

With a spatial index (KD-tree or ball tree), DBSCAN runs in roughly $O(n\log n)$ for low-dimensional data; the worst case is $O(n^2)$.

## Comparing clustering algorithms

| | k-means | Hierarchical | DBSCAN / HDBSCAN | Gaussian mixture |
|---|---|---|---|---|
| Needs $k$ | Yes | No (cut later) | No | Yes |
| Cluster shape | Spherical | Depends on linkage | Arbitrary | Ellipsoidal |
| Handles noise | No | No | Yes | Partially |
| Scales to large $n$ | Excellent | Poor | Good | Good |
| Soft assignments | No | No | HDBSCAN: yes | Yes |

:::exercise
1. Use the k-distance plot to choose $\varepsilon$ for the moons data and verify that the chosen value works.
2. Create two blobs with very different densities and show DBSCAN failing for every $\varepsilon$; then show HDBSCAN succeeding.
3. Cluster geographic coordinates (latitude/longitude) of points of interest using DBSCAN with the haversine metric.
:::

:::takeaway
- DBSCAN grows clusters from core points with at least `min_samples` neighbours within $\varepsilon$.
- It finds arbitrary shapes, determines the number of clusters and labels noise.
- Choose $\varepsilon$ with a k-distance plot; it struggles with varying densities.
- HDBSCAN removes $\varepsilon$, handles varying densities and is a strong modern default.
:::

=== POST ===
slug: gaussian-mixture-models-em
title: Gaussian Mixture Models and the EM Algorithm
category: machine-learning
level: Advanced
tags: gmm, em algorithm, clustering, latent variables, density estimation
summary: Gaussian mixtures model data as a blend of Gaussian components with soft cluster memberships. We derive the Expectation–Maximisation algorithm, prove it increases likelihood, and choose the number of components with BIC.
---
k-means gives every point a hard label and assumes round clusters. But real data often has overlapping, elliptical groups, and we may want to say "this point is 70% likely from cluster A and 30% from cluster B". **Gaussian Mixture Models (GMMs)** provide exactly this, and fitting them introduces one of the most important algorithms in statistics and ML: **Expectation–Maximisation (EM)**.

## The model

A GMM assumes each point is generated by first choosing a component $z \in \{1, \dots, K\}$ with probability $\pi_k$ (the **mixing weights**), then drawing $\mathbf{x}$ from that component's Gaussian:

$$
p(\mathbf{x}) = \sum_{k=1}^{K}\pi_k\,\mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k), \qquad \sum_k\pi_k = 1
$$

The component label $z$ is a **latent** (hidden) variable. A GMM is both a **clustering** method and a flexible **density estimator** — with enough components, it can approximate any smooth density.

## Why is fitting hard?

The log-likelihood is

$$
\ell(\boldsymbol{\theta}) = \sum_{i=1}^{n}\ln\sum_{k=1}^{K}\pi_k\,\mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)
$$

The **log of a sum** does not decompose, so there is no closed-form maximiser. But if we *knew* each point's component, the problem would be trivial: fit each Gaussian to its own points. EM exploits this.

## The EM algorithm

**E-step (Expectation).** Given current parameters, compute the **responsibility** of component $k$ for point $i$ — the posterior probability that $i$ came from $k$:

$$
\gamma_{ik} = \frac{\pi_k\,\mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}{\sum_{j}\pi_j\,\mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)}
$$

**M-step (Maximisation).** Re-estimate parameters as responsibility-weighted statistics, with $N_k = \sum_i\gamma_{ik}$:

$$
\pi_k = \frac{N_k}{n}, \qquad \boldsymbol{\mu}_k = \frac{1}{N_k}\sum_i\gamma_{ik}\mathbf{x}_i, \qquad \boldsymbol{\Sigma}_k = \frac{1}{N_k}\sum_i\gamma_{ik}(\mathbf{x}_i - \boldsymbol{\mu}_k)(\mathbf{x}_i - \boldsymbol{\mu}_k)^\top
$$

Repeat until the log-likelihood stops improving.

## Why EM works

For any distribution $q(z)$ over the latent variables, Jensen's inequality gives a lower bound on the log-likelihood — the **Evidence Lower BOund (ELBO)**:

$$
\ln p(\mathbf{x} \mid \boldsymbol{\theta}) \ge \sum_z q(z)\ln\frac{p(\mathbf{x}, z \mid \boldsymbol{\theta})}{q(z)} = \mathcal{L}(q, \boldsymbol{\theta})
$$

The gap equals $D_{\text{KL}}(q(z)\,\|\,p(z \mid \mathbf{x}, \boldsymbol{\theta}))$.

- The **E-step** sets $q$ to the exact posterior, closing the gap: the bound touches the likelihood.
- The **M-step** maximises the bound over $\boldsymbol{\theta}$, which can only raise the likelihood.

Hence **EM never decreases the likelihood**. It converges to a local maximum (or saddle point), so initialisation matters — typically with k-means and several restarts. This ELBO derivation is also the foundation of **variational autoencoders**, where the E-step is replaced by a learned neural network.

:::note
k-means is a limiting case of EM for GMMs: use equal, spherical covariances $\sigma^2\mathbf{I}$ and let $\sigma \to 0$. Responsibilities become hard 0/1 assignments to the nearest centre, and the M-step becomes the mean update. So k-means is "hard EM".
:::

## Implementation from scratch

```python
import numpy as np
from scipy.stats import multivariate_normal

def gmm_em(X, K, iters=200, seed=0, tol=1e-6):
    n, d = X.shape
    rng = np.random.default_rng(seed)
    mu = X[rng.choice(n, K, replace=False)]
    Sigma = np.array([np.cov(X.T) + 1e-6 * np.eye(d) for _ in range(K)])
    pi = np.full(K, 1 / K)
    prev = -np.inf
    for _ in range(iters):
        # E-step (in log space for stability)
        logp = np.column_stack([np.log(pi[k]) + multivariate_normal.logpdf(X, mu[k], Sigma[k])
                                for k in range(K)])
        ll = np.logaddexp.reduce(logp, axis=1)
        gamma = np.exp(logp - ll[:, None])
        # M-step
        Nk = gamma.sum(0)
        pi = Nk / n
        mu = (gamma.T @ X) / Nk[:, None]
        for k in range(K):
            D = X - mu[k]
            Sigma[k] = (gamma[:, k, None] * D).T @ D / Nk[k] + 1e-6 * np.eye(d)
        if ll.sum() - prev < tol:
            break
        prev = ll.sum()
    return pi, mu, Sigma, gamma, ll.sum()
```

The small ridge `1e-6 * I` prevents a **degenerate solution**: if a component collapses onto a single point, its variance shrinks to zero and the likelihood goes to infinity. This singularity is a real hazard of maximum-likelihood GMMs.

## Covariance structures

| `covariance_type` | Parameters per component | Cluster shape |
|---|---|---|
| `spherical` | 1 variance | Circles |
| `diag` | $d$ variances | Axis-aligned ellipses |
| `tied` | one shared full matrix | Same ellipse for all |
| `full` | $d(d+1)/2$ | Any orientation and shape |

## Choosing the number of components

Likelihood always increases with $K$, so use penalised criteria:

$$
\text{BIC} = -2\ell(\hat{\boldsymbol{\theta}}) + p\ln n, \qquad \text{AIC} = -2\ell(\hat{\boldsymbol{\theta}}) + 2p
$$

where $p$ is the number of free parameters. Choose the model with the **lowest BIC**. A Bayesian alternative, the **variational Dirichlet-process GMM** (`BayesianGaussianMixture`), switches off unnecessary components automatically.

```python
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=800, centers=3, cluster_std=[1.0, 2.0, 0.6], random_state=2)
X = X @ np.array([[0.6, -0.6], [-0.4, 0.8]])            # stretch into ellipses
for K in range(1, 7):
    g = GaussianMixture(K, covariance_type="full", n_init=5, random_state=0).fit(X)
    print(f"K={K}: BIC={g.bic(X):9.1f}")
```

## Uses of GMMs

- **Soft clustering** with probabilistic memberships.
- **Density estimation** and **anomaly detection** (flag points with low $p(\mathbf{x})$).
- **Speaker identification** — GMMs over acoustic features were the standard before deep learning.
- **Background subtraction** in video.
- **Generating synthetic data** by sampling from the fitted mixture.

:::exercise
1. Derive the M-step update for $\boldsymbol{\mu}_k$ by maximising the expected complete-data log-likelihood.
2. Fit GMMs with `spherical`, `diag` and `full` covariances to elongated clusters and compare BIC.
3. Use a GMM as an anomaly detector: fit on normal data, and flag the lowest 1% log-density points in new data.
:::

:::takeaway
- A GMM is a weighted sum of Gaussians with a latent component label.
- EM alternates computing responsibilities (E) and weighted parameter estimates (M); likelihood never decreases.
- The ELBO view of EM underlies variational inference and VAEs; k-means is hard EM.
- Choose $K$ with BIC; regularise covariances to avoid degenerate solutions.
:::

=== POST ===
slug: principal-component-analysis
title: Principal Component Analysis (PCA): Theory and Practice
category: machine-learning
level: Intermediate
tags: pca, dimensionality reduction, eigenvectors, svd, variance
summary: PCA finds the orthogonal directions of maximum variance. We derive it two ways — maximum variance and minimum reconstruction error — compute it via SVD, choose the number of components, and discuss its limits.
---
Datasets often contain dozens or thousands of correlated features. Many of those dimensions are redundant: height in centimetres and height in inches; the pixel values of neighbouring pixels. **Principal Component Analysis** (Pearson 1901, Hotelling 1933) finds a new coordinate system aligned with the directions along which the data varies most, letting us compress, visualise and denoise data. It is the most widely used dimensionality-reduction technique in science.

## Two equivalent goals

Let the data matrix $\mathbf{X} \in \mathbb{R}^{n \times d}$ be **centred** (each column has mean zero). We seek a unit vector $\mathbf{u}$ onto which to project the data.

### 1. Maximise variance
The variance of the projections $\mathbf{X}\mathbf{u}$ is $\mathbf{u}^\top\mathbf{S}\mathbf{u}$, where $\mathbf{S} = \frac{1}{n-1}\mathbf{X}^\top\mathbf{X}$ is the sample covariance matrix. Maximise it subject to $\|\mathbf{u}\| = 1$ using a Lagrange multiplier:

$$
\frac{\partial}{\partial\mathbf{u}}\left[\mathbf{u}^\top\mathbf{S}\mathbf{u} - \lambda(\mathbf{u}^\top\mathbf{u} - 1)\right] = 0 \;\Longrightarrow\; \mathbf{S}\mathbf{u} = \lambda\mathbf{u}
$$

So $\mathbf{u}$ must be an **eigenvector** of the covariance matrix, and the variance along it equals the eigenvalue $\lambda$. The **first principal component** is the eigenvector with the largest eigenvalue; the second is the next eigenvector (orthogonal to the first), and so on.

### 2. Minimise reconstruction error
Project onto a $k$-dimensional subspace spanned by orthonormal $\mathbf{U}_k$ and reconstruct: $\hat{\mathbf{x}} = \mathbf{U}_k\mathbf{U}_k^\top\mathbf{x}$. The subspace minimising the average squared reconstruction error $\frac{1}{n}\sum_i\|\mathbf{x}_i - \hat{\mathbf{x}}_i\|^2$ is spanned by the top $k$ eigenvectors. Since total variance = retained variance + reconstruction error, maximising one minimises the other. (This is the Eckart–Young theorem in disguise.)

## Computing PCA with the SVD

In practice we never form $\mathbf{X}^\top\mathbf{X}$ explicitly. Take the SVD of the centred data:

$$
\mathbf{X} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top
$$

- The columns of $\mathbf{V}$ are the principal directions (**loadings**).
- The eigenvalues are $\lambda_j = \sigma_j^2/(n - 1)$.
- The projected coordinates (**scores**) are $\mathbf{X}\mathbf{V}_k = \mathbf{U}_k\boldsymbol{\Sigma}_k$.

```python
import numpy as np
from sklearn.datasets import load_digits

X, y = load_digits(return_X_y=True)              # 1797 images, 64 pixels
Xc = X - X.mean(0)
U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
explained = s**2 / (s**2).sum()
cum = np.cumsum(explained)
print("variance explained by first 2 PCs:", explained[:2].round(3))
print("components needed for 90% variance:", int(np.searchsorted(cum, 0.90) + 1))

Z = Xc @ Vt[:2].T                                 # 2-D scores for plotting
X_rec = (Xc @ Vt[:20].T) @ Vt[:20] + X.mean(0)    # reconstruct from 20 components
print("relative reconstruction error (20 PCs):",
      round(np.linalg.norm(X - X_rec) / np.linalg.norm(X), 3))
```

The 64-dimensional digits need only around 20–30 components to retain 90% of the variance.

## Choosing the number of components

- **Cumulative explained variance** — keep enough components for, say, 90–95%.
- **Scree plot** — plot eigenvalues and look for an elbow.
- **Downstream performance** — choose $k$ by cross-validating the model that uses the PCA features.
- **Kaiser rule** (for standardised data) — keep components with eigenvalue > 1; crude but common in social science.

## Standardise or not?

PCA is sensitive to scale: a feature measured in large units dominates the variance. If features have different units, **standardise** them (PCA on the correlation matrix). If all features share meaningful units (pixel intensities, gene expression on the same scale), centring alone may be appropriate.

## Interpreting components

Each component is a weighted combination of original features. Inspecting the loadings can reveal latent factors — in a survey, one component might load on all income-related questions ("economic status"). But be careful: components are defined by variance, not meaning, and their signs are arbitrary.

:::warning
**PCA is unsupervised.** The directions of greatest variance are not necessarily the directions that best separate classes. A low-variance direction can carry all the discriminative information. If your goal is classification, validate that PCA helps — or consider supervised methods such as Linear Discriminant Analysis.
:::

## Uses

- **Visualisation** — project to 2-D or 3-D.
- **Compression and speed** — fewer features for downstream models.
- **Noise reduction** — discard low-variance components that mostly contain noise.
- **Decorrelation / whitening** — transform features to be uncorrelated with unit variance.
- **Eigenfaces** — the classic face-recognition representation.
- **Population genetics** — top PCs of genotype data mirror geography.
- **Analysing neural network representations** and embeddings.

## Variants

- **Incremental PCA** — for data that does not fit in memory.
- **Randomised PCA** — fast approximate SVD for large matrices.
- **Kernel PCA** — non-linear PCA via kernels.
- **Sparse PCA** — loadings with many zeros for interpretability.
- **Probabilistic PCA / Factor analysis** — generative latent-variable versions.

:::exercise
1. Show that the principal components are uncorrelated (their covariance matrix is diagonal).
2. Apply PCA to the digits and train logistic regression with $k = 5, 10, 20, 40$ components. Plot accuracy against $k$.
3. Construct a 2-D dataset where the first principal component is useless for classification but the second is perfect.
:::

:::takeaway
- PCA finds orthogonal directions of maximum variance: eigenvectors of the covariance matrix.
- Equivalently, it minimises squared reconstruction error; compute it with the SVD.
- Choose $k$ by explained variance or downstream validation; standardise features with different units.
- PCA is unsupervised and linear — variance is not the same as usefulness.
:::

=== POST ===
slug: tsne-and-umap
title: t-SNE and UMAP: Visualising High-Dimensional Data
category: machine-learning
level: Intermediate
tags: visualization, t-sne, umap, manifold learning, embeddings
summary: Non-linear embeddings reveal cluster structure that PCA hides. We explain how t-SNE and UMAP work, what their hyperparameters do, and — critically — how not to misread their plots.
---
You have trained a model that embeds 50,000 documents into 768-dimensional vectors. Are there natural groups? Are two classes confused? PCA to two dimensions often shows a shapeless cloud, because the structure lies on a curved, non-linear manifold. **t-SNE** and **UMAP** are non-linear methods designed to reveal such structure in 2-D plots. They are indispensable tools — and among the most frequently misinterpreted.

## t-SNE: matching neighbour probabilities

**t-distributed Stochastic Neighbour Embedding** (van der Maaten & Hinton, 2008) works in three steps.

**1. High-dimensional similarities.** For each point $i$, define a Gaussian-based probability that $i$ would pick $j$ as its neighbour:

$$
p_{j \mid i} = \frac{\exp(-\|\mathbf{x}_i - \mathbf{x}_j\|^2/2\sigma_i^2)}{\sum_{k \ne i}\exp(-\|\mathbf{x}_i - \mathbf{x}_k\|^2/2\sigma_i^2)}
$$

The bandwidth $\sigma_i$ is chosen per point so that the distribution's **perplexity** matches a user-set value — roughly, the effective number of neighbours. Symmetrise: $p_{ij} = (p_{j \mid i} + p_{i \mid j})/2n$.

**2. Low-dimensional similarities** use a heavy-tailed **Student-t** distribution with one degree of freedom:

$$
q_{ij} = \frac{(1 + \|\mathbf{y}_i - \mathbf{y}_j\|^2)^{-1}}{\sum_{k \ne l}(1 + \|\mathbf{y}_k - \mathbf{y}_l\|^2)^{-1}}
$$

**3. Optimise** the 2-D positions $\mathbf{y}_i$ by gradient descent to minimise

$$
D_{\text{KL}}(P\,\|\,Q) = \sum_{i \ne j}p_{ij}\ln\frac{p_{ij}}{q_{ij}}
$$

Why the heavy tail? In high dimensions, many points can be moderately far from a given point, but in 2-D there is not enough room — the **crowding problem**. The Student-t lets moderately distant points be placed much farther apart in 2-D, producing well-separated clusters.

## UMAP: fuzzy topology

**Uniform Manifold Approximation and Projection** (McInnes, Healy & Melville, 2018) builds a weighted k-nearest-neighbour graph in high dimensions (grounded in fuzzy simplicial sets from topology), then optimises a low-dimensional layout whose graph matches it using a **cross-entropy** objective with attractive forces along edges and repulsive forces via negative sampling.

Compared with t-SNE, UMAP typically:

- runs **much faster** and scales to millions of points;
- preserves **more global structure** (relative positions of clusters are somewhat more meaningful, though still not reliable);
- can **transform new points** into an existing embedding;
- works as a general-purpose reduction for clustering (e.g. UMAP → HDBSCAN on text embeddings).

## Hyperparameters

| Method | Parameter | Effect |
|---|---|---|
| t-SNE | `perplexity` (5–50) | Neighbourhood size; low → many tiny clusters, high → broader structure |
| t-SNE | learning rate, iterations | Too few iterations → unconverged "ball" |
| UMAP | `n_neighbors` (5–200) | Local vs global balance |
| UMAP | `min_dist` (0–0.99) | How tightly points pack within clusters |

```python
import numpy as np
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
# pip install umap-learn
import umap

X, y = load_digits(return_X_y=True)
X50 = PCA(n_components=30, random_state=0).fit_transform(X)   # common pre-step for speed/denoising

emb = {
    "PCA": PCA(n_components=2).fit_transform(X),
    "t-SNE (perplexity 30)": TSNE(perplexity=30, init="pca", random_state=0).fit_transform(X50),
    "UMAP (n_neighbors 15)": umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=0).fit_transform(X50),
}
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, (name, Z) in zip(axes, emb.items()):
    ax.scatter(Z[:, 0], Z[:, 1], c=y, cmap="tab10", s=5); ax.set_title(name)
plt.tight_layout(); plt.show()
```

PCA shows overlapping digit groups; t-SNE and UMAP show ten crisp clusters.

## How NOT to read these plots

:::warning
1. **Cluster sizes mean nothing.** t-SNE expands dense clusters and contracts sparse ones. A big blob is not a bigger or more diverse group.
2. **Distances between clusters mean little.** Two clusters far apart in t-SNE are not necessarily more different than two close ones (UMAP is somewhat better but not reliable).
3. **Random noise can look like clusters** at low perplexity. Always try several hyperparameters and random seeds.
4. **Axes have no meaning** — they are arbitrary coordinates.
5. **Do not cluster on t-SNE output** and then report those clusters as discoveries; cluster in the original (or PCA/UMAP-reduced) space and use the plot for visual inspection only.
:::

Wattenberg, Viégas and Johnson's interactive article "How to Use t-SNE Effectively" demonstrates these pitfalls vividly; every student should explore it.

## Good practice

- Standardise features; for very high-dimensional data, reduce to 30–50 dimensions with PCA first.
- Try multiple perplexities / neighbour counts and seeds; trust only structure that persists.
- Colour points by known labels, metadata and model predictions to generate hypotheses — then **test** those hypotheses with proper methods.
- Report hyperparameters with every plot.

## Uses in ML practice

- Inspecting learned embeddings (words, sentences, images).
- Diagnosing classifier confusion — misclassified points often lie between clusters.
- Detecting label noise and data-quality problems (points of one class inside another's cluster).
- Exploring single-cell biology data, where UMAP plots are standard.

:::exercise
1. Run t-SNE on the digits with perplexity 2, 30 and 200. Describe the differences.
2. Generate 1,000 points of pure Gaussian noise in 50 dimensions and run t-SNE with perplexity 5. Do you "see" clusters?
3. Embed sentence vectors of news headlines with UMAP and colour by category. Which categories overlap, and why might that be?
:::

:::takeaway
- t-SNE matches Gaussian neighbour probabilities with heavy-tailed Student-t similarities in 2-D.
- UMAP builds a fuzzy k-NN graph and optimises a layout; it is faster and preserves more global structure.
- Perplexity / `n_neighbors` control the local–global balance.
- Cluster sizes and inter-cluster distances are not reliable; use these plots for exploration, not conclusions.
:::

=== POST ===
slug: linear-discriminant-analysis
title: Linear Discriminant Analysis: Supervised Dimensionality Reduction
category: machine-learning
level: Intermediate
tags: lda, fisher discriminant, dimensionality reduction, classification, generative
summary: Unlike PCA, LDA uses labels to find projections that separate classes. We derive Fisher's criterion, the generative Gaussian view, and compare LDA with PCA, QDA and logistic regression.
---
PCA finds the directions of maximum variance, ignoring labels. But if our goal is classification, we want directions that **separate classes**. In 1936 Ronald Fisher, studying the famous Iris flowers, derived the linear projection that best discriminates between groups. **Linear Discriminant Analysis (LDA)** is both a classifier and a supervised dimensionality-reduction method — fast, closed-form and surprisingly competitive.

(Not to be confused with *Latent Dirichlet Allocation*, the topic model, which shares the acronym.)

## Fisher's criterion

For two classes, project data onto a direction $\mathbf{w}$. We want:

- the projected class means to be **far apart**;
- each class's projected points to be **tightly clustered**.

Define the **between-class scatter** $\mathbf{S}_B = (\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2)(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2)^\top$ and the **within-class scatter** $\mathbf{S}_W = \sum_{c}\sum_{i \in c}(\mathbf{x}_i - \boldsymbol{\mu}_c)(\mathbf{x}_i - \boldsymbol{\mu}_c)^\top$. Fisher's criterion is the ratio

$$
J(\mathbf{w}) = \frac{\mathbf{w}^\top\mathbf{S}_B\mathbf{w}}{\mathbf{w}^\top\mathbf{S}_W\mathbf{w}}
$$

Maximising it (a generalised Rayleigh quotient) gives the closed form

$$
\mathbf{w}^* \propto \mathbf{S}_W^{-1}(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2)
$$

Notice the role of $\mathbf{S}_W^{-1}$: the best direction is not simply the line joining the means. It is tilted to account for the shape of the classes — directions in which the classes are spread out are down-weighted.

## Multiclass LDA

For $K$ classes, define $\mathbf{S}_B = \sum_c n_c(\boldsymbol{\mu}_c - \boldsymbol{\mu})(\boldsymbol{\mu}_c - \boldsymbol{\mu})^\top$. The optimal projection directions are the top eigenvectors of $\mathbf{S}_W^{-1}\mathbf{S}_B$. Because $\mathbf{S}_B$ has rank at most $K - 1$, LDA yields **at most $K - 1$ discriminant components**. For 10 digit classes, LDA gives at most 9 dimensions — often an excellent compressed representation for classification.

## The generative (probabilistic) view

LDA can also be derived as a **generative classifier**. Assume each class is Gaussian with its own mean but a **shared covariance** $\boldsymbol{\Sigma}$:

$$
p(\mathbf{x} \mid y = c) = \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_c, \boldsymbol{\Sigma})
$$

Applying Bayes' theorem, the log-posterior ratio between two classes is

$$
\ln\frac{P(y = 1 \mid \mathbf{x})}{P(y = 2 \mid \mathbf{x})} = \mathbf{x}^\top\boldsymbol{\Sigma}^{-1}(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2) + \text{const}
$$

The quadratic terms cancel because the covariance is shared, so the decision boundary is **linear** — and its normal vector is exactly Fisher's direction. Two derivations, one answer.

If each class has its **own** covariance, the quadratic terms no longer cancel and we get **Quadratic Discriminant Analysis (QDA)** with curved boundaries — more flexible, but with many more parameters to estimate ($K \cdot d(d+1)/2$ covariance entries).

## LDA vs PCA

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

X, y = load_wine(return_X_y=True)
Xs = StandardScaler().fit_transform(X)
Z_pca = PCA(2).fit_transform(Xs)
Z_lda = LinearDiscriminantAnalysis(n_components=2).fit_transform(Xs, y)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, Z, t in [(axes[0], Z_pca, "PCA (unsupervised)"), (axes[1], Z_lda, "LDA (supervised)")]:
    ax.scatter(Z[:, 0], Z[:, 1], c=y, cmap="viridis", s=18); ax.set_title(t)
plt.tight_layout(); plt.show()

for name, m in [("LDA", LinearDiscriminantAnalysis()), ("QDA", QuadraticDiscriminantAnalysis()),
                ("Logistic", LogisticRegression(max_iter=2000))]:
    print(name, round(cross_val_score(m, Xs, y, cv=5).mean(), 3))
```

| | PCA | LDA |
|---|---|---|
| Uses labels | No | Yes |
| Objective | Maximise variance | Maximise class separation |
| Max components | $\min(n, d)$ | $K - 1$ |
| Assumptions | None (linear) | Gaussian classes, shared covariance (for optimality) |

## Regularisation and high dimensions

When $d$ is large relative to $n$, $\mathbf{S}_W$ is singular or poorly estimated. **Shrinkage LDA** replaces it with $(1 - \alpha)\hat{\mathbf{S}}_W + \alpha\,\nu\mathbf{I}$, with $\alpha$ chosen analytically (Ledoit–Wolf). In scikit-learn: `LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")`. Shrinkage LDA is a standard, strong method in brain–computer interfaces, where EEG data is high-dimensional and noisy with few trials.

## LDA vs logistic regression

Both produce linear boundaries. LDA is **generative** and assumes Gaussian classes with shared covariance; logistic regression is **discriminative** and assumes only the log-odds are linear.

- When LDA's assumptions hold, it is **more statistically efficient** (needs less data).
- When they fail (heavy tails, outliers, categorical features), logistic regression is more robust.

:::exercise
1. Derive $\mathbf{w}^* \propto \mathbf{S}_W^{-1}(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2)$ by setting the derivative of Fisher's criterion to zero.
2. Explain why LDA can produce at most $K - 1$ components.
3. Create a dataset with 200 features and 60 examples. Compare plain LDA with shrinkage LDA using cross-validation.
:::

:::takeaway
- LDA finds projections maximising between-class relative to within-class scatter.
- Its generative view — Gaussian classes with shared covariance — gives linear boundaries; QDA drops the shared covariance.
- It yields at most $K - 1$ discriminant directions.
- Use shrinkage in high dimensions; prefer logistic regression when Gaussian assumptions fail badly.
:::

=== POST ===
slug: feature-engineering
title: Feature Engineering: Turning Raw Data into Signal
category: machine-learning
level: Intermediate
tags: feature engineering, domain knowledge, transformations, interactions, time features
summary: Better features beat better algorithms. We survey the craft — transformations, interactions, aggregations, date and text features, domain-driven ratios — and how to engineer features without leaking the target.
---
Andrew Ng once remarked that applied machine learning is basically feature engineering. Deep learning has automated feature learning for images, audio and text, but for **tabular data** — the most common data in business, government and humanitarian work — thoughtful features remain the single biggest lever on model quality. Today we study the craft.

## What makes a good feature?

A good feature is:

- **Informative** — related to the target.
- **Available at prediction time** — no leakage from the future or the label.
- **Robust** — stable over time and across populations.
- **Simple to compute and explain** where possible.

## 1. Numeric transformations

- **Log / power transforms** for skewed variables (income, population, counts): $\log(1 + x)$ or Box–Cox/Yeo–Johnson. They stabilise variance and make relationships more linear for linear models.
- **Binning** into quantiles or domain-meaningful ranges (age groups) — can capture non-linearity for linear models and improve robustness to outliers, at the cost of information.
- **Clipping (winsorising)** extreme values.
- **Ratios and differences** often carry more meaning than raw values: debt-to-income, price per square metre, household size per room, change since last month.

:::note
Tree models are invariant to monotonic transformations of a single feature, so log transforms rarely help them. But **ratios and differences** between features help trees enormously, because a tree needs many axis-aligned splits to approximate a ratio.
:::

## 2. Interaction features

Products and combinations of features capture effects that depend on context: `rainfall × temperature` for crop yield, `is_weekend × hour` for demand. Linear models cannot discover interactions on their own; trees and neural networks can, but explicit interactions still help them learn faster with less data.

## 3. Date and time features

A timestamp is not a useful raw number. Extract: hour, day of week, month, quarter, holiday flags, days until/since an event, season, and **cyclical encodings** so that December is next to January:

$$
\text{month}_{\sin} = \sin\left(\frac{2\pi\,\text{month}}{12}\right), \qquad \text{month}_{\cos} = \cos\left(\frac{2\pi\,\text{month}}{12}\right)
$$

## 4. Aggregation (group) features

For relational data — transactions per customer, visits per patient, registrations per location — compute statistics over related records: counts, sums, means, maxima, standard deviations, recency (time since last event), frequency and trends. Aggregations over **time windows** (last 7, 30, 90 days) are extremely powerful.

:::warning
Aggregations are the most common source of **temporal leakage**. When computing "average spending of this customer", include only transactions **before** the prediction time for that row. Use point-in-time correct joins; feature stores exist largely to enforce this.
:::

## 5. Text features

Word counts, TF-IDF vectors, text length, presence of keywords, sentiment scores, and — increasingly — **embeddings** from pretrained language models, which convert free text into dense vectors that capture meaning.

## 6. Geographic features

Distances to key locations (nearest clinic, market, water point), density of points within a radius, administrative-region encodings, elevation, and features derived from satellite imagery.

## 7. Domain knowledge features

The most valuable features usually come from **talking to domain experts**. A nutritionist knows that the ratio of weight to height predicts malnutrition better than either alone (that is why weight-for-height z-scores exist). A logistics officer knows that road access in the rainy season matters more than distance. No algorithm will reliably discover what an expert already knows from a few thousand examples.

## A worked example

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
n = 1000
df = pd.DataFrame({
    "household_id": rng.integers(0, 200, n),
    "visit_time": pd.to_datetime("2025-01-01") + pd.to_timedelta(rng.integers(0, 365 * 24, n), unit="h"),
    "members": rng.integers(1, 10, n),
    "rooms": rng.integers(1, 5, n),
    "monthly_income": rng.lognormal(9, 0.8, n),
})
df = df.sort_values("visit_time")

# Numeric transforms and ratios
df["log_income"] = np.log1p(df["monthly_income"])
df["income_per_member"] = df["monthly_income"] / df["members"]
df["crowding"] = df["members"] / df["rooms"]

# Date features with cyclical encoding
df["hour"] = df["visit_time"].dt.hour
df["dow"] = df["visit_time"].dt.dayofweek
df["month_sin"] = np.sin(2 * np.pi * df["visit_time"].dt.month / 12)
df["month_cos"] = np.cos(2 * np.pi * df["visit_time"].dt.month / 12)

# Point-in-time aggregations: only PAST visits of the same household
g = df.groupby("household_id")
df["prior_visits"] = g.cumcount()
df["days_since_last_visit"] = g["visit_time"].diff().dt.total_seconds() / 86400
df["prior_mean_income"] = g["monthly_income"].transform(lambda s: s.shift().expanding().mean())

print(df.head(8).round(2).to_string())
```

Note the `shift()` in the aggregation: each row sees only earlier visits. Without it, the feature would include the current (and future) visits — leakage.

## Automated feature engineering

Tools like **Featuretools** (deep feature synthesis) generate many aggregation features automatically from relational tables. They are useful for exploration, but they generate hundreds of candidates — combine them with feature selection and careful leakage checks.

## A disciplined process

1. Start with a baseline on raw features.
2. Brainstorm features with domain experts; write down the hypothesis behind each.
3. Add features in groups; measure validation improvement for each group.
4. Check importance and **error analysis** to inspire the next features.
5. Keep all feature code in a reproducible pipeline shared by training and serving, so the model sees identical features in production.

:::exercise
1. For a dataset of taxi trips (pickup time, pickup/drop-off coordinates, fare), propose ten features and justify each.
2. Demonstrate leakage: compute a customer's mean purchase including the current row, and compare validation performance with the correct point-in-time version.
3. Add cyclical hour-of-day features to a demand-forecasting dataset and compare with a raw integer hour for a linear model.
:::

:::takeaway
- Features must be informative, available at prediction time and robust.
- Use transforms, ratios, interactions, cyclical dates, aggregations, text and geographic features.
- Domain expertise is the richest source of features.
- Compute aggregations point-in-time to avoid leakage; keep feature code identical in training and serving.
:::
