=== POST ===
slug: convex-optimization-basics
title: Convex Optimisation Basics: Why Some Problems Are Easy
category: math
level: Intermediate
tags: optimization, convexity, convex functions, global minimum
summary: In a convex problem every local minimum is global. We define convex sets and functions, learn practical tests for convexity, and see which ML models are convex and which are not.
---
Optimisation is the engine of learning. Some optimisation problems are treacherous — full of local minima and saddle points. Others are benign: any downhill path leads to the best answer. The dividing line is **convexity**. Linear regression, logistic regression, SVMs and lasso are convex; deep neural networks are not. Understanding convexity tells you when you can trust your optimiser's answer.

## Convex sets

A set $C$ is **convex** if the line segment between any two of its points lies entirely inside it:

$$
\mathbf{x}, \mathbf{y} \in C,\; \theta \in [0, 1] \;\Longrightarrow\; \theta\mathbf{x} + (1 - \theta)\mathbf{y} \in C
$$

Examples: balls, half-spaces $\{\mathbf{x} : \mathbf{a}^\top\mathbf{x} \le b\}$, polyhedra, the set of positive semi-definite matrices. The intersection of convex sets is convex — so constraint sets built from linear inequalities are convex. A crescent or a donut is not convex.

## Convex functions

A function $f$ is **convex** if its domain is convex and

$$
f(\theta\mathbf{x} + (1 - \theta)\mathbf{y}) \le \theta f(\mathbf{x}) + (1 - \theta)f(\mathbf{y})
$$

Geometrically: the chord between any two points on the graph lies **above** the graph. It is **strictly convex** if the inequality is strict for $\mathbf{x} \ne \mathbf{y}$ and $\theta \in (0,1)$.

### Practical tests

1. **First-order condition** (differentiable $f$): the tangent plane is a global under-estimator:
$$
f(\mathbf{y}) \ge f(\mathbf{x}) + \nabla f(\mathbf{x})^\top(\mathbf{y} - \mathbf{x})
$$
2. **Second-order condition** (twice differentiable): the Hessian is positive semi-definite everywhere, $\nabla^2 f(\mathbf{x}) \succeq 0$. In 1-D: $f''(x) \ge 0$.

### Building blocks

- Linear and affine functions (both convex and concave).
- Norms $\|\mathbf{x}\|_p$ for $p \ge 1$.
- $e^{x}$, $x^2$, $-\ln x$, $\ln(1 + e^x)$ (softplus), $\max(0, 1 - x)$ (hinge).
- **Log-sum-exp** $\ln\sum_i e^{x_i}$.

### Operations that preserve convexity

- Non-negative weighted sums: $\sum_i w_i f_i$ with $w_i \ge 0$.
- Composition with an affine map: $f(\mathbf{A}\mathbf{x} + \mathbf{b})$.
- Pointwise maximum: $\max_i f_i(\mathbf{x})$.

With these rules you can verify convexity of most ML objectives without computing a Hessian.

## The fundamental theorem

:::definition
For a convex function, **every local minimum is a global minimum**. If $f$ is differentiable, $\nabla f(\mathbf{x}^*) = \mathbf{0}$ is *sufficient* for global optimality. If $f$ is strictly convex, the global minimiser is **unique**.
:::

*Proof sketch.* Suppose $\mathbf{x}$ is a local minimum but some $\mathbf{y}$ has $f(\mathbf{y}) < f(\mathbf{x})$. Points on the segment near $\mathbf{x}$, $\mathbf{z} = \theta\mathbf{y} + (1-\theta)\mathbf{x}$ with small $\theta > 0$, satisfy $f(\mathbf{z}) \le \theta f(\mathbf{y}) + (1-\theta)f(\mathbf{x}) < f(\mathbf{x})$ — contradicting local minimality. $\blacksquare$

## Which ML problems are convex?

| Model / objective | Convex in parameters? | Why |
|---|---|---|
| Linear regression (MSE) | Yes | Quadratic with PSD Hessian $2\mathbf{X}^\top\mathbf{X}$ |
| Ridge / Lasso | Yes (ridge strictly) | Convex loss + convex norm penalty |
| Logistic regression | Yes | Log-loss is softplus of an affine function |
| Linear SVM | Yes | Hinge loss + squared norm |
| k-means objective | No | Discrete assignments create many local optima |
| Neural networks | No | Compositions of non-linearities; weight-space symmetries |
| Matrix factorisation | No (jointly) | Bilinear; but convex in each factor separately |

:::note
Neural networks are non-convex, yet gradient descent trains them remarkably well. Research suggests that in very high dimensions, most critical points are saddle points rather than bad local minima, and that for over-parameterised networks many global minima exist and are connected. Non-convexity is real, but it is less catastrophic than low-dimensional pictures suggest.
:::

## Strong convexity and smoothness

Two constants govern how fast gradient methods converge:

- **$\mu$-strongly convex**: $\nabla^2 f \succeq \mu\mathbf{I}$ — the function curves up at least as fast as a quadratic.
- **$L$-smooth**: $\nabla^2 f \preceq L\mathbf{I}$ — gradients do not change too abruptly.

Their ratio $\kappa = L/\mu$ is the **condition number**. Gradient descent with step $1/L$ on a strongly convex, smooth function converges **linearly**: the error shrinks by a factor of roughly $(1 - 1/\kappa)$ per step. Large $\kappa$ means slow convergence — a key motivation for feature standardisation, which improves conditioning.

```python
import numpy as np

def gd(H, x0, lr, steps=200):
    x = x0.copy()
    for _ in range(steps):
        x -= lr * (H @ x)                  # gradient of 0.5 x^T H x
    return np.linalg.norm(x)

x0 = np.array([1.0, 1.0])
for kappa in [2, 20, 200]:
    H = np.diag([1.0, float(kappa)])       # mu = 1, L = kappa
    print(f"kappa={kappa:>3}: distance to optimum after 200 steps = {gd(H, x0, 1 / kappa):.2e}")
```

## Convex solvers in practice

For convex problems we have excellent tools: `scikit-learn` solvers (liblinear, L-BFGS, SAGA), and modelling languages like **CVXPY** that let you write a problem in mathematical form and hand it to a specialised solver.

```python
import cvxpy as cp
import numpy as np

rng = np.random.default_rng(0)
X, y = rng.normal(size=(50, 10)), rng.normal(size=50)
w = cp.Variable(10)
problem = cp.Problem(cp.Minimize(cp.sum_squares(X @ w - y) + 0.5 * cp.norm1(w)))
problem.solve()
print(np.round(w.value, 3))   # lasso solution with some exact zeros
```

:::exercise
1. Prove that $f(x) = \ln(1 + e^x)$ is convex using the second-order condition.
2. Show that the logistic regression loss is convex in $\mathbf{w}$ by composing convex building blocks.
3. Is $f(x, y) = xy$ convex? Compute its Hessian and classify it.
:::

:::takeaway
- Convex sets contain all line segments between their points; convex functions lie below their chords.
- For convex functions every local minimum is global; zero gradient is sufficient for optimality.
- Linear/logistic regression, SVMs, ridge and lasso are convex; neural networks and k-means are not.
- The condition number $\kappa = L/\mu$ controls convergence speed of gradient methods.
:::

=== POST ===
slug: gradient-descent-theory
title: Gradient Descent: Theory, Step Sizes and Convergence
category: math
level: Intermediate
tags: optimization, gradient descent, sgd, learning rate, convergence
summary: The simplest optimisation algorithm trains the largest models in the world. We analyse gradient descent, the role of the learning rate, stochastic gradients, and the convergence rates you should know.
---
Gradient descent is almost embarrassingly simple: compute the slope, take a small step downhill, repeat. Yet a variant of it trains every large neural network. In this lecture we analyse it properly — why it works, how to choose the step size, what happens when gradients are noisy, and how fast it converges.

## The algorithm

To minimise a differentiable function $f(\mathbf{w})$:

$$
\mathbf{w}_{t+1} = \mathbf{w}_t - \eta\,\nabla f(\mathbf{w}_t)
$$

where $\eta > 0$ is the **learning rate** (step size).

### Why it decreases the loss

By Taylor expansion, for small $\eta$:

$$
f(\mathbf{w}_{t+1}) \approx f(\mathbf{w}_t) - \eta\,\|\nabla f(\mathbf{w}_t)\|^2
$$

The decrease is proportional to the squared gradient norm — guaranteed descent as long as the step is small enough that the linear approximation holds.

## Choosing the learning rate

If $f$ is **$L$-smooth** (its gradient is $L$-Lipschitz), the **descent lemma** gives

$$
f(\mathbf{w}_{t+1}) \le f(\mathbf{w}_t) - \eta\left(1 - \frac{L\eta}{2}\right)\|\nabla f(\mathbf{w}_t)\|^2
$$

So any $\eta < 2/L$ guarantees decrease, and $\eta = 1/L$ gives the best guaranteed decrease of $\frac{1}{2L}\|\nabla f\|^2$ per step.

For a quadratic $f(\mathbf{w}) = \frac{1}{2}\mathbf{w}^\top\mathbf{H}\mathbf{w}$, analysing each eigen-direction separately shows the update multiplies the component along eigenvector $i$ by $(1 - \eta\lambda_i)$:

- $\eta < 2/\lambda_{\max}$: all components shrink — convergence.
- $\eta > 2/\lambda_{\max}$: the steepest direction oscillates with growing amplitude — divergence.
- Small $\lambda_{\min}$: the flattest direction shrinks very slowly — the condition number bottleneck.

:::tip
**Learning rate symptoms.** Loss explodes or becomes NaN → too high. Loss decreases painfully slowly → too low. Loss decreases then plateaus noisily → consider a decay schedule. A practical recipe: try values on a log scale (1e-1, 3e-2, 1e-2, …) and pick the largest one that trains stably — or use a learning-rate range test.
:::

## Convergence rates

For gradient descent with $\eta = 1/L$:

| Function class | Rate | Iterations for accuracy $\epsilon$ |
|---|---|---|
| Convex, $L$-smooth | $f(\mathbf{w}_T) - f^* = O(1/T)$ | $O(1/\epsilon)$ |
| $\mu$-strongly convex, $L$-smooth | $O\big((1 - \mu/L)^T\big)$ — linear | $O(\kappa\log(1/\epsilon))$ |
| Non-convex, $L$-smooth | $\min_t \|\nabla f(\mathbf{w}_t)\|^2 = O(1/T)$ | finds approximate stationary points |

**Nesterov's accelerated gradient** improves the convex rate to $O(1/T^2)$ and the strongly convex rate to depend on $\sqrt{\kappa}$ instead of $\kappa$ — provably optimal among first-order methods.

## Stochastic gradient descent

In ML, the loss is an average over $n$ training examples: $f(\mathbf{w}) = \frac{1}{n}\sum_i \ell_i(\mathbf{w})$. Computing the full gradient costs $O(n)$ per step — prohibitive for millions of examples. **Stochastic gradient descent (SGD)** uses a random **mini-batch** $\mathcal{B}$:

$$
\mathbf{g}_t = \frac{1}{|\mathcal{B}|}\sum_{i \in \mathcal{B}} \nabla\ell_i(\mathbf{w}_t), \qquad \mathbb{E}[\mathbf{g}_t] = \nabla f(\mathbf{w}_t)
$$

The mini-batch gradient is an **unbiased** estimate with variance proportional to $1/|\mathcal{B}|$.

Key consequences:

1. **Cheap steps, many of them.** SGD makes progress long before a full pass over the data.
2. **Noise floor.** With a constant learning rate, SGD does not converge exactly — it bounces around the minimum in a region whose size scales with $\eta \times$ gradient variance. Hence **learning-rate decay**.
3. **Robbins–Monro conditions** for convergence: $\sum_t \eta_t = \infty$ and $\sum_t \eta_t^2 < \infty$ (e.g. $\eta_t \propto 1/t$).
4. **Implicit regularisation.** SGD's noise may help it find flatter minima that generalise better — an active research topic.

```python
import numpy as np

rng = np.random.default_rng(0)
n, d = 10_000, 5
X = rng.normal(size=(n, d)); w_true = rng.normal(size=d)
y = X @ w_true + 0.5 * rng.normal(size=n)

def run(batch, lr, epochs=5):
    w = np.zeros(d)
    for epoch in range(epochs):
        idx = rng.permutation(n)
        for start in range(0, n, batch):
            b = idx[start:start + batch]
            grad = 2 * X[b].T @ (X[b] @ w - y[b]) / len(b)
            w -= lr * grad
    return np.linalg.norm(w - w_true)

for batch in [1, 32, 1024, n]:
    print(f"batch {batch:>5}: error {run(batch, lr=0.01 if batch < 100 else 0.1):.4f}")
```

## Linear scaling and large batches

Doubling the batch halves gradient variance. A useful heuristic — the **linear scaling rule** — increases the learning rate proportionally to the batch size, with a **warm-up** period at the start of training. Beyond a "critical batch size", however, bigger batches give diminishing returns: you save wall-clock time on parallel hardware but spend more total computation.

## Beyond plain gradient descent

Plain SGD struggles with ill-conditioning and noisy gradients. The deep-learning track covers the fixes:

- **Momentum** and Nesterov momentum — accumulate velocity to move faster along consistent directions.
- **Adaptive methods** (AdaGrad, RMSProp, Adam, AdamW) — per-parameter step sizes.
- **Schedules** — warm-up, step decay, cosine annealing.
- **Second-order ideas** — Newton, quasi-Newton (L-BFGS), and approximate curvature methods.

:::exercise
1. For $f(w) = 5w^2$, find the largest learning rate for which gradient descent converges. Verify numerically at $\eta$ slightly below and above it.
2. Implement gradient descent with $\eta_t = \eta_0/(1 + t/100)$ on the SGD example and compare the final error with a constant learning rate.
3. Explain why shuffling data every epoch matters for SGD.
:::

:::takeaway
- Gradient descent steps against the gradient; decrease is guaranteed for $\eta < 2/L$.
- Convergence is $O(1/T)$ for convex, linear for strongly convex, and to stationary points for non-convex functions.
- SGD uses unbiased mini-batch gradients: cheap steps, a noise floor, and a need for decay schedules.
- Conditioning, batch size and learning rate interact — tune them together.
:::

=== POST ===
slug: lagrange-multipliers-constrained-optimization
title: Lagrange Multipliers and Constrained Optimisation (KKT Conditions)
category: math
level: Advanced
tags: optimization, lagrange multipliers, kkt, duality, svm
summary: Many ML problems impose constraints: margins, budgets, probabilities that sum to one. We derive Lagrange multipliers, the KKT conditions and duality — the mathematics behind support vector machines.
---
Unconstrained optimisation asks "where is the lowest point?". Constrained optimisation asks "where is the lowest point **within the allowed region**?". Probabilities must sum to one; an SVM's points must lie outside the margin; a fair classifier must satisfy fairness constraints. **Lagrange multipliers** and the **Karush–Kuhn–Tucker (KKT)** conditions are the tools, and duality theory reveals hidden structure — such as the support vectors of an SVM.

## Equality constraints

Minimise $f(\mathbf{x})$ subject to $h(\mathbf{x}) = 0$. At the optimum, you cannot move along the constraint surface and decrease $f$. That means $\nabla f$ has no component along the surface — it must be **parallel** to the constraint normal $\nabla h$:

$$
\nabla f(\mathbf{x}^*) + \lambda\,\nabla h(\mathbf{x}^*) = \mathbf{0}
$$

for some scalar $\lambda$, the **Lagrange multiplier**. We package this with the **Lagrangian**:

$$
\mathcal{L}(\mathbf{x}, \lambda) = f(\mathbf{x}) + \lambda\,h(\mathbf{x})
$$

and find its stationary points: $\nabla_{\mathbf{x}}\mathcal{L} = \mathbf{0}$ and $\partial\mathcal{L}/\partial\lambda = h(\mathbf{x}) = 0$.

:::example Maximum entropy distribution
Maximise $H(\mathbf{p}) = -\sum_i p_i\ln p_i$ subject to $\sum_i p_i = 1$. The Lagrangian is $-\sum_i p_i\ln p_i + \lambda(\sum_i p_i - 1)$. Setting $\partial/\partial p_i = -\ln p_i - 1 + \lambda = 0$ gives $p_i = e^{\lambda - 1}$ — the same for all $i$. So the maximum-entropy distribution with no other constraints is **uniform**. Add a constraint on the mean and you get the exponential family — including the softmax.
:::

### Interpretation of $\lambda$

The multiplier measures the **sensitivity** of the optimal value to the constraint: if the constraint is relaxed to $h(\mathbf{x}) = \epsilon$, the optimal value changes by approximately $-\lambda\epsilon$. In economics it is a "shadow price".

## Inequality constraints and KKT

Now minimise $f(\mathbf{x})$ subject to $g_i(\mathbf{x}) \le 0$ for $i = 1, \dots, m$ and $h_j(\mathbf{x}) = 0$. The Lagrangian is

$$
\mathcal{L}(\mathbf{x}, \boldsymbol{\alpha}, \boldsymbol{\lambda}) = f(\mathbf{x}) + \sum_i \alpha_i g_i(\mathbf{x}) + \sum_j \lambda_j h_j(\mathbf{x})
$$

The **KKT conditions** at an optimum $\mathbf{x}^*$ (under mild regularity conditions) are:

1. **Stationarity**: $\nabla f(\mathbf{x}^*) + \sum_i \alpha_i\nabla g_i(\mathbf{x}^*) + \sum_j \lambda_j\nabla h_j(\mathbf{x}^*) = \mathbf{0}$.
2. **Primal feasibility**: $g_i(\mathbf{x}^*) \le 0$, $h_j(\mathbf{x}^*) = 0$.
3. **Dual feasibility**: $\alpha_i \ge 0$.
4. **Complementary slackness**: $\alpha_i\,g_i(\mathbf{x}^*) = 0$ for every $i$.

Complementary slackness is the elegant part: for each inequality, either the constraint is **active** ($g_i = 0$, the solution sits on the boundary) or its multiplier is **zero** (the constraint does not matter). For convex problems, the KKT conditions are both necessary and sufficient.

## Duality

Define the **dual function** by minimising the Lagrangian over $\mathbf{x}$:

$$
d(\boldsymbol{\alpha}, \boldsymbol{\lambda}) = \min_{\mathbf{x}}\mathcal{L}(\mathbf{x}, \boldsymbol{\alpha}, \boldsymbol{\lambda})
$$

For any $\boldsymbol{\alpha} \ge 0$, $d \le f^*$ — the dual gives a **lower bound** (weak duality). The **dual problem** maximises this bound. For convex problems satisfying Slater's condition (a strictly feasible point exists), **strong duality** holds: the dual optimum equals the primal optimum.

The dual can be easier to solve, and it often exposes structure.

## The payoff: support vector machines

The hard-margin SVM solves

$$
\min_{\mathbf{w}, b}\; \frac{1}{2}\|\mathbf{w}\|^2 \quad \text{s.t.} \quad y_i(\mathbf{w}^\top\mathbf{x}_i + b) \ge 1 \;\; \forall i
$$

Forming the Lagrangian with multipliers $\alpha_i \ge 0$ and applying stationarity gives

$$
\mathbf{w} = \sum_i \alpha_i y_i\mathbf{x}_i, \qquad \sum_i \alpha_i y_i = 0
$$

Substituting back yields the **dual**:

$$
\max_{\boldsymbol{\alpha} \ge 0}\; \sum_i \alpha_i - \frac{1}{2}\sum_{i,j}\alpha_i\alpha_j y_i y_j\,\mathbf{x}_i^\top\mathbf{x}_j \quad \text{s.t.} \quad \sum_i \alpha_i y_i = 0
$$

Two profound consequences:

1. By complementary slackness, $\alpha_i > 0$ only for points **on the margin** — the **support vectors**. The solution depends only on them.
2. The data appear only through **dot products** $\mathbf{x}_i^\top\mathbf{x}_j$. Replace them with a kernel $k(\mathbf{x}_i, \mathbf{x}_j)$ and you get non-linear SVMs for free — the **kernel trick**.

```python
import numpy as np
from sklearn.svm import SVC

rng = np.random.default_rng(0)
X = np.vstack([rng.normal([-2, -2], 1, (50, 2)), rng.normal([2, 2], 1, (50, 2))])
y = np.array([-1] * 50 + [1] * 50)
clf = SVC(kernel="linear", C=1e3).fit(X, y)
print("number of support vectors:", clf.n_support_)       # only a handful
w_from_dual = (clf.dual_coef_ @ clf.support_vectors_).ravel()
print(np.allclose(w_from_dual, clf.coef_.ravel()))          # w = sum alpha_i y_i x_i
```

## Constraints in modern ML

- **Probability simplex** constraints appear in attention, mixtures and optimal transport.
- **Norm-ball** constraints define adversarial perturbations ($\|\boldsymbol{\delta}\|_\infty \le \epsilon$); **projected gradient descent** handles them by projecting after each step.
- **Trust regions** in reinforcement learning (TRPO) constrain policy change with a KL-divergence constraint, solved via Lagrangian methods; PPO approximates this with clipping.
- **Fairness constraints** can be imposed with Lagrangian (min–max) training.

:::exercise
1. Maximise $f(x, y) = xy$ subject to $x + y = 10$ using a Lagrange multiplier.
2. Write the KKT conditions for $\min x^2$ subject to $x \ge 1$. Which constraint is active?
3. Derive the SVM dual from the primal, showing each step of stationarity and substitution.
:::

:::takeaway
- At a constrained optimum, $\nabla f$ is a combination of constraint gradients (Lagrange multipliers).
- KKT = stationarity + primal feasibility + dual feasibility + complementary slackness.
- The dual gives lower bounds; strong duality holds for well-behaved convex problems.
- SVM duality reveals support vectors and enables the kernel trick.
:::

=== POST ===
slug: statistical-hypothesis-testing-for-ml
title: Statistical Hypothesis Testing for ML: Is Model A Really Better?
category: math
level: Intermediate
tags: statistics, hypothesis testing, p-value, confidence intervals, bootstrap
summary: A 1% accuracy gain may be noise. We cover confidence intervals, p-values, paired tests, McNemar's test, the bootstrap and multiple-comparison pitfalls so your experimental claims hold up.
---
Your new model scores 87.4% accuracy; the baseline scores 86.9%. Is your model better — or did you get lucky with this test set, this random seed, this data split? Many published ML "improvements" fail to replicate because authors never asked this question. Today we learn the statistical tools to answer it honestly.

## The logic of hypothesis testing

1. State a **null hypothesis** $H_0$: "the two models have the same true accuracy."
2. Choose a **test statistic** that measures the observed difference.
3. Compute the **p-value**: the probability, *assuming $H_0$ is true*, of observing a difference at least as extreme as the one you saw.
4. If $p < \alpha$ (commonly 0.05), reject $H_0$.

:::warning
A p-value is **not** the probability that $H_0$ is true, and it is **not** the probability your result is a fluke. It says how surprising your data would be if there were no real effect. A tiny p-value with a tiny effect size may be statistically significant but practically irrelevant. Always report **effect sizes with confidence intervals**, not p-values alone.
:::

Two error types:

| | $H_0$ true | $H_0$ false |
|---|---|---|
| Reject $H_0$ | Type I error (false positive), rate $\alpha$ | Correct (power $= 1 - \beta$) |
| Fail to reject | Correct | Type II error (false negative), rate $\beta$ |

## Confidence intervals for accuracy

Accuracy on $n$ independent test examples is a binomial proportion. A normal-approximation 95% confidence interval is

$$
\hat{p} \pm 1.96\sqrt{\frac{\hat{p}(1 - \hat{p})}{n}}
$$

For $\hat{p} = 0.87$ and $n = 1000$: $0.87 \pm 0.021$. A 0.5% difference between two models is well inside this uncertainty. (For small $n$ or extreme $\hat{p}$, use the Wilson interval.)

## Paired comparisons: McNemar's test

When two classifiers are evaluated **on the same test set**, their errors are correlated. Comparing their confidence intervals independently wastes information. **McNemar's test** focuses on disagreements:

| | B correct | B wrong |
|---|---|---|
| **A correct** | $n_{11}$ | $n_{10}$ |
| **A wrong** | $n_{01}$ | $n_{00}$ |

Under $H_0$, disagreements are equally likely in both directions. The statistic

$$
\chi^2 = \frac{(|n_{10} - n_{01}| - 1)^2}{n_{10} + n_{01}}
$$

is approximately $\chi^2$ with one degree of freedom. For small counts, use the exact binomial test on $n_{10}$ out of $n_{10} + n_{01}$.

## Comparing across data splits and seeds

Performance also varies with random initialisation and data splits. Good practice:

- Run each method with **multiple seeds** (e.g. 5–10) and report mean ± standard deviation.
- For cross-validation, use a **paired t-test** across folds — but note that folds share training data, which violates independence and inflates false positives. The **5×2 cross-validation paired t-test** (Dietterich) and the **corrected resampled t-test** (Nadeau & Bengio) address this.
- For many datasets, compare methods with the **Wilcoxon signed-rank test**, which does not assume normality.

## The bootstrap: a universal tool

When no formula exists (for F1, AUC, BLEU…), use the **bootstrap**: resample the test set with replacement many times, recompute the metric, and read confidence intervals from the empirical distribution.

```python
import numpy as np

rng = np.random.default_rng(0)
n = 1000
y_true = rng.integers(0, 2, n)
pred_a = np.where(rng.random(n) < 0.87, y_true, 1 - y_true)   # ~87% accurate
pred_b = np.where(rng.random(n) < 0.86, y_true, 1 - y_true)   # ~86% accurate

diffs = []
for _ in range(10_000):
    idx = rng.integers(0, n, n)                # paired resampling: same indices for both
    diffs.append((pred_a[idx] == y_true[idx]).mean() - (pred_b[idx] == y_true[idx]).mean())
lo, hi = np.percentile(diffs, [2.5, 97.5])
print(f"accuracy difference: {np.mean(diffs):.4f}, 95% CI [{lo:.4f}, {hi:.4f}]")

# McNemar exact test
from scipy.stats import binomtest
a_ok, b_ok = pred_a == y_true, pred_b == y_true
n10, n01 = int((a_ok & ~b_ok).sum()), int((~a_ok & b_ok).sum())
print("McNemar p-value:", binomtest(n10, n10 + n01, 0.5).pvalue)
```

If the interval for the difference includes zero, you cannot claim an improvement.

## Multiple comparisons and the garden of forking paths

Test 20 hyperparameter settings at $\alpha = 0.05$ and, on average, one will look "significant" by chance. Corrections:

- **Bonferroni**: use $\alpha/m$ for $m$ tests (conservative).
- **Holm–Bonferroni**: a uniformly more powerful step-down version.
- **Benjamini–Hochberg**: controls the false discovery rate.

More insidious is **test-set overfitting**: repeatedly evaluating on the test set and keeping what works turns the test set into a training set. Use a **validation set** for all decisions and touch the test set **once** at the end.

:::note
Benchmark culture amplifies these problems: thousands of researchers evaluate on the same test sets, and small gains get published. Careful replication studies have found that some reported improvements shrink or vanish under matched tuning budgets and multiple seeds. Rigor is a competitive advantage — reviewers and employers notice it.
:::

## A checklist for claiming "Model A beats Model B"

1. Same data splits, same preprocessing, comparable tuning budgets.
2. Multiple seeds; report mean, standard deviation and confidence intervals.
3. Paired test (McNemar, bootstrap on differences, Wilcoxon).
4. Correct for multiple comparisons.
5. Report the effect size and discuss practical significance.
6. The test set was used only once.

:::exercise
1. Compute the 95% confidence interval for 92% accuracy on 250 test examples.
2. Two models disagree on 60 test examples: A is right on 38, B on 22. Run McNemar's test.
3. Use the bootstrap to compute a 95% CI for the F1 score of a classifier on a test set of your choice.
:::

:::takeaway
- A p-value measures surprise under the null, not the probability of a fluke.
- Report confidence intervals and effect sizes; accuracy estimates on small test sets are noisy.
- Use paired tests (McNemar, paired bootstrap) when models share a test set.
- Beware multiple comparisons and test-set reuse.
:::

=== POST ===
slug: sampling-and-monte-carlo-methods
title: Sampling and Monte Carlo Methods
category: math
level: Intermediate
tags: sampling, monte carlo, importance sampling, mcmc, metropolis-hastings
summary: When integrals are intractable, we estimate them by sampling. We cover Monte Carlo estimation, inverse-transform and rejection sampling, importance sampling and Markov chain Monte Carlo.
---
Many quantities in machine learning are expectations: the expected loss, the posterior predictive, the normalising constant of a distribution, the value of a policy. Most of these integrals have no closed form. **Monte Carlo methods** estimate them by drawing random samples. The idea is old — it was named during the Manhattan Project after the casino in Monaco — and it is one of the most useful ideas in computational science.

## Monte Carlo estimation

To estimate $\mu = \mathbb{E}_{p}[f(X)] = \int f(x)p(x)\,dx$, draw $x_1, \dots, x_N \sim p$ and average:

$$
\hat\mu_N = \frac{1}{N}\sum_{i=1}^{N} f(x_i)
$$

The estimator is **unbiased**, and its standard error is $\sigma_f/\sqrt{N}$. Crucially, this rate **does not depend on dimension** — unlike grid-based numerical integration, whose cost explodes exponentially with dimension. That is why Monte Carlo dominates in high dimensions.

```python
import numpy as np

rng = np.random.default_rng(0)
# Estimate pi: fraction of random points in the unit square inside the quarter circle
for N in [100, 10_000, 1_000_000]:
    pts = rng.random((N, 2))
    est = 4 * np.mean((pts**2).sum(1) <= 1)
    print(f"N={N:>9}: pi ~ {est:.5f}   (error {abs(est - np.pi):.5f})")
```

Every factor of 100 in samples gives one more correct digit — the $1/\sqrt{N}$ law in action.

## Generating samples

### Inverse transform sampling
If $F$ is the CDF of $X$ and $U \sim \text{Uniform}(0,1)$, then $F^{-1}(U)$ has distribution $F$. For the exponential distribution, $F^{-1}(u) = -\ln(1 - u)/\lambda$.

### Rejection sampling
To sample from a target $p(x)$ known up to a constant, use a proposal $q(x)$ with $M q(x) \ge \tilde{p}(x)$ everywhere. Draw $x \sim q$ and $u \sim U(0,1)$; accept if $u < \tilde{p}(x)/(Mq(x))$. It is exact but becomes hopelessly inefficient in high dimensions because the acceptance rate collapses.

## Importance sampling

Sometimes we cannot sample from $p$, or sampling from $p$ rarely hits the region that matters (rare events). Sample from a proposal $q$ instead and reweight:

$$
\mathbb{E}_p[f(X)] = \mathbb{E}_q\left[f(X)\frac{p(X)}{q(X)}\right] \approx \frac{1}{N}\sum_{i=1}^{N} f(x_i)\,w_i, \qquad w_i = \frac{p(x_i)}{q(x_i)}
$$

When $p$ is only known up to a constant, normalise the weights (**self-normalised importance sampling**). The variance depends heavily on how well $q$ matches $|f|p$; a poor proposal produces a few enormous weights and a useless estimate. The **effective sample size** $\text{ESS} = (\sum w_i)^2/\sum w_i^2$ diagnoses this.

:::note
Importance sampling appears throughout ML: **off-policy reinforcement learning** reweights trajectories collected by an old policy (PPO's probability ratio is an importance weight); **particle filters** track hidden states with weighted samples; and **evaluation under distribution shift** reweights test examples.
:::

## Markov chain Monte Carlo (MCMC)

For complex high-dimensional distributions — such as Bayesian posteriors known only up to a normalising constant — we construct a **Markov chain** whose stationary distribution is the target. After a **burn-in** period, the chain's states are (correlated) samples from $p$.

### Metropolis–Hastings
From the current state $x$, propose $x' \sim q(x' \mid x)$ and accept with probability

$$
A = \min\left(1,\; \frac{\tilde{p}(x')\,q(x \mid x')}{\tilde{p}(x)\,q(x' \mid x)}\right)
$$

The normalising constant cancels — we only need the unnormalised density $\tilde{p}$. This is why MCMC is so powerful for Bayesian inference, where the evidence $p(\mathcal{D})$ is intractable.

```python
import numpy as np

def log_target(x):          # unnormalised mixture of two Gaussians
    return np.logaddexp(-0.5 * (x + 2) ** 2, -0.5 * ((x - 3) / 0.7) ** 2)

rng = np.random.default_rng(1)
x, samples, accepted = 0.0, [], 0
for t in range(60_000):
    prop = x + rng.normal(0, 1.5)                  # symmetric proposal: q terms cancel
    if np.log(rng.random()) < log_target(prop) - log_target(x):
        x, accepted = prop, accepted + 1
    if t >= 10_000:                                # discard burn-in
        samples.append(x)
print("acceptance rate:", accepted / 60_000, " mean:", np.mean(samples).round(3))
```

### Gibbs sampling
Update one variable at a time from its full conditional $p(x_j \mid \mathbf{x}_{-j})$. It is a special case of Metropolis–Hastings with acceptance probability one, ideal when conditionals are easy (e.g. conjugate models, LDA topic models).

### Hamiltonian Monte Carlo
Random-walk proposals explore high-dimensional spaces slowly. **HMC** uses gradients of $\log p$ to simulate physical dynamics and make long, informed moves. The **NUTS** variant tunes itself automatically and powers probabilistic programming tools such as Stan, PyMC and NumPyro.

## Diagnosing MCMC

- **Trace plots** should look like "fuzzy caterpillars", not slow drifts.
- Run **multiple chains** from different starts and compare with $\hat{R}$ (should be close to 1.0).
- Report **effective sample size**, since samples are autocorrelated.

## Monte Carlo in deep learning

- **SGD** itself is a Monte Carlo estimate of the full gradient.
- **Dropout at test time** (MC dropout) approximates Bayesian model averaging.
- **VAEs** estimate the ELBO with sampled latent variables.
- **Policy gradients** estimate expected returns from sampled trajectories.
- **Diffusion models** generate data by iteratively sampling from learned conditional distributions.

:::exercise
1. Use inverse-transform sampling to draw from an exponential distribution with $\lambda = 2$ and verify the mean.
2. Estimate $P(X > 4)$ for $X \sim \mathcal{N}(0,1)$ with plain Monte Carlo and with importance sampling from $\mathcal{N}(4, 1)$. Compare variances.
3. Run the Metropolis code with proposal standard deviations 0.1, 1.5 and 10. Explain the acceptance rates and mixing.
:::

:::takeaway
- Monte Carlo estimates expectations by averaging samples; error falls as $1/\sqrt{N}$ regardless of dimension.
- Inverse-transform and rejection sampling generate exact samples in simple cases.
- Importance sampling reweights samples from a proposal; watch the effective sample size.
- MCMC (Metropolis–Hastings, Gibbs, HMC) samples from unnormalised high-dimensional distributions.
:::

=== POST ===
slug: markov-chains-for-ml
title: Markov Chains: Memoryless Processes and Stationary Distributions
category: math
level: Intermediate
tags: probability, markov chains, stationary distribution, pagerank, random walks
summary: Markov chains model sequences where the future depends only on the present. We study transition matrices, stationary distributions, ergodicity and mixing, with applications from PageRank to MCMC and RL.
---
Weather tomorrow depends mostly on today's weather. A user's next click depends mostly on the current page. A board game's next position depends only on the current position and the dice. Processes like these, where **the future is independent of the past given the present**, are **Markov chains**. They underpin hidden Markov models, MCMC sampling, PageRank and — through Markov decision processes — reinforcement learning.

## Definition

A sequence of random variables $X_0, X_1, X_2, \dots$ on a state space $S$ is a (time-homogeneous) Markov chain if

$$
P(X_{t+1} = j \mid X_t = i, X_{t-1}, \dots, X_0) = P(X_{t+1} = j \mid X_t = i) = P_{ij}
$$

The **transition matrix** $\mathbf{P}$ has rows that sum to one (a *stochastic* matrix). If $\boldsymbol{\pi}_t$ is the row vector of state probabilities at time $t$:

$$
\boldsymbol{\pi}_{t+1} = \boldsymbol{\pi}_t\mathbf{P}, \qquad \boldsymbol{\pi}_t = \boldsymbol{\pi}_0\mathbf{P}^t
$$

:::example
A simple weather model with states Sunny (S) and Rainy (R):
$\mathbf{P} = \begin{bmatrix} 0.8 & 0.2 \\ 0.4 & 0.6 \end{bmatrix}$. If today is sunny, $\boldsymbol{\pi}_0 = (1, 0)$; tomorrow $(0.8, 0.2)$; the day after $(0.72, 0.28)$; and in the long run the probabilities approach $(2/3, 1/3)$ regardless of today's weather.
:::

## Stationary distributions

A distribution $\boldsymbol{\pi}$ is **stationary** if $\boldsymbol{\pi}\mathbf{P} = \boldsymbol{\pi}$ — once the chain is distributed this way, it stays so. It is a **left eigenvector** of $\mathbf{P}$ with eigenvalue 1. For the weather chain: $\pi_S = 0.8\pi_S + 0.4\pi_R$ with $\pi_S + \pi_R = 1$ gives $\boldsymbol{\pi} = (2/3, 1/3)$.

## When does a chain converge?

A finite chain has a unique stationary distribution and converges to it from any starting state if it is:

- **Irreducible** — every state can reach every other state;
- **Aperiodic** — the chain does not cycle with a fixed period (e.g. a self-loop somewhere suffices).

Such chains are called **ergodic**. The **ergodic theorem** adds that time averages equal expectations under $\boldsymbol{\pi}$:

$$
\frac{1}{T}\sum_{t=1}^{T} f(X_t) \to \mathbb{E}_{\boldsymbol{\pi}}[f]
$$

This is precisely why MCMC works: build an ergodic chain whose stationary distribution is your target, run it, and average.

## Detailed balance

A sufficient (not necessary) condition for $\boldsymbol{\pi}$ to be stationary is **detailed balance**:

$$
\pi_i P_{ij} = \pi_j P_{ji} \quad \text{for all } i, j
$$

The probability flow from $i$ to $j$ equals the flow back. The Metropolis–Hastings acceptance rule is designed exactly to enforce detailed balance with respect to the target distribution.

## Mixing time

How quickly does the chain forget its start? The **mixing time** depends on the **spectral gap** $1 - |\lambda_2|$, where $\lambda_2$ is the second-largest eigenvalue magnitude of $\mathbf{P}$. A small gap means slow mixing — for example, a chain on two well-separated modes that rarely jumps between them. Slow mixing is the central practical problem of MCMC.

## PageRank: a Markov chain on the web

Model a web surfer who follows a random outgoing link with probability $d$ (the damping factor, typically 0.85) and jumps to a random page with probability $1 - d$. The random jump makes the chain irreducible and aperiodic. Each page's **PageRank** is its stationary probability — the long-run fraction of time the surfer spends there.

```python
import numpy as np

links = {0: [1, 2], 1: [2], 2: [0], 3: [2]}      # page -> pages it links to
n, d = 4, 0.85
P = np.zeros((n, n))
for i, outs in links.items():
    for j in outs:
        P[i, j] = 1 / len(outs)
G = d * P + (1 - d) / n                           # "Google matrix"

pi = np.ones(n) / n
for _ in range(100):                              # power iteration
    pi = pi @ G
print("PageRank:", pi.round(4))

w, V = np.linalg.eig(G.T)                         # check: eigenvector with eigenvalue 1
v = np.real(V[:, np.argmin(abs(w - 1))]); print((v / v.sum()).round(4))
```

Page 2, which everyone links to, receives the highest rank; page 3, which nobody links to, receives only the random-jump share.

## Absorbing chains

Some states are **absorbing** — once entered, never left (game over, customer churn). With the fundamental matrix $\mathbf{N} = (\mathbf{I} - \mathbf{Q})^{-1}$, where $\mathbf{Q}$ holds transitions among transient states, we can compute expected steps before absorption and absorption probabilities. This is useful for modelling user journeys, student progression through a curriculum, or case-processing pipelines.

## Where Markov chains appear in ML

- **n-gram language models** are Markov chains over words.
- **Hidden Markov models** add noisy observations to a hidden chain.
- **MCMC** samples from posteriors.
- **Markov decision processes** add actions and rewards — the foundation of reinforcement learning.
- **Random-walk graph embeddings** (DeepWalk, node2vec) learn node representations from random walks.
- **Diffusion models** use a Markov chain that gradually adds noise, and learn the reverse chain.

:::exercise
1. Compute the stationary distribution of $\mathbf{P} = \begin{bmatrix} 0.5 & 0.5 & 0 \\ 0.25 & 0.5 & 0.25 \\ 0 & 0.5 & 0.5 \end{bmatrix}$ by solving $\boldsymbol{\pi}\mathbf{P} = \boldsymbol{\pi}$.
2. Construct a periodic chain and show numerically that $\boldsymbol{\pi}_0\mathbf{P}^t$ does not converge.
3. Build a bigram (Markov) text generator from a paragraph of English and generate a sentence.
:::

:::takeaway
- A Markov chain's future depends only on its present state; transitions form a stochastic matrix.
- The stationary distribution is the eigenvector of $\mathbf{P}^\top$ with eigenvalue 1.
- Irreducible + aperiodic ⇒ convergence to a unique stationary distribution; detailed balance ensures stationarity.
- PageRank, MCMC, HMMs, MDPs and diffusion models are all built on Markov chains.
:::

=== POST ===
slug: numerical-stability-in-ml
title: Numerical Stability: Floating Point, Log-Sum-Exp and Avoiding NaNs
category: math
level: Intermediate
tags: numerical computing, floating point, log-sum-exp, softmax, precision
summary: Mathematically correct code can still produce NaN. We study floating-point arithmetic, overflow and underflow, catastrophic cancellation, the log-sum-exp trick, stable softmax and mixed-precision pitfalls.
---
One of the most frustrating experiences in ML is watching a loss curve suddenly turn into `NaN`. The mathematics was correct; the arithmetic was not. Computers do not work with real numbers — they work with **floating-point** approximations. Knowing where those approximations break is a core professional skill.

## Floating-point numbers

A floating-point number stores a sign, an exponent and a mantissa (significand), representing $\pm m \times 2^e$. Common formats:

| Format | Bits | Approx. decimal digits | Max value | Smallest normal |
|---|---|---|---|---|
| float64 (double) | 64 | ~16 | $\sim 1.8 \times 10^{308}$ | $\sim 2.2 \times 10^{-308}$ |
| float32 (single) | 32 | ~7 | $\sim 3.4 \times 10^{38}$ | $\sim 1.2 \times 10^{-38}$ |
| float16 (half) | 16 | ~3 | 65,504 | $\sim 6.1 \times 10^{-5}$ |
| bfloat16 | 16 | ~2–3 | $\sim 3.4 \times 10^{38}$ | $\sim 1.2 \times 10^{-38}$ |

**Machine epsilon** is the gap between 1 and the next representable number: about $2.2 \times 10^{-16}$ for float64 and $1.2 \times 10^{-7}$ for float32. Every operation may introduce a relative error of this size.

```python
import numpy as np
print(0.1 + 0.2 == 0.3)                      # False!
print(np.float32(1) + np.float32(1e-8) == 1) # True: 1e-8 is below float32 epsilon
print(np.float16(70000))                     # inf: overflow in half precision
```

:::warning
Never compare floats with `==`. Use `np.isclose` or `math.isclose` with sensible tolerances.
:::

## The main failure modes

1. **Overflow** — a result exceeds the largest representable number and becomes `inf`. `np.exp(1000)` overflows in float64.
2. **Underflow** — a result is smaller than the smallest representable number and becomes 0. Multiplying hundreds of probabilities underflows.
3. **Catastrophic cancellation** — subtracting nearly equal numbers destroys significant digits. Computing variance as $\mathbb{E}[X^2] - \mathbb{E}[X]^2$ for data with a large mean can even produce negative variance.
4. **Invalid operations** — `inf - inf`, `0 * inf`, `0/0` and `log(0)` produce `NaN` or `-inf`, which then spread through every later computation.

## The log-sum-exp trick

Softmax and many likelihoods require $\ln\sum_i e^{x_i}$. If some $x_i = 1000$, $e^{1000}$ overflows. If all $x_i = -1000$, every term underflows to 0 and the log gives $-\infty$. The fix: factor out the maximum $m = \max_i x_i$:

$$
\ln\sum_i e^{x_i} = m + \ln\sum_i e^{x_i - m}
$$

Now the largest exponent is $e^0 = 1$ — no overflow — and at least one term equals 1 — no total underflow.

### Stable softmax

$$
\text{softmax}(\mathbf{x})_i = \frac{e^{x_i - m}}{\sum_j e^{x_j - m}}
$$

Subtracting the max does not change the result mathematically but keeps it computable.

```python
import numpy as np

def naive_softmax(x):
    e = np.exp(x); return e / e.sum()

def stable_softmax(x):
    z = x - x.max(); e = np.exp(z); return e / e.sum()

def log_softmax(x):
    m = x.max()
    return x - (m + np.log(np.exp(x - m).sum()))

x = np.array([1000.0, 1001.0, 1002.0])
with np.errstate(all="ignore"):
    print("naive: ", naive_softmax(x))      # [nan nan nan]
print("stable:", stable_softmax(x))         # [0.090 0.245 0.665]
print("log-softmax:", log_softmax(x))
```

## Stable cross-entropy and sigmoid

Computing `log(sigmoid(z))` naively fails for large negative $z$ (sigmoid underflows to 0). Use the identity

$$
\ln\sigma(z) = -\ln(1 + e^{-z}) = -\text{softplus}(-z)
$$

and implement softplus stably as $\max(z, 0) + \ln(1 + e^{-|z|})$. This is why deep-learning frameworks provide `BCEWithLogitsLoss` and `cross_entropy` functions that take **logits**, not probabilities. Always pass logits to these fused losses.

## Other practical techniques

- **Work in log space** for products of probabilities (HMMs, Naive Bayes, sequence likelihoods).
- **Add small epsilons carefully**: `log(p + 1e-12)` or `x / (norm + 1e-8)` prevent `log(0)` and division by zero — but choose epsilon relative to the precision (1e-12 is meaningless in float16).
- **Welford's algorithm** computes running mean and variance stably in one pass.
- **Solve, don't invert**: use `np.linalg.solve` or Cholesky decomposition rather than explicit matrix inverses.
- **Check condition numbers**: `np.linalg.cond(A)` — if it approaches $1/\epsilon$, results are unreliable.
- **Gradient clipping** limits the norm of gradients to prevent a single bad batch from causing overflow.

## Mixed-precision training

Modern GPUs are much faster in 16-bit arithmetic, so large models train in **mixed precision**: compute in float16 or bfloat16, keep a float32 "master copy" of weights, and accumulate sums in float32.

- **float16** has a narrow range: small gradients underflow to zero. **Loss scaling** multiplies the loss by a large factor before backpropagation (and divides the gradients afterwards) to keep gradients in range; dynamic loss scaling lowers the factor when overflow occurs.
- **bfloat16** keeps float32's exponent range with fewer mantissa bits, so overflow and underflow are rare and loss scaling is usually unnecessary — one reason it became the default for training large models.

:::tip
**Debugging NaNs, in order:** (1) check the input data for NaN/inf; (2) lower the learning rate; (3) make sure you use logits-based loss functions; (4) add gradient clipping; (5) enable anomaly detection (`torch.autograd.set_detect_anomaly(True)`) to find the first operation producing NaN; (6) look for `log`, `sqrt`, division and `exp` in your custom code.
:::

:::exercise
1. Show that subtracting any constant $c$ from all inputs leaves softmax unchanged.
2. Compute the variance of `[1e9 + 1, 1e9 + 2, 1e9 + 3]` in float32 using the naive formula and using Welford's algorithm.
3. Implement a numerically stable `log_sigmoid` and test it for $z \in \{-1000, 0, 1000\}$.
:::

:::takeaway
- Floats have finite range and precision; machine epsilon bounds relative error.
- Overflow, underflow, cancellation and invalid operations cause NaNs.
- Log-sum-exp (subtract the max) stabilises softmax and likelihoods; use logits-based fused losses.
- Mixed precision needs float32 accumulation and loss scaling (float16) or bfloat16.
:::

=== POST ===
slug: curse-of-dimensionality
title: The Curse of Dimensionality
category: math
level: Intermediate
tags: high dimensions, geometry, distance concentration, dimensionality reduction
summary: High-dimensional spaces behave strangely: volume hides in corners, distances concentrate and data becomes sparse. We quantify the curse, explain why ML still works, and survey the remedies.
---
Richard Bellman coined the phrase "curse of dimensionality" in 1957 while studying dynamic programming: the number of states explodes exponentially with the number of variables. In machine learning the curse has several faces — data sparsity, distance concentration, and counter-intuitive geometry. Understanding them explains why nearest-neighbour methods struggle on raw pixels, why feature selection matters, and why deep learning's ability to find low-dimensional structure is so valuable.

## Face 1: exponential sparsity

To cover the unit interval $[0, 1]$ with a grid of spacing 0.1 you need 10 points. To cover the unit cube $[0,1]^d$ at the same resolution you need $10^d$ points. At $d = 20$ that is $10^{20}$ — more than any dataset will ever contain. In high dimensions, **data is always sparse**: most of the space contains no training examples at all.

Consequence for local methods such as k-nearest neighbours: to capture a fraction $r$ of uniformly distributed data in a hypercube neighbourhood, the neighbourhood's edge length must be

$$
e_d(r) = r^{1/d}
$$

To capture 1% of the data in 10 dimensions, you need $0.01^{1/10} \approx 0.63$ — 63% of the range of *each* feature. Your "local" neighbourhood is not local at all.

## Face 2: volume concentrates in the shell

The volume of a $d$-dimensional ball of radius $r$ scales as $r^d$. The fraction of a unit ball's volume lying within the outer shell of thickness $\epsilon$ is

$$
1 - (1 - \epsilon)^d
$$

For $d = 100$ and $\epsilon = 0.05$ this is $1 - 0.95^{100} \approx 0.994$. **Almost all the volume is near the surface.** Similarly, a high-dimensional Gaussian's samples do not cluster near the mean; they concentrate on a thin shell of radius about $\sqrt{d}\,\sigma$.

:::note
This "Gaussian soap bubble" matters in practice. When interpolating between latent codes of a generative model with a Gaussian prior, straight-line interpolation passes through the low-probability interior. **Spherical interpolation (slerp)** stays on the shell and often produces better samples.
:::

## Face 3: distance concentration

For many distributions, as $d$ grows, the distances from a query point to its nearest and farthest neighbours become almost equal:

$$
\frac{\text{dist}_{\max} - \text{dist}_{\min}}{\text{dist}_{\min}} \to 0 \quad \text{as } d \to \infty
$$

If all points are roughly equally far away, "nearest neighbour" loses meaning.

```python
import numpy as np

rng = np.random.default_rng(0)
for d in [2, 10, 100, 1000, 10000]:
    X = rng.random((1000, d))
    q = rng.random(d)
    dist = np.linalg.norm(X - q, axis=1)
    contrast = (dist.max() - dist.min()) / dist.min()
    print(f"d={d:>5}: relative contrast = {contrast:.3f}")
```

Run it: the relative contrast collapses as $d$ grows.

## Face 4: random vectors are orthogonal

The cosine similarity of two random vectors in $\mathbb{R}^d$ concentrates around zero with standard deviation about $1/\sqrt{d}$. In 10,000 dimensions, random vectors are almost exactly orthogonal. This is a *blessing* too: high-dimensional spaces can host an enormous number of nearly-orthogonal directions, which lets embeddings and neural representations pack many features with little interference (the "superposition" hypothesis in interpretability research).

## Face 5: more parameters, more data

A model with more features has more parameters to estimate. Without enough data, it overfits. Classical rules of thumb suggest needing many examples per parameter; the **Hughes phenomenon** describes how, with a fixed training set, classifier accuracy first rises and then falls as features are added.

## Why does machine learning work at all?

If the curse were the whole story, image classification on $224 \times 224 \times 3 = 150{,}528$-dimensional inputs would be impossible. It works because real data is **not** uniformly spread:

1. **The manifold hypothesis** — natural data (images, speech, text) lies near low-dimensional manifolds embedded in the high-dimensional space. The set of plausible face images is a tiny, structured subset of all pixel arrays.
2. **Smoothness and structure** — nearby inputs usually have similar labels, and many features are correlated.
3. **Inductive biases** — convolutional networks assume locality and translation invariance; transformers exploit relational structure. These priors drastically reduce what must be learned from data.
4. **Compositionality** — deep networks build complex functions from simple parts, which can be exponentially more efficient than shallow ones for certain structured functions.

## Remedies

- **Feature selection** — keep only informative features.
- **Dimensionality reduction** — PCA (linear), autoencoders (non-linear), UMAP/t-SNE (for visualisation).
- **Regularisation** — constrain model complexity.
- **Better representations** — learned embeddings in which distances are meaningful; this is why semantic search uses learned embeddings rather than raw bag-of-words vectors.
- **Domain knowledge and architecture** — encode known structure into the model.
- **More data and data augmentation.**

:::exercise
1. Compute the edge length needed to capture 10% of uniform data in $d = 1, 2, 10, 100$ dimensions.
2. Sample from a 500-dimensional standard Gaussian and plot a histogram of sample norms. Where do they concentrate?
3. Train k-NN on a dataset, then append 100, 1,000 and 5,000 random noise features. Plot accuracy against the number of noise features.
:::

:::takeaway
- Covering high-dimensional space needs exponentially many points — data is always sparse.
- Volume concentrates near surfaces; Gaussian samples lie on a thin shell of radius $\approx\sqrt{d}\,\sigma$.
- Distances concentrate, weakening nearest-neighbour methods on raw features.
- ML succeeds because real data lies near low-dimensional manifolds and models encode useful inductive biases.
:::

=== POST ===
slug: tensors-and-tensor-operations
title: Tensors and Tensor Operations: Broadcasting, Reshaping and Einsum
category: math
level: Beginner
tags: tensors, numpy, pytorch, broadcasting, einsum
summary: Deep learning code manipulates multi-dimensional arrays. We master tensor shapes, indexing, broadcasting, reshaping versus transposing, reductions and einsum — the skills that prevent most deep-learning bugs.
---
In deep learning practice, most bugs are **shape bugs**. A batch of colour images is a 4-D array; a batch of token embeddings is 3-D; attention scores are 4-D. The mathematics of these objects is simple, but you must manipulate them fluently. This lecture is deliberately practical: by the end you should be able to read and write tensor code with confidence.

## What is a tensor?

In deep-learning libraries, a **tensor** is simply a multi-dimensional array with a data type and a device (CPU or GPU). Its **rank** (number of axes) and **shape** matter:

| Rank | Name | Example shape | Meaning |
|---|---|---|---|
| 0 | Scalar | `()` | A loss value |
| 1 | Vector | `(768,)` | One embedding |
| 2 | Matrix | `(32, 10)` | Batch of 32 logit vectors |
| 3 | 3-tensor | `(32, 128, 768)` | Batch × sequence length × hidden size |
| 4 | 4-tensor | `(32, 3, 224, 224)` | Batch × channels × height × width (PyTorch "NCHW") |

(Mathematicians use "tensor" for multilinear maps with transformation rules; in ML the word just means n-dimensional array.)

## Indexing and slicing

```python
import numpy as np

x = np.arange(24).reshape(2, 3, 4)      # shape (2, 3, 4)
print(x[0].shape)          # (3, 4)   first item in batch
print(x[:, 1].shape)       # (2, 4)   second row of every item
print(x[..., -1].shape)    # (2, 3)   last element along the final axis
print(x[:, :, ::2].shape)  # (2, 3, 2) every other column
mask = x > 10
print(x[mask].shape)       # (13,)    boolean indexing flattens
```

## Broadcasting

**Broadcasting** lets operations combine tensors of different shapes without copying data. Rules (NumPy and PyTorch):

1. Align shapes from the **right**.
2. Two dimensions are compatible if they are equal or one of them is 1.
3. Missing leading dimensions are treated as 1.
4. Size-1 dimensions are virtually stretched to match.

```python
X = np.random.rand(32, 10)        # batch of 32 feature vectors
mu = X.mean(axis=0)               # shape (10,)
Xc = X - mu                       # (32,10) - (10,) -> broadcast over batch

a = np.arange(3).reshape(3, 1)    # (3,1)
b = np.arange(4).reshape(1, 4)    # (1,4)
print((a + b).shape)              # (3,4): an "outer sum"

# Pairwise squared distances between rows of A (n,d) and B (m,d) -> (n,m)
A, B = np.random.rand(5, 3), np.random.rand(7, 3)
D = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
print(D.shape)                    # (5, 7)
```

:::warning
Broadcasting can silently produce the wrong shape. A classic bug: `y_pred` has shape `(32, 1)` and `y_true` has shape `(32,)`. Then `y_pred - y_true` broadcasts to `(32, 32)`, and the MSE loss is computed over 1,024 meaningless pairs — with no error message. **Assert shapes** in your code.
:::

## Reshape, view, transpose and permute

- `reshape` / `view` reinterpret the same elements in a new shape; the total size must match. Use `-1` for one inferred dimension.
- `transpose` / `permute` reorder axes.

These are **not interchangeable**. Reshaping a `(2, 3)` matrix to `(3, 2)` is different from transposing it:

```python
m = np.array([[1, 2, 3], [4, 5, 6]])
print(m.reshape(3, 2))   # [[1,2],[3,4],[5,6]]  - reads elements in memory order
print(m.T)               # [[1,4],[2,5],[3,6]]  - swaps axes
```

A common pattern in transformers — splitting the hidden dimension into attention heads:

```python
import torch
B, T, H, Dh = 2, 5, 4, 8                  # batch, tokens, heads, head dim
x = torch.randn(B, T, H * Dh)             # (2, 5, 32)
heads = x.view(B, T, H, Dh).transpose(1, 2)   # (2, 4, 5, 8): batch, head, token, dim
merged = heads.transpose(1, 2).reshape(B, T, H * Dh)
print(torch.allclose(merged, x))          # True
```

In PyTorch, `view` requires contiguous memory; after `transpose`, use `.reshape` or call `.contiguous()` first.

## Reductions

Reductions collapse axes: `sum`, `mean`, `max`, `argmax`, `logsumexp`, `norm`. The `axis` (NumPy) or `dim` (PyTorch) argument specifies which axis disappears; `keepdims=True` keeps it as size 1, which is often what you need for subsequent broadcasting (e.g. normalising each row).

```python
logits = np.random.randn(4, 10)
probs = np.exp(logits - logits.max(axis=1, keepdims=True))
probs /= probs.sum(axis=1, keepdims=True)   # each row sums to 1
print(probs.sum(axis=1))
```

## Einstein summation

`einsum` expresses products, transposes and reductions with index notation: indices that appear in inputs but not in the output are summed.

| Operation | einsum |
|---|---|
| Matrix multiply | `'ik,kj->ij'` |
| Batched matrix multiply | `'bik,bkj->bij'` |
| Dot product of rows | `'bd,bd->b'` |
| Outer product | `'i,j->ij'` |
| Trace | `'ii->'` |
| Attention scores | `'bhqd,bhkd->bhqk'` |

```python
Q = torch.randn(2, 4, 5, 8)   # batch, heads, queries, dim
K = torch.randn(2, 4, 5, 8)   # batch, heads, keys, dim
scores = torch.einsum("bhqd,bhkd->bhqk", Q, K) / 8 ** 0.5
print(scores.shape)           # (2, 4, 5, 5)
```

Einsum makes the *meaning* of each axis explicit, which is why many researchers prefer it for complex operations. Libraries such as `einops` add readable `rearrange` operations, e.g. `rearrange(x, 'b t (h d) -> b h t d', h=4)`.

## Memory layout and devices

Tensors live on a device; operations require all operands on the same device (`x.to('cuda')`). Moving data between CPU and GPU is slow — keep the training loop on the GPU. Choose data types deliberately (float32 for general training, bfloat16/float16 for mixed precision, int64 for class indices).

:::tip
Adopt a **shape-comment habit**: annotate every non-trivial line with the resulting shape, e.g. `# (B, T, D)`. It costs seconds and saves hours of debugging.
:::

:::exercise
1. Without loops, compute the cosine-similarity matrix between 100 query vectors and 1,000 document vectors of dimension 64.
2. Given images of shape `(B, H, W, C)`, convert them to `(B, C, H, W)` and flatten each image to a vector.
3. Rewrite batched matrix multiplication and the attention-weighted sum of values using `einsum`.
:::

:::takeaway
- Tensors are n-dimensional arrays; always know the meaning of every axis.
- Broadcasting aligns shapes from the right; it can hide bugs — assert shapes.
- `reshape` and `transpose` are different operations; use `keepdims` in reductions.
- `einsum` expresses complex contractions clearly — especially attention.
:::
