=== POST ===
slug: probability-fundamentals
title: Probability Fundamentals: Sample Spaces, Axioms and Conditional Probability
category: math
level: Beginner
tags: probability, axioms, conditional probability, independence
summary: Machine learning is reasoning under uncertainty. We build probability from Kolmogorov's axioms, then master conditional probability, the product and sum rules, and independence.
---
A classifier that says "this email is spam" is less useful than one that says "this email is spam with probability 0.97". Probability is the language that lets models express — and reason about — uncertainty. Today we lay its foundations with care, because sloppy probabilistic reasoning is one of the most common sources of error in data science.

## Sample spaces and events

- The **sample space** $\Omega$ is the set of all possible outcomes of an experiment. Rolling a die: $\Omega = \{1, 2, 3, 4, 5, 6\}$.
- An **event** $A \subseteq \Omega$ is a set of outcomes. "Even number": $A = \{2, 4, 6\}$.
- Events combine with set operations: $A \cup B$ (A or B), $A \cap B$ (A and B), $A^c$ (not A).

## Kolmogorov's axioms

A **probability measure** $P$ assigns numbers to events such that:

1. $P(A) \ge 0$ for every event $A$;
2. $P(\Omega) = 1$;
3. For **disjoint** events $A_1, A_2, \dots$: $P(\bigcup_i A_i) = \sum_i P(A_i)$.

Everything else follows. For example:

- $P(A^c) = 1 - P(A)$;
- $P(\emptyset) = 0$;
- **Inclusion–exclusion**: $P(A \cup B) = P(A) + P(B) - P(A \cap B)$;
- If $A \subseteq B$ then $P(A) \le P(B)$.

:::note
What does a probability *mean*? **Frequentists** interpret $P(A)$ as the long-run frequency of $A$ in repeated trials. **Bayesians** interpret it as a degree of belief, updated by evidence. The axioms are the same; the interpretation affects how we do statistics. Modern ML uses both freely — a Bayesian neural network and a frequentist confidence interval can coexist in one project.
:::

## Conditional probability

The probability of $A$ given that $B$ occurred is

$$
P(A \mid B) = \frac{P(A \cap B)}{P(B)}, \qquad P(B) > 0
$$

Conditioning **restricts the sample space** to $B$ and renormalises. Rearranging gives the **product rule**:

$$
P(A \cap B) = P(A \mid B)\,P(B) = P(B \mid A)\,P(A)
$$

and extending it to many events gives the **chain rule of probability**:

$$
P(A_1, A_2, \dots, A_n) = P(A_1)\,P(A_2 \mid A_1)\,P(A_3 \mid A_1, A_2) \cdots P(A_n \mid A_{1:n-1})
$$

:::note
The chain rule of probability is exactly how **autoregressive language models** work: the probability of a sentence is the product of the probability of each word given all previous words. GPT-style models learn $P(w_t \mid w_{1:t-1})$ and multiply.
:::

## The law of total probability

If $B_1, \dots, B_k$ partition $\Omega$ (disjoint and covering everything):

$$
P(A) = \sum_{i=1}^{k} P(A \mid B_i)\,P(B_i)
$$

This is the **sum rule** or **marginalisation**. It lets us compute a probability by considering all the ways it can happen.

:::example
A clinic receives patients from three camps: 50% from camp X, 30% from Y, 20% from Z. The rate of a certain infection is 2%, 5% and 10% respectively. The overall infection rate is $0.5(0.02) + 0.3(0.05) + 0.2(0.10) = 0.045$, i.e. 4.5%.
:::

## Independence

Events $A$ and $B$ are **independent** if $P(A \cap B) = P(A)P(B)$, equivalently $P(A \mid B) = P(A)$ — knowing $B$ tells you nothing about $A$.

**Conditional independence**: $A \perp B \mid C$ if $P(A, B \mid C) = P(A \mid C)\,P(B \mid C)$. Conditional independence is the assumption that makes Naive Bayes, HMMs and Bayesian networks tractable.

:::warning
Independence and conditional independence do not imply each other. Two symptoms may be *dependent* in the population (both caused by a disease) yet *independent given* the disease. Conversely, two independent causes become *dependent* given their common effect (explaining away). Confusing these is a classic error.
:::

## Simulation: the best way to build intuition

```python
import numpy as np

rng = np.random.default_rng(42)
N = 1_000_000

# The birthday problem: probability that at least two of 23 people share a birthday
bdays = rng.integers(0, 365, size=(100_000, 23))
shared = np.array([len(set(row)) < 23 for row in bdays])
print("birthday (simulated):", shared.mean())           # ~0.507

# Conditional probability by simulation: two dice, P(sum = 8 | first die is even)
d1, d2 = rng.integers(1, 7, N), rng.integers(1, 7, N)
cond = d1 % 2 == 0
print("P(sum=8 | d1 even):", ((d1 + d2 == 8) & cond).sum() / cond.sum())   # 3/18 = 0.1667
```

The birthday result surprises almost everyone: with only 23 people, a shared birthday is more likely than not. Our intuitions about probability are unreliable — which is precisely why we formalise and simulate.

## Probability in ML, in one paragraph

A supervised model estimates a **conditional distribution** $p(y \mid \mathbf{x})$. Training maximises the probability the model assigns to the observed data (maximum likelihood). Generative models estimate $p(\mathbf{x})$ itself. Evaluation estimates probabilities of errors from finite samples. Uncertainty estimates tell us when to trust predictions. Every one of these rests on today's axioms.

:::exercise
1. Prove $P(A \cup B) = P(A) + P(B) - P(A \cap B)$ from the axioms.
2. In a class, 60% study ML, 40% study NLP, and 25% study both. What is the probability a student studies ML given they study NLP? Are the two independent?
3. Simulate the Monty Hall problem and verify that switching wins with probability 2/3.
:::

:::takeaway
- Probability rests on three axioms; everything else is derived.
- Conditional probability restricts the sample space; the product and chain rules follow.
- The law of total probability marginalises over a partition.
- Independence and conditional independence are distinct — confusing them causes real errors.
:::

=== POST ===
slug: random-variables-and-distributions
title: Random Variables and Probability Distributions
category: math
level: Beginner
tags: probability, distributions, bernoulli, binomial, poisson, pdf
summary: A random variable turns outcomes into numbers. We study discrete and continuous distributions — Bernoulli, categorical, binomial, Poisson, uniform, exponential, Beta — and when ML uses each.
---
In the last lecture we assigned probabilities to events. In practice we care about *numbers*: the number of clicks, the pixel intensity, the class label, the waiting time. A **random variable** attaches numbers to outcomes, and its **distribution** summarises how likely each value is. Choosing the right distribution is choosing the right assumptions for your model.

## Random variables

A random variable $X$ is a function from the sample space to the real numbers. We distinguish:

- **Discrete** random variables take countable values; described by a **probability mass function (PMF)** $p(x) = P(X = x)$, with $\sum_x p(x) = 1$.
- **Continuous** random variables take values in intervals; described by a **probability density function (PDF)** $f(x)$ with $P(a \le X \le b) = \int_a^b f(x)\,dx$ and $\int f = 1$.

Both have a **cumulative distribution function (CDF)** $F(x) = P(X \le x)$.

:::warning
A density is **not** a probability. $f(x)$ can exceed 1 (a uniform density on $[0, 0.5]$ equals 2). For a continuous variable, $P(X = x) = 0$ for every single value; only intervals have positive probability. This matters when you compare "likelihoods" of continuous models.
:::

## Discrete distributions

| Distribution | PMF | Mean | Variance | ML use |
|---|---|---|---|---|
| Bernoulli($p$) | $p^x(1-p)^{1-x}$, $x \in \{0,1\}$ | $p$ | $p(1-p)$ | Binary labels; logistic regression output |
| Categorical($\boldsymbol{\pi}$) | $\pi_k$ for class $k$ | — | — | Multiclass labels; softmax output; next-token prediction |
| Binomial($n, p$) | $\binom{n}{x}p^x(1-p)^{n-x}$ | $np$ | $np(1-p)$ | Number of successes; accuracy on $n$ test items |
| Poisson($\lambda$) | $\frac{\lambda^x e^{-\lambda}}{x!}$ | $\lambda$ | $\lambda$ | Counts per interval: arrivals, clicks, events |
| Geometric($p$) | $(1-p)^{x-1}p$ | $1/p$ | $(1-p)/p^2$ | Trials until first success |

:::example Why the binomial matters for evaluation
If a classifier's true accuracy is 90% and you test it on 100 examples, the number correct is Binomial(100, 0.9) with standard deviation $\sqrt{100 \times 0.9 \times 0.1} = 3$. Observing 87% versus 93% could easily be noise. Small test sets give noisy accuracy estimates — always report uncertainty.
:::

## Continuous distributions

| Distribution | PDF | Mean | Variance | ML use |
|---|---|---|---|---|
| Uniform($a, b$) | $\frac{1}{b - a}$ on $[a, b]$ | $\frac{a+b}{2}$ | $\frac{(b-a)^2}{12}$ | Random initialisation; random search |
| Gaussian($\mu, \sigma^2$) | $\frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$ | $\mu$ | $\sigma^2$ | Noise models; VAEs; diffusion; weight init |
| Exponential($\lambda$) | $\lambda e^{-\lambda x}$, $x \ge 0$ | $1/\lambda$ | $1/\lambda^2$ | Waiting times; survival analysis |
| Beta($\alpha, \beta$) | $\propto x^{\alpha-1}(1-x)^{\beta-1}$ on $[0,1]$ | $\frac{\alpha}{\alpha+\beta}$ | — | Prior over probabilities; A/B testing; bandits |
| Laplace($\mu, b$) | $\frac{1}{2b}e^{-\lvert x-\mu \rvert/b}$ | $\mu$ | $2b^2$ | Robust regression; L1 regularisation prior; differential privacy noise |

The **Dirichlet** distribution generalises the Beta to probability vectors (e.g. topic proportions in LDA topic models).

## Joint, marginal and conditional distributions

For two variables, the **joint distribution** $p(x, y)$ describes them together. The **marginal** is obtained by summing or integrating out the other variable: $p(x) = \sum_y p(x, y)$. The **conditional** is $p(y \mid x) = p(x, y)/p(x)$. This vocabulary is how we describe every model:

- A discriminative classifier models $p(y \mid \mathbf{x})$.
- A generative model models $p(\mathbf{x}, y)$ or $p(\mathbf{x})$.

## Transformations of random variables

If $Y = g(X)$ with $g$ invertible and differentiable, the density transforms as

$$
f_Y(y) = f_X\big(g^{-1}(y)\big)\left|\frac{d}{dy}g^{-1}(y)\right|
$$

The derivative term accounts for how $g$ stretches or compresses space. In many dimensions it becomes the absolute value of a Jacobian determinant — the core of **normalising flows**.

## Sampling in code

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
samples = {
    "Bernoulli(0.3)": rng.binomial(1, 0.3, 10_000),
    "Poisson(4)":     rng.poisson(4, 10_000),
    "Gaussian(0,1)":  rng.normal(0, 1, 10_000),
    "Exponential(1)": rng.exponential(1.0, 10_000),
    "Beta(2,5)":      rng.beta(2, 5, 10_000),
}
fig, axes = plt.subplots(1, 5, figsize=(18, 3))
for ax, (name, s) in zip(axes, samples.items()):
    ax.hist(s, bins=40, density=True, color="#7c3aed", alpha=0.8)
    ax.set_title(f"{name}\nmean={s.mean():.2f} var={s.var():.2f}")
plt.tight_layout(); plt.show()
```

## Choosing a distribution for your model

Choosing the output distribution *is* choosing the loss function:

- Binary target → Bernoulli → **binary cross-entropy**.
- Multiclass target → Categorical → **softmax cross-entropy**.
- Real-valued target with Gaussian noise → **mean squared error**.
- Real-valued target with heavy-tailed noise → Laplace → **mean absolute error**.
- Count target → Poisson → **Poisson loss**.

We will prove this correspondence in the maximum likelihood lecture.

:::exercise
1. A camp receives on average 3 new arrivals per hour. Using the Poisson distribution, what is the probability of more than 6 arrivals in an hour?
2. Show that the variance of Uniform(0, 1) is 1/12 by integration.
3. Plot Beta densities for $(\alpha, \beta) = (1,1), (2,2), (10,2), (0.5, 0.5)$ and describe what prior belief each expresses about a probability.
:::

:::takeaway
- Discrete variables have PMFs; continuous variables have PDFs (densities, not probabilities).
- Know the Bernoulli, categorical, binomial, Poisson, uniform, Gaussian, exponential, Beta and Laplace distributions.
- Joint, marginal and conditional distributions describe discriminative and generative models.
- The output distribution you assume determines your loss function.
:::

=== POST ===
slug: bayes-theorem-explained
title: Bayes' Theorem: Updating Beliefs with Evidence
category: math
level: Beginner
tags: probability, bayes theorem, base rate, prior, posterior
summary: Bayes' theorem is the mathematical rule for learning from evidence. We derive it, work through the famous medical-test example, and see how it underlies Naive Bayes, Bayesian inference and spam filters.
---
If a test for a rare disease is 99% accurate and you test positive, how worried should you be? Most people — including, in published studies, many doctors — answer "99%". The correct answer is often *much* lower. The tool that gets this right is **Bayes' theorem**, and it is the mathematical formalisation of learning from evidence.

## Derivation

From the product rule, $P(A \cap B) = P(A \mid B)P(B) = P(B \mid A)P(A)$. Divide by $P(B)$:

$$
P(A \mid B) = \frac{P(B \mid A)\,P(A)}{P(B)}
$$

In the language of learning, with hypothesis $H$ and evidence (data) $D$:

$$
\underbrace{P(H \mid D)}_{\text{posterior}} = \frac{\overbrace{P(D \mid H)}^{\text{likelihood}}\;\overbrace{P(H)}^{\text{prior}}}{\underbrace{P(D)}_{\text{evidence}}}
$$

- **Prior** $P(H)$: belief before seeing data.
- **Likelihood** $P(D \mid H)$: how probable the data is if $H$ were true.
- **Evidence** $P(D) = \sum_{H'} P(D \mid H')P(H')$: a normalising constant.
- **Posterior** $P(H \mid D)$: updated belief.

In words: **posterior ∝ likelihood × prior**.

## The medical test example

A disease affects 1 in 1,000 people. A test has **sensitivity** 99% ($P(+ \mid \text{disease}) = 0.99$) and **specificity** 95% ($P(- \mid \text{healthy}) = 0.95$, so a 5% false-positive rate). You test positive. What is $P(\text{disease} \mid +)$?

$$
P(\text{disease} \mid +) = \frac{0.99 \times 0.001}{0.99 \times 0.001 + 0.05 \times 0.999} = \frac{0.00099}{0.05094} \approx 0.019
$$

Less than **2%**! Among 1,000 people, about 1 is sick and tests positive, but about 50 healthy people also test positive. A positive result raises the probability from 0.1% to 1.9% — a nineteen-fold increase — but the disease remains unlikely.

:::note
This is the **base-rate fallacy**: ignoring the prior. It matters enormously for AI systems that screen for rare events — fraud, security threats, rare diseases. Even a very accurate detector produces mostly false alarms when the base rate is tiny. This is why precision, not just accuracy, must be reported for imbalanced problems.
:::

A natural-frequencies table makes it vivid (per 100,000 people):

| | Test + | Test − | Total |
|---|---|---|---|
| Disease | 99 | 1 | 100 |
| Healthy | 4,995 | 94,905 | 99,900 |
| Total | 5,094 | 94,906 | 100,000 |

## Sequential updating

Today's posterior becomes tomorrow's prior. If the patient takes a second, independent test and is again positive:

$$
P(\text{disease} \mid +, +) = \frac{0.99 \times 0.019}{0.99 \times 0.019 + 0.05 \times 0.981} \approx 0.28
$$

Evidence accumulates. This is how Bayesian agents learn continuously — the same logic as the filtering algorithms in HMMs.

## Odds form

Bayes' theorem is often cleaner in **odds**:

$$
\underbrace{\frac{P(H \mid D)}{P(\neg H \mid D)}}_{\text{posterior odds}} = \underbrace{\frac{P(D \mid H)}{P(D \mid \neg H)}}_{\text{likelihood ratio}} \times \underbrace{\frac{P(H)}{P(\neg H)}}_{\text{prior odds}}
$$

For our test, the likelihood ratio of a positive result is $0.99/0.05 = 19.8$. Prior odds of 1:999 become posterior odds of about 19.8:999 ≈ 1:50. Taking logarithms turns multiplication into addition: each independent piece of evidence *adds* its log-likelihood ratio. Logistic regression's scores are exactly such log-odds.

## Naive Bayes: Bayes' theorem as a classifier

To classify an email with words $w_1, \dots, w_n$ as spam or ham, apply Bayes' theorem with the **naive** assumption that words are conditionally independent given the class:

$$
P(\text{spam} \mid w_{1:n}) \propto P(\text{spam})\prod_{i=1}^{n} P(w_i \mid \text{spam})
$$

```python
import math
from collections import Counter

train = [("win money now", "spam"), ("cheap money offer", "spam"),
         ("meeting at noon", "ham"), ("lecture notes attached", "ham"),
         ("win a free lecture", "spam"), ("noon meeting moved", "ham")]

counts = {"spam": Counter(), "ham": Counter()}
docs = Counter()
for text, label in train:
    docs[label] += 1
    counts[label].update(text.split())
vocab = set(w for c in counts.values() for w in c)

def log_posterior(text, label, alpha=1.0):               # Laplace smoothing
    total = sum(counts[label].values())
    lp = math.log(docs[label] / sum(docs.values()))
    for w in text.split():
        lp += math.log((counts[label][w] + alpha) / (total + alpha * len(vocab)))
    return lp

msg = "free money meeting"
scores = {c: log_posterior(msg, c) for c in counts}
print(max(scores, key=scores.get), scores)
```

We work with **log probabilities** to avoid underflow and use **Laplace smoothing** so that an unseen word does not force a probability of zero.

## Bayesian inference beyond classification

Bayes' theorem applies to **model parameters** too:

$$
p(\boldsymbol{\theta} \mid \mathcal{D}) \propto p(\mathcal{D} \mid \boldsymbol{\theta})\,p(\boldsymbol{\theta})
$$

The posterior over parameters expresses what we know and how uncertain we remain. Predictions average over it. This is the foundation of Bayesian linear regression, Gaussian processes, Bayesian optimisation for hyperparameter tuning and Thompson sampling for bandits.

:::exercise
1. Recompute the medical example with a prevalence of 10%. How does the posterior change? What does this imply for screening high-risk versus general populations?
2. Derive the odds form of Bayes' theorem from the standard form.
3. Extend the Naive Bayes code to handle a third class ("event announcements") with your own training sentences.
:::

:::takeaway
- Posterior ∝ likelihood × prior; the evidence normalises.
- Ignoring base rates (priors) produces badly wrong conclusions — especially for rare events.
- In odds form, evidence multiplies prior odds by a likelihood ratio; log-odds add.
- Naive Bayes classifiers and full Bayesian inference both follow directly from the theorem.
:::

=== POST ===
slug: expectation-variance-covariance
title: Expectation, Variance, Covariance and Correlation
category: math
level: Beginner
tags: probability, expectation, variance, covariance, correlation, statistics
summary: Summaries of distributions drive everything from loss functions to PCA. We define expectation, variance, covariance and correlation, prove linearity of expectation, and study the covariance matrix.
---
A full probability distribution can be complicated. Often we summarise it with a few numbers: its centre, its spread and how variables move together. These summaries — **expectation, variance, covariance** — are not just descriptive statistics. Loss functions are expectations, the bias–variance trade-off is about variance, and PCA is about covariance.

## Expectation

The **expected value** (mean) of a random variable is its probability-weighted average:

$$
\mathbb{E}[X] = \sum_x x\,p(x) \quad \text{(discrete)}, \qquad \mathbb{E}[X] = \int x\,f(x)\,dx \quad \text{(continuous)}
$$

For a function $g$: $\mathbb{E}[g(X)] = \sum_x g(x)p(x)$ — the **law of the unconscious statistician**.

### Linearity of expectation

$$
\mathbb{E}[aX + bY + c] = a\,\mathbb{E}[X] + b\,\mathbb{E}[Y] + c
$$

This holds **always** — even when $X$ and $Y$ are dependent. It is one of the most powerful tools in probability.

:::example
Shuffle a list of $n$ exam papers and hand them back randomly. How many students expect to get their own paper? Let $X_i = 1$ if student $i$ gets their own paper. $\mathbb{E}[X_i] = 1/n$. By linearity, $\mathbb{E}[\sum X_i] = n \times 1/n = 1$ — regardless of $n$, and despite the $X_i$ being dependent.
:::

In ML, the **risk** of a model is an expectation: $R(f) = \mathbb{E}_{(\mathbf{x}, y)}[\ell(f(\mathbf{x}), y)]$. Training minimises the **empirical** average over the training set as an estimate of this expectation.

## Variance and standard deviation

$$
\text{Var}(X) = \mathbb{E}\big[(X - \mathbb{E}[X])^2\big] = \mathbb{E}[X^2] - \mathbb{E}[X]^2
$$

The standard deviation $\sigma = \sqrt{\text{Var}(X)}$ has the same units as $X$. Properties:

- $\text{Var}(aX + b) = a^2\,\text{Var}(X)$ — shifting does not change spread.
- For **independent** $X, Y$: $\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y)$.
- In general: $\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y) + 2\,\text{Cov}(X, Y)$.

### Why averaging reduces noise

For $n$ independent samples each with variance $\sigma^2$, the sample mean $\bar{X}$ has

$$
\text{Var}(\bar{X}) = \frac{\sigma^2}{n}, \qquad \text{SD}(\bar{X}) = \frac{\sigma}{\sqrt{n}}
$$

This $1/\sqrt{n}$ law explains why larger mini-batches give less noisy gradient estimates (with diminishing returns — quadrupling the batch only halves the noise), why **ensembles** of independent models reduce variance, and why test-set accuracy estimates improve slowly with test-set size.

## Covariance and correlation

**Covariance** measures how two variables vary together:

$$
\text{Cov}(X, Y) = \mathbb{E}\big[(X - \mu_X)(Y - \mu_Y)\big] = \mathbb{E}[XY] - \mu_X\mu_Y
$$

Positive covariance: they tend to be large together. Its magnitude depends on units, so we normalise to get the **Pearson correlation**:

$$
\rho(X, Y) = \frac{\text{Cov}(X, Y)}{\sigma_X\,\sigma_Y} \in [-1, 1]
$$

:::warning
1. **Correlation measures only *linear* association.** If $Y = X^2$ with $X$ symmetric around zero, $\rho = 0$ even though $Y$ is completely determined by $X$.
2. **Zero correlation does not imply independence** (except for jointly Gaussian variables).
3. **Correlation is not causation.** Ice-cream sales and drowning correlate because both rise in summer. Models trained on correlations can fail when the world changes.
:::

## The covariance matrix

For a random vector $\mathbf{x} \in \mathbb{R}^d$ with mean $\boldsymbol{\mu}$:

$$
\boldsymbol{\Sigma} = \mathbb{E}\big[(\mathbf{x} - \boldsymbol{\mu})(\mathbf{x} - \boldsymbol{\mu})^\top\big], \qquad \Sigma_{ij} = \text{Cov}(x_i, x_j)
$$

Properties:

- Symmetric and **positive semi-definite**: $\mathbf{a}^\top\boldsymbol{\Sigma}\mathbf{a} = \text{Var}(\mathbf{a}^\top\mathbf{x}) \ge 0$.
- The variance of any linear projection $\mathbf{a}^\top\mathbf{x}$ is $\mathbf{a}^\top\boldsymbol{\Sigma}\mathbf{a}$ — so the direction of maximum variance is the top eigenvector of $\boldsymbol{\Sigma}$. **That is PCA.**
- Under a linear transform $\mathbf{y} = \mathbf{A}\mathbf{x}$: $\boldsymbol{\Sigma}_y = \mathbf{A}\boldsymbol{\Sigma}_x\mathbf{A}^\top$.

The **sample covariance** from a centred data matrix $\mathbf{X}_c \in \mathbb{R}^{n \times d}$ is $\hat{\boldsymbol{\Sigma}} = \frac{1}{n-1}\mathbf{X}_c^\top\mathbf{X}_c$. The $n - 1$ (Bessel's correction) makes it unbiased.

```python
import numpy as np

rng = np.random.default_rng(0)
n = 5000
study_hours = rng.normal(10, 3, n)
exam_score  = 40 + 4 * study_hours + rng.normal(0, 8, n)
sleep_hours = rng.normal(7, 1, n)
X = np.column_stack([study_hours, exam_score, sleep_hours])

print("means:", X.mean(axis=0).round(2))
print("covariance:\n", np.cov(X, rowvar=False).round(2))
print("correlation:\n", np.corrcoef(X, rowvar=False).round(2))

x = rng.normal(size=n); y = x**2
print("corr(x, x^2):", np.corrcoef(x, y)[0, 1].round(3))   # ~0 despite dependence
```

## Higher moments

- **Skewness** $\mathbb{E}[(X - \mu)^3]/\sigma^3$ measures asymmetry (incomes are right-skewed).
- **Kurtosis** $\mathbb{E}[(X - \mu)^4]/\sigma^4$ measures tail heaviness. Heavy-tailed data (financial returns, network traffic) produce outliers that break methods assuming Gaussian noise.

:::exercise
1. Prove $\text{Var}(X) = \mathbb{E}[X^2] - \mathbb{E}[X]^2$.
2. Show that $\text{Var}(aX + bY) = a^2\text{Var}(X) + b^2\text{Var}(Y) + 2ab\,\text{Cov}(X, Y)$.
3. Simulate: estimate accuracy of a 90%-accurate classifier with test sets of size 100, 1,000 and 10,000, repeated 1,000 times each. Plot the spread and compare with $\sqrt{p(1-p)/n}$.
:::

:::takeaway
- Expectation is linear — always, even for dependent variables.
- Variance of an average shrinks as $\sigma^2/n$: the basis of mini-batching, ensembling and evaluation error bars.
- Correlation captures only linear association and is not causation.
- The covariance matrix is symmetric PSD; its eigenvectors are the principal components.
:::

=== POST ===
slug: gaussian-distribution-deep-dive
title: The Gaussian Distribution: Why It Is Everywhere
category: math
level: Intermediate
tags: probability, gaussian, normal distribution, central limit theorem, multivariate
summary: The bell curve appears in noise models, weight initialisation, VAEs and diffusion models. We study univariate and multivariate Gaussians, the central limit theorem, and the closure properties that make Gaussians so convenient.
---
If one distribution deserves a full lecture, it is the **Gaussian** (normal) distribution. It models measurement noise, underlies least squares, initialises neural network weights, defines the latent space of VAEs and drives the noise process of diffusion models. Understanding *why* it appears so often, and how to manipulate it, is essential.

## The univariate Gaussian

$$
\mathcal{N}(x \mid \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)
$$

- Mean $\mu$, variance $\sigma^2$.
- About 68% of the mass lies within $\mu \pm \sigma$, 95% within $\mu \pm 1.96\sigma$, and 99.7% within $\mu \pm 3\sigma$.
- **Standardisation**: if $X \sim \mathcal{N}(\mu, \sigma^2)$ then $Z = (X - \mu)/\sigma \sim \mathcal{N}(0, 1)$.

The log-density is a **quadratic** in $x$:

$$
\ln \mathcal{N}(x \mid \mu, \sigma^2) = -\frac{(x - \mu)^2}{2\sigma^2} - \frac{1}{2}\ln(2\pi\sigma^2)
$$

This is why maximising Gaussian likelihood is equivalent to minimising **squared error**.

## Why Gaussians are everywhere

### 1. The Central Limit Theorem
If $X_1, \dots, X_n$ are independent and identically distributed with mean $\mu$ and finite variance $\sigma^2$, then

$$
\sqrt{n}\,\frac{\bar{X}_n - \mu}{\sigma} \xrightarrow{d} \mathcal{N}(0, 1)
$$

Sums of many small independent effects are approximately Gaussian, whatever their individual distributions. Measurement errors, heights and many aggregate quantities behave this way.

### 2. Maximum entropy
Among all distributions with a given mean and variance, the Gaussian has the **highest entropy** — it makes the fewest additional assumptions. Choosing a Gaussian is the most "honest" choice when you only know the first two moments.

### 3. Mathematical convenience
Gaussians are closed under linear transformations, marginalisation, conditioning and products (up to normalisation). Everything stays Gaussian, and everything has a closed form.

```python
import numpy as np

rng = np.random.default_rng(0)
# CLT demo: averages of skewed exponential variables become Gaussian
for n in [1, 2, 10, 50]:
    means = rng.exponential(1.0, size=(100_000, n)).mean(axis=1)
    z = (means - 1.0) / (1.0 / np.sqrt(n))
    skew = np.mean(z**3)
    print(f"n={n:>2}: skewness of standardised mean = {skew:.3f}")   # -> 0 as n grows
```

## The multivariate Gaussian

For $\mathbf{x} \in \mathbb{R}^d$ with mean $\boldsymbol{\mu}$ and covariance $\boldsymbol{\Sigma}$ (positive definite):

$$
\mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}, \boldsymbol{\Sigma}) = \frac{1}{(2\pi)^{d/2}|\boldsymbol{\Sigma}|^{1/2}}\exp\left(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^\top\boldsymbol{\Sigma}^{-1}(\mathbf{x} - \boldsymbol{\mu})\right)
$$

The quantity in the exponent is the squared **Mahalanobis distance**. Contours of constant density are **ellipsoids** whose axes are the eigenvectors of $\boldsymbol{\Sigma}$, with lengths proportional to $\sqrt{\lambda_i}$.

- $\boldsymbol{\Sigma} = \sigma^2\mathbf{I}$: spherical contours (isotropic).
- Diagonal $\boldsymbol{\Sigma}$: axis-aligned ellipses (independent coordinates).
- Full $\boldsymbol{\Sigma}$: rotated ellipses (correlated coordinates).

## Closure properties (memorise these)

**Linear transformation.** If $\mathbf{x} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ then $\mathbf{A}\mathbf{x} + \mathbf{b} \sim \mathcal{N}(\mathbf{A}\boldsymbol{\mu} + \mathbf{b}, \mathbf{A}\boldsymbol{\Sigma}\mathbf{A}^\top)$.

**Sampling via the reparameterisation.** To sample from $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$, factor $\boldsymbol{\Sigma} = \mathbf{L}\mathbf{L}^\top$ (Cholesky), draw $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ and set $\mathbf{x} = \boldsymbol{\mu} + \mathbf{L}\boldsymbol{\epsilon}$. This **reparameterisation trick** is what lets VAEs backpropagate through random sampling.

**Marginals and conditionals.** Partition $\mathbf{x} = (\mathbf{x}_a, \mathbf{x}_b)$. The marginal $p(\mathbf{x}_a)$ is Gaussian with mean $\boldsymbol{\mu}_a$ and covariance $\boldsymbol{\Sigma}_{aa}$. The conditional is also Gaussian:

$$
\boldsymbol{\mu}_{a \mid b} = \boldsymbol{\mu}_a + \boldsymbol{\Sigma}_{ab}\boldsymbol{\Sigma}_{bb}^{-1}(\mathbf{x}_b - \boldsymbol{\mu}_b), \qquad \boldsymbol{\Sigma}_{a \mid b} = \boldsymbol{\Sigma}_{aa} - \boldsymbol{\Sigma}_{ab}\boldsymbol{\Sigma}_{bb}^{-1}\boldsymbol{\Sigma}_{ba}
$$

These formulas are the engine of **Gaussian process regression** and the **Kalman filter**. Notice that the conditional covariance never exceeds the marginal: observing $\mathbf{x}_b$ can only reduce uncertainty about $\mathbf{x}_a$.

**Sums of independent Gaussians.** $\mathcal{N}(\mu_1, \sigma_1^2) + \mathcal{N}(\mu_2, \sigma_2^2) = \mathcal{N}(\mu_1 + \mu_2, \sigma_1^2 + \sigma_2^2)$. Diffusion models exploit this: adding Gaussian noise step after step still yields a Gaussian with a closed-form variance, so we can jump to any noise level in one step.

```python
import numpy as np

mu = np.array([1.0, 2.0])
Sigma = np.array([[2.0, 1.2], [1.2, 1.0]])
L = np.linalg.cholesky(Sigma)
eps = np.random.default_rng(0).normal(size=(100_000, 2))
x = mu + eps @ L.T                                # reparameterised samples
print("sample mean:", x.mean(0).round(3))
print("sample cov:\n", np.cov(x, rowvar=False).round(3))
```

## Where you will meet the Gaussian in this course

- **Linear regression**: Gaussian noise ⇒ MSE loss.
- **Weight initialisation**: Xavier/He schemes draw weights from Gaussians with carefully chosen variances.
- **Gaussian Mixture Models**: clusters as Gaussians, fitted by EM.
- **VAEs**: Gaussian prior and posterior in latent space; KL divergence between Gaussians in closed form.
- **Diffusion models**: forward process adds Gaussian noise; the model learns to remove it.
- **Bayesian optimisation**: Gaussian processes model unknown functions.

:::warning
Real data is often *not* Gaussian: incomes are skewed, sensor failures create outliers, financial returns have heavy tails. Using Gaussian assumptions on heavy-tailed data leads to underestimated risks and outlier-dominated fits. Plot your data and consider robust alternatives (Laplace, Student-t, Huber loss).
:::

:::exercise
1. Show that the maximum likelihood estimates for a univariate Gaussian are the sample mean and the (biased) sample variance.
2. For $\boldsymbol{\Sigma} = \begin{bmatrix} 2 & 1.2 \\ 1.2 & 1 \end{bmatrix}$ compute the conditional distribution of $x_1$ given $x_2 = 3$ with $\boldsymbol{\mu} = (1, 2)$.
3. Plot density contours of 2-D Gaussians with isotropic, diagonal and full covariance matrices.
:::

:::takeaway
- The Gaussian's log-density is quadratic ⇒ Gaussian likelihood ↔ squared error.
- It arises from the CLT and is the maximum-entropy distribution for fixed mean and variance.
- Gaussians are closed under linear maps, marginals, conditionals and sums.
- The reparameterisation $\mathbf{x} = \boldsymbol{\mu} + \mathbf{L}\boldsymbol{\epsilon}$ powers VAEs; closed-form noise sums power diffusion.
:::

=== POST ===
slug: maximum-likelihood-estimation
title: Maximum Likelihood Estimation: How Models Learn from Data
category: math
level: Intermediate
tags: statistics, mle, likelihood, loss functions, estimation
summary: Most loss functions in ML are negative log-likelihoods in disguise. We define MLE, derive estimators for Bernoulli and Gaussian models, and prove that MSE and cross-entropy arise from maximum likelihood.
---
Why do we train regression models with mean squared error and classifiers with cross-entropy? Are these arbitrary choices? No — both arise from a single principle: **maximum likelihood estimation (MLE)**. Once you understand MLE, loss functions stop being a list to memorise and become consequences of your assumptions about the data.

## The likelihood function

Suppose data $\mathcal{D} = \{x_1, \dots, x_n\}$ are drawn independently from a model $p(x \mid \boldsymbol{\theta})$ with unknown parameters $\boldsymbol{\theta}$. The **likelihood** is the probability of the observed data viewed as a function of the parameters:

$$
\mathcal{L}(\boldsymbol{\theta}) = p(\mathcal{D} \mid \boldsymbol{\theta}) = \prod_{i=1}^{n} p(x_i \mid \boldsymbol{\theta})
$$

The **maximum likelihood estimate** is

$$
\hat{\boldsymbol{\theta}}_{\text{MLE}} = \arg\max_{\boldsymbol{\theta}} \prod_{i=1}^{n} p(x_i \mid \boldsymbol{\theta}) = \arg\min_{\boldsymbol{\theta}} \left[ -\sum_{i=1}^{n} \ln p(x_i \mid \boldsymbol{\theta}) \right]
$$

We take logarithms because (1) products of many small numbers underflow, (2) sums are easier to differentiate, and (3) the log is monotonic, so the maximiser is unchanged. The quantity in brackets is the **negative log-likelihood (NLL)** — the loss function.

:::warning
The likelihood is **not** a probability distribution over $\boldsymbol{\theta}$. It need not integrate to one over parameters. "The likelihood of $\theta = 0.7$ is higher than of $\theta = 0.5$" is correct; "the probability that $\theta = 0.7$" requires a prior and Bayes' theorem.
:::

## Example 1: coin flips (Bernoulli)

Observing $k$ heads in $n$ flips, with $p(x \mid \theta) = \theta^x(1-\theta)^{1-x}$:

$$
\ln\mathcal{L}(\theta) = k\ln\theta + (n - k)\ln(1 - \theta)
$$

Setting the derivative to zero: $\frac{k}{\theta} - \frac{n - k}{1 - \theta} = 0 \Rightarrow \hat{\theta} = \frac{k}{n}$. The MLE is the observed frequency — reassuringly intuitive.

## Example 2: Gaussian

For $x_i \sim \mathcal{N}(\mu, \sigma^2)$:

$$
\ln\mathcal{L} = -\frac{n}{2}\ln(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_i(x_i - \mu)^2
$$

giving $\hat\mu = \frac{1}{n}\sum_i x_i$ and $\hat\sigma^2 = \frac{1}{n}\sum_i(x_i - \hat\mu)^2$. Note the MLE variance divides by $n$, not $n - 1$: it is slightly **biased**, underestimating variance for small samples.

## From MLE to loss functions

### Regression → Mean Squared Error
Model the target as $y = f_{\boldsymbol{\theta}}(\mathbf{x}) + \epsilon$ with $\epsilon \sim \mathcal{N}(0, \sigma^2)$. Then

$$
-\ln p(y \mid \mathbf{x}, \boldsymbol{\theta}) = \frac{1}{2\sigma^2}\big(y - f_{\boldsymbol{\theta}}(\mathbf{x})\big)^2 + \text{const}
$$

Summing over data, **minimising NLL = minimising squared error**. MSE is the loss that corresponds to assuming Gaussian noise. Assume Laplace noise instead and you get **mean absolute error**.

### Classification → Cross-Entropy
Model $y \in \{0, 1\}$ as Bernoulli with probability $\hat{p} = \sigma(f_{\boldsymbol{\theta}}(\mathbf{x}))$:

$$
-\ln p(y \mid \mathbf{x}, \boldsymbol{\theta}) = -\big[y\ln\hat{p} + (1 - y)\ln(1 - \hat{p})\big]
$$

That is **binary cross-entropy**. For $K$ classes with a categorical model and softmax outputs, the NLL is **categorical cross-entropy** $-\ln\hat{p}_{y}$. Language models are trained by exactly this loss on next-token prediction.

| Assumed output distribution | Negative log-likelihood = loss |
|---|---|
| Gaussian (fixed variance) | Mean squared error |
| Laplace | Mean absolute error |
| Bernoulli | Binary cross-entropy |
| Categorical | Softmax cross-entropy |
| Poisson | Poisson deviance |

## MLE with gradient descent

When no closed form exists (logistic regression, neural networks), we minimise the NLL numerically:

```python
import numpy as np

rng = np.random.default_rng(0)
n = 2000
X = np.column_stack([np.ones(n), rng.normal(size=(n, 2))])
true_w = np.array([-0.5, 2.0, -1.0])
y = rng.random(n) < 1 / (1 + np.exp(-X @ true_w))        # Bernoulli labels

w = np.zeros(3)
for step in range(2000):
    p = 1 / (1 + np.exp(-X @ w))
    nll = -np.mean(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))
    grad = X.T @ (p - y) / n
    w -= 0.5 * grad
print("estimated:", w.round(2), " true:", true_w, " final NLL:", round(nll, 4))
```

The estimate recovers the true parameters closely — MLE is **consistent**.

## Properties of MLE

Under regularity conditions, as $n \to \infty$ the MLE is:

- **Consistent** — converges to the true parameter.
- **Asymptotically normal** — its sampling distribution approaches a Gaussian with covariance given by the inverse **Fisher information**.
- **Asymptotically efficient** — achieves the lowest possible variance (the Cramér–Rao bound).
- **Invariant** — the MLE of $g(\theta)$ is $g(\hat\theta)$.

## The weakness of MLE: overfitting

With little data, MLE trusts the data completely. Flip a coin three times, get three heads, and MLE says $\hat\theta = 1$: tails are impossible. With a flexible model, MLE fits noise. The remedies are **regularisation** and, equivalently, **priors** — which leads to MAP estimation in the next lecture.

:::note
MLE also connects to information theory: minimising average NLL is equivalent to minimising the **KL divergence** from the empirical data distribution to the model. We will make this precise in the information theory lecture.
:::

:::exercise
1. Derive the MLE of the rate $\lambda$ of a Poisson distribution from counts $x_1, \dots, x_n$.
2. Show that assuming Laplace noise gives mean absolute error as the NLL.
3. Modify the code to estimate a linear regression by maximising Gaussian likelihood; compare with `np.linalg.lstsq`.
:::

:::takeaway
- MLE chooses parameters that make the observed data most probable.
- Minimise the negative log-likelihood; it is the loss function.
- Gaussian noise ⇒ MSE; Bernoulli/categorical outputs ⇒ cross-entropy.
- MLE is consistent and efficient asymptotically but overfits small data — motivating priors and regularisation.
:::

=== POST ===
slug: map-estimation-and-priors
title: MAP Estimation, Priors and the Bayesian View of Regularisation
category: math
level: Intermediate
tags: statistics, map, bayesian, priors, regularization, conjugate
summary: Adding a prior to maximum likelihood gives MAP estimation — and reveals that L2 and L1 regularisation are Gaussian and Laplace priors. We also meet conjugate priors and full Bayesian inference.
---
Maximum likelihood trusts the data completely. With three coin flips all heads, MLE concludes the coin can never land tails. A sensible person would say: "Probably biased towards heads, but I've seen only three flips." That common-sense caution is a **prior**, and combining it with the likelihood gives **Maximum A Posteriori (MAP)** estimation. Along the way we will discover that regularisation is secretly Bayesian.

## From MLE to MAP

Bayes' theorem for parameters:

$$
p(\boldsymbol{\theta} \mid \mathcal{D}) = \frac{p(\mathcal{D} \mid \boldsymbol{\theta})\,p(\boldsymbol{\theta})}{p(\mathcal{D})}
$$

MAP chooses the mode of the posterior:

$$
\hat{\boldsymbol{\theta}}_{\text{MAP}} = \arg\max_{\boldsymbol{\theta}}\; p(\mathcal{D} \mid \boldsymbol{\theta})\,p(\boldsymbol{\theta}) = \arg\min_{\boldsymbol{\theta}}\; \Big[\underbrace{-\ln p(\mathcal{D} \mid \boldsymbol{\theta})}_{\text{data fit (NLL)}} \; \underbrace{- \ln p(\boldsymbol{\theta})}_{\text{penalty}}\Big]
$$

The negative log-prior acts as a **regulariser**.

## Regularisation is a prior

### Gaussian prior → L2 (ridge, weight decay)
Place an independent Gaussian prior on each weight: $w_j \sim \mathcal{N}(0, \tau^2)$. Then

$$
-\ln p(\mathbf{w}) = \frac{1}{2\tau^2}\|\mathbf{w}\|_2^2 + \text{const}
$$

So MAP with a Gaussian prior equals **L2-regularised** maximum likelihood with $\lambda = \sigma^2/\tau^2$ in linear regression. A small prior variance $\tau^2$ (strong belief that weights are near zero) means strong regularisation.

### Laplace prior → L1 (lasso)
With $w_j \sim \text{Laplace}(0, b)$: $-\ln p(\mathbf{w}) = \frac{1}{b}\|\mathbf{w}\|_1 + \text{const}$ — the **lasso** penalty. The Laplace density has a sharp peak at zero, encouraging many weights to be exactly zero.

:::note
This correspondence gives you a principled way to think about regularisation strength: it encodes how strongly you believe weights should be small *before* seeing data. It also explains why **weight decay** in neural networks is often described as a Gaussian prior on weights (strictly true for plain SGD; with adaptive optimisers like Adam, decoupled weight decay — AdamW — behaves differently).
:::

## Conjugate priors

A prior is **conjugate** to a likelihood if the posterior is in the same family as the prior. Then Bayesian updating is simple arithmetic.

**Beta–Bernoulli.** Prior $\theta \sim \text{Beta}(\alpha, \beta)$. After $k$ heads and $n - k$ tails:

$$
\theta \mid \mathcal{D} \sim \text{Beta}(\alpha + k,\; \beta + n - k)
$$

The prior parameters behave like **pseudo-counts**: $\alpha - 1$ imaginary heads and $\beta - 1$ imaginary tails. The MAP estimate is

$$
\hat\theta_{\text{MAP}} = \frac{k + \alpha - 1}{n + \alpha + \beta - 2}
$$

With $\alpha = \beta = 2$ and three heads out of three, $\hat\theta_{\text{MAP}} = 4/5 = 0.8$ — confident but not certain. With $\alpha = \beta = 1$ (uniform prior) MAP equals MLE. Laplace smoothing in Naive Bayes is exactly this idea.

Other conjugate pairs: Gamma–Poisson, Dirichlet–Categorical, and Gaussian–Gaussian (for the mean with known variance).

```python
import numpy as np
from scipy import stats

k, n = 3, 3                     # three heads in three flips
for a, b in [(1, 1), (2, 2), (10, 10)]:
    post = stats.beta(a + k, b + n - k)
    map_est = (k + a - 1) / (n + a + b - 2)
    lo, hi = post.ppf([0.025, 0.975])
    print(f"prior Beta({a},{b}): MAP={map_est:.2f}, posterior mean={post.mean():.2f}, "
          f"95% credible interval=({lo:.2f}, {hi:.2f})")
```

## MAP versus full Bayesian inference

MAP returns a single point — the posterior mode. **Full Bayesian inference** keeps the whole posterior and makes predictions by averaging over it:

$$
p(y^* \mid \mathbf{x}^*, \mathcal{D}) = \int p(y^* \mid \mathbf{x}^*, \boldsymbol{\theta})\,p(\boldsymbol{\theta} \mid \mathcal{D})\,d\boldsymbol{\theta}
$$

This **posterior predictive** distribution accounts for parameter uncertainty — predictions become less confident far from the training data. The integral is usually intractable, so we approximate it using:

- **Conjugate** closed forms (Bayesian linear regression, Gaussian processes);
- **MCMC** sampling (Metropolis–Hastings, Hamiltonian Monte Carlo);
- **Variational inference** — optimise a simple distribution to approximate the posterior (the basis of VAEs);
- **Practical deep-learning approximations** — deep ensembles, MC dropout, Laplace approximations.

| | MLE | MAP | Full Bayes |
|---|---|---|---|
| Uses prior | No | Yes | Yes |
| Output | Point estimate | Point estimate | Distribution |
| Overfitting on small data | High | Reduced | Reduced |
| Uncertainty estimates | No | No | Yes |
| Cost | Low | Low | High |

## Caveats of MAP

- The mode can be unrepresentative of the posterior (e.g. a sharp spike with little mass).
- MAP is **not invariant to reparameterisation**: the mode of $p(\theta)$ does not map to the mode of $p(g(\theta))$, because densities pick up a Jacobian factor.
- As $n \to \infty$ the likelihood dominates the prior and MAP → MLE. Priors matter most when data is scarce — exactly when you need them.

:::tip
In practice: use MAP (i.e. regularised training) as the default, tune regularisation strength on validation data, and use ensembles or Bayesian approximations when you need calibrated uncertainty — for example in medical or humanitarian decision support, where knowing "the model is unsure" can be as valuable as the prediction itself.
:::

:::exercise
1. Derive the MAP estimate for linear regression with a Gaussian prior and show it equals the ridge solution $(\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^\top\mathbf{y}$.
2. For the Beta–Bernoulli model, show that the posterior mean is $(k + \alpha)/(n + \alpha + \beta)$.
3. Simulate 10 coin flips from a coin with $\theta = 0.7$, and plot the posterior after each flip for a Beta(2, 2) prior.
:::

:::takeaway
- MAP = maximise likelihood × prior = minimise NLL + negative log-prior.
- Gaussian prior ↔ L2 regularisation; Laplace prior ↔ L1 regularisation.
- Conjugate priors give closed-form posteriors; prior parameters act as pseudo-counts.
- Full Bayesian inference averages over the posterior and yields uncertainty, at higher computational cost.
:::

=== POST ===
slug: information-theory-entropy-kl
title: Information Theory for ML: Entropy, Cross-Entropy and KL Divergence
category: math
level: Intermediate
tags: information theory, entropy, cross-entropy, kl divergence, mutual information
summary: Shannon's theory of information explains our loss functions. We derive entropy, cross-entropy, KL divergence and mutual information, and show why minimising cross-entropy is maximum likelihood.
---
In 1948 Claude Shannon asked how to measure information, and founded a field that now sits at the core of machine learning. The loss you minimise when training a classifier or a language model is called *cross-entropy* for a reason. The regulariser in a VAE is a *KL divergence*. Decision trees split on *information gain*. Today we build these concepts from first principles.

## Information content

How surprising is an event? An event with probability 1 carries no information; a rare event carries a lot. Shannon defined the **information content** (surprisal) of an outcome $x$ as

$$
I(x) = -\log_2 p(x) \quad \text{bits}
$$

(Using natural logs gives **nats**.) A fair coin flip yields 1 bit. An event with probability 1/1024 yields 10 bits. Information from independent events adds, because probabilities multiply and logs turn products into sums.

## Entropy

**Entropy** is the expected surprisal — the average uncertainty of a distribution:

$$
H(p) = -\sum_x p(x)\log p(x)
$$

- Maximum for the uniform distribution: $H = \log K$ for $K$ outcomes.
- Zero for a deterministic distribution.
- For a Bernoulli($q$): $H = -q\log q - (1-q)\log(1-q)$, maximised at $q = 0.5$.

Shannon's **source coding theorem** gives entropy an operational meaning: it is the minimum average number of bits per symbol needed to encode messages from $p$ losslessly. English text has a much lower entropy per character than $\log_2 26 \approx 4.7$ bits, because letters are predictable — which is why text compresses well and why language models can predict it.

## Cross-entropy

Suppose data comes from a true distribution $p$ but we encode it with a code optimised for a model $q$. The average code length is the **cross-entropy**:

$$
H(p, q) = -\sum_x p(x)\log q(x)
$$

Always $H(p, q) \ge H(p)$: using the wrong model costs extra bits.

### Cross-entropy as a loss
In classification, the "true distribution" for one example is the one-hot label $\mathbf{y}$, and the model outputs $\hat{\mathbf{p}}$:

$$
H(\mathbf{y}, \hat{\mathbf{p}}) = -\sum_k y_k \log\hat{p}_k = -\log\hat{p}_{\text{true class}}
$$

Averaged over the dataset, this is exactly the **negative log-likelihood** — so minimising cross-entropy *is* maximum likelihood.

:::note
For language models, the average cross-entropy per token relates directly to **perplexity**: $\text{PPL} = \exp(H)$ with $H$ in nats. A perplexity of 20 means the model is, on average, as uncertain as if choosing uniformly among 20 tokens.
:::

## KL divergence

The **Kullback–Leibler divergence** measures the extra cost of using $q$ instead of $p$:

$$
D_{\text{KL}}(p \,\|\, q) = \sum_x p(x)\log\frac{p(x)}{q(x)} = H(p, q) - H(p)
$$

Properties:

- $D_{\text{KL}}(p\|q) \ge 0$, with equality iff $p = q$ (Gibbs' inequality, from Jensen's inequality).
- **Not symmetric**: $D_{\text{KL}}(p\|q) \ne D_{\text{KL}}(q\|p)$ in general, so it is not a true distance.

Since $H(p)$ is fixed by the data, **minimising cross-entropy equals minimising KL divergence** from the data distribution to the model.

### Forward vs reverse KL
- **Forward KL** $D_{\text{KL}}(p\|q)$ (used in MLE) heavily penalises $q(x) \approx 0$ where $p(x) > 0$. The model must cover all data modes — **mode-covering**, producing blurry averages.
- **Reverse KL** $D_{\text{KL}}(q\|p)$ (used in variational inference) penalises $q$ putting mass where $p$ has none — **mode-seeking**, locking onto one mode.

This distinction explains qualitative behaviour of generative models and variational approximations.

### KL between Gaussians
A closed form used in every VAE:

$$
D_{\text{KL}}\big(\mathcal{N}(\mu, \sigma^2)\,\|\,\mathcal{N}(0, 1)\big) = \frac{1}{2}\left(\mu^2 + \sigma^2 - \ln\sigma^2 - 1\right)
$$

## Mutual information

**Mutual information** measures how much knowing one variable reduces uncertainty about another:

$$
I(X; Y) = H(X) - H(X \mid Y) = D_{\text{KL}}\big(p(x, y)\,\|\,p(x)p(y)\big)
$$

It is zero iff $X$ and $Y$ are independent and, unlike correlation, captures **non-linear** dependence. Uses include feature selection, the **information gain** criterion for decision-tree splits, contrastive representation learning (InfoNCE bounds mutual information), and the information bottleneck theory of deep learning.

## Computing it

```python
import numpy as np

def entropy(p):
    p = np.asarray(p, float); p = p[p > 0]
    return -(p * np.log2(p)).sum()

def cross_entropy(p, q):
    p, q = np.asarray(p, float), np.asarray(q, float)
    return -(p * np.log2(q + 1e-12)).sum()

def kl(p, q):
    return cross_entropy(p, q) - entropy(p)

p = [0.7, 0.2, 0.1]
q = [0.5, 0.3, 0.2]
print("H(p) =", round(entropy(p), 3), "bits")
print("H(p,q) =", round(cross_entropy(p, q), 3), " KL(p||q) =", round(kl(p, q), 3), " KL(q||p) =", round(kl(q, p), 3))

# Information gain of a split: parent labels vs children
parent = [0.5, 0.5]
left, right = [0.9, 0.1], [0.2, 0.8]          # each child holds half the data
gain = entropy(parent) - 0.5 * entropy(left) - 0.5 * entropy(right)
print("information gain:", round(gain, 3), "bits")
```

:::tip
In code, never compute `log(softmax(z))` in two steps — it underflows for very negative logits. Use `log_softmax` or a fused cross-entropy function (e.g. `torch.nn.functional.cross_entropy`), which applies the log-sum-exp trick internally.
:::

:::exercise
1. Compute the entropy of a fair six-sided die and of a die that lands on 6 half the time.
2. Show that $D_{\text{KL}}(p\|q) \ge 0$ using Jensen's inequality.
3. Derive the closed-form KL between two univariate Gaussians $\mathcal{N}(\mu_1, \sigma_1^2)$ and $\mathcal{N}(\mu_2, \sigma_2^2)$.
:::

:::takeaway
- Entropy = expected surprisal = minimum average code length.
- Cross-entropy with one-hot labels = negative log-likelihood; minimising it = maximum likelihood.
- KL divergence = cross-entropy − entropy; asymmetric; forward KL covers modes, reverse KL seeks modes.
- Mutual information measures any (including non-linear) dependence; information gain drives decision trees.
:::
