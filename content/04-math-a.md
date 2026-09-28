=== POST ===
slug: why-mathematics-matters-for-ml
title: Why Mathematics Matters for Machine Learning
category: math
level: Beginner
tags: mathematics, linear algebra, calculus, probability, study guide
summary: You can call library functions without mathematics — until something breaks. We explain which branches of mathematics ML uses, why, and how to learn them efficiently.
---
Every year a student asks me: "Sir, the libraries do everything. Why must I learn mathematics?" It is a fair question, and it deserves an honest answer. You can train a model with five lines of code. But the moment your model does not converge, overfits, produces NaNs, or behaves unfairly, those five lines offer no help. Mathematics is the language in which the *reasons* are written.

## The four pillars

Machine learning rests on four mathematical pillars:

| Pillar | What it gives ML | Where you will meet it |
|---|---|---|
| **Linear algebra** | Language for data and models: vectors, matrices, transformations | Every model; embeddings; PCA; attention |
| **Calculus** | How outputs change with inputs; gradients | Training by gradient descent; backpropagation |
| **Probability & statistics** | Modelling uncertainty; learning from samples | Loss functions, Bayesian methods, evaluation |
| **Optimisation** | Finding the best parameters | Every training algorithm |

Information theory — entropy, cross-entropy, KL divergence — ties probability to learning and appears in almost every loss function.

## One equation, all four pillars

Consider logistic regression trained with gradient descent:

$$
\mathbf{w} \leftarrow \mathbf{w} - \eta \, \frac{1}{n} \sum_{i=1}^{n} \big(\sigma(\mathbf{w}^\top \mathbf{x}_i) - y_i\big)\, \mathbf{x}_i
$$

- $\mathbf{w}^\top \mathbf{x}_i$ is a **dot product** — linear algebra.
- $\sigma$ turns a score into a **probability** — probability.
- The term in the sum is the **gradient** of the cross-entropy loss — calculus and information theory.
- The update rule is **gradient descent** — optimisation.

Every idea in this course is some elaboration of this pattern.

## What mathematics lets you do

1. **Debug.** Exploding losses? You will recognise exploding gradients from the chain rule. Model predicts only one class? You will reason about class imbalance and decision thresholds.
2. **Choose.** Should you use MSE or cross-entropy? L1 or L2 regularisation? Mathematics explains the consequences.
3. **Read research.** Papers are written in mathematics. Without it you are limited to blog summaries.
4. **Invent.** New methods come from understanding why old ones work.

## How much do you need?

You do not need to be a mathematician. You need **fluency** in a specific toolkit:

- Linear algebra: vectors, matrices, matrix multiplication, transpose, inverse, rank, eigenvalues, SVD, norms, projections.
- Calculus: derivatives, partial derivatives, the chain rule, gradients, Jacobians, a little Taylor expansion.
- Probability: random variables, distributions (Bernoulli, categorical, Gaussian), expectation, variance, Bayes' rule, independence, maximum likelihood.
- Optimisation: convexity, gradient descent and its variants, constrained optimisation basics.

## Learn mathematics with code

The best way for engineers to learn mathematics is to *compute* it. Verify every identity numerically:

```python
import numpy as np

rng = np.random.default_rng(0)
A = rng.normal(size=(3, 3))
B = rng.normal(size=(3, 3))

# Identity: (AB)^T = B^T A^T
print(np.allclose((A @ B).T, B.T @ A.T))          # True

# Numerical derivative vs analytic derivative of f(x) = x^3
f = lambda x: x**3
x, h = 2.0, 1e-5
numeric = (f(x + h) - f(x - h)) / (2 * h)
print(numeric, 3 * x**2)                           # ~12.0, 12.0
```

:::tip
Adopt the **"paper, then code"** habit. Derive a result by hand, then check it numerically. When the two disagree, you have found either a mistake in your derivation or a bug in your code — both are valuable discoveries.
:::

## Notation used in this course

- Scalars: lowercase italics $x, \eta$.
- Vectors: bold lowercase $\mathbf{x}$ (column vectors by default).
- Matrices: bold uppercase $\mathbf{X}, \mathbf{W}$.
- $\mathbf{x}^\top$: transpose. $\|\mathbf{x}\|$: Euclidean norm.
- $\nabla_{\mathbf{w}} L$: gradient of $L$ with respect to $\mathbf{w}$.
- $\mathbb{E}[X]$: expectation. $p(x)$: probability density or mass.
- Dataset: $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^n$ with $n$ examples of dimension $d$.

:::exercise
1. Verify numerically that matrix multiplication is not commutative: find $\mathbf{A}, \mathbf{B}$ with $\mathbf{AB} \ne \mathbf{BA}$.
2. Compute the numerical derivative of $\sin(x)$ at $x = 1$ and compare with $\cos(1)$. Try $h = 10^{-2}, 10^{-5}, 10^{-12}$. What happens with the smallest $h$ and why?
3. Identify the four pillars in the update rule of linear regression with squared loss.
:::

:::takeaway
- ML rests on linear algebra, calculus, probability/statistics and optimisation.
- Mathematics lets you debug, choose methods, read papers and invent.
- Aim for working fluency, reinforced by verifying results in code.
:::

=== POST ===
slug: vectors-and-vector-spaces
title: Vectors and Vector Spaces: The Language of Data
category: math
level: Beginner
tags: linear algebra, vectors, basis, span, linear independence
summary: Every data point, word and image becomes a vector. We define vector spaces, linear combinations, span, independence, basis and dimension — and see why embeddings live in them.
---
In machine learning, everything becomes a vector. A house becomes (area, rooms, age). A grayscale image of $28 \times 28$ pixels becomes a vector in $\mathbb{R}^{784}$. A word becomes a 768-dimensional embedding. Today we learn the algebra of these objects rigorously, because the geometry of vector spaces is the geometry of data.

## Vectors

A vector $\mathbf{x} \in \mathbb{R}^d$ is an ordered list of $d$ real numbers:

$$
\mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_d \end{bmatrix}
$$

We can view it in two ways: as a **point** in $d$-dimensional space, or as an **arrow** (a displacement) from the origin. Both views are useful.

Two basic operations:

- **Addition**: $(\mathbf{x} + \mathbf{y})_i = x_i + y_i$ — place arrows head to tail.
- **Scalar multiplication**: $(c\mathbf{x})_i = c\,x_i$ — stretch or flip the arrow.

## Vector spaces

A **vector space** over $\mathbb{R}$ is a set $V$ with addition and scalar multiplication satisfying eight axioms (associativity, commutativity, zero vector, additive inverses, distributivity, etc.). The essential point: **closed under linear combinations**. $\mathbb{R}^d$ is the standard example, but polynomials, matrices and functions also form vector spaces — which is why techniques like kernel methods can treat functions as vectors.

A **subspace** is a subset that is itself a vector space: it contains the zero vector and is closed under addition and scaling. In $\mathbb{R}^3$, subspaces are the origin, lines through the origin, planes through the origin, and $\mathbb{R}^3$ itself.

## Linear combinations and span

A **linear combination** of vectors $\mathbf{v}_1, \dots, \mathbf{v}_k$ is

$$
c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + \dots + c_k\mathbf{v}_k
$$

The **span** of the vectors is the set of all such combinations. Two non-parallel vectors in $\mathbb{R}^3$ span a plane.

:::note
A linear model's prediction $\hat{y} = \mathbf{w}^\top\mathbf{x}$ is a linear combination of features. The output of a linear layer in a neural network, $\mathbf{W}\mathbf{x}$, is a linear combination of the columns of $\mathbf{W}$ weighted by the entries of $\mathbf{x}$. Keep that "column picture" in mind — it makes many results obvious.
:::

## Linear independence

Vectors are **linearly independent** if the only solution to $c_1\mathbf{v}_1 + \dots + c_k\mathbf{v}_k = \mathbf{0}$ is $c_1 = \dots = c_k = 0$. Otherwise one of them can be written as a combination of the others — it is redundant.

In data terms: if one feature equals twice another plus a third (e.g. "total price" = "unit price × quantity" already stored), the feature vectors are dependent. This **multicollinearity** makes linear regression coefficients unstable, a problem we fix with regularisation.

## Basis and dimension

A **basis** of a space is a set of linearly independent vectors that span it. Every vector in the space can be written *uniquely* as a combination of basis vectors; the coefficients are its **coordinates** in that basis. The number of vectors in any basis is the **dimension**.

The **standard basis** of $\mathbb{R}^3$ is $\mathbf{e}_1 = (1,0,0)$, $\mathbf{e}_2 = (0,1,0)$, $\mathbf{e}_3 = (0,0,1)$. But other bases can be far more useful: PCA finds a basis aligned with the directions of greatest variance; the Fourier basis represents signals as frequencies.

```python
import numpy as np

V = np.array([[1, 2, 3],
              [2, 4, 6],      # = 2 * first row -> dependent
              [0, 1, 1]]).T   # columns are our vectors
print("rank:", np.linalg.matrix_rank(V))   # 2 -> the three vectors span only a plane

# Coordinates of x in a new basis B (columns): solve B c = x
B = np.array([[1, 1], [1, -1]]).T
x = np.array([3, 1])
c = np.linalg.solve(B, x)
print("coordinates:", c)                   # [2, 1] because 2*(1,1) + 1*(1,-1) = (3,1)
```

## The geometry of high dimensions

Your intuition, built in 2-D and 3-D, can mislead you in high dimensions:

- Most of the volume of a high-dimensional ball lies near its surface.
- Two random vectors in high dimensions are almost always nearly **orthogonal**.
- Distances between random points concentrate — the nearest and farthest neighbours become almost equally far.

These facts are the **curse of dimensionality**, which we will study in its own lecture. They also explain why embeddings can store so much information: high-dimensional spaces have room for many nearly independent directions.

## Vectors as meaning: embeddings

Word embeddings famously capture relationships as vector arithmetic, for example $\mathbf{v}_{\text{king}} - \mathbf{v}_{\text{man}} + \mathbf{v}_{\text{woman}} \approx \mathbf{v}_{\text{queen}}$. Directions in embedding space can correspond to concepts such as gender, tense or sentiment. This is linear algebra acting as a model of meaning — and a reason that biased training data produces biased directions, which we must detect and address.

:::exercise
1. Are $(1, 2, 0)$, $(0, 1, 1)$ and $(1, 3, 1)$ linearly independent? Justify by hand, then check with `matrix_rank`.
2. Find the coordinates of $(4, 2)$ in the basis $\{(1, 1), (1, -1)\}$.
3. Generate 1,000 pairs of random unit vectors in $\mathbb{R}^{2}$ and $\mathbb{R}^{1000}$. Plot histograms of their dot products. What do you observe?
:::

:::takeaway
- Data points are vectors; vector spaces are closed under linear combinations.
- Span, linear independence, basis and dimension describe the structure of a space.
- Dependent features cause multicollinearity; good bases (PCA, Fourier) reveal structure.
- High-dimensional geometry is counter-intuitive — random vectors are nearly orthogonal.
:::

=== POST ===
slug: matrices-and-linear-transformations
title: Matrices and Linear Transformations
category: math
level: Beginner
tags: linear algebra, matrices, matrix multiplication, rank, inverse
summary: A matrix is not just a table of numbers — it is a function that transforms space. We cover matrix multiplication four ways, rank, inverses, determinants and the fundamental subspaces.
---
Last lecture we treated vectors as data. Today we study the functions that act on them. A **matrix** is the concrete representation of a **linear transformation**, and every layer of a neural network begins with one. Understanding matrices as transformations — rotating, stretching, projecting space — is the single most useful intuition in linear algebra.

## Linear transformations

A function $T: \mathbb{R}^n \to \mathbb{R}^m$ is **linear** if

$$
T(a\mathbf{x} + b\mathbf{y}) = aT(\mathbf{x}) + bT(\mathbf{y})
$$

Every such $T$ can be written as $T(\mathbf{x}) = \mathbf{A}\mathbf{x}$ for a unique $m \times n$ matrix $\mathbf{A}$. The columns of $\mathbf{A}$ are simply where the standard basis vectors land: column $j$ is $T(\mathbf{e}_j)$.

:::example
The matrix $\mathbf{R} = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$ rotates the plane by angle $\theta$. The matrix $\begin{bmatrix} 2 & 0 \\ 0 & 0.5 \end{bmatrix}$ stretches the x-axis by 2 and squashes the y-axis by half. The matrix $\begin{bmatrix} 1 & 0 \\ 0 & 0 \end{bmatrix}$ projects every point onto the x-axis.
:::

## Four ways to see matrix multiplication

For $\mathbf{C} = \mathbf{A}\mathbf{B}$ with $\mathbf{A} \in \mathbb{R}^{m \times k}$, $\mathbf{B} \in \mathbb{R}^{k \times n}$:

1. **Entry view**: $C_{ij} = \sum_{l} A_{il}B_{lj}$ — dot product of row $i$ of $\mathbf{A}$ and column $j$ of $\mathbf{B}$.
2. **Column view**: each column of $\mathbf{C}$ is $\mathbf{A}$ times the corresponding column of $\mathbf{B}$ — a combination of $\mathbf{A}$'s columns.
3. **Row view**: each row of $\mathbf{C}$ is a combination of $\mathbf{B}$'s rows.
4. **Outer-product view**: $\mathbf{C} = \sum_{l} \mathbf{a}_{:l}\,\mathbf{b}_{l:}$ — a sum of rank-one matrices.

The outer-product view is the key to understanding low-rank approximations, SVD and LoRA fine-tuning.

**Composition**: applying $\mathbf{B}$ then $\mathbf{A}$ equals applying $\mathbf{AB}$. This is why a stack of linear layers without non-linearities collapses into a *single* linear layer — the reason neural networks need activation functions.

## Key properties

- Associative: $(\mathbf{AB})\mathbf{C} = \mathbf{A}(\mathbf{BC})$.
- Distributive: $\mathbf{A}(\mathbf{B} + \mathbf{C}) = \mathbf{AB} + \mathbf{AC}$.
- **Not commutative**: generally $\mathbf{AB} \ne \mathbf{BA}$.
- Transpose: $(\mathbf{AB})^\top = \mathbf{B}^\top\mathbf{A}^\top$.

## Rank and the four fundamental subspaces

The **rank** of $\mathbf{A}$ is the dimension of its **column space** — the set of all outputs $\mathbf{A}\mathbf{x}$. Gilbert Strang's four fundamental subspaces of an $m \times n$ matrix of rank $r$:

| Subspace | Lives in | Dimension |
|---|---|---|
| Column space $C(\mathbf{A})$ | $\mathbb{R}^m$ | $r$ |
| Null space $N(\mathbf{A})$: $\mathbf{Ax} = \mathbf{0}$ | $\mathbb{R}^n$ | $n - r$ |
| Row space $C(\mathbf{A}^\top)$ | $\mathbb{R}^n$ | $r$ |
| Left null space $N(\mathbf{A}^\top)$ | $\mathbb{R}^m$ | $m - r$ |

The **rank–nullity theorem**: $\text{rank} + \text{nullity} = n$. The row space and null space are orthogonal complements.

## Inverses and determinants

A square matrix $\mathbf{A}$ is **invertible** if there is $\mathbf{A}^{-1}$ with $\mathbf{A}\mathbf{A}^{-1} = \mathbf{I}$. Equivalent conditions: full rank, trivial null space, non-zero determinant, no zero eigenvalue.

The **determinant** measures how the transformation scales volume: $|\det \mathbf{A}|$ is the volume of the image of the unit cube; the sign says whether orientation flips. $\det(\mathbf{AB}) = \det\mathbf{A}\,\det\mathbf{B}$. Determinants appear in the Gaussian density and in **normalising flows**, where we must track how a transformation changes probability volume.

:::warning
In code, almost **never** compute `inv(A) @ b` to solve $\mathbf{Ax} = \mathbf{b}$. Use `np.linalg.solve(A, b)` (or a least-squares solver). It is faster and far more numerically stable, especially when $\mathbf{A}$ is nearly singular (ill-conditioned).
:::

## Special matrices you will meet constantly

- **Identity** $\mathbf{I}$ and **diagonal** matrices — scale each axis independently.
- **Symmetric** $\mathbf{A} = \mathbf{A}^\top$ — covariance matrices, Hessians, kernel matrices.
- **Orthogonal** $\mathbf{Q}^\top\mathbf{Q} = \mathbf{I}$ — rotations and reflections; preserve lengths and angles.
- **Positive (semi-)definite** $\mathbf{x}^\top\mathbf{A}\mathbf{x} \ge 0$ — covariance matrices; convex quadratic forms.
- **Sparse** matrices — graphs, text term–document matrices.

```python
import numpy as np

theta = np.pi / 4
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])
x = np.array([1.0, 0.0])
print(R @ x)                           # rotated 45 degrees
print(np.allclose(R.T @ R, np.eye(2))) # orthogonal
print(np.linalg.det(R))                # 1.0 -> preserves area

A = np.array([[4., 1.], [2., 3.]])
b = np.array([1., 2.])
print(np.linalg.solve(A, b))           # preferred over inv(A) @ b
```

## Matrices in neural networks

A fully connected layer computes $\mathbf{h} = \phi(\mathbf{W}\mathbf{x} + \mathbf{b})$. For a mini-batch stored as rows of $\mathbf{X} \in \mathbb{R}^{B \times d}$, it is $\mathbf{H} = \phi(\mathbf{X}\mathbf{W}^\top + \mathbf{b})$ — one matrix multiplication for the whole batch. GPUs are, above all, machines for fast matrix multiplication, and that is why deep learning became practical.

:::exercise
1. Write the matrix that reflects points across the line $y = x$. Verify with code.
2. Show that the composition of two rotations by $\alpha$ and $\beta$ is a rotation by $\alpha + \beta$.
3. Find the null space of $\begin{bmatrix} 1 & 2 & 3 \\ 2 & 4 & 6 \end{bmatrix}$ and verify the rank–nullity theorem.
:::

:::takeaway
- Matrices represent linear transformations; columns show where basis vectors go.
- Master four views of multiplication — especially the column and outer-product views.
- Rank, null space and the four fundamental subspaces describe what a matrix can and cannot do.
- Solve systems with `solve`, not explicit inverses.
:::

=== POST ===
slug: eigenvalues-and-eigenvectors
title: Eigenvalues and Eigenvectors: The Natural Axes of a Transformation
category: math
level: Intermediate
tags: linear algebra, eigenvalues, eigenvectors, diagonalization, spectral theorem
summary: Eigenvectors are directions a matrix merely stretches. We derive the characteristic equation, diagonalisation and the spectral theorem, and connect them to PCA, PageRank, Markov chains and training stability.
---
Most vectors change direction when a matrix acts on them. But some special vectors only get stretched or shrunk — they keep their direction. These are **eigenvectors**, and the stretching factors are **eigenvalues**. They reveal the "natural axes" of a transformation and appear throughout machine learning: PCA, spectral clustering, PageRank, the convergence of gradient descent and the stability of recurrent networks.

## Definition

For a square matrix $\mathbf{A} \in \mathbb{R}^{n \times n}$, a non-zero vector $\mathbf{v}$ is an **eigenvector** with **eigenvalue** $\lambda$ if

$$
\mathbf{A}\mathbf{v} = \lambda\mathbf{v}
$$

Rearranging, $(\mathbf{A} - \lambda\mathbf{I})\mathbf{v} = \mathbf{0}$ must have a non-zero solution, so $\mathbf{A} - \lambda\mathbf{I}$ must be singular:

$$
\det(\mathbf{A} - \lambda\mathbf{I}) = 0
$$

This **characteristic equation** is a polynomial of degree $n$ in $\lambda$, so there are $n$ eigenvalues (counted with multiplicity, possibly complex).

:::example
$\mathbf{A} = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}$. Then $\det(\mathbf{A} - \lambda\mathbf{I}) = (2 - \lambda)^2 - 1 = 0$, giving $\lambda_1 = 3$, $\lambda_2 = 1$. For $\lambda = 3$: $\mathbf{v}_1 = (1, 1)/\sqrt{2}$. For $\lambda = 1$: $\mathbf{v}_2 = (1, -1)/\sqrt{2}$. The matrix stretches space by 3 along the diagonal and leaves the anti-diagonal unchanged.
:::

Two useful facts: the **trace** equals the sum of eigenvalues, and the **determinant** equals their product.

## Diagonalisation

If $\mathbf{A}$ has $n$ linearly independent eigenvectors, collect them as columns of $\mathbf{V}$ and the eigenvalues in diagonal $\boldsymbol{\Lambda}$:

$$
\mathbf{A} = \mathbf{V}\boldsymbol{\Lambda}\mathbf{V}^{-1}
$$

Read right to left: change to the eigenvector coordinate system, scale each axis independently, change back. Powers become trivial:

$$
\mathbf{A}^k = \mathbf{V}\boldsymbol{\Lambda}^k\mathbf{V}^{-1}
$$

So the long-term behaviour of repeatedly applying $\mathbf{A}$ is governed by the **largest** eigenvalue in magnitude. If $|\lambda_{\max}| > 1$, repeated application explodes; if all $|\lambda| < 1$, it decays to zero.

:::note
This is exactly the **vanishing/exploding gradient** problem in recurrent neural networks. Backpropagating through $T$ time steps multiplies by (roughly) the same recurrent weight matrix $T$ times. Eigenvalues above 1 make gradients explode; below 1 make them vanish. LSTMs, gradient clipping and careful initialisation all address this.
:::

## The spectral theorem

For **symmetric** real matrices ($\mathbf{A} = \mathbf{A}^\top$), things are especially nice:

1. All eigenvalues are **real**.
2. Eigenvectors for distinct eigenvalues are **orthogonal**.
3. $\mathbf{A}$ can be diagonalised by an **orthogonal** matrix:

$$
\mathbf{A} = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^\top = \sum_{i=1}^n \lambda_i\, \mathbf{q}_i\mathbf{q}_i^\top
$$

Covariance matrices, kernel (Gram) matrices, graph Laplacians and Hessians are all symmetric, so the spectral theorem is everywhere in ML. A symmetric matrix is **positive definite** iff all eigenvalues are positive; **positive semi-definite** iff all are non-negative.

## Applications in ML

### Principal Component Analysis
The eigenvectors of the data covariance matrix $\mathbf{\Sigma}$ are the **principal directions**; the eigenvalues are the variances along them. Keeping the top $k$ eigenvectors gives the best $k$-dimensional linear approximation of the data.

### Optimisation and curvature
Near a minimum, a loss looks like a quadratic with Hessian $\mathbf{H}$. The eigenvalues of $\mathbf{H}$ are curvatures along principal directions. Gradient descent converges only if the learning rate satisfies $\eta < 2/\lambda_{\max}(\mathbf{H})$, and it is slow when the **condition number** $\kappa = \lambda_{\max}/\lambda_{\min}$ is large — long narrow valleys. Momentum, Adam and normalisation layers all help with ill-conditioning.

### PageRank and Markov chains
The stationary distribution of a Markov chain is the eigenvector of the transition matrix with eigenvalue 1. Google's PageRank is exactly this for the web graph.

### Spectral clustering
Eigenvectors of the graph Laplacian with the smallest eigenvalues reveal cluster structure, even for non-convex clusters.

## Computing eigenvectors: power iteration

The simplest algorithm repeatedly multiplies a random vector by $\mathbf{A}$ and normalises. It converges to the dominant eigenvector at a rate governed by $|\lambda_2/\lambda_1|$.

```python
import numpy as np

def power_iteration(A, iters=100):
    v = np.random.default_rng(0).normal(size=A.shape[0])
    for _ in range(iters):
        v = A @ v
        v /= np.linalg.norm(v)
    return v @ A @ v, v             # Rayleigh quotient gives the eigenvalue

A = np.array([[2., 1.], [1., 2.]])
lam, v = power_iteration(A)
print(lam, v)                        # ~3.0, ~[0.707, 0.707]

w, Q = np.linalg.eigh(A)             # eigh: for symmetric matrices (sorted ascending)
print(w, Q)
```

:::tip
Use `np.linalg.eigh` for symmetric matrices — it is faster, more stable and guarantees real eigenvalues and orthonormal eigenvectors. Use `np.linalg.eig` only for general matrices.
:::

:::exercise
1. Find the eigenvalues and eigenvectors of $\begin{bmatrix} 4 & 2 \\ 1 & 3 \end{bmatrix}$ by hand. Verify trace and determinant properties.
2. Show that a rotation matrix by 90° has no real eigenvectors. Interpret geometrically.
3. Run gradient descent on $f(\mathbf{x}) = \frac{1}{2}\mathbf{x}^\top \mathbf{H}\mathbf{x}$ with $\mathbf{H} = \text{diag}(1, 100)$. Find the largest learning rate that converges and compare with $2/\lambda_{\max}$.
:::

:::takeaway
- Eigenvectors keep their direction under $\mathbf{A}$; eigenvalues are the scaling factors.
- Diagonalisation $\mathbf{A} = \mathbf{V}\boldsymbol{\Lambda}\mathbf{V}^{-1}$ makes powers easy; the largest $|\lambda|$ governs growth.
- Symmetric matrices have real eigenvalues and orthonormal eigenvectors (spectral theorem).
- Eigen-analysis explains PCA, PageRank, gradient-descent step limits and exploding/vanishing gradients.
:::

=== POST ===
slug: singular-value-decomposition
title: Singular Value Decomposition: The Swiss Army Knife of Linear Algebra
category: math
level: Intermediate
tags: linear algebra, svd, low-rank, pca, pseudoinverse
summary: Every matrix — any shape, any rank — factors as rotation, scaling, rotation. We derive the SVD, prove the Eckart–Young low-rank theorem, and apply it to compression, PCA, recommenders and least squares.
---
Eigendecomposition works only for square matrices, and nicely only for symmetric ones. The **Singular Value Decomposition (SVD)** works for *every* matrix. Gilbert Strang calls it the climax of linear algebra, and in machine learning it is everywhere: PCA, latent semantic analysis, recommender systems, model compression, numerical least squares and the analysis of neural network weights.

## The theorem

Any real matrix $\mathbf{A} \in \mathbb{R}^{m \times n}$ of rank $r$ can be written as

$$
\mathbf{A} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top
$$

where

- $\mathbf{U} \in \mathbb{R}^{m \times m}$ is orthogonal — its columns $\mathbf{u}_i$ are **left singular vectors**;
- $\mathbf{V} \in \mathbb{R}^{n \times n}$ is orthogonal — its columns $\mathbf{v}_i$ are **right singular vectors**;
- $\boldsymbol{\Sigma} \in \mathbb{R}^{m \times n}$ is diagonal with **singular values** $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_r > 0$.

Equivalently, as a sum of rank-one pieces:

$$
\mathbf{A} = \sum_{i=1}^{r} \sigma_i\, \mathbf{u}_i \mathbf{v}_i^\top
$$

## Geometric meaning

Every linear map, however complicated, does three simple things in sequence: **rotate** (by $\mathbf{V}^\top$), **stretch** along the axes (by $\boldsymbol{\Sigma}$), and **rotate** again (by $\mathbf{U}$). The unit sphere is mapped to an ellipsoid whose semi-axes have lengths $\sigma_i$ in directions $\mathbf{u}_i$.

## Connection to eigenvalues

$$
\mathbf{A}^\top\mathbf{A} = \mathbf{V}\boldsymbol{\Sigma}^\top\boldsymbol{\Sigma}\mathbf{V}^\top, \qquad \mathbf{A}\mathbf{A}^\top = \mathbf{U}\boldsymbol{\Sigma}\boldsymbol{\Sigma}^\top\mathbf{U}^\top
$$

So the right singular vectors are eigenvectors of $\mathbf{A}^\top\mathbf{A}$, the left singular vectors are eigenvectors of $\mathbf{A}\mathbf{A}^\top$, and $\sigma_i^2$ are their eigenvalues. (In practice we never form $\mathbf{A}^\top\mathbf{A}$ explicitly — it squares the condition number and loses precision.)

## The Eckart–Young theorem: best low-rank approximation

Truncate the SVD to the top $k$ terms:

$$
\mathbf{A}_k = \sum_{i=1}^{k} \sigma_i\, \mathbf{u}_i\mathbf{v}_i^\top
$$

**Theorem (Eckart–Young–Mirsky).** Among all matrices of rank at most $k$, $\mathbf{A}_k$ is the closest to $\mathbf{A}$ in both the spectral and Frobenius norms:

$$
\|\mathbf{A} - \mathbf{A}_k\|_F = \sqrt{\sigma_{k+1}^2 + \dots + \sigma_r^2}
$$

This is the mathematical basis of every "keep the important directions, drop the noise" technique.

## Application 1: image compression

A grayscale image is a matrix. Storing a rank-$k$ approximation needs $k(m + n + 1)$ numbers instead of $mn$.

```python
import numpy as np

rng = np.random.default_rng(0)
# A synthetic "image": smooth structure plus noise
x = np.linspace(0, 1, 200)
img = np.outer(np.sin(6 * x), np.cos(4 * x)) + 0.5 * np.outer(x, x**2) + 0.05 * rng.normal(size=(200, 200))

U, s, Vt = np.linalg.svd(img, full_matrices=False)
for k in [1, 2, 5, 20]:
    approx = (U[:, :k] * s[:k]) @ Vt[:k]
    err = np.linalg.norm(img - approx) / np.linalg.norm(img)
    print(f"rank {k:>2}: relative error {err:.3f}, storage {k * (200 + 200 + 1)} vs {200 * 200}")
```

## Application 2: PCA

Centre the data matrix $\mathbf{X}$ (subtract column means). Its SVD $\mathbf{X} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$ directly gives the principal directions (columns of $\mathbf{V}$) and the variance explained by each ($\sigma_i^2/(n-1)$). This is how scikit-learn implements PCA.

## Application 3: recommender systems

A user–item rating matrix is approximately low rank: tastes are driven by a few latent factors (genre, mood, price sensitivity). Low-rank factorisation — conceptually an SVD of a matrix with missing entries — powered the winning approaches in the Netflix Prize.

## Application 4: least squares and the pseudoinverse

The **Moore–Penrose pseudoinverse** is

$$
\mathbf{A}^+ = \mathbf{V}\boldsymbol{\Sigma}^+\mathbf{U}^\top
$$

where $\boldsymbol{\Sigma}^+$ inverts the non-zero singular values. Then $\mathbf{x} = \mathbf{A}^+\mathbf{b}$ is the **minimum-norm least-squares solution** of $\mathbf{Ax} \approx \mathbf{b}$, even when $\mathbf{A}$ is rank-deficient.

## Application 5: conditioning and neural networks

The **condition number** $\kappa(\mathbf{A}) = \sigma_{\max}/\sigma_{\min}$ measures sensitivity of $\mathbf{Ax} = \mathbf{b}$ to perturbations. In deep learning, the singular values of weight matrices govern how signals and gradients grow through layers. **Spectral normalisation**, used to stabilise GAN training, divides a weight matrix by $\sigma_{\max}$. And **LoRA** fine-tuning exploits the empirical observation that useful weight *updates* are approximately low rank.

:::tip
For very large matrices, use **truncated** or **randomised SVD** (`sklearn.utils.extmath.randomized_svd`, `scipy.sparse.linalg.svds`) to compute only the top $k$ components — often orders of magnitude faster.
:::

:::exercise
1. Compute the SVD of $\begin{bmatrix} 3 & 0 \\ 4 & 5 \end{bmatrix}$ with NumPy and verify $\mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top = \mathbf{A}$.
2. Load any photo, convert it to grayscale and plot reconstructions at ranks 5, 20 and 50. Plot the singular value spectrum on a log scale.
3. Prove that $\|\mathbf{A}\|_F^2 = \sum_i \sigma_i^2$.
:::

:::takeaway
- SVD factors any matrix as rotation × scaling × rotation: $\mathbf{A} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$.
- Truncated SVD is the optimal low-rank approximation (Eckart–Young).
- It underlies PCA, compression, recommenders, pseudoinverses and low-rank fine-tuning.
- Singular values measure conditioning and signal growth in networks.
:::

=== POST ===
slug: norms-inner-products-and-distances
title: Norms, Inner Products, Projections and Distances
category: math
level: Beginner
tags: linear algebra, norms, dot product, cosine similarity, projection
summary: How long is a vector, and how similar are two vectors? We study L1, L2 and L∞ norms, dot products, cosine similarity, orthogonal projections and distance metrics used across ML.
---
Machine learning constantly asks "how big?" and "how similar?". How large are the weights (regularisation)? How far is this point from that cluster centre (k-means)? How similar are two sentences (semantic search)? The mathematical tools for these questions are **norms**, **inner products** and **distances**.

## Norms

A **norm** $\|\cdot\|$ assigns a length to each vector and satisfies:

1. $\|\mathbf{x}\| \ge 0$, with equality only for $\mathbf{x} = \mathbf{0}$;
2. $\|c\mathbf{x}\| = |c|\,\|\mathbf{x}\|$;
3. **Triangle inequality**: $\|\mathbf{x} + \mathbf{y}\| \le \|\mathbf{x}\| + \|\mathbf{y}\|$.

The family of **$L_p$ norms**:

$$
\|\mathbf{x}\|_p = \left( \sum_{i=1}^d |x_i|^p \right)^{1/p}
$$

| Norm | Formula | Unit ball shape (2-D) | ML use |
|---|---|---|---|
| $L_1$ (Manhattan) | $\sum_i \lvert x_i \rvert$ | Diamond | Lasso — produces sparse weights |
| $L_2$ (Euclidean) | $\sqrt{\sum_i x_i^2}$ | Circle | Ridge / weight decay; distances |
| $L_\infty$ (max) | $\max_i \lvert x_i \rvert$ | Square | Adversarial perturbation budgets |
| $L_0$ ("norm") | number of non-zeros | — | Sparsity (not a true norm) |

:::note
Why does $L_1$ regularisation produce **sparse** solutions while $L_2$ does not? Picture the elliptical contours of a loss function meeting the constraint region. The $L_1$ diamond has sharp corners on the axes, and the ellipse typically touches it at a corner — where some coordinates are exactly zero. The round $L_2$ ball has no corners, so the solution shrinks all coordinates without zeroing them.
:::

For matrices, the **Frobenius norm** $\|\mathbf{A}\|_F = \sqrt{\sum_{ij} A_{ij}^2}$ treats the matrix as a long vector; the **spectral norm** $\|\mathbf{A}\|_2 = \sigma_{\max}$ is the maximum stretch factor.

## Inner products

The standard **inner (dot) product** on $\mathbb{R}^d$ is

$$
\langle \mathbf{x}, \mathbf{y} \rangle = \mathbf{x}^\top\mathbf{y} = \sum_i x_i y_i = \|\mathbf{x}\|\,\|\mathbf{y}\|\cos\theta
$$

It links algebra with geometry: the sign tells whether vectors point the same way; zero means **orthogonal**. The **Cauchy–Schwarz inequality** $|\mathbf{x}^\top\mathbf{y}| \le \|\mathbf{x}\|\,\|\mathbf{y}\|$ guarantees $|\cos\theta| \le 1$.

Generalised inner products $\langle \mathbf{x}, \mathbf{y}\rangle_{\mathbf{M}} = \mathbf{x}^\top\mathbf{M}\mathbf{y}$ with positive definite $\mathbf{M}$ define different geometries — the basis of the Mahalanobis distance and kernel methods.

## Cosine similarity

$$
\cos(\mathbf{x}, \mathbf{y}) = \frac{\mathbf{x}^\top\mathbf{y}}{\|\mathbf{x}\|\,\|\mathbf{y}\|}
$$

Cosine similarity ignores length and compares only direction. It is the standard similarity for text embeddings and TF-IDF vectors, where document length should not dominate. If vectors are L2-normalised, cosine similarity equals the dot product, and squared Euclidean distance equals $2 - 2\cos\theta$ — so nearest-neighbour search by either criterion gives the same ranking.

## Orthogonal projection

The projection of $\mathbf{y}$ onto the line spanned by $\mathbf{a}$ is

$$
\text{proj}_{\mathbf{a}}(\mathbf{y}) = \frac{\mathbf{a}^\top\mathbf{y}}{\mathbf{a}^\top\mathbf{a}}\,\mathbf{a}
$$

The projection onto the column space of a matrix $\mathbf{A}$ (full column rank) is $\mathbf{P}\mathbf{y}$ with

$$
\mathbf{P} = \mathbf{A}(\mathbf{A}^\top\mathbf{A})^{-1}\mathbf{A}^\top
$$

**Linear regression is projection.** The least-squares prediction $\hat{\mathbf{y}} = \mathbf{X}\hat{\mathbf{w}}$ is the orthogonal projection of the target vector $\mathbf{y}$ onto the column space of the feature matrix $\mathbf{X}$. The residual $\mathbf{y} - \hat{\mathbf{y}}$ is orthogonal to every feature column — that orthogonality condition *is* the normal equation $\mathbf{X}^\top(\mathbf{y} - \mathbf{X}\mathbf{w}) = \mathbf{0}$.

## Distance metrics

A **metric** $d(\mathbf{x}, \mathbf{y})$ is non-negative, zero only for identical points, symmetric, and obeys the triangle inequality. Common choices:

- **Euclidean**: $\|\mathbf{x} - \mathbf{y}\|_2$ — default for continuous features (scale them first!).
- **Manhattan**: $\|\mathbf{x} - \mathbf{y}\|_1$ — robust to outliers in single coordinates.
- **Mahalanobis**: $\sqrt{(\mathbf{x} - \mathbf{y})^\top\boldsymbol{\Sigma}^{-1}(\mathbf{x} - \mathbf{y})}$ — accounts for feature correlations and scales; used in anomaly detection.
- **Hamming**: number of differing positions — for binary codes and strings.
- **Cosine distance** $1 - \cos$ — not a true metric, but widely used for embeddings.

```python
import numpy as np

x = np.array([3.0, 4.0]); y = np.array([4.0, 3.0])
print("L1:", np.linalg.norm(x, 1), "L2:", np.linalg.norm(x), "Linf:", np.linalg.norm(x, np.inf))
cos = x @ y / (np.linalg.norm(x) * np.linalg.norm(y))
print("cosine:", round(cos, 3))

a = np.array([1.0, 1.0])
proj = (a @ x) / (a @ a) * a
print("projection of x on a:", proj, " residual . a =", (x - proj) @ a)   # ~0
```

:::warning
Distance-based methods (k-NN, k-means, SVM with RBF kernel) are **sensitive to feature scale**. If one feature is income in taka (values ~100,000) and another is age (~30), Euclidean distance is dominated by income. Always standardise features before using such methods.
:::

:::exercise
1. Sketch the unit balls of $L_1$, $L_2$ and $L_\infty$ in 2-D and explain Lasso's sparsity using the sketch.
2. Prove that for unit vectors $\|\mathbf{x} - \mathbf{y}\|_2^2 = 2 - 2\,\mathbf{x}^\top\mathbf{y}$.
3. Fit linear regression with NumPy and verify that the residual is orthogonal to every column of $\mathbf{X}$.
:::

:::takeaway
- $L_1$, $L_2$, $L_\infty$ norms measure size differently; $L_1$ encourages sparsity.
- The dot product encodes angle; cosine similarity compares direction only.
- Least squares is orthogonal projection onto the column space of the features.
- Choose distance metrics deliberately and scale features first.
:::

=== POST ===
slug: derivatives-gradients-chain-rule
title: Derivatives, Gradients and the Chain Rule
category: math
level: Beginner
tags: calculus, derivatives, gradient, chain rule, taylor series
summary: Learning is the art of nudging parameters in the right direction. We review derivatives, partial derivatives, gradients, directional derivatives, Taylor expansions and the chain rule that makes backpropagation possible.
---
Training a model means adjusting parameters to reduce a loss. To know *which way* to adjust, we need to know how the loss changes when each parameter changes. That is exactly what derivatives measure. In this lecture we build the calculus you need for the rest of the course — especially the chain rule, which is the mathematical heart of backpropagation.

## The derivative

For $f: \mathbb{R} \to \mathbb{R}$, the derivative at $x$ is

$$
f'(x) = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h}
$$

It is the slope of the tangent line — the best **linear approximation** of $f$ near $x$:

$$
f(x + h) \approx f(x) + f'(x)\,h
$$

Rules to know by heart: power rule $(x^n)' = nx^{n-1}$; $(e^x)' = e^x$; $(\ln x)' = 1/x$; product rule $(uv)' = u'v + uv'$; quotient rule; and the chain rule below.

Derivatives of ML's favourite functions:

| Function | Derivative |
|---|---|
| Sigmoid $\sigma(x) = \frac{1}{1 + e^{-x}}$ | $\sigma(x)(1 - \sigma(x))$ |
| $\tanh(x)$ | $1 - \tanh^2(x)$ |
| ReLU $\max(0, x)$ | $1$ if $x > 0$, $0$ if $x < 0$ |
| Softplus $\ln(1 + e^x)$ | $\sigma(x)$ |

## Partial derivatives and the gradient

For $f: \mathbb{R}^d \to \mathbb{R}$, the **partial derivative** $\partial f/\partial x_i$ measures change along coordinate $i$ holding the others fixed. Stacking them gives the **gradient**:

$$
\nabla f(\mathbf{x}) = \begin{bmatrix} \frac{\partial f}{\partial x_1} & \cdots & \frac{\partial f}{\partial x_d} \end{bmatrix}^\top
$$

### Why the gradient points uphill

The **directional derivative** in unit direction $\mathbf{u}$ is $D_{\mathbf{u}}f = \nabla f^\top\mathbf{u} = \|\nabla f\|\cos\theta$. It is maximised when $\mathbf{u}$ points along $\nabla f$. Therefore:

- $\nabla f$ points in the direction of **steepest ascent**;
- $-\nabla f$ points in the direction of **steepest descent**;
- $\nabla f$ is perpendicular to the level sets (contours) of $f$.

This one fact justifies gradient descent: $\mathbf{w} \leftarrow \mathbf{w} - \eta\nabla L(\mathbf{w})$.

## The chain rule

If $y = f(u)$ and $u = g(x)$, then

$$
\frac{dy}{dx} = \frac{dy}{du}\cdot\frac{du}{dx}
$$

For a long composition $L = f_4(f_3(f_2(f_1(x))))$ — which is what a neural network is — the derivative is a *product* of local derivatives:

$$
\frac{dL}{dx} = f_4'(\cdot)\; f_3'(\cdot)\; f_2'(\cdot)\; f_1'(x)
$$

**Multivariable chain rule.** If $L$ depends on $x$ through several intermediate variables $u_1, \dots, u_k$, we *sum* over paths:

$$
\frac{\partial L}{\partial x} = \sum_{j=1}^{k} \frac{\partial L}{\partial u_j}\,\frac{\partial u_j}{\partial x}
$$

:::note
Backpropagation is nothing more than the multivariable chain rule, applied from the output backwards, reusing intermediate results so that each local derivative is computed once. The "product of many local derivatives" also explains vanishing gradients: multiply many sigmoid derivatives (each at most 0.25) and the product shrinks exponentially.
:::

## Worked example: logistic loss

Let $z = wx + b$, $p = \sigma(z)$, $L = -[y\ln p + (1 - y)\ln(1 - p)]$. By the chain rule:

$$
\frac{\partial L}{\partial p} = -\frac{y}{p} + \frac{1 - y}{1 - p}, \qquad \frac{\partial p}{\partial z} = p(1 - p), \qquad \frac{\partial z}{\partial w} = x
$$

Multiplying, most terms cancel:

$$
\frac{\partial L}{\partial w} = (p - y)\,x, \qquad \frac{\partial L}{\partial b} = p - y
$$

This beautifully simple result — *prediction minus target, times input* — is why sigmoid pairs naturally with cross-entropy.

## Taylor series

Near a point $\mathbf{x}_0$:

$$
f(\mathbf{x}_0 + \boldsymbol{\delta}) \approx f(\mathbf{x}_0) + \nabla f(\mathbf{x}_0)^\top\boldsymbol{\delta} + \tfrac{1}{2}\boldsymbol{\delta}^\top\mathbf{H}(\mathbf{x}_0)\boldsymbol{\delta}
$$

where $\mathbf{H}$ is the **Hessian** matrix of second derivatives. The first-order term justifies gradient descent; the second-order term underlies Newton's method and explains why curvature limits the step size.

## Checking gradients numerically

```python
import numpy as np

def loss(w, x, y):
    p = 1 / (1 + np.exp(-(w @ x)))
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))

def analytic_grad(w, x, y):
    p = 1 / (1 + np.exp(-(w @ x)))
    return (p - y) * x

def numeric_grad(f, w, eps=1e-6):
    g = np.zeros_like(w)
    for i in range(len(w)):
        e = np.zeros_like(w); e[i] = eps
        g[i] = (f(w + e) - f(w - e)) / (2 * eps)      # central difference
    return g

rng = np.random.default_rng(1)
w, x, y = rng.normal(size=3), rng.normal(size=3), 1.0
a = analytic_grad(w, x, y)
n = numeric_grad(lambda v: loss(v, x, y), w)
print(a, n, "rel. error:", np.linalg.norm(a - n) / np.linalg.norm(a + n))
```

:::tip
**Gradient checking** with central differences is the standard way to debug a hand-written backward pass. A relative error below about $10^{-7}$ is excellent; above $10^{-3}$ almost certainly signals a bug. Use float64 when checking.
:::

:::exercise
1. Derive $\sigma'(x) = \sigma(x)(1 - \sigma(x))$ from the definition of the sigmoid.
2. Compute the gradient of $f(x, y) = x^2y + \sin(xy)$ and evaluate it at $(1, \pi)$.
3. Using the chain rule, derive $\partial L/\partial w$ for squared loss $L = (wx + b - y)^2$. Verify with the gradient checker.
:::

:::takeaway
- The derivative is the best local linear approximation; the gradient collects partial derivatives.
- $-\nabla f$ is the direction of steepest descent — the basis of gradient descent.
- The chain rule multiplies local derivatives along paths and sums across paths: this is backpropagation.
- Verify analytic gradients numerically with central differences.
:::

=== POST ===
slug: matrix-calculus-essentials
title: Matrix Calculus Essentials: Jacobians, Hessians and Vector Derivatives
category: math
level: Intermediate
tags: calculus, jacobian, hessian, matrix calculus, backpropagation
summary: Deep learning differentiates vectors with respect to matrices. We learn the Jacobian, Hessian, key vector-derivative identities, and the shape-checking discipline that makes backprop derivations painless.
---
Neural networks do not have one parameter — they have millions, arranged in matrices. To derive gradients efficiently we need calculus that speaks the language of vectors and matrices. Matrix calculus looks intimidating, but it reduces to a handful of identities and one golden rule: **check the shapes**.

## Layout convention

In this course we use the **denominator layout** for gradients of scalars: if $L$ is a scalar and $\mathbf{W}$ is an $m \times n$ matrix, then $\partial L/\partial \mathbf{W}$ is also $m \times n$, with entry $(i,j)$ equal to $\partial L/\partial W_{ij}$. This matches what PyTorch's `.grad` stores: the gradient has the same shape as the parameter.

## The Jacobian

For a vector function $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$, the **Jacobian** is the $m \times n$ matrix of all first partial derivatives:

$$
\mathbf{J} = \frac{\partial \mathbf{f}}{\partial \mathbf{x}} = \begin{bmatrix}
\frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\
\vdots & \ddots & \vdots \\
\frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n}
\end{bmatrix}
$$

It is the best linear approximation of $\mathbf{f}$: $\mathbf{f}(\mathbf{x} + \boldsymbol{\delta}) \approx \mathbf{f}(\mathbf{x}) + \mathbf{J}\boldsymbol{\delta}$. The chain rule for vector functions becomes **matrix multiplication of Jacobians**:

$$
\frac{\partial \mathbf{z}}{\partial \mathbf{x}} = \frac{\partial \mathbf{z}}{\partial \mathbf{y}}\,\frac{\partial \mathbf{y}}{\partial \mathbf{x}}
$$

Important Jacobians:

- Linear map $\mathbf{y} = \mathbf{W}\mathbf{x}$: $\partial\mathbf{y}/\partial\mathbf{x} = \mathbf{W}$.
- Element-wise activation $\mathbf{y} = \phi(\mathbf{x})$: $\partial\mathbf{y}/\partial\mathbf{x} = \text{diag}(\phi'(\mathbf{x}))$ — diagonal, so we implement it as an element-wise product, never as a full matrix.
- Softmax $\mathbf{s} = \text{softmax}(\mathbf{z})$: $\partial s_i/\partial z_j = s_i(\delta_{ij} - s_j)$, i.e. $\text{diag}(\mathbf{s}) - \mathbf{s}\mathbf{s}^\top$.

## The Hessian

For scalar $f: \mathbb{R}^n \to \mathbb{R}$, the **Hessian** $\mathbf{H}_{ij} = \partial^2 f/\partial x_i\partial x_j$ is symmetric (for smooth $f$). It describes curvature: positive definite at a point with zero gradient means a local minimum; indefinite means a **saddle point**. In high-dimensional deep learning loss surfaces, saddle points vastly outnumber poor local minima.

## Identities to memorise

| Scalar function $f$ | Gradient $\nabla_{\mathbf{x}} f$ |
|---|---|
| $\mathbf{a}^\top\mathbf{x}$ | $\mathbf{a}$ |
| $\mathbf{x}^\top\mathbf{x}$ | $2\mathbf{x}$ |
| $\mathbf{x}^\top\mathbf{A}\mathbf{x}$ | $(\mathbf{A} + \mathbf{A}^\top)\mathbf{x}$ (equals $2\mathbf{A}\mathbf{x}$ if symmetric) |
| $\|\mathbf{A}\mathbf{x} - \mathbf{b}\|^2$ | $2\mathbf{A}^\top(\mathbf{A}\mathbf{x} - \mathbf{b})$ |

| Scalar function of a matrix | Gradient w.r.t. $\mathbf{W}$ |
|---|---|
| $\mathbf{a}^\top\mathbf{W}\mathbf{b}$ | $\mathbf{a}\mathbf{b}^\top$ |
| $\text{tr}(\mathbf{A}\mathbf{W})$ | $\mathbf{A}^\top$ |
| $\|\mathbf{W}\|_F^2$ | $2\mathbf{W}$ |

## Deriving linear regression in one line

Minimise $L(\mathbf{w}) = \|\mathbf{X}\mathbf{w} - \mathbf{y}\|^2$. Using the table:

$$
\nabla_{\mathbf{w}}L = 2\mathbf{X}^\top(\mathbf{X}\mathbf{w} - \mathbf{y}) = \mathbf{0} \;\Longrightarrow\; \mathbf{X}^\top\mathbf{X}\mathbf{w} = \mathbf{X}^\top\mathbf{y}
$$

These are the **normal equations**. The Hessian is $2\mathbf{X}^\top\mathbf{X}$, which is positive semi-definite, so the loss is convex and any solution is a global minimum.

## The backprop rule for a linear layer

Consider a layer $\mathbf{Y} = \mathbf{X}\mathbf{W}$ with a batch $\mathbf{X} \in \mathbb{R}^{B \times d}$, weights $\mathbf{W} \in \mathbb{R}^{d \times k}$, output $\mathbf{Y} \in \mathbb{R}^{B \times k}$. Suppose we already know the **upstream gradient** $\mathbf{G} = \partial L/\partial \mathbf{Y} \in \mathbb{R}^{B \times k}$. Then

$$
\frac{\partial L}{\partial \mathbf{W}} = \mathbf{X}^\top\mathbf{G} \quad (d \times k), \qquad \frac{\partial L}{\partial \mathbf{X}} = \mathbf{G}\mathbf{W}^\top \quad (B \times d)
$$

:::tip
**The shape trick.** You can often *reconstruct* these formulas just by making shapes match. $\partial L/\partial\mathbf{W}$ must be $d \times k$. The ingredients are $\mathbf{X}$ ($B \times d$) and $\mathbf{G}$ ($B \times k$). The only product giving $d \times k$ is $\mathbf{X}^\top\mathbf{G}$. Done. Always sanity-check derivations this way.
:::

```python
import numpy as np

rng = np.random.default_rng(0)
B, d, k = 4, 3, 2
X, W = rng.normal(size=(B, d)), rng.normal(size=(d, k))
T = rng.normal(size=(B, k))                     # targets

def L(W):                                       # L = 0.5 * ||XW - T||_F^2
    return 0.5 * np.sum((X @ W - T) ** 2)

G = X @ W - T                                   # dL/dY
grad_W = X.T @ G                                # our formula

num = np.zeros_like(W); eps = 1e-6
for i in range(d):
    for j in range(k):
        E = np.zeros_like(W); E[i, j] = eps
        num[i, j] = (L(W + E) - L(W - E)) / (2 * eps)
print(np.allclose(grad_W, num, atol=1e-6))      # True
```

## Softmax + cross-entropy

For logits $\mathbf{z}$, probabilities $\mathbf{p} = \text{softmax}(\mathbf{z})$ and one-hot target $\mathbf{y}$, the loss $L = -\sum_i y_i\ln p_i$ has the elegant gradient

$$
\frac{\partial L}{\partial \mathbf{z}} = \mathbf{p} - \mathbf{y}
$$

— the same "prediction minus target" pattern we found for the sigmoid. Frameworks fuse softmax and cross-entropy into one operation precisely to use this simple, numerically stable gradient.

## Vector–Jacobian products

Frameworks never build full Jacobians — for a layer with a million inputs and outputs that would be $10^{12}$ numbers. Reverse-mode automatic differentiation computes **vector–Jacobian products** $\mathbf{g}^\top\mathbf{J}$ directly, which cost about as much as the forward computation. This is why training costs only a small constant factor more than inference.

:::exercise
1. Derive $\nabla_{\mathbf{x}}(\mathbf{x}^\top\mathbf{A}\mathbf{x})$ component-wise and confirm the table entry.
2. Derive $\partial L/\partial\mathbf{z} = \mathbf{p} - \mathbf{y}$ for softmax cross-entropy using the softmax Jacobian.
3. Extend the code to a layer $\mathbf{Y} = \mathbf{X}\mathbf{W} + \mathbf{b}$ and derive and verify $\partial L/\partial\mathbf{b}$.
:::

:::takeaway
- Gradients of scalars have the same shape as the parameter (denominator layout).
- The Jacobian generalises the derivative; the chain rule becomes Jacobian multiplication.
- Key results: normal equations, $\partial L/\partial\mathbf{W} = \mathbf{X}^\top\mathbf{G}$, softmax-CE gradient $\mathbf{p} - \mathbf{y}$.
- Autodiff computes vector–Jacobian products, never full Jacobians.
:::
