=== POST ===
slug: from-neurons-to-neural-networks
title: From Biological Neurons to Artificial Neural Networks
category: deep-learning
level: Beginner
tags: neural networks, neurons, history, deep learning, representation learning
summary: We open the Deep Learning track by tracing the path from biological neurons to artificial ones, defining a neural network precisely, and explaining why depth and learned representations changed AI.
---
Welcome to the Deep Learning track. Over the next lectures we will build — from first principles — the technology behind modern image recognition, speech recognition, machine translation and generative AI. Let us begin where the field began: with the neuron.

## The biological inspiration

A biological neuron receives signals through thousands of **dendrites**, integrates them in the cell body, and — if the combined input exceeds a threshold — fires an electrical spike down its **axon** to other neurons via **synapses**. Learning in the brain involves changing synaptic strengths; Donald Hebb (1949) summarised it as "cells that fire together wire together". The human brain has roughly 86 billion neurons and on the order of $10^{14}$ synapses.

:::warning
Artificial neural networks are *inspired* by the brain, not models of it. Real neurons communicate with spike timing, have complex dendritic computation, and learn with local rules we only partly understand. Backpropagation, as used in deep learning, is not known to occur in the brain in the same form. Treat the analogy as motivation, not explanation.
:::

## The artificial neuron

An artificial neuron computes a weighted sum of its inputs plus a bias, then applies a non-linear **activation function** $\phi$:

$$
a = \phi\left(\sum_{j=1}^{d}w_jx_j + b\right) = \phi(\mathbf{w}^\top\mathbf{x} + b)
$$

With $\phi$ = step function, this is McCulloch and Pitts' 1943 threshold unit; with $\phi$ = sigmoid it is logistic regression. The weights $\mathbf{w}$ and bias $b$ are learned.

## Networks of neurons

A **feedforward neural network** (multilayer perceptron) arranges neurons in **layers**. Each layer computes

$$
\mathbf{h}^{(l)} = \phi\left(\mathbf{W}^{(l)}\mathbf{h}^{(l-1)} + \mathbf{b}^{(l)}\right), \qquad \mathbf{h}^{(0)} = \mathbf{x}
$$

and the final layer produces the output (logits for classification, values for regression). The whole network is a composition of functions:

$$
f(\mathbf{x}) = f^{(L)}\big(f^{(L-1)}(\dots f^{(1)}(\mathbf{x}))\big)
$$

"Deep" learning simply means many layers.

## Why non-linearity is essential

Without activation functions, each layer is a linear map, and a composition of linear maps is linear: $\mathbf{W}^{(2)}\mathbf{W}^{(1)}\mathbf{x} = \mathbf{W}'\mathbf{x}$. A hundred linear layers are no more expressive than one. Non-linear activations let networks represent curved decision boundaries and complex functions.

## Representation learning: the key idea

Classical ML relies on hand-engineered features. Deep networks **learn features** automatically, layer by layer. In an image classifier:

- early layers detect edges and colour blobs;
- middle layers combine them into textures and parts (eyes, wheels);
- late layers represent whole objects.

Each layer transforms the data into a representation in which the next task is easier. Remember the phrase from the ML track: a neural network is "linear regression on learned features" — the last layer is a linear model, and everything before it learns the features.

## Why deep learning took off after 2010

The ideas are decades old. Three ingredients arrived together:

1. **Data** — large labelled datasets like ImageNet (over a million images) and web-scale text.
2. **Compute** — GPUs made the massive matrix multiplications fast and affordable.
3. **Algorithms and tricks** — ReLU activations, good initialisation, dropout, batch normalisation, residual connections, the Adam optimiser.

Together they allowed networks with many layers to be trained reliably, and performance scaled with data and model size.

## A first network in PyTorch

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(2, 16), nn.ReLU(),
    nn.Linear(16, 16), nn.ReLU(),
    nn.Linear(16, 1),                 # a logit for binary classification
)
print(model)
print("parameters:", sum(p.numel() for p in model.parameters()))

x = torch.randn(4, 2)                 # a batch of 4 examples with 2 features
print(torch.sigmoid(model(x)).squeeze())   # predicted probabilities (untrained)
```

This tiny network has 337 parameters. In the coming lectures we will learn how to train it — computing gradients by backpropagation and updating weights by gradient descent.

## What deep learning is good (and not so good) at

**Excels at:** perception (images, audio), language, large-scale pattern recognition, generating content, learning from raw high-dimensional data.

**Struggles with:** small datasets without pretrained models, guarantees and verification, out-of-distribution robustness, explaining its decisions, and tasks where tabular gradient boosting is simpler and stronger.

:::exercise
1. Show algebraically that two stacked linear layers without activations equal one linear layer.
2. Count the parameters of a network with layer sizes 784 → 256 → 128 → 10.
3. Write a one-page comparison of a biological neuron and an artificial neuron, listing three similarities and three differences.
:::

:::takeaway
- An artificial neuron computes $\phi(\mathbf{w}^\top\mathbf{x} + b)$; networks stack layers of them.
- Non-linear activations are essential; linear layers alone collapse into one.
- Deep networks learn hierarchical representations instead of hand-crafted features.
- Data, GPUs and training innovations made deep learning practical after 2010.
:::

=== POST ===
slug: the-perceptron
title: The Perceptron: The First Learning Machine
category: deep-learning
level: Beginner
tags: perceptron, linear classifier, convergence theorem, xor, history
summary: Rosenblatt's perceptron learned to classify by correcting its mistakes. We derive its learning rule, prove the convergence theorem, reveal its XOR limitation, and see how it foreshadowed modern networks.
---
In 1958 Frank Rosenblatt demonstrated a machine at Cornell that learned to distinguish simple visual patterns. The *New York Times* reported excitedly that the navy expected the device to one day "walk, talk, see, write, reproduce itself and be conscious of its existence". The reality was more modest — but the **perceptron** was the first practical learning algorithm for a neural model, and its ideas live on in every network today.

## The model

For input $\mathbf{x} \in \mathbb{R}^d$ and label $y \in \{-1, +1\}$, the perceptron predicts

$$
\hat{y} = \text{sign}(\mathbf{w}^\top\mathbf{x} + b)
$$

a linear classifier with a hard threshold.

## The learning rule

Loop over training examples. Whenever an example is **misclassified** ($y_i(\mathbf{w}^\top\mathbf{x}_i + b) \le 0$), update:

$$
\mathbf{w} \leftarrow \mathbf{w} + \eta\,y_i\mathbf{x}_i, \qquad b \leftarrow b + \eta\,y_i
$$

Correct examples cause no change. Intuition: if a positive example scored too low, add it to $\mathbf{w}$, increasing $\mathbf{w}^\top\mathbf{x}_i$ next time; if a negative example scored too high, subtract it.

This is **stochastic gradient descent** on the **perceptron loss** $\max(0, -y_i(\mathbf{w}^\top\mathbf{x}_i + b))$ — a cousin of the SVM hinge loss without the margin.

```python
import numpy as np

def perceptron(X, y, epochs=50, lr=1.0):
    w, b = np.zeros(X.shape[1]), 0.0
    for epoch in range(epochs):
        mistakes = 0
        for xi, yi in zip(X, y):
            if yi * (w @ xi + b) <= 0:
                w += lr * yi * xi; b += lr * yi; mistakes += 1
        if mistakes == 0:
            print(f"converged after {epoch + 1} epochs")
            break
    return w, b

rng = np.random.default_rng(0)
X = np.vstack([rng.normal([2, 2], 0.7, (50, 2)), rng.normal([-2, -2], 0.7, (50, 2))])
y = np.r_[np.ones(50), -np.ones(50)]
w, b = perceptron(X, y)
print("accuracy:", np.mean(np.sign(X @ w + b) == y))
```

## The perceptron convergence theorem

**Theorem (Novikoff, 1962).** Suppose the data is linearly separable with **margin** $\gamma$: there is a unit vector $\mathbf{w}^*$ with $y_i\,\mathbf{w}^{*\top}\mathbf{x}_i \ge \gamma > 0$ for all $i$, and all $\|\mathbf{x}_i\| \le R$. Then the perceptron (with $b$ absorbed into $\mathbf{w}$) makes at most

$$
\left(\frac{R}{\gamma}\right)^2
$$

mistakes, regardless of the order of examples or the dimension.

*Proof sketch.* After $k$ mistakes starting from $\mathbf{w} = \mathbf{0}$:

- Each update increases the projection onto $\mathbf{w}^*$ by at least $\gamma$: $\mathbf{w}_k^\top\mathbf{w}^* \ge k\gamma$.
- Each update increases the squared norm by at most $R^2$ (because the example was misclassified, the cross term is non-positive): $\|\mathbf{w}_k\|^2 \le kR^2$.

By Cauchy–Schwarz, $k\gamma \le \mathbf{w}_k^\top\mathbf{w}^* \le \|\mathbf{w}_k\| \le \sqrt{k}R$, hence $k \le R^2/\gamma^2$. $\blacksquare$

:::note
The bound depends on the **margin**, not the number of features. Data that is separable with a wide margin is easy to learn even in very high dimensions — the same insight that later motivated support vector machines.
:::

## The XOR problem and the first AI winter

If data is **not** linearly separable, the perceptron never converges; it cycles forever. The simplest example is XOR:

| $x_1$ | $x_2$ | XOR |
|---|---|---|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

No single line separates the ones from the zeros. Minsky and Papert's 1969 book *Perceptrons* rigorously analysed such limitations of single-layer perceptrons (for instance, computing connectedness of a figure). Although they knew multilayer networks could represent more, there was no known way to **train** them, and research funding for neural networks declined sharply.

## The solution: hidden layers

A two-layer network solves XOR easily: one hidden unit computes OR, another computes AND, and the output computes "OR and not AND". The problem was learning the hidden-layer weights — solved by **backpropagation** in the 1980s, as we will see shortly.

## Variants

- **Averaged perceptron** — average weights over all steps; generalises much better and was a strong NLP tagger for years.
- **Voted perceptron** — weighted vote of intermediate weight vectors.
- **Kernel perceptron** — replace dot products with kernels for non-linear boundaries.

:::exercise
1. Run the perceptron on XOR and observe that it never converges.
2. Construct weights by hand for a two-layer network of step units that computes XOR.
3. Generate separable data with a small margin and a large margin; compare the number of mistakes with the bound $(R/\gamma)^2$.
:::

:::takeaway
- The perceptron is a linear classifier updated only on mistakes: $\mathbf{w} \leftarrow \mathbf{w} + \eta y_i\mathbf{x}_i$.
- On separable data it converges in at most $(R/\gamma)^2$ mistakes — independent of dimension.
- It cannot learn non-separable functions like XOR, which contributed to the first AI winter.
- Hidden layers plus backpropagation overcame the limitation.
:::

=== POST ===
slug: multilayer-perceptrons-universal-approximation
title: Multilayer Perceptrons and the Universal Approximation Theorem
category: deep-learning
level: Intermediate
tags: mlp, universal approximation, depth, width, expressivity
summary: With one hidden layer, a network can approximate any continuous function — so why go deep? We state the universal approximation theorem, build intuition with bumps, and explain the efficiency advantages of depth.
---
A multilayer perceptron (MLP) with even a single hidden layer is astonishingly expressive. This is formalised by the **universal approximation theorem**, one of the most quoted — and most misunderstood — results in deep learning. Today we state it, see why it is true, and then ask the more important question: if one layer is enough in principle, why do we use many?

## The MLP

A network with one hidden layer of $m$ units computes

$$
f(\mathbf{x}) = \sum_{j=1}^{m}v_j\,\phi(\mathbf{w}_j^\top\mathbf{x} + b_j) + c
$$

— a weighted sum of $m$ "ridge" functions, each a non-linearity applied to a projection of the input. Deeper MLPs stack such layers.

## The theorem

**Universal Approximation Theorem** (Cybenko 1989 for sigmoids; Hornik 1991; Leshno et al. 1993 for any non-polynomial activation, including ReLU). Let $\phi$ be continuous and not a polynomial. For any continuous function $g$ on a compact set $K \subset \mathbb{R}^d$ and any $\epsilon > 0$, there exist $m$ and parameters such that

$$
\sup_{\mathbf{x} \in K}|f(\mathbf{x}) - g(\mathbf{x})| < \epsilon
$$

In words: a single hidden layer, made wide enough, can approximate any continuous function on a bounded region as closely as we like.

## Intuition: building functions from bumps

With ReLU $\phi(z) = \max(0, z)$ in one dimension:

- A single ReLU is a hinge.
- The difference of two shifted ReLUs creates a ramp that flattens out: a **step**.
- The difference of two steps creates a **bump** — a small tower over an interval.
- Sum many bumps of chosen heights and you can trace any continuous curve, like a histogram approximating a density.

```python
import numpy as np
import torch, torch.nn as nn

x = torch.linspace(-3, 3, 400).unsqueeze(1)
target = torch.sin(2 * x) + 0.3 * x**2                     # an arbitrary continuous function

for width in [2, 8, 64]:
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(1, width), nn.ReLU(), nn.Linear(width, 1))
    opt = torch.optim.Adam(net.parameters(), lr=0.01)
    for _ in range(3000):
        opt.zero_grad(); loss = ((net(x) - target) ** 2).mean(); loss.backward(); opt.step()
    print(f"width {width:>3}: MSE = {loss.item():.5f}")
```

Wider single-layer networks approximate the target ever more closely — the theorem in action.

## What the theorem does NOT say

:::warning
1. **It says nothing about how many units are needed.** The required width can grow exponentially with input dimension.
2. **It says nothing about learning.** Gradient descent may not find the approximating weights.
3. **It says nothing about generalisation.** Fitting a function on training points does not mean predicting well elsewhere.
4. **Many other model families are also universal approximators** — polynomials, kernel machines, decision trees with enough leaves. Universality is not what makes neural networks special.
:::

## Why depth?

If shallow networks are universal, why use deep ones? Because **depth can be exponentially more efficient**.

- **Depth-separation results**: there exist functions computable by small deep networks that require exponentially many units in shallow networks (e.g. Telgarsky 2016, using compositions of "sawtooth" functions; Eldan & Shamir 2016 for 3 vs 2 layers).
- **Compositionality**: a ReLU network partitions input space into linear regions. Each additional layer can "fold" space, so the number of linear regions can grow **exponentially with depth** but only polynomially with width.
- **Hierarchical structure of real data**: images are composed of objects, objects of parts, parts of edges; language of sentences, phrases and words. Deep networks mirror this compositional structure, reusing intermediate features across many higher-level concepts.

:::example Folding intuition
Consider $h(x) = 2x$ for $x < 0.5$ and $2 - 2x$ otherwise — a "tent" map computable by two ReLUs. Composing it $k$ times produces a sawtooth with $2^{k-1}$ teeth using only $O(k)$ units. A one-hidden-layer network needs about $2^k$ units to draw the same sawtooth.
:::

## Width vs depth in practice

- Very deep plain networks are hard to train (vanishing gradients) — solved by residual connections and normalisation, discussed in later lectures.
- Very wide shallow networks tend to memorise rather than build reusable features.
- Modern architectures balance both, and **scaling laws** show performance improves predictably when depth, width and data grow together.

## Designing an MLP

For tabular data or as a component inside larger models:

- 2–4 hidden layers; widths of 64–1024; ReLU/GELU activations.
- Normalisation (batch or layer norm) and dropout for regularisation.
- Output layer: linear for regression; logits + softmax/sigmoid for classification.
- Tune learning rate first, then width/depth and regularisation.

In transformers, every block contains an MLP (the "feed-forward network") that holds a large share of the parameters and is believed to store much of the model's factual knowledge.

:::exercise
1. Construct by hand a one-hidden-layer ReLU network that outputs a bump: 0 outside $[1, 2]$, 1 on $[1.25, 1.75]$, linear in between.
2. Train depth-1 and depth-4 networks with equal parameter counts to fit a sawtooth made by composing the tent map five times. Which fits better?
3. Explain in your own words why universality does not imply that a network will generalise.
:::

:::takeaway
- One hidden layer with a non-polynomial activation can approximate any continuous function on a compact set.
- The theorem guarantees neither efficient size, learnability nor generalisation.
- Depth can be exponentially more efficient than width, matching the compositional structure of real data.
- Practical networks balance depth and width with residuals and normalisation.
:::

=== POST ===
slug: activation-functions
title: "Activation Functions: Sigmoid, Tanh, ReLU, GELU, SwiGLU and Softmax"
category: deep-learning
level: Beginner
tags: activation functions, relu, gelu, sigmoid, tanh, swiglu
summary: The choice of non-linearity shapes how gradients flow and how networks learn. We compare the classical and modern activations, their derivatives and failure modes, and which to use where.
---
Activation functions are the non-linear heart of a neural network. Change them and you change how fast a network trains, whether gradients survive through many layers, and sometimes the final accuracy. The history of deep learning's progress is partly the history of better activations.

## Sigmoid

$$
\sigma(z) = \frac{1}{1 + e^{-z}}, \qquad \sigma'(z) = \sigma(z)(1 - \sigma(z))
$$

Outputs lie in $(0, 1)$ — useful for probabilities in output layers and gates. As a hidden activation it has serious problems:

- **Saturation**: for large $|z|$, $\sigma'(z) \approx 0$ — gradients vanish.
- **Maximum derivative 0.25**: backpropagating through many sigmoid layers multiplies many factors ≤ 0.25, shrinking gradients exponentially.
- **Not zero-centred**: outputs are always positive, making gradients of the next layer's weights all share a sign, which causes zig-zagging updates.

## Tanh

$$
\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}} = 2\sigma(2z) - 1, \qquad \tanh'(z) = 1 - \tanh^2(z)
$$

Zero-centred with outputs in $(-1, 1)$ and maximum derivative 1 — better than sigmoid, but still saturates. It is still used inside LSTMs and GRUs.

## ReLU: the revolution

$$
\text{ReLU}(z) = \max(0, z), \qquad \text{ReLU}'(z) = \begin{cases} 1 & z > 0 \\ 0 & z < 0 \end{cases}
$$

Popularised around 2010–2012 (Nair & Hinton; Glorot et al.; AlexNet), ReLU transformed deep learning:

- **No saturation for positive inputs**: gradient is exactly 1, so it passes through many layers undiminished.
- **Cheap**: a comparison.
- **Sparse activations**: many units output exactly zero.

Its weakness is the **dying ReLU** problem: if a unit's pre-activation becomes negative for all inputs (e.g. after a large update), its gradient is zero forever and it never recovers.

## ReLU variants

| Activation | Formula | Notes |
|---|---|---|
| Leaky ReLU | $\max(\alpha z, z)$, $\alpha \approx 0.01$ | Small negative slope avoids dead units |
| PReLU | Leaky ReLU with learned $\alpha$ | Used in some vision models |
| ELU | $z$ if $z > 0$, else $\alpha(e^z - 1)$ | Smooth, negative values push mean activations towards 0 |
| SELU | Scaled ELU | Self-normalising networks under specific conditions |

## Smooth modern activations: GELU and Swish/SiLU

**GELU** (Gaussian Error Linear Unit, Hendrycks & Gimpel 2016) weights the input by the probability that a standard Gaussian is below it:

$$
\text{GELU}(z) = z\,\Phi(z) \approx 0.5z\left(1 + \tanh\left[\sqrt{2/\pi}\,(z + 0.044715z^3)\right]\right)
$$

**Swish / SiLU**: $z\,\sigma(\beta z)$ (with $\beta = 1$ for SiLU).

Both are smooth, non-monotonic near zero, and behave like ReLU for large positive inputs. GELU is the default in BERT and GPT-style transformers; SiLU is common in vision models such as EfficientNet.

## Gated linear units: SwiGLU

Many recent large language models use a **gated** feed-forward layer:

$$
\text{SwiGLU}(\mathbf{x}) = \big(\text{Swish}(\mathbf{x}\mathbf{W}_1) \odot \mathbf{x}\mathbf{W}_2\big)\mathbf{W}_3
$$

One projection acts as a learned, input-dependent gate on another. Shazeer (2020) found GLU variants improved transformer quality at equal compute, and SwiGLU has since been adopted widely in open LLMs.

## Output-layer activations

The output activation must match the task and loss:

| Task | Output activation | Loss |
|---|---|---|
| Binary classification | Sigmoid (usually fused into the loss) | Binary cross-entropy with logits |
| Multiclass | Softmax (fused) | Cross-entropy |
| Multi-label | Independent sigmoids | Binary cross-entropy per label |
| Regression | None (identity) | MSE / MAE / Huber |
| Positive targets | Softplus or exp | Appropriate likelihood |

## Visualising and checking gradients

```python
import torch

z = torch.linspace(-5, 5, 11, requires_grad=True)
acts = {"sigmoid": torch.sigmoid, "tanh": torch.tanh, "relu": torch.relu,
        "gelu": torch.nn.functional.gelu, "silu": torch.nn.functional.silu}
for name, f in acts.items():
    g, = torch.autograd.grad(f(z).sum(), z)
    print(f"{name:<8} grad at z=-5..5: {[round(v, 2) for v in g.tolist()]}")

# Gradient surviving through 20 layers of each activation (product of derivatives at z = 1)
for name, f in [("sigmoid", torch.sigmoid), ("tanh", torch.tanh), ("relu", torch.relu)]:
    x = torch.tensor(1.0, requires_grad=True); y = x
    for _ in range(20):
        y = f(y)
    y.backward()
    print(f"{name:<8} d(output)/d(input) through 20 layers: {x.grad.item():.2e}")
```

The sigmoid chain's gradient collapses towards zero; ReLU's survives.

:::tip
**Default choices today:** ReLU or GELU/SiLU for hidden layers in CNNs and MLPs; GELU or SwiGLU inside transformers; tanh/sigmoid inside recurrent gates; and never a sigmoid in deep hidden stacks. Pair ReLU-family activations with **He initialisation** (covered in a later lecture).
:::

:::exercise
1. Derive the derivative of tanh and show $\tanh(z) = 2\sigma(2z) - 1$.
2. Train the same MLP on MNIST with sigmoid, tanh, ReLU and GELU hidden layers for 5 epochs. Compare training curves.
3. Count the fraction of dead ReLU units in a network trained with a deliberately large learning rate.
:::

:::takeaway
- Sigmoid and tanh saturate, causing vanishing gradients in deep stacks.
- ReLU keeps gradients alive and is cheap, but units can die; leaky variants help.
- GELU, SiLU and SwiGLU are smooth modern defaults, especially in transformers.
- Choose output activations to match the task and loss (usually fused into the loss for stability).
:::

=== POST ===
slug: loss-functions-deep-learning
title: Loss Functions in Deep Learning: What Are We Really Optimising?
category: deep-learning
level: Beginner
tags: loss functions, cross-entropy, mse, huber, focal loss, contrastive loss
summary: The loss function defines what "good" means to a network. We survey regression, classification, ranking and representation-learning losses, their probabilistic meaning, and common implementation mistakes.
---
A neural network does exactly one thing during training: it reduces its loss. If the loss does not capture what you actually care about, the network will faithfully optimise the wrong thing. Choosing a loss is therefore a modelling decision, not a detail. Today we survey the losses you will use most often and connect each to its underlying assumptions.

## Regression losses

**Mean squared error (MSE / L2):**

$$
L = \frac{1}{n}\sum_i(y_i - \hat{y}_i)^2
$$

Negative log-likelihood of Gaussian noise; predicts the conditional mean; sensitive to outliers.

**Mean absolute error (MAE / L1):** $\frac{1}{n}\sum_i|y_i - \hat{y}_i|$. Laplace noise; predicts the median; robust but has a constant gradient magnitude, which can make fine convergence slow.

**Huber (smooth L1):**

$$
L_\delta(r) = \begin{cases} \frac{1}{2}r^2 & |r| \le \delta \\ \delta\left(|r| - \frac{1}{2}\delta\right) & |r| > \delta \end{cases}
$$

Quadratic near zero, linear for large errors — the best of both. Used in object-detection box regression and in DQN.

**Quantile (pinball) loss** for prediction intervals; **Gaussian NLL** with a predicted variance for heteroscedastic uncertainty.

## Classification losses

**Binary cross-entropy (BCE):** for logit $z$ and label $y \in \{0,1\}$:

$$
L = -\big[y\log\sigma(z) + (1 - y)\log(1 - \sigma(z))\big]
$$

**Categorical cross-entropy:** for logits $\mathbf{z}$ and class $c$:

$$
L = -\log\frac{e^{z_c}}{\sum_k e^{z_k}} = -z_c + \log\sum_k e^{z_k}
$$

Both are negative log-likelihoods and have the clean gradient "probability minus target".

:::warning
**The most common implementation bug:** applying softmax (or sigmoid) in the model *and then* using a loss that applies it again. In PyTorch, `nn.CrossEntropyLoss` and `nn.BCEWithLogitsLoss` expect **raw logits**. Passing probabilities produces wrong gradients and slow, poor training — without any error message.
:::

### Refinements

- **Class weights** — up-weight rare classes.
- **Label smoothing** — replace one-hot targets with $(1 - \epsilon)$ on the true class and $\epsilon/(K-1)$ elsewhere. It discourages over-confident logits and often improves calibration and generalisation (widely used in image classification and translation).
- **Focal loss** (Lin et al., 2017):

$$
L = -(1 - p_t)^\gamma\log p_t
$$

where $p_t$ is the predicted probability of the true class. The factor $(1 - p_t)^\gamma$ down-weights easy, well-classified examples, focusing training on hard ones. It was introduced for dense object detection, where easy background examples vastly outnumber objects.

## Margin and ranking losses

- **Hinge loss** $\max(0, 1 - yz)$ — SVM-style classification.
- **Pairwise ranking loss** — for a relevant item $i$ and irrelevant $j$: $\max(0, m - s_i + s_j)$ or the logistic version $\log(1 + e^{-(s_i - s_j)})$ (as in BPR for recommenders).
- **Triplet loss** — for an anchor $a$, positive $p$ (same identity) and negative $n$:

$$
L = \max\big(0,\; \|f(a) - f(p)\|^2 - \|f(a) - f(n)\|^2 + m\big)
$$

Used in face recognition (FaceNet) to learn embeddings where same-identity faces are close.

## Contrastive losses for representation learning

**InfoNCE** (used in SimCLR, CLIP and dense retrieval): given a query $\mathbf{q}$, one positive key $\mathbf{k}^+$ and many negatives, with similarity $s$ and temperature $\tau$:

$$
L = -\log\frac{\exp(s(\mathbf{q}, \mathbf{k}^+)/\tau)}{\sum_{j}\exp(s(\mathbf{q}, \mathbf{k}_j)/\tau)}
$$

It is a softmax cross-entropy where the "class" is "which key is the positive". Other examples in the batch serve as negatives, so larger batches give harder, more informative contrasts.

## Generative-model losses

- **Reconstruction + KL** (VAE ELBO).
- **Adversarial losses** (GANs).
- **Denoising MSE on predicted noise** (diffusion models).
- **Next-token cross-entropy** (language models).

We will meet each in the Generative AI track.

## Composite losses

Real systems combine terms: detection = classification + box regression; multi-task models = weighted sum of task losses; regularised objectives add weight decay or auxiliary losses. Balancing weights matters — a term with a larger scale can dominate gradients. Normalise terms or tune their weights on validation data.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

logits = torch.tensor([[2.0, 0.5, -1.0], [0.1, 0.2, 3.0]])
target = torch.tensor([0, 2])

print("CE:", F.cross_entropy(logits, target).item())
print("CE + label smoothing 0.1:", F.cross_entropy(logits, target, label_smoothing=0.1).item())
# WRONG: softmax applied twice
print("double-softmax bug:", F.cross_entropy(F.softmax(logits, 1), target).item())

def focal_loss(logits, target, gamma=2.0):
    logp = F.log_softmax(logits, dim=1).gather(1, target[:, None]).squeeze(1)
    return (-(1 - logp.exp()) ** gamma * logp).mean()
print("focal:", focal_loss(logits, target).item())

pred, y = torch.tensor([1.0, 2.0, 10.0]), torch.tensor([1.2, 1.8, 2.0])
print("MSE:", F.mse_loss(pred, y).item(), " Huber:", F.huber_loss(pred, y, delta=1.0).item())
```

:::exercise
1. Derive the gradient of focal loss with respect to $p_t$ and explain how $\gamma$ changes it.
2. Show that label smoothing corresponds to cross-entropy against a mixture of the one-hot label and a uniform distribution.
3. Train a regressor on data with 5% extreme outliers using MSE, MAE and Huber. Compare test MAE on clean data.
:::

:::takeaway
- The loss encodes your assumptions: MSE ↔ Gaussian, MAE ↔ Laplace, cross-entropy ↔ categorical likelihood.
- Pass logits to fused losses; never apply softmax twice.
- Label smoothing, class weights and focal loss refine classification.
- Triplet and InfoNCE losses learn embeddings; composite losses need balancing.
:::

=== POST ===
slug: backpropagation-derived
title: Backpropagation Derived Step by Step
category: deep-learning
level: Intermediate
tags: backpropagation, chain rule, gradients, neural networks, training
summary: Backpropagation computes every gradient in a network at about the cost of one forward pass. We derive it for a two-layer network by hand, generalise to any depth, implement it in NumPy and verify it numerically.
---
Backpropagation is the algorithm that made deep learning possible. It was discovered several times — Linnainmaa's reverse-mode differentiation in 1970, Werbos' application to neural networks in 1974 — and brought to prominence by Rumelhart, Hinton and Williams in 1986. Every time you call `loss.backward()`, this algorithm runs. Today we derive it completely, by hand, so that it never feels like magic again.

## The setting

Consider a two-layer network for classification with input $\mathbf{x} \in \mathbb{R}^d$, a hidden layer of width $h$ and $K$ output classes:

$$
\begin{aligned}
\mathbf{z}_1 &= \mathbf{W}_1\mathbf{x} + \mathbf{b}_1 && (h) \\
\mathbf{a}_1 &= \text{ReLU}(\mathbf{z}_1) && (h) \\
\mathbf{z}_2 &= \mathbf{W}_2\mathbf{a}_1 + \mathbf{b}_2 && (K) \\
\mathbf{p} &= \text{softmax}(\mathbf{z}_2) && (K) \\
L &= -\log p_y
\end{aligned}
$$

We want $\frac{\partial L}{\partial\mathbf{W}_1}, \frac{\partial L}{\partial\mathbf{b}_1}, \frac{\partial L}{\partial\mathbf{W}_2}, \frac{\partial L}{\partial\mathbf{b}_2}$.

## The key idea

Define the **error signal** at each layer, $\boldsymbol{\delta} = \partial L/\partial\mathbf{z}$. Compute it at the output, then **propagate it backwards** layer by layer using the chain rule. Once you know $\boldsymbol{\delta}$ for a layer, the weight gradients of that layer follow immediately.

## Step 1: output layer

For softmax with cross-entropy, we showed earlier that

$$
\boldsymbol{\delta}_2 = \frac{\partial L}{\partial\mathbf{z}_2} = \mathbf{p} - \mathbf{y}
$$

where $\mathbf{y}$ is the one-hot label.

## Step 2: gradients of the output weights

Since $\mathbf{z}_2 = \mathbf{W}_2\mathbf{a}_1 + \mathbf{b}_2$, each entry $z_{2,k} = \sum_j W_{2,kj}a_{1,j} + b_{2,k}$, so $\partial z_{2,k}/\partial W_{2,kj} = a_{1,j}$. Therefore

$$
\frac{\partial L}{\partial\mathbf{W}_2} = \boldsymbol{\delta}_2\,\mathbf{a}_1^\top \quad (K \times h), \qquad \frac{\partial L}{\partial\mathbf{b}_2} = \boldsymbol{\delta}_2
$$

**Pattern:** weight gradient = (error at the layer's output) × (input to the layer)ᵀ.

## Step 3: propagate to the hidden layer

The hidden activation $\mathbf{a}_1$ influences every output through $\mathbf{W}_2$. Summing over those paths (the multivariable chain rule):

$$
\frac{\partial L}{\partial\mathbf{a}_1} = \mathbf{W}_2^\top\boldsymbol{\delta}_2
$$

Then pass through the ReLU, whose derivative is 1 where $z_1 > 0$ and 0 elsewhere:

$$
\boldsymbol{\delta}_1 = \frac{\partial L}{\partial\mathbf{z}_1} = \left(\mathbf{W}_2^\top\boldsymbol{\delta}_2\right) \odot \mathbb{1}[\mathbf{z}_1 > 0]
$$

## Step 4: gradients of the first layer

Exactly the same pattern as Step 2:

$$
\frac{\partial L}{\partial\mathbf{W}_1} = \boldsymbol{\delta}_1\,\mathbf{x}^\top, \qquad \frac{\partial L}{\partial\mathbf{b}_1} = \boldsymbol{\delta}_1
$$

## The general algorithm

For a network with layers $\mathbf{z}_l = \mathbf{W}_l\mathbf{a}_{l-1} + \mathbf{b}_l$, $\mathbf{a}_l = \phi(\mathbf{z}_l)$:

1. **Forward pass** — compute and **store** all $\mathbf{z}_l$ and $\mathbf{a}_l$.
2. **Output error** — $\boldsymbol{\delta}_L = \partial L/\partial\mathbf{z}_L$.
3. **Backward pass** — for $l = L, \dots, 1$:

$$
\frac{\partial L}{\partial\mathbf{W}_l} = \boldsymbol{\delta}_l\mathbf{a}_{l-1}^\top, \qquad \frac{\partial L}{\partial\mathbf{b}_l} = \boldsymbol{\delta}_l, \qquad \boldsymbol{\delta}_{l-1} = \left(\mathbf{W}_l^\top\boldsymbol{\delta}_l\right) \odot \phi'(\mathbf{z}_{l-1})
$$

For a mini-batch stored as rows of $\mathbf{X}$, the products become $\mathbf{A}_{l-1}^\top\boldsymbol{\Delta}_l$ summed (or averaged) over the batch.

## Why it is efficient

A naive approach would compute each of the $P$ parameter derivatives separately by perturbation, costing $O(P)$ forward passes. Backpropagation reuses the shared error signals $\boldsymbol{\delta}_l$, computing **all** gradients in one backward pass whose cost is about **two to three times** the forward pass. For a model with a billion parameters, that is the difference between impossible and routine. The price is **memory**: stored activations from the forward pass.

## Implementation in NumPy

```python
import numpy as np

rng = np.random.default_rng(0)
N, d, h, K = 64, 10, 32, 3
X = rng.normal(size=(N, d)); y = rng.integers(0, K, N); Y = np.eye(K)[y]
W1 = rng.normal(0, np.sqrt(2 / d), (d, h)); b1 = np.zeros(h)
W2 = rng.normal(0, np.sqrt(2 / h), (h, K)); b2 = np.zeros(K)

def forward(W1, b1, W2, b2):
    Z1 = X @ W1 + b1; A1 = np.maximum(0, Z1)
    Z2 = A1 @ W2 + b2
    Z2s = Z2 - Z2.max(1, keepdims=True)
    P = np.exp(Z2s) / np.exp(Z2s).sum(1, keepdims=True)
    loss = -np.log(P[np.arange(N), y]).mean()
    return loss, (Z1, A1, P)

def backward(cache):
    Z1, A1, P = cache
    D2 = (P - Y) / N                        # dL/dZ2, averaged over the batch
    dW2, db2 = A1.T @ D2, D2.sum(0)
    D1 = (D2 @ W2.T) * (Z1 > 0)             # dL/dZ1
    dW1, db1 = X.T @ D1, D1.sum(0)
    return dW1, db1, dW2, db2

loss, cache = forward(W1, b1, W2, b2)
grads = backward(cache)

# Gradient check on W1 with central differences
num = np.zeros_like(W1); eps = 1e-5
for i in range(d):
    for j in range(h):
        W1[i, j] += eps; lp, _ = forward(W1, b1, W2, b2)
        W1[i, j] -= 2 * eps; lm, _ = forward(W1, b1, W2, b2)
        W1[i, j] += eps; num[i, j] = (lp - lm) / (2 * eps)
print("relative error:", np.linalg.norm(num - grads[0]) / np.linalg.norm(num + grads[0]))

# Train with plain gradient descent
params = [W1, b1, W2, b2]
for step in range(500):
    loss, cache = forward(*params)
    for p, g in zip(params, backward(cache)):
        p -= 0.5 * g
print("final training loss:", round(loss, 4))
```

The gradient check should report a relative error around $10^{-8}$ or smaller — confirmation that our derivation and code agree.

:::note
Notice where vanishing and exploding gradients come from: each backward step multiplies by $\mathbf{W}_l^\top$ and by $\phi'$. Over many layers, these products can shrink or grow exponentially. Good initialisation, ReLU-type activations, normalisation and residual connections all exist to keep this product well-behaved.
:::

:::exercise
1. Derive backpropagation for the same network with a sigmoid hidden layer and MSE loss.
2. Extend the NumPy code to three layers and verify all gradients with the checker.
3. Explain why the forward activations must be stored, and estimate the activation memory for a batch of 256 examples through ten layers of width 4,096 in float32.
:::

:::takeaway
- Backprop propagates error signals $\boldsymbol{\delta}_l = \partial L/\partial\mathbf{z}_l$ backwards with the chain rule.
- Weight gradient = error × input ᵀ; $\boldsymbol{\delta}_{l-1} = (\mathbf{W}_l^\top\boldsymbol{\delta}_l) \odot \phi'(\mathbf{z}_{l-1})$.
- One backward pass computes all gradients at a small multiple of forward cost, trading memory for speed.
- Always verify hand-written gradients numerically.
:::

=== POST ===
slug: computational-graphs-autodiff
title: Computational Graphs and Automatic Differentiation
category: deep-learning
level: Intermediate
tags: autodiff, computational graph, reverse mode, pytorch, jax
summary: Frameworks compute gradients of arbitrary programs automatically. We compare symbolic, numerical and automatic differentiation, contrast forward and reverse mode, and build a tiny reverse-mode autodiff engine.
---
In the previous lecture we derived backpropagation by hand for a specific network. Nobody does that for a 100-layer transformer. Instead, frameworks like PyTorch, JAX and TensorFlow **automatically** differentiate any program built from differentiable operations. Understanding how they do it will make you a far better debugger and will demystify errors like "one of the variables needed for gradient computation has been modified by an inplace operation".

## Three ways to compute derivatives

1. **Numerical differentiation** — finite differences $\frac{f(x + h) - f(x - h)}{2h}$. Easy but approximate and costs $O(P)$ evaluations for $P$ parameters. Useful only for checking.
2. **Symbolic differentiation** — manipulate formulas like a computer-algebra system. Exact, but expressions can blow up in size ("expression swell") and it struggles with loops and branches.
3. **Automatic differentiation (autodiff)** — decompose the program into elementary operations with known derivatives and apply the chain rule **numerically** as the program runs. Exact to floating-point precision and efficient.

## Computational graphs

Any computation can be written as a directed acyclic graph whose nodes are elementary operations. For

$$
f(x, y) = (x + y)\cdot\sin(x)
$$

the graph is: $a = x + y$, $b = \sin(x)$, $f = a\cdot b$. Each node knows its **local derivative** with respect to its inputs: $\partial a/\partial x = 1$, $\partial b/\partial x = \cos x$, $\partial f/\partial a = b$, $\partial f/\partial b = a$.

## Forward mode

Forward mode propagates derivatives **with** the computation, from inputs to outputs, carrying a "tangent" $\dot{v} = \partial v/\partial x$ for one chosen input direction. It computes a **Jacobian–vector product** $\mathbf{J}\mathbf{v}$ in one pass. It is efficient when there are **few inputs and many outputs**. It can be implemented elegantly with **dual numbers** $a + b\varepsilon$ where $\varepsilon^2 = 0$.

## Reverse mode

Reverse mode first runs the program forward, recording the graph and intermediate values. Then it propagates **adjoints** $\bar{v} = \partial L/\partial v$ backwards from the output:

$$
\bar{v}_i = \sum_{j \in \text{children}(i)}\bar{v}_j\,\frac{\partial v_j}{\partial v_i}
$$

It computes a **vector–Jacobian product** $\mathbf{u}^\top\mathbf{J}$ — the gradient of one scalar with respect to **all** inputs — in a single backward pass. Since training has **one scalar loss and millions of parameters**, reverse mode is exactly what we need. Backpropagation is reverse-mode autodiff applied to neural networks.

| | Forward mode | Reverse mode |
|---|---|---|
| Computes | $\mathbf{J}\mathbf{v}$ (JVP) | $\mathbf{u}^\top\mathbf{J}$ (VJP) |
| Cost per pass | ~1 forward | ~1 forward + 1 backward |
| Efficient when | inputs ≪ outputs | outputs ≪ inputs (e.g. a loss) |
| Memory | Low | Stores intermediate values |

## Build a tiny autodiff engine

In the spirit of Andrej Karpathy's *micrograd*:

```python
import math

class Value:
    def __init__(self, data, parents=(), op=""):
        self.data, self.grad = data, 0.0
        self._parents, self._op = parents, op
        self._backward = lambda: None

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def sin(self):
        out = Value(math.sin(self.data), (self,), "sin")
        def _backward():
            self.grad += math.cos(self.data) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        order, seen = [], set()
        def topo(v):
            if v not in seen:
                seen.add(v)
                for p in v._parents:
                    topo(p)
                order.append(v)
        topo(self)
        self.grad = 1.0
        for v in reversed(order):          # reverse topological order
            v._backward()

x, y = Value(2.0), Value(3.0)
f = (x + y) * x.sin()
f.backward()
print("f =", round(f.data, 4))
print("df/dx =", round(x.grad, 4), " expected:", round(math.sin(2) + (2 + 3) * math.cos(2), 4))
print("df/dy =", round(y.grad, 4), " expected:", round(math.sin(2), 4))
```

Note the `+=` in each backward function: when a variable is used in several places ($x$ appears twice here), gradients from all paths **accumulate** — the "sum over paths" in the multivariable chain rule.

## How PyTorch does it

PyTorch builds a **dynamic graph** ("define-by-run"): each tensor operation records a node with a `grad_fn` as your Python code executes, so ordinary control flow (loops, if-statements) just works. Calling `.backward()` traverses the recorded graph in reverse.

```python
import torch
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)
f = (x + y) * torch.sin(x)
print(f.grad_fn)          # <MulBackward0 ...>
f.backward()
print(x.grad, y.grad)
```

Practical consequences:

- **Gradients accumulate** in `.grad` — call `optimizer.zero_grad()` every step.
- **`torch.no_grad()`** disables graph recording for inference, saving memory.
- **`.detach()`** cuts a tensor out of the graph (used for target networks, stop-gradient tricks).
- **In-place operations** can overwrite values needed for the backward pass, hence the famous error.
- The graph is freed after `backward()` unless `retain_graph=True`.

**JAX** takes a functional approach: `jax.grad(f)` returns a new function computing the gradient, composable with `jax.jit` (compilation), `jax.vmap` (vectorisation) and `jax.jvp`/`jax.vjp` for forward and reverse mode.

## Memory tricks

Reverse mode stores activations, which dominates memory for large models. **Gradient (activation) checkpointing** stores only some activations and recomputes the rest during the backward pass — trading about one extra forward pass for a large memory reduction, which lets much bigger models or batches fit on a GPU.

:::exercise
1. Add `__pow__`, `exp` and `relu` to the `Value` class and train a single neuron with it.
2. Implement forward-mode differentiation with a `Dual` number class and compute $f'(2)$ for the same $f$.
3. Explain why a Hessian–vector product can be computed with one forward-over-reverse pass without forming the Hessian.
:::

:::takeaway
- Autodiff applies the chain rule to elementary operations — exact and efficient.
- Forward mode computes JVPs (few inputs); reverse mode computes VJPs (one loss, many parameters).
- Backpropagation is reverse-mode autodiff; gradients accumulate across paths.
- PyTorch records dynamic graphs; checkpointing trades compute for memory.
:::

=== POST ===
slug: sgd-momentum-nesterov
title: "Optimisers I: SGD, Momentum and Nesterov Acceleration"
category: deep-learning
level: Intermediate
tags: optimization, sgd, momentum, nesterov, training
summary: Plain SGD zig-zags through ravines and crawls across plateaus. Momentum accumulates velocity to fix both. We derive heavy-ball and Nesterov momentum, analyse their effect on ill-conditioned problems, and give tuning advice.
---
In the mathematics track we analysed gradient descent and saw its Achilles' heel: on ill-conditioned problems — long, narrow valleys — it oscillates across the steep walls while creeping along the floor. Deep-learning loss surfaces are full of such ravines, plateaus and noisy gradients. **Momentum** is the simplest and most important fix, and it remains part of almost every optimiser used today.

## Mini-batch SGD recap

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta\,\mathbf{g}_t, \qquad \mathbf{g}_t = \nabla_{\boldsymbol{\theta}}\frac{1}{|\mathcal{B}_t|}\sum_{i \in \mathcal{B}_t}\ell_i(\boldsymbol{\theta}_t)
$$

Problems:

1. **Ravines** — a learning rate small enough to be stable along the steep direction is tiny along the shallow direction.
2. **Noise** — mini-batch gradients fluctuate, causing jittery progress.
3. **Plateaus and saddle points** — tiny gradients mean tiny steps.

## Momentum (heavy ball)

Polyak's **heavy-ball** method (1964) keeps a **velocity** — an exponentially decaying accumulation of past gradients:

$$
\mathbf{v}_{t+1} = \beta\,\mathbf{v}_t + \mathbf{g}_t, \qquad \boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta\,\mathbf{v}_{t+1}
$$

with momentum coefficient $\beta$ typically 0.9.

**Physical intuition:** a ball rolling downhill gathers speed in consistent directions and is not deflected much by small bumps.

**Why it fixes ravines:** across the valley, gradients alternate in sign, so they **cancel** in the velocity; along the valley floor they point the same way every step, so they **accumulate**. Unrolling the recursion, the effective step in a consistent direction approaches

$$
\eta\,(1 + \beta + \beta^2 + \dots) = \frac{\eta}{1 - \beta}
$$

— with $\beta = 0.9$, ten times the plain step. Momentum also **averages out gradient noise** over roughly $1/(1 - \beta)$ steps.

:::note
For a quadratic with condition number $\kappa$, well-tuned heavy-ball momentum reduces the number of iterations needed from $O(\kappa)$ to $O(\sqrt{\kappa})$. For $\kappa = 10{,}000$ that is the difference between about 10,000 and about 100 iterations.
:::

## Nesterov accelerated gradient

Nesterov's idea (1983): since we are going to move by roughly $\beta\mathbf{v}_t$ anyway, compute the gradient at that **look-ahead** point rather than the current one:

$$
\mathbf{v}_{t+1} = \beta\,\mathbf{v}_t + \nabla f(\boldsymbol{\theta}_t - \eta\beta\,\mathbf{v}_t), \qquad \boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta\,\mathbf{v}_{t+1}
$$

If momentum is about to carry us past the minimum, the look-ahead gradient already points back and applies the brakes earlier. Nesterov momentum reduces overshooting and oscillation, and for smooth convex problems it achieves the optimal $O(1/t^2)$ convergence rate among first-order methods. In practice it is a small but reliable improvement over classical momentum.

## Seeing the difference

```python
import numpy as np

H = np.diag([1.0, 50.0])                  # ill-conditioned quadratic, kappa = 50
grad = lambda th: H @ th
start = np.array([10.0, 1.0])

def run(method, lr, beta=0.9, steps=100):
    th, v = start.copy(), np.zeros(2)
    for _ in range(steps):
        if method == "sgd":
            th = th - lr * grad(th)
        elif method == "momentum":
            v = beta * v + grad(th); th = th - lr * v
        elif method == "nesterov":
            v = beta * v + grad(th - lr * beta * v); th = th - lr * v
    return np.linalg.norm(th)

for m in ["sgd", "momentum", "nesterov"]:
    print(f"{m:<9} distance from optimum after 100 steps: {run(m, lr=0.035):.2e}")
```

With the same learning rate, momentum methods reach the optimum orders of magnitude faster than plain gradient descent.

## Momentum in PyTorch

```python
import torch
model = torch.nn.Linear(10, 1)
opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True, weight_decay=5e-4)
```

(PyTorch's formulation folds the learning rate slightly differently from the equations above, but the behaviour is equivalent for a constant learning rate.)

## Tuning SGD with momentum

- **$\beta = 0.9$** is a robust default; 0.95–0.99 for very noisy or large-batch settings.
- The **effective learning rate** is about $\eta/(1 - \beta)$ — if you increase $\beta$, reduce $\eta$ accordingly.
- Use a **learning-rate schedule** (step decay, cosine) — SGD with momentum depends heavily on it.
- Add **weight decay** for regularisation.

## SGD + momentum versus adaptive optimisers

Adaptive methods such as Adam (next lecture) usually converge faster with less tuning, and dominate transformer training. But **SGD with momentum** remains competitive — and sometimes generalises slightly better — for convolutional networks in image classification, where many well-known results were obtained with it. It also uses less memory (one state vector per parameter versus two for Adam).

:::exercise
1. Unroll the momentum recursion to express $\mathbf{v}_t$ as a weighted sum of past gradients.
2. In the example, find the largest stable learning rate for plain SGD and for momentum ($\beta = 0.9$).
3. Train a small CNN on CIFAR-10 with SGD (no momentum), SGD + momentum and SGD + Nesterov. Compare curves over 10 epochs.
:::

:::takeaway
- Momentum accumulates an exponentially weighted velocity of gradients, damping oscillations and accelerating consistent directions.
- The effective step is about $\eta/(1 - \beta)$; it improves ill-conditioned convergence from $O(\kappa)$ to $O(\sqrt{\kappa})$ on quadratics.
- Nesterov evaluates the gradient at a look-ahead point, braking earlier.
- SGD + momentum with a good schedule remains strong for CNNs.
:::
