=== POST ===
slug: adaptive-optimizers-adam-adamw
title: "Optimisers II: AdaGrad, RMSProp, Adam and AdamW"
category: deep-learning
level: Intermediate
tags: optimization, adam, adamw, rmsprop, adagrad
summary: Adaptive optimisers give each parameter its own learning rate. We derive AdaGrad, RMSProp and Adam including bias correction, explain why AdamW decouples weight decay, and survey newer optimisers.
---
Different parameters see very different gradients. The embedding of a rare word receives a gradient once in a thousand batches; a bias in the output layer receives one every step. A single global learning rate is too large for some parameters and too small for others. **Adaptive optimisers** scale each parameter's step by the history of its own gradients. Adam, the most famous of them, is probably the most widely used optimiser in deep learning.

## AdaGrad

Duchi, Hazan and Singer (2011) accumulate the sum of squared gradients per parameter:

$$
\mathbf{s}_t = \mathbf{s}_{t-1} + \mathbf{g}_t^2, \qquad \boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \frac{\eta}{\sqrt{\mathbf{s}_t} + \epsilon}\odot\mathbf{g}_t
$$

(all operations element-wise). Parameters with large past gradients get smaller steps; rarely updated parameters get larger ones — excellent for sparse features. Its flaw: $\mathbf{s}_t$ only grows, so the effective learning rate decays towards zero and training stalls in long non-convex runs.

## RMSProp

Hinton (in a 2012 lecture) replaced the sum with an **exponential moving average**, so old gradients are forgotten:

$$
\mathbf{s}_t = \rho\,\mathbf{s}_{t-1} + (1 - \rho)\,\mathbf{g}_t^2, \qquad \boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \frac{\eta}{\sqrt{\mathbf{s}_t} + \epsilon}\odot\mathbf{g}_t
$$

with $\rho \approx 0.9$–$0.99$. Dividing by the root-mean-square gradient normalises step sizes across parameters.

## Adam

Kingma and Ba (2014) combined **momentum** (first moment) with **RMSProp** (second moment):

$$
\begin{aligned}
\mathbf{m}_t &= \beta_1\mathbf{m}_{t-1} + (1 - \beta_1)\,\mathbf{g}_t \\
\mathbf{v}_t &= \beta_2\mathbf{v}_{t-1} + (1 - \beta_2)\,\mathbf{g}_t^2 \\
\hat{\mathbf{m}}_t &= \frac{\mathbf{m}_t}{1 - \beta_1^t}, \qquad \hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1 - \beta_2^t} \\
\boldsymbol{\theta}_{t+1} &= \boldsymbol{\theta}_t - \eta\,\frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon}
\end{aligned}
$$

Defaults: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$, $\eta = 10^{-3}$ (much smaller for large models).

### Why bias correction?

$\mathbf{m}_0 = \mathbf{v}_0 = \mathbf{0}$, so early averages are biased towards zero. At step 1, $\mathbf{v}_1 = 0.001\,\mathbf{g}_1^2$ — a thousand times too small. Dividing by $1 - \beta^t$ corrects this exactly in expectation (if gradients were stationary). Without it, the first steps would be enormous because the denominator is tiny.

### Properties

- The step size per parameter is roughly bounded by $\eta$, since $|\hat{m}|/\sqrt{\hat{v}} \lesssim 1$ — making Adam robust to gradient scale.
- It works well out of the box across many architectures, especially transformers, RNNs and GANs.
- It stores **two** extra values per parameter (memory = 3× parameters, excluding gradients).

## AdamW: decoupled weight decay

With plain SGD, L2 regularisation (adding $\frac{\lambda}{2}\|\boldsymbol{\theta}\|^2$ to the loss) and **weight decay** (shrinking weights each step, $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \eta\lambda\boldsymbol{\theta}$) are equivalent. With Adam they are **not**: an L2 gradient term gets divided by $\sqrt{\hat{\mathbf{v}}}$, so parameters with large gradients are barely regularised. Loshchilov and Hutter (2017) proposed **AdamW**, which applies weight decay directly:

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta\left(\frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} + \lambda\,\boldsymbol{\theta}_t\right)
$$

AdamW generalises better and is the standard optimiser for training transformers. Typical weight decay: 0.01–0.1, usually **not applied** to biases and normalisation parameters.

```python
import torch

model = torch.nn.Sequential(torch.nn.Linear(128, 256), torch.nn.GELU(),
                            torch.nn.LayerNorm(256), torch.nn.Linear(256, 10))
decay, no_decay = [], []
for name, p in model.named_parameters():
    (no_decay if p.ndim == 1 else decay).append(p)       # biases & norm weights are 1-D
opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.05},
                         {"params": no_decay, "weight_decay": 0.0}],
                        lr=3e-4, betas=(0.9, 0.999), eps=1e-8)
```

## Adam from scratch

```python
import numpy as np

def adam_step(theta, g, state, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, wd=0.0):
    state["t"] += 1
    state["m"] = b1 * state["m"] + (1 - b1) * g
    state["v"] = b2 * state["v"] + (1 - b2) * g**2
    m_hat = state["m"] / (1 - b1 ** state["t"])
    v_hat = state["v"] / (1 - b2 ** state["t"])
    return theta - lr * (m_hat / (np.sqrt(v_hat) + eps) + wd * theta)

# Minimise a badly scaled quadratic: 0.5*(x^2 + 1000*y^2)
theta = np.array([5.0, 5.0]); state = {"t": 0, "m": np.zeros(2), "v": np.zeros(2)}
for _ in range(2000):
    g = np.array([1.0, 1000.0]) * theta
    theta = adam_step(theta, g, state, lr=0.05)
print(theta.round(5))
```

Adam handles the 1000× difference in curvature gracefully because it normalises each coordinate's step.

## Known issues and newer optimisers

- **Generalisation gap**: on some vision tasks, Adam-trained models generalised slightly worse than SGD + momentum; AdamW and good schedules narrow this.
- **Instability early in training** for large models — mitigated by **learning-rate warm-up** (next lecture).
- **AMSGrad** fixes a theoretical non-convergence example by keeping the maximum of past $\mathbf{v}_t$.
- **LAMB/LARS** add layer-wise trust ratios for very large batch training.
- **Adafactor** factorises the second-moment matrix to save memory.
- **Lion** (found by automated search) uses only the sign of a momentum-like update — memory-efficient.
- Second-order-inspired methods such as **Shampoo** and **SOAP** precondition with approximate curvature matrices and have shown strong results in recent large-scale training.

| Optimiser | State per parameter | Typical use |
|---|---|---|
| SGD + momentum | 1 | CNNs for vision, well-tuned schedules |
| Adam / AdamW | 2 | Transformers, most new projects |
| Adafactor | < 1 (factored) | Memory-constrained large models |
| Lion | 1 | Memory-efficient alternative to AdamW |

:::tip
Start new projects with **AdamW**, learning rate around $3\times10^{-4}$ for small models (lower for large ones), weight decay 0.01–0.1, warm-up plus cosine decay, and gradient clipping at norm 1.0. Tune the learning rate before anything else.
:::

:::exercise
1. Show that without bias correction, Adam's first update has magnitude about $\eta\sqrt{1 - \beta_2}/(1 - \beta_1)$ times smaller or larger than intended. Which is it?
2. Compare Adam with L2 regularisation and AdamW on a small classification task; plot validation accuracy.
3. Explain why AdaGrad suits sparse features such as word counts.
:::

:::takeaway
- AdaGrad scales steps by accumulated squared gradients; RMSProp uses a moving average.
- Adam = momentum + RMSProp + bias correction; robust defaults make it widely used.
- AdamW decouples weight decay from the adaptive scaling and is the transformer standard.
- Adaptive optimisers cost extra memory; newer optimisers trade memory, speed and stability.
:::

=== POST ===
slug: learning-rate-schedules-warmup
title: Learning Rate Schedules, Warm-up and the LR Range Test
category: deep-learning
level: Intermediate
tags: learning rate, schedules, warmup, cosine annealing, one-cycle
summary: The learning rate is the most important hyperparameter, and it should change during training. We compare step, exponential, cosine and one-cycle schedules, explain why warm-up stabilises large models, and find good rates quickly.
---
If you tune only one hyperparameter, tune the learning rate. And once you have a good value, recognise that the best learning rate **changes during training**: large steps early to make rapid progress and escape poor regions, small steps late to settle into a good minimum. A **learning rate schedule** encodes this. For large models, a short **warm-up** at the start is also essential.

## Why decay the learning rate?

With stochastic gradients, SGD with a constant learning rate $\eta$ does not converge to a point; it bounces around the minimum in a region whose size scales with $\eta$ times the gradient noise. Reducing $\eta$ shrinks that region. Empirically, loss curves often show a sharp drop right after each learning-rate decrease.

## Common schedules

**Step decay:** multiply by $\gamma$ (e.g. 0.1) at fixed epochs (e.g. 30, 60, 90). Classic for ResNets on ImageNet.

**Exponential decay:** $\eta_t = \eta_0\gamma^t$.

**Inverse square root:** $\eta_t \propto 1/\sqrt{t}$ — used in the original Transformer paper after warm-up.

**Cosine annealing** (Loshchilov & Hutter, 2016):

$$
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\frac{\pi t}{T}\right)
$$

Smooth, only one real hyperparameter (the total length $T$), and very popular for both vision and language models. **Cosine with warm restarts** periodically resets to $\eta_{\max}$.

**One-cycle policy** (Smith, 2018): increase the learning rate linearly from low to a high maximum over the first part of training, then decrease it to a very low value, while momentum moves inversely. It often trains faster ("super-convergence").

**Warm-up–stable–decay (WSD):** warm up, hold a constant rate for most of training, then decay quickly at the end. It is convenient for large-model training because you can branch off decayed checkpoints at different points without committing to a total length in advance.

**Reduce on plateau:** decrease when validation loss stops improving — adaptive but reactive.

## Warm-up

For the first few hundred or thousand steps, increase the learning rate linearly from near zero to its target:

$$
\eta_t = \eta_{\max}\cdot\frac{t}{T_{\text{warmup}}} \quad (t \le T_{\text{warmup}})
$$

Why does this help?

1. At initialisation, gradients can be large and poorly aligned; big steps can push the network into regions it never recovers from (divergence, dead units, loss spikes).
2. Adam's second-moment estimate $\hat{\mathbf{v}}$ is unreliable for the first steps (few samples), so update magnitudes are noisy; warm-up keeps them small until statistics stabilise.
3. For transformers with post-layer normalisation, early training is especially unstable; warm-up was critical in the original Transformer.

Typical warm-up: 1–5% of total steps, or a few thousand steps for large models.

## Implementing a schedule

```python
import math
import torch

model = torch.nn.Linear(10, 1)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
total_steps, warmup = 10_000, 500

def lr_lambda(step):
    if step < warmup:
        return (step + 1) / warmup                      # linear warm-up
    progress = (step - warmup) / max(1, total_steps - warmup)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))   # cosine to 10% of peak

sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
for step in range(total_steps):
    # ... forward, loss.backward(), opt.step(), opt.zero_grad()
    sched.step()
    if step in (0, 250, 500, 5000, 9999):
        print(step, f"{sched.get_last_lr()[0]:.2e}")
```

Call `scheduler.step()` in the right place — per **step** for step-based schedules, per **epoch** for epoch-based ones. Mixing these up is a common silent bug.

## Finding the learning rate: the range test

Leslie Smith's **LR range test**: train for a few hundred steps while increasing the learning rate exponentially from very small (e.g. $10^{-7}$) to large (e.g. 10), recording the loss. Plot loss against learning rate:

- the loss first stays flat (too small), then decreases (good range), then explodes (too large);
- pick a maximum learning rate somewhat below the point where the loss is lowest — often about a tenth of the value where it starts to blow up.

```python
import numpy as np

def lr_range_test(train_step, lr_min=1e-7, lr_max=10, steps=200):
    lrs = np.geomspace(lr_min, lr_max, steps)
    losses = []
    for lr in lrs:
        loss = train_step(lr)                 # performs one update with this lr, returns loss
        losses.append(loss)
        if not np.isfinite(loss) or loss > 4 * min(losses):
            break
    return lrs[:len(losses)], np.array(losses)
```

## Batch size and learning rate

Larger batches give less noisy gradients and allow larger learning rates. The **linear scaling rule** — multiply the learning rate by $k$ when the batch is multiplied by $k$, with warm-up — works well up to a point (Goyal et al. trained ResNet-50 on ImageNet in one hour with this rule). For Adam, a square-root scaling is sometimes used. Beyond the "critical batch size", gains diminish.

:::tip
Practical defaults: AdamW with linear warm-up (2–5% of steps) followed by cosine decay to 5–10% of the peak. For SGD on CNNs: momentum 0.9 with cosine or step decay. Always plot the learning rate alongside the loss curve in your experiment logs.
:::

:::exercise
1. Train a CNN on CIFAR-10 with a constant learning rate, step decay and cosine decay for the same number of epochs. Compare final accuracy.
2. Remove warm-up from a small transformer's training and observe early loss behaviour at a high learning rate.
3. Run the LR range test on an MLP for MNIST and choose a learning rate from the plot.
:::

:::takeaway
- Decaying the learning rate reduces SGD noise and improves final performance.
- Cosine annealing and one-cycle are strong defaults; WSD is convenient for long runs.
- Warm-up stabilises early training, especially with Adam and transformers.
- Use the LR range test to find a good peak; scale learning rate with batch size carefully.
:::

=== POST ===
slug: weight-initialization
title: Weight Initialisation: Xavier, He and Why It Matters
category: deep-learning
level: Intermediate
tags: initialization, xavier, he, variance, deep networks
summary: Bad initial weights make signals explode or vanish before training even starts. We derive variance-preserving initialisation for tanh (Xavier) and ReLU (He) networks and discuss modern practice for deep and residual models.
---
Before a network learns anything, we must choose starting values for its weights. This seems like a detail, but it was one of the key obstacles that held back deep networks for years. Initialise too small and signals shrink to nothing as they pass through layers; too large and they explode. Today we derive the principled answer.

## Why not zeros?

If all weights in a layer start equal (e.g. all zero), every neuron in the layer computes the same output and receives the same gradient. They remain identical forever — the layer behaves like a single neuron. **Random initialisation breaks this symmetry.** (Biases can safely start at zero.)

## The variance argument

Consider a layer $z_i = \sum_{j=1}^{n_{\text{in}}}w_{ij}x_j$ with independent zero-mean weights of variance $\text{Var}(w)$ and independent inputs with zero mean and variance $\text{Var}(x)$. Then

$$
\text{Var}(z_i) = n_{\text{in}}\,\text{Var}(w)\,\text{Var}(x)
$$

After $L$ layers, the variance is multiplied by $\left(n_{\text{in}}\text{Var}(w)\right)^L$ (ignoring activations). If $n_{\text{in}}\text{Var}(w) = 1.5$ and $L = 50$, activations grow by a factor of about $1.5^{50} \approx 6 \times 10^{8}$; if it equals 0.5, they shrink by $10^{15}$. The same argument applies to gradients flowing backwards. We want this factor to be **exactly 1**.

## Xavier / Glorot initialisation

For activations that are roughly linear around zero (tanh, sigmoid's central region), Glorot and Bengio (2010) required variance preservation in both the forward pass ($n_{\text{in}}\text{Var}(w) = 1$) and the backward pass ($n_{\text{out}}\text{Var}(w) = 1$), and compromised with the average:

$$
\text{Var}(w) = \frac{2}{n_{\text{in}} + n_{\text{out}}}
$$

Uniform version: $w \sim U\left[-\sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}},\; \sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}\right]$.

## He / Kaiming initialisation

ReLU sets half its inputs to zero, which halves the second moment of the signal. He et al. (2015) compensated with a factor of 2:

$$
\text{Var}(w) = \frac{2}{n_{\text{in}}}, \qquad w \sim \mathcal{N}\left(0, \frac{2}{n_{\text{in}}}\right)
$$

This let them train very deep plain ReLU networks (30 layers) from scratch where Xavier initialisation stalled. It is the default for ReLU-family networks.

| Activation | Recommended initialisation |
|---|---|
| tanh, sigmoid, linear | Xavier / Glorot |
| ReLU, Leaky ReLU, GELU, SiLU | He / Kaiming (with the appropriate gain) |
| SELU | LeCun normal $\text{Var}(w) = 1/n_{\text{in}}$ |

## Seeing it happen

```python
import torch

def activation_stats(init, act, depth=50, width=512):
    x = torch.randn(1000, width)
    for _ in range(depth):
        W = torch.empty(width, width)
        init(W)
        x = act(x @ W.T)
    return x.std().item()

inits = {
    "N(0, 0.01)":  lambda W: torch.nn.init.normal_(W, 0, 0.01),
    "N(0, 1)":     lambda W: torch.nn.init.normal_(W, 0, 1.0),
    "Xavier":      torch.nn.init.xavier_normal_,
    "He (Kaiming)": lambda W: torch.nn.init.kaiming_normal_(W, nonlinearity="relu"),
}
for name, init in inits.items():
    print(f"ReLU net, {name:<13}: std after 50 layers = {activation_stats(init, torch.relu):.3e}")
```

Small fixed variance collapses activations to zero; unit variance explodes them; Xavier slowly shrinks them with ReLU; He keeps them stable.

## Beyond variance: modern practice

- **Orthogonal initialisation** — initialise weight matrices as random orthogonal matrices (all singular values 1), which preserves norms exactly in linear networks and helps RNNs.
- **Residual networks**: each block adds to the input, $\mathbf{x} + F(\mathbf{x})$, so variance grows with depth. Common fixes: initialise the last layer of each residual branch to zero ("zero-init residual" / Fixup) so each block starts as the identity, or scale residual-branch initialisation by $1/\sqrt{2L}$ as in GPT-2.
- **Normalisation layers** (BatchNorm, LayerNorm) make networks much less sensitive to initialisation, but good initialisation still speeds training.
- **Transformers**: small normal initialisation (e.g. standard deviation 0.02) is common, with scaled residual projections.
- **Maximal update parametrisation (μP)** prescribes how initialisation and learning rates should scale with width so that hyperparameters tuned on a small model transfer to a large one.

:::note
Frameworks already apply sensible defaults: PyTorch's `nn.Linear` uses a Kaiming-uniform variant. You rarely need to write initialisation code for standard layers — but when you build custom architectures, very deep networks or unusual activations, understanding these derivations is what lets you debug a network that "refuses to train".
:::

:::exercise
1. Derive the He initialisation variance by computing $\mathbb{E}[\text{ReLU}(z)^2]$ for symmetric $z$.
2. Modify the experiment to track the **gradient** norm at the first layer after backpropagating from layer 50 for each initialisation.
3. Initialise a 20-block residual MLP with and without zero-init of the final layer of each block; compare the output variance at initialisation.
:::

:::takeaway
- Identical weights never differentiate — random initialisation breaks symmetry.
- Variance multiplies by $n_{\text{in}}\text{Var}(w)$ per layer; keep it near 1 in both directions.
- Xavier ($2/(n_{\text{in}} + n_{\text{out}})$) suits tanh; He ($2/n_{\text{in}}$) suits ReLU.
- Residual scaling, zero-init and normalisation extend stable initialisation to very deep networks.
:::

=== POST ===
slug: vanishing-exploding-gradients
title: Vanishing and Exploding Gradients — Causes and Cures
category: deep-learning
level: Intermediate
tags: gradients, vanishing gradients, exploding gradients, gradient clipping, deep networks
summary: Gradients are products of many Jacobians, so they can shrink or grow exponentially with depth. We analyse why, how to diagnose it, and the arsenal of fixes from ReLU and initialisation to residuals, normalisation, clipping and gating.
---
For roughly two decades, deep networks had a reputation for being nearly impossible to train. The main culprit had a precise mathematical cause, identified by Sepp Hochreiter in his 1991 diploma thesis and by Bengio, Simard and Frasconi in 1994: **gradients vanish or explode** as they propagate through many layers or time steps. Understanding this problem explains the design of nearly every modern architecture.

## The mathematics

By the chain rule, the gradient of the loss with respect to an early layer's activations is a **product of Jacobians**:

$$
\frac{\partial L}{\partial\mathbf{h}_1} = \frac{\partial L}{\partial\mathbf{h}_L}\prod_{l=2}^{L}\frac{\partial\mathbf{h}_l}{\partial\mathbf{h}_{l-1}}, \qquad \frac{\partial\mathbf{h}_l}{\partial\mathbf{h}_{l-1}} = \text{diag}\big(\phi'(\mathbf{z}_l)\big)\,\mathbf{W}_l
$$

The norm of a product of $L$ matrices behaves roughly like the product of their typical scaling factors. If each Jacobian shrinks vectors by a factor $\alpha < 1$, the gradient shrinks like $\alpha^L$ — **vanishing**. If each stretches by $\alpha > 1$, it grows like $\alpha^L$ — **exploding**.

## Symptoms

**Vanishing gradients:**
- early layers barely change; their weights stay near initialisation;
- training loss plateaus early;
- in RNNs, the model cannot learn long-range dependencies.

**Exploding gradients:**
- loss spikes or suddenly becomes NaN or inf;
- weights grow very large;
- training is unstable and sensitive to the learning rate.

## Diagnosis

Log the **gradient norm per layer** during training. A healthy network has gradient norms of similar order of magnitude across layers.

```python
import torch
import torch.nn as nn

def layer_grad_norms(depth, act):
    torch.manual_seed(0)
    layers = []
    for _ in range(depth):
        layers += [nn.Linear(64, 64), act()]
    net = nn.Sequential(*layers, nn.Linear(64, 1))
    x, y = torch.randn(256, 64), torch.randn(256, 1)
    nn.functional.mse_loss(net(x), y).backward()
    linears = [m for m in net if isinstance(m, nn.Linear)]
    return [f"{m.weight.grad.norm().item():.1e}" for m in linears[::max(1, depth // 5)]]

print("sigmoid, 30 layers:", layer_grad_norms(30, nn.Sigmoid))
print("relu,    30 layers:", layer_grad_norms(30, nn.ReLU))
```

With sigmoid, early-layer gradient norms are many orders of magnitude smaller than late-layer norms.

## The cures

### 1. Better activations
ReLU's derivative is exactly 1 for active units, unlike sigmoid's maximum of 0.25. This single change made much deeper networks trainable.

### 2. Careful initialisation
Xavier and He initialisation set weight variances so the Jacobians neither shrink nor stretch signals on average (previous lecture).

### 3. Normalisation layers
Batch normalisation and layer normalisation keep activations in a well-scaled range at every layer, preventing drift in scale that compounds with depth.

### 4. Residual (skip) connections
With $\mathbf{h}_l = \mathbf{h}_{l-1} + F(\mathbf{h}_{l-1})$, the Jacobian becomes

$$
\frac{\partial\mathbf{h}_l}{\partial\mathbf{h}_{l-1}} = \mathbf{I} + \frac{\partial F}{\partial\mathbf{h}_{l-1}}
$$

The identity term gives gradients a **direct highway** back to early layers. This is why ResNets train with hundreds of layers and why every transformer uses residual connections.

### 5. Gating
LSTMs and GRUs control information flow with multiplicative gates and an additive cell state, allowing gradients to flow across many time steps (the "constant error carousel").

### 6. Gradient clipping (for explosions)
Rescale the gradient when its norm exceeds a threshold $c$:

$$
\mathbf{g} \leftarrow \mathbf{g}\cdot\min\left(1, \frac{c}{\|\mathbf{g}\|}\right)
$$

Clipping by global norm preserves the gradient's direction while bounding its size. It is standard for RNNs and transformers (often $c = 1.0$).

```python
optimizer.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

### 7. Smaller learning rates, warm-up and mixed-precision care
Explosions are often triggered by an overly aggressive learning rate early in training. Warm-up and appropriate loss scaling (for float16) reduce the risk.

## A summary table

| Problem | Main causes | Fixes |
|---|---|---|
| Vanishing | Saturating activations, small weights, long chains | ReLU/GELU, He init, residuals, normalisation, LSTM/GRU gating |
| Exploding | Large weights, recurrent loops, high learning rate | Clipping, proper init, normalisation, warm-up, lower LR |

:::note
These fixes are complementary, and modern architectures use several at once: a transformer block combines residual connections, layer normalisation, GELU activations, scaled initialisation, warm-up and gradient clipping. When you see these ingredients in an architecture, you now know *why* each one is there.
:::

:::exercise
1. For a scalar RNN $h_t = w\,h_{t-1}$, compute $\partial h_T/\partial h_0$ and explain the cases $|w| < 1$ and $|w| > 1$.
2. Add residual connections to the 30-layer sigmoid network in the code and compare gradient norms.
3. Train an RNN on a long-sequence task with and without gradient clipping; plot the loss.
:::

:::takeaway
- Gradients are products of layer Jacobians and can vanish or explode exponentially with depth or time.
- Monitor per-layer gradient norms to diagnose problems.
- ReLU, good initialisation, normalisation, residuals and gating prevent vanishing.
- Gradient clipping, warm-up and sensible learning rates control explosions.
:::

=== POST ===
slug: batch-normalization
title: Batch Normalisation: Faster, More Stable Training
category: deep-learning
level: Intermediate
tags: batch normalization, normalization, training stability, cnn
summary: BatchNorm normalises each feature using mini-batch statistics, then rescales it with learned parameters. We derive the forward pass, explain training-versus-inference behaviour, debate why it works, and list its pitfalls.
---
In 2015 Sergey Ioffe and Christian Szegedy introduced **Batch Normalisation**, and it almost immediately became standard in convolutional networks. It allowed much higher learning rates, reduced sensitivity to initialisation, and often improved accuracy. It also has subtle behaviour that causes some of the most confusing bugs in deep learning. Today we cover both.

## The operation

For a mini-batch $\{x_1, \dots, x_m\}$ of values of one feature (one neuron, or one channel in a CNN):

$$
\mu_B = \frac{1}{m}\sum_{i=1}^{m}x_i, \qquad \sigma_B^2 = \frac{1}{m}\sum_{i=1}^{m}(x_i - \mu_B)^2
$$

$$
\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \qquad y_i = \gamma\,\hat{x}_i + \beta
$$

The learned **scale** $\gamma$ and **shift** $\beta$ let the network undo the normalisation if that is optimal — so BatchNorm never reduces what the layer can represent. For convolutional layers, statistics are computed per channel over the batch **and** spatial positions.

## Training versus inference

At **training** time, BatchNorm uses the current batch's statistics. At **inference** time, a single example has no batch, and predictions must be deterministic, so BatchNorm uses **running averages** of $\mu$ and $\sigma^2$ accumulated during training:

$$
\mu_{\text{run}} \leftarrow (1 - \alpha)\,\mu_{\text{run}} + \alpha\,\mu_B
$$

:::warning
Forgetting to switch modes is a classic bug. In PyTorch, call `model.eval()` before validation or inference, and `model.train()` before training. If you evaluate in training mode, predictions depend on which other examples share the batch; if you train in eval mode, BatchNorm stops adapting. Symptoms: validation accuracy much worse than expected, or results that change with batch size.
:::

## Why does BatchNorm help?

The original paper attributed its success to reducing **internal covariate shift** — the change in each layer's input distribution as earlier layers update. Later work questioned this. Santurkar et al. (2018) showed that BatchNorm networks train well even when distribution shift is deliberately re-injected, and argued the main effect is a **smoother optimisation landscape**: gradients become more predictable (smaller Lipschitz constants), allowing larger learning rates. Other analyses emphasise that BatchNorm makes a layer's output invariant to the scale of its incoming weights, which interacts with weight decay to produce an effective learning-rate schedule.

Practical benefits are undisputed:

- **Higher learning rates** and faster convergence;
- **Less sensitivity** to initialisation;
- **Regularisation** — batch statistics add noise, sometimes reducing the need for dropout.

## Where to place it

The original paper placed BatchNorm **before** the activation: `Linear/Conv → BN → ReLU`. Placing it after the activation also works in practice. A bias in the preceding layer is redundant (BN's $\beta$ replaces it), so set `bias=False`.

```python
import torch.nn as nn

block = nn.Sequential(
    nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
    nn.BatchNorm2d(128),
    nn.ReLU(inplace=True),
)
```

## BatchNorm from scratch

```python
import numpy as np

class BatchNorm1d:
    def __init__(self, dim, momentum=0.1, eps=1e-5):
        self.gamma, self.beta = np.ones(dim), np.zeros(dim)
        self.run_mean, self.run_var = np.zeros(dim), np.ones(dim)
        self.m, self.eps, self.training = momentum, eps, True

    def __call__(self, x):
        if self.training:
            mu, var = x.mean(0), x.var(0)
            self.run_mean = (1 - self.m) * self.run_mean + self.m * mu
            self.run_var = (1 - self.m) * self.run_var + self.m * var * len(x) / (len(x) - 1)
        else:
            mu, var = self.run_mean, self.run_var
        return self.gamma * (x - mu) / np.sqrt(var + self.eps) + self.beta

bn = BatchNorm1d(3)
for _ in range(200):
    bn(np.random.default_rng().normal([5, -2, 100], [2, 0.5, 30], size=(64, 3)))
print("running mean:", bn.run_mean.round(2), " running var:", bn.run_var.round(2))
bn.training = False
print(bn(np.array([[5.0, -2.0, 100.0]])).round(3))   # ~0 after normalisation
```

## Pitfalls and limitations

- **Small batches**: with batch sizes of 1–4 (common in detection, segmentation and 3-D medical imaging), batch statistics are too noisy. Use **Group Normalisation** or **Layer Normalisation**, or synchronise statistics across GPUs (SyncBatchNorm).
- **Sequence models**: variable sequence lengths and autoregressive decoding make batch statistics awkward — transformers use LayerNorm instead.
- **Train/test discrepancy**: if the data distribution shifts, running statistics may no longer fit; re-estimating them on target-domain data is a simple adaptation technique.
- **Batch dependence** leaks information between examples in a batch, which can break certain contrastive-learning setups unless handled carefully.
- **Fine-tuning** with small batches often works better with BatchNorm layers frozen (kept in eval mode).

:::exercise
1. Train a 10-layer MLP on MNIST with and without BatchNorm at learning rates 0.01, 0.1 and 1.0. Which configurations train?
2. Evaluate a trained BatchNorm model in train mode with batch sizes 1, 8 and 256. What happens and why?
3. Derive the gradient of the BatchNorm output with respect to its input $x_i$, noting that $\mu_B$ and $\sigma_B$ also depend on $x_i$.
:::

:::takeaway
- BatchNorm normalises each feature with mini-batch mean/variance, then applies learned $\gamma, \beta$.
- Inference uses running statistics — always switch between `train()` and `eval()`.
- It enables higher learning rates and smoother optimisation; the "covariate shift" explanation is debated.
- For small batches or sequences, prefer GroupNorm or LayerNorm.
:::

=== POST ===
slug: layer-norm-group-norm-rmsnorm
title: "Beyond BatchNorm: Layer, Group, Instance and RMS Normalisation"
category: deep-learning
level: Intermediate
tags: layer normalization, rmsnorm, group normalization, transformers, normalization
summary: Normalisation layers differ only in which axes they average over — yet that choice decides where they work. We compare LayerNorm, GroupNorm, InstanceNorm and RMSNorm, and the pre-norm versus post-norm debate in transformers.
---
Batch normalisation computes statistics across the **batch**. That fails with small batches and is awkward for sequences. A family of alternatives normalises over **other axes** of the activation tensor, removing the dependence on batch size. One of them — Layer Normalisation — is in every transformer; its simplified cousin RMSNorm is in most recent large language models.

## One formula, different axes

All these methods compute

$$
\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}, \qquad y = \gamma\,\hat{x} + \beta
$$

and differ only in **which elements** share $\mu$ and $\sigma$. For an image activation tensor of shape $(N, C, H, W)$:

| Method | Statistics computed over | Depends on batch? | Typical use |
|---|---|---|---|
| BatchNorm | $N, H, W$ (per channel) | Yes | CNNs with moderate/large batches |
| LayerNorm | $C, H, W$ (per example) | No | Transformers, RNNs |
| InstanceNorm | $H, W$ (per example, per channel) | No | Style transfer, image generation |
| GroupNorm | groups of channels × $H, W$ | No | Detection/segmentation with small batches |

For a transformer activation of shape $(N, T, D)$, LayerNorm normalises each token's $D$-dimensional vector independently.

## Layer Normalisation

Ba, Kiros and Hinton (2016) normalise across the features of **each example** (each token, in a transformer). Consequences:

- identical computation in training and inference — no running statistics;
- works with batch size 1 and variable-length sequences;
- well suited to recurrent and attention models.

## Group Normalisation

Wu and He (2018) divide channels into $G$ groups (e.g. 32) and normalise within each group per example. With $G = 1$ it equals LayerNorm; with $G = C$ it equals InstanceNorm. GroupNorm's accuracy is stable across batch sizes, making it the standard in detection and segmentation, where memory-hungry high-resolution images force tiny batches.

## Instance Normalisation

Normalises each channel of each image separately, removing instance-specific contrast and colour statistics. That is exactly what style transfer wants: the "style" of an image lives largely in these per-channel statistics. **Adaptive Instance Normalisation (AdaIN)** replaces $\gamma$ and $\beta$ with statistics of a style image — the core idea behind StyleGAN's style modulation.

## RMSNorm

Zhang and Sennrich (2019) observed that re-centring (subtracting the mean) contributes little; re-scaling is what matters. **RMSNorm** drops the mean and the bias:

$$
y = \gamma\odot\frac{\mathbf{x}}{\text{RMS}(\mathbf{x})}, \qquad \text{RMS}(\mathbf{x}) = \sqrt{\frac{1}{D}\sum_{i=1}^{D}x_i^2 + \epsilon}
$$

It is simpler and cheaper while matching LayerNorm's quality, and it has become the default in many modern LLMs (the LLaMA family, among others).

```python
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps, self.weight = eps, nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

x = torch.randn(2, 5, 8) * 3 + 1                     # (batch, tokens, features)
ln, rms = nn.LayerNorm(8), RMSNorm(8)
print("LayerNorm per-token mean/std:", ln(x).mean(-1)[0, :3].detach(), ln(x).std(-1, unbiased=False)[0, :3].detach())
print("RMSNorm per-token RMS:", rms(x).pow(2).mean(-1).sqrt()[0, :3].detach())
```

## Pre-norm versus post-norm transformers

The original Transformer placed LayerNorm **after** the residual addition (**post-norm**):

$$
\mathbf{x}_{l+1} = \text{LN}\big(\mathbf{x}_l + F(\mathbf{x}_l)\big)
$$

Most modern models use **pre-norm**, normalising the input to each sub-layer:

$$
\mathbf{x}_{l+1} = \mathbf{x}_l + F\big(\text{LN}(\mathbf{x}_l)\big)
$$

Pre-norm keeps an untouched residual path from the output back to the input, so gradients flow more easily; it trains stably without long warm-up and scales to very deep models. Post-norm can achieve slightly better final quality when it trains successfully but is more fragile. Hybrids and extra normalisations (e.g. normalising queries and keys before attention, "QK-norm") are used to improve stability at scale.

:::note
Why does normalisation matter so much in transformers? Residual streams accumulate contributions from dozens of blocks; without normalisation, activation scales drift and attention logits can grow until softmax saturates. Normalisation keeps each sub-layer operating in a well-conditioned range.
:::

## Choosing a normalisation

- **CNN, batch ≥ 16**: BatchNorm.
- **CNN, small batches (detection, segmentation, medical 3-D)**: GroupNorm.
- **Transformers and RNNs**: LayerNorm or RMSNorm, usually pre-norm.
- **Style transfer / image generation**: InstanceNorm or AdaIN.
- **When in doubt for new sequence models**: pre-norm RMSNorm.

:::exercise
1. Implement GroupNorm manually for a $(N, C, H, W)$ tensor and compare with `nn.GroupNorm`.
2. Show that GroupNorm with $G = 1$ equals LayerNorm over $(C, H, W)$.
3. Train small pre-norm and post-norm transformers without warm-up at a high learning rate; compare stability.
:::

:::takeaway
- Normalisation methods differ only in the axes used for statistics.
- LayerNorm and RMSNorm are batch-independent and standard in transformers; RMSNorm drops mean-centring.
- GroupNorm handles small-batch vision; InstanceNorm/AdaIN serve style and generation.
- Pre-norm transformers train more stably than post-norm.
:::

=== POST ===
slug: dropout-regularization
title: Dropout: Regularisation by Random Deletion
category: deep-learning
level: Beginner
tags: dropout, regularization, ensembles, overfitting, mc dropout
summary: Randomly switching off neurons during training prevents co-adaptation and approximates an ensemble of exponentially many networks. We cover inverted dropout, where to apply it, its variants, and Monte Carlo dropout for uncertainty.
---
Large neural networks can memorise training data. In 2012, Hinton and colleagues proposed a remarkably simple remedy: during training, **randomly drop** each neuron (set its output to zero) with some probability. This technique, **dropout**, was a key ingredient in AlexNet's ImageNet victory and became one of the most widely used regularisers in deep learning.

## The mechanism

During training, for each example and each unit, sample a mask $m_i \sim \text{Bernoulli}(1 - p)$, where $p$ is the **drop probability**:

$$
\tilde{h}_i = m_i\,h_i
$$

At test time, all units are active. To keep the **expected** activation the same in both phases, we must scale. Modern implementations use **inverted dropout**, scaling during training so that inference needs no change:

$$
\tilde{h}_i = \frac{m_i}{1 - p}\,h_i, \qquad \mathbb{E}[\tilde{h}_i] = h_i
$$

```python
import torch

def dropout(h, p=0.5, training=True):
    if not training or p == 0:
        return h
    mask = (torch.rand_like(h) > p).float()
    return h * mask / (1 - p)

h = torch.ones(8)
print(dropout(h, 0.5))                     # some zeros, others scaled to 2.0
print(dropout(torch.ones(100000), 0.5).mean())   # ~1.0 in expectation
```

## Why does it work?

### 1. Preventing co-adaptation
Without dropout, a neuron can rely on specific other neurons to correct its mistakes, forming fragile, co-adapted feature detectors. With dropout, any neuron may disappear, so each must be useful on its own and in many different combinations — encouraging robust, redundant features.

### 2. An implicit ensemble
A network with $n$ droppable units defines $2^n$ "thinned" sub-networks that share weights. Each training step trains one random sub-network. At test time, using all units with (inverted) scaling approximates the **geometric average** of the predictions of this exponentially large ensemble — a cheap form of bagging.

### 3. Noise as regularisation
For linear models, dropout on inputs is approximately equivalent to an adaptive L2 penalty that depends on feature variance. More generally, injecting noise during training discourages the model from depending on fine details of individual examples.

## Where and how much?

- **Fully connected layers**: $p = 0.5$ was the classic choice for large hidden layers (AlexNet, VGG).
- **Input layer**: small $p$ (0.1–0.2) — dropping too many inputs destroys information.
- **Convolutional layers**: standard dropout is less effective because neighbouring pixels are correlated; use **spatial dropout** (drop whole channels) or **DropBlock** (drop contiguous regions), or rely on data augmentation.
- **Transformers**: dropout of about 0.1 on attention weights, residual branches and embeddings is common for moderate-sized models; very large language models trained on huge datasets for a single epoch often use little or no dropout, because they are not in an overfitting regime.
- **RNNs**: apply the **same** dropout mask at every time step (variational dropout) rather than resampling per step.

## Variants

| Variant | Drops | Use |
|---|---|---|
| Standard dropout | individual activations | MLP layers |
| Spatial dropout | entire feature maps | CNNs |
| DropBlock | contiguous spatial regions | CNNs |
| DropConnect | individual weights | Regularising weights directly |
| Stochastic depth / DropPath | entire residual blocks | Deep ResNets, vision transformers |
| Attention dropout | attention probabilities | Transformers |

**Stochastic depth** deserves special mention: randomly skipping whole residual blocks during training regularises very deep networks and speeds training; it is standard in modern vision transformers and ConvNeXt.

## Train/eval mode

Dropout behaves differently in training and inference, so — as with BatchNorm — you must call `model.train()` and `model.eval()` correctly. Forgetting `eval()` makes predictions noisy and systematically worse.

## Monte Carlo dropout for uncertainty

Gal and Ghahramani (2016) showed that keeping dropout **active at test time** and averaging many stochastic forward passes approximates Bayesian inference over the weights. The spread of the predictions gives an estimate of **model uncertainty**.

```python
import torch, torch.nn as nn

model = nn.Sequential(nn.Linear(1, 128), nn.ReLU(), nn.Dropout(0.2),
                      nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 1))
# ... train on data x in [-2, 2] ...

def mc_predict(model, x, n=100):
    model.train()                             # keep dropout ON
    with torch.no_grad():
        preds = torch.stack([model(x) for _ in range(n)])
    return preds.mean(0), preds.std(0)        # prediction and uncertainty

x_query = torch.tensor([[0.0], [5.0]])        # in-distribution vs far outside
mean, std = mc_predict(model, x_query)
print(mean.squeeze(), std.squeeze())          # after training, std should be larger at x = 5
```

(If your model contains BatchNorm, switch only the dropout layers to training mode.) MC dropout is a practical, inexpensive way to flag inputs on which the model should not be trusted — valuable in medical and humanitarian decision support.

:::exercise
1. Show that inverted dropout preserves the expected activation.
2. Train an over-parameterised MLP on 1,000 MNIST examples with dropout 0, 0.2 and 0.5. Compare the gap between training and validation accuracy.
3. Implement spatial dropout for a CNN feature map of shape $(N, C, H, W)$.
:::

:::takeaway
- Dropout zeroes units randomly during training; inverted scaling keeps expectations unchanged.
- It prevents co-adaptation and approximates an ensemble of $2^n$ sub-networks.
- Use channel/block dropout for CNNs and stochastic depth for deep residual networks.
- Keep dropout on at test time (MC dropout) to estimate uncertainty.
:::

=== POST ===
slug: regularization-in-deep-learning
title: Regularisation in Deep Learning: Weight Decay, Early Stopping, Augmentation and More
category: deep-learning
level: Intermediate
tags: regularization, weight decay, early stopping, data augmentation, mixup
summary: Deep networks can memorise anything, yet generalise well when regularised properly. We survey the toolkit — weight decay, early stopping, data augmentation, mixup and cutmix, label smoothing — and how to combine them.
---
Zhang et al. (2017) showed that standard image-classification networks can reach 100% training accuracy on data with **completely random labels**. These networks have more than enough capacity to memorise. So why do they generalise on real data — and how do we help them? The answer is a combination of the implicit biases of training and a toolkit of explicit **regularisers**. This lecture surveys that toolkit.

## 1. Weight decay (L2 regularisation)

Shrink weights towards zero at every step:

$$
\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \eta\left(\nabla L + \lambda\boldsymbol{\theta}\right)
$$

Small weights mean smoother functions that are less sensitive to input perturbations. For SGD this equals an L2 penalty; for Adam, use **decoupled** weight decay (AdamW). Typical values: $10^{-4}$–$5 \times 10^{-4}$ for SGD on CNNs; 0.01–0.1 for AdamW on transformers. Usually exclude biases and normalisation parameters.

## 2. Early stopping

Monitor validation loss and stop when it stops improving, keeping the best checkpoint.

```python
best, patience, bad_epochs = float("inf"), 10, 0
for epoch in range(max_epochs):
    train_one_epoch(model)
    val = evaluate(model)
    if val < best - 1e-4:
        best, bad_epochs = val, 0
        torch.save(model.state_dict(), "best.pt")
    else:
        bad_epochs += 1
        if bad_epochs >= patience:
            break
model.load_state_dict(torch.load("best.pt"))
```

For gradient descent on a quadratic loss, early stopping is provably related to L2 regularisation: the number of steps plays the role of the inverse penalty strength. It is cheap and nearly universal.

## 3. Data augmentation

The most powerful regulariser in vision and speech: create new training examples by applying label-preserving transformations. It encodes **invariances** we know the task should have.

- **Images**: random crops, flips, rotations, colour jitter, blur, random erasing; learned policies like RandAugment and TrivialAugment.
- **Audio**: time shifting, speed perturbation, adding background noise, SpecAugment (masking time and frequency bands).
- **Text**: back-translation, synonym replacement, random deletion (use carefully — meaning changes easily).
- **Tabular**: harder; noise injection, or synthetic data generation, with caution.

:::warning
Augmentations must preserve the label. Flipping a photo of a cat horizontally is fine; flipping a handwritten "6" vertically makes a "9"; horizontally flipping an X-ray swaps the patient's left and right sides, which matters for diagnosis. Design augmentations with domain experts.
:::

## 4. Mixup and CutMix

**Mixup** (Zhang et al., 2018) trains on convex combinations of pairs of examples and their labels:

$$
\tilde{\mathbf{x}} = \lambda\mathbf{x}_i + (1 - \lambda)\mathbf{x}_j, \qquad \tilde{\mathbf{y}} = \lambda\mathbf{y}_i + (1 - \lambda)\mathbf{y}_j, \qquad \lambda \sim \text{Beta}(\alpha, \alpha)
$$

It encourages linear behaviour between training examples, improving generalisation and calibration. **CutMix** pastes a rectangular patch from one image into another, mixing labels in proportion to the patch area — it keeps images locally realistic.

```python
import torch
import numpy as np

def mixup(x, y_onehot, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0))
    return lam * x + (1 - lam) * x[idx], lam * y_onehot + (1 - lam) * y_onehot[idx]
```

## 5. Label smoothing

Replace hard one-hot targets with soft ones (e.g. 0.9 for the true class, the rest spread over others). It prevents logits from growing without bound and typically improves calibration and accuracy.

## 6. Dropout and stochastic depth

Covered in the previous lecture — noise injected into activations or whole blocks.

## 7. Other techniques

- **Noise injection** into inputs, weights or gradients.
- **Parameter sharing** — convolution shares weights across positions, a powerful built-in regulariser.
- **Transfer learning** — starting from pretrained weights acts as a strong prior.
- **Ensembling and weight averaging** — averaging weights along the training trajectory (Stochastic Weight Averaging, exponential moving averages of weights) finds flatter solutions and often generalises better.
- **Sharpness-Aware Minimisation (SAM)** — explicitly seeks parameters whose whole neighbourhood has low loss.

## Implicit regularisation

Even without explicit regularisers, **SGD** tends to find solutions that generalise: from small random initialisations, gradient descent on over-parameterised models favours low-norm, "simple" solutions, and SGD's noise biases it towards flat minima. This implicit bias is a major reason over-parameterised networks generalise despite their capacity — and why the **double descent** curve appears.

## Putting it together

| Regime | Emphasise |
|---|---|
| Small dataset, vision | Pretrained backbone, strong augmentation, weight decay, early stopping |
| Medium dataset, from scratch | Augmentation (+ mixup/cutmix), weight decay, label smoothing, stochastic depth |
| Huge dataset, single epoch (LLMs) | Little dropout; weight decay; the data itself regularises |
| Tabular deep learning | Weight decay, dropout, early stopping — and compare with gradient boosting |

:::tip
Regularisers interact. Strong augmentation may make dropout unnecessary; more weight decay may require a longer schedule. Add regularisation incrementally and validate each addition. And remember: **more real data** is the best regulariser of all.
:::

:::exercise
1. Train a ResNet-18 on 10% of CIFAR-10 with (a) no regularisation, (b) augmentation, (c) augmentation + mixup, (d) all of the above + weight decay and label smoothing. Tabulate test accuracy.
2. Replicate the random-label experiment on a small subset of MNIST. How many epochs does memorisation take compared with real labels?
3. Implement CutMix and visualise a batch of mixed images with their mixed labels.
:::

:::takeaway
- Deep networks can memorise random labels; generalisation relies on regularisation and implicit biases.
- Weight decay, early stopping and data augmentation are the core tools.
- Mixup, CutMix and label smoothing improve generalisation and calibration.
- Weight averaging, SAM and pretrained priors add further gains; more data beats all.
:::
