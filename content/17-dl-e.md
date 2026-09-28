=== POST ===
slug: graph-neural-networks
title: Graph Neural Networks: Learning on Relational Data
category: deep-learning
level: Advanced
tags: gnn, graphs, message passing, gcn, graph attention
summary: Molecules, social networks, road maps and knowledge graphs are graphs. GNNs learn from them by passing messages between neighbours. We derive message passing, GCN and GAT layers, and survey node, edge and graph-level tasks.
---
Images live on grids and text on sequences, but much of the world's data is **relational**: atoms bonded in molecules, people connected in social networks, roads linking towns, citations between papers, transactions between accounts. Such data forms a **graph** $G = (V, E)$ — nodes with features, connected by edges. Graphs have no fixed ordering or size, so CNNs and RNNs do not apply directly. **Graph Neural Networks (GNNs)** extend deep learning to this setting.

## Graph basics

- **Adjacency matrix** $\mathbf{A} \in \{0,1\}^{n \times n}$: $A_{ij} = 1$ if nodes $i$ and $j$ are connected.
- **Degree matrix** $\mathbf{D}$: diagonal, $D_{ii} = \sum_j A_{ij}$.
- **Node features** $\mathbf{X} \in \mathbb{R}^{n \times d}$; optionally edge features.

A key requirement: the output should not depend on the arbitrary order in which we number the nodes. Node-level outputs must be **permutation equivariant**; graph-level outputs must be **permutation invariant**.

## Message passing: the unifying framework

Almost all GNNs follow the **message-passing** paradigm (Gilmer et al., 2017). In each layer, every node:

1. **Collects messages** from its neighbours;
2. **Aggregates** them with a permutation-invariant function (sum, mean, max);
3. **Updates** its own representation.

$$
\mathbf{h}_v^{(l+1)} = \text{UPDATE}\left(\mathbf{h}_v^{(l)},\; \text{AGG}_{u \in \mathcal{N}(v)}\,\text{MSG}\big(\mathbf{h}_v^{(l)}, \mathbf{h}_u^{(l)}, \mathbf{e}_{uv}\big)\right)
$$

After $k$ layers, each node's representation depends on its **$k$-hop neighbourhood** — analogous to a CNN's growing receptive field.

## Graph Convolutional Network (GCN)

Kipf and Welling's GCN (2017) uses a normalised average of neighbour features, including a self-loop ($\tilde{\mathbf{A}} = \mathbf{A} + \mathbf{I}$):

$$
\mathbf{H}^{(l+1)} = \sigma\left(\tilde{\mathbf{D}}^{-1/2}\tilde{\mathbf{A}}\tilde{\mathbf{D}}^{-1/2}\mathbf{H}^{(l)}\mathbf{W}^{(l)}\right)
$$

The symmetric normalisation prevents high-degree nodes from dominating and keeps feature scales stable. The weight matrix $\mathbf{W}^{(l)}$ is shared across all nodes — weight sharing, as in convolution.

## Graph Attention Network (GAT)

GCN weights neighbours by fixed degree-based coefficients. **GAT** (Veličković et al., 2018) learns them with attention:

$$
\alpha_{vu} = \text{softmax}_{u \in \mathcal{N}(v)}\Big(\text{LeakyReLU}\big(\mathbf{a}^\top[\mathbf{W}\mathbf{h}_v \,\|\, \mathbf{W}\mathbf{h}_u]\big)\Big), \qquad \mathbf{h}_v' = \sigma\Big(\sum_{u \in \mathcal{N}(v)}\alpha_{vu}\mathbf{W}\mathbf{h}_u\Big)
$$

Different neighbours can matter differently, and multi-head attention stabilises learning. (A transformer is, in fact, a GAT on a fully connected graph of tokens.)

## Other important layers

- **GraphSAGE** — samples a fixed number of neighbours and aggregates them, enabling inductive learning on huge graphs (used for recommendation at web scale).
- **GIN** (Graph Isomorphism Network) — uses sum aggregation and an MLP; provably as expressive as the Weisfeiler–Lehman graph isomorphism test, the theoretical limit for standard message passing.
- **Edge-conditioned / MPNN layers** — use edge features (bond types, distances) in messages; common in chemistry.
- **Equivariant GNNs** — respect 3-D rotations and translations for molecules and physics.

## A GCN from scratch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNLayer(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.lin = nn.Linear(d_in, d_out, bias=False)
    def forward(self, X, A):
        A_hat = A + torch.eye(A.size(0))                      # add self-loops
        d = A_hat.sum(1)
        D_inv_sqrt = torch.diag(d.pow(-0.5))
        return D_inv_sqrt @ A_hat @ D_inv_sqrt @ self.lin(X)

class GCN(nn.Module):
    def __init__(self, d_in, d_hid, n_cls):
        super().__init__()
        self.l1, self.l2 = GCNLayer(d_in, d_hid), GCNLayer(d_hid, n_cls)
    def forward(self, X, A):
        return self.l2(F.dropout(F.relu(self.l1(X, A)), 0.5, self.training), A)

# Two communities of 20 nodes with dense internal links and a few cross links
torch.manual_seed(0)
n = 40; labels = torch.tensor([0] * 20 + [1] * 20)
A = (torch.rand(n, n) < 0.02).float()
same = labels[:, None] == labels[None, :]
A = torch.maximum(A, ((torch.rand(n, n) < 0.3) & same).float())
A = torch.triu(A, 1); A = A + A.T                             # symmetric, no self-loops
X = torch.randn(n, 8)                                         # uninformative features
train_mask = torch.zeros(n, dtype=torch.bool); train_mask[[0, 1, 20, 21]] = True   # 4 labels only

model = GCN(8, 16, 2); opt = torch.optim.Adam(model.parameters(), 0.01, weight_decay=5e-4)
for epoch in range(200):
    model.train(); opt.zero_grad()
    loss = F.cross_entropy(model(X, A)[train_mask], labels[train_mask]); loss.backward(); opt.step()
model.eval()
print("accuracy on all nodes:", (model(X, A).argmax(1) == labels).float().mean().item())
```

Even with random node features and only four labelled nodes, the GCN classifies most nodes correctly by exploiting the **graph structure** — a semi-supervised setting where GNNs shine. In practice, use libraries such as **PyTorch Geometric** or **DGL**, which handle sparse adjacency efficiently.

## Tasks

| Level | Example | Readout |
|---|---|---|
| Node | Classify users, papers, proteins | Node embeddings |
| Edge | Predict links: friendships, drug–target interactions, recommendations | Pairs of node embeddings |
| Graph | Predict molecular toxicity or solubility | Pool all nodes (sum/mean/attention) |

## Challenges

- **Over-smoothing** — after many layers, node representations become indistinguishable, so GNNs are usually shallow (2–4 layers); residual connections and normalisation help.
- **Over-squashing** — information from exponentially growing neighbourhoods is compressed into fixed-size vectors, limiting long-range reasoning; graph rewiring and graph transformers address this.
- **Expressivity limits** — standard message passing cannot distinguish certain non-isomorphic graphs.
- **Scalability** — graphs with billions of edges need neighbour sampling and mini-batching.

## Applications

Drug discovery and molecular property prediction, materials science, traffic and travel-time forecasting (used in widely deployed map services), fraud detection in transaction networks, recommendation, knowledge-graph completion, physics simulation, and protein-structure pipelines where graph-like reasoning over residues plays a central role.

:::exercise
1. Show that the GCN layer is permutation equivariant: permuting node order permutes the output rows identically.
2. Stack 2, 4, 8 and 16 GCN layers on the toy graph and measure accuracy and the variance of node embeddings. Observe over-smoothing.
3. Use PyTorch Geometric to train a GCN and a GAT on the Cora citation dataset and compare accuracy.
:::

:::takeaway
- GNNs learn from graphs by message passing: collect, aggregate (permutation-invariantly) and update.
- GCN averages normalised neighbour features; GAT learns attention weights; GIN maximises expressivity.
- Tasks exist at node, edge and graph level; structure alone can carry strong signal.
- Over-smoothing, over-squashing and scale are the main challenges.
:::

=== POST ===
slug: knowledge-distillation
title: Knowledge Distillation: Teaching Small Models with Large Ones
category: deep-learning
level: Intermediate
tags: distillation, model compression, soft targets, teacher student, efficiency
summary: A large "teacher" model's soft predictions contain rich information that can train a much smaller "student". We derive the distillation loss with temperature, discuss dark knowledge, and survey feature and LLM distillation.
---
The most accurate models are often too large, slow or expensive to deploy on a phone, in a browser or at scale. **Knowledge distillation**, popularised by Hinton, Vinyals and Dean (2015), trains a compact **student** model to mimic a large **teacher**. The student often performs far better than the same small model trained on labels alone. Distillation is behind many efficient deployed models, including compact language models such as DistilBERT.

## Dark knowledge in soft targets

A one-hot label for an image of a "2" says only "this is a 2". A trained teacher's output might say: 2 with probability 0.90, 3 with 0.06, 7 with 0.03, and nearly zero for others. These small probabilities encode **similarity structure** — this "2" looks a bit like a 3 and a 7, and nothing like a 4. Hinton called this **dark knowledge**. Learning from full distributions gives the student far more information per example than hard labels.

## Temperature

Softmax outputs of a confident teacher are nearly one-hot, hiding the small probabilities. We soften them with a **temperature** $T > 1$:

$$
p_i^{(T)} = \frac{\exp(z_i/T)}{\sum_j\exp(z_j/T)}
$$

Higher $T$ produces a softer distribution that reveals the relative ranking of wrong classes.

## The distillation loss

The student is trained on a weighted combination of two terms:

$$
\mathcal{L} = \alpha\,T^2\,D_{\text{KL}}\left(\mathbf{p}_{\text{teacher}}^{(T)}\,\Big\|\,\mathbf{p}_{\text{student}}^{(T)}\right) + (1 - \alpha)\,\text{CE}\left(\mathbf{y},\, \mathbf{p}_{\text{student}}^{(1)}\right)
$$

- The first term matches the softened teacher distribution.
- The second term is ordinary cross-entropy with the true labels.
- The factor $T^2$ keeps the gradient magnitude of the soft term roughly independent of $T$, since softened gradients scale as $1/T^2$.

Typical settings: $T \in [2, 10]$, $\alpha \in [0.5, 0.9]$.

```python
import torch
import torch.nn.functional as F

def distillation_loss(student_logits, teacher_logits, labels, T=4.0, alpha=0.7):
    soft = F.kl_div(F.log_softmax(student_logits / T, dim=-1),
                    F.softmax(teacher_logits / T, dim=-1),
                    reduction="batchmean") * (T * T)
    hard = F.cross_entropy(student_logits, labels)
    return alpha * soft + (1 - alpha) * hard

# Training loop sketch
teacher.eval()
for xb, yb in loader:
    with torch.no_grad():
        t_logits = teacher(xb)                 # teacher is frozen
    s_logits = student(xb)
    loss = distillation_loss(s_logits, t_logits, yb)
    opt.zero_grad(); loss.backward(); opt.step()
```

## Why does it work so well?

- **Richer supervision** — each example carries a full probability vector rather than one bit of class information.
- **Regularisation** — soft targets act like label smoothing informed by real class similarity.
- **Easier function** — the teacher has already smoothed away label noise and found a simpler decision function that the student can approximate.
- **Unlabelled data** — the teacher can label unlimited unlabelled (or augmented) data for the student.

## Variants

- **Feature (hint) distillation** — FitNets match intermediate representations of teacher and student via a small projection; attention-transfer matches attention maps.
- **Relational distillation** — match the similarity structure between examples rather than individual outputs.
- **Self-distillation** — a model distils into a copy of the same architecture, which surprisingly often improves it; **born-again networks** iterate this.
- **Online / mutual distillation** — several students teach each other during training, without a pre-trained teacher.
- **Data-free distillation** — synthesise inputs when the original training data cannot be shared.

## Distillation for language models

For large language models, distillation takes several forms:

1. **Logit distillation** — match the teacher's next-token distributions (as in DistilBERT, which retained most of BERT's performance with about 40% fewer parameters and higher speed).
2. **Sequence-level distillation** — train the student on text *generated* by the teacher (e.g. teacher-written answers or reasoning traces). Many small instruction-following and reasoning models are trained this way.
3. **On-policy distillation** — the student generates, and the teacher provides token-level feedback on the student's own outputs, reducing the mismatch between training and use.

:::warning
Distillation copies the teacher's **errors and biases** along with its knowledge, and the student may learn the teacher's style without its underlying competence — for example, reproducing confident-sounding reasoning that is wrong. Evaluate the student independently on realistic, held-out data. Also check model licences: some providers' terms restrict using their outputs to train competing models.
:::

## Where distillation fits among compression methods

| Method | Reduces | Needs retraining |
|---|---|---|
| Distillation | Architecture size (new smaller model) | Yes (train the student) |
| Pruning | Parameters/structures in the same model | Usually fine-tuning |
| Quantisation | Bits per parameter | Optional (calibration or QAT) |
| Low-rank factorisation | Matrix sizes | Usually fine-tuning |

They combine well: distil into a small student, then prune and quantise it for deployment.

:::exercise
1. Train a small CNN on CIFAR-10 with labels only, then distil it from a ResNet-18 teacher. Compare accuracy.
2. Vary the temperature $T \in \{1, 2, 4, 8, 16\}$ and plot the student's accuracy.
3. Show mathematically that the gradient of the soft term with respect to student logits scales as $1/T^2$ for large $T$, justifying the $T^2$ factor.
:::

:::takeaway
- Distillation trains a small student to match a large teacher's softened output distribution.
- Temperature reveals "dark knowledge" in the relative probabilities of wrong classes; scale the soft loss by $T^2$.
- Feature, relational, self- and sequence-level distillation extend the idea.
- Students inherit teachers' errors; evaluate independently and respect licences.
:::

=== POST ===
slug: pruning-and-quantization
title: "Model Compression: Pruning and Quantisation"
category: deep-learning
level: Advanced
tags: pruning, quantization, compression, edge ai, efficiency
summary: Neural networks are highly redundant. We remove unnecessary weights with pruning, represent the rest with fewer bits via quantisation, and discuss the lottery ticket hypothesis and deployment on edge devices.
---
Trained networks contain astonishing redundancy. Han et al. (2015) showed that most weights of AlexNet and VGG could be removed with little loss in accuracy. Compressing models matters wherever resources are limited: smartphones, microcontrollers in the field, browsers, or servers handling millions of requests where every millisecond and watt counts. The two main tools are **pruning** — removing parameters — and **quantisation** — storing and computing with fewer bits.

## Pruning

### Unstructured pruning
Remove individual weights, most commonly those with the smallest magnitude (**magnitude pruning**). The resulting sparse matrices can reach very high sparsity (80–95% on many networks) with small accuracy loss when combined with fine-tuning. But **unstructured sparsity rarely speeds up** standard hardware, which is optimised for dense matrix multiplication; it mainly reduces storage (with sparse formats) unless special kernels or hardware are used.

### Structured pruning
Remove entire **channels, filters, attention heads or layers**. The model becomes genuinely smaller and faster on ordinary hardware, at the cost of lower achievable sparsity. Importance can be measured by the norm of a filter's weights, the scaling factors of BatchNorm layers (network slimming), or the effect on the loss (Taylor-expansion criteria).

A middle ground, **N:M semi-structured sparsity** (e.g. 2 non-zeros in every block of 4), is accelerated by recent GPUs.

### The pruning workflow
1. Train a dense model.
2. Prune a fraction of weights by some importance criterion.
3. **Fine-tune** to recover accuracy.
4. Repeat (iterative pruning usually beats one-shot pruning at high sparsity).

```python
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

model = nn.Sequential(nn.Linear(784, 300), nn.ReLU(), nn.Linear(300, 100), nn.ReLU(), nn.Linear(100, 10))
params = [(m, "weight") for m in model if isinstance(m, nn.Linear)]
prune.global_unstructured(params, pruning_method=prune.L1Unstructured, amount=0.8)   # prune 80% globally

total = sum(m.weight.nelement() for m, _ in params)
zeros = sum((m.weight == 0).sum().item() for m, _ in params)
print(f"global sparsity: {zeros / total:.1%}")
for m, _ in params:
    prune.remove(m, "weight")                  # make pruning permanent (bake mask into weights)
```

### The lottery ticket hypothesis
Frankle and Carbin (2019) found that dense networks contain small sub-networks ("winning tickets") that, when **reset to their original initialisation** and trained in isolation, match the full network's accuracy. This suggests over-parameterisation helps optimisation by providing many candidate sub-networks, rather than being needed for the final function. For large networks, rewinding to an early training checkpoint rather than initialisation works better.

## Quantisation

Replace 32-bit floating-point numbers with low-bit representations, typically 8-bit integers (INT8) or even 4-bit. Benefits: 4–8× smaller models, less memory bandwidth, and faster integer arithmetic on CPUs, mobile NPUs and GPUs.

**Affine (asymmetric) quantisation** maps a real value $x$ to an integer $q$ with a **scale** $s$ and **zero point** $z$:

$$
q = \text{clamp}\left(\text{round}\left(\frac{x}{s}\right) + z,\; q_{\min},\; q_{\max}\right), \qquad \hat{x} = s\,(q - z)
$$

For $b$ bits and a real range $[x_{\min}, x_{\max}]$: $s = \frac{x_{\max} - x_{\min}}{2^b - 1}$. **Symmetric** quantisation fixes $z = 0$.

```python
import numpy as np

def quantize(x, bits=8):
    qmin, qmax = 0, 2**bits - 1
    s = (x.max() - x.min()) / (qmax - qmin)
    z = np.round(qmin - x.min() / s)
    q = np.clip(np.round(x / s + z), qmin, qmax).astype(np.int32)
    return q, s, z

def dequantize(q, s, z):
    return s * (q - z)

w = np.random.default_rng(0).normal(0, 0.05, 10_000)
for bits in [8, 4, 2]:
    q, s, z = quantize(w, bits)
    err = np.abs(dequantize(q, s, z) - w).mean()
    print(f"{bits}-bit: mean abs error {err:.5f} ({err / np.abs(w).mean():.1%} of mean |w|)")
```

### Granularity
- **Per-tensor**: one scale for the whole tensor — simple but hurt by outliers.
- **Per-channel**: one scale per output channel — standard for weights.
- **Per-group**: one scale per block of (e.g.) 64–128 weights — standard for 4-bit LLM weights.

### Post-training quantisation vs quantisation-aware training
- **Post-training quantisation (PTQ)**: quantise a trained model directly. **Dynamic** PTQ quantises weights ahead of time and activations on the fly; **static** PTQ uses a small **calibration set** to fix activation ranges. Fast and often sufficient for INT8.
- **Quantisation-aware training (QAT)**: simulate quantisation during training ("fake quantisation") and backpropagate with the **straight-through estimator** (treat rounding as identity in the backward pass). Recovers accuracy at low bit-widths.

### Quantising large language models
LLM activations contain rare, very large **outlier features** that break naive INT8 quantisation. Methods such as LLM.int8() (handles outliers in higher precision), SmoothQuant (migrates difficulty from activations to weights), GPTQ and AWQ (accurate 4-bit weight-only quantisation using second-order or activation-aware criteria) make it possible to run large models on consumer GPUs and laptops with modest quality loss. Formats such as GGUF package quantised models for local inference.

## Choosing a compression strategy

| Goal | Recommended |
|---|---|
| Faster CPU/mobile inference | INT8 static PTQ; structured pruning; distillation to a small architecture |
| Fit an LLM on a small GPU | 4-bit weight-only quantisation (GPTQ/AWQ) |
| Microcontroller (TinyML) | Small architecture + INT8 QAT |
| Maximum accuracy at low bits | QAT, per-channel/group scales |

:::tip
Always measure what matters on the **target device** — latency, memory, energy and accuracy on realistic data. A 90%-sparse model that runs no faster on your phone is not a win; a quantised model that fails on a minority dialect is not acceptable.
:::

:::exercise
1. Prune an MNIST MLP to 50%, 80%, 95% and 99% sparsity with and without fine-tuning; plot accuracy against sparsity.
2. Apply PyTorch dynamic INT8 quantisation (`torch.ao.quantization.quantize_dynamic`) to an LSTM or transformer and compare size and CPU latency.
3. Explain why a single large outlier value harms per-tensor quantisation, using the scale formula.
:::

:::takeaway
- Unstructured pruning reaches high sparsity but needs special kernels for speed; structured pruning directly shrinks computation.
- The lottery ticket hypothesis suggests over-parameterisation aids optimisation.
- Quantisation maps floats to low-bit integers with a scale and zero point; per-channel/group scales handle ranges.
- PTQ is quick; QAT recovers accuracy at low bits; LLMs need outlier-aware methods.
:::

=== POST ===
slug: neural-architecture-search
title: Neural Architecture Search and Automated Machine Learning
category: deep-learning
level: Advanced
tags: nas, automl, architecture design, efficientnet, search
summary: Can algorithms design better networks than humans? We review search spaces, reinforcement-learning and evolutionary search, differentiable NAS, weight sharing and hardware-aware search — and the lessons of the NAS era.
---
Designing neural architectures — choosing layer types, widths, depths, connections — was long a craft of human intuition and trial and error. **Neural Architecture Search (NAS)** automates this: an algorithm searches over a space of possible architectures for the one that performs best. NAS produced several state-of-the-art models, most famously the EfficientNet family, and it belongs to the broader field of **AutoML**, which also automates hyperparameter tuning and feature engineering.

## The three components of NAS

Elsken, Metzen and Hutter (2019) describe every NAS method by three choices:

1. **Search space** — which architectures can be expressed.
2. **Search strategy** — how to explore that space.
3. **Performance estimation** — how to evaluate a candidate cheaply.

## Search spaces

- **Global / chain-structured**: choose each layer's type and hyperparameters in sequence.
- **Cell-based**: search for a small repeated building block (a "cell") and stack it; this made search tractable and transferable (NASNet learned cells on CIFAR-10 and transferred them to ImageNet).
- **Hierarchical / macro spaces**: choose depths, widths and resolutions per stage.

The search space encodes strong human priors. A space made only of good building blocks makes even random search competitive — an important and humbling lesson.

## Search strategies

### Reinforcement learning
Zoph and Le (2017) used an RNN **controller** to generate architecture descriptions, trained with policy gradients using validation accuracy as the reward. It produced excellent architectures but required enormous compute (hundreds of GPUs for weeks).

### Evolutionary algorithms
Maintain a population of architectures; mutate the good ones (add a layer, change a kernel); select by fitness. **Regularized (aging) evolution** (Real et al., 2019) — removing the oldest rather than the worst individuals — produced AmoebaNet, matching RL-based results.

### Bayesian optimisation
Model architecture performance with surrogates (e.g. with graph kernels or neural predictors) to choose promising candidates.

### Differentiable NAS (DARTS)
Liu et al. (2019) relaxed the discrete choice of operation on each edge into a softmax-weighted mixture:

$$
\bar{o}(\mathbf{x}) = \sum_{o \in \mathcal{O}}\frac{\exp(\alpha_o)}{\sum_{o'}\exp(\alpha_{o'})}\,o(\mathbf{x})
$$

Architecture parameters $\alpha$ and network weights are optimised jointly by gradient descent (a bilevel problem), and the strongest operation on each edge is kept at the end. This cut search cost to a few GPU-days, though DARTS can be unstable (e.g. collapsing towards parameter-free skip connections).

## Cheap performance estimation

Training every candidate to convergence is prohibitively expensive. Shortcuts:

- **Low-fidelity estimates** — fewer epochs, smaller images, data subsets (combine with Hyperband).
- **Learning-curve extrapolation** — stop unpromising runs early.
- **Weight sharing / one-shot supernets** — train one large "supernet" containing all candidate architectures as sub-graphs; evaluate candidates by inheriting its weights (ENAS, Once-for-All). Fast, but inherited-weight rankings can correlate imperfectly with stand-alone performance.
- **Zero-cost proxies** — score untrained networks by properties at initialisation (gradient statistics, activation patterns).

## Hardware-aware NAS

For deployment, accuracy is not the only objective. **Multi-objective** NAS includes latency, energy or memory on a specific device:

$$
\max_{a}\;\text{Acc}(a)\cdot\left(\frac{\text{Latency}(a)}{\text{target}}\right)^w
$$

(MnasNet's reward). **MobileNetV3** and **EfficientNet**'s base network were found with such searches. Once-for-All trains one supernet and then extracts specialised sub-networks for many devices without retraining.

## EfficientNet's compound scaling

Tan and Le (2019) combined NAS with a principled scaling rule. After searching for a small baseline (EfficientNet-B0), they scaled depth $d$, width $w$ and resolution $r$ together:

$$
d = \alpha^\phi, \quad w = \beta^\phi, \quad r = \gamma^\phi, \qquad \alpha\beta^2\gamma^2 \approx 2
$$

so each increment of $\phi$ roughly doubles compute. The family B0–B7 achieved excellent accuracy–efficiency trade-offs.

## Lessons from the NAS era

:::note
1. **Baselines matter.** Carefully tuned random search within a good search space is often competitive with sophisticated methods (Li & Talwalkar, 2019).
2. **Training recipes matter as much as architectures.** Improvements attributed to new architectures sometimes vanish when baselines receive the same modern training tricks (augmentation, schedules, longer training).
3. **Scaling beats searching.** In the transformer era, simple architectures scaled up with more data and compute have often outperformed intricate searched designs — but NAS remains valuable for **efficient** models under tight hardware constraints.
:::

## AutoML in practice

For most practitioners, AutoML means tools that automate model selection and tuning: Auto-sklearn, AutoGluon, H2O AutoML, FLAML, cloud AutoML services and Optuna-based pipelines. AutoGluon, for example, combines many model families with stacking and often provides a very strong tabular baseline with a few lines of code. Use AutoML as a **baseline and accelerator**, not a substitute for understanding the problem, the data and the evaluation.

:::exercise
1. Implement random search over a small CNN search space (number of blocks, channels, kernel sizes) on CIFAR-10 with a 2-epoch budget per candidate. How well do 2-epoch rankings predict 20-epoch rankings?
2. Explain the bilevel optimisation in DARTS and why it can collapse towards skip connections.
3. Run AutoGluon or FLAML on a tabular dataset and compare it with your manually tuned gradient-boosting model.
:::

:::takeaway
- NAS = search space + search strategy + performance estimation.
- Strategies include RL, evolution, Bayesian optimisation and differentiable search (DARTS).
- Weight sharing and low-fidelity estimates make search affordable; hardware-aware NAS targets real devices.
- Good search spaces, strong baselines and training recipes matter as much as search algorithms.
:::

=== POST ===
slug: double-descent-generalization
title: Double Descent and the Generalisation Mystery of Deep Learning
category: deep-learning
level: Advanced
tags: generalization, double descent, overparameterization, implicit regularization, theory
summary: Over-parameterised networks can fit random labels yet generalise on real data, and test error can fall again beyond the interpolation threshold. We explore double descent, benign overfitting and implicit regularisation.
---
Classical statistics teaches a U-shaped curve: as model complexity grows, test error first falls (less bias) and then rises (more variance). Past the point where a model perfectly fits the training data, the prediction is disaster. Yet modern deep networks, with far more parameters than training examples, interpolate their training data and **generalise well**. This lecture examines one of the most interesting open questions in machine learning.

## The puzzle

Zhang et al. (2017), "Understanding deep learning requires rethinking generalization", showed that:

1. Standard CNNs reach 100% training accuracy on CIFAR-10 with **completely random labels** — they can memorise.
2. The same architectures, on real labels, generalise well.
3. Explicit regularisers (weight decay, dropout, augmentation) help but are **not necessary** for good generalisation.

So capacity-based explanations (VC dimension, Rademacher complexity) cannot explain why these networks generalise: the same model class that generalises on real data can fit noise perfectly.

## Double descent

Belkin, Hsu, Ma and Mandal (2019) unified classical and modern regimes in the **double descent** curve:

- In the **under-parameterised** regime, test error follows the classical U-shape.
- At the **interpolation threshold** — when the model has just enough capacity to fit the training data exactly — test error **peaks**.
- In the **over-parameterised** regime, test error **decreases again**, often below the classical minimum.

Nakkiran et al. (2019) showed double descent occurs in deep networks along several axes:

- **Model-wise**: increasing width;
- **Epoch-wise**: training longer can make test error rise then fall again;
- **Sample-wise**: in a critical regime, adding more data can temporarily **hurt**.

The peak is most pronounced with **label noise** and little regularisation.

## Why the peak at the threshold?

Near the interpolation threshold there is essentially **one** way to fit the training data exactly, and that solution must contort itself to pass through every noisy point — huge norm, wild behaviour between points. Past the threshold there are **infinitely many** interpolating solutions, and the training algorithm can choose a well-behaved one.

## Minimum-norm interpolation

For linear regression with more features than examples ($d > n$), infinitely many $\mathbf{w}$ satisfy $\mathbf{X}\mathbf{w} = \mathbf{y}$. Gradient descent initialised at zero converges to the **minimum-norm** solution:

$$
\hat{\mathbf{w}} = \mathbf{X}^\top(\mathbf{X}\mathbf{X}^\top)^{-1}\mathbf{y} = \mathbf{X}^+\mathbf{y}
$$

As $d$ grows beyond $n$, the minimum norm solution can become smoother and test error can fall — double descent in its simplest form.

```python
import numpy as np

rng = np.random.default_rng(0)
n_train, n_test, D = 40, 1000, 400
w_true = rng.normal(size=D) / np.sqrt(D)
X_all = rng.normal(size=(n_train + n_test, D))
y_all = X_all @ w_true + 0.3 * rng.normal(size=n_train + n_test)
Xtr, ytr, Xte, yte = X_all[:n_train], y_all[:n_train], X_all[n_train:], y_all[n_train:]

for p in [5, 10, 20, 30, 38, 40, 42, 50, 80, 150, 400]:
    w = np.linalg.pinv(Xtr[:, :p]) @ ytr          # least squares / minimum-norm solution
    mse = np.mean((Xte[:, :p] @ w - yte) ** 2)
    print(f"features={p:>3}  test MSE={mse:9.3f}")
```

Test error spikes as the number of features approaches the number of training examples (40), then falls again as features increase further.

## Implicit regularisation

The key idea: **the training algorithm itself prefers certain solutions**.

- Gradient descent on linear models finds the minimum-norm interpolant.
- On separable classification with logistic loss, gradient descent converges in direction to the **maximum-margin** solution (Soudry et al., 2018) — like an SVM, without explicit regularisation.
- **SGD noise** biases training towards **flat minima** — wide basins where the loss changes little under parameter perturbations — which are associated with better generalisation (though "flatness" depends on parameterisation and is debated).
- Small initialisation and architecture choices induce **simplicity biases**: networks tend to learn simple functions (low-frequency, low-complexity) first, a phenomenon called **spectral bias**.

## Benign overfitting

Bartlett et al. (2020) characterised **benign overfitting** in linear regression: interpolating noisy data can still give near-optimal test error when the data has many "unimportant" directions that can absorb the noise harmlessly, while the important directions are learned accurately. The noise is fitted, but in a way that barely affects predictions on new data.

## Grokking

Power et al. (2022) observed that small transformers trained on algorithmic tasks (e.g. modular arithmetic) can first memorise the training set, and then — long after training accuracy reaches 100% and with weight decay — suddenly **generalise**. Mechanistic analyses found the networks eventually form structured algorithms (e.g. Fourier-based representations of modular addition). Grokking illustrates how optimisation dynamics and regularisation interact over long training.

## What this means in practice

:::note
- **Bigger models are often better**, even beyond the point of fitting the training set — provided they are trained well. Avoid the danger zone near the interpolation threshold, especially with noisy labels.
- **Regularisation still helps**: tuned weight decay and augmentation can suppress the double-descent peak entirely (Nakkiran et al.).
- **Evaluate empirically**: when theory is incomplete, careful validation is your best guide.
- **Label quality matters**: noise sharpens the peak and harms interpolating models.
:::

Theory is catching up — neural tangent kernels, mean-field analyses, PAC-Bayes and compression bounds, and studies of feature learning — but a complete explanation of deep learning's generalisation remains an active research frontier, and an excellent area for a thesis.

:::exercise
1. Reproduce the double-descent curve above, then add ridge regularisation and observe how the peak changes.
2. Train CNNs of increasing width on CIFAR-10 with 20% label noise and plot test error against width.
3. Show that gradient descent from zero on underdetermined least squares stays in the row space of $\mathbf{X}$ and converges to the minimum-norm solution.
:::

:::takeaway
- Deep networks can memorise random labels yet generalise on real data — capacity alone cannot explain it.
- Double descent: test error peaks at the interpolation threshold and falls again with over-parameterisation.
- Implicit regularisation (minimum norm, max margin, flat minima, simplicity bias) selects good interpolants.
- Label noise sharpens the peak; tuned regularisation and larger models mitigate it.
:::

=== POST ===
slug: loss-landscapes
title: Loss Landscapes, Saddle Points and Flat Minima
category: deep-learning
level: Advanced
tags: loss landscape, optimization, saddle points, flat minima, mode connectivity
summary: What does the surface that SGD descends actually look like? We study critical points in high dimensions, visualise loss landscapes, discuss sharp versus flat minima, mode connectivity and why architecture shapes trainability.
---
Training a neural network is a descent across a loss surface with millions or billions of dimensions. We cannot see it directly, yet its geometry determines whether training succeeds, how fast it converges and how well the result generalises. Over the past decade, researchers have developed surprising insights into this geometry — many of which overturn low-dimensional intuitions.

## Critical points in high dimensions

A **critical point** has zero gradient. Its type is determined by the eigenvalues of the Hessian:

- all positive → local minimum;
- all negative → local maximum;
- mixed signs → **saddle point**.

In low dimensions we picture landscapes full of local minima. In high dimensions, a random critical point needs *every one* of its millions of eigenvalues to be positive to be a minimum — an exponentially unlikely event unless the loss is already low. Dauphin et al. (2014) argued, drawing on random-matrix and statistical-physics results, that:

- **saddle points vastly outnumber local minima** in high dimensions;
- local minima tend to have loss **close to the global minimum**;
- high-loss critical points are mostly saddles, which slow training (plateaus) but can be escaped.

Gradient noise in SGD and momentum help escape saddles; theory shows perturbed gradient descent escapes strict saddles efficiently.

## Visualising loss landscapes

We cannot plot millions of dimensions, but we can plot slices. Li et al. (2018) evaluated the loss along two random directions $\boldsymbol{\delta}, \boldsymbol{\eta}$ around trained parameters $\boldsymbol{\theta}^*$:

$$
f(\alpha, \beta) = L(\boldsymbol{\theta}^* + \alpha\boldsymbol{\delta} + \beta\boldsymbol{\eta})
$$

with **filter normalisation** (scaling each random direction's filters to match the trained filters' norms, removing scale artefacts). Their striking finding: deep networks **without skip connections** have chaotic, highly non-convex landscapes, while **ResNets** and wider networks have smooth, nearly convex-looking basins — a visual explanation of why residual connections make training easier.

```python
import torch
import numpy as np

def loss_slice(model, loss_fn, data, steps=21, span=1.0):
    """Evaluate the loss along one filter-normalised random direction."""
    theta = [p.detach().clone() for p in model.parameters()]
    direction = []
    for p in theta:
        d = torch.randn_like(p)
        if p.dim() > 1:                                   # filter/row-wise normalisation
            d = d * (p.norm(dim=tuple(range(1, p.dim())), keepdim=True) /
                     (d.norm(dim=tuple(range(1, p.dim())), keepdim=True) + 1e-10))
        else:
            d = torch.zeros_like(p)                        # common choice: ignore biases/BN params
        direction.append(d)
    alphas, losses = np.linspace(-span, span, steps), []
    with torch.no_grad():
        for a in alphas:
            for p, t, d in zip(model.parameters(), theta, direction):
                p.copy_(t + a * d)
            losses.append(sum(loss_fn(model(x), y).item() for x, y in data) / len(data))
        for p, t in zip(model.parameters(), theta):
            p.copy_(t)                                     # restore trained weights
    return alphas, losses
```

## Sharp versus flat minima

Hochreiter and Schmidhuber (1997) proposed that **flat minima** — wide regions where the loss stays low — generalise better than **sharp minima**, because a flat solution is robust to the differences between the training and test loss surfaces and needs less precision to describe (a minimum-description-length argument). Keskar et al. (2017) reported that **large-batch** training tended to converge to sharper minima and generalise worse than small-batch training.

Caveats: Dinh et al. (2017) showed that sharpness is not invariant to reparameterisation — for ReLU networks, rescaling weights between layers can make a minimum arbitrarily sharp without changing the function. So flatness must be measured carefully (e.g. with scale-invariant definitions). Nevertheless, methods that explicitly seek flat regions help in practice:

- **Sharpness-Aware Minimisation (SAM)** minimises the worst-case loss in a small neighbourhood:

$$
\min_{\boldsymbol{\theta}}\;\max_{\|\boldsymbol{\epsilon}\| \le \rho}L(\boldsymbol{\theta} + \boldsymbol{\epsilon})
$$

approximated with one extra gradient step, often improving generalisation.
- **Stochastic Weight Averaging (SWA)** averages weights along the trajectory, landing in the centre of a wide basin.

## Mode connectivity

Train two networks from different random seeds and they converge to different minima. Are these isolated valleys? Garipov et al. (2018) and Draxler et al. (2018) found they can be joined by simple **curves** (e.g. a quadratic Bézier curve) along which the loss stays low: the minima are **connected** by low-loss paths. Further work found that, after accounting for permutation symmetries of hidden units (neurons can be reordered without changing the function), many solutions are connected even by **straight lines** (linear mode connectivity), especially in wide networks. This geometry underpins **model merging** and **model soups** — averaging the weights of fine-tuned models to combine their strengths.

## The edge of stability

Cohen et al. (2021) observed that with full-batch gradient descent, the largest Hessian eigenvalue (sharpness) rises during training until it reaches about $2/\eta$ — the classical stability limit — and then hovers there, while the loss keeps decreasing non-monotonically. Training operates at the **edge of stability**, contradicting the assumption that step sizes stay safely below $2/\lambda_{\max}$. The learning rate therefore implicitly controls the sharpness of the solutions found.

## Practical takeaways

:::note
- Local minima are rarely the obstacle in large networks; **plateaus, saddles and poor conditioning** are.
- Architecture shapes the landscape: residual connections, normalisation and width make it smoother.
- Learning rate and batch size influence which minima you reach, not only how fast.
- Weight averaging (SWA, EMA, model soups) and SAM exploit flat, connected regions for better generalisation.
:::

:::exercise
1. Use `loss_slice` to compare 1-D landscapes of a plain 20-layer CNN and a ResNet-20 trained on CIFAR-10.
2. Train two MLPs from different seeds and evaluate the loss along the straight line between their weights. Is there a barrier? Why might permutations matter?
3. Implement one step of SAM (ascent step of size $\rho$ along the normalised gradient, then descent using the gradient at the perturbed point).
:::

:::takeaway
- In high dimensions, saddle points dominate and local minima are usually nearly as good as the global minimum.
- Filter-normalised slices show skip connections make landscapes smoother.
- Flat minima tend to generalise better (with careful definitions); SAM and SWA seek them.
- Minima are connected by low-loss paths, enabling weight averaging and model merging.
:::
