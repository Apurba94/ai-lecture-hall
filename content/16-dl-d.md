=== POST ===
slug: autoencoders
title: "Autoencoders: Compression, Denoising and Representation Learning"
category: deep-learning
level: Intermediate
tags: autoencoders, representation learning, denoising, dimensionality reduction, unsupervised
summary: An autoencoder learns to reconstruct its input through a bottleneck, discovering compact representations without labels. We cover undercomplete, denoising, sparse and convolutional autoencoders and their uses.
---
Can a network learn useful features without any labels? One elegant answer is to ask it to **reproduce its own input** — but through a narrow bottleneck that forces it to discover the data's essential structure. This is the **autoencoder**, an early and still-important form of unsupervised (self-supervised) representation learning, and the direct ancestor of variational autoencoders and latent diffusion models.

## Architecture

An autoencoder has two parts:

- an **encoder** $\mathbf{z} = f_\phi(\mathbf{x})$ mapping the input to a **latent code** (the bottleneck);
- a **decoder** $\hat{\mathbf{x}} = g_\theta(\mathbf{z})$ mapping the code back to input space.

Training minimises a **reconstruction loss**, e.g.

$$
\mathcal{L}(\phi, \theta) = \frac{1}{n}\sum_{i=1}^{n}\|\mathbf{x}_i - g_\theta(f_\phi(\mathbf{x}_i))\|^2
$$

(or binary cross-entropy for inputs in $[0, 1]$). No labels are needed.

## Why a bottleneck?

If the latent dimension were at least as large as the input and the network flexible enough, it could learn the identity and nothing useful. An **undercomplete** autoencoder ($\dim\mathbf{z} \ll \dim\mathbf{x}$) must compress, so it keeps the factors that explain most of the data.

:::note
A **linear** autoencoder with squared-error loss learns the same subspace as **PCA** (Baldi & Hornik, 1989): its optimal decoder spans the top principal components. A **non-linear** autoencoder is therefore a non-linear generalisation of PCA, able to capture curved manifolds.
:::

## Regularised autoencoders

Instead of (or in addition to) a narrow bottleneck, we can constrain the code in other ways:

- **Sparse autoencoders** add a penalty encouraging most latent units to be inactive (e.g. L1 on activations or a KL penalty towards a small target activation). Each input is explained by a few active features. Sparse autoencoders have become a key tool in **mechanistic interpretability**, used to decompose a language model's internal activations into more interpretable features.
- **Denoising autoencoders** (Vincent et al., 2008) corrupt the input — add noise, mask pixels — and train the network to reconstruct the **clean** input. To denoise, the model must learn the structure of the data manifold rather than copy pixels. This idea — *reconstruct what was corrupted* — anticipates masked language modelling (BERT), masked image modelling (MAE) and diffusion models.
- **Contractive autoencoders** penalise the Jacobian of the encoder, making the code insensitive to small input changes.

## A convolutional denoising autoencoder

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

class ConvAE(nn.Module):
    def __init__(self, latent=32):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1), nn.ReLU(),          # 28 -> 14
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),         # 14 -> 7
            nn.Flatten(), nn.Linear(64 * 7 * 7, latent))
        self.dec = nn.Sequential(
            nn.Linear(latent, 64 * 7 * 7), nn.ReLU(), nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),   # 7 -> 14
            nn.ConvTranspose2d(32, 1, 4, 2, 1), nn.Sigmoid()) # 14 -> 28
    def forward(self, x):
        z = self.enc(x)
        return self.dec(z), z

data = datasets.MNIST(".", train=True, download=True, transform=transforms.ToTensor())
loader = torch.utils.data.DataLoader(data, batch_size=128, shuffle=True)
model = ConvAE(); opt = torch.optim.Adam(model.parameters(), 1e-3)
for epoch in range(3):
    for x, _ in loader:                                   # labels unused!
        noisy = (x + 0.4 * torch.randn_like(x)).clamp(0, 1)
        recon, _ = model(noisy)
        loss = F.binary_cross_entropy(recon, x)           # reconstruct the CLEAN image
        opt.zero_grad(); loss.backward(); opt.step()
    print(f"epoch {epoch}: loss {loss.item():.4f}")
```

After training, feeding a noisy digit returns a clean-looking one, and the 32-dimensional codes cluster by digit identity — even though the model never saw a label.

## Applications

1. **Dimensionality reduction and visualisation** — non-linear alternative to PCA.
2. **Denoising** — images, audio, sensor signals.
3. **Anomaly detection** — train on normal data; high reconstruction error flags anomalies (defects on a production line, unusual network traffic).
4. **Pretraining / feature learning** — use the encoder as a feature extractor for downstream tasks with few labels.
5. **Compression** — learned image codecs use autoencoders with quantised latents.
6. **Latent spaces for generation** — latent diffusion models (e.g. Stable Diffusion) first train an autoencoder to compress images into a compact latent space, then run diffusion there.

## Limitations

A plain autoencoder is **not a good generative model**. Its latent space is not organised in any particular way: decoding a random point or the midpoint between two codes often produces garbage, because the model was never asked to make the whole latent space meaningful. **Variational autoencoders** fix this by imposing a probabilistic structure on the latent space — the subject of a lecture in the Generative AI track.

:::exercise
1. Train a linear autoencoder with a 2-D bottleneck on MNIST and compare its latent space with the first two PCA components.
2. Plot the 2-D latent codes of a non-linear autoencoder coloured by digit label. Do classes separate?
3. Use reconstruction error from an autoencoder trained on digits 0–4 to detect digits 5–9. Report ROC-AUC.
:::

:::takeaway
- Autoencoders learn to reconstruct inputs through a bottleneck; no labels are needed.
- Linear autoencoders recover PCA; non-linear ones learn curved manifolds.
- Sparse, denoising and contractive variants regularise the code; denoising foreshadows masked modelling and diffusion.
- Plain autoencoders are poor generators — VAEs add probabilistic structure.
:::

=== POST ===
slug: embeddings-explained
title: "Embeddings: Turning Discrete Things into Meaningful Vectors"
category: deep-learning
level: Beginner
tags: embeddings, representation learning, word vectors, similarity, vector search
summary: Words, users, products and categories become dense vectors whose geometry encodes meaning. We explain embedding layers, how embeddings are learned, how to measure similarity, and how they power search and recommendation.
---
Neural networks work with numbers, but much of the world is discrete: words, product IDs, user IDs, diagnoses, postcodes. One-hot encoding a vocabulary of 50,000 words produces 50,000-dimensional vectors in which every pair of words is equally distant — "cat" is as far from "kitten" as from "carburettor". **Embeddings** fix this by mapping each discrete item to a **dense, low-dimensional vector** learned so that similar items end up close together. Embeddings are one of the most important ideas in modern AI.

## The embedding layer

An embedding layer is simply a lookup table — a matrix $\mathbf{E} \in \mathbb{R}^{V \times d}$ with one row per item:

$$
\text{embed}(i) = \mathbf{E}[i, :] = \mathbf{e}_i^\top\mathbf{E}
$$

where $\mathbf{e}_i$ is the one-hot vector for item $i$. Mathematically it is a linear layer applied to a one-hot input; computationally it is an efficient row lookup. The rows are ordinary parameters, learned by backpropagation like any weights. Typical sizes: $d$ from 16 (small categorical features) to several thousand (large language models).

```python
import torch
import torch.nn as nn

emb = nn.Embedding(num_embeddings=10_000, embedding_dim=64)
token_ids = torch.tensor([[12, 845, 3], [7, 7, 9999]])     # (batch, sequence)
vectors = emb(token_ids)
print(vectors.shape)                                        # (2, 3, 64)
```

## How embeddings acquire meaning

An embedding has no meaning on its own; it acquires meaning from the **task** it is trained on.

- In a **sentiment classifier**, word embeddings organise along positive/negative lines.
- In **word2vec**, trained to predict neighbouring words, words appearing in similar contexts get similar vectors — the distributional hypothesis: "you shall know a word by the company it keeps" (Firth, 1957).
- In a **recommender**, user and item embeddings are trained so their dot product predicts interactions — similar users and similar items cluster.
- In **contrastive models** like CLIP, images and captions are embedded in a shared space where matching pairs are close.
- In **language models**, token embeddings are trained jointly with the whole network; deeper layers produce **contextual** embeddings in which "bank" in "river bank" differs from "bank" in "bank account".

## Measuring similarity

The standard measure is **cosine similarity**:

$$
\cos(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u}^\top\mathbf{v}}{\|\mathbf{u}\|\,\|\mathbf{v}\|}
$$

which compares direction and ignores length. Many systems L2-normalise embeddings so that cosine similarity equals the dot product and nearest-neighbour search is simple.

## Geometry of meaning

Well-trained embeddings exhibit remarkable structure:

- **Clusters** of related items (countries, verbs, sports).
- **Linear relationships**: $\mathbf{v}_{\text{Paris}} - \mathbf{v}_{\text{France}} + \mathbf{v}_{\text{Japan}} \approx \mathbf{v}_{\text{Tokyo}}$ — the famous analogies of word2vec (these hold approximately and less reliably than early demonstrations suggested).
- **Directions** corresponding to attributes such as gender, tense or sentiment.

:::warning
Embeddings absorb the **biases** in their training data. Word embeddings trained on web text have been shown to associate occupations with genders in stereotyped ways (Bolukbasi et al., 2016). When embeddings feed into hiring, lending or eligibility decisions, such associations can cause real harm. Audit embeddings for bias and evaluate downstream systems for fairness.
:::

## Embeddings as a universal interface

Because embeddings turn anything into vectors with meaningful distances, they power a wide range of systems:

1. **Semantic search** — embed documents and queries; retrieve nearest neighbours. This finds relevant results even without shared keywords ("How do I register a newborn?" matches "birth registration procedure").
2. **Retrieval-augmented generation** — retrieve relevant passages by embedding similarity and give them to a language model.
3. **Recommendation** — nearest item embeddings to a user embedding.
4. **Clustering and deduplication** — group similar support tickets or detect near-duplicate records.
5. **Classification with few labels** — train a simple classifier on top of pretrained embeddings.
6. **Categorical features in tabular models** — entity embeddings for high-cardinality columns.

```python
import numpy as np
from sentence_transformers import SentenceTransformer   # pip install sentence-transformers

model = SentenceTransformer("all-MiniLM-L6-v2")
docs = ["How to register the birth of a child",
        "Where can I get vaccinations for my baby?",
        "Steps to renew an expired passport",
        "Cash assistance eligibility criteria"]
query = "My newborn needs a birth certificate"
D = model.encode(docs, normalize_embeddings=True)
q = model.encode([query], normalize_embeddings=True)[0]
for i in np.argsort(-(D @ q)):
    print(f"{D[i] @ q:.3f}  {docs[i]}")
```

The top result shares almost no keywords with the query, yet is semantically closest.

## Practical tips

- **Pretrained embeddings** (from sentence encoders, CLIP, language models) are often better than training from scratch when data is limited.
- Use **multilingual** embedding models for multilingual content — essential in contexts where users write in Bangla, Arabic, English and other languages.
- Evaluate embeddings on **your task**: retrieval metrics (recall@k) with real queries matter more than generic benchmarks.
- For large collections, index embeddings with an **approximate nearest-neighbour** library or vector database.
- Store the **model version** with every embedding; vectors from different models are not comparable.

:::exercise
1. Train a small classifier with a trainable `nn.Embedding` on movie reviews; then find the nearest neighbours of "excellent" and "boring" in the learned space.
2. Using a pretrained sentence encoder, build a mini FAQ search engine over 30 questions and evaluate recall@3 on 10 paraphrased queries.
3. Measure a gender direction in pretrained word vectors (e.g. $\mathbf{v}_{\text{he}} - \mathbf{v}_{\text{she}}$) and project ten occupation words onto it.
:::

:::takeaway
- An embedding layer is a learned lookup table mapping discrete items to dense vectors.
- Embeddings acquire meaning from their training objective; similar items end up close.
- Cosine similarity and nearest-neighbour search make embeddings a universal interface for search, RAG and recommendation.
- Embeddings inherit biases from data — audit them.
:::

=== POST ===
slug: transfer-learning-fine-tuning
title: Transfer Learning and Fine-Tuning
category: deep-learning
level: Intermediate
tags: transfer learning, fine-tuning, pretrained models, feature extraction, domain adaptation
summary: Pretrained models let you achieve strong results with small datasets. We compare feature extraction and fine-tuning, explain discriminative learning rates and layer freezing, and discuss when transfer helps or hurts.
---
Training a large network from scratch requires enormous data and compute. Yet most practitioners today achieve excellent results with a few thousand — sometimes a few hundred — labelled examples. The secret is **transfer learning**: start from a model pretrained on a large, general dataset, and adapt it to your task. It is the single most practical technique in modern deep learning.

## Why transfer works

Features learned on large, diverse data are **general**. In a CNN pretrained on ImageNet, early layers detect edges, colours and textures useful for almost any visual task; middle layers detect parts and patterns; only the last layers are specific to ImageNet's 1,000 classes. Yosinski et al. (2014) quantified this: features become progressively more task-specific with depth. In language models, pretraining on vast text teaches grammar, facts and reasoning patterns that transfer to classification, extraction and question answering.

## Strategy 1: feature extraction

Freeze the pretrained network, remove its original output layer, and train only a new **head** on your data:

$$
\hat{y} = h_\psi\big(f_{\theta_{\text{frozen}}}(\mathbf{x})\big)
$$

- Very fast and cheap; works with tiny datasets.
- You can precompute features once and train a logistic regression on them.
- Limited if your domain differs substantially from the pretraining domain.

## Strategy 2: fine-tuning

Initialise from pretrained weights and continue training **some or all** layers on your task with a small learning rate.

- Usually better accuracy than feature extraction, especially with more data or a domain shift.
- Risk of **catastrophic forgetting** and overfitting on small data.

## Best practices for fine-tuning

1. **Replace the head** with a freshly initialised layer for your classes.
2. **Warm up the head first**: train only the new head for a few epochs (backbone frozen), then unfreeze. A random head produces large, noisy gradients that can damage pretrained features.
3. **Use smaller learning rates** than training from scratch — typically 10–100× smaller (e.g. $10^{-5}$–$10^{-4}$ for transformers, $10^{-4}$–$10^{-3}$ for CNNs).
4. **Discriminative (layer-wise) learning rates**: lower rates for early, general layers; higher for later layers and the head.
5. **Gradual unfreezing**: unfreeze layers from top to bottom over epochs (as in ULMFiT).
6. **Use the pretrained preprocessing**: same input size, normalisation statistics and tokeniser.
7. **Regularise**: weight decay, augmentation, early stopping.
8. **BatchNorm with small batches**: keep BN layers in eval mode (frozen statistics).

```python
import torch
import torch.nn as nn
from torchvision import models

model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
num_classes = 5                                     # e.g. five crop-disease categories
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Stage 1: train the head only
for p in model.parameters():
    p.requires_grad = False
for p in model.fc.parameters():
    p.requires_grad = True
opt = torch.optim.AdamW(model.fc.parameters(), lr=1e-3)
# ... train a few epochs ...

# Stage 2: unfreeze everything with discriminative learning rates
for p in model.parameters():
    p.requires_grad = True
opt = torch.optim.AdamW([
    {"params": list(model.conv1.parameters()) + list(model.layer1.parameters()), "lr": 1e-5},
    {"params": list(model.layer2.parameters()) + list(model.layer3.parameters()), "lr": 5e-5},
    {"params": model.layer4.parameters(), "lr": 1e-4},
    {"params": model.fc.parameters(), "lr": 5e-4},
], weight_decay=0.01)
```

## Choosing a strategy

| Your data size | Similar to pretraining domain | Different domain |
|---|---|---|
| Small | Feature extraction (or fine-tune head + top layers) | Fine-tune top layers carefully; consider domain-specific pretrained models |
| Large | Fine-tune everything | Fine-tune everything (or pretrain on in-domain unlabelled data first) |

For very different domains — satellite imagery, medical scans, microscopy, low-resource languages — look for models pretrained on **similar data**, or continue self-supervised pretraining on your unlabelled in-domain data before fine-tuning (**domain-adaptive pretraining**).

## Parameter-efficient fine-tuning

For very large models, updating all parameters is expensive and requires storing a full copy per task. **Parameter-efficient fine-tuning (PEFT)** trains only a small number of added or selected parameters:

- **Adapters** — small bottleneck layers inserted into each block.
- **LoRA** — low-rank updates to weight matrices (covered in detail in the LLM track).
- **Prompt/prefix tuning** — learned input vectors.
- **BitFit** — train only biases.

These often match full fine-tuning at a tiny fraction of trainable parameters.

## When transfer can hurt

**Negative transfer** occurs when source and target tasks are too different or when pretrained biases conflict with the target task. Pretrained models also carry the **biases and blind spots** of their data: a model pretrained mostly on images from wealthy countries may perform worse on photos of households, crops or objects from elsewhere. Always evaluate on data representative of your deployment population.

:::exercise
1. Compare feature extraction and full fine-tuning of a pretrained ResNet-18 on a dataset of 500 images from 5 classes.
2. Fine-tune with a learning rate of $10^{-2}$ versus $10^{-4}$ and observe catastrophic forgetting.
3. Train a small CNN from scratch on the same data and quantify the benefit of pretraining.
:::

:::takeaway
- Pretrained features are general in early layers and specific in later ones.
- Feature extraction freezes the backbone; fine-tuning adapts it with small, often layer-wise, learning rates.
- Warm up the new head first; keep pretrained preprocessing; regularise.
- PEFT methods adapt huge models cheaply; beware negative transfer and inherited biases.
:::

=== POST ===
slug: pytorch-fundamentals
title: PyTorch Fundamentals: Tensors, Autograd, Modules and the Training Loop
category: deep-learning
level: Beginner
tags: pytorch, training loop, dataloader, nn.module, gpu
summary: A practical tour of PyTorch — tensors and devices, autograd, nn.Module, Dataset and DataLoader, optimisers, and a complete, correct training and evaluation loop you can reuse in every project.
---
Theory becomes skill only when you can implement it. PyTorch has become the most widely used framework in deep-learning research and a major one in industry, thanks to its Pythonic, define-by-run design. This lecture is a hands-on tour. By the end you will have a reusable training template that avoids the most common mistakes.

## Tensors and devices

```python
import torch

x = torch.tensor([[1., 2.], [3., 4.]])
y = torch.randn(2, 2)
print(x @ y, x.shape, x.dtype)

device = "cuda" if torch.cuda.is_available() else "cpu"
x = x.to(device)                      # move data to GPU if available
print(x.device)

a = torch.arange(6).reshape(2, 3)
print(a.float().mean(dim=1, keepdim=True))   # reductions, broadcasting — like NumPy
print(torch.from_numpy(a.numpy()))            # zero-copy bridge to NumPy (on CPU)
```

All operands of an operation must live on the same device; mixing CPU and GPU tensors raises an error.

## Autograd

Set `requires_grad=True` and PyTorch records operations to compute gradients:

```python
w = torch.tensor(2.0, requires_grad=True)
loss = (3 * w - 1) ** 2
loss.backward()
print(w.grad)                         # d/dw (3w-1)^2 = 6(3w-1) = 30
```

Remember: gradients **accumulate** in `.grad`, so zero them each iteration; use `torch.no_grad()` during evaluation.

## Defining models with nn.Module

```python
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, p_drop=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(d_hidden, d_hidden), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(d_hidden, d_out))
    def forward(self, x):
        return self.net(x)
```

Sub-modules assigned as attributes are registered automatically, so `model.parameters()`, `model.to(device)`, `model.train()`, `model.eval()` and `model.state_dict()` all work.

## Datasets and DataLoaders

```python
from torch.utils.data import Dataset, DataLoader

class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        return self.X[i], self.y[i]
```

`DataLoader` handles batching, shuffling and parallel loading (`num_workers`), and `pin_memory=True` speeds up CPU-to-GPU transfer.

## A complete, correct training template

```python
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from sklearn.datasets import make_classification

torch.manual_seed(0)
X, y = make_classification(n_samples=5000, n_features=20, n_informative=10, n_classes=3, random_state=0)
full = TabularDataset(X, y)
train_ds, val_ds = random_split(full, [4000, 1000], generator=torch.Generator().manual_seed(0))
train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=256)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = MLP(20, 128, 3).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)

def run_epoch(loader, train):
    model.train(train)                           # sets dropout/BN behaviour
    total, correct, loss_sum = 0, 0, 0.0
    with torch.set_grad_enabled(train):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)   # logits, not softmax!
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            loss_sum += loss.item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
            total += len(yb)
    return loss_sum / total, correct / total

best = 0.0
for epoch in range(30):
    tr_loss, tr_acc = run_epoch(train_dl, train=True)
    va_loss, va_acc = run_epoch(val_dl, train=False)
    sched.step()
    if va_acc > best:
        best = va_acc
        torch.save({"model": model.state_dict(), "epoch": epoch}, "best.pt")
    if epoch % 5 == 0:
        print(f"epoch {epoch:2d}  train {tr_loss:.3f}/{tr_acc:.3f}  val {va_loss:.3f}/{va_acc:.3f}")
print("best val acc:", round(best, 3))
```

## The five steps of every training iteration

1. **Forward**: `logits = model(xb)`
2. **Loss**: `loss = criterion(logits, yb)`
3. **Zero gradients**: `opt.zero_grad()`
4. **Backward**: `loss.backward()`
5. **Update**: `opt.step()`

## Checklist of common mistakes

:::warning
- Forgetting `opt.zero_grad()` → gradients accumulate across steps.
- Forgetting `model.eval()` / `torch.no_grad()` during validation → wrong dropout/BN behaviour and wasted memory.
- Applying `softmax` before `CrossEntropyLoss` → wrong gradients.
- Labels with wrong dtype (`float` instead of `long` for class indices) or shape.
- Data and model on different devices.
- Calling `loss` (a tensor) in logging instead of `loss.item()` → memory leak via the graph.
- Not shuffling the training loader, or shuffling the validation loader for no reason.
- Forgetting to set seeds when comparing experiments.
:::

## Saving, loading and deploying

Save `state_dict()` (weights), not the whole model object. For deployment, export with `torch.export` / TorchScript or to ONNX, and use `torch.compile(model)` to speed up training and inference with graph compilation.

## Ecosystem

- **torchvision, torchaudio, torchtext** — datasets and pretrained models.
- **Hugging Face Transformers and Datasets** — pretrained NLP, vision and audio models.
- **PyTorch Lightning / Accelerate** — reduce boilerplate, handle multi-GPU and mixed precision.
- **TorchMetrics** — correct metrics across devices.

:::exercise
1. Adapt the template to Fashion-MNIST with the CNN from the CNN lecture.
2. Add early stopping with patience 5 based on validation loss.
3. Deliberately introduce each bug from the checklist, one at a time, and describe the symptom you observe.
:::

:::takeaway
- Tensors live on devices; autograd records operations and accumulates gradients.
- `nn.Module`, `Dataset`/`DataLoader` and optimisers are the core building blocks.
- Every step: forward → loss → zero_grad → backward → step; switch `train()`/`eval()` properly.
- Use a tested template to avoid silent bugs; save `state_dict()`s.
:::

=== POST ===
slug: tensorflow-keras-fundamentals
title: TensorFlow and Keras Fundamentals
category: deep-learning
level: Beginner
tags: tensorflow, keras, deep learning frameworks, callbacks, deployment
summary: Keras offers a high-level, productive API for deep learning. We build models with the Sequential and Functional APIs, train with fit and callbacks, write a custom training step, and export for deployment.
---
PyTorch is dominant in research, but **Keras** remains popular for its concise, beginner-friendly API and strong deployment ecosystem (TensorFlow Lite for mobile, TensorFlow.js for browsers, TensorFlow Serving for servers). Since Keras 3, the same Keras code can run on TensorFlow, JAX or PyTorch backends. Knowing both frameworks makes you versatile — many organisations and tutorials use Keras, and the concepts transfer directly.

## Tensors and gradients in TensorFlow

```python
import tensorflow as tf

x = tf.constant([[1., 2.], [3., 4.]])
w = tf.Variable(2.0)
with tf.GradientTape() as tape:
    loss = (3 * w - 1) ** 2
print(tape.gradient(loss, w))          # 30.0
```

`tf.GradientTape` records operations for automatic differentiation — the counterpart of PyTorch's autograd.

## Three ways to build models

### 1. Sequential API — a simple stack

```python
import keras
from keras import layers

model = keras.Sequential([
    keras.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding="same", activation="relu"),
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(10),                     # logits
])
model.summary()
```

### 2. Functional API — any directed acyclic graph

Multiple inputs and outputs, shared layers, skip connections:

```python
inputs = keras.Input(shape=(32, 32, 3))
x = layers.Conv2D(64, 3, padding="same", activation="relu")(inputs)
shortcut = x
x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = layers.Conv2D(64, 3, padding="same")(x)
x = layers.Activation("relu")(layers.Add()([x, shortcut]))     # residual connection
x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(10)(x)
resnet_like = keras.Model(inputs, outputs)
```

### 3. Model subclassing — full flexibility

```python
class MLP(keras.Model):
    def __init__(self, hidden=128, classes=10):
        super().__init__()
        self.d1 = layers.Dense(hidden, activation="relu")
        self.drop = layers.Dropout(0.2)
        self.out = layers.Dense(classes)
    def call(self, x, training=False):
        return self.out(self.drop(self.d1(x), training=training))
```

## Compile, fit, evaluate

```python
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
x_train = x_train[..., None].astype("float32") / 255.0
x_test = x_test[..., None].astype("float32") / 255.0

model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),   # logits!
    metrics=["accuracy"],
)
callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
    keras.callbacks.ModelCheckpoint("best.keras", save_best_only=True),
]
history = model.fit(x_train, y_train, validation_split=0.1, epochs=20, batch_size=128,
                    callbacks=callbacks, verbose=2)
print(model.evaluate(x_test, y_test, verbose=0))
```

**Callbacks** are hooks that run during training — early stopping, checkpointing, learning-rate scheduling, logging to TensorBoard. They replace much of the manual loop code you write in plain PyTorch.

:::warning
The same logits pitfall exists in Keras: if the last layer has no softmax, set `from_logits=True` in the loss. If you add `activation="softmax"` to the last layer, use `from_logits=False`. Mismatches silently degrade training.
:::

## Efficient input pipelines with tf.data

```python
ds = (tf.data.Dataset.from_tensor_slices((x_train, y_train))
      .shuffle(10_000)
      .map(lambda x, y: (tf.image.random_flip_left_right(x), y), num_parallel_calls=tf.data.AUTOTUNE)
      .batch(128)
      .prefetch(tf.data.AUTOTUNE))
```

`prefetch` overlaps data preparation with training so the accelerator never waits.

## Custom training steps

When you need non-standard training (GANs, custom losses), override `train_step` or write a loop with `GradientTape`:

```python
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = keras.optimizers.Adam(1e-3)

@tf.function                              # compiles the step into a fast graph
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss = loss_fn(y, logits)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss
```

## Transfer learning in Keras

```python
base = keras.applications.EfficientNetB0(include_top=False, weights="imagenet",
                                         input_shape=(224, 224, 3), pooling="avg")
base.trainable = False                    # feature extraction first
clf = keras.Sequential([base, layers.Dropout(0.2), layers.Dense(5)])
```

Then unfreeze top layers and recompile with a smaller learning rate for fine-tuning.

## Deployment

- `model.save("model.keras")` — full model for later training or inference.
- `model.export("saved_model_dir")` — SavedModel for TensorFlow Serving.
- **TensorFlow Lite** — convert for Android/iOS and microcontrollers, with quantisation for small, fast models.
- **TensorFlow.js** — run in the browser, with no server and data staying on the user's device.

## PyTorch vs Keras: a quick comparison

| | PyTorch | Keras / TensorFlow |
|---|---|---|
| Style | Explicit training loop, Pythonic | High-level `fit()` with callbacks |
| Research adoption | Dominant | Smaller share |
| Mobile/browser deployment | ExecuTorch, ONNX | TF Lite, TF.js — mature |
| Multi-backend | — | Keras 3: TF, JAX, PyTorch |

Both are excellent. Choose based on your team, ecosystem and deployment target — and remember that the concepts (tensors, autodiff, layers, optimisers, losses) are identical.

:::exercise
1. Rebuild the PyTorch MLP template from the previous lecture in Keras and compare the code length.
2. Add a TensorBoard callback and inspect training curves.
3. Convert a trained Fashion-MNIST model to TensorFlow Lite with dynamic-range quantisation and compare file size and accuracy.
:::

:::takeaway
- Keras offers Sequential, Functional and subclassing APIs for building models.
- `compile` + `fit` with callbacks handles most training; `GradientTape` enables custom loops.
- Use `from_logits=True` with raw logits; build fast pipelines with `tf.data`.
- TF Lite and TF.js make on-device and in-browser deployment straightforward.
:::

=== POST ===
slug: debugging-neural-network-training
title: Debugging Neural Network Training: A Systematic Recipe
category: deep-learning
level: Intermediate
tags: debugging, training, best practices, overfitting, diagnostics
summary: Neural networks fail silently — they train, but badly. We present a systematic recipe for finding bugs, from data inspection and overfitting a single batch to monitoring activations, gradients and learning curves.
---
Andrej Karpathy has observed that neural network training "fails silently". A bug in ordinary software usually crashes the program; a bug in a training pipeline often just produces a model that is somewhat worse than it should be. You may never know. This lecture presents a disciplined, step-by-step process for building and debugging deep-learning systems — a process I ask all my project students to follow.

## Principle: start simple, add complexity gradually

Begin with the simplest pipeline that could work, verify each component, and add one piece at a time. When something breaks, you know exactly which change caused it.

## Step 1: become one with the data

Before writing a model, **look** at the data:

- Visualise dozens of random examples with their labels. Are labels correct? Are images rotated? Are texts truncated?
- Check class balance, duplicates, missing values, value ranges and units.
- Inspect the data **after** preprocessing and augmentation — the exact tensors the model sees. Many bugs live here: images normalised twice, channels swapped (RGB vs BGR), augmentations destroying labels, tokenisation errors.

```python
import matplotlib.pyplot as plt

xb, yb = next(iter(train_loader))
print(xb.shape, xb.dtype, xb.min().item(), xb.max().item(), xb.mean().item())
print("label counts in batch:", yb.bincount())
fig, axes = plt.subplots(2, 8, figsize=(14, 4))
for ax, img, lab in zip(axes.flat, xb[:16], yb[:16]):
    ax.imshow(img.permute(1, 2, 0).squeeze(), cmap="gray"); ax.set_title(int(lab)); ax.axis("off")
plt.show()
```

## Step 2: establish baselines and sanity checks

- **Check the initial loss.** For $K$ balanced classes with a well-initialised network, the initial cross-entropy should be about $\ln K$ (2.30 for 10 classes). If it is much higher, the output layer is initialised poorly or the loss is misconfigured.
- **Initialise the output bias** sensibly: for imbalanced binary problems, set it to $\log(p/(1-p))$ of the base rate.
- **Human or simple baseline**: a logistic regression or the majority class. Your network must beat it.
- **Input-independent baseline**: train with inputs set to zero. If the real model does not do better, it is not using the inputs.

## Step 3: overfit a single batch

Take one small batch (e.g. 2–32 examples) and train on it repeatedly. The loss should go to nearly **zero**. If it cannot, there is a bug — in the model, loss, labels, optimiser or data pipeline. This is the single most valuable debugging test.

```python
xb, yb = next(iter(train_loader))
model.train()
for step in range(300):
    loss = F.cross_entropy(model(xb), yb)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 50 == 0:
        print(step, round(loss.item(), 4))       # should approach 0
```

## Step 4: verify the plumbing

- **Gradient check** custom layers with finite differences.
- **Check that every parameter receives a gradient**: after `backward()`, any parameter with `grad is None` or all-zero gradients is disconnected.
- **Batch independence test**: set the loss to depend only on example $i$ and check that gradients with respect to other examples' inputs are zero. Non-zero gradients reveal accidental mixing across the batch dimension — e.g. a wrong `view` or reduction over the wrong axis.

```python
for name, p in model.named_parameters():
    if p.grad is None or p.grad.abs().sum() == 0:
        print("NO GRADIENT:", name)
```

## Step 5: monitor the right signals

Log, per step or epoch:

- training and validation **loss and metrics**;
- **learning rate**;
- **gradient norms** (global and per layer) — spikes precede divergence;
- **update-to-weight ratio** $\|\Delta\mathbf{w}\|/\|\mathbf{w}\|$ — around $10^{-3}$ per step is a common healthy range; much larger suggests too high a learning rate;
- **activation statistics** — fraction of dead ReLUs, saturated sigmoids, activation means and standard deviations per layer;
- **predictions on a fixed set of examples** over time.

Tools: TensorBoard, Weights & Biases, MLflow.

## Step 6: read the curves

| Symptom | Likely cause | Try |
|---|---|---|
| Loss is NaN/inf | LR too high, log(0), bad data, fp16 overflow | Lower LR, logits-based losses, clip gradients, check data |
| Loss flat from the start | LR too low, dead units, bug, frozen params | LR range test, overfit-one-batch, check gradients |
| Loss decreases then explodes | LR too high late, no clipping | Schedule/decay, clipping, warm-up |
| Train ↓, val ↑ early | Overfitting | Regularise, augment, more data, early stop |
| Train and val both high | Underfitting | Bigger model, train longer, better features |
| Val better than train | Dropout/augmentation in train only, or leakage | Check splits for leakage |
| Great val, poor production | Distribution shift, preprocessing mismatch | Compare pipelines, monitor inputs |

## Step 7: error analysis

Examine the examples the model gets **wrong**, and the ones it gets wrong **with high confidence**. Group them into categories (blurry images, rare classes, label errors, ambiguous cases). Fix the largest category first. You will often discover label noise — confident "mistakes" that are actually mislabelled data.

## Step 8: then tune and scale

Only after the pipeline is correct: tune the learning rate, regularisation and architecture; scale data and model; ensemble. Change **one thing at a time**, fix random seeds, and keep an experiment log.

:::tip
Keep a small "smoke test" configuration (tiny model, a few hundred examples, a few steps) that runs in under a minute. Run it after every code change. It catches most bugs before you waste hours of GPU time.
:::

:::exercise
1. Verify that your classifier's initial loss is approximately $\ln K$. Then deliberately scale the final layer's weights by 100 and observe the initial loss.
2. Introduce a bug that mixes information across the batch (e.g. normalise with statistics over the wrong axis) and detect it with the batch-independence test.
3. Log the update-to-weight ratio per layer during training and relate it to learning-rate changes.
:::

:::takeaway
- Inspect the exact tensors the model sees; many bugs hide in data pipelines.
- Check the initial loss, beat simple baselines, and overfit a single batch.
- Verify gradients reach every parameter; monitor gradient norms, update ratios and activations.
- Diagnose from curves, do error analysis, and only then tune and scale.
:::

=== POST ===
slug: mixed-precision-training
title: Mixed-Precision Training and GPU Efficiency
category: deep-learning
level: Advanced
tags: mixed precision, fp16, bf16, gpu, performance, tensor cores
summary: Training in 16-bit arithmetic roughly halves memory and can multiply throughput. We explain float16 and bfloat16, loss scaling, automatic mixed precision, and other practical techniques to make GPUs work harder.
---
Modern GPUs contain specialised units (Tensor Cores and their equivalents) that perform matrix multiplication far faster in 16-bit or lower precision than in 32-bit. **Mixed-precision training** exploits this: most computation happens in 16-bit, while numerically sensitive parts stay in 32-bit. It typically roughly halves activation memory and speeds up training substantially, usually with no loss in accuracy. It is standard practice for training any sizeable model.

## Floating-point formats

| Format | Exponent bits | Mantissa bits | Range (approx.) | Precision (decimal digits) |
|---|---|---|---|---|
| FP32 | 8 | 23 | $10^{-38}$ to $3 \times 10^{38}$ | ~7 |
| FP16 | 5 | 10 | $6 \times 10^{-5}$ (normal) to 65,504 | ~3 |
| BF16 | 8 | 7 | same as FP32 | ~2–3 |
| FP8 (E4M3 / E5M2) | 4 / 5 | 3 / 2 | narrow | ~1 |

- **FP16** has decent precision but a **narrow range**: small gradients underflow to zero, large activations overflow to infinity.
- **BF16** ("brain float") keeps FP32's exponent — same range — with less precision. Overflow and underflow are rare, which makes it the preferred format for training on hardware that supports it.
- **FP8** is used on the newest accelerators for further speed-ups, with careful per-tensor scaling.

## The mixed-precision recipe

Micikevicius et al. (2018) established the approach:

1. **Keep an FP32 master copy of the weights.** Tiny updates ($\eta \times$ gradient) can be smaller than FP16's precision relative to the weight; accumulating them in FP16 would lose them entirely.
2. **Run the forward and backward passes in 16-bit** for matrix multiplications and convolutions.
3. **Keep sensitive operations in FP32**: reductions (sums, softmax, normalisation statistics), loss computation, and large accumulations.
4. **Loss scaling (for FP16)**: multiply the loss by a factor $S$ (e.g. $2^{16}$) before backpropagation so small gradients are shifted into FP16's representable range; divide gradients by $S$ before the optimiser step. **Dynamic loss scaling** increases $S$ periodically and halves it (skipping the step) whenever infinities or NaNs appear.

With BF16, loss scaling is generally unnecessary.

## Automatic mixed precision in PyTorch

```python
import torch

device = "cuda"
model = MyModel().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
use_bf16 = torch.cuda.is_bf16_supported()
dtype = torch.bfloat16 if use_bf16 else torch.float16
scaler = torch.amp.GradScaler("cuda", enabled=not use_bf16)   # loss scaling only for fp16

for xb, yb in loader:
    xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
    with torch.autocast(device_type="cuda", dtype=dtype):
        logits = model(xb)
        loss = torch.nn.functional.cross_entropy(logits, yb)
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)                                     # so clipping sees true gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt)                                         # skips the step if inf/NaN found
    scaler.update()
```

`autocast` chooses the precision per operation automatically: matmuls and convolutions in 16-bit, softmax, layer norm and losses in FP32. (With `enabled=False`, the scaler's methods become pass-throughs, so the same loop works for BF16.)

## Where the time goes: other efficiency techniques

Mixed precision is one lever among several. Profile first (`torch.profiler`, NVIDIA Nsight) to find the real bottleneck.

1. **Keep the GPU fed**: slow data loading is the most common bottleneck. Use multiple `num_workers`, `pin_memory=True`, prefetching, and preprocess data offline where possible.
2. **Larger batches** use hardware more efficiently (with appropriate learning-rate scaling). **Gradient accumulation** simulates large batches when memory is limited:

```python
accum = 4
for i, (xb, yb) in enumerate(loader):
    with torch.autocast("cuda", dtype=dtype):
        loss = criterion(model(xb), yb) / accum
    scaler.scale(loss).backward()
    if (i + 1) % accum == 0:
        scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
```

3. **Tensor-friendly shapes**: dimensions that are multiples of 8 (or 64) map better onto Tensor Cores.
4. **Compilation**: `torch.compile(model)` fuses operations and reduces Python overhead.
5. **Fused and memory-efficient kernels**: FlashAttention computes attention without materialising the full attention matrix; fused optimisers and fused layer norms reduce memory traffic.
6. **Activation checkpointing**: recompute activations in the backward pass instead of storing them — trading compute for memory to fit larger batches or models.
7. **Channels-last memory format** for CNNs on modern GPUs.
8. **Avoid host–device synchronisation** inside the loop: calling `.item()` or printing tensors every step forces the CPU to wait for the GPU.

## Memory budgeting

For training with Adam in mixed precision, a rough per-parameter memory cost is: 2 bytes (16-bit weights) + 4 bytes (FP32 master weights) + 8 bytes (two FP32 Adam moments) + 2–4 bytes (gradients) ≈ **16–18 bytes per parameter**, plus activations. A 1-billion-parameter model therefore needs about 16–18 GB just for weights, gradients and optimiser state — before activations. This arithmetic explains why large models require memory-sharding techniques (next lecture).

:::note
Precision reduction continues to push further. Inference commonly uses 8-bit or 4-bit integer weights (quantisation, covered later), and training research explores FP8 and even lower. Each step requires careful handling of the numerical issues we studied in the numerical-stability lecture.
:::

:::exercise
1. Train a ResNet-18 on CIFAR-10 in FP32, FP16 (with scaling) and BF16 if your GPU supports it. Compare throughput (images/second), peak memory and final accuracy.
2. Disable loss scaling in FP16 training and inspect the fraction of gradient entries that are exactly zero.
3. Estimate the memory needed to fine-tune a 7-billion-parameter model with AdamW in mixed precision. Which components dominate?
:::

:::takeaway
- Mixed precision runs matmuls in 16-bit while keeping FP32 master weights and sensitive operations.
- FP16 needs (dynamic) loss scaling; BF16 has FP32's range and usually does not.
- Use `torch.autocast` + `GradScaler`; unscale before clipping.
- Profile and fix data loading, batch size, compilation, fused kernels and checkpointing for further gains.
:::

=== POST ===
slug: distributed-training
title: "Distributed Training: Data, Model, Pipeline and Sharded Parallelism"
category: deep-learning
level: Advanced
tags: distributed training, data parallel, fsdp, zero, model parallel, pipeline parallel
summary: Large models and datasets need many accelerators. We explain data parallelism with all-reduce, sharded data parallelism (ZeRO/FSDP), tensor and pipeline model parallelism, and how they combine at scale.
---
Training modern foundation models requires thousands of GPUs working together for weeks. Even university projects often benefit from spreading training across several GPUs. Distributed training is fundamentally about two constraints: **time** (we want training to finish sooner) and **memory** (the model and its training state may not fit on one device). Different parallelism strategies address each.

## Data parallelism

The simplest and most common approach: **replicate the model** on every device, give each device a different slice of the mini-batch, and average gradients before updating.

1. Each of $N$ workers holds a full model copy.
2. Each computes forward and backward passes on its local batch $\mathcal{B}_k$.
3. Gradients are averaged across workers with an **all-reduce** operation:

$$
\mathbf{g} = \frac{1}{N}\sum_{k=1}^{N}\mathbf{g}_k
$$

4. Every worker applies the identical update, so replicas stay synchronised.

The effective batch size is $N \times$ the per-device batch, so the learning rate and warm-up usually need adjusting.

### All-reduce

A naive approach sends all gradients to one parameter server — a bandwidth bottleneck. **Ring all-reduce** arranges workers in a ring; each sends and receives chunks so that the per-worker communication volume is about $2\frac{N-1}{N}$ times the gradient size — nearly independent of $N$. Libraries such as NCCL implement efficient all-reduce over NVLink and InfiniBand.

### DDP in PyTorch

`DistributedDataParallel` overlaps communication with computation: it all-reduces gradients in **buckets** as soon as they are computed during the backward pass.

```python
# launch with:  torchrun --nproc_per_node=4 train.py
import os, torch, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

dist.init_process_group("nccl")
rank = int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(rank)

model = MyModel().cuda(rank)
model = DDP(model, device_ids=[rank])
sampler = DistributedSampler(train_ds, shuffle=True)          # each rank gets a distinct shard
loader = torch.utils.data.DataLoader(train_ds, batch_size=64, sampler=sampler, num_workers=4)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

for epoch in range(epochs):
    sampler.set_epoch(epoch)                                   # reshuffle differently each epoch
    for xb, yb in loader:
        loss = torch.nn.functional.cross_entropy(model(xb.cuda(rank)), yb.cuda(rank))
        opt.zero_grad(); loss.backward(); opt.step()           # all-reduce happens in backward
    if rank == 0:
        torch.save(model.module.state_dict(), f"ckpt_{epoch}.pt")   # save from one rank only
dist.destroy_process_group()
```

## Sharded data parallelism: ZeRO and FSDP

Plain data parallelism replicates **everything** — weights, gradients and optimiser states — on every GPU. With Adam in mixed precision that is roughly 16 bytes per parameter per GPU: a 10-billion-parameter model needs about 160 GB per GPU, exceeding any single device.

**ZeRO** (Zero Redundancy Optimizer, Rajbhandari et al., 2020) removes the redundancy by **sharding** state across the $N$ data-parallel workers:

| Stage | Sharded | Memory per GPU (approx.) |
|---|---|---|
| ZeRO-1 | Optimiser states | ~4 + 12/N bytes/param |
| ZeRO-2 | + Gradients | ~2 + 14/N |
| ZeRO-3 / FSDP | + Parameters | ~16/N |

In ZeRO-3 / PyTorch **FSDP** (Fully Sharded Data Parallel), each layer's parameters are gathered from all shards just before they are needed (all-gather), used, and released; gradients are reduce-scattered back to their owning shards. Memory scales down with $N$ at the cost of extra communication. **Offloading** optimiser states or parameters to CPU memory or NVMe extends this further.

## Model parallelism

When a single layer or the activations are too large, or to reduce per-device compute, split the model itself.

### Tensor parallelism
Split individual weight matrices across devices. For a linear layer $\mathbf{Y} = \mathbf{X}\mathbf{W}$, partition $\mathbf{W}$ by columns so each device computes part of the output; Megatron-LM pairs a column-split first MLP layer with a row-split second layer so only one all-reduce is needed per MLP block. Attention heads split naturally across devices. Tensor parallelism requires very fast interconnects and is typically used **within** a server.

### Pipeline parallelism
Place consecutive groups of layers ("stages") on different devices. Naively, only one stage works at a time — a large **pipeline bubble**. **Micro-batching** (GPipe) splits each batch into micro-batches that flow through the stages concurrently; schedules like 1F1B (one-forward-one-backward) further reduce idle time and memory.

### Sequence / context parallelism
Split long sequences across devices so attention over very long contexts fits in memory.

### Expert parallelism
In mixture-of-experts models, place different experts on different devices and route tokens to them.

## 3-D parallelism

Frontier-scale training combines these: tensor parallelism inside a node, pipeline parallelism across nodes, and (sharded) data parallelism across replicas of the pipeline. Choosing the right combination depends on model size, sequence length, cluster topology and interconnect bandwidth.

| Strategy | Solves | Communication | Typical scope |
|---|---|---|---|
| Data parallel (DDP) | Time | Gradient all-reduce per step | Any |
| ZeRO / FSDP | Memory (states, params) | All-gather + reduce-scatter | Many GPUs |
| Tensor parallel | Memory + per-layer compute | Frequent, within layers | Within a node |
| Pipeline parallel | Memory (layers) | Activations between stages | Across nodes |

## Practical concerns

- **Reproducibility and randomness**: seed each rank; use `DistributedSampler`; only rank 0 logs and saves.
- **Batch normalisation**: statistics are per-GPU unless you use SyncBatchNorm.
- **Fault tolerance**: long runs on thousands of GPUs *will* see failures — checkpoint frequently and support automatic restart.
- **Communication efficiency**: gradient compression, overlapping communication with computation, and topology-aware placement.
- **Tools**: PyTorch DDP/FSDP, DeepSpeed, Megatron-LM, Hugging Face Accelerate, JAX with `pjit`/sharding annotations.

:::exercise
1. Estimate per-GPU memory for training a 3-billion-parameter model with AdamW on 8 GPUs under plain DDP, ZeRO-2 and FSDP (ZeRO-3).
2. Convert a single-GPU training script to DDP using `torchrun`, and measure the speed-up on 2 and 4 GPUs.
3. Explain why tensor parallelism is usually restricted to GPUs within the same server.
:::

:::takeaway
- Data parallelism replicates the model and all-reduces gradients; ring all-reduce scales well.
- ZeRO/FSDP shard optimiser states, gradients and parameters to cut memory by ~$N$.
- Tensor and pipeline parallelism split the model itself; micro-batching reduces pipeline bubbles.
- Large-scale training combines strategies (3-D parallelism) and needs checkpointing and fault tolerance.
:::
