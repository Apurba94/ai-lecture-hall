=== POST ===
slug: generative-vs-discriminative-models
title: "Generative vs Discriminative Models: Learning to Create"
category: generative-ai
level: Beginner
tags: generative models, discriminative models, density estimation, likelihood, overview
summary: We open the Generative AI track by contrasting models that draw boundaries with models that learn the data distribution itself, and map the families of deep generative models — autoregressive, VAEs, GANs, flows and diffusion.
---
Welcome to the Generative AI track. So far, most of our models have been **discriminative**: given an input, predict a label. Generative models do something more ambitious: they learn the distribution of the data itself, so they can **create** new examples — images, text, audio, molecules — that look like they came from the real world. This is the technology behind text-to-image systems and large language models.

## Two ways to model data

- A **discriminative** model learns $p(y \mid \mathbf{x})$ — or just a decision boundary. Logistic regression, SVMs and most classifiers are discriminative.
- A **generative** model learns $p(\mathbf{x})$ (unconditional) or $p(\mathbf{x}, y)$ / $p(\mathbf{x} \mid y)$ (conditional). Naive Bayes, Gaussian mixtures, HMMs, VAEs, GANs, diffusion models and language models are generative.

A generative model can, in principle, be turned into a classifier with Bayes' rule, $p(y \mid \mathbf{x}) \propto p(\mathbf{x} \mid y)p(y)$. But modelling all of $\mathbf{x}$ is much harder than modelling a boundary: a $256 \times 256$ colour image has almost 200,000 dimensions, and the model must capture everything — textures, lighting, object shapes, their relationships.

## What can generative models do?

1. **Sampling** — generate new, realistic data.
2. **Density estimation** — evaluate how likely a data point is (useful for anomaly detection).
3. **Conditional generation** — generate given a condition: a class, a caption, a sketch, a prompt.
4. **Representation learning** — latent variables often capture meaningful factors.
5. **Imputation, editing and restoration** — fill in missing parts, super-resolve, denoise.
6. **Simulation and data augmentation** — synthetic training data (with caution).

## The families of deep generative models

| Family | Core idea | Exact likelihood? | Sample quality | Sampling speed |
|---|---|---|---|---|
| Autoregressive (PixelCNN, GPT) | Factorise $p(\mathbf{x}) = \prod_i p(x_i \mid x_{<i})$ | Yes | High | Slow (sequential) |
| Variational autoencoders | Latent variable model trained with a lower bound (ELBO) | Lower bound | Moderate (blurry for images) | Fast |
| GANs | Generator vs discriminator game | No | High (sharp) | Fast |
| Normalising flows | Invertible transformations with tractable Jacobians | Yes | Moderate | Fast |
| Diffusion models | Learn to reverse a gradual noising process | Lower bound | Very high, diverse | Slower (many steps; improving) |
| Energy-based models | Unnormalised density $e^{-E(\mathbf{x})}$ | No (intractable $Z$) | Varies | Slow (MCMC) |

Each makes a different trade-off among **quality**, **diversity** (mode coverage), **speed** and **likelihood evaluation**. We study each in turn.

## Evaluating generative models

This is notoriously hard:

- **Likelihood / bits per dimension**: measures fit to data, but high likelihood does not guarantee good-looking samples (and vice versa).
- **Fréchet Inception Distance (FID)**: compares the mean and covariance of Inception-network features of real and generated images:

$$
\text{FID} = \|\boldsymbol{\mu}_r - \boldsymbol{\mu}_g\|^2 + \text{Tr}\left(\boldsymbol{\Sigma}_r + \boldsymbol{\Sigma}_g - 2(\boldsymbol{\Sigma}_r\boldsymbol{\Sigma}_g)^{1/2}\right)
$$

  Lower is better. It captures both quality and diversity roughly, but depends on the feature network and sample size.
- **Precision and recall for distributions**: fidelity (samples look real) vs coverage (all modes represented).
- **CLIP score**: text–image alignment for text-to-image models.
- **Human evaluation**: still the gold standard for perceived quality.

## A tiny generative model

Even a Gaussian mixture is a generative model — fit, then sample:

```python
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_moons

X, _ = make_moons(n_samples=2000, noise=0.06, random_state=0)
gmm = GaussianMixture(n_components=12, covariance_type="full", random_state=0).fit(X)
samples, _ = gmm.sample(1000)                           # generate new data
print("avg log-likelihood of real data:", round(gmm.score(X), 3))
print("avg log-likelihood of noise:   ", round(gmm.score(np.random.uniform(-2, 3, (1000, 2))), 3))
```

Deep generative models do the same things — learn, sample, score — but in extremely high dimensions.

:::note
Generative AI raises questions discriminative models rarely did: authorship and copyright of training data and outputs, realistic misinformation (deepfakes), consent over one's likeness or voice, and the environmental cost of training. We address these throughout the track and in the Ethics track.
:::

:::exercise
1. Classify each model as generative or discriminative: logistic regression, Naive Bayes, HMM, random forest, GPT, BERT with a classification head.
2. Fit GMMs with 2, 12 and 50 components to the moons data. Plot samples and compare training vs held-out log-likelihood.
3. Explain why a model could achieve a good FID while memorising training images, and how you would test for memorisation.
:::

:::takeaway
- Discriminative models learn $p(y \mid \mathbf{x})$; generative models learn $p(\mathbf{x})$ and can create data.
- Families — autoregressive, VAE, GAN, flow, diffusion, energy-based — trade off quality, diversity, speed and likelihood.
- Evaluation uses likelihood, FID, precision/recall, CLIP score and human judgement.
- Generative AI brings new ethical and legal questions.
:::

=== POST ===
slug: variational-autoencoders
title: "Variational Autoencoders: Probabilistic Latent Spaces"
category: generative-ai
level: Advanced
tags: vae, elbo, reparameterization trick, latent variables, generative models
summary: VAEs turn autoencoders into generative models by learning a smooth, probabilistic latent space. We derive the evidence lower bound, the reparameterisation trick and the KL term, and discuss blurriness, posterior collapse and β-VAE.
---
A plain autoencoder compresses data but its latent space is disorganised: decoding a random point gives garbage. Kingma and Welling's **Variational Autoencoder** (2013) — with the parallel work of Rezende, Mohamed and Wierstra — made the latent space **probabilistic and smooth**, so we can sample new data by decoding random latent vectors. VAEs introduced ideas central to modern generative modelling, and their autoencoder component lives on inside latent diffusion models.

## The generative model

Assume each data point $\mathbf{x}$ is generated from a latent variable $\mathbf{z}$:

$$
\mathbf{z} \sim p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, \mathbf{I}), \qquad \mathbf{x} \sim p_\theta(\mathbf{x} \mid \mathbf{z})
$$

where the **decoder** $p_\theta(\mathbf{x} \mid \mathbf{z})$ is a neural network (e.g. outputting pixel means). The marginal likelihood

$$
p_\theta(\mathbf{x}) = \int p_\theta(\mathbf{x} \mid \mathbf{z})\,p(\mathbf{z})\,d\mathbf{z}
$$

is intractable: we cannot integrate over all $\mathbf{z}$, and the true posterior $p_\theta(\mathbf{z} \mid \mathbf{x})$ is intractable too.

## Amortised variational inference

Introduce an **encoder** $q_\phi(\mathbf{z} \mid \mathbf{x})$ — a neural network outputting the mean $\boldsymbol{\mu}$ and (log-)variance $\boldsymbol{\sigma}^2$ of a Gaussian — to approximate the posterior. "Amortised" means one network infers latents for all data points rather than optimising a separate distribution for each.

## The Evidence Lower Bound (ELBO)

For any $q$, the log-likelihood decomposes as

$$
\log p_\theta(\mathbf{x}) = \underbrace{\mathbb{E}_{q_\phi(\mathbf{z} \mid \mathbf{x})}\big[\log p_\theta(\mathbf{x} \mid \mathbf{z})\big] - D_{\text{KL}}\big(q_\phi(\mathbf{z} \mid \mathbf{x})\,\|\,p(\mathbf{z})\big)}_{\text{ELBO}} + D_{\text{KL}}\big(q_\phi(\mathbf{z} \mid \mathbf{x})\,\|\,p_\theta(\mathbf{z} \mid \mathbf{x})\big)
$$

The last KL term is non-negative, so the ELBO is a **lower bound** on $\log p_\theta(\mathbf{x})$. We maximise it jointly over encoder and decoder. Its two terms have clear meanings:

1. **Reconstruction term**: decoded samples should reproduce the input.
2. **KL regulariser**: each encoding distribution should stay close to the prior $\mathcal{N}(\mathbf{0}, \mathbf{I})$ — packing the latent space densely and smoothly so random prior samples decode to sensible data.

For Gaussian $q$ and standard normal prior, the KL has a closed form:

$$
D_{\text{KL}} = \frac{1}{2}\sum_{j=1}^{d}\left(\mu_j^2 + \sigma_j^2 - \log\sigma_j^2 - 1\right)
$$

## The reparameterisation trick

We need gradients of $\mathbb{E}_{q_\phi}[\cdot]$ with respect to $\phi$, but sampling is not differentiable. Rewrite the sample as a deterministic function of $\phi$ and independent noise:

$$
\mathbf{z} = \boldsymbol{\mu}_\phi(\mathbf{x}) + \boldsymbol{\sigma}_\phi(\mathbf{x})\odot\boldsymbol{\epsilon}, \qquad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
$$

Now gradients flow through $\boldsymbol{\mu}$ and $\boldsymbol{\sigma}$; the randomness sits in $\boldsymbol{\epsilon}$. This low-variance gradient estimator is what made VAEs trainable with ordinary backpropagation.

## Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, x_dim=784, h=400, z_dim=20):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(x_dim, h), nn.ReLU())
        self.mu, self.logvar = nn.Linear(h, z_dim), nn.Linear(h, z_dim)
        self.dec = nn.Sequential(nn.Linear(z_dim, h), nn.ReLU(), nn.Linear(h, x_dim))
    def forward(self, x):
        hdn = self.enc(x)
        mu, logvar = self.mu(hdn), self.logvar(hdn)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)          # reparameterisation
        return self.dec(z), mu, logvar

def vae_loss(logits, x, mu, logvar, beta=1.0):
    recon = F.binary_cross_entropy_with_logits(logits, x, reduction="sum")
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (recon + beta * kl) / x.size(0)

model = VAE()
x = torch.rand(64, 784)                        # e.g. flattened MNIST images in [0, 1]
logits, mu, logvar = model(x)
print(vae_loss(logits, x, mu, logvar).item())

with torch.no_grad():                           # generation: decode samples from the prior
    samples = torch.sigmoid(model.dec(torch.randn(16, 20)))
```

After training on MNIST, decoding a grid of latent points shows digits morphing smoothly into one another — evidence of a continuous, structured latent space.

## Known issues

- **Blurry samples**: with a Gaussian (MSE-like) or per-pixel likelihood, the decoder averages over uncertainty, producing blurry images compared with GANs and diffusion models.
- **Posterior collapse**: with powerful decoders (e.g. autoregressive), the model may ignore $\mathbf{z}$, making the KL term zero. Remedies include KL annealing (gradually increasing its weight) and "free bits".
- **The prior hole problem**: regions of the prior rarely covered by any encoding decode poorly.

## Variants

- **β-VAE** (Higgins et al., 2017): weight the KL by $\beta > 1$ to encourage **disentangled** latents where individual dimensions capture separate factors (e.g. rotation, scale) — at some cost to reconstruction.
- **Conditional VAE (CVAE)**: condition encoder and decoder on a label or other input.
- **VQ-VAE** (van den Oord et al., 2017): a **discrete** latent codebook with vector quantisation; paired with an autoregressive prior it generated high-quality images and audio, and discrete codebooks underlie many image and audio tokenisers used by modern multimodal models.
- **Hierarchical VAEs** (NVAE, VDVAE): many latent layers; much sharper samples.
- The **autoencoder in Stable Diffusion** is a VAE-style model (with perceptual and adversarial losses) that compresses images into a latent space where diffusion operates.

:::exercise
1. Derive the ELBO using Jensen's inequality on $\log\int p_\theta(\mathbf{x} \mid \mathbf{z})p(\mathbf{z})\,d\mathbf{z}$.
2. Train the VAE on MNIST with a 2-D latent space and plot the decoded grid over $[-3, 3]^2$.
3. Train with $\beta = 0.1, 1, 4$ and compare reconstructions and samples.
:::

:::takeaway
- VAEs learn an encoder $q_\phi(\mathbf{z} \mid \mathbf{x})$ and decoder $p_\theta(\mathbf{x} \mid \mathbf{z})$ by maximising the ELBO.
- ELBO = reconstruction − KL to the prior; the KL organises a smooth latent space.
- The reparameterisation trick makes sampling differentiable.
- VAEs can be blurry and suffer posterior collapse; VQ-VAE and hierarchical VAEs improved them, and VAE-style autoencoders power latent diffusion.
:::

=== POST ===
slug: generative-adversarial-networks
title: "Generative Adversarial Networks: The Generator–Discriminator Game"
category: generative-ai
level: Advanced
tags: gan, adversarial training, minimax, mode collapse, generative models
summary: GANs train a generator to fool a discriminator in a two-player game. We derive the minimax objective and its optimum, study training dynamics, mode collapse and the non-saturating loss, and build a DCGAN.
---
In 2014 Ian Goodfellow and colleagues proposed a strikingly original idea: train two networks against each other. A **generator** creates fake data; a **discriminator** tries to tell fake from real. As each improves, the fakes become more convincing. **Generative Adversarial Networks (GANs)** went on to produce photorealistic faces, art and image translations, and dominated image generation until diffusion models arrived around 2021.

## The game

- **Generator** $G$: maps random noise $\mathbf{z} \sim p(\mathbf{z})$ (e.g. Gaussian) to a sample $G(\mathbf{z})$.
- **Discriminator** $D$: outputs $D(\mathbf{x}) \in (0, 1)$, the probability that $\mathbf{x}$ is real.

They play a **minimax game**:

$$
\min_G\max_D\; V(D, G) = \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}}\big[\log D(\mathbf{x})\big] + \mathbb{E}_{\mathbf{z} \sim p(\mathbf{z})}\big[\log(1 - D(G(\mathbf{z})))\big]
$$

The discriminator maximises a binary cross-entropy objective (label real as 1, fake as 0); the generator minimises it — it wants $D(G(\mathbf{z}))$ close to 1.

## The theory

For a fixed generator with sample distribution $p_g$, the optimal discriminator is

$$
D^*(\mathbf{x}) = \frac{p_{\text{data}}(\mathbf{x})}{p_{\text{data}}(\mathbf{x}) + p_g(\mathbf{x})}
$$

Substituting it back gives

$$
V(D^*, G) = -\log 4 + 2\,\text{JSD}(p_{\text{data}}\,\|\,p_g)
$$

where JSD is the **Jensen–Shannon divergence**. So, with an optimal discriminator, the generator minimises the JS divergence between the data and generated distributions; the global optimum is $p_g = p_{\text{data}}$, where $D^* = 1/2$ everywhere. Notice that GANs never evaluate a likelihood — they are **implicit** generative models.

## The non-saturating loss

Early in training the discriminator easily rejects poor fakes, so $\log(1 - D(G(\mathbf{z})))$ saturates and gives the generator vanishing gradients. In practice the generator instead **maximises** $\log D(G(\mathbf{z}))$ — the same fixed point, much stronger gradients.

## Training loop

Alternate: one (or a few) discriminator steps on a batch of real and fake data, then one generator step.

```python
import torch
import torch.nn as nn

z_dim = 100
G = nn.Sequential(                                    # DCGAN-style generator for 28x28 images
    nn.Linear(z_dim, 128 * 7 * 7), nn.BatchNorm1d(128 * 7 * 7), nn.ReLU(),
    nn.Unflatten(1, (128, 7, 7)),
    nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),   # 14x14
    nn.ConvTranspose2d(64, 1, 4, 2, 1), nn.Tanh())                        # 28x28, values in [-1, 1]
D = nn.Sequential(
    nn.Conv2d(1, 64, 4, 2, 1), nn.LeakyReLU(0.2),
    nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
    nn.Flatten(), nn.Linear(128 * 7 * 7, 1))                              # logit

bce = nn.BCEWithLogitsLoss()
optG = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
optD = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

def train_step(real):                                 # real: (B, 1, 28, 28) scaled to [-1, 1]
    B = real.size(0)
    fake = G(torch.randn(B, z_dim))
    # Discriminator: real -> 1, fake -> 0
    lossD = bce(D(real), torch.ones(B, 1)) + bce(D(fake.detach()), torch.zeros(B, 1))
    optD.zero_grad(); lossD.backward(); optD.step()
    # Generator: non-saturating loss, wants D(fake) -> 1
    lossG = bce(D(fake), torch.ones(B, 1))
    optG.zero_grad(); lossG.backward(); optG.step()
    return lossD.item(), lossG.item()

print(train_step(torch.rand(16, 1, 28, 28) * 2 - 1))
```

The **DCGAN** guidelines (Radford et al., 2016) made convolutional GANs train reliably: strided convolutions instead of pooling, batch normalisation, ReLU in the generator and LeakyReLU in the discriminator, tanh output, and Adam with $\beta_1 = 0.5$.

## Why GANs are hard to train

- **Non-convergence**: gradient descent on a two-player game can oscillate rather than converge to an equilibrium.
- **Mode collapse**: the generator produces only a few kinds of outputs that fool the discriminator (e.g. only one digit), ignoring the diversity of the data.
- **Vanishing gradients** when the discriminator becomes too strong — related to the JS divergence being constant when distributions do not overlap.
- **Sensitivity** to architecture and hyperparameters; loss values are not a reliable progress indicator — inspect samples and FID.

## Stabilisation techniques

- **Wasserstein GAN** (Arjovsky et al., 2017): replace JS with the **Earth-Mover (Wasserstein-1) distance**, which gives meaningful gradients even when distributions do not overlap; the "critic" must be 1-Lipschitz, enforced by weight clipping or, better, a **gradient penalty** (WGAN-GP).
- **Spectral normalisation** of discriminator weights (Miyato et al., 2018).
- **Two time-scale update rule** (different learning rates for G and D).
- Label smoothing, minibatch discrimination, feature matching, R1 regularisation, exponential moving averages of generator weights.

## Strengths and legacy

GANs produce **sharp**, high-fidelity samples and generate in a single forward pass — much faster than early diffusion models. They lost the lead in general image generation to diffusion models, which train more stably and cover modes better, but adversarial losses remain widely used: as a perceptual component in image super-resolution, in the autoencoders of latent diffusion, in neural vocoders for speech (HiFi-GAN), and to distil diffusion models into fast few-step generators.

:::exercise
1. Derive $D^*(\mathbf{x})$ by maximising $V(D, G)$ pointwise.
2. Train the DCGAN on MNIST, saving samples every epoch. Do you observe mode collapse?
3. Replace the loss with WGAN-GP and compare training stability.
:::

:::takeaway
- A GAN's generator and discriminator play a minimax game; at the optimum the generator matches the data distribution.
- With an optimal discriminator, the objective minimises Jensen–Shannon divergence; use the non-saturating generator loss in practice.
- Training is unstable: mode collapse, oscillation and vanishing gradients.
- WGAN-GP, spectral normalisation and DCGAN practices stabilise training; adversarial losses remain widely used.
:::

=== POST ===
slug: gan-variants-stylegan-cyclegan
title: "GAN Variants: Conditional GANs, Pix2Pix, CycleGAN and StyleGAN"
category: generative-ai
level: Advanced
tags: stylegan, cyclegan, pix2pix, conditional gan, image translation, deepfakes
summary: A tour of influential GAN architectures — conditional GANs, paired and unpaired image-to-image translation, progressive growing and StyleGAN's style-based generator — plus the deepfake concerns they raised.
---
The basic GAN generates random samples. Researchers quickly extended the idea to **control** what is generated, to **translate** images between domains, and to reach **photorealistic** quality. This lecture surveys the architectures that defined the GAN era and whose ideas still shape image synthesis.

## Conditional GANs

Mirza and Osindero (2014) fed a condition $\mathbf{y}$ — e.g. a class label — to both generator and discriminator: $G(\mathbf{z}, \mathbf{y})$ and $D(\mathbf{x}, \mathbf{y})$. The discriminator now judges whether $\mathbf{x}$ is a real example **of class $\mathbf{y}$**. Refinements such as the **projection discriminator** and class-conditional batch normalisation led to **BigGAN** (Brock et al., 2019), which generated diverse, high-fidelity ImageNet images at scale, and introduced the **truncation trick** (sampling $\mathbf{z}$ from a truncated distribution to trade diversity for fidelity).

## Pix2Pix: paired image-to-image translation

Isola et al. (2017) framed many problems as translating one image into another: sketches → photos, segmentation maps → street scenes, day → night, aerial photo → map. With **paired** training data $(\mathbf{x}, \mathbf{y})$:

- **Generator**: a U-Net mapping $\mathbf{x}$ to $\hat{\mathbf{y}}$ (skip connections carry low-level detail).
- **Discriminator**: a **PatchGAN** that classifies each $N \times N$ patch as real or fake, focusing on local texture.
- **Loss**: adversarial loss + an **L1 reconstruction** term:

$$
\mathcal{L} = \mathcal{L}_{\text{cGAN}}(G, D) + \lambda\,\mathbb{E}\big[\|\mathbf{y} - G(\mathbf{x})\|_1\big]
$$

L1 captures low-frequency correctness; the adversarial term makes outputs sharp.

## CycleGAN: unpaired translation

Paired data is often unavailable (there are no photos of the same scene painted by Monet). Zhu et al. (2017) learned mappings $G: X \to Y$ and $F: Y \to X$ from **unpaired** collections, with adversarial losses in both domains plus a **cycle-consistency loss**:

$$
\mathcal{L}_{\text{cyc}} = \mathbb{E}_{\mathbf{x}}\big[\|F(G(\mathbf{x})) - \mathbf{x}\|_1\big] + \mathbb{E}_{\mathbf{y}}\big[\|G(F(\mathbf{y})) - \mathbf{y}\|_1\big]
$$

Translating a horse to a zebra and back should return the original horse — preventing the generator from producing arbitrary zebras unrelated to the input. CycleGAN powered photo↔painting style transfer, summer↔winter, and domain adaptation (e.g. synthetic → realistic driving scenes).

:::warning
Unpaired translation can **hallucinate** content. In one well-known demonstration, a CycleGAN-style model translating medical images added or removed tumour-like features because the target domain's statistics differed. Never use such translated images for diagnosis without rigorous validation.
:::

## Progressive GAN and StyleGAN

**Progressive growing** (Karras et al., 2018) started training at $4 \times 4$ resolution and progressively added layers up to $1024 \times 1024$, stabilising high-resolution training.

**StyleGAN** (Karras, Laine & Aila, 2019) redesigned the generator:

1. A **mapping network** (8-layer MLP) transforms $\mathbf{z}$ into an intermediate latent $\mathbf{w}$, which is less entangled than $\mathbf{z}$.
2. The synthesis network starts from a **learned constant**, and $\mathbf{w}$ controls each layer through **adaptive instance normalisation (AdaIN)** — injecting "style" at every resolution.
3. **Per-pixel noise** inputs add stochastic detail (hair strands, freckles).

Styles at coarse layers control pose and face shape; middle layers control features; fine layers control colour and micro-texture. **Style mixing** combines coarse styles from one image with fine styles from another. StyleGAN2 (2020) removed characteristic artefacts (weight demodulation replaced AdaIN, path-length regularisation), and StyleGAN3 (2021) tackled "texture sticking" with alias-free operations. The results — faces of people who do not exist — were strikingly photorealistic.

## GAN inversion and editing

Because StyleGAN's $\mathcal{W}$ space is well organised, you can **invert** a real image (find the latent that reproduces it) and then **edit** it by moving along semantic directions (age, smile, lighting) discovered with labels or unsupervised methods (e.g. PCA in latent space, GANSpace). This enabled powerful photo-editing tools.

## Summary of variants

| Model | Year | Key contribution |
|---|---|---|
| DCGAN | 2016 | Stable convolutional architecture guidelines |
| Conditional GAN | 2014 | Class/attribute control |
| Pix2Pix | 2017 | Paired translation: U-Net + PatchGAN + L1 |
| CycleGAN | 2017 | Unpaired translation via cycle consistency |
| WGAN-GP | 2017 | Wasserstein loss with gradient penalty |
| Progressive GAN | 2018 | Grow resolution during training |
| BigGAN | 2019 | Large-scale class-conditional generation; truncation trick |
| StyleGAN 1–3 | 2019–2021 | Style-based generator; photoreal faces; editable latent space |

## Deepfakes and responsibility

GAN-based face synthesis and face swapping enabled "deepfakes" — realistic fabricated images and videos of real people. Harms include non-consensual intimate imagery, fraud, harassment and political misinformation; the mere possibility of fakes also lets real evidence be dismissed ("the liar's dividend"). Countermeasures include detection models (an arms race), **content provenance** standards such as C2PA that cryptographically sign media at capture or creation, watermarking, platform policies and laws against specific abuses.

:::exercise
1. Explain why cycle consistency alone does not guarantee semantically faithful translation. Give a failure example.
2. Train a small Pix2Pix model to colourise grayscale images of a single object category.
3. Using a pretrained StyleGAN, interpolate between two latent vectors in $\mathcal{Z}$ and in $\mathcal{W}$. Which interpolation looks smoother?
:::

:::takeaway
- Conditional GANs control outputs with labels; BigGAN scaled class-conditional generation.
- Pix2Pix translates paired images with U-Net, PatchGAN and L1; CycleGAN handles unpaired data with cycle consistency.
- StyleGAN's mapping network and per-layer style modulation yield photorealistic, editable images.
- The same power enables deepfakes — provenance, detection and policy are part of responsible practice.
:::

=== POST ===
slug: normalizing-flows
title: "Normalising Flows: Exact Likelihood with Invertible Networks"
category: generative-ai
level: Advanced
tags: normalizing flows, change of variables, realnvp, glow, density estimation
summary: Normalising flows transform a simple distribution into a complex one through invertible mappings, giving exact likelihoods and fast sampling. We derive the change-of-variables formula, coupling layers, RealNVP and Glow, and continuous flows.
---
VAEs optimise a lower bound; GANs provide no likelihood at all. **Normalising flows** offer something rare: **exact** log-likelihood evaluation **and** efficient sampling, both through the same invertible network. They are valuable when precise densities matter — for anomaly detection, scientific inference and as building blocks in other models — and their continuous-time relatives connect directly to modern flow-matching generators.

## The change-of-variables formula

Take a simple base distribution $p_Z(\mathbf{z})$, e.g. a standard Gaussian, and an **invertible, differentiable** function $f$ with $\mathbf{x} = f(\mathbf{z})$ and $\mathbf{z} = f^{-1}(\mathbf{x})$. Then

$$
p_X(\mathbf{x}) = p_Z\big(f^{-1}(\mathbf{x})\big)\left|\det\frac{\partial f^{-1}(\mathbf{x})}{\partial\mathbf{x}}\right|
$$

or in log form, writing the inverse as a composition of $K$ steps $\mathbf{x} = \mathbf{h}_0 \to \mathbf{h}_1 \to \dots \to \mathbf{h}_K = \mathbf{z}$:

$$
\log p_X(\mathbf{x}) = \log p_Z(\mathbf{z}) + \sum_{k=1}^{K}\log\left|\det\frac{\partial\mathbf{h}_k}{\partial\mathbf{h}_{k-1}}\right|
$$

The Jacobian determinant accounts for how the transformation stretches or compresses volume — probability mass is conserved. A sequence of simple invertible steps "flows" the Gaussian into a complex distribution, hence the name.

- **Training**: maximise $\log p_X(\mathbf{x})$ on data (exact maximum likelihood).
- **Sampling**: draw $\mathbf{z} \sim p_Z$ and compute $\mathbf{x} = f(\mathbf{z})$.

## The design challenge

A general $d \times d$ Jacobian determinant costs $O(d^3)$ — impossible for images with $d$ in the hundreds of thousands. Flow layers are therefore designed so that (1) they are easily invertible and (2) their Jacobians are **triangular**, whose determinant is the product of the diagonal.

## Coupling layers: RealNVP

Dinh et al.'s **affine coupling layer** (RealNVP, 2017) splits the input into two halves $(\mathbf{x}_a, \mathbf{x}_b)$:

$$
\mathbf{y}_a = \mathbf{x}_a, \qquad \mathbf{y}_b = \mathbf{x}_b\odot\exp\big(s(\mathbf{x}_a)\big) + t(\mathbf{x}_a)
$$

where $s$ and $t$ are **arbitrary** neural networks. Properties:

- **Inverse** is trivial: $\mathbf{x}_b = (\mathbf{y}_b - t(\mathbf{y}_a))\odot\exp(-s(\mathbf{y}_a))$ — no need to invert $s$ or $t$.
- **Jacobian** is triangular, so $\log|\det| = \sum_j s(\mathbf{x}_a)_j$ — essentially free.

Alternating which half is transformed (and permuting dimensions) lets all dimensions interact across layers.

```python
import torch
import torch.nn as nn

class AffineCoupling(nn.Module):
    def __init__(self, dim, hidden=128, flip=False):
        super().__init__()
        self.flip, self.d = flip, dim // 2
        self.net = nn.Sequential(nn.Linear(self.d, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 2 * (dim - self.d)))
    def forward(self, x):                                  # data -> latent direction
        xa, xb = x[:, :self.d], x[:, self.d:]
        if self.flip: xa, xb = xb, xa
        s, t = self.net(xa).chunk(2, dim=1)
        s = torch.tanh(s)                                  # keep scales stable
        yb = xb * torch.exp(s) + t
        y = torch.cat([yb, xa], 1) if self.flip else torch.cat([xa, yb], 1)
        return y, s.sum(1)                                 # output and log|det J|

class Flow(nn.Module):
    def __init__(self, dim=2, n_layers=8):
        super().__init__()
        self.layers = nn.ModuleList([AffineCoupling(dim, flip=i % 2 == 1) for i in range(n_layers)])
        self.base = torch.distributions.MultivariateNormal(torch.zeros(dim), torch.eye(dim))
    def log_prob(self, x):
        logdet = 0.0
        for layer in self.layers:
            x, ld = layer(x); logdet = logdet + ld
        return self.base.log_prob(x) + logdet

from sklearn.datasets import make_moons
X = torch.tensor(make_moons(4000, noise=0.05)[0], dtype=torch.float32)
flow = Flow(); opt = torch.optim.Adam(flow.parameters(), 1e-3)
for step in range(2000):
    idx = torch.randint(0, len(X), (256,))
    loss = -flow.log_prob(X[idx]).mean()                   # exact negative log-likelihood
    opt.zero_grad(); loss.backward(); opt.step()
print("final NLL:", round(loss.item(), 3))
```

(Sampling requires implementing each layer's inverse — a good exercise.)

## Other flow architectures

- **Glow** (Kingma & Dhariwal, 2018): adds **invertible $1 \times 1$ convolutions** (learned channel permutations, with LU-decomposed weights for cheap determinants) and **ActNorm**; generated high-resolution faces with exact likelihood.
- **Autoregressive flows**: MAF (Masked Autoregressive Flow) has fast density evaluation but slow sampling; IAF (Inverse Autoregressive Flow) has the opposite trade-off — IAF was used to make VAE posteriors more flexible.
- **Neural spline flows**: use monotonic rational-quadratic splines as element-wise transforms — more expressive than affine coupling.
- **Continuous normalising flows** (Neural ODEs, FFJORD): define the transformation as the solution of an ODE $d\mathbf{z}/dt = f_\theta(\mathbf{z}, t)$; the log-density evolves by the trace of the Jacobian, computable with stochastic estimators.

## Flow matching

Training continuous flows by maximum likelihood requires expensive ODE solves. **Flow matching** (Lipman et al., 2023) and **rectified flow** (Liu et al., 2023) instead regress the network's velocity field directly onto simple target velocities along paths between noise and data (e.g. straight lines $\mathbf{x}_t = (1 - t)\mathbf{z} + t\mathbf{x}$ with target velocity $\mathbf{x} - \mathbf{z}$). Training becomes simulation-free regression, closely related to diffusion models; several recent state-of-the-art image and video generators are trained with flow matching.

## Strengths and weaknesses

**Strengths:** exact likelihood; exact latent inference ($\mathbf{z} = f^{-1}(\mathbf{x})$); fast sampling (for coupling flows); stable maximum-likelihood training.

**Weaknesses:** invertibility forces the latent to have the **same dimension** as the data; architectures are constrained; sample quality of discrete flows lagged behind GANs and diffusion models for images.

:::exercise
1. Derive the log-determinant of the affine coupling layer's Jacobian.
2. Implement the inverse of `AffineCoupling` and generate samples from the trained moons flow.
3. Use the trained flow as an anomaly detector: compare log-densities of moon points and uniform random points.
:::

:::takeaway
- Flows map a simple base distribution through invertible layers; the change-of-variables formula gives exact log-likelihood.
- Coupling layers (RealNVP, Glow) make inverses and Jacobian determinants cheap.
- Autoregressive, spline and continuous flows trade off expressivity and speed.
- Flow matching trains continuous flows by simple regression and powers modern generators.
:::

=== POST ===
slug: diffusion-models
title: "Diffusion Models: Generating by Learning to Denoise"
category: generative-ai
level: Advanced
tags: diffusion, ddpm, denoising, score matching, generative models
summary: Diffusion models gradually add noise to data and train a network to reverse the process. We derive the DDPM forward and reverse processes, the simple noise-prediction loss, sampling, the score-based view, and faster samplers like DDIM.
---
Since around 2021, **diffusion models** have produced the most impressive generated images, and they now power text-to-image, video, audio and even molecule generation. The idea is almost paradoxical: to learn how to create data, learn how to **remove noise** from it. Start with pure noise, denoise step by step, and a realistic image emerges. This lecture derives the method carefully.

## The forward (noising) process

Take a data point $\mathbf{x}_0$ and gradually add Gaussian noise over $T$ steps (e.g. $T = 1000$), with a small variance schedule $\beta_1, \dots, \beta_T$:

$$
q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) = \mathcal{N}\left(\mathbf{x}_t;\; \sqrt{1 - \beta_t}\,\mathbf{x}_{t-1},\; \beta_t\mathbf{I}\right)
$$

After enough steps, $\mathbf{x}_T$ is essentially pure Gaussian noise. Crucially, because sums of Gaussians are Gaussian, we can jump to **any** step in closed form. Define $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_{s=1}^{t}\alpha_s$:

$$
q(\mathbf{x}_t \mid \mathbf{x}_0) = \mathcal{N}\left(\mathbf{x}_t;\; \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0,\; (1 - \bar{\alpha}_t)\mathbf{I}\right) \quad\Longleftrightarrow\quad \mathbf{x}_t = \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\,\boldsymbol{\epsilon}, \;\; \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
$$

## The reverse (denoising) process

We want to run the process backwards: start from $\mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ and repeatedly sample $\mathbf{x}_{t-1}$ from $\mathbf{x}_t$. For small $\beta_t$, the true reverse step is approximately Gaussian, so we learn

$$
p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t) = \mathcal{N}\left(\mathbf{x}_{t-1};\; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t),\; \sigma_t^2\mathbf{I}\right)
$$

## DDPM: the simple loss

Ho, Jain and Abbeel (2020) — **Denoising Diffusion Probabilistic Models** — showed that the variational bound on the likelihood simplifies beautifully if the network $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$ **predicts the noise** that was added. The training objective becomes:

$$
\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}}\left[\big\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta\big(\sqrt{\bar{\alpha}_t}\,\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\,\boldsymbol{\epsilon},\; t\big)\big\|^2\right]
$$

Training loop: pick a random image, a random timestep and random noise; noise the image in one shot; ask the network to predict the noise; minimise MSE. No adversarial game, no mode collapse — just regression. The network is typically a **U-Net** with timestep embeddings (and attention layers), or a transformer (**DiT**).

## Sampling

Given the predicted noise, the reverse mean is

$$
\boldsymbol{\mu}_\theta(\mathbf{x}_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}}\,\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\right)
$$

and we sample $\mathbf{x}_{t-1} = \boldsymbol{\mu}_\theta(\mathbf{x}_t, t) + \sigma_t\mathbf{z}$ (with $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$, and $\sigma_t^2 = \beta_t$ a common choice), for $t = T, \dots, 1$.

```python
import torch
import torch.nn as nn

T = 1000
betas = torch.linspace(1e-4, 0.02, T)
alphas = 1 - betas
alpha_bar = torch.cumprod(alphas, dim=0)

class Denoiser(nn.Module):                          # a tiny MLP denoiser for 2-D toy data
    def __init__(self, dim=2, hidden=256):
        super().__init__()
        self.t_emb = nn.Embedding(T, hidden)
        self.net = nn.Sequential(nn.Linear(dim + hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, dim))
    def forward(self, x, t):
        return self.net(torch.cat([x, self.t_emb(t)], dim=1))

def training_loss(model, x0):
    t = torch.randint(0, T, (x0.size(0),))
    eps = torch.randn_like(x0)
    ab = alpha_bar[t].unsqueeze(1)
    xt = ab.sqrt() * x0 + (1 - ab).sqrt() * eps     # jump straight to step t
    return ((eps - model(xt, t)) ** 2).mean()       # predict the noise

@torch.no_grad()
def sample(model, n=1000, dim=2):
    x = torch.randn(n, dim)
    for t in reversed(range(T)):
        tt = torch.full((n,), t, dtype=torch.long)
        eps = model(x, tt)
        mean = (x - betas[t] / (1 - alpha_bar[t]).sqrt() * eps) / alphas[t].sqrt()
        x = mean + (betas[t].sqrt() * torch.randn_like(x) if t > 0 else 0)
    return x

from sklearn.datasets import make_moons
data = torch.tensor(make_moons(8000, noise=0.03)[0], dtype=torch.float32)
model = Denoiser(); opt = torch.optim.Adam(model.parameters(), 1e-3)
for step in range(3000):
    loss = training_loss(model, data[torch.randint(0, len(data), (512,))])
    opt.zero_grad(); loss.backward(); opt.step()
print("generated:", sample(model, 5))
```

Plot the samples: they trace out the two moons, although the model never saw a formula for moons.

## The score-based view

Song and Ermon (2019) and Song et al. (2021) connected diffusion to **score matching**. The **score** is the gradient of the log-density, $\nabla_{\mathbf{x}}\log p(\mathbf{x})$ — it points towards regions of higher probability. The noise-prediction network estimates the score of the noised data distribution:

$$
\nabla_{\mathbf{x}_t}\log q(\mathbf{x}_t) \approx -\frac{\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)}{\sqrt{1 - \bar{\alpha}_t}}
$$

In continuous time, the forward process is a **stochastic differential equation (SDE)**, and generation solves the reverse-time SDE — or an equivalent deterministic **probability-flow ODE**, which also enables exact likelihood computation. This unified view explains why diffusion models work: they learn the direction to move a noisy sample towards the data manifold at every noise level.

## Faster sampling

A thousand network evaluations per image is slow. Improvements:

- **DDIM** (Song, Meng & Ermon, 2021): a non-Markovian, deterministic sampler using the same trained model, producing good samples in 20–50 steps and enabling latent interpolation and inversion.
- **Higher-order ODE solvers** (DPM-Solver, Heun/EDM samplers): high quality in 10–25 steps.
- **Distillation**: progressive distillation, **consistency models** and adversarial distillation produce images in 1–4 steps.

## Why diffusion won

- **Stable training** — a simple regression loss, no adversarial dynamics.
- **Excellent mode coverage and diversity**, with sample quality that surpassed GANs on standard benchmarks (Dhariwal & Nichol, 2021, "Diffusion Models Beat GANs on Image Synthesis").
- **Flexible conditioning** — text, images, masks, poses — and guidance techniques (next lectures).
- **Scalability** to video, audio, 3-D and scientific data (protein structure generation, weather forecasting).

:::exercise
1. Show that $\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}$ follows from composing the single-step Gaussian transitions.
2. Plot $\sqrt{\bar{\alpha}_t}$ (signal) and $\sqrt{1 - \bar{\alpha}_t}$ (noise) against $t$ for linear and cosine schedules.
3. Train the toy model, then implement DDIM sampling with 50 steps and compare sample quality and time with the 1000-step sampler.
:::

:::takeaway
- The forward process adds Gaussian noise; any step can be sampled in closed form from $\mathbf{x}_0$.
- DDPM trains a network to predict the added noise with a simple MSE loss, then samples by iterative denoising.
- The noise predictor estimates the score; continuous-time SDE/ODE views unify diffusion with score-based models.
- DDIM, better solvers and distillation make sampling fast; stable training and diversity made diffusion dominant.
:::

=== POST ===
slug: latent-diffusion-stable-diffusion
title: "Latent Diffusion and Stable Diffusion: Text-to-Image at Scale"
category: generative-ai
level: Advanced
tags: stable diffusion, latent diffusion, text-to-image, cross-attention, controlnet
summary: Running diffusion in a compressed latent space made high-resolution text-to-image generation affordable. We dissect latent diffusion — autoencoder, U-Net with cross-attention, text encoder — and practical techniques like img2img, inpainting, ControlNet and fine-tuning.
---
Pixel-space diffusion at high resolution is expensive: every denoising step processes hundreds of thousands of pixel values. In 2022, Rombach, Blattmann, Lorenz, Esser and Ommer published **Latent Diffusion Models (LDMs)**, which run diffusion in the compact latent space of an autoencoder. Released openly as **Stable Diffusion**, it made high-quality text-to-image generation run on consumer GPUs and sparked an explosion of creative tools, research — and debate.

## Architecture overview

A latent diffusion model has three components:

1. **Autoencoder (VAE)**: the encoder $\mathcal{E}$ compresses an image $\mathbf{x}$ (e.g. $512 \times 512 \times 3$) into a latent $\mathbf{z} = \mathcal{E}(\mathbf{x})$ (e.g. $64 \times 64 \times 4$ — a factor of 48 fewer values); the decoder $\mathcal{D}$ maps latents back to images. It is trained once, with reconstruction, perceptual and adversarial losses plus a small KL penalty, so that latents keep perceptually important detail.
2. **Denoising network**: a U-Net (or transformer) performing diffusion **in latent space**, with the usual noise-prediction objective:

$$
\mathcal{L}_{\text{LDM}} = \mathbb{E}_{\mathcal{E}(\mathbf{x}), \mathbf{c}, \boldsymbol{\epsilon}, t}\left[\big\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{z}_t, t, \tau_\theta(\mathbf{c}))\big\|^2\right]
$$

3. **Conditioning encoder** $\tau_\theta$: for text-to-image, a text encoder (a CLIP text encoder in Stable Diffusion 1.x; larger and multiple encoders in later versions) turns the prompt into a sequence of embeddings.

## Conditioning through cross-attention

The U-Net's intermediate feature maps attend to the text embeddings via **cross-attention**:

$$
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d}}\right)\mathbf{V}, \quad \mathbf{Q} = \mathbf{W}_Q\,\varphi(\mathbf{z}_t), \; \mathbf{K} = \mathbf{W}_K\,\tau_\theta(\mathbf{c}), \; \mathbf{V} = \mathbf{W}_V\,\tau_\theta(\mathbf{c})
$$

Each spatial location "looks up" the relevant words of the prompt. Visualising these cross-attention maps shows the word "dog" attending to the dog's region — the basis of attention-based editing methods like Prompt-to-Prompt.

## Generation pipeline

1. Encode the prompt with the text encoder.
2. Sample a random latent $\mathbf{z}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$.
3. Denoise for 20–50 steps with a fast sampler, using **classifier-free guidance** (next lecture) to strengthen prompt adherence.
4. Decode $\mathbf{z}_0$ with the VAE decoder into the final image.

```python
# pip install diffusers transformers accelerate
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

pipe = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5",
                                               torch_dtype=torch.float16).to("cuda")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)   # fast sampler
image = pipe("a watercolour illustration of children reading under a tree, soft morning light",
             negative_prompt="blurry, distorted, text",
             num_inference_steps=25, guidance_scale=7.0,
             generator=torch.Generator("cuda").manual_seed(42)).images[0]
image.save("reading_tree.png")
```

(Check the licence of any checkpoint you use; model licences often include use restrictions.)

## Beyond text-to-image

- **Image-to-image (img2img / SDEdit)**: encode an input image, add noise to an intermediate step, then denoise with a new prompt — the noise strength trades faithfulness to the input against creativity.
- **Inpainting**: regenerate only a masked region, conditioned on the rest.
- **ControlNet** (Zhang et al., 2023): a trainable copy of the encoder, connected through zero-initialised convolutions, adds spatial conditions — edges, depth maps, human poses, segmentation maps — while keeping the base model frozen. It gives precise control over composition.
- **Personalisation**: **DreamBooth** fine-tunes on a few images of a subject bound to a rare token; **Textual Inversion** learns a new word embedding; **LoRA** adapters fine-tune efficiently and are easily shared.
- **Super-resolution and upscaling** with dedicated diffusion upscalers.
- **Video**: extending latents with a time dimension and temporal attention.

## Architecture evolution

Later models replaced parts of the recipe: larger and multiple text encoders (including T5-style encoders that improve text rendering and prompt understanding), higher-resolution training, and **transformer backbones** (Diffusion Transformers, DiT; Multimodal DiT) trained with **flow matching** / rectified flow. The core idea — generate in a learned latent space — has remained.

## Limitations and responsibility

:::warning
Text-to-image models reflect and can amplify **biases** in web-scraped training data (e.g. stereotyped depictions of professions, nationalities and genders). They can generate misleading imagery, non-consensual depictions of real people and copyrighted characters or styles. Training on scraped images has prompted lawsuits and debates over consent and compensation for artists. Responsible deployment includes safety filters, provenance metadata and watermarks, clear usage policies, respect for opt-outs, and never presenting generated images as documentary evidence — especially in humanitarian and news contexts.
:::

:::exercise
1. Compute the compression factor from a $512 \times 512 \times 3$ image to a $64 \times 64 \times 4$ latent, and explain why this speeds up diffusion.
2. Generate the same prompt with guidance scales 1, 4, 7 and 15. Describe how prompt adherence and image quality change.
3. Use img2img with noise strengths 0.3, 0.6 and 0.9 on one of your photos and explain the results.
:::

:::takeaway
- Latent diffusion runs diffusion in a compressed autoencoder latent space — far cheaper than pixel space.
- A U-Net (or DiT) denoiser is conditioned on text embeddings via cross-attention.
- img2img, inpainting, ControlNet, DreamBooth and LoRA extend control and personalisation.
- Bias, misuse and copyright concerns require safeguards, provenance and clear policies.
:::

=== POST ===
slug: classifier-free-guidance
title: "Guidance in Diffusion Models: Classifier and Classifier-Free Guidance"
category: generative-ai
level: Advanced
tags: classifier-free guidance, conditional diffusion, guidance scale, diffusion
summary: Conditional diffusion models often ignore their prompt. Guidance amplifies the condition. We derive classifier guidance from Bayes' rule, then classifier-free guidance, and analyse the fidelity–diversity trade-off controlled by the guidance scale.
---
A conditional diffusion model trained on image–caption pairs learns $p(\mathbf{x} \mid \mathbf{c})$. Sampled naively, its images are often only loosely related to the prompt and of mediocre quality. **Guidance** techniques steer sampling towards images that strongly match the condition. **Classifier-free guidance (CFG)** is arguably the single most important trick behind the quality of modern text-to-image and text-to-video systems.

## Score functions and conditioning

Recall that a diffusion model estimates the score $\nabla_{\mathbf{x}_t}\log p(\mathbf{x}_t)$ at each noise level. For conditional generation we want the conditional score. By Bayes' rule, $p(\mathbf{x}_t \mid \mathbf{c}) \propto p(\mathbf{c} \mid \mathbf{x}_t)\,p(\mathbf{x}_t)$, so

$$
\nabla_{\mathbf{x}_t}\log p(\mathbf{x}_t \mid \mathbf{c}) = \nabla_{\mathbf{x}_t}\log p(\mathbf{x}_t) + \nabla_{\mathbf{x}_t}\log p(\mathbf{c} \mid \mathbf{x}_t)
$$

The conditional score is the unconditional score plus the gradient of a "classifier" that tells how well $\mathbf{x}_t$ matches the condition.

## Classifier guidance

Dhariwal and Nichol (2021) trained a separate classifier $p_\phi(\mathbf{c} \mid \mathbf{x}_t)$ **on noisy images** and scaled its gradient by a **guidance weight** $s$:

$$
\nabla\log p_s(\mathbf{x}_t \mid \mathbf{c}) = \nabla\log p(\mathbf{x}_t) + s\,\nabla\log p_\phi(\mathbf{c} \mid \mathbf{x}_t)
$$

This corresponds to sampling from a sharpened distribution $\propto p(\mathbf{x})\,p(\mathbf{c} \mid \mathbf{x})^s$. Larger $s$ yields images that are more clearly of the class and higher in perceived quality, but less diverse. Drawbacks: you must train an extra classifier on noisy data, and its gradients can behave like adversarial perturbations.

## Classifier-free guidance

Ho and Salimans (2022) removed the classifier. Train **one** network to predict noise both **with** and **without** the condition, by randomly replacing the condition with a null token $\varnothing$ (e.g. an empty caption) for some fraction of training examples (often 10–20%). At sampling time, combine the two predictions:

$$
\tilde{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, \mathbf{c}) = \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, \varnothing) + w\,\big(\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, \mathbf{c}) - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, \varnothing)\big)
$$

- $w = 0$: unconditional generation.
- $w = 1$: plain conditional generation.
- $w > 1$: **extrapolate** beyond the conditional prediction, in the direction that distinguishes "with prompt" from "without prompt".

Why does this work? Since noise predictions are proportional to (negative) scores, the difference $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, \mathbf{c}) - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, \varnothing)$ is proportional to $-\nabla\log p(\mathbf{x}_t \mid \mathbf{c}) + \nabla\log p(\mathbf{x}_t) = -\nabla\log p(\mathbf{c} \mid \mathbf{x}_t)$ — an **implicit classifier** gradient. CFG therefore approximates classifier guidance using the diffusion model itself.

```python
import torch

def cfg_noise(model, x_t, t, cond_emb, null_emb, w=7.0):
    """One classifier-free-guided noise prediction (batched: unconditional + conditional)."""
    x_in = torch.cat([x_t, x_t])
    t_in = torch.cat([t, t])
    c_in = torch.cat([null_emb, cond_emb])
    eps_uncond, eps_cond = model(x_in, t_in, c_in).chunk(2)
    return eps_uncond + w * (eps_cond - eps_uncond)

# Training-side: condition dropout
def maybe_drop_condition(cond_emb, null_emb, p_drop=0.1):
    drop = torch.rand(cond_emb.size(0), *[1] * (cond_emb.dim() - 1)) < p_drop
    return torch.where(drop, null_emb.expand_as(cond_emb), cond_emb)
```

CFG doubles the cost of each sampling step (two forward passes, usually batched together).

## The guidance scale trade-off

| Guidance scale $w$ | Effect |
|---|---|
| 1 | Diverse but often weakly aligned, lower perceived quality |
| 3–8 | Typical sweet spot for text-to-image |
| 10–20 | Strong prompt adherence, less diversity, oversaturated colours and artefacts |

Increasing $w$ moves along a **fidelity–diversity curve**: FID first improves, then worsens as diversity collapses, while CLIP-score (prompt alignment) keeps increasing. Practitioners tune $w$ per model and use case.

## Negative prompts

Replace the null condition with a **negative prompt** $\mathbf{c}_{\text{neg}}$:

$$
\tilde{\boldsymbol{\epsilon}} = \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, \mathbf{c}_{\text{neg}}) + w\,\big(\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, \mathbf{c}) - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, \mathbf{c}_{\text{neg}})\big)
$$

Sampling moves towards the prompt **and away from** the negative prompt ("blurry, extra fingers, watermark").

## Refinements

- **Guidance rescaling / dynamic thresholding** (Imagen) prevents oversaturation at high $w$ by clipping or rescaling predicted pixel values.
- **Guidance intervals**: apply guidance only in the middle range of noise levels, improving diversity.
- **Guidance distillation**: train a student to reproduce guided outputs in a single forward pass, halving cost.
- **Autoguidance**: guide with a weaker version of the model instead of an unconditional one.
- CFG applies beyond images — text-to-audio, text-to-video, and even language models (e.g. to strengthen adherence to instructions or context).

:::note
CFG illustrates a recurring theme in generative AI: models trained purely by likelihood often produce samples that are "too diverse" for human taste, including low-quality regions of the distribution. Techniques that sharpen the distribution — guidance, truncation in GANs, lower temperature in language models — trade diversity for perceived quality.
:::

:::exercise
1. Derive the relationship between the noise prediction and the score, and show that the CFG difference term approximates the gradient of an implicit classifier.
2. Train the toy 2-D diffusion model from the diffusion lecture conditionally on two classes (the two moons) with condition dropout, and sample with $w = 0, 1, 3, 8$.
3. Measure CLIP score and a diversity metric for 50 images per guidance scale on a text-to-image model; plot the trade-off.
:::

:::takeaway
- Conditional score = unconditional score + gradient of $\log p(\mathbf{c} \mid \mathbf{x}_t)$ (Bayes' rule).
- Classifier guidance uses an external noisy-image classifier scaled by $s$.
- Classifier-free guidance trains one model with condition dropout and extrapolates between conditional and unconditional predictions with weight $w$.
- Higher $w$ increases prompt adherence and perceived quality but reduces diversity; negative prompts steer away from unwanted content.
:::
