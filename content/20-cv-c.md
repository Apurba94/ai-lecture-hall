=== POST ===
slug: self-supervised-vision-simclr-dino-mae
title: "Self-Supervised Vision: SimCLR, MoCo, DINO and Masked Autoencoders"
category: computer-vision
level: Advanced
tags: self-supervised learning, contrastive learning, simclr, dino, mae
summary: Labels are expensive, images are abundant. Self-supervised methods learn visual representations from unlabelled images through contrastive learning, self-distillation or masked reconstruction. We compare the main families and how to use them.
---
ImageNet's 1.3 million labelled images took years of human effort. Meanwhile, billions of unlabelled images exist. **Self-supervised learning (SSL)** designs "pretext" tasks whose labels come from the data itself, so a network can learn general-purpose visual features without human annotation. By around 2020–2021, self-supervised features rivalled and then, in many settings, surpassed supervised ImageNet pretraining for transfer learning.

## Early pretext tasks

Predicting image **rotations** (0°, 90°, 180°, 270°), solving **jigsaw puzzles** of shuffled patches, **colourising** grayscale images, predicting relative patch positions. These worked to a degree, but the learned features were tied to the pretext task.

## Contrastive learning

The key idea: two **augmented views** of the same image should have similar representations (**positives**), while views of different images should differ (**negatives**).

### SimCLR
Chen et al. (2020):

1. Create two random augmentations of each image (crop + resize, colour distortion, blur).
2. Encode with a backbone $f$ (e.g. ResNet-50) and a small **projection head** $g$ (an MLP): $\mathbf{z} = g(f(\mathbf{x}))$.
3. Apply the **NT-Xent (InfoNCE) loss**: for a positive pair $(i, j)$ in a batch of $2N$ views,

$$
\ell_{i,j} = -\log\frac{\exp(\text{sim}(\mathbf{z}_i, \mathbf{z}_j)/\tau)}{\sum_{k \ne i}\exp(\text{sim}(\mathbf{z}_i, \mathbf{z}_k)/\tau)}
$$

with cosine similarity and temperature $\tau$.
4. After training, discard $g$ and use $f$'s features.

Findings: **strong augmentation composition** (especially crop + colour distortion) is crucial; the projection head improves the representation below it; **large batches** (thousands) provide many negatives.

```python
import torch
import torch.nn.functional as F

def nt_xent(z1, z2, tau=0.2):
    """z1, z2: (N, d) projections of two views of the same N images."""
    z = F.normalize(torch.cat([z1, z2]), dim=1)             # (2N, d)
    sim = z @ z.T / tau
    sim.fill_diagonal_(float("-inf"))                        # exclude self-similarity
    n = z1.size(0)
    targets = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)])   # index of each positive
    return F.cross_entropy(sim, targets)

print(nt_xent(torch.randn(8, 128), torch.randn(8, 128)))
```

### MoCo
He et al. (2020) decoupled the number of negatives from batch size with a **queue** of past embeddings and a slowly updated **momentum encoder** (an exponential moving average of the online encoder), keeping queue embeddings consistent.

## Non-contrastive methods

Can we avoid negatives entirely? Naively, the network could collapse — map every image to the same vector. Methods prevent collapse differently:

- **BYOL** (2020): an online network predicts the target network's representation of another view; the target is a momentum (EMA) copy, and an extra **predictor** head creates asymmetry.
- **SimSiam** (2021): like BYOL without momentum; a **stop-gradient** on one branch is essential.
- **Barlow Twins / VICReg**: make the cross-correlation matrix of two views' embeddings close to identity — decorrelating dimensions prevents collapse.
- **DINO** (Caron et al., 2021): **self-distillation** with no labels. A student network matches the output distribution of a momentum teacher across different crops (the teacher sees large global crops, the student also sees small local crops). **Centring** and **sharpening** of teacher outputs prevent collapse. Remarkably, DINO-trained ViTs' attention maps segment objects without any supervision. **DINOv2** (2023) scaled this with curated data and produced strong general-purpose features usable frozen for classification, segmentation and depth estimation.

## Masked image modelling

Inspired by BERT's masked language modelling:

- **MAE** (He et al., 2022): mask a **large** fraction (~75%) of ViT patches, encode only the visible patches (cheap), and reconstruct the missing **pixels** with a lightweight decoder. The high mask ratio makes the task non-trivial (images are spatially redundant) and training efficient. MAE features fine-tune excellently.
- **BEiT, SimMIM**: predict discrete tokens or pixels of masked patches.
- **I-JEPA** (2023): predict the **representations** (not pixels) of masked regions from context, focusing on semantics rather than low-level detail.

## Comparing the families

| Family | Examples | Signal | Needs negatives | Notes |
|---|---|---|---|---|
| Contrastive | SimCLR, MoCo | Invariance across views | Yes | Strong linear-probe features; batch/queue size matters |
| Self-distillation | BYOL, DINO | Match EMA teacher | No | DINO attention maps localise objects |
| Redundancy reduction | Barlow Twins, VICReg | Decorrelate dimensions | No | Simple, stable |
| Masked modelling | MAE, BEiT, I-JEPA | Reconstruct/predict masked content | No | Excellent fine-tuning; efficient with ViTs |

## Evaluating representations

- **Linear probing**: freeze the backbone, train a linear classifier on top — measures how linearly separable the features are.
- **k-NN evaluation**: classify by nearest neighbours in feature space — no training needed.
- **Fine-tuning** and **transfer** to detection, segmentation and low-label regimes.

## Why this matters in practice

Self-supervised pretraining lets you exploit **your own unlabelled domain data**: thousands of unlabelled satellite tiles, X-rays, microscopy slides or field photos. Pretrain (or continue pretraining a public model) on them, then fine-tune on a small labelled set. In specialised domains this frequently beats ImageNet-pretrained initialisation.

:::exercise
1. Implement SimCLR on CIFAR-10 with a ResNet-18 for a few epochs and evaluate with a linear probe.
2. Visualise DINO ViT attention maps from the [CLS] token for several images using a pretrained checkpoint.
3. Compare linear probes on frozen DINOv2 features versus supervised ImageNet features for a small dataset of your domain.
:::

:::takeaway
- SSL learns visual features from unlabelled images via pretext tasks.
- Contrastive methods (SimCLR, MoCo) pull positive views together and push negatives apart with InfoNCE.
- Non-contrastive methods (BYOL, DINO, VICReg) avoid collapse with EMA teachers, stop-gradients or decorrelation.
- Masked image modelling (MAE) reconstructs hidden patches; SSL on in-domain data is powerful for specialised applications.
:::

=== POST ===
slug: clip-vision-language
title: "CLIP: Connecting Images and Language"
category: computer-vision
level: Intermediate
tags: clip, vision-language, contrastive learning, zero-shot, multimodal
summary: CLIP learns a shared embedding space for images and text from hundreds of millions of image–caption pairs. We explain its contrastive training, zero-shot classification with prompts, retrieval, limitations and its role in generative models.
---
Traditional classifiers recognise a fixed list of classes chosen before training. To add "solar panel" or "water tank", you must collect labelled data and retrain. In 2021 OpenAI's **CLIP** (Contrastive Language–Image Pre-training) showed a different path: learn from **natural-language captions** of images on the web, and you can recognise almost any concept that can be described in words — **zero-shot**, with no task-specific training.

## Training objective

CLIP has two encoders:

- an **image encoder** (ResNet or ViT) producing $\mathbf{u}_i$;
- a **text encoder** (transformer) producing $\mathbf{v}_i$;

each followed by a linear projection into a shared space and L2 normalisation. It was trained on about **400 million** image–text pairs collected from the internet.

For a batch of $N$ pairs, compute the $N \times N$ matrix of cosine similarities $s_{ij} = \mathbf{u}_i^\top\mathbf{v}_j / \tau$ (with learned temperature $\tau$). The $N$ matching pairs on the diagonal are positives; all $N^2 - N$ mismatched pairs are negatives. The loss is a **symmetric cross-entropy**: classify the correct caption for each image, and the correct image for each caption.

```python
import torch
import torch.nn.functional as F

def clip_loss(img_emb, txt_emb, logit_scale):
    img, txt = F.normalize(img_emb, dim=-1), F.normalize(txt_emb, dim=-1)
    logits = logit_scale * img @ txt.T                      # (N, N)
    labels = torch.arange(len(img))
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2

print(clip_loss(torch.randn(16, 512), torch.randn(16, 512), logit_scale=torch.tensor(100.0)))
```

Captions describe objects, scenes, actions, styles and attributes, so the model learns a broad visual vocabulary aligned with language.

## Zero-shot classification

To classify an image among classes {cat, dog, tent}:

1. Turn each class into a **prompt**: "a photo of a cat", "a photo of a dog", "a photo of a tent".
2. Embed the prompts with the text encoder and the image with the image encoder.
3. Predict the class whose text embedding has the highest cosine similarity with the image.

The text embeddings act as the weights of a classifier generated on the fly from language. CLIP's zero-shot ImageNet accuracy matched the original ResNet-50 trained with 1.28 million labelled examples — without using any of them.

```python
# pip install open_clip_torch
import torch, open_clip
from PIL import Image

model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
tokenizer = open_clip.get_tokenizer("ViT-B-32")
classes = ["a flooded road", "a dry road", "a collapsed building", "an intact building"]
image = preprocess(Image.open("scene.jpg")).unsqueeze(0)
text = tokenizer([f"a satellite photo of {c}" for c in classes])
with torch.no_grad():
    img_f = model.encode_image(image); txt_f = model.encode_text(text)
    img_f /= img_f.norm(dim=-1, keepdim=True); txt_f /= txt_f.norm(dim=-1, keepdim=True)
    probs = (100 * img_f @ txt_f.T).softmax(dim=-1)
print(dict(zip(classes, probs[0].round(decimals=3).tolist())))
```

## Prompt engineering and ensembling

Wording matters. "A photo of a {label}" usually beats the bare label. For satellite images, "a satellite photo of {label}" helps. **Prompt ensembling** — averaging text embeddings over many templates ("a blurry photo of", "a close-up photo of", "an origami") — improves accuracy further. For some fine-grained domains, adding context ("a type of pet") resolves ambiguity.

## Other capabilities

- **Image–text retrieval**: search images by text, or find captions for images.
- **Linear probes**: CLIP image features with a simple classifier are strong for many tasks with few labels.
- **Robustness**: CLIP's zero-shot models were notably more robust to certain natural distribution shifts (sketches, renditions) than ImageNet-trained models of similar accuracy.
- **Foundation for generation**: CLIP-style text encoders guide text-to-image diffusion models; CLIP scores are used to evaluate image–text alignment.
- **Open-vocabulary detection and segmentation** build on CLIP-like embeddings.

## Limitations

:::warning
- **Weak at counting, spatial relations and fine-grained or specialised domains** (medical images, specific species, satellite analysis) unless adapted.
- **Typographic attacks**: an apple with a paper label reading "iPod" may be classified as an iPod — the model reads text in images.
- **Biases from web data**: CLIP has been shown to associate demographic groups with harmful labels. The original paper itself documented such issues. Avoid using zero-shot CLIP labels for decisions about people without careful evaluation.
- **Distribution of web data** under-represents many regions, languages and contexts.
:::

## Successors

Open reproductions (OpenCLIP trained on LAION datasets), **ALIGN** (noisier but larger data), **SigLIP** (a sigmoid pairwise loss that removes the need for batch-wide softmax normalisation and works well at small batch sizes), multilingual CLIP variants, and domain-specific versions for medicine and remote sensing. CLIP-style vision encoders are the visual front end of most modern multimodal large language models.

:::exercise
1. Evaluate zero-shot CLIP on a small labelled dataset from your domain with three prompt templates and with an ensemble. Compare accuracy.
2. Build a text-to-image search engine over 1,000 of your own photos using CLIP embeddings.
3. Train a logistic regression on CLIP image features with 5, 20 and 100 labels per class and compare with zero-shot accuracy.
:::

:::takeaway
- CLIP trains image and text encoders contrastively on image–caption pairs into a shared space.
- Zero-shot classification compares an image with text prompts for each class; prompt wording and ensembling matter.
- CLIP enables retrieval, robust transfer, open-vocabulary recognition and text-to-image guidance.
- It struggles with counting, specialised domains and typographic attacks, and carries web-data biases.
:::

=== POST ===
slug: human-pose-estimation
title: Human Pose Estimation
category: computer-vision
level: Intermediate
tags: pose estimation, keypoints, heatmaps, openpose, action recognition
summary: Pose estimation locates body joints in images and video. We cover keypoint heatmap regression, top-down versus bottom-up approaches, part affinity fields, evaluation with OKS, 3-D pose, and applications from health to sport.
---
Where are a person's shoulders, elbows, wrists, hips, knees and ankles? **Human pose estimation** answers this by locating **keypoints** (joints) and connecting them into a skeleton. It supports physiotherapy and rehabilitation apps that check exercise form, fall detection for elderly care, sports analysis, animation and motion capture without suits, sign-language recognition and human–robot interaction.

## Problem formulation

For each person, predict $K$ keypoints (COCO uses 17: nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles), each with coordinates $(x_k, y_k)$ and a visibility or confidence score.

## Direct regression vs heatmaps

**Direct regression** (DeepPose, 2014) predicts coordinates with a fully connected layer. It is simple but struggles to learn precise spatial mappings.

**Heatmap regression** became standard: for each keypoint, the network outputs a 2-D map $H_k$ whose values peak at the joint's location. The training target is a Gaussian centred on the ground truth:

$$
H_k^*(x, y) = \exp\left(-\frac{(x - x_k)^2 + (y - y_k)^2}{2\sigma^2}\right)
$$

trained with per-pixel MSE. At inference, the keypoint is the argmax of the heatmap (with sub-pixel refinement). Heatmaps preserve spatial structure and let fully convolutional networks exploit locality.

**Architectures** that keep high resolution matter here: **Stacked Hourglass** networks (repeated down/up-sampling with skips), and **HRNet**, which maintains a high-resolution branch throughout while exchanging information with lower-resolution branches.

## Multi-person pose: top-down vs bottom-up

| Approach | How | Pros | Cons |
|---|---|---|---|
| Top-down | Detect each person, crop, run single-person pose estimation | High accuracy | Cost grows with number of people; depends on detector |
| Bottom-up | Detect all keypoints in the image, then group them into people | Constant cost regardless of crowd size | Grouping is hard in crowds |

**OpenPose** (Cao et al., 2017), a landmark bottom-up method, predicts keypoint heatmaps plus **Part Affinity Fields (PAFs)** — 2-D vector fields encoding the direction of limbs between joints. Grouping keypoints into skeletons becomes a matching problem scored by integrating the PAF along candidate limbs. It ran in real time on multiple people.

Modern one-stage methods (e.g. pose heads in YOLO-style detectors) predict boxes and keypoints together, and transformer-based models (e.g. ViTPose) achieve strong accuracy with plain ViT backbones.

## Evaluation: Object Keypoint Similarity

Accuracy is measured with **OKS**, analogous to IoU for keypoints:

$$
\text{OKS} = \frac{\sum_k\exp\left(-\frac{d_k^2}{2s^2\kappa_k^2}\right)\delta(v_k > 0)}{\sum_k\delta(v_k > 0)}
$$

where $d_k$ is the distance between predicted and true keypoint $k$, $s$ is the object scale, and $\kappa_k$ is a per-keypoint constant reflecting annotation variability (hips are harder to place precisely than eyes). AP is then averaged over OKS thresholds, just as detection AP is averaged over IoU thresholds. Single-person benchmarks also use **PCK** (percentage of correct keypoints within a normalised distance).

## Using a pretrained model

```python
import torch
from torchvision.models.detection import keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights
from torchvision.io import read_image
from torchvision.transforms.functional import convert_image_dtype

weights = KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
model = keypointrcnn_resnet50_fpn(weights=weights).eval()
img = convert_image_dtype(read_image("people.jpg"), torch.float)
with torch.no_grad():
    out = model([img])[0]
keep = out["scores"] > 0.8
print("people:", int(keep.sum()), " keypoints tensor:", out["keypoints"][keep].shape)  # (P, 17, 3)
names = weights.meta["keypoint_names"]
print(dict(zip(names[:5], out["keypoints"][keep][0, :5, :2].round().tolist())))
```

## From 2-D to 3-D and to action

- **3-D pose estimation** lifts 2-D keypoints to 3-D, either from multiple calibrated cameras (triangulation) or from a single image by learned priors (a 2-D-to-3-D "lifting" network), often exploiting temporal consistency in video.
- **Body models** such as SMPL represent full 3-D body shape and pose with a compact set of parameters.
- **Action recognition from skeletons**: sequences of keypoints feed temporal models or graph convolutional networks treating the skeleton as a graph (ST-GCN), giving privacy-friendlier activity recognition than raw video.

## Challenges

Occlusion (people behind objects or each other), unusual poses, crowded scenes, motion blur, loose clothing (e.g. long robes hide leg joints), low resolution, and — importantly — **dataset bias**: training sets over-represent certain body types, clothing and activities. Evaluate on the population you intend to serve.

:::note
Pose estimation can analyse movement without storing identifiable images: an app can compute joint angles on the device and keep only the skeleton. This privacy-by-design pattern is valuable in health, education and workplace-safety applications.
:::

:::exercise
1. Write the Gaussian heatmap target for a keypoint and implement argmax decoding with sub-pixel refinement.
2. Run a pretrained pose model on a video and compute the knee angle over time during a squat.
3. Explain why top-down methods slow down in crowds while bottom-up methods do not.
:::

:::takeaway
- Pose estimation locates body keypoints; heatmap regression with high-resolution networks (Hourglass, HRNet) is standard.
- Top-down methods detect people first; bottom-up methods (OpenPose with PAFs) group keypoints afterwards.
- OKS-based AP evaluates keypoint accuracy relative to scale and annotation variability.
- 3-D lifting, body models and skeleton-based action recognition extend pose to rich, privacy-friendly applications.
:::

=== POST ===
slug: face-recognition-and-ethics
title: "Face Recognition: Metric Learning, ArcFace and Responsible Use"
category: computer-vision
level: Advanced
tags: face recognition, metric learning, arcface, triplet loss, ethics
summary: Face recognition maps faces to embeddings where the same person is close. We study the pipeline, triplet and angular-margin losses, verification versus identification, evaluation — and the serious ethical questions this technology raises.
---
Face recognition is among the most commercially deployed — and most controversial — applications of computer vision. It unlocks phones and verifies identity for services, but it also enables mass surveillance and has shown unequal accuracy across demographic groups. As engineers, we must understand both how it works and why its use demands exceptional care.

## The pipeline

1. **Detection** — find faces (e.g. RetinaFace, MTCNN).
2. **Alignment** — detect landmarks (eyes, nose, mouth corners) and warp the face to a canonical pose.
3. **Embedding** — a deep network maps the aligned face to a vector (e.g. 512 dimensions), normalised to unit length.
4. **Comparison** — cosine similarity between embeddings, with a threshold.

## Verification vs identification

- **Verification (1:1)**: "Is this person who they claim to be?" Compare one probe with one enrolled template. Used for phone unlock or document checks.
- **Identification (1:N)**: "Who is this person?" Compare a probe against a gallery of $N$ identities. Error rates grow with $N$, because more gallery faces mean more chances of a false match.

## Why not ordinary classification?

A face system must recognise people **not seen during training** (new users enrol later). We therefore learn an **embedding space** where distances reflect identity — **metric learning** — rather than a fixed classifier.

## Triplet loss

FaceNet (Schroff et al., 2015) trained with triplets: an **anchor** $a$, a **positive** $p$ (same identity) and a **negative** $n$ (different identity):

$$
\mathcal{L} = \max\left(0,\; \|f(a) - f(p)\|^2 - \|f(a) - f(n)\|^2 + m\right)
$$

The margin $m$ forces negatives to be farther than positives by at least $m$. Performance depends heavily on **mining** informative (semi-hard) triplets.

## Angular-margin softmax losses

Later methods train a classifier over training identities but modify the softmax to enforce margins **on the hypersphere**. With normalised embedding $\mathbf{x}$ and normalised class weights $\mathbf{W}_j$, the logit is $s\cos\theta_j$. **ArcFace** (Deng et al., 2019) adds an **additive angular margin** $m$ to the true class:

$$
\mathcal{L} = -\log\frac{e^{s\cos(\theta_{y} + m)}}{e^{s\cos(\theta_{y} + m)} + \sum_{j \ne y}e^{s\cos\theta_j}}
$$

This forces embeddings of the same identity to cluster tightly with clear angular gaps between identities. After training, the classifier is discarded and the embeddings are used. (Related losses: SphereFace, CosFace.)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ArcFaceHead(nn.Module):
    def __init__(self, dim, n_ids, s=64.0, m=0.5):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_ids, dim) * 0.01)
        self.s, self.m = s, m
    def forward(self, emb, labels):
        cos = F.linear(F.normalize(emb), F.normalize(self.W)).clamp(-1 + 1e-7, 1 - 1e-7)
        theta = torch.acos(cos)
        target = torch.cos(theta + self.m)                   # add margin to the true class angle
        onehot = F.one_hot(labels, cos.size(1)).bool()
        logits = torch.where(onehot, target, cos) * self.s
        return F.cross_entropy(logits, labels)

head = ArcFaceHead(512, 1000)
print(head(torch.randn(8, 512), torch.randint(0, 1000, (8,))))
```

## Evaluation

Operating points are defined by thresholds on similarity:

- **False Match Rate (FMR / FAR)** — different people accepted as the same.
- **False Non-Match Rate (FNMR / FRR)** — same person rejected.

Systems report, for example, FNMR at FMR = $10^{-6}$. Benchmarks range from LFW (now saturated) to harder datasets with pose, age and quality variation. Independent evaluations such as NIST's Face Recognition Vendor Tests report accuracy — including **demographic differentials** — across many commercial algorithms.

## Bias and fairness

Studies including Buolamwini and Gebru's *Gender Shades* (2018) found commercial facial-analysis systems had much higher error rates for darker-skinned women than for lighter-skinned men. NIST's 2019 demographic study found that many algorithms showed higher false-positive rates for some demographic groups, with large variation between algorithms. Causes include unbalanced training data and image-capture conditions. A false match in a policing context can lead to wrongful suspicion or arrest — documented cases exist.

## Responsible use

:::warning
Face recognition processes **biometric data** — among the most sensitive personal data, because you cannot change your face. Before building or deploying a system, ask:
- **Is it necessary and proportionate?** Would a less intrusive method (a card, a PIN) suffice?
- **Consent and choice**: can people opt out without losing access to essential services?
- **Legal basis**: data-protection laws (e.g. the GDPR treats biometric data as a special category) and emerging AI regulations restrict or ban certain uses, such as untargeted scraping of facial images and many forms of real-time remote biometric identification in public spaces.
- **Accuracy across groups**: measure FMR and FNMR per demographic group on representative data.
- **Human review**: never allow a match alone to trigger consequential action.
- **Security**: templates must be encrypted and protected; liveness detection must prevent spoofing with photos or masks.
- **Special care with vulnerable populations**: for refugees and displaced people, biometric databases can create serious risks if data is shared, breached or misused. Humanitarian organisations have developed specific data-protection policies for this reason.
:::

Many cities and organisations have restricted facial recognition use by public authorities; several technology companies paused or limited sales to law enforcement. Responsible engineers treat this technology as high-risk by default.

:::exercise
1. Explain why identification error grows with gallery size, using the probability of at least one false match among $N$ comparisons.
2. Train a small embedding network with triplet loss on a public face dataset subset and plot the distribution of same-person and different-person similarities.
3. Write a one-page risk assessment for a proposed face-verification system for aid distribution, including alternatives and safeguards.
:::

:::takeaway
- Face recognition = detection + alignment + embedding + similarity comparison.
- Metric learning (triplet loss, ArcFace angular margins) creates identity-discriminative embeddings for unseen people.
- Evaluate with FMR/FNMR at operating thresholds — per demographic group.
- It processes sensitive biometric data; necessity, consent, fairness, security and human oversight are mandatory considerations.
:::

=== POST ===
slug: video-understanding
title: "Video Understanding: Action Recognition and Temporal Modelling"
category: computer-vision
level: Advanced
tags: video, action recognition, 3d cnn, optical flow, temporal models
summary: Video adds time to vision. We study optical flow, two-stream networks, 3-D convolutions, (2+1)-D factorisation, video transformers and tracking, plus the computational tricks that make video models practical.
---
A single frame shows a person with an arm raised. Are they waving, throwing or reaching? The answer lies in **motion**. Video understanding extends computer vision along the time axis for tasks such as **action recognition**, temporal localisation of events, video captioning, object tracking and anomaly detection in surveillance or industrial footage. Video is data-rich and computationally expensive, which shapes every design choice.

## Representing video

A video clip is a 4-D tensor $T \times H \times W \times C$ — for example, 16 frames of $224 \times 224$ RGB. Consecutive frames are highly redundant, so models usually **sample** frames sparsely (e.g. 8–32 frames spread across the clip).

## Optical flow

**Optical flow** estimates, for each pixel, its apparent motion between two frames. Classical methods rely on the **brightness constancy** assumption: a moving point keeps its intensity, giving

$$
I_x u + I_y v + I_t = 0
$$

for flow $(u, v)$ — one equation in two unknowns (the **aperture problem**). Lucas–Kanade solves it by assuming constant flow in a small window; Horn–Schunck adds a global smoothness term. Deep networks (FlowNet, RAFT) now estimate flow far more accurately.

## Two-stream networks

Simonyan and Zisserman (2014) processed video with two CNNs:

- a **spatial stream** on single RGB frames (appearance);
- a **temporal stream** on stacks of optical-flow fields (motion);

and fused their predictions. Motion information substantially improved action recognition, but computing optical flow is expensive.

## 3-D convolutions

A **3-D convolution** extends kernels across time: a $3 \times 3 \times 3$ kernel slides over frames and pixels, learning spatio-temporal features directly.

- **C3D** (2015) showed 3-D CNNs learn useful video features.
- **I3D** (Carreira & Zisserman, 2017) "inflated" pretrained 2-D ImageNet filters into 3-D (repeating weights along time and rescaling), giving 3-D networks a strong initialisation. Trained on the large **Kinetics** dataset, I3D set new standards.
- **(2+1)-D convolutions** (R(2+1)D, 2018) factorise a 3-D conv into a 2-D spatial conv followed by a 1-D temporal conv — fewer parameters, an extra non-linearity, easier optimisation.
- **SlowFast networks** (2019) use a slow pathway (few frames, many channels) for appearance and a fast pathway (many frames, few channels) for motion — inspired by parallel processing streams in primate vision.

```python
import torch
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights

weights = R2Plus1D_18_Weights.DEFAULT
model = r2plus1d_18(weights=weights).eval()
clip = torch.rand(1, 3, 16, 112, 112)               # (batch, channels, frames, H, W)
with torch.no_grad():
    probs = model(clip).softmax(-1)
top = probs.topk(3)
print([weights.meta["categories"][i] for i in top.indices[0]], top.values[0].round(decimals=3))
```

## Efficient temporal modelling

- **Temporal Segment Networks (TSN)**: sample one frame (or snippet) from each of several segments across the whole video and average predictions — cheap long-range coverage.
- **Temporal Shift Module (TSM)**: shift a fraction of channels forward and backward in time within a 2-D CNN, exchanging information between neighbouring frames at almost zero cost.

## Video transformers

Treat video as a sequence of **spatio-temporal patches ("tubelets")**. Full attention over all patches in all frames is expensive, so architectures factorise it:

- **TimeSformer**: divided attention — temporal attention across frames at the same location, then spatial attention within a frame.
- **ViViT**: several factorised variants.
- **Video Swin**: local 3-D window attention.
- **VideoMAE**: masked autoencoding with very high mask ratios (~90%), exploiting video's redundancy for efficient self-supervised pretraining.

Large multimodal models now also process sampled video frames alongside text for video question answering and captioning.

## Other video tasks

- **Temporal action localisation**: find start and end times of actions in untrimmed videos.
- **Multi-object tracking**: detect objects in each frame and link them over time — "tracking by detection" with motion models (Kalman filters) and appearance embeddings (SORT, DeepSORT, ByteTrack).
- **Video object segmentation** and **video anomaly detection**.

## Datasets and challenges

Kinetics (hundreds of action classes), Something-Something (fine-grained actions requiring temporal reasoning, e.g. "pushing something from left to right" vs "right to left"), AVA (spatio-temporal action localisation), and Ego4D (egocentric video). A known pitfall: many actions in Kinetics can be recognised from **scene context** in a single frame (a swimming pool implies swimming), so strong scores do not always mean a model understands motion. Datasets like Something-Something test genuine temporal reasoning.

:::note
Video raises heightened privacy concerns: it can reveal identities, behaviour and locations over time. Prefer on-device processing, store derived signals (counts, skeletons, events) rather than raw footage where possible, and follow data-minimisation principles.
:::

:::exercise
1. Compute optical flow between two frames with OpenCV's Farnebäck method and visualise it as a colour-coded image.
2. Count the parameters of a $3 \times 3 \times 3$ convolution with 64 input and 64 output channels versus its (2+1)-D factorisation with 144 intermediate channels.
3. Fine-tune a pretrained video model on a small action dataset, comparing 8 and 32 sampled frames.
:::

:::takeaway
- Video adds time; motion is key, captured via optical flow or learned spatio-temporal features.
- Two-stream networks, 3-D and (2+1)-D convolutions, SlowFast and TSM model motion with different trade-offs.
- Video transformers factorise space–time attention; VideoMAE pretrains efficiently.
- Beware scene-context shortcuts in benchmarks; handle video data with strong privacy practices.
:::

=== POST ===
slug: 3d-vision-depth-nerf
title: "3-D Vision: Stereo, Depth Estimation, Point Clouds and NeRF"
category: computer-vision
level: Advanced
tags: 3d vision, depth estimation, stereo, point clouds, nerf, gaussian splatting
summary: Images are 2-D projections of a 3-D world. We cover camera geometry, stereo and monocular depth, structure from motion, point-cloud networks like PointNet, and neural scene representations such as NeRF and Gaussian splatting.
---
Robots must know how far away obstacles are; augmented-reality apps must place virtual objects on real tables; drones map terrain; archaeologists and disaster responders reconstruct buildings in 3-D. **3-D vision** recovers geometric structure from images and other sensors. It combines classical projective geometry — which remains essential — with deep learning.

## The pinhole camera model

A 3-D point $\mathbf{X} = (X, Y, Z)$ in camera coordinates projects to pixel $(u, v)$:

$$
\begin{bmatrix} u \\ v \\ 1 \end{bmatrix} \sim \mathbf{K}\begin{bmatrix} X/Z \\ Y/Z \\ 1 \end{bmatrix}, \qquad \mathbf{K} = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}
$$

$\mathbf{K}$ holds the **intrinsic** parameters (focal lengths, principal point); the camera's rotation $\mathbf{R}$ and translation $\mathbf{t}$ relative to the world are the **extrinsics**. Division by $Z$ is why depth is lost in a single image: every point along a ray projects to the same pixel. **Camera calibration** (e.g. with a checkerboard) estimates $\mathbf{K}$ and lens distortion.

## Stereo vision

Two cameras separated by a **baseline** $B$ see a point at slightly different horizontal positions. The difference, the **disparity** $d = u_L - u_R$, gives depth for rectified cameras:

$$
Z = \frac{f\,B}{d}
$$

Nearby objects have large disparity; distant ones small. Depth precision therefore degrades with distance. The hard part is **stereo matching** — finding corresponding pixels — which is ambiguous in textureless regions, reflections and occlusions. Classical semi-global matching and learned stereo networks (e.g. RAFT-Stereo) address it.

```python
import cv2
left = cv2.imread("left_rectified.png", cv2.IMREAD_GRAYSCALE)
right = cv2.imread("right_rectified.png", cv2.IMREAD_GRAYSCALE)
sgbm = cv2.StereoSGBM_create(minDisparity=0, numDisparities=128, blockSize=5)
disparity = sgbm.compute(left, right).astype("float32") / 16.0
f, B = 700.0, 0.12                                    # focal length (px), baseline (m)
depth = f * B / (disparity + 1e-6)                    # metres, where disparity > 0
```

## Monocular depth estimation

From a **single image**, depth is ambiguous in principle, but humans use cues — perspective, relative size, texture gradients, occlusion, shading. Deep networks learn these cues from data:

- Supervised with depth sensors (LiDAR, RGB-D cameras).
- **Self-supervised** from video or stereo pairs: predict depth and camera motion so that one frame can be warped to reconstruct its neighbour (Monodepth).
- Large models trained on diverse mixed datasets (e.g. MiDaS, Depth Anything) produce robust **relative** depth for arbitrary images; metric depth requires scale information.

## Structure from Motion and SLAM

**Structure from Motion (SfM)** reconstructs 3-D points and camera poses from many overlapping photos: detect and match features (SIFT), estimate relative poses (essential matrix + RANSAC), triangulate points, and refine everything jointly with **bundle adjustment** — minimising total reprojection error. COLMAP is the standard open-source tool. **SLAM** (Simultaneous Localisation and Mapping) does this incrementally and in real time for robots and AR devices.

## Point clouds

LiDAR and depth cameras produce **point clouds** — unordered sets of 3-D points. **PointNet** (Qi et al., 2017) processes them directly: apply a shared MLP to each point, then aggregate with a symmetric function (max pooling), guaranteeing permutation invariance:

$$
f(\{\mathbf{x}_1, \dots, \mathbf{x}_n\}) = \gamma\left(\max_{i}h(\mathbf{x}_i)\right)
$$

**PointNet++** adds hierarchical local grouping; other approaches voxelise space (sparse 3-D convolutions) or project to bird's-eye views, as in autonomous-driving detectors.

## Neural scene representations

**NeRF** (Neural Radiance Fields, Mildenhall et al., 2020) represents a scene as an MLP mapping a 3-D position and viewing direction to colour and density:

$$
F_\Theta: (x, y, z, \theta, \phi) \mapsto (\mathbf{c}, \sigma)
$$

Images are rendered by **volume rendering** along camera rays:

$$
\hat{C}(\mathbf{r}) = \sum_{i=1}^{N}T_i\,\big(1 - e^{-\sigma_i\delta_i}\big)\,\mathbf{c}_i, \qquad T_i = \exp\Big(-\sum_{j<i}\sigma_j\delta_j\Big)
$$

Training minimises the difference between rendered and real pixels from posed photos. **Positional encoding** of inputs with sinusoids lets the MLP represent fine detail. NeRF produced photorealistic novel views but was slow; hash-grid encodings (Instant-NGP) made it fast.

**3-D Gaussian Splatting** (2023) represents scenes as millions of anisotropic 3-D Gaussians with colours and opacities, rendered by fast rasterisation — real-time photorealistic rendering, rapidly adopted in graphics, mapping and cultural-heritage digitisation.

## Applications

Autonomous driving and robotics (depth, 3-D detection), AR/VR, drone mapping and photogrammetry for damage assessment and settlement planning, 3-D medical imaging, digital preservation of heritage sites, and e-commerce product visualisation.

:::exercise
1. With $f = 800$ px and $B = 0.1$ m, compute depths for disparities of 80, 8 and 0.8 px. What does this imply about long-range stereo accuracy?
2. Run a pretrained monocular depth model on several photos and discuss where it fails (glass, mirrors, sky).
3. Reconstruct a small object from 30 phone photos with COLMAP and inspect the sparse point cloud and camera poses.
:::

:::takeaway
- The pinhole model projects 3-D to 2-D with intrinsics $\mathbf{K}$ and extrinsics; depth is lost in projection.
- Stereo depth $Z = fB/d$; monocular depth relies on learned cues; SfM and SLAM recover structure and camera motion.
- PointNet handles unordered point clouds with shared MLPs and symmetric pooling.
- NeRF and Gaussian splatting represent scenes neurally for photorealistic novel-view synthesis.
:::
