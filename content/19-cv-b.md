=== POST ===
slug: image-classification-pipeline
title: Building an Image Classification Pipeline End to End
category: computer-vision
level: Beginner
tags: image classification, transfer learning, dataset, evaluation, pytorch
summary: A practical walkthrough of a real image classifier — collecting and splitting data, preprocessing, choosing a pretrained backbone, training, evaluating per class, inspecting errors and exporting the model.
---
Theory is necessary but not sufficient. Today we build a complete image classification system the way a professional would — for example, a model that classifies photos of crop leaves as healthy or diseased. Every step contains decisions that affect whether the final system works in the field.

## 1. Define the task and collect data

- **Classes**: clearly defined and mutually exclusive (or use multi-label if not).
- **Coverage**: images should reflect **deployment conditions** — phones used by real users, lighting at different times of day, backgrounds, disease stages, crop varieties, regions.
- **Labels**: agreed guidelines; ideally two annotators per image to measure agreement. Label noise directly caps accuracy.
- **Quantity**: with transfer learning, a few hundred images per class can already give a useful model.

Organise images in the conventional folder layout:

```text
data/
  train/healthy/*.jpg   train/leaf_blight/*.jpg   train/rust/*.jpg
  val/...               test/...
```

## 2. Split correctly

Split by **source**, not just randomly: if many photos come from the same field or the same plant, keep them in the same split, otherwise the test set contains near-duplicates of training images and accuracy is inflated. Check for exact and near duplicates (perceptual hashing) across splits.

## 3. Preprocess and augment

Use the preprocessing expected by the pretrained backbone (resize, crop, normalise with ImageNet mean and standard deviation). Augment only the training set:

```python
import torch
from torchvision import datasets, transforms, models

mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.3, 0.3, 0.2),
    transforms.ToTensor(), transforms.Normalize(mean, std)])
eval_tf = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(), transforms.Normalize(mean, std)])

train_ds = datasets.ImageFolder("data/train", train_tf)
val_ds = datasets.ImageFolder("data/val", eval_tf)
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
val_dl = torch.utils.data.DataLoader(val_ds, batch_size=64, num_workers=4)
print(train_ds.classes, len(train_ds), len(val_ds))
```

## 4. Choose a backbone and fine-tune

Start with a strong, efficient pretrained model (ResNet-50, EfficientNet, ConvNeXt-Tiny, or a small ViT; MobileNet for phones). Replace the head, train the head first, then fine-tune the whole network with a lower learning rate.

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
model.classifier[2] = torch.nn.Linear(model.classifier[2].in_features, len(train_ds.classes))
model.to(device)

def train(epochs, params, lr):
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * len(train_dl))
    for epoch in range(epochs):
        model.train()
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            loss = torch.nn.functional.cross_entropy(model(x), y, label_smoothing=0.1)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        print(epoch, "val acc:", evaluate())

@torch.no_grad()
def evaluate():
    model.eval(); correct = total = 0
    for x, y in val_dl:
        pred = model(x.to(device)).argmax(1).cpu()
        correct += (pred == y).sum().item(); total += len(y)
    return round(correct / total, 4)

for p in model.features.parameters():
    p.requires_grad = False
train(3, model.classifier.parameters(), 1e-3)          # stage 1: head only
for p in model.parameters():
    p.requires_grad = True
train(10, model.parameters(), 1e-4)                    # stage 2: full fine-tuning
```

## 5. Evaluate beyond accuracy

- **Confusion matrix** — which diseases are confused?
- **Per-class precision and recall** — a rare but dangerous disease needs high recall.
- **Calibration** — are confidence scores meaningful? Consider temperature scaling.
- **Slices** — performance by phone model, region, lighting, crop variety.
- **Test-time augmentation** (averaging predictions over flips/crops) can add a little accuracy.

## 6. Inspect errors

Look at misclassified images and the highest-confidence mistakes. Typical findings: mislabelled training data, blurry photos, multiple diseases in one image, backgrounds the model latched onto (a **shortcut**: e.g. all "diseased" photos taken on one field's soil). Tools like **Grad-CAM** (covered later) show which regions drive a prediction — useful for detecting shortcuts.

:::warning
Shortcut learning is common in vision. Famous examples include classifiers detecting hospital-specific markers on X-rays instead of pathology, and "wolf vs husky" models relying on snow in the background. Always check what your model is looking at, and test on data from new sources.
:::

## 7. Handle "none of the above"

Deployed models will receive photos that belong to no class (a hand, a car, a blurry image). Options: add an "other/unknown" class with diverse examples, threshold on confidence, or use out-of-distribution detection. Design the user interface to request a retake when confidence is low.

## 8. Export and monitor

Export to ONNX, TorchScript or TFLite; quantise for mobile; version the model with its class list and preprocessing; log predictions and confidence in production to detect drift; collect corrected labels from experts to retrain.

:::exercise
1. Build the pipeline on a public plant-disease or flower dataset and report per-class metrics.
2. Compare a random split with a source-grouped split. How much does accuracy change?
3. Add an "unknown" class using random images from another dataset and evaluate how often out-of-distribution images are rejected.
:::

:::takeaway
- Collect data that matches deployment conditions; split by source to avoid leakage.
- Use pretrained backbones with matched preprocessing; fine-tune in two stages.
- Evaluate per class, per slice and for calibration; inspect errors for shortcuts.
- Plan for out-of-distribution inputs, export efficiently and monitor in production.
:::

=== POST ===
slug: data-augmentation-for-vision
title: Data Augmentation for Computer Vision
category: computer-vision
level: Intermediate
tags: data augmentation, randaugment, mixup, cutmix, albumentations
summary: Augmentation multiplies your data by encoding known invariances. We survey geometric, photometric and occlusion augmentations, automated policies like RandAugment, mixing methods, and augmentation for detection and segmentation.
---
In vision, data augmentation is often the single most effective regulariser. By transforming training images in ways that preserve their labels, we teach the model the **invariances** we know the task has — a leaf is still diseased when rotated, a car is still a car in dim light. Good augmentation can be worth as much as doubling the dataset.

## Categories of augmentation

**Geometric**
- Random resized crops (scale and aspect-ratio jitter) — the most important single augmentation for classification.
- Horizontal (sometimes vertical) flips, small rotations, translations, shear.
- Perspective warps and elastic deformations (useful for handwriting and medical images).

**Photometric**
- Brightness, contrast, saturation and hue jitter.
- Gaussian blur, noise, JPEG compression artefacts.
- Grayscale conversion, gamma changes, simulated shadows, fog or rain.

**Occlusion**
- **Random erasing / Cutout**: blank out rectangles, forcing the model to use multiple cues rather than one discriminative patch.

**Mixing**
- **Mixup**: blend two images and their labels.
- **CutMix**: paste a patch from one image into another; mix labels by area.
- **Mosaic** (popular in YOLO): tile four images into one, exposing objects at varied scales and contexts.

## Choosing augmentations: preserve the label

:::warning
An augmentation is valid only if the label remains correct:
- Vertical flips are fine for satellite and microscopy images, wrong for street scenes (the sky is not below).
- Horizontal flips can invert meaning in text, traffic signs with arrows, or left–right anatomy.
- Strong colour jitter can erase the evidence in colour-dependent tasks (ripeness of fruit, skin lesions, leaf yellowing).
- Aggressive crops can remove the object entirely.
Always visualise augmented batches before training.
:::

## Automated augmentation policies

Choosing magnitudes by hand is tedious. **AutoAugment** (Cubuk et al., 2019) searched for augmentation policies with reinforcement learning — effective but expensive. **RandAugment** (2020) simplified this to two hyperparameters: apply $N$ random operations from a fixed list, each with global magnitude $M$. **TrivialAugment** (2021) goes further: one random operation with a random magnitude per image — and performs comparably. These are standard in modern training recipes.

```python
from torchvision import transforms

train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.TrivialAugmentWide(),              # or transforms.RandAugment(num_ops=2, magnitude=9)
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.25),
])
```

## Mixup and CutMix in practice

```python
import numpy as np
import torch

def cutmix(x, y, num_classes, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0))
    H, W = x.shape[2:]
    rh, rw = int(H * np.sqrt(1 - lam)), int(W * np.sqrt(1 - lam))
    cy, cx = np.random.randint(H), np.random.randint(W)
    y1, y2 = np.clip(cy - rh // 2, 0, H), np.clip(cy + rh // 2, 0, H)
    x1, x2 = np.clip(cx - rw // 2, 0, W), np.clip(cx + rw // 2, 0, W)
    x = x.clone(); x[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam = 1 - (y2 - y1) * (x2 - x1) / (H * W)            # actual area ratio
    y1h = torch.nn.functional.one_hot(y, num_classes).float()
    return x, lam * y1h + (1 - lam) * y1h[idx]
```

Train with soft-target cross-entropy: `loss = -(targets * logits.log_softmax(1)).sum(1).mean()`.

## Augmentation for detection and segmentation

Geometric transforms must be applied consistently to **boxes, masks and keypoints**. Libraries such as **Albumentations** and torchvision's `transforms.v2` handle this:

```python
import albumentations as A

aug = A.Compose([
    A.RandomResizedCrop(size=(512, 512), scale=(0.5, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.GaussNoise(p=0.2),
], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"], min_visibility=0.3))
# out = aug(image=img, bboxes=boxes, labels=labels)
```

`min_visibility` drops boxes that become mostly cropped out — otherwise the model is trained to find objects that are no longer visible.

## Test-time augmentation (TTA)

At inference, average predictions over several augmented versions (flips, a few crops). It often adds a small accuracy gain and improves calibration, at the cost of extra inference time.

## Synthetic data

When real data is scarce — rare classes, dangerous scenarios, privacy constraints — synthetic images from simulation, 3-D rendering or generative models can help. Watch for the **domain gap**: models may learn artefacts of synthetic images. Mixing synthetic with real data and applying domain randomisation reduces this.

## Designing an augmentation strategy

1. Start with random resized crop + flip (if valid).
2. Add photometric jitter matching real-world variation in your deployment environment.
3. Add RandAugment/TrivialAugment and random erasing for larger training runs.
4. Add Mixup/CutMix for longer training from scratch or when overfitting persists.
5. Validate each addition; stronger augmentation often needs **longer training**.

:::exercise
1. Visualise 16 augmented versions of one image under your policy. Are all labels still correct?
2. Train a ResNet-18 on 10% of CIFAR-10 with (a) no augmentation, (b) crop + flip, (c) + TrivialAugment, (d) + CutMix. Report test accuracy.
3. Design an augmentation policy for chest X-rays and justify each choice with a radiologist's perspective.
:::

:::takeaway
- Augmentation encodes invariances: geometric, photometric, occlusion and mixing transforms.
- Every augmentation must preserve the label for your domain — visualise to check.
- RandAugment and TrivialAugment automate policies; Mixup/CutMix regularise strongly.
- Transform boxes and masks consistently; use TTA and synthetic data judiciously.
:::

=== POST ===
slug: object-detection-rcnn-family
title: "Object Detection I: R-CNN, Fast R-CNN and Faster R-CNN"
category: computer-vision
level: Intermediate
tags: object detection, faster r-cnn, region proposals, anchors, iou
summary: Detection asks what objects are in an image and where. We define bounding boxes, IoU and mAP, then trace the two-stage R-CNN family from selective search to region proposal networks and feature pyramids.
---
Classification says "there is a person in this image". **Object detection** says "there are three people, here, here and here, and a bicycle there". It must output a variable number of **bounding boxes**, each with a class label and a confidence score. Detection underpins autonomous driving, retail analytics, wildlife counting from camera traps, and counting shelters or vehicles in satellite imagery after disasters.

## Representing and evaluating boxes

A box is typically $(x_{\min}, y_{\min}, x_{\max}, y_{\max})$ or (centre, width, height). Box overlap is measured by **Intersection over Union**:

$$
\text{IoU}(A, B) = \frac{|A \cap B|}{|A \cup B|}
$$

A prediction is a **true positive** if its IoU with an unmatched ground-truth box of the same class exceeds a threshold (commonly 0.5).

**Mean Average Precision (mAP)**: for each class, rank predictions by confidence, compute the precision–recall curve and its area (**AP**), then average over classes. The COCO benchmark reports mAP averaged over IoU thresholds 0.50 to 0.95 in steps of 0.05 (written mAP@[.5:.95]), rewarding precise localisation.

```python
def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])
    return inter / (area(a) + area(b) - inter + 1e-9)

print(iou([10, 10, 50, 50], [30, 30, 70, 70]))   # 400 / 2800 ≈ 0.143
```

## Non-maximum suppression (NMS)

Detectors produce many overlapping boxes for the same object. **NMS** keeps the highest-scoring box and removes others that overlap it by more than a threshold (e.g. IoU > 0.5), repeating for the remaining boxes. Variants: Soft-NMS (decays scores instead of removing), class-aware NMS.

## R-CNN (2014)

Girshick et al.'s **Regions with CNN features**:

1. Generate ~2,000 **region proposals** with selective search (a classical segmentation-based method).
2. Warp each region to a fixed size and run a CNN on **each** one to extract features.
3. Classify each region with SVMs; refine boxes with **bounding-box regression**.

It dramatically improved accuracy on PASCAL VOC but was extremely slow — tens of seconds per image — because the CNN ran thousands of times.

## Fast R-CNN (2015)

Run the CNN **once** on the whole image to get a feature map. For each proposal, **RoI pooling** extracts a fixed-size feature grid from the corresponding feature-map region. A single network then outputs class scores and box refinements, trained jointly with a multi-task loss (classification + smooth-L1 box regression). Much faster — but selective search proposals remained the bottleneck.

## Faster R-CNN (2015)

Ren, He, Girshick and Sun replaced selective search with a learned **Region Proposal Network (RPN)** that shares the backbone features:

- At each feature-map location, place $k$ **anchor boxes** of different scales and aspect ratios (e.g. 3 × 3 = 9).
- For each anchor, predict an **objectness** score and box offsets.
- Keep the top proposals after NMS; pass them to the Fast R-CNN head.

Box regression predicts offsets relative to an anchor $(x_a, y_a, w_a, h_a)$:

$$
t_x = \frac{x - x_a}{w_a}, \quad t_y = \frac{y - y_a}{h_a}, \quad t_w = \log\frac{w}{w_a}, \quad t_h = \log\frac{h}{h_a}
$$

This parameterisation makes regression targets scale-invariant. Faster R-CNN became a near-real-time, end-to-end trainable detector and the standard two-stage design.

## Feature Pyramid Networks (2017)

Small objects are hard for detectors that use only the coarse final feature map. **FPN** (Lin et al.) builds a top-down pathway with lateral connections, producing semantically strong feature maps at **multiple resolutions**. Small objects are detected on high-resolution levels, large ones on coarse levels. Faster R-CNN + FPN became a very strong baseline.

```python
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn_v2(weights=weights).eval()
img = torch.rand(3, 480, 640)
with torch.no_grad():
    out = model([img])[0]
print(out["boxes"].shape, out["labels"][:5], out["scores"][:5])

# Fine-tuning for your own classes (e.g. 3 classes + background)
in_feat = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, num_classes=4)
# Training: model.train(); loss_dict = model(images, targets); sum(loss_dict.values()).backward()
```

Targets are dictionaries with `boxes` (N×4) and `labels` (N) per image.

## Two-stage vs one-stage

Two-stage detectors (propose, then classify and refine) are typically accurate, especially for small objects, but slower. One-stage detectors (YOLO, SSD, RetinaNet — next lecture) predict boxes directly in one pass and are faster. The gap in accuracy has narrowed considerably over the years.

:::exercise
1. Implement NMS from scratch and compare it with `torchvision.ops.nms`.
2. Compute the anchor offsets $(t_x, t_y, t_w, t_h)$ for a ground-truth box and an anchor of your choice, then invert them.
3. Fine-tune Faster R-CNN on a small detection dataset (e.g. a few hundred annotated images) and report mAP@0.5.
:::

:::takeaway
- Detection outputs boxes, classes and scores; evaluate with IoU and mAP.
- NMS removes duplicate detections.
- R-CNN → Fast R-CNN (shared features, RoI pooling) → Faster R-CNN (learned RPN with anchors).
- FPN adds multi-scale features, greatly helping small objects.
:::

=== POST ===
slug: yolo-real-time-detection
title: "Object Detection II: YOLO and Real-Time Detection"
category: computer-vision
level: Intermediate
tags: yolo, real-time detection, one-stage detectors, anchor-free
summary: "You Only Look Once" reframed detection as a single regression problem, enabling real-time performance. We study the original YOLO grid formulation, its evolution, anchor-free heads, and practical training with modern tools.
---
In 2016 Joseph Redmon and colleagues published **YOLO — You Only Look Once**. Instead of proposing regions and classifying each one, YOLO looks at the whole image **once** with a single network and directly predicts all boxes and classes. It ran at tens of frames per second on a GPU, bringing detection to real-time video. Successive YOLO versions have made it the most widely used family of practical detectors.

## The original formulation

YOLOv1 divides the image into an $S \times S$ grid (e.g. $7 \times 7$). The cell containing an object's **centre** is responsible for detecting it. Each cell predicts:

- $B$ boxes, each with $(x, y, w, h)$ and a **confidence** $= P(\text{object}) \times \text{IoU}$;
- $C$ class probabilities.

The output is a tensor of shape $S \times S \times (5B + C)$ — for PASCAL VOC with $S = 7$, $B = 2$, $C = 20$: $7 \times 7 \times 30$. Training uses a sum-of-squares loss with weights that emphasise box coordinates and down-weight cells without objects; width and height are predicted as square roots so errors on small boxes count more.

**Strengths:** very fast; sees the whole image, so it makes fewer background false positives than sliding-window methods. **Weaknesses:** each cell predicts few boxes, so crowds of small objects (flocks of birds) are hard; coarse localisation.

## Evolution of YOLO

| Version | Key ideas |
|---|---|
| YOLOv2 / YOLO9000 (2017) | Anchor boxes from k-means on training boxes, batch norm, higher resolution, multi-scale training |
| YOLOv3 (2018) | Darknet-53 backbone, predictions at three scales (FPN-like), logistic class predictions |
| YOLOv4 (2020) | "Bag of freebies" (mosaic augmentation, CIoU loss) and "bag of specials" (CSP backbone, PANet neck) |
| YOLOv5 onwards (community/industry) | PyTorch implementations, auto-anchors, strong training recipes, easy tooling |
| YOLOX, YOLOv8 and later | **Anchor-free** heads, decoupled classification/regression heads, improved label assignment |

(From version 4 onward, YOLO versions were developed by different groups, so the numbering does not indicate a single lineage.)

## Anatomy of a modern one-stage detector

1. **Backbone** — extracts features (CSPDarknet, efficient CNNs).
2. **Neck** — fuses multi-scale features (FPN + PAN: top-down and bottom-up paths).
3. **Head** — at each location of each scale, predicts class scores and box geometry.

**Anchor-free** heads predict the distance from a location to the four box sides (as in FCOS) or the box centre and size directly, avoiding hand-tuned anchor shapes.

**Label assignment** — deciding which predictions are responsible for which ground-truth boxes — matters as much as architecture. Modern methods (SimOTA, task-aligned assignment) dynamically assign positives based on both classification and localisation quality.

**Box losses** based on IoU (GIoU, DIoU, CIoU) directly optimise overlap rather than coordinate differences, which aligns training with evaluation. GIoU, for instance, adds a penalty based on the smallest enclosing box $C$:

$$
\text{GIoU} = \text{IoU} - \frac{|C \setminus (A \cup B)|}{|C|}
$$

so non-overlapping boxes still receive a useful gradient.

## Training a YOLO model in practice

Tools like the Ultralytics package make training straightforward. Labels use one text file per image with rows `class x_center y_center width height`, normalised to $[0, 1]$.

```python
# pip install ultralytics
from ultralytics import YOLO

model = YOLO("yolov8n.pt")                  # small pretrained model (COCO)
model.train(data="shelters.yaml",           # paths to train/val images and class names
            epochs=100, imgsz=640, batch=16, patience=20)
metrics = model.val()                        # reports mAP50 and mAP50-95
results = model.predict("aerial_tile.jpg", conf=0.25, iou=0.5)
results[0].show()
model.export(format="onnx")                  # for deployment
```

```yaml
# shelters.yaml
path: datasets/shelters
train: images/train
val: images/val
names: {0: tent, 1: building, 2: vehicle}
```

:::tip
For small objects (e.g. vehicles or tents in aerial imagery), use higher input resolution, tile large images into overlapping patches with sliced inference, and make sure small boxes survive augmentation. Check the licence of any detection framework before commercial use.
:::

## Speed–accuracy trade-offs

YOLO families come in sizes (nano, small, medium, large, extra-large). Pick the smallest model meeting the accuracy requirement at your target frame rate on your hardware. Measure **end-to-end latency** including preprocessing and NMS, not just model FLOPs.

## Beyond YOLO

**DETR** (2020) reframed detection as set prediction with a transformer and bipartite matching, removing anchors and NMS entirely; later variants (Deformable DETR, DINO, RT-DETR) made transformer detectors fast and accurate. **Open-vocabulary detectors** such as OWL-ViT and Grounding DINO detect objects described by arbitrary text prompts — "detect water tanks" — without training on those classes.

:::exercise
1. Compute YOLOv1's output tensor size for $S = 13$, $B = 3$ and 80 classes.
2. Implement GIoU and compare its value with IoU for overlapping, touching and far-apart boxes.
3. Train a small YOLO model on a public dataset of your choice and report mAP50 and frames per second on CPU and GPU.
:::

:::takeaway
- YOLO predicts all boxes in one pass over a grid — fast, real-time detection.
- Modern YOLOs use CSP backbones, FPN/PAN necks, anchor-free decoupled heads and smart label assignment.
- IoU-based losses (GIoU, CIoU) align training with evaluation.
- Choose model size by accuracy and measured end-to-end latency; transformers and open-vocabulary detectors extend the field.
:::

=== POST ===
slug: ssd-retinanet-focal-loss
title: "Object Detection III: SSD, RetinaNet and the Focal Loss"
category: computer-vision
level: Advanced
tags: ssd, retinanet, focal loss, class imbalance, one-stage detectors
summary: One-stage detectors face an extreme imbalance between background and objects. We study SSD's multi-scale default boxes, then derive RetinaNet's focal loss, which let one-stage detectors match two-stage accuracy.
---
Why were one-stage detectors less accurate than two-stage detectors for years? A 2017 paper from Facebook AI Research gave a precise answer — **class imbalance** — and a simple, elegant fix: the **focal loss**. Along the way we meet **SSD**, which introduced the multi-scale design that most one-stage detectors still follow.

## SSD: Single Shot MultiBox Detector

Liu et al. (2016) proposed SSD:

- A backbone (originally VGG-16) followed by extra convolutional layers of decreasing resolution.
- **Predictions from multiple feature maps**: high-resolution maps detect small objects, low-resolution maps large ones.
- At each location of each map, a set of **default boxes** (anchors) with several aspect ratios; small convolutional filters predict class scores and box offsets for each.
- **Hard negative mining**: because most default boxes are background, SSD keeps only the hardest negatives, at a negative:positive ratio of at most 3:1.

SSD was fast and more accurate than YOLOv1, but still behind Faster R-CNN on small objects.

## The imbalance problem

A one-stage detector evaluates on the order of $10^4$–$10^5$ candidate locations per image, while an image contains only a handful of objects. The vast majority are **easy negatives** — plain background, correctly classified with high confidence. Individually, each contributes a small loss; collectively, they **dominate** the total loss and gradient, swamping the few informative examples.

Two-stage detectors sidestep this: the RPN filters out most background, and the second stage samples a balanced mix of foreground and background. One-stage detectors used heuristics like hard negative mining, which discard data and require tuning.

## The focal loss

Lin, Goyal, Girshick, He and Dollár (2017) reshaped the cross-entropy loss. Let $p_t$ be the model's probability for the true class:

$$
p_t = \begin{cases} p & \text{if } y = 1 \\ 1 - p & \text{otherwise} \end{cases}, \qquad \text{CE}(p_t) = -\log p_t
$$

The **focal loss** adds a modulating factor:

$$
\text{FL}(p_t) = -\alpha_t\,(1 - p_t)^\gamma\,\log p_t
$$

- For **easy** examples ($p_t \to 1$), $(1 - p_t)^\gamma \to 0$ — their loss is strongly down-weighted.
- For **hard** examples ($p_t$ small), the factor is near 1 — the loss is almost unchanged.
- $\gamma$ controls the strength of focusing; $\gamma = 2$ worked best. With $\gamma = 2$, an example with $p_t = 0.9$ has its loss reduced 100×, and with $p_t \approx 0.968$ about 1000×.
- $\alpha_t$ is a class-balancing weight (0.25 for the positive class worked well with $\gamma = 2$).

```python
import torch
import torch.nn.functional as F

def sigmoid_focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    return (alpha_t * (1 - p_t) ** gamma * ce).sum()

logits = torch.tensor([4.0, -4.0, 0.0, -1.0]); targets = torch.tensor([1.0, 0.0, 1.0, 1.0])
print("CE per example:   ", F.binary_cross_entropy_with_logits(logits, targets, reduction="none"))
print("focal (summed):   ", sigmoid_focal_loss(logits, targets))
```

(`torchvision.ops.sigmoid_focal_loss` provides an optimised version.)

## RetinaNet

To test the focal loss, the authors built **RetinaNet**: a ResNet + FPN backbone with two small subnetworks applied at every pyramid level — one for classification (with sigmoid outputs per class) and one for box regression — using 9 anchors per location. Two details matter:

- **Normalisation**: the total loss is divided by the number of anchors assigned to ground-truth boxes, not the total number of anchors.
- **Prior initialisation**: the classification layer's bias is initialised so every anchor starts with a foreground probability of about $\pi = 0.01$:

$$
b = -\log\frac{1 - \pi}{\pi}
$$

Without this, the huge number of background anchors produces a destabilising loss at the start of training — the same trick we discussed for imbalanced problems generally.

RetinaNet matched or surpassed two-stage detectors' accuracy on COCO while keeping one-stage simplicity — a striking demonstration that **the loss function, not the architecture, had been the bottleneck**.

## Legacy

- Focal loss (and variants such as quality focal loss and varifocal loss) is used throughout detection, segmentation and other imbalanced classification problems — medical lesion detection, rare-event prediction.
- The "backbone + FPN + shared dense heads" design became the template for one-stage detectors (FCOS, ATSS, modern YOLOs).
- The idea of **reweighting examples by difficulty** connects to boosting, hard-example mining and curriculum learning.

:::note
When you face severe imbalance in any dense prediction problem, consider three levers together: a focal-style loss, a sensible bias initialisation reflecting the prior, and normalisation by the number of positives. Together they often remove the need for manual sampling heuristics.
:::

:::exercise
1. Plot CE and FL (for $\gamma = 0.5, 1, 2, 5$) as functions of $p_t$ and explain the curves.
2. Compute the down-weighting factor $(1 - p_t)^\gamma$ for $p_t = 0.5, 0.9, 0.99$ with $\gamma = 2$.
3. Train a binary classifier on a 1:1000 imbalanced dataset with CE and with focal loss; compare average precision.
:::

:::takeaway
- SSD predicts from multiple feature maps with default boxes; it relied on hard negative mining.
- One-stage detectors suffer extreme foreground–background imbalance dominated by easy negatives.
- Focal loss $-\alpha_t(1 - p_t)^\gamma\log p_t$ down-weights easy examples; prior bias initialisation stabilises training.
- RetinaNet matched two-stage accuracy and set the template for modern dense detectors.
:::

=== POST ===
slug: semantic-segmentation-fcn-unet
title: "Semantic Segmentation: FCN, U-Net and DeepLab"
category: computer-vision
level: Intermediate
tags: segmentation, u-net, fcn, deeplab, dice loss, medical imaging
summary: Segmentation labels every pixel. We cover fully convolutional networks, the encoder–decoder U-Net with skip connections, DeepLab's atrous convolutions, loss functions like Dice, and evaluation with IoU.
---
Classification labels an image; detection draws boxes; **semantic segmentation** assigns a class to **every pixel**. It outlines tumours in MRI scans, maps flooded areas and informal settlements from satellite images, identifies drivable road for autonomous vehicles and separates crops from weeds. It is the most detailed form of image understanding among the classic tasks.

## The task

Input: an image $H \times W \times 3$. Output: a label map $H \times W$ with one of $K$ classes per pixel (or $K$ probability maps). "Semantic" segmentation does not separate instances: two adjacent people are both simply "person". (Instance segmentation, next lecture, separates them.)

## Fully Convolutional Networks (FCN)

Long, Shelhamer and Darrell (2015) observed that a classification CNN becomes a dense predictor if you:

1. **Replace fully connected layers with convolutions**, so the network accepts any input size and outputs a coarse spatial map of class scores.
2. **Upsample** the coarse map back to input resolution with learnable **transposed convolutions**.
3. Add **skip connections** from earlier, higher-resolution layers (FCN-16s, FCN-8s) to recover fine detail.

This end-to-end, pixels-to-pixels formulation founded modern segmentation.

## U-Net

Ronneberger, Fischer and Brox (2015) designed **U-Net** for biomedical images with very few training examples. Its symmetric encoder–decoder has a characteristic U shape:

- **Encoder (contracting path)**: repeated conv blocks and downsampling capture context ("what").
- **Decoder (expanding path)**: upsampling and conv blocks recover resolution ("where").
- **Skip connections**: at each level, encoder feature maps are **concatenated** with decoder feature maps, giving the decoder precise spatial detail.

U-Net trained well from only dozens of annotated images with heavy augmentation (elastic deformations) and became the default architecture for medical segmentation — and, later, the backbone of diffusion image generators.

```python
import torch
import torch.nn as nn

def block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

class UNet(nn.Module):
    def __init__(self, in_ch=3, n_classes=2, base=32):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8]
        self.enc = nn.ModuleList([block(in_ch, c[0]), block(c[0], c[1]), block(c[1], c[2])])
        self.pool = nn.MaxPool2d(2)
        self.mid = block(c[2], c[3])
        self.up = nn.ModuleList([nn.ConvTranspose2d(c[3], c[2], 2, 2), nn.ConvTranspose2d(c[2], c[1], 2, 2),
                                 nn.ConvTranspose2d(c[1], c[0], 2, 2)])
        self.dec = nn.ModuleList([block(c[3], c[2]), block(c[2], c[1]), block(c[1], c[0])])
        self.head = nn.Conv2d(c[0], n_classes, 1)
    def forward(self, x):
        skips = []
        for e in self.enc:
            x = e(x); skips.append(x); x = self.pool(x)
        x = self.mid(x)
        for up, d, s in zip(self.up, self.dec, reversed(skips)):
            x = d(torch.cat([up(x), s], dim=1))          # skip connection by concatenation
        return self.head(x)                               # (N, n_classes, H, W) logits

print(UNet()(torch.randn(1, 3, 128, 128)).shape)
```

## DeepLab and atrous convolution

Downsampling loses detail, but context needs large receptive fields. The **DeepLab** family (Chen et al., 2015–2018) used:

- **Atrous (dilated) convolutions** to enlarge receptive fields while keeping feature maps at higher resolution;
- **Atrous Spatial Pyramid Pooling (ASPP)**: parallel dilated convolutions at several rates plus global pooling, capturing multi-scale context;
- **DeepLabv3+**: adds a lightweight decoder for sharper boundaries.

## Loss functions

- **Pixel-wise cross-entropy** — the default.
- **Weighted cross-entropy** — up-weight rare classes (small lesions, thin roads).
- **Dice loss** — directly optimises overlap, robust to foreground–background imbalance:

$$
\mathcal{L}_{\text{Dice}} = 1 - \frac{2\sum_i p_i g_i + \epsilon}{\sum_i p_i + \sum_i g_i + \epsilon}
$$

where $p_i$ are predicted probabilities and $g_i$ ground-truth labels. Combining cross-entropy and Dice is common in medical imaging.
- **Focal loss** and **boundary losses** for hard pixels and edges.

```python
def dice_loss(logits, target, eps=1.0):
    p = torch.sigmoid(logits).flatten(1); g = target.flatten(1).float()
    return 1 - ((2 * (p * g).sum(1) + eps) / (p.sum(1) + g.sum(1) + eps)).mean()
```

## Evaluation

- **Pixel accuracy** — misleading when background dominates.
- **IoU (Jaccard) per class** and **mean IoU (mIoU)** — the standard metric.
- **Dice coefficient** (equivalent to F1 over pixels) — standard in medicine.
- **Boundary metrics** (e.g. Hausdorff distance) when contours matter clinically.

## Practical tips

- High-resolution images (satellite, pathology slides) are processed in **tiles** with overlap; predictions are stitched and blended.
- Use pretrained encoders (e.g. a ResNet or EfficientNet encoder in a U-Net) — libraries like `segmentation_models_pytorch` make this easy.
- Annotation is expensive; consider weak labels (scribbles, boxes), active learning, and foundation models like SAM (covered later) to accelerate labelling.

:::exercise
1. Train the U-Net above on a small segmentation dataset (e.g. an oxford pets trimap or a public satellite building dataset) and report mIoU.
2. Compare cross-entropy, Dice and a combination on a dataset where the foreground covers less than 5% of pixels.
3. Remove the skip connections from U-Net and compare boundary quality.
:::

:::takeaway
- Semantic segmentation classifies every pixel; FCNs made classification networks dense predictors.
- U-Net's encoder–decoder with concatenated skip connections recovers detail; it dominates medical imaging.
- DeepLab uses atrous convolutions and ASPP for multi-scale context.
- Use cross-entropy, Dice or focal losses; evaluate with IoU/mIoU and Dice.
:::

=== POST ===
slug: instance-segmentation-mask-rcnn
title: "Instance Segmentation: Mask R-CNN and Beyond"
category: computer-vision
level: Advanced
tags: instance segmentation, mask r-cnn, roialign, panoptic segmentation
summary: Instance segmentation separates each individual object with its own mask. We study Mask R-CNN's mask branch and RoIAlign, compare instance, semantic and panoptic segmentation, and survey query-based models like Mask2Former.
---
Semantic segmentation tells us which pixels are "person"; it does not tell us **how many** people there are or where one ends and another begins. **Instance segmentation** combines detection and segmentation: for every object instance it outputs a class, a confidence and a pixel mask. It is used to count and measure cells in microscopy, delineate individual buildings for damage assessment, separate overlapping fruit for robotic harvesting, and track individual animals.

## Three flavours of segmentation

| Task | Output | "Stuff" (sky, road) | "Things" (people, cars) |
|---|---|---|---|
| Semantic | Class per pixel | Yes | Class only, instances merged |
| Instance | Mask per object | Ignored | Separate instances |
| Panoptic | Class + instance ID per pixel | Yes | Separate instances |

**Panoptic segmentation** (Kirillov et al., 2019) unifies both: every pixel gets a class, and pixels of countable things also get an instance ID.

## Mask R-CNN

He, Gkioxari, Dollár and Girshick (2017) extended Faster R-CNN with a third, parallel branch:

1. Backbone + FPN extract features.
2. The RPN proposes regions.
3. For each region, three heads run in parallel:
   - classification,
   - box regression,
   - a **mask head**: a small fully convolutional network that predicts a $28 \times 28$ binary mask **for each class** (the mask for the predicted class is used).

The loss is the sum $\mathcal{L} = \mathcal{L}_{\text{cls}} + \mathcal{L}_{\text{box}} + \mathcal{L}_{\text{mask}}$, where the mask loss is per-pixel binary cross-entropy on the ground-truth class's mask only. **Decoupling** mask prediction from classification (no competition between classes at each pixel) improved results.

## RoIAlign: the crucial detail

Faster R-CNN's **RoI pooling** quantises region coordinates to the feature-map grid (rounding), then quantises again into pooling bins. For classification this misalignment is harmless, but for pixel-accurate masks, errors of even a feature-map cell (which may correspond to 16–32 image pixels) are significant.

**RoIAlign** removes all rounding: it samples feature values at exact, fractional locations using **bilinear interpolation**, then pools. This simple change improved mask accuracy substantially and also helped box detection.

```python
import torch
from torchvision.ops import roi_align

features = torch.randn(1, 256, 50, 50)                 # feature map at stride 16
boxes = torch.tensor([[0, 103.7, 58.2, 331.9, 287.4]])  # (batch_idx, x1, y1, x2, y2) in image pixels
pooled = roi_align(features, boxes, output_size=(7, 7), spatial_scale=1 / 16, sampling_ratio=2, aligned=True)
print(pooled.shape)                                     # (1, 256, 7, 7)
```

## Using and fine-tuning Mask R-CNN

```python
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

model = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
num_classes = 2                                          # background + "building"
model.roi_heads.box_predictor = FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, num_classes)
model.roi_heads.mask_predictor = MaskRCNNPredictor(256, 256, num_classes)
# Targets per image: {"boxes": (N,4), "labels": (N,), "masks": (N,H,W) uint8}
```

At inference, each detection comes with a soft mask; threshold at 0.5 to obtain a binary mask in image coordinates.

## Evaluation

Instance segmentation uses **mask AP**: the same AP computation as detection, but IoU is computed between **masks** rather than boxes. Panoptic segmentation uses **Panoptic Quality**: $PQ = \text{SQ} \times \text{RQ}$ (segmentation quality × recognition quality).

## Beyond Mask R-CNN

- **One-stage / bottom-up methods**: YOLACT (prototype masks combined with per-instance coefficients, real-time), SOLO (predict masks by location).
- **Query-based transformers**: **DETR**-style models use learned object queries that each predict a class and mask, trained with bipartite matching — no anchors or NMS. **MaskFormer** and **Mask2Former** unified semantic, instance and panoptic segmentation in one architecture with state-of-the-art results.
- **Promptable foundation models**: **Segment Anything (SAM)** produces masks for any object given a point, box or (in later versions) text/concept prompt — transforming annotation workflows (see the lecture on vision foundation models).

:::note
For measurement applications — counting and sizing cells, estimating roof areas, measuring crop canopy — instance masks give far more information than boxes. But annotation of polygons is slow; use model-assisted labelling (pre-annotate with a pretrained or promptable model, then correct) to cut labelling time dramatically.
:::

:::exercise
1. Explain with a numerical example how RoI pooling's rounding misaligns a region by several image pixels at stride 16.
2. Fine-tune Mask R-CNN on a small instance-segmentation dataset (e.g. a pedestrian dataset) and report box AP and mask AP.
3. Compare semantic, instance and panoptic outputs on one street-scene image and describe what each captures.
:::

:::takeaway
- Instance segmentation outputs a class, score and mask per object; panoptic segmentation also labels "stuff".
- Mask R-CNN adds a per-class FCN mask branch to Faster R-CNN, with decoupled mask and class prediction.
- RoIAlign's bilinear sampling removes quantisation misalignment, essential for pixel-accurate masks.
- Query-based transformers (Mask2Former) unify segmentation tasks; SAM makes masks promptable.
:::

=== POST ===
slug: vision-transformers
title: "Vision Transformers (ViT): Images as Sequences of Patches"
category: computer-vision
level: Advanced
tags: vision transformer, vit, attention, patches, swin, deit
summary: Transformers conquered language, then vision. We dissect ViT's patch embeddings, class token and positional encodings, compare inductive biases with CNNs, and survey DeiT, Swin and hierarchical designs.
---
In 2020, Dosovitskiy and colleagues at Google published a paper with a memorable title: "An Image is Worth 16×16 Words". They applied a nearly unmodified **Transformer** — the architecture dominating NLP — directly to images, treating an image as a sequence of patches. Trained on enough data, the **Vision Transformer (ViT)** matched or beat the best CNNs. Today, transformer-based vision backbones are central to foundation models, multimodal systems and image generation.

(If you have not yet studied transformers, read the NLP track's lecture on the Transformer architecture alongside this one.)

## From image to token sequence

1. **Split** the image $H \times W \times C$ into $N = HW/P^2$ non-overlapping patches of size $P \times P$ (e.g. $16 \times 16$; a $224 \times 224$ image gives $14 \times 14 = 196$ patches).
2. **Flatten** each patch into a vector of length $P^2C$ (768 for RGB $16 \times 16$ patches) and apply a learned **linear projection** to dimension $D$. (Equivalently, a convolution with kernel size and stride $P$.)
3. **Prepend a learnable [CLS] token** whose final representation summarises the image for classification.
4. **Add positional embeddings** (learned, one per position), since self-attention alone is permutation-invariant and would otherwise ignore where each patch came from.

$$
\mathbf{z}_0 = [\mathbf{x}_{\text{cls}};\; \mathbf{x}_p^1\mathbf{E};\; \dots;\; \mathbf{x}_p^N\mathbf{E}] + \mathbf{E}_{\text{pos}}
$$

5. Pass through $L$ standard **transformer encoder blocks** (pre-norm multi-head self-attention + MLP, with residual connections).
6. Classify from the final [CLS] representation (or from the mean of patch tokens).

```python
import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    def __init__(self, img=224, patch=16, in_ch=3, dim=384):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)
        self.n = (img // patch) ** 2
    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)        # (B, N, dim)

class TinyViT(nn.Module):
    def __init__(self, n_classes=10, dim=384, depth=6, heads=6, img=224, patch=16):
        super().__init__()
        self.embed = PatchEmbed(img, patch, 3, dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.randn(1, self.embed.n + 1, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(dim, heads, 4 * dim, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, depth)
        self.norm, self.head = nn.LayerNorm(dim), nn.Linear(dim, n_classes)
    def forward(self, x):
        t = self.embed(x)
        t = torch.cat([self.cls.expand(len(t), -1, -1), t], dim=1) + self.pos
        return self.head(self.norm(self.blocks(t))[:, 0])     # classify from [CLS]

print(TinyViT()(torch.randn(2, 3, 224, 224)).shape)
```

## Inductive bias: why data scale matters

CNNs build in **locality** and **translation equivariance**. ViT builds in almost nothing: apart from patch extraction, every patch can attend to every other from the first layer, and spatial relationships must be **learned** through positional embeddings and attention.

Consequence: on ImageNet-1k alone (1.3M images), the original ViT underperformed comparable ResNets. Pretrained on much larger datasets (e.g. ImageNet-21k with 14M images, or the proprietary JFT-300M), ViT **surpassed** them, and it scaled better with data and compute. With little data, strong priors help; with abundant data, flexible models win — a recurring theme.

## Making ViTs data-efficient: DeiT

**DeiT** (Touvron et al., 2021) trained competitive ViTs on ImageNet-1k alone using a strong recipe — heavy augmentation (RandAugment, Mixup, CutMix, random erasing), regularisation (stochastic depth), and **distillation** from a CNN teacher via an extra distillation token. Training recipes, once again, mattered as much as architecture.

## What ViTs learn

Visualisations show that some attention heads in early layers attend locally (like convolutions), while others attend globally from the start. Attention **distance** increases with depth. Learned positional embeddings reproduce the 2-D grid structure. ViTs tend to be more **shape-biased** and somewhat more robust to certain corruptions and occlusions than CNNs.

## Hierarchical vision transformers: Swin

Global self-attention costs $O(N^2)$ in the number of patches, which becomes prohibitive for high-resolution dense tasks like detection and segmentation. **Swin Transformer** (Liu et al., 2021):

- computes attention within **local windows** (linear cost in image size);
- **shifts** the windows between consecutive layers so information flows across window boundaries;
- **merges patches** between stages to build a CNN-like pyramid of feature maps.

Swin became a strong general backbone for detection and segmentation. Other hybrids (ConViT, CoAtNet, MobileViT) blend convolution and attention.

## Self-supervised ViTs

ViTs pair naturally with self-supervised learning: **MAE** (masked autoencoders) masks ~75% of patches and reconstructs them; **DINO / DINOv2** learn powerful general-purpose features by self-distillation, which transfer well to many tasks with little labelled data. These are covered in a later lecture.

## Choosing between CNNs and ViTs

| Situation | Suggestion |
|---|---|
| Small dataset, limited compute | Pretrained CNN (ResNet, ConvNeXt, EfficientNet) or pretrained ViT fine-tuned |
| Large-scale pretraining | ViT scales very well |
| Dense prediction at high resolution | Hierarchical (Swin-like) or CNN backbones |
| Multimodal models (image + text) | ViT encoders are the norm (CLIP, most vision-language models) |
| Edge devices | Efficient CNNs or mobile hybrids |

:::exercise
1. Compute the number of tokens and the cost of one self-attention layer for $224 \times 224$ images with patch sizes 32, 16 and 8.
2. Fine-tune a pretrained ViT-B/16 and a ResNet-50 on a small dataset; compare accuracy and training time.
3. Visualise the cosine similarity between the learned positional embeddings of a pretrained ViT. What structure appears?
:::

:::takeaway
- ViT splits images into patches, embeds them linearly, adds positional embeddings and a [CLS] token, and applies a transformer encoder.
- With weak inductive biases, ViTs need large data or strong recipes (DeiT) — but scale excellently.
- Swin's shifted-window attention gives hierarchical, efficient backbones for dense tasks.
- ViTs underpin modern self-supervised and multimodal vision models.
:::
