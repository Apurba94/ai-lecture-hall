=== POST ===
slug: introduction-to-computer-vision
title: "Introduction to Computer Vision: From Pixels to Perception"
category: computer-vision
level: Beginner
tags: computer vision, images, pixels, tasks, overview
summary: We open the Computer Vision track by asking how a machine can see. We cover how images are represented, why vision is hard, the landscape of vision tasks, and how deep learning transformed the field.
---
Welcome to the Computer Vision track. Humans recognise a friend's face in a crowd in a fraction of a second, yet for decades this was one of the hardest problems in computing. In 1966, a famous MIT summer project aimed to "solve" vision in a few months; the field has been working on it ever since. Today we set out the problem and the map of the territory.

## How computers see an image

To a computer, an image is an array of numbers. A grayscale image of height $H$ and width $W$ is a matrix in $\mathbb{R}^{H \times W}$, with each **pixel** typically an integer from 0 (black) to 255 (white). A colour image adds a **channel** dimension: $H \times W \times 3$ for red, green and blue. A 1080p photo therefore contains about 6.2 million numbers.

Other representations matter too: **HSV** separates hue from brightness (useful for colour-based segmentation), depth images store distance, multispectral satellite images have many bands beyond visible light, and medical scans (CT, MRI) are 3-D volumes.

## Why vision is hard

The same object produces wildly different pixel arrays because of:

- **Viewpoint** — a chair seen from above vs from the side.
- **Illumination** — bright sunlight vs shadow change every pixel value.
- **Scale** — near vs far.
- **Deformation** — a sitting cat vs a stretching cat.
- **Occlusion** — objects partly hidden.
- **Background clutter** — the object blends into its surroundings.
- **Intra-class variation** — thousands of chair designs.

Meanwhile, tiny pixel changes can alter meaning (a "stop" sign vs a "slow" sign). A good vision system must be **invariant** to nuisance variation while remaining **sensitive** to meaningful differences.

## The landscape of vision tasks

| Task | Output | Example |
|---|---|---|
| Image classification | One label per image | "This X-ray shows pneumonia" |
| Object detection | Boxes + labels | Locate every vehicle in a street scene |
| Semantic segmentation | Label per pixel | Road, building, vegetation in satellite images |
| Instance segmentation | Mask per object instance | Separate each person in a crowd |
| Pose estimation | Keypoints | Body joints for physiotherapy apps |
| Depth / 3-D reconstruction | Distances, meshes | Robot navigation |
| Tracking | Identities over video | Following players in sports |
| OCR / document understanding | Text and layout | Digitising registration forms |
| Image generation | New images | Text-to-image models |
| Vision–language | Captions, answers | "What is the child holding?" |

## Three eras of computer vision

1. **Geometry and hand-crafted rules (1960s–1990s)** — edge detection, shape from shading, stereo geometry.
2. **Hand-crafted features + machine learning (2000s)** — SIFT, HOG and bag-of-visual-words features fed to SVMs; the Viola–Jones face detector.
3. **Deep learning (2012–present)** — CNNs learn features end-to-end; since 2020, Vision Transformers and large pretrained vision–language models learned from web-scale image–text data.

The 2012 ImageNet moment was dramatic: AlexNet reduced the top-5 error from about 26% to about 15%. Within a few years, error on this benchmark fell below reported human estimates.

## A first look in code

```python
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

img = np.asarray(Image.open("photo.jpg").convert("RGB"))     # (H, W, 3), uint8
print(img.shape, img.dtype, img.min(), img.max())

gray = img.mean(axis=2)                                        # naive grayscale
red = img.copy(); red[..., 1:] = 0                             # keep only the red channel
fig, ax = plt.subplots(1, 3, figsize=(12, 4))
for a, im, t in zip(ax, [img, gray, red], ["RGB", "Gray", "Red channel"]):
    a.imshow(im, cmap="gray" if im.ndim == 2 else None); a.set_title(t); a.axis("off")
plt.show()
```

## Vision for good — and with care

Computer vision assists doctors reading scans, maps informal settlements and flood extents from satellites after disasters, monitors crops, reads handwritten records and helps visually impaired people navigate. It also powers surveillance and facial recognition with serious risks to privacy and civil liberties, and has shown unequal accuracy across skin tones and genders. As engineers, we must evaluate across diverse populations and consider how a system will be used — themes we revisit in the Ethics track.

:::exercise
1. Load an image and convert it to HSV. Threshold the hue channel to isolate green vegetation.
2. For each task in the table, name one application relevant to your community.
3. List three nuisance factors that would affect a model classifying crop diseases from farmers' phone photos, and how you would collect data to cover them.
:::

:::takeaway
- Images are arrays: $H \times W \times C$ numbers.
- Vision is hard because of viewpoint, lighting, scale, occlusion and intra-class variation.
- Tasks range from classification to detection, segmentation, 3-D, OCR and generation.
- Deep learning transformed the field after 2012; responsible evaluation is essential.
:::

=== POST ===
slug: image-processing-fundamentals
title: "Image Processing Fundamentals: Filtering, Convolution and Edge Detection"
category: computer-vision
level: Beginner
tags: image processing, filters, convolution, edge detection, opencv
summary: Before deep learning, images were processed with hand-designed filters. We study histograms, blurring, sharpening, gradients, the Sobel and Canny edge detectors, and morphological operations — the vocabulary CNNs later learned.
---
Classical image processing is not obsolete. It is used for preprocessing, data augmentation, fast embedded systems and interpretable pipelines — and it provides the intuition for what convolutional networks learn in their first layers. Today we learn the essential operations with OpenCV.

## Point operations and histograms

A **point operation** changes each pixel independently: brightness ($I + b$), contrast ($aI$), gamma correction ($I^\gamma$), thresholding. An image's **histogram** counts pixel intensities. **Histogram equalisation** spreads intensities across the full range, improving contrast in dark or washed-out images. **CLAHE** (Contrast-Limited Adaptive Histogram Equalisation) does this locally and is widely used for medical and low-light images.

## Filtering by convolution

Linear filtering replaces each pixel with a weighted sum of its neighbourhood — a convolution with a **kernel**:

$$
I'(x, y) = \sum_{u, v}K(u, v)\,I(x + u, y + v)
$$

| Kernel | Effect |
|---|---|
| Box / mean ($\frac{1}{9}$ everywhere in $3\times3$) | Blur |
| Gaussian $\propto e^{-(u^2+v^2)/2\sigma^2}$ | Smooth blur, removes noise |
| Sharpen $\begin{bmatrix}0&-1&0\\-1&5&-1\\0&-1&0\end{bmatrix}$ | Enhances edges |
| Sobel-x $\begin{bmatrix}-1&0&1\\-2&0&2\\-1&0&1\end{bmatrix}$ | Horizontal gradient (vertical edges) |
| Laplacian | Second derivative; responds to edges and blobs |

**Non-linear filters**: the **median filter** replaces each pixel by the median of its neighbourhood and removes salt-and-pepper noise while preserving edges; the **bilateral filter** smooths while respecting edges by weighting neighbours by both spatial and intensity similarity.

## Image gradients and edges

Edges are where intensity changes sharply. The gradient $\nabla I = (I_x, I_y)$ has magnitude and direction:

$$
|\nabla I| = \sqrt{I_x^2 + I_y^2}, \qquad \theta = \arctan\frac{I_y}{I_x}
$$

Sobel filters approximate $I_x$ and $I_y$ with built-in smoothing.

### The Canny edge detector

Canny (1986) remains the classic edge detector:

1. Smooth with a Gaussian to reduce noise.
2. Compute gradient magnitude and direction.
3. **Non-maximum suppression** — keep only pixels that are local maxima along the gradient direction, producing thin edges.
4. **Double thresholding** — strong edges above a high threshold, weak edges between thresholds.
5. **Hysteresis** — keep weak edges only if connected to strong ones.

```python
import cv2
import numpy as np

img = cv2.imread("photo.jpg", cv2.IMREAD_GRAYSCALE)
blur = cv2.GaussianBlur(img, (5, 5), sigmaX=1.4)
gx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
gy = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
magnitude = np.hypot(gx, gy)
edges = cv2.Canny(img, threshold1=50, threshold2=150)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img)
denoised = cv2.medianBlur(img, 5)
cv2.imwrite("edges.png", edges)
```

## Morphological operations

For binary images (e.g. after thresholding), morphology cleans shapes using a **structuring element**:

- **Erosion** shrinks foreground, removing small specks.
- **Dilation** grows foreground, filling small gaps.
- **Opening** (erode then dilate) removes noise; **closing** (dilate then erode) fills holes.

Combined with **connected-component labelling** and **contour detection**, these let you count cells in microscopy, segment printed characters, or measure objects.

## Geometric transformations

Resizing (with interpolation: nearest, bilinear, bicubic), rotation, affine and **perspective (homography)** transforms. A homography can "flatten" a photo of a document taken at an angle — the first step of any document-scanning app:

```python
src = np.float32([[120, 80], [880, 60], [920, 700], [90, 720]])   # document corners in photo
dst = np.float32([[0, 0], [800, 0], [800, 1000], [0, 1000]])
H = cv2.getPerspectiveTransform(src, dst)
flat = cv2.warpPerspective(cv2.imread("form_photo.jpg"), H, (800, 1000))
```

## From hand-designed to learned filters

The first layer of a trained CNN learns filters that look strikingly like Gaussian derivatives, oriented edge detectors and colour-opponent blobs — similar to classical filters and to receptive fields found in the mammalian visual cortex. Deep learning did not discard these ideas; it learned them, and then learned far more complex ones on top.

:::note
Classical operations remain the backbone of **data augmentation** (blur, colour jitter, rotations, perspective warps) and preprocessing (resizing, normalisation, CLAHE for X-rays). Understanding them helps you design augmentations that preserve labels.
:::

:::exercise
1. Add Gaussian and salt-and-pepper noise to an image and compare Gaussian, median and bilateral filtering.
2. Implement Sobel filtering with `scipy.signal.convolve2d` and compare with OpenCV.
3. Build a small pipeline that counts coins in a photo using thresholding, morphology and connected components.
:::

:::takeaway
- Histograms and equalisation adjust contrast; CLAHE works locally.
- Convolution with kernels blurs, sharpens and computes gradients; median and bilateral filters preserve edges.
- Canny edge detection combines smoothing, gradients, non-maximum suppression and hysteresis.
- Morphology and geometric transforms clean shapes and rectify views; CNNs learn similar filters.
:::

=== POST ===
slug: classical-features-sift-hog
title: "Classical Features: Harris Corners, SIFT, HOG and Bag of Visual Words"
category: computer-vision
level: Intermediate
tags: sift, hog, feature detection, descriptors, image matching
summary: Before CNNs, vision relied on carefully engineered features. We study corner detection, SIFT keypoints and descriptors, HOG for pedestrian detection and the bag-of-visual-words model — ideas still used in geometry and robotics.
---
In the 2000s, the most important question in computer vision was: **what features should we extract from an image?** Brilliant hand-engineered answers — Harris corners, SIFT, HOG — powered panorama stitching, object recognition and pedestrian detection. They remain widely used in 3-D reconstruction, visual SLAM for robots and image registration, and they illuminate what deep networks later learned automatically.

## Interest points: Harris corners

Good features are distinctive and repeatable. **Corners** qualify: shifting a small window at a corner changes its contents in every direction. Harris and Stephens (1988) formalised this with the **structure tensor**:

$$
\mathbf{M} = \sum_{(x,y) \in W}w(x, y)\begin{bmatrix} I_x^2 & I_xI_y \\ I_xI_y & I_y^2 \end{bmatrix}
$$

Eigenvalues of $\mathbf{M}$ describe intensity change: both small → flat region; one large → edge; both large → **corner**. The Harris response $R = \det\mathbf{M} - k(\text{tr}\,\mathbf{M})^2$ avoids computing eigenvalues explicitly.

## SIFT: scale-invariant features

David Lowe's **Scale-Invariant Feature Transform** (1999, 2004) was one of the most influential vision algorithms ever. It finds keypoints and descriptors that are invariant to scale and rotation and robust to illumination and viewpoint changes.

1. **Scale-space extrema**: build a pyramid of Gaussian-blurred images and compute **differences of Gaussians** (DoG), approximating the scale-normalised Laplacian. Keypoints are local extrema in space *and* scale — each has a characteristic size.
2. **Keypoint refinement**: sub-pixel localisation; reject low-contrast points and points on edges.
3. **Orientation assignment**: the dominant gradient direction in the neighbourhood, so descriptors can be rotated to a canonical orientation.
4. **Descriptor**: a $16 \times 16$ neighbourhood divided into $4 \times 4$ cells, each summarised by an 8-bin histogram of gradient orientations → a **128-dimensional** vector, normalised for illumination robustness.

**Matching**: find each descriptor's nearest neighbour in another image, and accept it only if it is much closer than the second-nearest (**Lowe's ratio test**, e.g. ratio < 0.75). Then use **RANSAC** to fit a geometric transformation (homography or fundamental matrix) robustly, rejecting outlier matches.

```python
import cv2
import numpy as np

img1 = cv2.imread("scene_left.jpg", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("scene_right.jpg", cv2.IMREAD_GRAYSCALE)
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

matches = cv2.BFMatcher().knnMatch(des1, des2, k=2)
good = [m for m, n in matches if m.distance < 0.75 * n.distance]     # ratio test
src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
H, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)          # robust alignment
print(len(kp1), "keypoints;", len(good), "good matches;", int(inliers.sum()), "RANSAC inliers")
```

This is the core of **panorama stitching**. Faster alternatives include SURF and **ORB** (binary descriptors, free of patent restrictions historically associated with SIFT and SURF; SIFT's patent has since expired).

## HOG: histograms of oriented gradients

Dalal and Triggs (2005) designed **HOG** for pedestrian detection:

1. Compute gradients.
2. Divide the detection window into small **cells** (e.g. $8 \times 8$ pixels) and build a histogram of gradient orientations (9 bins) per cell, weighted by magnitude.
3. Normalise histograms over overlapping **blocks** of cells for illumination invariance.
4. Concatenate into a feature vector and classify with a linear **SVM**, sliding the window over the image at multiple scales.

HOG captures **shape** through the distribution of edge directions. The later **Deformable Part Models** (Felzenszwalb et al.) extended HOG with movable parts and dominated detection benchmarks until deep learning.

```python
from skimage.feature import hog
from skimage import io, color
features, hog_image = hog(color.rgb2gray(io.imread("person.jpg")), orientations=9,
                          pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
print(features.shape)
```

## Bag of visual words

To classify whole images, the **bag-of-visual-words** model borrowed from text retrieval:

1. Extract local descriptors (e.g. SIFT) from many training images.
2. Cluster them with k-means into a **visual vocabulary** of, say, 1,000 "words".
3. Represent each image as a **histogram** of visual-word occurrences.
4. Train an SVM (often with a $\chi^2$ kernel) on the histograms.

Extensions — **spatial pyramid matching**, **VLAD**, **Fisher vectors** — added spatial layout and richer statistics and were state of the art just before 2012.

## Classical vs learned features

| | Hand-crafted (SIFT, HOG) | Learned (CNN, ViT) |
|---|---|---|
| Design | Human expertise | Learned from data |
| Data needed | Little | Large (or pretrained) |
| Invariances | Engineered explicitly | Learned (plus architecture and augmentation) |
| Recognition accuracy | Lower | Much higher |
| Geometric matching | Still widely used | Learned matchers (SuperPoint, LoFTR, LightGlue) increasingly competitive |

:::note
The deep-learning revolution replaced hand-crafted features for *recognition*, but the **pipeline ideas** — detect keypoints, describe them, match robustly, verify geometrically with RANSAC — remain the foundation of structure-from-motion, SLAM and augmented reality, now often with learned components plugged in.
:::

:::exercise
1. Stitch two overlapping photos into a panorama using SIFT, the ratio test, RANSAC and `cv2.warpPerspective`.
2. Train a HOG + linear SVM classifier to distinguish two object categories and compare with a pretrained CNN's features.
3. Build a bag-of-visual-words classifier with ORB descriptors and k-means on a small image dataset.
:::

:::takeaway
- Corners are distinctive; the Harris detector uses the structure tensor's eigenvalues.
- SIFT gives scale- and rotation-invariant 128-D descriptors; match with the ratio test and RANSAC.
- HOG describes shape by gradient-orientation histograms; with an SVM it detected pedestrians.
- Bag-of-visual-words classified images before CNNs; matching pipelines remain central in 3-D vision.
:::

=== POST ===
slug: lenet-and-alexnet
title: "LeNet and AlexNet: The Birth of Deep Vision"
category: computer-vision
level: Beginner
tags: lenet, alexnet, cnn history, imagenet, architectures
summary: Two architectures bookend the rise of CNNs. LeNet-5 read handwritten digits in the 1990s; AlexNet won ImageNet in 2012 and launched the deep learning era. We dissect both and the innovations that made AlexNet work.
---
Every modern vision architecture descends from two landmark networks. **LeNet-5** (1998) proved that convolutional networks trained with backpropagation could solve a real problem — reading handwritten digits on cheques. **AlexNet** (2012) proved they could scale to a million natural images and beat every alternative by a wide margin. Studying them shows which ideas were present from the beginning and which innovations unlocked scale.

## LeNet-5

Yann LeCun and colleagues developed a series of convolutional networks from the late 1980s; LeNet-5 was described in the 1998 paper "Gradient-Based Learning Applied to Document Recognition". Systems based on this work were deployed to read a significant share of cheques in the United States.

Architecture for $32 \times 32$ grayscale input:

| Layer | Operation | Output |
|---|---|---|
| C1 | 6 filters $5\times5$ | $6 \times 28 \times 28$ |
| S2 | Subsampling (average pool) $2\times2$ | $6 \times 14 \times 14$ |
| C3 | 16 filters $5\times5$ | $16 \times 10 \times 10$ |
| S4 | Subsampling $2\times2$ | $16 \times 5 \times 5$ |
| C5 | 120 filters $5\times5$ | 120 |
| F6 | Fully connected | 84 |
| Output | 10 classes | 10 |

About 60,000 parameters, tanh-like activations. All the core ideas are already here: local receptive fields, weight sharing, subsampling, hierarchical features and end-to-end training by gradient descent.

```python
import torch.nn as nn

class LeNet5(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, 5), nn.Tanh(), nn.AvgPool2d(2),
            nn.Conv2d(6, 16, 5), nn.Tanh(), nn.AvgPool2d(2),
            nn.Conv2d(16, 120, 5), nn.Tanh())
        self.classifier = nn.Sequential(nn.Linear(120, 84), nn.Tanh(), nn.Linear(84, n_classes))
    def forward(self, x):                       # x: (N, 1, 32, 32)
        return self.classifier(self.features(x).flatten(1))
```

So why did CNNs not take over computer vision in the 1990s? Datasets were small, computers were slow, and on the datasets of the time, SVMs with hand-crafted features were competitive and easier to use.

## ImageNet

Fei-Fei Li and colleagues built **ImageNet**, eventually containing over 14 million labelled images, labelled via crowdsourcing. The ImageNet Large Scale Visual Recognition Challenge (ILSVRC), run from 2010 to 2017, used a 1,000-class subset with about 1.2 million training images. It provided exactly what deep networks needed: **scale**.

## AlexNet

In 2012 Alex Krizhevsky, Ilya Sutskever and Geoffrey Hinton entered a deep CNN in ILSVRC and achieved a top-5 error of about **15.3%**, versus about **26.2%** for the next-best entry — an unprecedented gap.

Architecture (for $224 \times 224 \times 3$ input): five convolutional layers (the first with $11 \times 11$ filters and stride 4), three max-pooling layers and three fully connected layers (4096, 4096, 1000), with about **60 million parameters** — a thousand times more than LeNet.

## The innovations that mattered

1. **ReLU activations** — trained several times faster than tanh units, avoiding saturation.
2. **GPU training** — trained on two NVIDIA GTX 580 GPUs (3 GB each) for about a week; the model was split across the two GPUs.
3. **Dropout** (p = 0.5) in the fully connected layers — crucial against overfitting the 60 million parameters.
4. **Data augmentation** — random $224 \times 224$ crops from $256 \times 256$ images, horizontal flips, and PCA-based colour perturbation, enlarging the effective dataset.
5. **Overlapping max pooling** ($3 \times 3$ windows, stride 2).
6. **Local response normalisation** — later abandoned in favour of batch normalisation.
7. **SGD with momentum 0.9 and weight decay**, with the learning rate reduced manually when validation error plateaued.

```python
from torchvision import models
alexnet = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
print(alexnet)
print("parameters:", sum(p.numel() for p in alexnet.parameters()))   # ~61 million
```

:::note
AlexNet's first-layer filters, visualised in the paper, show oriented edges and colour blobs — learned automatically, resembling Gabor filters. Notice also where the parameters live: the vast majority are in the fully connected layers. Later architectures replaced these with global average pooling, cutting parameters dramatically.
:::

## Impact

After 2012, almost every competitive ILSVRC entry used deep CNNs, and the error rate fell year after year (ZFNet 2013, VGG and GoogLeNet 2014, ResNet 2015). Industry invested heavily in GPUs and deep learning, and the approach spread to speech, language and beyond. The recipe — **big data + big compute + deep networks trained end-to-end** — became the paradigm of modern AI.

:::exercise
1. Compute the output shape after each layer of AlexNet and identify which layer holds the most parameters.
2. Train LeNet-5 on MNIST (pad images to $32 \times 32$). Then replace tanh with ReLU and average pooling with max pooling. Compare accuracy and convergence speed.
3. Visualise the first-layer filters of a pretrained AlexNet.
:::

:::takeaway
- LeNet-5 established convolution, pooling and end-to-end training for digit recognition.
- ImageNet provided the scale deep networks needed.
- AlexNet (2012) cut ImageNet top-5 error from ~26% to ~15% using ReLU, GPUs, dropout and augmentation.
- The big-data + big-compute + deep-network recipe launched the modern AI era.
:::

=== POST ===
slug: vgg-and-inception
title: VGG and GoogLeNet/Inception — Depth and Multi-Scale Design
category: computer-vision
level: Intermediate
tags: vgg, inception, googlenet, cnn architectures, 1x1 convolution
summary: In 2014 two architectures pushed CNNs deeper in opposite styles: VGG with uniform stacks of 3×3 convolutions, GoogLeNet with parallel multi-scale Inception modules and 1×1 bottlenecks. We compare their designs and lessons.
---
The 2014 ImageNet challenge produced two architectures that shaped the next decade. **VGGNet** from Oxford's Visual Geometry Group showed that simply going deeper with a clean, uniform design works. **GoogLeNet** (Inception v1) from Google showed that careful design could achieve higher accuracy with far fewer parameters. Their ideas — small kernels, 1×1 bottlenecks, global pooling, multi-branch blocks — are everywhere today.

## VGG: simplicity and depth

Simonyan and Zisserman (2014) used a single design rule: **only $3 \times 3$ convolutions (stride 1, padding 1) and $2 \times 2$ max pooling**, with the number of channels doubling after each pool (64 → 128 → 256 → 512 → 512). VGG-16 has 13 convolutional layers and 3 fully connected layers.

Why $3 \times 3$? As we computed in the receptive-field lecture:

- Two stacked $3 \times 3$ layers see a $5 \times 5$ region; three see $7 \times 7$.
- Three $3 \times 3$ layers use $3 \times 9C^2 = 27C^2$ weights versus $49C^2$ for one $7 \times 7$ layer.
- Each extra layer adds a non-linearity, making the function more expressive.

VGG-16 reached about 7.3% top-5 error. Its weaknesses: roughly **138 million parameters** (most in the first fully connected layer: $7 \times 7 \times 512 \times 4096 \approx 103$ million) and heavy computation (~15 GFLOPs per image). Yet its simplicity made it a favourite backbone for transfer learning, style transfer (VGG features define "perceptual loss") and early detection and segmentation systems.

## GoogLeNet and the Inception module

Szegedy et al. (2014) asked: which filter size should a layer use — $1\times1$, $3\times3$ or $5\times5$? **Why not all of them in parallel?** The **Inception module** runs several branches side by side and concatenates their outputs along the channel dimension:

```text
            ┌── 1×1 conv ─────────────────────┐
            ├── 1×1 conv → 3×3 conv ──────────┤
input ──────┼── 1×1 conv → 5×5 conv ──────────┼── concatenate channels
            └── 3×3 max pool → 1×1 conv ──────┘
```

This captures features at multiple scales simultaneously.

### The 1×1 bottleneck

Naively, $5 \times 5$ convolutions on many channels are expensive. The key trick: first reduce channels with a cheap **$1 \times 1$ convolution**. For 256 input channels and 64 output channels:

- Direct $5 \times 5$: $256 \times 64 \times 25 \approx 410{,}000$ weights.
- $1 \times 1$ to 32 channels, then $5 \times 5$ to 64: $256 \times 32 + 32 \times 64 \times 25 \approx 59{,}000$ weights — about 7× fewer.

A $1 \times 1$ convolution is a small fully connected layer applied at every pixel across channels (the "network in network" idea of Lin et al., 2013).

### Other GoogLeNet features

- **22 layers deep** yet only about **7 million parameters** — roughly 20× fewer than VGG-16.
- **Global average pooling** instead of large fully connected layers.
- **Auxiliary classifiers** attached to intermediate layers during training to inject extra gradient into the middle of the network (later found to act mainly as regularisers).
- Top-5 error of about 6.7%, winning ILSVRC 2014.

```python
import torch
import torch.nn as nn

class InceptionModule(nn.Module):
    def __init__(self, cin, c1, c3r, c3, c5r, c5, cp):
        super().__init__()
        conv = lambda i, o, k: nn.Sequential(nn.Conv2d(i, o, k, padding=k // 2), nn.ReLU(inplace=True))
        self.b1 = conv(cin, c1, 1)
        self.b2 = nn.Sequential(conv(cin, c3r, 1), conv(c3r, c3, 3))
        self.b3 = nn.Sequential(conv(cin, c5r, 1), conv(c5r, c5, 5))
        self.b4 = nn.Sequential(nn.MaxPool2d(3, 1, 1), conv(cin, cp, 1))
    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)

m = InceptionModule(192, 64, 96, 128, 16, 32, 32)       # "inception 3a" configuration
print(m(torch.randn(1, 192, 28, 28)).shape)              # (1, 256, 28, 28)
```

## Later Inception versions

- **Inception v2/v3** (2015): factorised $5 \times 5$ into two $3 \times 3$, and $n \times n$ into $1 \times n$ followed by $n \times 1$; added batch normalisation and label smoothing (which was introduced in this line of work).
- **Inception v4 / Inception-ResNet** (2016): combined Inception modules with residual connections.
- **Xception** (2017): pushed the idea to its extreme with depthwise separable convolutions — each channel's spatial pattern processed separately, then mixed with $1 \times 1$ convs.

## Comparing the two philosophies

| | VGG-16 | GoogLeNet |
|---|---|---|
| Depth | 16 | 22 |
| Parameters | ~138M | ~7M |
| Design | Uniform, simple | Multi-branch, engineered |
| Strength | Easy to understand and adapt | Efficient, accurate |
| Lasting ideas | Small kernels, doubling channels | 1×1 bottlenecks, multi-scale branches, global average pooling |

:::note
Both networks were near the limit of trainable depth in 2014: going much deeper made training worse (the degradation problem). The next year, ResNet's residual connections broke this barrier — the subject of the next lecture.
:::

:::exercise
1. Verify the parameter savings of the 1×1 bottleneck for your own choice of channel counts.
2. Count VGG-16's parameters layer by layer and compute the fraction in the fully connected layers.
3. Fine-tune pretrained VGG-16 and GoogLeNet on a small dataset and compare accuracy, speed and memory.
:::

:::takeaway
- VGG stacks uniform 3×3 convolutions: simple, deep, but parameter-heavy (~138M).
- Inception modules run parallel multi-scale branches and use 1×1 bottlenecks to stay efficient (~7M parameters).
- Global average pooling replaced giant fully connected layers.
- Factorised convolutions and depthwise separability grew out of the Inception line.
:::

=== POST ===
slug: resnet-architecture
title: "ResNet in Depth: Architecture, Bottlenecks and Variants"
category: computer-vision
level: Intermediate
tags: resnet, residual networks, bottleneck, resnext, architectures
summary: ResNet's residual blocks enabled 152-layer networks and became the default vision backbone. We study basic and bottleneck blocks, the full ResNet-50 layout, training recipe, and descendants such as ResNeXt and ConvNeXt.
---
In the Deep Learning track we met the **idea** of residual connections. Here we study **ResNet** as a concrete vision architecture — the backbone behind countless detection, segmentation and medical-imaging systems, and still one of the first models practitioners try. ResNet won ILSVRC 2015 with a top-5 error of about 3.6% using an ensemble of networks up to 152 layers deep.

## Building blocks

### Basic block (ResNet-18/34)
Two $3 \times 3$ convolutions with BatchNorm and ReLU, plus an identity shortcut:

```text
x ──► conv3×3 ─ BN ─ ReLU ─ conv3×3 ─ BN ──(+)── ReLU ──►
│                                         ▲
└─────────────── identity ────────────────┘
```

### Bottleneck block (ResNet-50/101/152)
For deeper networks, a three-layer design keeps computation manageable:

1. $1 \times 1$ conv **reduces** channels (e.g. 256 → 64);
2. $3 \times 3$ conv operates on the reduced channels;
3. $1 \times 1$ conv **restores** channels (64 → 256).

The expensive $3 \times 3$ operation runs on 4× fewer channels. A bottleneck block with 256 input/output channels has about 70,000 weights, compared with about 1.2 million for two plain $3 \times 3$ layers at 256 channels.

When the spatial size halves (stride 2) and channels double, the shortcut uses a $1 \times 1$ convolution with stride 2 (a "projection shortcut").

## The ResNet-50 layout

| Stage | Output size (224 input) | Blocks |
|---|---|---|
| Stem: $7 \times 7$ conv, stride 2 + $3 \times 3$ max pool | $56 \times 56$ | — |
| conv2_x | $56 \times 56$, 256 channels | 3 bottlenecks |
| conv3_x | $28 \times 28$, 512 | 4 |
| conv4_x | $14 \times 14$, 1024 | 6 |
| conv5_x | $7 \times 7$, 2048 | 3 |
| Global average pool + FC | 1000 classes | — |

About **25.6 million parameters** and ~4 GFLOPs per image — fewer parameters than VGG-16 despite being far deeper and more accurate.

```python
import torch
from torchvision import models

resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
print(sum(p.numel() for p in resnet.parameters()) / 1e6, "M parameters")

# Use as a feature extractor: drop the classification head
backbone = torch.nn.Sequential(*list(resnet.children())[:-1])
feats = backbone(torch.randn(2, 3, 224, 224)).flatten(1)
print(feats.shape)                      # (2, 2048) image embeddings

# Multi-scale feature maps for detection/segmentation
from torchvision.models.feature_extraction import create_feature_extractor
fx = create_feature_extractor(resnet, {"layer2": "c3", "layer3": "c4", "layer4": "c5"})
print({k: v.shape for k, v in fx(torch.randn(1, 3, 224, 224)).items()})
```

## Training recipe and "ResNet strikes back"

The original recipe: SGD with momentum 0.9, weight decay $10^{-4}$, batch size 256, learning rate 0.1 divided by 10 when error plateaus, simple crop-and-flip augmentation, ~90 epochs.

Later work showed the **same ResNet-50** can gain several points of ImageNet accuracy from a modern recipe alone — longer training, cosine schedules, label smoothing, mixup/cutmix, RandAugment, stochastic depth, EMA of weights (Bello et al., 2021 "Revisiting ResNets"; Wightman et al., 2021 "ResNet strikes back"). A valuable lesson: **compare architectures under equal training recipes**.

## Variants and descendants

- **Pre-activation ResNet** (2016): BN → ReLU → conv ordering inside the branch, leaving a clean identity path; trained 1,000-layer networks.
- **Wide ResNet** (2016): fewer, wider layers can match or beat very deep thin ones.
- **ResNeXt** (2017): **grouped convolutions** in the bottleneck — many parallel paths ("cardinality") — a cleaner form of Inception's multi-branch idea; better accuracy at similar cost.
- **SE-Net** (2018): **squeeze-and-excitation** blocks reweight channels using global context:

$$
\mathbf{s} = \sigma\big(\mathbf{W}_2\,\text{ReLU}(\mathbf{W}_1\,\text{GAP}(\mathbf{X}))\big), \qquad \tilde{\mathbf{X}}_c = s_c\,\mathbf{X}_c
$$

  A form of channel attention; won ILSVRC 2017.
- **ResNet-D and bag of tricks** (He et al., 2019): small stem and downsampling changes that add accuracy for free.
- **ConvNeXt** (2022): modernised ResNet design choices one by one towards Vision-Transformer-style components (larger $7 \times 7$ depthwise kernels, LayerNorm, GELU, inverted bottlenecks, fewer activations) and showed pure CNNs can match ViTs at similar scale.

:::note
ResNets remain excellent default backbones when you need a well-understood, efficient model with strong transfer learning — especially for medium-sized datasets, medical imaging and deployment on modest hardware. Many detection and segmentation frameworks default to ResNet-50 with a Feature Pyramid Network.
:::

:::exercise
1. Count parameters of a bottleneck block (256 → 64 → 64 → 256) and compare with two $3 \times 3$ convs at 256 channels.
2. Fine-tune ResNet-18 and ResNet-50 on a small dataset; compare accuracy, speed and memory.
3. Add squeeze-and-excitation to the basic block from the Deep Learning track and measure its effect on CIFAR-10.
:::

:::takeaway
- ResNet uses basic (2-layer) or bottleneck (1×1–3×3–1×1) residual blocks with projection shortcuts when shapes change.
- ResNet-50 has ~25.6M parameters, four stages and global average pooling.
- Modern training recipes add substantial accuracy to the same architecture.
- ResNeXt, SE-Net and ConvNeXt extend the design; ResNets remain strong default backbones.
:::

=== POST ===
slug: efficientnet-and-model-scaling
title: "EfficientNet and Principled Model Scaling"
category: computer-vision
level: Intermediate
tags: efficientnet, compound scaling, mbconv, efficiency, architectures
summary: How should a network grow when you have more compute — deeper, wider or higher resolution? EfficientNet's compound scaling answers "all three, in balance". We cover MBConv blocks, the B0–B7 family and EfficientNetV2.
---
Given twice the compute, how should you enlarge a CNN? Before 2019, practitioners typically scaled one dimension: ResNet grew deeper (18 → 152 layers), Wide ResNets grew wider, and some models used higher-resolution input. Mingxing Tan and Quoc Le's **EfficientNet** (2019) showed that scaling all three dimensions together, in a fixed ratio, gives much better accuracy per unit of compute.

## Three scaling dimensions

- **Depth** $d$ — more layers capture richer, more complex features but face diminishing returns and harder optimisation.
- **Width** $w$ — more channels capture finer-grained features; very wide shallow networks struggle to learn high-level features.
- **Resolution** $r$ — larger input images reveal finer detail; returns diminish at very high resolution.

Experiments showed that scaling any single dimension saturates quickly (e.g. accuracy stalls around 80% on ImageNet), and the dimensions interact: higher-resolution images benefit from deeper networks (bigger receptive fields) and wider networks (more fine-grained patterns).

## Compound scaling

EfficientNet scales all three with a single **compound coefficient** $\phi$:

$$
d = \alpha^\phi, \qquad w = \beta^\phi, \qquad r = \gamma^\phi, \qquad \text{subject to } \alpha\cdot\beta^2\cdot\gamma^2 \approx 2, \;\; \alpha, \beta, \gamma \ge 1
$$

FLOPs of a convolutional network scale roughly linearly with depth and quadratically with width and resolution, so this constraint makes total FLOPs grow by about $2^\phi$. A small grid search on the baseline found $\alpha = 1.2$, $\beta = 1.1$, $\gamma = 1.15$.

## The baseline: EfficientNet-B0 and MBConv

The baseline network was found by **hardware-aware neural architecture search** optimising accuracy and FLOPs. Its building block is the **MBConv** (mobile inverted bottleneck) from MobileNetV2, plus squeeze-and-excitation:

1. $1 \times 1$ conv **expands** channels (e.g. 6×);
2. **depthwise** $3 \times 3$ or $5 \times 5$ conv filters each channel spatially;
3. **squeeze-and-excitation** reweights channels;
4. $1 \times 1$ conv **projects** back to fewer channels;
5. residual connection when shapes match (with stochastic depth during training).

It uses the **SiLU (Swish)** activation.

```python
import torch
import torch.nn as nn

class MBConv(nn.Module):
    def __init__(self, cin, cout, expand=6, k=3, stride=1, se_ratio=0.25):
        super().__init__()
        mid = cin * expand
        self.use_res = stride == 1 and cin == cout
        self.block = nn.Sequential(
            nn.Conv2d(cin, mid, 1, bias=False), nn.BatchNorm2d(mid), nn.SiLU(),
            nn.Conv2d(mid, mid, k, stride, k // 2, groups=mid, bias=False), nn.BatchNorm2d(mid), nn.SiLU())
        se = max(1, int(cin * se_ratio))
        self.se = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(mid, se, 1), nn.SiLU(),
                                nn.Conv2d(se, mid, 1), nn.Sigmoid())
        self.project = nn.Sequential(nn.Conv2d(mid, cout, 1, bias=False), nn.BatchNorm2d(cout))
    def forward(self, x):
        h = self.block(x)
        h = self.project(h * self.se(h))
        return x + h if self.use_res else h

print(MBConv(40, 40)(torch.randn(1, 40, 28, 28)).shape)
```

## The EfficientNet family

Applying $\phi = 1, \dots, 7$ to B0 produced **EfficientNet-B1 to B7**. At publication, EfficientNet-B7 reached about 84.3% ImageNet top-1 accuracy while being 8.4× smaller and 6.1× faster at inference than the best existing CNN of similar accuracy. Smaller variants matched ResNet-50 with far fewer FLOPs and parameters, and the models transferred well to other datasets.

| Model | Input resolution | Parameters (approx.) |
|---|---|---|
| B0 | 224 | 5.3M |
| B3 | 300 | 12M |
| B5 | 456 | 30M |
| B7 | 600 | 66M |

## EfficientNetV2

EfficientNet's FLOP efficiency did not always translate into fast **training**: depthwise convolutions underuse accelerators, and large images consume memory. **EfficientNetV2** (Tan & Le, 2021) addressed this with:

- **Fused-MBConv** blocks in early stages (a regular $3 \times 3$ conv replaces the expansion + depthwise pair, which runs faster on GPUs/TPUs);
- training-aware NAS optimising accuracy, speed and parameter efficiency;
- **progressive learning**: train with small images and weak regularisation first, then larger images with stronger regularisation.

## Lessons

:::note
1. **Balance beats extremes** — grow depth, width and resolution together.
2. **FLOPs are not latency** — memory access, parallelism and operator support on the target hardware determine real speed. Always benchmark on the actual device.
3. **Scaling rules generalise** — the same thinking underlies scaling laws for language models, where the balance is between parameters and training tokens.
:::

:::exercise
1. Using $\alpha = 1.2$, $\beta = 1.1$, $\gamma = 1.15$, compute depth, width and resolution multipliers for $\phi = 3$ and verify FLOPs roughly multiply by 8.
2. Fine-tune `torchvision.models.efficientnet_b0` and `resnet50` on a small dataset; compare accuracy, GPU throughput and CPU latency.
3. Explain why depthwise convolutions can have low FLOPs yet run slowly on GPUs.
:::

:::takeaway
- Scaling one dimension saturates; compound scaling grows depth, width and resolution together.
- EfficientNet-B0 was found by NAS; its MBConv blocks use expansion, depthwise convs and squeeze-and-excitation.
- The B0–B7 family achieved strong accuracy per FLOP; V2 optimised training speed with Fused-MBConv and progressive learning.
- Benchmark latency on real hardware — FLOPs can mislead.
:::

=== POST ===
slug: mobilenet-efficient-architectures
title: "MobileNet and Efficient Architectures for Edge Devices"
category: computer-vision
level: Intermediate
tags: mobilenet, depthwise separable convolution, edge ai, efficiency, shufflenet
summary: Phones, drones and microcontrollers need vision models that are small and fast. We derive the cost savings of depthwise separable convolutions and study MobileNet V1–V3, ShuffleNet and design principles for efficient inference.
---
A crop-disease detector used by farmers without reliable internet, a reading aid for the visually impaired, a camera that counts vehicles on a solar-powered pole — these applications must run **on the device**, with limited memory, compute and battery. Server-class models like ResNet-50 are too heavy. A family of architectures, led by **MobileNet**, was designed specifically for this setting.

## The cost of a standard convolution

For an input with $M$ channels, output $N$ channels, kernel $D_K \times D_K$ and output feature map $D_F \times D_F$, a standard convolution costs

$$
D_K \cdot D_K \cdot M \cdot N \cdot D_F \cdot D_F
$$

multiply–accumulate operations: every output channel mixes every input channel at every spatial position.

## Depthwise separable convolutions

MobileNet (Howard et al., 2017) factorises this into two steps:

1. **Depthwise convolution** — one $D_K \times D_K$ filter **per input channel** (spatial filtering only): cost $D_K^2 \cdot M \cdot D_F^2$.
2. **Pointwise convolution** — a $1 \times 1$ convolution mixing channels: cost $M \cdot N \cdot D_F^2$.

The ratio of costs is

$$
\frac{D_K^2 M D_F^2 + M N D_F^2}{D_K^2 M N D_F^2} = \frac{1}{N} + \frac{1}{D_K^2}
$$

With $3 \times 3$ kernels, depthwise separable convolution is about **8–9× cheaper**, with only a small accuracy drop.

```python
import torch.nn as nn

def depthwise_separable(cin, cout, stride=1):
    return nn.Sequential(
        nn.Conv2d(cin, cin, 3, stride, 1, groups=cin, bias=False),   # depthwise
        nn.BatchNorm2d(cin), nn.ReLU6(inplace=True),
        nn.Conv2d(cin, cout, 1, bias=False),                         # pointwise
        nn.BatchNorm2d(cout), nn.ReLU6(inplace=True))

std = nn.Conv2d(256, 256, 3, padding=1, bias=False)
sep = depthwise_separable(256, 256)
count = lambda m: sum(p.numel() for p in m.parameters() if p.dim() > 1)
print("standard:", count(std), " separable:", count(sep))   # ~589k vs ~68k weights
```

MobileNetV1 also introduced two global knobs: a **width multiplier** $\alpha$ (thin every layer's channels) and a **resolution multiplier** $\rho$ (smaller inputs), letting one design family trade accuracy for speed across devices.

## MobileNetV2: inverted residuals and linear bottlenecks

Sandler et al. (2018) introduced the **inverted residual block**:

- **Expand** a thin input (e.g. 24 channels) with a $1 \times 1$ conv by a factor of 6;
- apply a **depthwise** $3 \times 3$ conv in the wide space;
- **project** back to a thin output with a $1 \times 1$ conv — with **no activation** (a *linear bottleneck*), because ReLU in a low-dimensional space destroys information;
- add a residual connection between the **thin** ends.

It is "inverted" relative to ResNet's bottleneck, which is wide at the ends and thin in the middle. Keeping the residual stream thin saves memory — critical on mobile devices.

## MobileNetV3

Howard et al. (2019) combined hardware-aware NAS (for block choices) with **NetAdapt** (for layer widths), added **squeeze-and-excitation** modules, and introduced the cheap **hard-swish** activation:

$$
\text{h-swish}(x) = x\cdot\frac{\text{ReLU6}(x + 3)}{6}
$$

a piecewise-linear approximation of Swish that is fast and quantisation-friendly. It was released in Large and Small variants for different budgets.

## Other efficient designs

- **ShuffleNet** (2018): grouped $1 \times 1$ convolutions plus a **channel shuffle** so information flows between groups. **ShuffleNetV2** proposed practical guidelines for real speed: equal input/output channel widths minimise memory access cost; excessive group convolution and network fragmentation slow things down; element-wise operations are not free.
- **SqueezeNet** (2016): AlexNet-level accuracy with 50× fewer parameters using "fire" modules.
- **GhostNet**: generate some feature maps with cheap linear operations.
- **EfficientNet-Lite, MobileViT, EfficientFormer, MobileNetV4**: efficient CNN and hybrid CNN–transformer designs tuned for mobile accelerators.

## Deploying on the edge

Architecture is only part of the story. The full recipe:

1. **Choose an efficient backbone** matched to the hardware (CPU, mobile GPU, NPU, microcontroller).
2. **Quantise** to INT8 (post-training or quantisation-aware).
3. **Prune** or distil if needed.
4. **Convert** to a mobile runtime: TensorFlow Lite / LiteRT, ONNX Runtime Mobile, Core ML, ExecuTorch, or TensorFlow Lite Micro for microcontrollers.
5. **Benchmark on the actual device** — latency, memory, battery drain and accuracy on realistic, locally collected images.

```python
import torch
from torchvision import models

model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1).eval()
print(sum(p.numel() for p in model.parameters()) / 1e6, "M parameters")   # ~2.5M
example = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, example, "mobilenet_v3_small.onnx", opset_version=17)
```

:::note
On-device inference is not only about speed. It keeps personal images **on the user's phone** (privacy), works **offline** in areas with poor connectivity, and avoids per-request server costs. For humanitarian and development applications, these properties are often decisive.
:::

:::exercise
1. Derive the cost ratio $1/N + 1/D_K^2$ and evaluate it for $N = 256$, $D_K = 3$.
2. Fine-tune MobileNetV3-Small and ResNet-50 on a small dataset; compare accuracy and CPU latency on a laptop.
3. Convert a fine-tuned MobileNet to TensorFlow Lite or ONNX with INT8 quantisation and measure size and accuracy changes.
:::

:::takeaway
- Depthwise separable convolutions cut computation by ~8–9× for 3×3 kernels.
- MobileNetV2's inverted residuals with linear bottlenecks keep memory low.
- MobileNetV3 adds NAS, squeeze-and-excitation and hard-swish; ShuffleNetV2 gives practical speed guidelines.
- Real edge deployment combines efficient architectures, quantisation, mobile runtimes and on-device benchmarking.
:::
