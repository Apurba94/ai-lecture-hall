=== POST ===
slug: ocr-and-document-ai
title: "OCR and Document AI: From Scanned Forms to Structured Data"
category: computer-vision
level: Intermediate
tags: ocr, document ai, text recognition, ctc, layout analysis
summary: Much of the world's information is locked in scanned forms, receipts and handwritten records. We cover text detection, recognition with CRNN and CTC, layout-aware models, multilingual challenges and end-to-end document understanding.
---
Registration forms, identity documents, invoices, historical archives, handwritten case notes — enormous amounts of information exist only as images of text. **Optical Character Recognition (OCR)** and broader **Document AI** convert them into searchable, structured data. For organisations processing thousands of paper forms, this can save enormous amounts of manual data entry — but errors can have real consequences for the people whose records are processed.

## The classical OCR pipeline

1. **Preprocessing**: deskew, denoise, binarise (e.g. Otsu or adaptive thresholding), correct perspective (homography for phone photos).
2. **Text detection**: find regions containing text — lines or words.
3. **Text recognition**: convert each region's image into a character string.
4. **Post-processing**: language models, dictionaries and validation rules (dates, ID number checksums).
5. **Information extraction**: map text to fields ("Name", "Date of birth").

## Text detection

Scene and document text varies in orientation, size and shape. Deep detectors include **EAST** (predicts rotated boxes per pixel), **CRAFT** (predicts character regions and the affinity between neighbouring characters, handling curved text) and **DBNet** (differentiable binarisation of a text probability map). For clean scanned documents, simpler line segmentation often suffices.

## Text recognition with CRNN and CTC

A line image has variable width and text has variable length, and we rarely know exactly which pixels correspond to which character. The **CRNN** architecture (Shi et al., 2015) solved this elegantly:

1. A **CNN** extracts a feature sequence along the horizontal axis (one vector per column-slice).
2. A **bidirectional LSTM** models context along the sequence.
3. **Connectionist Temporal Classification (CTC)** loss (Graves et al., 2006) trains without character-level alignment.

CTC adds a **blank** symbol and sums over all alignments that collapse to the target string: repeated characters are merged, then blanks removed ("hh-e-ll-ll-oo" → "hello"; a blank between the two l's preserves the double letter). The loss is

$$
\mathcal{L}_{\text{CTC}} = -\log\sum_{\pi \in \mathcal{B}^{-1}(\mathbf{y})}\prod_{t=1}^{T}p(\pi_t \mid \mathbf{x})
$$

computed efficiently by dynamic programming (a forward–backward algorithm, like HMMs).

```python
import torch
import torch.nn as nn

T, N, C = 32, 4, 37                         # time steps, batch, classes (blank + 26 letters + 10 digits)
log_probs = torch.randn(T, N, C).log_softmax(2).requires_grad_()   # model output per column
targets = torch.randint(1, C, (N, 8))        # label indices (0 is reserved for blank)
loss = nn.CTCLoss(blank=0)(log_probs, targets,
                           input_lengths=torch.full((N,), T), target_lengths=torch.full((N,), 8))
loss.backward(); print(loss.item())

def greedy_decode(log_probs_1seq, alphabet):
    best = log_probs_1seq.argmax(-1).tolist()
    out, prev = [], None
    for k in best:
        if k != prev and k != 0:
            out.append(alphabet[k - 1])
        prev = k
    return "".join(out)
```

Transformer-based recognisers (e.g. **TrOCR**, pairing a ViT encoder with a text decoder) now achieve excellent results, including on handwriting.

## Practical OCR tools

```python
# pip install pytesseract easyocr   (Tesseract binary must also be installed)
import easyocr
reader = easyocr.Reader(["en", "bn"])        # English and Bangla
for box, text, conf in reader.readtext("form_photo.jpg"):
    print(f"{conf:.2f}  {text}")
```

Tesseract, EasyOCR, PaddleOCR and cloud APIs differ in language coverage, accuracy on handwriting and deployment options (on-premise vs cloud — important for sensitive documents).

## Layout and document understanding

Documents are not just text: meaning depends on **layout** — tables, key–value pairs, headers, checkboxes. Modern Document AI models combine three signals:

- **Text** (OCR tokens),
- **Layout** (2-D positions of tokens),
- **Image** (visual appearance).

The **LayoutLM** family adds 2-D position embeddings to BERT-like models and pretrains on millions of document pages; it excels at form understanding (extracting fields) and document classification. **OCR-free** models such as Donut read document images directly and output structured JSON. Large multimodal language models can now answer questions about documents and extract fields from prompts — powerful, but they must be validated carefully because they can **hallucinate** plausible values.

## Multilingual and handwritten challenges

- Scripts with complex shaping and conjuncts (Bangla, Devanagari, Arabic with right-to-left text and cursive joining) are harder and often under-resourced.
- Handwriting varies enormously; historical documents add degradation.
- Mixed-language forms are common in humanitarian and administrative settings.

Fine-tuning recognisers on in-domain, local-script samples usually yields large gains.

## Evaluating OCR

- **Character Error Rate (CER)** and **Word Error Rate (WER)**: edit distance (substitutions + deletions + insertions) divided by reference length.
- **Field-level accuracy** for extraction: exact match per field matters more than average CER — one wrong digit in an ID number invalidates the field.

:::warning
For forms that affect people's access to services, design a **human-in-the-loop** workflow: route low-confidence fields to reviewers, validate formats (dates, check digits), and never let OCR errors silently change a person's record. Process sensitive documents on secure infrastructure, and minimise retention of images.
:::

:::exercise
1. Compute CER and WER between "Janin Apurba, 12/05/1997" and an OCR output of "Janln Apurba 12/06/1997".
2. Walk through CTC collapsing for the path "--hh-ee-l-ll-oo--" and explain why the blank is needed for "hello".
3. Run two OCR engines on 20 photographed forms in your local language and compare field-level accuracy.
:::

:::takeaway
- OCR = preprocessing → text detection → recognition → post-processing → extraction.
- CRNN + CTC recognises variable-length text without character alignment; transformers (TrOCR) push accuracy further.
- Layout-aware models (LayoutLM, Donut) understand forms; multimodal LLMs help but must be validated.
- Measure CER/WER and field accuracy; keep humans in the loop for consequential documents.
:::

=== POST ===
slug: medical-imaging-ai
title: "AI in Medical Imaging: Opportunities, Pitfalls and Validation"
category: computer-vision
level: Intermediate
tags: medical imaging, radiology, pathology, validation, healthcare ai
summary: Deep learning can detect disease in X-rays, retinal scans and pathology slides. We survey modalities and tasks, discuss data and labelling challenges, shortcut learning, rigorous clinical validation, and deployment responsibilities.
---
Medical imaging is one of the most promising — and most demanding — application areas of computer vision. Models have matched specialists on specific tasks such as detecting diabetic retinopathy in retinal photographs or finding certain findings on chest X-rays. In places with too few radiologists, AI screening could extend care to millions. But the history of medical AI also contains many models that performed brilliantly in papers and poorly in hospitals. Today we study both the potential and the discipline required.

## Modalities and tasks

| Modality | Typical tasks |
|---|---|
| Chest X-ray | Classification (pneumonia, tuberculosis, nodules), triage |
| CT / MRI (3-D volumes) | Segmentation of organs and tumours, detection of haemorrhage |
| Retinal fundus / OCT | Diabetic retinopathy grading, glaucoma screening |
| Digital pathology (gigapixel slides) | Tumour detection, grading, cell counting |
| Dermatology photographs | Lesion classification |
| Ultrasound | Fetal biometry, cardiac function, point-of-care screening |
| Mammography | Breast cancer screening |

Architectures are familiar: CNNs and ViTs for classification, U-Net variants (including 3-D U-Net and the self-configuring **nnU-Net**, a very strong baseline) for segmentation, and **multiple-instance learning** for gigapixel pathology slides, where a slide-level label must be learned from thousands of tiles.

## Data challenges

- **Small labelled datasets** — expert labelling is expensive and slow.
- **Label noise and disagreement** — radiologists disagree; use multiple readers, adjudication, or labels from stronger reference standards (biopsy, follow-up) where possible.
- **Class imbalance** — disease is rarer than normal findings.
- **Privacy** — de-identify DICOM headers and burned-in text; follow legal and ethical approval processes.
- **Heterogeneity** — different scanners, protocols, populations and hospitals.

Transfer learning, self-supervised pretraining on unlabelled scans, heavy but anatomically valid augmentation, and federated learning (training across hospitals without sharing raw data) help address these.

## Shortcut learning: the central danger

Models learn whatever predicts the label in the training data — including artefacts:

- A pneumonia classifier trained on data from several hospitals learned to recognise **which hospital** an image came from (hospitals had different pneumonia prevalence and scanner markers), inflating accuracy (Zech et al., 2018).
- Studies found models using **chest tubes** (a treatment already given) to "detect" pneumothorax, or ruler markings in dermatology photos (rulers were more common beside suspicious lesions).
- Deep networks have been shown able to predict patient attributes such as self-reported race from X-rays in ways humans cannot, raising concerns about hidden biases in downstream predictions.

:::warning
High AUC on an internal test set is not evidence of clinical value. Check what the model uses (saliency maps, stratified analysis by hospital and device), test on **external** data from different sites, and look for confounders such as treatment devices or text markers.
:::

## Rigorous evaluation

1. **Split by patient**, never by image — images of the same patient must not appear in both training and test sets.
2. **External validation** on data from other hospitals, countries and devices.
3. **Clinically meaningful metrics**: sensitivity and specificity at an operating point chosen with clinicians; negative predictive value for rule-out tools; calibration.
4. **Subgroup analysis**: age, sex, ethnicity, disease severity, device type.
5. **Compare with clinicians** on the same cases — and, better, measure **clinician + AI** versus clinician alone, since most systems assist rather than replace.
6. **Prospective studies** and, ideally, randomised trials measuring patient outcomes.
7. **Reporting guidelines**: CLAIM, TRIPOD+AI, CONSORT-AI and STARD-AI promote transparent reporting.

```python
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix

def clinical_report(y_true, scores, threshold, groups):
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    print(f"AUC {roc_auc_score(y_true, scores):.3f} | sensitivity {tp / (tp + fn):.3f} | "
          f"specificity {tn / (tn + fp):.3f} | NPV {tn / (tn + fn):.3f}")
    for g in np.unique(groups):                      # subgroup analysis
        m = groups == g
        tn, fp, fn, tp = confusion_matrix(y_true[m], pred[m], labels=[0, 1]).ravel()
        print(f"  {g:<12} n={m.sum():>4}  sensitivity {tp / max(tp + fn, 1):.3f}  specificity {tn / max(tn + fp, 1):.3f}")
```

## Deployment and regulation

Medical AI is typically regulated as a **medical device** (e.g. by the FDA in the United States or under the Medical Device Regulation in the EU). Hundreds of AI-enabled devices have received regulatory clearance, most in radiology. Post-deployment, performance must be **monitored** — new scanners, software updates or population changes can degrade it. Integration into clinical workflow, clear communication of uncertainty and responsibility, and avoiding **automation bias** (clinicians over-trusting the AI) are as important as the model.

## Global health perspective

AI screening for tuberculosis on chest X-rays has been recommended by the WHO for use in certain screening settings, and diabetic retinopathy screening tools have been deployed in primary care. In low-resource settings, benefits can be greatest — and so can risks if models trained on data from high-income hospitals are applied to different populations and equipment without local validation.

:::exercise
1. Train a chest X-ray classifier on a public dataset with an image-level split and a patient-level split. Compare test AUC.
2. Use Grad-CAM (next lectures) to inspect whether your model attends to lungs or to text markers and devices.
3. Draft an external-validation plan for deploying a tuberculosis screening model in a new country.
:::

:::takeaway
- Medical imaging AI spans X-ray, CT/MRI, retina, pathology, dermatology and ultrasound; U-Net and MIL are key tools.
- Small, noisy, heterogeneous data and privacy constraints shape methods.
- Shortcut learning is the central danger — validate externally, by patient, by subgroup.
- Clinical value requires prospective evaluation, regulation, monitoring and careful workflow integration.
:::

=== POST ===
slug: segment-anything-vision-foundation-models
title: "Vision Foundation Models: Segment Anything and Promptable Vision"
category: computer-vision
level: Advanced
tags: foundation models, segment anything, sam, open vocabulary, promptable
summary: Vision is following language towards general-purpose foundation models. We study the Segment Anything Model's promptable design and data engine, open-vocabulary detection, and how foundation models change vision workflows.
---
For most of deep learning's history, each vision task required its own labelled dataset and model. Foundation models change this: trained once on enormous data, they can be **adapted or prompted** for many tasks. The **Segment Anything Model (SAM)**, released by Meta AI in 2023, brought this paradigm to segmentation, and open-vocabulary detectors brought it to detection. These tools are transforming how practitioners annotate data and build vision systems.

## Promptable segmentation

SAM defines a new task: given an image and a **prompt** — a point, a box, a rough mask (and, in later versions, text or concept prompts) — output a valid segmentation mask for the indicated object. Because a single point can be ambiguous (the shirt, or the whole person?), SAM outputs **multiple masks** with predicted quality scores.

## Architecture

1. **Image encoder** — a large ViT pretrained with MAE, run **once** per image to produce an embedding. This is the expensive part.
2. **Prompt encoder** — encodes points and boxes with positional encodings, and masks with convolutions.
3. **Mask decoder** — a lightweight transformer that combines image and prompt embeddings, using two-way attention, to predict masks and their IoU scores in milliseconds.

The design separates a heavy, cacheable image embedding from a fast, interactive decoder — so a user can click repeatedly and see masks update in real time.

## The data engine

SAM's power comes from data: the **SA-1B** dataset with over **1 billion masks** on 11 million licensed, privacy-protected images. It was built with a **model-in-the-loop data engine**:

1. **Assisted-manual stage**: annotators click; SAM proposes masks; annotators correct them. The model is retrained as data grows.
2. **Semi-automatic stage**: SAM pre-fills confident masks; annotators add missing objects.
3. **Fully automatic stage**: SAM is prompted with a grid of points and generates masks, filtered by predicted quality and stability.

This virtuous cycle — better model → faster annotation → more data → better model — is a template for building datasets at scale.

## Using SAM

```python
# pip install segment-anything  (and download a checkpoint, e.g. sam_vit_b)
import numpy as np
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor

sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")
predictor = SamPredictor(sam)
image = np.array(Image.open("satellite_tile.jpg").convert("RGB"))
predictor.set_image(image)                               # heavy encoder runs once

masks, scores, _ = predictor.predict(point_coords=np.array([[420, 310]]),
                                     point_labels=np.array([1]),        # 1 = foreground click
                                     multimask_output=True)
best = masks[scores.argmax()]
print("mask area (pixels):", int(best.sum()), " predicted quality:", round(float(scores.max()), 3))
```

Later versions extended SAM to **video** (propagating masks across frames with memory) and to text/concept prompts, and efficient variants run on mobile devices.

## Open-vocabulary detection and grounding

Detectors traditionally recognise fixed class lists. **Open-vocabulary** models align visual regions with text embeddings, so you can detect categories specified at inference time:

- **OWL-ViT / OWLv2**: CLIP-style models adapted for detection with text or image queries.
- **Grounding DINO**: detects objects described by free-form phrases ("the blue water tank on the roof").
- **Grounded-SAM**: combines a grounding detector (text → boxes) with SAM (boxes → masks) for text-prompted segmentation.

## Foundation-model workflows

:::tip
A modern practical workflow for a new vision task:
1. **Try zero-shot** — CLIP for classification, an open-vocabulary detector for detection, SAM for masks. It may already be good enough.
2. **Pre-annotate** your data with foundation models, then have humans **correct** labels — often several times faster than labelling from scratch.
3. **Fine-tune** or **distil** into a smaller, specialised model (e.g. a YOLO or U-Net) that runs fast and cheaply in deployment.
4. **Evaluate** on in-domain data — foundation models can fail on specialised imagery (medical, microscopy, some satellite products) without adaptation.
:::

## Limitations

- **Class-agnostic masks**: SAM segments "things" but does not say what they are — combine it with classifiers or detectors.
- **Fine structures and domain shift**: thin structures, low-contrast medical images and unusual sensors may need fine-tuning (e.g. medical adaptations such as MedSAM).
- **Compute**: large image encoders are heavy; use efficient variants for edge devices.
- **Biases and coverage** of training data persist, as with all foundation models.

## The bigger picture

General-purpose vision backbones (DINOv2), vision–language models (CLIP, SigLIP), promptable segmenters (SAM) and multimodal large language models that "see" are converging towards general visual intelligence. For practitioners, the skill shifts from building every model from scratch to **selecting, prompting, adapting, evaluating and combining** foundation models responsibly.

:::exercise
1. Use SAM with box prompts to segment 20 objects in images from your domain and measure how long correction takes compared with manual polygon annotation.
2. Combine an open-vocabulary detector with SAM to segment "trees" in aerial images; evaluate against manual masks.
3. Fine-tune a small segmentation model on SAM-assisted labels and compare it with SAM zero-shot on held-out data.
:::

:::takeaway
- SAM performs promptable segmentation with a heavy cached image encoder and a fast prompt-conditioned mask decoder.
- Its model-in-the-loop data engine produced over a billion masks — a template for scalable annotation.
- Open-vocabulary detectors (OWL-ViT, Grounding DINO) detect objects named by text.
- Practical workflow: zero-shot → pre-annotate → fine-tune/distil → evaluate in-domain.
:::

=== POST ===
slug: explainability-for-vision-grad-cam
title: "Explaining Vision Models: Saliency Maps, Grad-CAM and Their Limits"
category: computer-vision
level: Intermediate
tags: explainability, grad-cam, saliency, interpretability, xai
summary: Which pixels made the model decide? We study gradient saliency, Grad-CAM, integrated gradients and occlusion, show how they reveal shortcuts, and discuss sanity checks that expose unreliable explanations.
---
A model classifies a chest X-ray as "pneumonia" with 94% confidence. A doctor asks: why? Did it look at the lungs — or at a hospital's text marker in the corner? **Explanation methods** for vision produce heatmaps highlighting the image regions that most influenced a prediction. They are invaluable for debugging and building appropriate trust — but they can also mislead if used naively.

## Gradient saliency

The simplest idea (Simonyan et al., 2013): compute the gradient of the class score $y_c$ with respect to the input pixels,

$$
S(\mathbf{x}) = \left|\frac{\partial y_c}{\partial\mathbf{x}}\right|
$$

(maximum over colour channels). Large values mark pixels whose small changes would most affect the score. Vanilla gradients are noisy; **SmoothGrad** averages gradients over several noisy copies of the input for cleaner maps.

## Grad-CAM

**Grad-CAM** (Selvaraju et al., 2017) is the most widely used method for CNNs. It works at the last convolutional layer, whose feature maps $A^k$ retain spatial layout while encoding high-level semantics.

1. Compute gradients of the class score with respect to each feature map.
2. **Global-average-pool** the gradients to get an importance weight per channel:
$$
\alpha_k^c = \frac{1}{Z}\sum_{i,j}\frac{\partial y_c}{\partial A_{ij}^k}
$$
3. Take a weighted combination of feature maps and keep positive evidence:
$$
L^c_{\text{Grad-CAM}} = \text{ReLU}\left(\sum_k\alpha_k^c A^k\right)
$$
4. Upsample to image size and overlay as a heatmap.

Grad-CAM is class-discriminative (different classes highlight different regions), needs no retraining and works with any CNN.

```python
import torch
import torch.nn.functional as F
from torchvision import models

model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).eval()
activations, gradients = {}, {}
layer = model.layer4
layer.register_forward_hook(lambda m, i, o: activations.__setitem__("a", o))
layer.register_full_backward_hook(lambda m, gi, go: gradients.__setitem__("g", go[0]))

def grad_cam(x, class_idx=None):
    logits = model(x)
    c = logits.argmax(1).item() if class_idx is None else class_idx
    model.zero_grad(); logits[0, c].backward()
    A, G = activations["a"], gradients["g"]              # (1, K, h, w)
    weights = G.mean(dim=(2, 3), keepdim=True)           # alpha_k
    cam = F.relu((weights * A).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
    return ((cam - cam.min()) / (cam.max() - cam.min() + 1e-8))[0, 0].detach(), c

x = torch.randn(1, 3, 224, 224)                          # replace with a normalised real image
heatmap, predicted = grad_cam(x)
print(heatmap.shape, predicted)
```

The `pytorch-grad-cam` library provides Grad-CAM and many variants (Grad-CAM++, Score-CAM, Eigen-CAM) for CNNs and ViTs.

## Integrated gradients

Plain gradients can be near zero when a feature is saturated even though it matters. **Integrated Gradients** (Sundararajan et al., 2017) integrates gradients along a path from a **baseline** $\mathbf{x}'$ (e.g. a black image) to the input:

$$
\text{IG}_i(\mathbf{x}) = (x_i - x_i')\int_0^1\frac{\partial F(\mathbf{x}' + \alpha(\mathbf{x} - \mathbf{x}'))}{\partial x_i}\,d\alpha
$$

It satisfies **completeness**: attributions sum to $F(\mathbf{x}) - F(\mathbf{x}')$. The choice of baseline matters and should be justified.

## Perturbation methods

**Occlusion**: slide a grey patch over the image and record how much the class score drops — model-agnostic and intuitive, but slow. **RISE** averages random masks weighted by the resulting scores. **LIME** fits a local interpretable model over superpixels (see the XAI lecture in the Ethics track).

## Explanations reveal shortcuts

Saliency maps have exposed many shortcut behaviours: models attending to watermarks or copyright text, to snow in "wolf" photos, to hospital markers in X-rays, or to the background rather than the object. Routinely inspect explanations for a sample of correct **and** incorrect predictions during model development.

## Explanations can mislead

:::warning
Adebayo et al. (2018), "Sanity Checks for Saliency Maps", showed that some popular methods produce **nearly identical** maps when the model's weights are randomised or when it is trained on random labels — the maps reflected image edges, not the model's reasoning. Before trusting a method:
- **Model randomisation test**: explanations should change substantially when weights are randomised.
- **Data randomisation test**: explanations of a model trained on random labels should differ from those of a real model.
- **Faithfulness checks**: deleting the most highlighted pixels should reduce the score more than deleting random pixels (deletion/insertion curves).

Grad-CAM passes these sanity checks reasonably well, but it is coarse (the resolution of the last conv layer). Heatmaps show *where*, not *why*, and a plausible-looking map is not proof of correct reasoning.
:::

## Beyond heatmaps

- **Concept-based explanations** (TCAV) test whether a human-defined concept (e.g. "striped") influences a class.
- **Prototype networks** explain by comparing parts of the input with learned prototypical parts ("this looks like that").
- **Counterfactual explanations** show a minimally changed image that would change the decision.
- **Mechanistic interpretability** studies individual neurons and circuits inside networks.

:::exercise
1. Apply Grad-CAM to a pretrained classifier on ten images, including misclassified ones. Describe what you learn.
2. Perform the model-randomisation sanity check: compare Grad-CAM and vanilla gradient maps before and after randomising the last block's weights.
3. Compute a deletion curve for Grad-CAM and for random ordering on the same image.
:::

:::takeaway
- Gradient saliency, SmoothGrad, Grad-CAM, integrated gradients and occlusion attribute predictions to image regions.
- Grad-CAM weights last-layer feature maps by pooled gradients — class-discriminative and easy to use.
- Explanations expose shortcuts and dataset artefacts.
- Always sanity-check explanation methods; heatmaps show where, not why.
:::

=== POST ===
slug: adversarial-examples-robustness
title: "Adversarial Examples: Fooling Neural Networks and Defending Them"
category: computer-vision
level: Advanced
tags: adversarial examples, robustness, fgsm, pgd, adversarial training, security
summary: Imperceptible perturbations can make a network confidently wrong. We derive FGSM and PGD attacks, explain why adversarial examples exist, cover physical and black-box attacks, and evaluate defences including adversarial training.
---
In 2013 Szegedy and colleagues discovered something unsettling: adding a tiny, carefully crafted perturbation — invisible to humans — to an image could make a state-of-the-art network misclassify it with high confidence. A panda becomes a "gibbon"; a stop sign becomes a "speed limit". These **adversarial examples** reveal that networks do not see the world the way we do, and they raise security concerns for any system where an attacker might manipulate inputs.

## Threat model

An adversarial example $\mathbf{x}' = \mathbf{x} + \boldsymbol{\delta}$ changes the model's prediction while the perturbation is small under some norm:

$$
\|\boldsymbol{\delta}\|_p \le \epsilon, \qquad f(\mathbf{x} + \boldsymbol{\delta}) \ne y
$$

Commonly $\ell_\infty$ with $\epsilon = 8/255$ (each pixel changes by at most about 3%), or $\ell_2$. Attacks can be **untargeted** (any wrong class) or **targeted** (a chosen class).

Knowledge levels:

- **White-box** — the attacker knows the model and its gradients.
- **Black-box** — only queries (or nothing at all) are available.

## FGSM: the fast gradient sign method

Goodfellow, Shlens and Szegedy (2015) took one step in the direction that increases the loss the most under an $\ell_\infty$ constraint:

$$
\mathbf{x}' = \mathbf{x} + \epsilon\cdot\text{sign}\big(\nabla_{\mathbf{x}}\mathcal{L}(f(\mathbf{x}), y)\big)
$$

## PGD: projected gradient descent

Madry et al. (2018) iterate smaller steps and **project** back into the allowed ball after each:

$$
\mathbf{x}^{t+1} = \Pi_{\mathcal{B}_\epsilon(\mathbf{x})}\Big(\mathbf{x}^t + \alpha\cdot\text{sign}\big(\nabla_{\mathbf{x}}\mathcal{L}(f(\mathbf{x}^t), y)\big)\Big)
$$

starting from a random point in the ball. PGD is a strong, standard first-order attack.

```python
import torch
import torch.nn.functional as F

def fgsm(model, x, y, eps=8 / 255):
    x = x.clone().requires_grad_(True)
    F.cross_entropy(model(x), y).backward()
    return (x + eps * x.grad.sign()).clamp(0, 1).detach()

def pgd(model, x, y, eps=8 / 255, alpha=2 / 255, steps=10):
    x_adv = (x + torch.empty_like(x).uniform_(-eps, eps)).clamp(0, 1)
    for _ in range(steps):
        x_adv.requires_grad_(True)
        grad, = torch.autograd.grad(F.cross_entropy(model(x_adv), y), x_adv)
        x_adv = x_adv.detach() + alpha * grad.sign()
        x_adv = torch.min(torch.max(x_adv, x - eps), x + eps).clamp(0, 1)   # project to the eps-ball
    return x_adv.detach()

# robust_acc = (model(pgd(model, x, y)).argmax(1) == y).float().mean()
```

(Here the model is assumed to include its own input normalisation, so attacks operate in $[0, 1]$ pixel space.) On an undefended CIFAR-10 classifier, PGD with $\epsilon = 8/255$ typically drives accuracy close to **zero**.

## Why do adversarial examples exist?

- **Linearity hypothesis** (Goodfellow et al.): in high dimensions, many tiny coordinated changes add up. For a linear score $\mathbf{w}^\top\mathbf{x}$, a perturbation $\epsilon\,\text{sign}(\mathbf{w})$ changes the score by $\epsilon\|\mathbf{w}\|_1$, which grows with dimension — even though each pixel barely changes.
- **Non-robust features** (Ilyas et al., 2019): datasets contain genuinely predictive but imperceptible patterns; standard training exploits them, and adversaries flip them. Adversarial vulnerability is partly a property of the data, not only of the model.
- **Geometry**: decision boundaries lie close to most data points in some directions of high-dimensional space.

## Beyond the digital lab

- **Transferability**: adversarial examples crafted on one model often fool others, enabling **black-box transfer attacks** via a surrogate model.
- **Query-based attacks** estimate gradients from outputs alone.
- **Physical attacks**: printed adversarial patches, stickers on stop signs, or adversarial eyeglass frames have fooled classifiers and detectors in real-world conditions.
- **Universal perturbations**: a single perturbation that fools a model on most images.

## Defences

Many proposed defences were later broken. Athalye, Carlini and Wagner (2018) showed that defences relying on **obfuscated gradients** (non-differentiable preprocessing, randomness) gave a false sense of security and were defeated by adaptive attacks.

The most reliable empirical defence is **adversarial training** — train on adversarial examples generated on the fly, solving the min–max problem:

$$
\min_{\boldsymbol{\theta}}\;\mathbb{E}_{(\mathbf{x}, y)}\left[\max_{\|\boldsymbol{\delta}\| \le \epsilon}\mathcal{L}\big(f_{\boldsymbol{\theta}}(\mathbf{x} + \boldsymbol{\delta}), y\big)\right]
$$

It works, but it multiplies training cost (several attack steps per batch), lowers clean accuracy, and robustness typically holds only for the threat model used in training. Variants such as TRADES balance clean and robust accuracy. **Certified defences** (e.g. randomised smoothing) provide provable guarantees within a radius, at a cost in accuracy.

:::note
**Evaluate robustness properly.** Use strong, adaptive attacks (PGD with many steps and restarts; standard suites such as AutoAttack), report the threat model ($\ell_p$ norm and $\epsilon$), and check for gradient masking (e.g. if black-box attacks succeed more than white-box ones, something is wrong). The RobustBench leaderboard tracks standardised evaluations.
:::

## Why it matters beyond security

Adversarial robustness connects to general reliability: robust models often have more interpretable gradients and features aligned with human perception. And adversarial thinking extends to other modalities — adversarial audio, text perturbations, and **prompt injection and jailbreaks** against language models, which we meet in the LLM and Ethics tracks.

:::exercise
1. Attack a pretrained CIFAR-10 model with FGSM and PGD for $\epsilon \in \{1, 2, 4, 8\}/255$ and plot accuracy against $\epsilon$.
2. Craft adversarial examples on one model and test their transferability to a different architecture.
3. Adversarially train a small CNN with PGD for a few epochs; compare clean and robust accuracy with the standard model.
:::

:::takeaway
- Tiny, crafted perturbations within an $\epsilon$-ball can make networks confidently wrong.
- FGSM takes one signed-gradient step; PGD iterates with projection and is the standard strong attack.
- Causes include high-dimensional linearity and non-robust but predictive features; attacks transfer and work physically.
- Adversarial training is the most reliable defence but is costly; evaluate with strong adaptive attacks.
:::
