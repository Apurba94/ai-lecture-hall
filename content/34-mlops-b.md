=== POST ===
slug: edge-ai-tinyml
title: "Edge AI and TinyML: Running Models on Phones and Microcontrollers"
category: mlops
level: Intermediate
tags: edge ai, tinyml, on-device, tflite, onnx, offline
summary: Running models on-device brings privacy, offline operation, low latency and low cost. We cover the edge hardware spectrum, the optimisation pipeline, TensorFlow Lite, ONNX Runtime and TinyML on microcontrollers, and field-deployment lessons.
---
Not every AI system can rely on a fast internet connection and a cloud server. A health worker in a remote area needs a diagnostic aid that works offline; a sensor in a field runs for months on a battery; a user's photos should not leave their phone. **Edge AI** runs inference directly on devices — phones, embedded computers, cameras, microcontrollers. **TinyML** pushes this to tiny microcontrollers with kilobytes of memory and milliwatts of power.

## Why run on the edge?

| Benefit | Explanation |
|---|---|
| Privacy | Raw data (images, audio, health signals) stays on the device |
| Offline operation | Works without connectivity — essential in many rural and crisis settings |
| Latency | No network round-trip; real-time response |
| Cost | No per-request server cost; scales with devices |
| Bandwidth | Send only results or alerts, not raw video |
| Energy | Local inference can use less energy than transmitting data |

Trade-offs: limited compute and memory, heterogeneous hardware, harder updates and monitoring, and model theft risk (the model file is on the device).

## The hardware spectrum

| Device class | Memory | Typical models |
|---|---|---|
| Smartphones (CPU/GPU/NPU) | GBs | MobileNet/EfficientNet, small LLMs (quantised), on-device speech |
| Single-board computers / edge accelerators (Raspberry Pi, Jetson, Coral) | 1–16 GB | Detection, segmentation, small transformers |
| Microcontrollers (Arm Cortex-M, ESP32) | 32 KB – 1 MB RAM | Keyword spotting, anomaly detection, tiny vision, gesture recognition |

Many phones include **neural processing units (NPUs)** that accelerate quantised models dramatically.

## The optimisation pipeline

1. **Choose an efficient architecture** (MobileNet, EfficientNet-Lite, small transformers, or tiny CNNs for microcontrollers).
2. **Train** (often with transfer learning).
3. **Compress**: quantisation (INT8 is standard; sometimes lower), pruning, knowledge distillation.
4. **Convert** to an on-device format: TensorFlow Lite / LiteRT, ONNX, Core ML, ExecuTorch.
5. **Benchmark on the target device**: latency, memory, energy, accuracy on realistic data.
6. **Deploy and update** through app updates or over-the-air model delivery, with versioning.

## TensorFlow Lite with full-integer quantisation

```python
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("crop_disease.keras")      # a trained Keras model

def representative_data():
    for img in np.load("calibration_images.npy")[:200]:       # ~100–500 real samples
        yield [img[None].astype(np.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data         # calibrate activation ranges
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
open("crop_disease_int8.tflite", "wb").write(converter.convert())

interpreter = tf.lite.Interpreter(model_path="crop_disease_int8.tflite")
interpreter.allocate_tensors()
print(interpreter.get_input_details()[0]["shape"], interpreter.get_input_details()[0]["dtype"])
```

Always compare the quantised model's accuracy with the float model on held-out data — especially per class.

## ONNX Runtime

**ONNX** is an open format supported by many frameworks. Export from PyTorch and run with ONNX Runtime on CPUs, GPUs and mobile, with execution providers for NPUs:

```python
import torch, onnxruntime as ort, numpy as np
from torchvision.models import mobilenet_v3_small

m = mobilenet_v3_small(weights="DEFAULT").eval()
torch.onnx.export(m, torch.randn(1, 3, 224, 224), "mnv3.onnx", input_names=["image"],
                  output_names=["logits"], dynamic_axes={"image": {0: "batch"}}, opset_version=17)
sess = ort.InferenceSession("mnv3.onnx", providers=["CPUExecutionProvider"])
print(sess.run(None, {"image": np.random.rand(1, 3, 224, 224).astype(np.float32)})[0].shape)
```

## TinyML on microcontrollers

TinyML models fit in tens to hundreds of kilobytes. Classic applications: **keyword spotting** ("Hey device"), vibration-based **predictive maintenance**, **anomaly detection** on sensor data, simple person detection, and wildlife or gunshot sound detection for conservation.

Workflow: train a tiny model (often a small CNN on spectrograms or sensor windows), quantise to INT8, convert with **TensorFlow Lite for Microcontrollers** (or tools like Edge Impulse), and compile into firmware as a C array. Constraints drive design: no operating system, no dynamic memory allocation, fixed "tensor arena" memory, and careful power management (often running inference only when a cheap trigger fires).

## Field-deployment lessons

:::tip
- **Collect evaluation data on the real devices in real conditions** — lighting, dust, noise, camera quality and user behaviour differ from the lab.
- **Design the UX for uncertainty**: ask users to retake unclear photos; show confidence carefully; provide a "not sure" outcome.
- **Plan updates and monitoring**: log anonymised, consented summary statistics when devices connect, so drift can be detected.
- **Battery and heat** matter: continuous inference drains phones quickly.
- **Support low-end devices** — many users do not have flagship phones.
:::

:::exercise
1. Convert a small image classifier to TFLite in float32, dynamic-range INT8 and full INT8; compare size, CPU latency and accuracy.
2. Export a PyTorch model to ONNX and measure latency on your laptop with ONNX Runtime versus PyTorch.
3. Design (on paper) a TinyML system for detecting pump failures from vibration: data collection, model, memory budget, power strategy and alerting.
:::

:::takeaway
- Edge AI brings privacy, offline operation, low latency and low cost, with tight compute and memory limits.
- The pipeline: efficient architecture → compression (INT8 quantisation, pruning, distillation) → conversion → on-device benchmarking.
- TFLite, ONNX Runtime, Core ML and ExecuTorch target phones; TFLite Micro targets microcontrollers.
- Validate on real devices and conditions, design for uncertainty, and plan updates and monitoring.
:::

=== POST ===
slug: gpus-and-ml-hardware
title: "GPUs and Hardware for Machine Learning"
category: mlops
level: Intermediate
tags: gpu, hardware, tpu, memory, compute, cloud
summary: Understanding hardware helps you train faster and cheaper. We explain why GPUs suit deep learning, the roles of memory capacity and bandwidth, precision and tensor cores, estimating requirements, and choosing between local, cloud and free resources.
---
Deep learning became practical because of **GPUs** — processors originally built for video games. Knowing what your hardware can and cannot do helps you choose batch sizes, precision and model sizes, estimate training time and cost, and diagnose slow training. This lecture gives students the hardware literacy needed for practical ML.

## Why GPUs?

A **CPU** has a few powerful cores optimised for sequential, branching logic and low latency. A **GPU** has thousands of simpler cores optimised for **parallel arithmetic** on large arrays. Neural network training is dominated by matrix multiplications and convolutions — massively parallel operations — so GPUs can be orders of magnitude faster.

Modern NVIDIA GPUs include **Tensor Cores**, specialised units for mixed-precision matrix multiply–accumulate, delivering far higher throughput in FP16/BF16/FP8/INT8 than in FP32. Other accelerators include Google **TPUs** (systolic arrays for matrix operations), AMD GPUs (ROCm software stack), Apple silicon (unified memory, MPS/MLX), and mobile **NPUs**.

## The two numbers that matter most

### 1. Memory capacity (GB of VRAM)
Determines what **fits**: model weights, gradients, optimiser states, activations and batch. Out-of-memory errors are the most common limit students hit.

Rough training memory for a model with $P$ parameters using Adam in mixed precision: about **16–18 bytes per parameter** (weights, gradients, FP32 master weights and two Adam moments) **plus activations**. Inference in FP16 needs about **2 bytes per parameter** plus activations/KV cache.

### 2. Memory bandwidth (GB/s) and compute (FLOP/s)
Determine **speed**. An operation is:

- **compute-bound** if it performs many operations per byte moved (large matrix multiplications);
- **memory-bound** if it moves a lot of data per operation (element-wise ops, normalisation, LLM decoding).

The ratio of FLOPs to bytes (**arithmetic intensity**) tells which applies — the "roofline model". This explains why LLM token generation depends on bandwidth, while training large batches depends on compute.

## Estimating training time

For transformers, training compute is about $6ND$ FLOPs (parameters × tokens). Divide by achievable throughput (peak × utilisation, typically 30–60%):

```python
def training_days(params, tokens, n_gpus, peak_tflops, mfu=0.4):
    flops = 6 * params * tokens
    per_second = n_gpus * peak_tflops * 1e12 * mfu
    return flops / per_second / 86_400

# A 125M-parameter model on 2.5B tokens with one GPU at 100 TFLOP/s (bf16) peak
print(round(training_days(125e6, 2.5e9, 1, 100), 2), "days")
# A 7B model on 1T tokens with 64 GPUs at 800 TFLOP/s peak
print(round(training_days(7e9, 1e12, 64, 800), 1), "days")
```

## Practical tips to use hardware well

1. **Keep the GPU busy**: profile data loading; use multiple DataLoader workers, pinned memory and prefetching. A GPU at 30% utilisation is usually waiting for data.
2. **Use mixed precision** (BF16/FP16) — often 2–3× faster and half the activation memory.
3. **Maximise batch size** within memory; use **gradient accumulation** for larger effective batches.
4. **Activation checkpointing** trades compute for memory.
5. **Compile** models (`torch.compile`) and use fused kernels (FlashAttention).
6. **Avoid CPU–GPU synchronisation** in the training loop (`.item()`, printing tensors each step).
7. **Monitor**: `nvidia-smi` for utilisation and memory; the PyTorch profiler for bottlenecks.

```bash
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv -l 5
```

## Choosing hardware as a student or small organisation

| Option | Pros | Cons |
|---|---|---|
| Laptop CPU / Apple silicon | Free, always available | Slow for deep learning; fine for classical ML and small models |
| Free cloud notebooks (e.g. Colab, Kaggle) | Free GPUs for learning | Session limits, variable availability |
| Consumer GPU workstation | Full control; cost-effective for continuous use | Upfront cost, power, maintenance |
| Cloud GPU instances | Scale on demand; latest hardware | Can become expensive; must manage data security and shut down idle machines |
| University/HPC clusters | Powerful, often free for students | Queues, scheduler learning curve |

:::tip
Before buying or renting hardware, ask: can a **smaller model**, **transfer learning**, **quantisation** or **LoRA** achieve the goal? Most practical projects do not need frontier-scale compute. And always set **budget alerts** and auto-shutdown on cloud instances — forgotten GPU machines are a classic, expensive mistake.
:::

## Energy and environmental cost

Training and serving large models consume significant electricity; the carbon footprint depends heavily on the energy mix of the data centre. Report compute and energy where possible (tools like CodeCarbon estimate emissions), choose efficient models, and avoid unnecessary large-scale runs — themes discussed further in the Ethics track.

:::exercise
1. Estimate the training memory for 350M-, 1.3B- and 7B-parameter models with AdamW in mixed precision, and identify which fit on a 24 GB GPU.
2. Profile a training loop with `torch.profiler` and find the largest time consumer; improve it.
3. Measure throughput (samples/second) of the same model in FP32 and BF16/FP16 on an available GPU.
:::

:::takeaway
- GPUs excel at parallel matrix arithmetic; Tensor Cores accelerate low-precision math.
- Memory capacity limits what fits; bandwidth and compute limit speed (roofline: compute- vs memory-bound).
- Estimate training compute with $6ND$ and realistic utilisation; keep GPUs fed and use mixed precision.
- Choose hardware pragmatically; control cloud costs and consider energy use.
:::

=== POST ===
slug: data-labeling-and-annotation
title: "Data Labelling and Annotation: Building High-Quality Datasets"
category: mlops
level: Beginner
tags: data labeling, annotation, inter-annotator agreement, label noise, data-centric ai
summary: Labels are the foundation of supervised learning, yet labelling is often rushed. We cover annotation guidelines, workflows and tools, measuring agreement, handling label noise, model-assisted labelling, and fair treatment of annotators.
---
Andrew Ng's "data-centric AI" movement argues that, for many applications, improving the **data** — especially labels — yields bigger gains than tweaking models. Yet labelling is often treated as a cheap afterthought. Studies have found label errors even in famous benchmark test sets (Northcutt et al., 2021, estimated an average of several percent across ten popular datasets), enough to change which model appears best. This lecture covers how to create labels you can trust.

## Step 1: define the task precisely

Write **annotation guidelines** before labelling begins:

- Clear definitions of every label, with **positive and negative examples**.
- Rules for **ambiguous and edge cases** ("if a message mentions both food and shelter, label both" or "label the primary need").
- What to do when unsure (an "unclear" option, or escalation).
- For spans/boxes/masks: exact boundary rules (include the title "Dr." in a PERSON entity? box around visible pixels only or the whole occluded object?).

Pilot the guidelines on a small batch, discuss disagreements, and revise. Good guidelines evolve through several iterations.

## Step 2: choose annotators and tools

- **Domain experts** (clinicians, protection officers, agronomists) for specialised judgements — expensive but essential for high-stakes labels.
- **Trained annotators** with guidelines for general tasks.
- **Crowdsourcing** for simple, high-volume tasks — with quality control.
- **Native speakers** for language tasks — critical for multilingual data.

Tools: Label Studio, CVAT (vision), doccano (text), Prodigy, Argilla, cloud labelling services. Choose tools that support your data types, quality workflows and **data security** requirements (self-hosting for sensitive data).

## Step 3: measure agreement

Have multiple annotators label an overlapping subset. Raw agreement overstates reliability because some agreement happens by chance. **Cohen's kappa** corrects for this:

$$
\kappa = \frac{p_o - p_e}{1 - p_e}
$$

where $p_o$ is observed agreement and $p_e$ is agreement expected by chance. For more than two annotators, use **Fleiss' kappa** or **Krippendorff's alpha**. Rough interpretation: above 0.8 strong, 0.6–0.8 substantial, below 0.4 weak — but acceptable levels depend on the task.

```python
from sklearn.metrics import cohen_kappa_score, confusion_matrix

a1 = ["food", "shelter", "health", "food", "water", "health", "food", "shelter", "water", "food"]
a2 = ["food", "shelter", "health", "water", "water", "health", "food", "health", "water", "food"]
print("raw agreement:", sum(x == y for x, y in zip(a1, a2)) / len(a1))
print("Cohen's kappa:", round(cohen_kappa_score(a1, a2), 3))
labels = ["food", "water", "shelter", "health"]
print(confusion_matrix(a1, a2, labels=labels))        # where do annotators disagree?
```

Low agreement signals unclear guidelines or a genuinely subjective task — no model can reliably exceed the consistency of its labels.

## Step 4: quality control

- **Gold questions**: items with known answers mixed into tasks to monitor annotator accuracy.
- **Multiple annotations + aggregation**: majority vote, or statistical models (Dawid–Skene) that estimate each annotator's reliability.
- **Adjudication**: an expert resolves disagreements.
- **Regular feedback** to annotators and guideline updates.
- **Audit samples** throughout, not only at the start.

## Handling label noise

Some noise is unavoidable. Strategies:

- **Find likely errors**: train a model with cross-validation and flag examples where it confidently disagrees with the label (the idea behind **confident learning** and the cleanlab library); send them for review.
- **Robust training**: label smoothing, noise-robust losses, co-teaching.
- **Keep disagreement as information**: for subjective tasks (e.g. offensiveness), store the distribution of annotator labels rather than forcing a single "truth" — annotators' perspectives may legitimately differ.

## Model-assisted labelling and active learning

- **Pre-annotation**: a model (or an LLM, or SAM for masks) proposes labels; humans correct them — often several times faster. Beware **anchoring bias**: annotators may accept wrong suggestions. Audit with unassisted gold items.
- **Active learning**: prioritise the most informative items for labelling (see the active learning lecture).
- **Weak supervision**: combine heuristic labelling functions statistically (e.g. Snorkel) to create noisy labels at scale.

## Annotators are people

:::warning
Annotation work — especially content moderation and labelling of violent or disturbing material — can harm annotators' wellbeing. Investigations have documented low pay and psychological harm among outsourced data workers. Responsible practice includes fair pay, reasonable targets, the right to skip distressing items, psychological support, clear information about the work, and credit for contributions. When labelling sensitive personal data, also ensure annotators are trained in confidentiality and data protection.
:::

## Document your dataset

Record who labelled the data, how, with which guidelines, agreement statistics, known limitations and intended uses — a **datasheet for datasets** (Gebru et al., 2021; see the documentation lecture).

:::exercise
1. Write annotation guidelines for classifying community feedback into five categories; have two classmates label 50 messages and compute Cohen's kappa. Revise the guidelines and repeat.
2. Use cross-validated predictions to flag 20 likely mislabelled examples in a dataset you use; review them manually. How many are real errors?
3. Compare annotation speed and accuracy with and without model pre-annotation on 40 items.
:::

:::takeaway
- Precise, iterated annotation guidelines with examples and edge cases are the foundation of good labels.
- Measure inter-annotator agreement with chance-corrected statistics (Cohen's kappa, Krippendorff's alpha).
- Use gold items, aggregation, adjudication, confident learning and robust training to manage noise.
- Model-assisted labelling speeds work but risks anchoring; treat annotators fairly and document the dataset.
:::

=== POST ===
slug: ab-testing-ml-models
title: "A/B Testing and Online Evaluation of ML Models"
category: mlops
level: Intermediate
tags: a/b testing, online evaluation, experimentation, statistics, causal inference
summary: Offline metrics do not guarantee real-world impact. Online experiments measure what a model actually changes. We design randomised A/B tests for ML, compute sample sizes, avoid common pitfalls, and discuss ethics of experimenting with people.
---
A new recommendation model improves offline accuracy by 3%. Will users actually find what they need faster? A new triage model has better AUC. Will caseworkers resolve urgent cases sooner? Offline metrics are **proxies**; the real question is causal: **what happens to outcomes when we deploy this model?** Randomised online experiments — **A/B tests** — answer it with the same logic as clinical trials.

## Why offline evaluation is not enough

- Offline data reflects the **old** system's behaviour (recommendations shape clicks), creating feedback loops.
- Proxy metrics (accuracy, AUC) may not map to outcomes (user satisfaction, time saved, harm avoided).
- Real-world effects include **user adaptation**, latency, UI interaction and workflow changes.

## The basic design

1. **Randomise** units (users, sessions, households, clinics) into **control** (current model, A) and **treatment** (new model, B).
2. Run both concurrently for a pre-specified period.
3. Compare a pre-registered **primary metric** between groups.

Randomisation makes the groups comparable in expectation, so a significant difference can be attributed to the model.

## Choosing metrics

- **Primary (decision) metric**: the outcome you care about — e.g. proportion of urgent cases handled within 48 hours.
- **Guardrail metrics**: things that must not get worse — latency, error rates, complaint rates, fairness across groups, cost.
- **Diagnostic metrics**: help explain results (clicks, model confidence distributions).

Specify metrics, analysis method and stopping rules **before** starting (pre-registration) to avoid cherry-picking.

## Sample size and power

To detect a difference between two proportions $p_A$ and $p_B$ with significance level $\alpha$ and power $1 - \beta$, the required sample size per group is approximately

$$
n \approx \frac{\left(z_{1-\alpha/2} + z_{1-\beta}\right)^2\,\big[p_A(1 - p_A) + p_B(1 - p_B)\big]}{(p_B - p_A)^2}
$$

Small effects require large samples: halving the detectable effect quadruples $n$.

```python
from scipy.stats import norm
import numpy as np
from statsmodels.stats.proportion import proportions_ztest

def sample_size(p_a, p_b, alpha=0.05, power=0.8):
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    return int(np.ceil(z**2 * (p_a * (1 - p_a) + p_b * (1 - p_b)) / (p_b - p_a) ** 2))

print("per-group n to detect 10% -> 11%:", sample_size(0.10, 0.11))
print("per-group n to detect 10% -> 12%:", sample_size(0.10, 0.12))

# Analysis after the experiment
successes = np.array([1_130, 1_020])          # treatment, control
totals = np.array([10_000, 10_000])
stat, p = proportions_ztest(successes, totals)
diff = successes[0] / totals[0] - successes[1] / totals[1]
se = np.sqrt(sum(s / t * (1 - s / t) / t for s, t in zip(successes, totals)))
print(f"difference = {diff:.4f}, 95% CI [{diff - 1.96 * se:.4f}, {diff + 1.96 * se:.4f}], p = {p:.4f}")
```

## Common pitfalls

1. **Peeking**: checking results repeatedly and stopping when $p < 0.05$ inflates false positives dramatically. Use a fixed horizon, or **sequential testing** methods designed for continuous monitoring (e.g. alpha-spending, always-valid confidence sequences).
2. **Multiple comparisons**: testing many metrics or segments guarantees some false "wins"; correct for it or pre-specify.
3. **Wrong randomisation unit**: randomising by request when users see both variants causes contamination; analyse at the level you randomise.
4. **Interference / network effects**: treating one unit affects others (e.g. shared resources, marketplaces, social networks). Use **cluster randomisation** (by region, clinic or camp) when needed.
5. **Novelty and learning effects**: users react differently at first; run long enough.
6. **Sample ratio mismatch**: if groups are not the expected sizes, something is broken in assignment or logging — investigate before trusting results.
7. **Simpson's paradox and heterogeneity**: an overall improvement can hide harm to a subgroup — examine pre-specified segments.

## Alternatives and complements

- **Interleaving** (for ranking/search): mix results from both rankers in one list and see which gets more engagement — very sensitive with small samples.
- **Multi-armed bandits**: shift traffic adaptively towards better variants (less regret, harder inference).
- **Quasi-experimental methods** when randomisation is impossible: difference-in-differences, regression discontinuity, synthetic controls.
- **Off-policy evaluation** from logs before any live test.

## Ethics of experimentation

:::warning
Experimenting on people requires care, especially when outcomes involve health, safety, income or access to services:
- Is there genuine uncertainty about which option is better (equipoise)?
- Could either arm cause harm? Set guardrails and stopping rules for harm.
- Do people need to be informed or to consent? Follow applicable ethics review and legal requirements.
- Are vulnerable groups protected from bearing disproportionate risk?
Many organisations route such experiments through ethics review boards.
:::

:::exercise
1. Compute the required sample size to detect an increase in a completion rate from 30% to 32% with 80% power.
2. Simulate 1,000 A/A tests (no true difference) where you "peek" daily and stop at the first $p < 0.05$. What fraction falsely declare a winner?
3. Design an A/B test for a new message-routing model in a support centre: unit of randomisation, primary and guardrail metrics, duration and ethical safeguards.
:::

:::takeaway
- Offline metrics are proxies; randomised online experiments measure causal impact.
- Pre-register primary and guardrail metrics, randomisation unit and analysis; compute sample size for adequate power.
- Avoid peeking, multiple comparisons, contamination, interference and sample ratio mismatch; check subgroups.
- Use interleaving, bandits or quasi-experiments where appropriate, and apply ethical safeguards when experimenting on people.
:::

=== POST ===
slug: ml-project-structure-clean-code
title: "From Notebook to Production Code: Structuring Clean ML Projects"
category: mlops
level: Beginner
tags: software engineering, project structure, clean code, testing, notebooks
summary: Notebooks are great for exploration and terrible for production. We cover a clean project layout, configuration, modular code, typing and testing, logging, packaging, and a workflow for moving from exploration to maintainable software.
---
Jupyter notebooks are wonderful for exploring data and prototyping models: immediate feedback, inline plots, narrative. But notebooks that grow into production systems become fragile: hidden state from out-of-order cell execution, copy-pasted code, no tests, hard-to-review diffs. Most of your career's ML work will be maintained by others — or by you in a year, having forgotten everything. Writing clean, well-structured ML code is a professional skill that makes your work reproducible, reviewable and reliable.

## Notebooks: use them for what they are good at

- ✅ Exploration, visualisation, quick experiments, teaching, reports.
- ❌ Core logic that must be reused, tested or scheduled.

A good workflow: explore in a notebook → move stable functions into a Python package → import them back into notebooks → run training and inference as scripts or pipelines. Restart the kernel and "run all" regularly to catch hidden-state bugs. Tools like `nbstripout` remove outputs before committing; `jupytext` pairs notebooks with plain `.py` files for readable diffs.

## A clean project layout

```text
aid-triage/
├── pyproject.toml            # package metadata + dependencies + tool config
├── README.md                 # purpose, setup, how to train/evaluate/serve
├── configs/
│   ├── train.yaml
│   └── gates.yaml
├── src/aid_triage/
│   ├── __init__.py
│   ├── data.py               # loading, validation
│   ├── features.py           # feature engineering (shared by train and serve)
│   ├── model.py              # model definition / pipeline construction
│   ├── train.py              # CLI entry point
│   ├── evaluate.py           # metrics, slices, reports
│   └── serve.py              # API
├── notebooks/                # exploration only; import from src
├── tests/
│   ├── test_features.py
│   └── test_model.py
└── .github/workflows/ci.yml
```

Templates such as Cookiecutter Data Science provide similar starting structures.

## Principles of clean ML code

### 1. Small, pure functions
Functions that take inputs and return outputs without hidden side effects are easy to test and reuse.

```python
# src/aid_triage/features.py
from __future__ import annotations
import pandas as pd

def dependency_ratio(members: pd.Series, working_age: pd.Series) -> pd.Series:
    """Number of dependants per working-age member; 0 working-age members -> NaN."""
    dependants = members - working_age
    return dependants / working_age.where(working_age > 0)

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["dependency_ratio"] = dependency_ratio(df["members"], df["working_age_members"])
    out["income_per_member"] = df["monthly_income"] / df["members"].clip(lower=1)
    out["has_child_under_5"] = (df["children_under_5"] > 0).astype(int)
    return out
```

### 2. Configuration, not constants
Put paths, hyperparameters, thresholds and seeds in config files (YAML/TOML) or typed config objects — not scattered through code.

```python
from dataclasses import dataclass
import yaml

@dataclass(frozen=True)
class TrainConfig:
    train_path: str
    valid_path: str
    target: str
    learning_rate: float = 0.05
    max_leaf_nodes: int = 31
    seed: int = 42

cfg = TrainConfig(**yaml.safe_load(open("configs/train.yaml")))
```

### 3. Type hints and docstrings
They document intent, enable editor support and catch bugs with type checkers (mypy, pyright).

### 4. Tests
Test feature functions with small, hand-computed examples, including edge cases:

```python
# tests/test_features.py
import numpy as np, pandas as pd
from aid_triage.features import dependency_ratio

def test_dependency_ratio_basic():
    assert dependency_ratio(pd.Series([5]), pd.Series([2])).iloc[0] == 1.5

def test_dependency_ratio_no_workers_is_nan():
    assert np.isnan(dependency_ratio(pd.Series([3]), pd.Series([0])).iloc[0])
```

### 5. Logging instead of print
Use the `logging` module with levels and timestamps; log configuration, data versions and metrics.

### 6. Formatting and linting
Automate style with tools such as `ruff` (linting and formatting) and pre-commit hooks — reviews can then focus on substance.

### 7. Command-line entry points
Make training runnable as `python -m aid_triage.train --config configs/train.yaml`, so it can be scheduled, containerised and run in CI.

## Code review and collaboration

- Use Git branches and pull requests; review code **and** results (metrics, plots).
- Keep pull requests small and focused.
- Write a README that lets a new person set up and run the project in under an hour.
- Record decisions (why this metric, why this threshold) in the repository — future readers need the reasoning, not just the code.

:::tip
A useful test of project quality: could a classmate clone your repository on a fresh machine, follow the README, reproduce your main result, and run the tests — without messaging you? If not, fix that first.
:::

:::exercise
1. Refactor one of your notebooks into a package with at least three modules, a config file and a CLI entry point.
2. Write five unit tests for your feature functions, including edge cases, and run them with pytest.
3. Add ruff and a pre-commit hook to your project, and ask a classmate to follow your README from scratch.
:::

:::takeaway
- Use notebooks for exploration; move reusable logic into a tested package run by scripts or pipelines.
- Organise projects clearly: src package, configs, tests, notebooks, CI, README.
- Write small pure functions with type hints, externalised configuration, logging and automated formatting.
- Collaborate through Git, reviews and documentation that lets others reproduce your work.
:::

=== POST ===
slug: model-cards-and-documentation
title: "Model Cards, Datasheets and Responsible Documentation"
category: mlops
level: Beginner
tags: model cards, datasheets, documentation, transparency, governance
summary: Documentation is how models and datasets are understood, audited and used responsibly. We cover model cards, datasheets for datasets, system cards, what to include, and how documentation supports accountability and regulation.
---
A model file tells you nothing about what it is for, how well it works, for whom it fails, or what data made it. Without documentation, models get reused in contexts they were never designed for — with predictable harm. In 2018–2019, researchers proposed simple, structured documentation practices: **model cards** (Mitchell et al., 2019) and **datasheets for datasets** (Gebru et al., 2018/2021). They have since become widely adopted norms — Hugging Face model pages use model cards — and they help meet the transparency obligations of emerging AI regulations.

## Model cards

A model card is a short document accompanying a trained model. Recommended sections (adapted from Mitchell et al.):

1. **Model details**: developers, date, version, architecture, training approach, licence, contact.
2. **Intended use**: primary intended uses and users; **out-of-scope uses**.
3. **Factors**: relevant groups (demographics, languages, regions), instruments (devices, sensors) and environments that may affect performance.
4. **Metrics**: which metrics, why, decision thresholds, and how uncertainty is measured.
5. **Evaluation data**: datasets, why chosen, preprocessing.
6. **Training data**: sources and characteristics (or reference to the datasheet).
7. **Quantitative analyses**: results **disaggregated** by relevant factors and their intersections.
8. **Ethical considerations**: sensitive data, risks, harms, mitigations.
9. **Caveats and recommendations**: known limitations, conditions for safe use, monitoring advice.

The most important innovation is **disaggregated evaluation**: reporting performance for different groups, not just an overall number.

## A model card template

```markdown
# Model Card: Household Priority Classifier v2.1

## Model details
- Developed by: Data & Analytics Unit, (organisation). Contact: data-team@example.org
- Version 2.1, trained 2025-09-10. Gradient-boosted trees (LightGBM). Licence: internal use.

## Intended use
- Supports caseworkers in **ordering** follow-up visits. Human caseworkers make all decisions.
- Out of scope: determining eligibility, reducing or denying assistance, use outside the three pilot districts.

## Factors
- District, household size, language of interview, presence of disability, head-of-household gender.

## Metrics
- Recall of "urgent" households at the operating threshold (primary), precision, calibration (ECE).
- 95% confidence intervals by bootstrap.

## Evaluation data
- 4,200 households surveyed Jan–Jun 2025, labelled by two protection officers (Cohen's kappa 0.78).

## Quantitative analyses
| Group            | n     | Recall (urgent) | Precision |
|------------------|-------|-----------------|-----------|
| All              | 4,200 | 0.86 [0.83, 0.89] | 0.61    |
| District A       | 1,900 | 0.88            | 0.63      |
| District C       |   700 | 0.79            | 0.55      |
| Disability = yes |   520 | 0.84            | 0.58      |

## Ethical considerations
- Uses sensitive personal data (processed under the data-protection policy; pseudonymised).
- Risk of under-prioritising households under-represented in training data (District C) — mitigated by manual review quotas.

## Caveats and recommendations
- Retrain or re-validate if intake forms change. Monitor recall by district monthly.
- Do not use scores as a proxy for "deservingness".
```

## Datasheets for datasets

Inspired by electronics datasheets, a **datasheet** answers questions across a dataset's lifecycle:

- **Motivation**: why was the dataset created, by whom, funded by whom?
- **Composition**: what do instances represent; how many; missing data; sensitive information; does it relate to people?
- **Collection process**: how, when, by whom; consent; sampling strategy.
- **Preprocessing/cleaning/labelling**: what was done; is raw data available; annotation guidelines and agreement.
- **Uses**: what it has been used for; what it should **not** be used for.
- **Distribution**: licence, access restrictions.
- **Maintenance**: who maintains it; how errors are reported and corrected; versioning; deletion requests.

Related formats include **Data Statements** for NLP (describing speaker demographics, language varieties and annotator backgrounds) and **Data Cards**.

## System cards and AI documentation at scale

For complex systems (e.g. LLM applications combining models, retrieval, tools and filters), **system cards** document the whole system: components, safety evaluations, red-teaming results, mitigations and deployment safeguards. Major AI developers publish system cards for frontier models; organisations deploying AI should maintain internal equivalents.

## Why documentation matters

- **Informed use**: prevents misuse outside intended conditions.
- **Accountability**: records decisions, responsibilities and known risks.
- **Auditing**: gives internal and external reviewers what they need.
- **Regulation**: many emerging frameworks require documentation of high-risk systems — technical documentation, data governance, performance, human oversight and logging (e.g. under the EU AI Act for high-risk systems).
- **Institutional memory**: survives staff turnover.

:::tip
Write the model card **during** development, not after. Deciding what goes in "intended use", "out-of-scope uses" and "factors" early forces the right conversations about evaluation and risk — and those conversations often change the design.
:::

:::exercise
1. Write a complete model card for a model you have trained, including a disaggregated results table with confidence intervals.
2. Write a datasheet for a dataset you have collected or used, answering at least 20 of the datasheet questions.
3. Review a public model card on Hugging Face. What important information is missing, and why might it matter?
:::

:::takeaway
- Model cards document details, intended and out-of-scope uses, factors, metrics, data, disaggregated results, ethics and caveats.
- Datasheets document a dataset's motivation, composition, collection, labelling, uses, distribution and maintenance.
- System cards document whole AI systems, including safety evaluations and mitigations.
- Documentation enables informed use, accountability, audits and regulatory compliance — write it during development.
:::
