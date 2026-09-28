=== POST ===
slug: what-is-mlops
title: "What Is MLOps? From Notebook to Reliable Production System"
category: mlops
level: Beginner
tags: mlops, production ml, lifecycle, technical debt, devops
summary: Most ML projects fail not because of the model but because of everything around it. We define MLOps, walk through the ML lifecycle, describe maturity levels, and examine the hidden technical debt of machine learning systems.
---
A student trains a model in a notebook, reaches 94% accuracy, and considers the project done. In an organisation, that is where the real work begins. The model must receive fresh data reliably, produce predictions for real users with acceptable latency, be monitored as the world changes, be retrained and redeployed safely, and be auditable months later. **MLOps** — machine learning operations — is the set of practices that makes this possible.

## Why ML in production is hard

In their influential paper "Hidden Technical Debt in Machine Learning Systems" (Sculley et al., NeurIPS 2015), Google engineers showed a now-famous diagram: the ML code is a tiny box surrounded by much larger boxes — data collection, data verification, feature extraction, configuration, serving infrastructure, monitoring, analysis tools, process management. ML systems also accumulate unique forms of **technical debt**:

- **Entanglement**: "changing anything changes everything" — adding or modifying one feature changes how the model uses all others.
- **Data dependencies**: upstream data sources change silently (a sensor is recalibrated; a form field changes meaning).
- **Feedback loops**: model predictions influence the data used to retrain it (recommendations shape clicks).
- **Glue code and pipeline jungles**: ad-hoc scripts that nobody fully understands.
- **Configuration debt**: hundreds of settings with poor documentation.
- **Undeclared consumers**: other systems quietly depend on your model's outputs.

## The ML lifecycle

```text
Problem framing → Data collection & labelling → Data validation → Feature engineering
      → Training & experiment tracking → Evaluation & validation → Packaging
      → Deployment (batch / online / edge) → Monitoring → Retraining → (repeat)
```

Unlike traditional software, ML systems can degrade **without any code change**, because the data changes. MLOps treats data, models and code as first-class, versioned artefacts.

## Core principles

1. **Versioning everything**: code (Git), data (dataset versions/snapshots), models (registry), configurations and environments.
2. **Automation**: reproducible pipelines for training, evaluation and deployment instead of manual notebook steps.
3. **Continuous testing**: code tests, data tests (schemas, distributions), model tests (performance thresholds, fairness slices, robustness).
4. **Continuous delivery**: safe, repeatable deployment with rollbacks.
5. **Monitoring**: of inputs, predictions, performance, latency, cost — and of harm.
6. **Reproducibility and auditability**: be able to answer "which data, code and settings produced this prediction?"
7. **Collaboration**: shared tools and conventions between data scientists, engineers, domain experts and risk/compliance teams.

## MLOps maturity levels

A widely cited framework (Google Cloud's MLOps levels) describes:

- **Level 0 — manual**: notebooks, manual hand-off of a model file to engineers, rare releases, no monitoring.
- **Level 1 — ML pipeline automation**: automated, reproducible training pipelines; continuous training triggered by new data; model registry; monitoring.
- **Level 2 — CI/CD pipeline automation**: automated testing and deployment of the pipelines themselves, enabling rapid, reliable iteration.

Most organisations — including many NGOs and public agencies — are at level 0. Moving to level 1 for important models brings most of the benefit.

## A minimal production-minded project

```text
project/
├── data/                 # raw data is NOT committed; use versioned storage + a manifest
├── src/
│   ├── data.py           # loading + validation
│   ├── features.py       # feature engineering (shared by training AND serving)
│   ├── train.py          # training entry point, reads config
│   ├── evaluate.py       # metrics, slices, thresholds
│   └── serve.py          # prediction API
├── configs/train.yaml    # hyperparameters, paths, seeds
├── tests/                # unit, data and model tests
├── Dockerfile            # reproducible environment
└── pipeline.yaml         # orchestration definition (e.g. DVC, Airflow, Kubeflow, Prefect)
```

```python
# src/train.py — a pipeline step that is reproducible and self-documenting
import json, hashlib, yaml, joblib, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

def file_hash(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:12]

def main(cfg_path="configs/train.yaml"):
    cfg = yaml.safe_load(open(cfg_path))
    train, valid = pd.read_parquet(cfg["train_path"]), pd.read_parquet(cfg["valid_path"])
    X, y = train.drop(columns=cfg["target"]), train[cfg["target"]]
    model = HistGradientBoostingClassifier(**cfg["params"], random_state=cfg["seed"]).fit(X, y)
    auc = roc_auc_score(valid[cfg["target"]], model.predict_proba(valid.drop(columns=cfg["target"]))[:, 1])
    assert auc >= cfg["min_auc"], f"Model below quality gate: {auc:.3f}"
    joblib.dump(model, "artifacts/model.joblib")
    json.dump({"auc": auc, "train_data": file_hash(cfg["train_path"]), "config": cfg},
              open("artifacts/metadata.json", "w"), indent=2)

if __name__ == "__main__":
    main()
```

Note the **quality gate** (the model is not saved if it underperforms) and the **metadata** linking the model to its data and configuration.

## Roles

Data scientists, ML engineers, data engineers, platform/DevOps engineers, product owners, domain experts and governance teams all contribute. In small teams one person wears many hats — which makes simple, well-documented practices even more valuable.

:::tip
You do not need a large platform to practise MLOps. Start with Git, a requirements/lock file or Docker image, a config file, a training script instead of a notebook, a saved evaluation report, and a simple monitoring dashboard. These alone put you ahead of most projects.
:::

:::exercise
1. Take one of your notebook projects and convert it into a script-based pipeline with a config file and a quality gate.
2. List the data sources of a model you know and, for each, write what could change silently upstream.
3. Assess an ML system you know against the three maturity levels and propose the next improvement.
:::

:::takeaway
- MLOps applies engineering discipline to the full ML lifecycle, not just model training.
- ML systems accumulate unique technical debt: entanglement, data dependencies, feedback loops, glue code.
- Core practices: version everything, automate pipelines, test data and models, deploy safely, monitor continuously.
- Maturity progresses from manual notebooks to automated training and CI/CD; start simple.
:::

=== POST ===
slug: data-versioning-and-pipelines
title: "Data Versioning, Validation and Pipelines"
category: mlops
level: Intermediate
tags: data versioning, dvc, data validation, pipelines, data quality
summary: Models are only as good as their data — and data changes. We cover versioning datasets, validating schemas and distributions, building reproducible data pipelines, and orchestration tools.
---
When a model's performance suddenly drops, the cause is usually not the model code — it is the **data**. A column changed units, a new category appeared, a join duplicated rows, a sensor failed, an upstream team changed a definition. Managing data with the same rigour as code — **versioning**, **validation** and **reproducible pipelines** — is the foundation of reliable ML.

## Why version data?

- **Reproducibility**: re-create the exact training set behind any model.
- **Debugging**: compare the data of a good model and a bad one.
- **Auditability**: answer regulators, partners or affected people about what data a decision relied on.
- **Collaboration**: everyone trains on the same, identified dataset.

Git is designed for code, not multi-gigabyte files. Options:

- **DVC (Data Version Control)**: stores small pointer files (with content hashes) in Git while the data lives in remote storage (S3, Azure Blob, Google Drive, a server). Checking out a Git commit restores the matching data version.
- **Lakehouse table formats** (Delta Lake, Apache Iceberg, Apache Hudi): versioned tables with **time travel** ("query this table as of last Tuesday").
- **Dataset registries** and immutable, dated snapshots in object storage with manifests.

```bash
# DVC basics
git init && dvc init
dvc remote add -d storage s3://my-bucket/dvc-store
dvc add data/registrations_2025.parquet      # creates data/registrations_2025.parquet.dvc
git add data/registrations_2025.parquet.dvc data/.gitignore
git commit -m "Add registrations dataset v1"
dvc push                                      # upload data to remote storage
```

## Data validation

Validate data **before** it reaches training or inference. Typical checks:

1. **Schema**: expected columns, types, allowed categories, nullability.
2. **Ranges and constraints**: ages between 0 and 120; dates not in the future; IDs unique.
3. **Distribution checks**: compare feature distributions with a reference (training) dataset to detect drift or pipeline bugs.
4. **Volume and freshness**: row counts within expected bounds; latest timestamp recent enough.
5. **Referential integrity**: joins do not drop or duplicate records unexpectedly.
6. **Label quality**: class balance, annotator agreement, suspicious labels.

```python
import pandas as pd
import pandera as pa
from pandera import Column, Check

schema = pa.DataFrameSchema({
    "household_id": Column(str, Check.str_matches(r"^HH-\d{6}$"), unique=True),
    "members": Column(int, Check.in_range(1, 30)),
    "district": Column(str, Check.isin(["Dhaka", "Chattogram", "Sylhet", "Khulna", "Rajshahi"])),
    "registered_at": Column(pd.Timestamp, Check.le(pd.Timestamp.now())),
    "monthly_income": Column(float, Check.ge(0), nullable=True),
})

df = pd.DataFrame({"household_id": ["HH-000001", "HH-000002"], "members": [4, 45],
                   "district": ["Dhaka", "Barishal"], "registered_at": pd.to_datetime(["2025-01-03", "2025-02-11"]),
                   "monthly_income": [12000.0, None]})
try:
    schema.validate(df, lazy=True)
except pa.errors.SchemaErrors as err:
    print(err.failure_cases[["column", "check", "failure_case"]])
```

Tools include Pandera, Great Expectations, TensorFlow Data Validation and dbt tests. Decide what happens on failure: block the pipeline, quarantine bad rows, or alert a human.

## Pipelines

A pipeline is a directed acyclic graph (DAG) of steps: ingest → validate → clean → feature engineering → split → train → evaluate → register. Good pipelines are:

- **Declarative and reproducible**: defined in code/config, with pinned environments.
- **Idempotent**: re-running a step with the same inputs gives the same outputs.
- **Cached**: unchanged steps are skipped.
- **Observable**: logs, lineage and metrics for every run.

```yaml
# dvc.yaml — a reproducible pipeline; `dvc repro` re-runs only what changed
stages:
  validate:
    cmd: python src/validate.py data/raw.parquet data/valid.parquet
    deps: [src/validate.py, data/raw.parquet]
    outs: [data/valid.parquet]
  features:
    cmd: python src/features.py data/valid.parquet data/features.parquet
    deps: [src/features.py, data/valid.parquet]
    outs: [data/features.parquet]
  train:
    cmd: python src/train.py
    deps: [src/train.py, data/features.parquet, configs/train.yaml]
    outs: [artifacts/model.joblib]
    metrics: [artifacts/metrics.json]
```

**Orchestrators** schedule and monitor pipelines in production: Apache Airflow, Prefect, Dagster, Kubeflow Pipelines, cloud-native services.

## Data lineage and privacy

**Lineage** records where each dataset came from and how it was transformed — vital for debugging and audits. Combine it with **data-protection practices**: minimise personal data, pseudonymise identifiers, restrict access, record legal bases and retention periods, and support deletion requests (which means you must know which datasets and models used a person's data).

:::warning
Never commit raw personal data to Git — even in private repositories. Git history is hard to erase. Store sensitive data in access-controlled storage and version it with pointers.
:::

:::exercise
1. Put a small project's dataset under DVC with a remote, then change the data and switch between versions.
2. Write a Pandera or Great Expectations suite with at least eight checks for a dataset you use.
3. Build a three-stage `dvc.yaml` pipeline and verify that changing only the training config re-runs only the training stage.
:::

:::takeaway
- Most production ML failures are data failures; treat data as a versioned artefact.
- Use DVC, lakehouse table formats or immutable snapshots to version datasets.
- Validate schema, ranges, distributions, volume, freshness and label quality before training and inference.
- Build reproducible, cached, observable pipelines with lineage — and protect personal data.
:::

=== POST ===
slug: experiment-tracking
title: "Experiment Tracking: Never Lose a Result Again"
category: mlops
level: Beginner
tags: experiment tracking, mlflow, weights and biases, model registry, reproducibility
summary: ML development involves hundreds of runs with different data, code and hyperparameters. We cover what to track, how to use MLflow for runs, metrics and artefacts, the model registry, and good experiment hygiene.
---
"Which settings produced that 91% model from two weeks ago?" If you cannot answer this in seconds, you need **experiment tracking**. ML development is empirical: you try many ideas, most fail, and the few that succeed must be reproducible. Spreadsheets and filenames like `model_final_v3_really_final.pkl` do not scale. Tracking tools record every run systematically.

## What to track for every run

- **Code version**: Git commit hash (and whether the working tree had uncommitted changes).
- **Data version**: dataset identifier or hash, split definitions.
- **Configuration**: all hyperparameters, feature lists, preprocessing options, random seeds.
- **Environment**: library versions, hardware (GPU type).
- **Metrics**: training/validation curves, final metrics, per-slice metrics (by language, region, class).
- **Artefacts**: the model file, plots (confusion matrix, calibration), sample predictions, evaluation reports.
- **Notes and tags**: the hypothesis being tested, the conclusion.

## Tools

- **MLflow**: open source; tracking server, UI, model registry, model packaging. Can run locally or self-hosted — useful for sensitive environments.
- **Weights & Biases**, **Neptune**, **Comet**: hosted platforms with rich visualisation and collaboration.
- **TensorBoard**: training curves, especially for deep learning.
- **Aim**, **ClearML**, **DVC experiments**: other open-source options.

## MLflow in practice

```python
import mlflow, mlflow.sklearn, subprocess
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

mlflow.set_tracking_uri("file:./mlruns")                 # or a shared tracking server URL
mlflow.set_experiment("triage-classifier")

X, y = load_breast_cancer(return_X_y=True, as_frame=True)
X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)

for n_estimators in [100, 300]:
    for max_depth in [4, None]:
        with mlflow.start_run(run_name=f"rf-{n_estimators}-{max_depth}"):
            commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
            mlflow.set_tags({"git_commit": commit or "not-a-git-repo", "data_version": "bc-sklearn-v1",
                             "hypothesis": "deeper trees improve recall"})
            params = {"n_estimators": n_estimators, "max_depth": max_depth, "random_state": 0}
            mlflow.log_params(params)
            model = RandomForestClassifier(**params).fit(X_tr, y_tr)
            proba = model.predict_proba(X_va)[:, 1]
            mlflow.log_metrics({"val_auc": roc_auc_score(y_va, proba),
                                "val_f1": f1_score(y_va, proba > 0.5)})
            ConfusionMatrixDisplay.from_predictions(y_va, proba > 0.5)
            mlflow.log_figure(plt.gcf(), "confusion_matrix.png"); plt.close()
            mlflow.sklearn.log_model(model, name="model", input_example=X_va.head(3))
# Then run `mlflow ui` and compare runs side by side.
```

Autologging (`mlflow.autolog()`) captures parameters and metrics automatically for many libraries.

## The model registry

A **model registry** is the catalogue of models that might be deployed:

- each registered model has **versions**, linked to the runs (and thus data and code) that produced them;
- versions carry **aliases or stages** (e.g. "candidate", "champion", "archived");
- promotion requires review — metrics, fairness checks, sign-off;
- deployment systems load "the current champion" rather than a file path, enabling clean rollbacks.

```python
from mlflow import MlflowClient
client = MlflowClient()
best = mlflow.search_runs(experiment_names=["triage-classifier"], order_by=["metrics.val_auc DESC"]).iloc[0]
mv = mlflow.register_model(f"runs:/{best.run_id}/model", "triage-classifier")
client.set_registered_model_alias("triage-classifier", "candidate", mv.version)
```

## Experiment hygiene

:::tip
- **One change at a time** where possible, and write the hypothesis before running.
- **Fix seeds** and run multiple seeds for important comparisons; report variance.
- **Keep a fixed validation set** and a locked test set; never tune on the test set.
- **Log failures too** — negative results prevent repeating dead ends.
- **Name and tag runs meaningfully**; archive clutter.
- **Record the conclusion** in the run notes: what did you learn?
:::

## Beyond metrics: comparing models properly

Leaderboards of runs invite chasing tiny improvements. Compare candidates with confidence intervals, paired tests and slice metrics (see the hypothesis-testing lecture). A model 0.3% better on average but 5% worse for a minority language may be the wrong choice.

:::exercise
1. Track a hyperparameter sweep for one of your projects in MLflow and identify the best run from the UI.
2. Register two model versions and practise promoting and rolling back using aliases.
3. Add per-slice metrics (e.g. per class or subgroup) to your logged metrics and find a case where the overall best model is worse on some slice.
:::

:::takeaway
- Track code version, data version, configuration, environment, metrics, artefacts and conclusions for every run.
- MLflow provides runs, metrics, artefacts, model packaging and a registry; hosted tools add collaboration.
- A model registry versions deployable models with aliases/stages for safe promotion and rollback.
- Good hygiene — hypotheses, seeds, fixed splits, logged failures — makes results trustworthy.
:::

=== POST ===
slug: reproducibility-in-ml
title: "Reproducibility in Machine Learning"
category: mlops
level: Intermediate
tags: reproducibility, random seeds, determinism, environments, research practice
summary: Can someone else — or you, in six months — get the same result? We examine sources of non-reproducibility, from seeds and GPUs to data and environments, and practical steps for reproducible research and production.
---
Reproducibility is a cornerstone of science and a practical necessity in engineering. Yet surveys and replication studies across ML have found that many published results are hard to reproduce: code is missing, hyperparameters are unreported, baselines are under-tuned, and results vary with random seeds. In production, irreproducibility means you cannot debug a model, explain a decision, or safely retrain. This lecture covers the causes and cures.

## Levels of reproducibility

- **Repeatability**: the same team, same setup, gets the same result.
- **Reproducibility**: a different team, using the original code and data, gets the same result.
- **Replicability**: a different team, with independent implementation or data, reaches the same **conclusion**.

Aim for the first two in every project; the third is the gold standard of science.

## Sources of non-reproducibility

1. **Randomness**: weight initialisation, data shuffling, dropout, augmentation, train/test splits, sampling.
2. **Non-deterministic hardware operations**: some GPU kernels (e.g. atomic additions in certain convolution backward passes and scatter operations) produce slightly different results run to run; different GPU models or library versions change floating-point results.
3. **Software environment**: library versions (a new scikit-learn default can change results), CUDA/cuDNN versions, operating system.
4. **Data**: unversioned datasets, changing upstream sources, undocumented filtering, time-dependent queries ("last 30 days").
5. **Hidden configuration**: unrecorded hyperparameters, manual notebook steps executed out of order.
6. **Evaluation variance**: small test sets and single-seed results make "improvements" indistinguishable from noise.

## Controlling randomness

```python
import os, random
import numpy as np
import torch

def set_seed(seed: int = 42, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False                 # avoid auto-tuned, variable algorithms
        torch.use_deterministic_algorithms(True, warn_only=True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"      # needed for deterministic cuBLAS ops

set_seed(42)

# DataLoader workers need their own seeding
def worker_init_fn(worker_id):
    s = torch.initial_seed() % 2**32
    np.random.seed(s); random.seed(s)

g = torch.Generator().manual_seed(42)
# loader = DataLoader(ds, shuffle=True, generator=g, worker_init_fn=worker_init_fn, num_workers=4)
```

Deterministic mode can slow training; for many projects, fixing seeds (without full determinism) plus **reporting variance across seeds** is the pragmatic choice.

:::note
Seeds are not a hyperparameter to tune. Picking the "lucky" seed that gives the best test score is a form of overfitting to the test set. Report the mean and spread over several seeds.
:::

## Controlling the environment

- **Pin dependencies**: `requirements.txt` with exact versions, or lock files (`pip-tools`, `poetry.lock`, `uv.lock`, `conda-lock`).
- **Containers**: Docker images capture the operating system, libraries and CUDA stack (next lectures).
- **Record hardware** and driver versions with each run.

```text
# requirements.txt (pinned)
numpy==2.1.3
pandas==2.2.3
scikit-learn==1.5.2
torch==2.5.1
```

## Controlling data and configuration

- Version datasets (DVC, snapshots) and record their hashes with each run.
- Store all hyperparameters in config files; log them automatically.
- Replace notebooks with scripts for final pipelines — or use tools that execute notebooks top-to-bottom in CI.
- Save split indices, not just the splitting code.

## Reporting for research

The NeurIPS and ICML reproducibility checklists ask authors to report, among other things: dataset details and preprocessing; hyperparameters and how they were chosen; number of runs, seeds and variance; compute used; and code availability. A good paper or thesis includes:

1. Code and instructions to reproduce each table and figure.
2. Exact data versions (or instructions to obtain them).
3. Hyperparameter search spaces and budgets **for baselines too**.
4. Means and confidence intervals over multiple seeds.
5. Compute requirements.
6. Known limitations.

## Reproducibility in production

For deployed models, reproducibility supports **accountability**: for any prediction, you should be able to identify the model version, its training data version, the code and configuration, and the input features used. Log prediction requests with model version identifiers (respecting privacy rules), and keep trained artefacts immutable in a registry.

:::exercise
1. Train the same small network three times without seeds, then three times with `set_seed`. Compare results.
2. Create a pinned environment for a project and verify that a fresh machine or container reproduces your metrics.
3. Take a published paper's code and try to reproduce one table. Document every obstacle you hit.
:::

:::takeaway
- Distinguish repeatability, reproducibility and replicability.
- Non-reproducibility comes from randomness, non-deterministic GPU ops, environments, data and hidden configuration.
- Fix seeds (and optionally enforce determinism), pin environments, version data and configs, script pipelines.
- Report variance over seeds and full experimental details; log model and data versions for every prediction.
:::

=== POST ===
slug: model-serving-and-apis
title: "Model Serving: Batch, Real-Time APIs and Streaming"
category: mlops
level: Intermediate
tags: model serving, fastapi, rest api, batch inference, latency
summary: A model creates value only when its predictions reach people and systems. We compare batch, online and streaming serving, build a FastAPI prediction service with validation, and cover latency, scaling and safe rollout.
---
Once a model is trained and validated, it must be **served**: made available to produce predictions for applications, dashboards or users. The right serving pattern depends on how fresh predictions must be, how many are needed, and how quickly. Choosing well saves cost and complexity; choosing poorly can make a good model useless.

## Serving patterns

| Pattern | How | Latency | Example |
|---|---|---|---|
| **Batch** | Score many records on a schedule; store results | Minutes–hours (predictions precomputed) | Nightly risk scores for all registered households |
| **Online (real-time) API** | Request–response over HTTP/gRPC | Milliseconds–seconds | Classify a message when it arrives |
| **Streaming** | Consume events from a queue (e.g. Kafka), emit predictions | Seconds | Fraud detection on transactions |
| **Edge / on-device** | Model runs on phone, browser or device | Local, offline | Crop-disease detection in the field |

**Prefer batch** when predictions are not needed instantly: it is simpler, cheaper, easier to monitor and retry. Use online serving only when fresh, per-request predictions are required.

## Building an online prediction API with FastAPI

```python
# serve.py — run with:  uvicorn serve:app --host 0.0.0.0 --port 8000
import time, logging, joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_VERSION = "triage-classifier:7"
model = joblib.load("artifacts/model.joblib")          # load ONCE at startup, not per request
app = FastAPI(title="Triage model API", version=MODEL_VERSION)
log = logging.getLogger("predictions")

class Household(BaseModel):                             # input validation with types and ranges
    members: int = Field(ge=1, le=30)
    children_under_5: int = Field(ge=0, le=15)
    monthly_income: float = Field(ge=0)
    has_disability: bool
    district: str

class Prediction(BaseModel):
    priority_score: float
    priority: str
    model_version: str

@app.get("/health")
def health():
    return {"status": "ok", "model_version": MODEL_VERSION}

@app.post("/predict", response_model=Prediction)
def predict(h: Household):
    t0 = time.perf_counter()
    try:
        X = pd.DataFrame([h.model_dump()])
        score = float(model.predict_proba(X)[0, 1])
    except Exception:
        log.exception("prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed")
    priority = "high" if score >= 0.7 else "medium" if score >= 0.4 else "low"
    log.info("latency_ms=%.1f score=%.3f version=%s", (time.perf_counter() - t0) * 1000, score, MODEL_VERSION)
    return Prediction(priority_score=round(score, 4), priority=priority, model_version=MODEL_VERSION)
```

Key practices illustrated:

- **Load the model once** at startup.
- **Validate inputs** with a schema (Pydantic) — reject malformed requests with clear errors rather than producing garbage predictions.
- **Return the model version** with every prediction for traceability.
- **Health endpoints** for load balancers and orchestrators.
- **Structured logging** of latency and outputs (never log sensitive personal data unnecessarily).
- **Same feature code** in training and serving — share the preprocessing pipeline (e.g. a scikit-learn `Pipeline` saved as one artefact) to avoid **training–serving skew**.

## Latency and throughput

- Measure **p50, p95 and p99** latency — tail latency matters for user experience.
- Optimise the model (smaller architecture, quantisation, ONNX Runtime/TensorRT), batch requests (**dynamic batching** groups concurrent requests for GPU efficiency), cache frequent results, and keep feature lookups fast.
- Scale **horizontally** (more replicas behind a load balancer) and autoscale on load.

## Dedicated serving frameworks

For deep learning and larger deployments: **TorchServe**, **TensorFlow Serving**, **NVIDIA Triton Inference Server** (multi-framework, dynamic batching, GPU sharing), **BentoML**, **Ray Serve**, **KServe** on Kubernetes, and LLM-specific engines (vLLM, TGI). Managed cloud endpoints are an option when data policies allow.

## Safe rollout strategies

Never replace a production model all at once:

- **Shadow deployment**: the new model receives a copy of live traffic and its predictions are logged but not used — compare with the current model safely.
- **Canary release**: route a small fraction (e.g. 5%) of traffic to the new model; watch metrics; increase gradually.
- **Blue–green deployment**: run old and new environments side by side and switch traffic instantly, with instant rollback.
- **A/B testing**: randomised comparison of business/outcome metrics (see the A/B testing lecture).

:::warning
When a model's output affects people (priority for assistance, eligibility checks), the serving layer is part of a decision system. Provide explanations or reason codes where appropriate, log decisions for audit, allow human override and appeal, and fail **safely**: if the model is unavailable, fall back to a documented manual process rather than a default decision.
:::

:::exercise
1. Wrap one of your trained models in the FastAPI template, add input validation, and load-test it (e.g. with `locust` or `hey`) to measure p95 latency.
2. Implement a batch scoring job that reads a CSV, validates rows, writes predictions with the model version, and reports rejected rows.
3. Design a shadow-deployment plan for replacing a production model: what would you log and compare, and for how long?
:::

:::takeaway
- Choose batch, online, streaming or edge serving based on freshness, volume and latency needs; prefer batch when possible.
- A good prediction API loads the model once, validates inputs, returns model versions, exposes health checks and logs responsibly.
- Share preprocessing between training and serving to avoid skew; measure tail latency and scale horizontally.
- Roll out safely with shadow, canary, blue–green or A/B strategies, and design fallbacks for decision systems.
:::

=== POST ===
slug: docker-for-machine-learning
title: "Docker and Containers for Machine Learning"
category: mlops
level: Beginner
tags: docker, containers, environments, gpu, deployment
summary: "It works on my machine" is not a deployment strategy. We explain containers, write efficient Dockerfiles for ML training and serving, handle GPUs, and cover image size, security and orchestration basics.
---
A model trained with one version of PyTorch, CUDA and NumPy may fail — or silently behave differently — on a machine with other versions. **Containers** package code together with its entire runtime environment — operating system libraries, Python, dependencies, configuration — so it runs identically on a laptop, a server, a cloud cluster or a partner's infrastructure. **Docker** is the most widely used container tool, and container skills are essential for any ML engineer.

## Containers vs virtual machines

- A **virtual machine** emulates a full computer, with its own operating-system kernel — heavy (gigabytes), slow to start.
- A **container** shares the host kernel and isolates processes, filesystems and networking — lightweight (megabytes to a few gigabytes), starts in seconds.

Key concepts:

- **Image**: an immutable template built from a **Dockerfile**, composed of cached **layers**.
- **Container**: a running instance of an image.
- **Registry**: where images are stored and shared (Docker Hub, GitHub Container Registry, cloud registries).
- **Volume**: persistent storage mounted into a container (for data and model files).

## A Dockerfile for a prediction service

```dockerfile
# Dockerfile
FROM python:3.11-slim AS base

# System settings: no .pyc files, unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

WORKDIR /app

# 1) Install dependencies first — this layer is cached until requirements change
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2) Copy application code and model artefact
COPY src/ ./src/
COPY artifacts/model.joblib ./artifacts/model.joblib

# 3) Run as a non-root user for security
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t triage-api:1.0.0 .
docker run --rm -p 8000:8000 triage-api:1.0.0
curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
     -d '{"members": 5, "children_under_5": 2, "monthly_income": 8000, "has_disability": false, "district": "Dhaka"}'
```

## Layer caching and image size

Order instructions from **least** to **most** frequently changing: base image → system packages → Python dependencies → application code. Then editing code does not reinstall dependencies.

Keep images small and secure:

- Use slim base images; avoid installing build tools in the final image.
- Use **multi-stage builds**: compile or install in a "builder" stage and copy only what is needed into the final stage.
- Add a `.dockerignore` (exclude `.git`, datasets, notebooks, virtual environments, secrets).
- Clean package caches (`--no-cache-dir`).

## GPUs in containers

Deep-learning images need CUDA libraries matching the framework version. Use official base images (e.g. PyTorch or NVIDIA CUDA images) and run with GPU access via the NVIDIA Container Toolkit:

```bash
docker run --rm --gpus all pytorch/pytorch:latest python -c "import torch; print(torch.cuda.is_available())"
```

The host needs a compatible NVIDIA driver; the container brings the CUDA runtime.

## Training in containers

Containers make training jobs reproducible and portable to clusters:

```bash
docker run --rm --gpus all \
  -v "$PWD/data:/app/data:ro" \          # mount data read-only
  -v "$PWD/artifacts:/app/artifacts" \   # write model outputs to the host
  -e MLFLOW_TRACKING_URI=http://mlflow:5000 \
  trainer:2.3.0 python src/train.py --config configs/train.yaml
```

Do not bake large datasets into images; mount them or read them from storage.

## Docker Compose for local stacks

Run multi-service setups (API + database + MLflow + monitoring) with one file:

```yaml
# compose.yaml
services:
  api:
    build: .
    ports: ["8000:8000"]
    depends_on: [mlflow]
  mlflow:
    image: ghcr.io/mlflow/mlflow:latest
    command: mlflow server --host 0.0.0.0 --port 5000
    ports: ["5000:5000"]
```

## Security essentials

:::warning
- **Never put secrets** (API keys, passwords, database credentials) in Dockerfiles or images — anyone with the image can extract them. Inject secrets at runtime via environment variables from a secret manager, or mounted secret files.
- Run as a **non-root** user; keep base images updated; **scan images** for known vulnerabilities (e.g. Trivy, Docker Scout).
- Pin base image versions (or digests) for reproducibility, and rebuild regularly for security patches.
:::

## Orchestration

In production, containers are managed by orchestrators — most commonly **Kubernetes** — which handle scheduling, scaling, restarts, rolling updates, service discovery and resource limits (CPU, memory, GPUs). ML platforms (Kubeflow, KServe, cloud ML services) build on these foundations. For smaller deployments, a single server with Docker Compose, or a managed container service, may be entirely sufficient.

:::exercise
1. Containerise one of your models with the Dockerfile template; verify it runs on a second machine.
2. Convert it to a multi-stage build and compare image sizes before and after.
3. Scan your image for vulnerabilities and fix at least one finding (e.g. by updating the base image).
:::

:::takeaway
- Containers package code with its full environment so it runs identically everywhere.
- Write Dockerfiles with cache-friendly layer ordering, slim or multi-stage images, `.dockerignore` and non-root users.
- Use NVIDIA base images and `--gpus` for GPU workloads; mount data rather than baking it in.
- Keep secrets out of images, scan for vulnerabilities, and use orchestrators like Kubernetes when scale requires.
:::

=== POST ===
slug: model-monitoring-and-drift
title: "Model Monitoring: Detecting Data Drift and Concept Drift"
category: mlops
level: Intermediate
tags: monitoring, data drift, concept drift, psi, observability
summary: Deployed models degrade silently as the world changes. We define data drift, concept drift and prediction drift, measure drift with PSI and statistical tests, monitor without labels, and design alerts and retraining triggers.
---
A model is a snapshot of the world at training time. The world keeps moving: a new policy changes who applies for services, a pandemic changes behaviour, a new phone model changes image colours, an upstream form changes a field's meaning. Without monitoring, a model can become inaccurate — or unfair — for months before anyone notices. **Monitoring** is how we detect such problems early.

## What can go wrong

- **Data (covariate) drift**: the input distribution changes, $P(\mathbf{x})$ shifts. Example: a larger share of applicants from a new region.
- **Concept drift**: the relationship between inputs and outputs changes, $P(y \mid \mathbf{x})$ shifts. Example: the same household characteristics now imply different needs after price inflation.
- **Label (prior) drift**: the class balance changes, $P(y)$ shifts. Example: a disease outbreak increases positive cases.
- **Prediction drift**: the distribution of model outputs changes — often the first visible symptom.
- **Data quality issues**: missing values surge, units change, a feature is silently set to a default — technically a pipeline bug rather than real-world drift, but the most common cause of sudden degradation.
- **Upstream/downstream changes**: new software versions, changed feature definitions, new consumers using outputs differently.

## Monitoring layers

1. **Operational**: uptime, latency (p50/p95/p99), error rates, throughput, resource usage, cost.
2. **Data quality**: schema violations, missing-value rates, range violations, new categories, volume.
3. **Distribution drift**: per-feature and prediction distributions versus a reference window.
4. **Performance**: accuracy, precision/recall, calibration — when (delayed) ground-truth labels arrive.
5. **Fairness and slices**: performance and outcome rates per subgroup.
6. **Business/outcome metrics**: is the system still achieving its purpose?

## Measuring drift

**Population Stability Index (PSI)** — widely used in credit scoring. Bin a feature (using reference quantiles), then compare the proportions of reference ($p_i$) and current ($q_i$) data per bin:

$$
\text{PSI} = \sum_i(q_i - p_i)\ln\frac{q_i}{p_i}
$$

Common rules of thumb: PSI < 0.1 little change; 0.1–0.25 moderate; > 0.25 significant shift (treat thresholds as starting points, not laws).

Other measures: **Kolmogorov–Smirnov** test for continuous features, **chi-squared** test for categorical features, **Jensen–Shannon** or **Wasserstein** distances, and **domain classifiers** — train a model to distinguish reference from current data; if it succeeds (AUC well above 0.5), the distributions differ, and its feature importances show where.

```python
import numpy as np
from scipy.stats import ks_2samp

def psi(reference, current, bins=10, eps=1e-6):
    edges = np.quantile(reference, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    p = np.histogram(reference, edges)[0] / len(reference) + eps
    q = np.histogram(current, edges)[0] / len(current) + eps
    return float(np.sum((q - p) * np.log(q / p)))

rng = np.random.default_rng(0)
ref = rng.lognormal(9.0, 0.6, 20_000)                 # income at training time
cur_same = rng.lognormal(9.0, 0.6, 5_000)
cur_shift = rng.lognormal(9.25, 0.7, 5_000)            # inflation + more variance
for name, cur in [("same", cur_same), ("shifted", cur_shift)]:
    print(f"{name:<8} PSI={psi(ref, cur):.3f}  KS p-value={ks_2samp(ref, cur).pvalue:.2e}")
```

:::note
With large samples, statistical tests flag even trivial differences as "significant". Focus on **effect sizes** (PSI, distances) and on whether drift affects **model performance** — not every drift matters. Prioritise features that are important to the model.
:::

## Monitoring without labels

Ground truth often arrives late (did the applicant's situation actually worsen?) or never. Proxies:

- input and prediction drift;
- changes in the model's **confidence** distribution;
- agreement between the production model and a reference or shadow model;
- **performance estimation** methods that predict accuracy from confidence under covariate shift (e.g. confidence-based estimators), with caution;
- targeted **human review** of a random sample of predictions — often the most reliable signal.

## Alerts and responses

Design alerting deliberately: too many alerts cause fatigue; too few miss problems.

- Define **thresholds per metric**, windows (hourly, daily), and minimum sample sizes.
- Route alerts to owners with **runbooks**: what to check, how to roll back, when to retrain.
- Responses range from investigating a data pipeline bug, to recalibrating thresholds, retraining on recent data, rolling back, or pausing automated decisions in favour of manual processing.

## Retraining strategies

- **Scheduled** (e.g. monthly) — simple, predictable.
- **Triggered** by drift or performance degradation.
- **Continuous/online learning** — powerful but risky (feedback loops, poisoning); use with safeguards.

Every retrained model must pass the same validation gates as the original — including fairness checks — before deployment.

Tools: Evidently, NannyML, WhyLabs/whylogs, Arize, cloud monitoring services, and Prometheus/Grafana for operational metrics.

:::exercise
1. Simulate a dataset whose distribution shifts gradually over 12 months; compute monthly PSI for each feature and plot them.
2. Train a domain classifier to distinguish training data from "current" data and use its feature importances to explain the drift.
3. Write a runbook for a drift alert on a model you know: checks, decision criteria, rollback and communication steps.
:::

:::takeaway
- Models degrade through data drift, concept drift, label drift and — most often — data quality bugs.
- Monitor operations, data quality, distributions, performance, fairness slices and outcomes.
- Measure drift with PSI, KS/chi-squared tests, distances or domain classifiers; focus on effect sizes and important features.
- When labels are delayed, use proxies and human review; alert with runbooks and retrain behind validation gates.
:::

=== POST ===
slug: ci-cd-for-machine-learning
title: "CI/CD for Machine Learning: Testing and Automating ML Systems"
category: mlops
level: Intermediate
tags: ci/cd, testing, github actions, continuous training, quality gates
summary: Continuous integration and delivery bring software-engineering discipline to ML. We cover the testing pyramid for ML — code, data and model tests — quality gates, continuous training, and a practical GitHub Actions workflow.
---
In software engineering, **continuous integration (CI)** automatically builds and tests every code change, and **continuous delivery/deployment (CD)** automatically releases changes that pass. ML systems need the same discipline — plus more, because behaviour depends on data and trained artefacts, not just code. This lecture shows how to test ML systems and automate their release safely.

## What changes in an ML system?

A new model can result from changes to:

- **code** (feature logic, training script, serving code);
- **data** (new training data, schema changes);
- **configuration** (hyperparameters, thresholds);
- **dependencies** (library versions).

CI/CD for ML (sometimes called CI/CD/CT — adding **continuous training**) must test all of them.

## The testing pyramid for ML

### 1. Code tests (fast, run on every commit)
- **Unit tests** for feature functions, preprocessing and utilities.
- **Edge cases**: missing values, empty inputs, unseen categories, extreme values.
- **API contract tests** for the serving interface.

### 2. Data tests
- Schema and constraint validation (types, ranges, uniqueness).
- Distribution checks against reference data.
- **Leakage checks**: no target-derived features; no overlap between train and test by ID.

### 3. Model tests
- **Performance gates**: metrics must exceed a threshold and not regress versus the current production model beyond a tolerance.
- **Slice tests**: minimum performance per subgroup/language/region.
- **Behavioural tests** (inspired by CheckList, Ribeiro et al., 2020): invariance ("changing a name should not change the prediction"), directional expectations ("more household members should not decrease need score, all else equal"), and minimum functionality tests.
- **Robustness** to noise and typos; **calibration** checks.
- **Training smoke test**: train on a tiny sample for a few steps and confirm loss decreases.

### 4. Integration and end-to-end tests
- The full pipeline runs from raw data to a served prediction in a staging environment.
- The packaged model loads and produces the same predictions as in training (**training–serving parity**).

```python
# tests/test_model_behaviour.py  (run with pytest)
import joblib, pandas as pd, pytest

model = joblib.load("artifacts/model.joblib")
BASE = {"members": 5, "children_under_5": 1, "monthly_income": 9000.0, "has_disability": False, "district": "Dhaka"}

def score(**overrides):
    return model.predict_proba(pd.DataFrame([{**BASE, **overrides}]))[0, 1]

def test_directional_income():
    # Lower income should not reduce the priority score (monotonic expectation)
    assert score(monthly_income=3000.0) >= score(monthly_income=15000.0) - 1e-6

def test_deterministic_predictions():
    # The same input must always produce the same score (no hidden randomness at inference)
    assert abs(score() - score()) < 1e-12

@pytest.mark.parametrize("children", [0, 3, 6])
def test_valid_probability(children):
    s = score(children_under_5=children)
    assert 0.0 <= s <= 1.0
```

## Quality gates and model promotion

A trained candidate is promoted only if it passes gates, for example:

- validation AUC ≥ 0.85 **and** not worse than production by more than 0.005;
- recall for the high-priority class ≥ 0.80 in **every** region;
- calibration error below a threshold;
- all behavioural tests pass;
- latency and model size within budget;
- human sign-off for high-impact models.

Results are recorded in the model registry with the run that produced them.

## A GitHub Actions workflow

```yaml
# .github/workflows/ml-ci.yml
name: ml-ci
on:
  pull_request:
  push:
    branches: [main]
jobs:
  test-and-train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: "3.11"}
      - run: pip install -r requirements.txt
      - name: Lint and unit tests
        run: |
          ruff check src tests
          pytest tests/unit -q
      - name: Data validation
        run: python src/validate.py data/sample.parquet
      - name: Smoke-train on a sample
        run: python src/train.py --config configs/smoke.yaml
      - name: Model quality gates and behavioural tests
        run: |
          python src/evaluate.py --gates configs/gates.yaml
          pytest tests/model -q
      - name: Build container
        run: docker build -t triage-api:${{ github.sha }} .
```

Full training on large data usually runs on dedicated infrastructure (triggered by the pipeline or schedule), not inside a CI runner; CI runs fast checks and smoke tests.

## Continuous training and delivery

- **Continuous training (CT)**: pipelines retrain automatically on new data (scheduled or drift-triggered), then run the same gates.
- **Continuous delivery**: passing models are packaged and deployed to staging automatically; production promotion may be automatic (low-risk models) or require approval (high-risk models).
- **Rollback**: keep the previous model ready; automate rollback on alert.

:::warning
Automation amplifies both good and bad processes. For models that affect people's access to services, keep a **human approval step** before production promotion, and include fairness and slice gates — an automated pipeline must never be able to silently ship a model that performs worse for a vulnerable group.
:::

:::exercise
1. Write five behavioural tests (invariance, directional, minimum functionality) for a text classifier you have trained.
2. Set up a GitHub Actions workflow that runs unit tests, data validation and a smoke-training step on every pull request.
3. Define a quality-gate configuration for a model in your domain, including at least two slice-based gates, and justify the thresholds.
:::

:::takeaway
- CI/CD for ML must test code, data, configuration and trained models.
- Use code tests, data validation, model performance and slice gates, behavioural tests and end-to-end parity checks.
- Promote models only through explicit quality gates recorded in the registry; keep rollback ready.
- Automate training and delivery, but keep human approval for high-impact models.
:::

=== POST ===
slug: feature-stores
title: "Feature Stores and Training–Serving Consistency"
category: mlops
level: Advanced
tags: feature store, training-serving skew, point-in-time, feast, feature engineering
summary: Feature stores manage features as shared, versioned assets available offline for training and online for serving. We explain training–serving skew, point-in-time correctness, online vs offline stores, and when a feature store is worth it.
---
Imagine a feature "number of assistance requests by this household in the last 90 days". The data scientist computes it in a notebook with SQL for training. Months later, an engineer re-implements it in the serving system — with a subtly different time window, time zone or treatment of cancelled requests. The model now receives different inputs in production than in training, and its accuracy quietly drops. This **training–serving skew** is one of the most common and damaging bugs in production ML. **Feature stores** exist largely to prevent it.

## What is a feature store?

A feature store is a data system that manages ML features as **reusable, documented, versioned assets**, with:

1. **Feature definitions** written once (transformations, entity keys, time windows).
2. An **offline store** holding historical feature values for training — large, append-only, often in a data warehouse or lakehouse.
3. An **online store** holding the **latest** feature values for low-latency serving — a key–value database (e.g. Redis, DynamoDB, Bigtable).
4. **Materialisation** jobs that compute features and keep both stores in sync.
5. **Point-in-time correct** retrieval for building training sets.
6. A **registry** with metadata, ownership, documentation and lineage.

Examples: Feast (open source), Hopsworks, Tecton, and feature stores in cloud ML platforms (Vertex AI, SageMaker, Databricks).

## Point-in-time correctness

When building a training set of labelled events (e.g. "was this household later found to be in urgent need?" at time $t$), each row must use feature values **as they were known at time $t$** — not later values. Otherwise future information leaks into training, inflating offline metrics that the production model cannot reproduce.

A naive join of the event table with the latest feature table is **wrong**. A correct "as-of" join selects, for each event, the most recent feature value with timestamp $\le$ the event time:

```python
import pandas as pd

events = pd.DataFrame({
    "household_id": ["H1", "H1", "H2"],
    "event_time": pd.to_datetime(["2025-03-01", "2025-06-01", "2025-04-15"]),
    "label": [0, 1, 0]})
features = pd.DataFrame({
    "household_id": ["H1", "H1", "H1", "H2"],
    "feature_time": pd.to_datetime(["2025-02-01", "2025-05-01", "2025-07-01", "2025-04-01"]),
    "requests_90d": [1, 4, 9, 2]})

training = pd.merge_asof(events.sort_values("event_time"), features.sort_values("feature_time"),
                         left_on="event_time", right_on="feature_time", by="household_id",
                         direction="backward")                 # only values known at event time
print(training[["household_id", "event_time", "requests_90d", "label"]])
# H1 on 2025-06-01 gets 4 (from May), NOT 9 (from July, which would be leakage)
```

Feature stores implement this at scale, including features with their own processing delays (a value computed on the 1st may only be **available** on the 3rd).

## A Feast example

```python
# feature_repo/features.py
from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Int64, Float32

household = Entity(name="household", join_keys=["household_id"])
source = FileSource(path="data/household_features.parquet", timestamp_field="feature_time")

household_stats = FeatureView(
    name="household_stats",
    entities=[household],
    ttl=timedelta(days=120),
    schema=[Field(name="requests_90d", dtype=Int64), Field(name="avg_income_6m", dtype=Float32)],
    source=source,
)
```

```python
from feast import FeatureStore
store = FeatureStore(repo_path="feature_repo")
# Offline: point-in-time correct training data
train_df = store.get_historical_features(entity_df=events.rename(columns={"event_time": "event_timestamp"}),
                                         features=["household_stats:requests_90d",
                                                   "household_stats:avg_income_6m"]).to_df()
# Online: latest values at prediction time (after `feast materialize`)
online = store.get_online_features(features=["household_stats:requests_90d"],
                                   entity_rows=[{"household_id": "H1"}]).to_dict()
```

The **same definitions** feed both training and serving — eliminating skew by construction.

## Benefits

- **Consistency** between training and serving.
- **Reuse** across teams and models; less duplicated feature engineering.
- **Discoverability**: a catalogue of documented features with owners.
- **Governance**: access control on sensitive features; lineage for audits.
- **Low-latency serving** of precomputed aggregates.

## When is a feature store worth it?

Feature stores add infrastructure and operational cost. They pay off when you have **many models** sharing features, **real-time** serving needs with complex aggregates, multiple teams, and recurring skew or leakage bugs. A single batch model can often achieve consistency more simply: a shared Python feature module used by both training and batch scoring, plus point-in-time joins in SQL.

:::tip
Whatever your tooling, follow the principle: **define each feature once, in code, with explicit timestamps and availability delays, and use that single definition everywhere.**
:::

:::exercise
1. Create a dataset where a naive join leaks future information, train a model both ways, and compare offline accuracy with realistic "production" accuracy.
2. Implement the same aggregate feature (requests in the last 30 days) in batch SQL and in a real-time Python function, and write a test proving they agree.
3. Set up a minimal Feast repository with one entity and one feature view, and retrieve both historical and online features.
:::

:::takeaway
- Training–serving skew arises when features are computed differently in training and production.
- Feature stores define features once and serve them from an offline store (training) and an online store (serving).
- Point-in-time correct joins use only feature values available at each event's time, preventing leakage.
- Feature stores suit many-model, real-time, multi-team settings; simpler shared code can suffice otherwise.
:::
