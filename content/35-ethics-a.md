=== POST ===
slug: why-ai-ethics-matters
title: "Why AI Ethics Matters: Principles for Responsible Engineers"
category: ethics
level: Beginner
tags: ai ethics, responsible ai, principles, accountability, harms
summary: AI systems make or shape decisions about people at scale. We examine real harms, the major principles of responsible AI, why good intentions are not enough, and how ethics becomes concrete engineering practice.
---
Welcome to the final track. Everything you have learned — from gradient descent to transformers — is powerful precisely because it scales: one model can make millions of decisions. That power is why ethics is not an optional appendix to AI education. A biased model does not discriminate once; it discriminates systematically, at scale, often invisibly. As engineers, we are responsible not only for whether our systems work, but for **whom** they work, **how**, and **with what consequences**.

## Real harms, not hypotheticals

Documented cases include:

- **Criminal justice**: ProPublica's 2016 analysis of the COMPAS recidivism tool reported that Black defendants who did not reoffend were almost twice as likely as white defendants to be labelled high risk — igniting a debate about fairness definitions that continues today.
- **Hiring**: Amazon reportedly abandoned an experimental résumé-screening tool after finding it penalised résumés containing the word "women's", having learned from historically male-dominated hiring data.
- **Healthcare**: Obermeyer et al. (2019, *Science*) found that a widely used algorithm for allocating extra care used past **health costs** as a proxy for health **needs**; because less money had historically been spent on Black patients with the same needs, the algorithm substantially under-identified Black patients who needed extra care.
- **Face recognition**: Buolamwini and Gebru's *Gender Shades* (2018) found commercial gender classifiers had far higher error rates for darker-skinned women; wrongful arrests based on facial-recognition matches have been reported.
- **Public services**: the Dutch childcare-benefits scandal involved risk-scoring and profiling practices that wrongly accused thousands of families of fraud, with devastating consequences, contributing to the government's resignation in 2021.

In each case, the technology "worked" by some metric — and still caused serious harm.

## Where harms come from

Harms can enter at every stage of the ML lifecycle (Suresh & Guttag, 2021):

1. **Problem formulation**: choosing a proxy target (cost instead of need; arrests instead of crime).
2. **Historical bias**: data reflecting past discrimination.
3. **Representation bias**: some groups under-represented in data.
4. **Measurement bias**: features or labels measured differently across groups.
5. **Aggregation bias**: one model for groups with different relationships.
6. **Evaluation bias**: benchmarks not representative of the deployment population.
7. **Deployment bias**: using a system in contexts it was not designed for, or in ways that shift power.

## Principles of responsible AI

Hundreds of AI ethics guidelines have been published; analyses (e.g. Jobin, Ienca & Vayena, 2019) found convergence around a few principles:

| Principle | Meaning in practice |
|---|---|
| Fairness / non-discrimination | Equitable performance and outcomes across groups |
| Transparency / explainability | People can understand how systems work and why decisions were made |
| Accountability | Clear responsibility; ability to contest and seek redress |
| Privacy | Data minimisation, consent, security, purpose limitation |
| Safety / robustness | Reliable behaviour, resistance to misuse and attack |
| Beneficence / human wellbeing | Systems should genuinely benefit people |
| Human autonomy / oversight | Humans remain in meaningful control of consequential decisions |

Humanitarian and development work adds principles such as **"do no harm"**, humanity, impartiality, and particular care for the **dignity and protection** of people in vulnerable situations.

## From principles to practice

Principles are easy to endorse and hard to implement. Concrete practices include:

- **Stakeholder engagement**: involve affected communities and domain experts from problem framing onward.
- **Impact assessments** before building and before deployment.
- **Data documentation** (datasheets) and **model documentation** (model cards) with disaggregated evaluation.
- **Fairness testing** across relevant groups.
- **Explanations** and **contestability** — ways for people to question and appeal decisions.
- **Human oversight** with real authority and time to exercise judgement (not rubber-stamping).
- **Monitoring** after deployment, including for harms.
- **The option not to build**: sometimes the responsible choice is that a problem should not be automated.

:::note
Ask four questions about any AI system: **Who benefits? Who bears the risks? Who decides? Who can say no?** If the people bearing the risks have no voice in the decision and no way to contest outcomes, the system has an ethics problem — however accurate it is.
:::

## Ethics is part of engineering quality

A model that is 95% accurate overall but 70% accurate for a minority language, or that can be easily manipulated, or that leaks personal data, is a **defective product**. Responsible AI is not in tension with good engineering; it is a dimension of it. The rest of this track gives you tools: fairness metrics, explainability, privacy techniques, safety, regulation, and career guidance for building AI that serves people well.

:::exercise
1. Choose one of the documented cases above. Identify at which lifecycle stages harm entered and what practices could have prevented it.
2. Write the four-question analysis (who benefits, bears risks, decides, can say no) for an AI system you use.
3. Draft three concrete engineering requirements that would implement "accountability" for an AI system in your field.
:::

:::takeaway
- AI harms are documented and systematic, often arising from proxies, biased data and deployment context.
- Harms enter at every lifecycle stage, from problem formulation to deployment.
- Responsible AI principles — fairness, transparency, accountability, privacy, safety, human oversight — must be made concrete.
- Ethics is a dimension of engineering quality; sometimes the right decision is not to automate.
:::

=== POST ===
slug: bias-and-fairness-in-ml
title: "Bias and Fairness in Machine Learning"
category: ethics
level: Intermediate
tags: fairness, bias, discrimination, protected attributes, proxies
summary: What does it mean for a model to be fair, and where does unfairness come from? We examine sources of bias, the failure of "fairness through unawareness", proxy variables, and practical strategies across the ML pipeline.
---
"Our model does not use race or gender, so it cannot be biased." This is one of the most common — and most mistaken — beliefs in applied ML. Bias can arise from data, labels, features, objectives and deployment, even when sensitive attributes are excluded. This lecture explains how unfairness arises and what we can do about it; the next lecture formalises fairness metrics.

## Sources of bias in data

- **Historical bias**: data faithfully records an unjust world. A model trained on past lending or hiring decisions learns past discrimination.
- **Representation (sampling) bias**: some groups are under-represented. Many image datasets over-represent certain regions; many speech datasets under-represent some accents and languages.
- **Measurement bias**: features or labels are measured differently across groups. Arrest records measure policing intensity, not just crime; "customer satisfaction" surveys may have different response rates by group.
- **Label bias**: human annotators' judgements (e.g. of toxicity or "professionalism") can encode stereotypes — studies found toxicity classifiers flagging text in African-American English as offensive more often.

## Sources of bias in modelling

- **Objective choice**: minimising overall error favours the majority group; small groups contribute little to the average loss.
- **Aggregation bias**: one model for groups whose feature–outcome relationships differ (e.g. a clinical risk score calibrated on one population applied to another).
- **Feedback loops**: predictive policing sends more patrols where more crime was recorded, which records more crime there, reinforcing the pattern.

## Why "fairness through unawareness" fails

Removing a protected attribute (gender, ethnicity, religion, disability) does not remove its influence, because other features act as **proxies**: postcode, name, language, school, shopping patterns, device type. A model can reconstruct protected attributes from such correlations. Worse, dropping the attribute makes it **harder to measure** and correct unfairness.

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

rng = np.random.default_rng(0)
n = 5000
group = rng.integers(0, 2, n)                               # protected attribute (not given to model)
neighbourhood = np.where(rng.random(n) < 0.85, group, 1 - group)   # strongly correlated proxy
income = rng.normal(50 - 8 * group, 10, n)                  # historical inequality in income
X_without_group = np.column_stack([neighbourhood, income])

# Can the "unaware" features predict the protected attribute?
auc = cross_val_score(LogisticRegression(), X_without_group, group, cv=5, scoring="roc_auc").mean()
print(f"protected attribute recoverable from proxies with AUC = {auc:.2f}")
```

An AUC far above 0.5 means the model can effectively "see" the protected attribute anyway.

## Strategies across the pipeline

### Before training (pre-processing)
- Collect **more representative data**; involve communities in data collection.
- Audit labels for bias; improve annotation guidelines and annotator diversity.
- **Reweighting** or resampling to balance groups; transformations that reduce dependence on protected attributes (e.g. learning fair representations).
- Examine whether the **target** is a biased proxy (cost vs need).

### During training (in-processing)
- Add **fairness constraints** or penalties to the objective (e.g. reductions approach of Agarwal et al., 2018, implemented in Fairlearn).
- **Adversarial debiasing**: train the model so an adversary cannot predict the protected attribute from its representation or predictions.
- **Group-robust optimisation**: minimise the worst-group loss (e.g. Group DRO) rather than the average.

### After training (post-processing)
- **Group-specific thresholds** to equalise chosen error rates (Hardt et al., 2016) — effective, but may be legally sensitive in some jurisdictions.
- **Calibration** per group.

### At deployment
- Monitor **outcomes by group** over time.
- Provide explanations, appeals and human review.
- Limit use to validated contexts.

## Fairness requires context

:::note
There is no purely technical definition of fairness that fits every situation. Whether equal error rates, equal selection rates or equal calibration matter most depends on the **decision**, the **harms of each error type**, the **legal framework** and the **values of affected people**. Technical tools measure and enforce a chosen notion; choosing the notion is a normative, participatory decision.
:::

## Intersectionality

Bias often concentrates at intersections — e.g. older women from a minority language group — which single-attribute analyses miss (Crenshaw's concept of intersectionality, applied to AI by Buolamwini and Gebru). Evaluate intersections where sample sizes allow, and report uncertainty where they do not.

:::exercise
1. Using the proxy example, train a model to predict an outcome that depends on income, with and without the protected attribute; compare selection rates by group.
2. Audit a public dataset you have used for representation: which groups, regions or languages are under-represented?
3. For a model in your domain, write down which errors (false positives, false negatives) harm whom, and which fairness goal you would prioritise and why.
:::

:::takeaway
- Bias arises from historical, representation, measurement and label bias, and from objectives, aggregation and feedback loops.
- Removing protected attributes fails because proxies encode them — and it hinders measurement.
- Mitigation spans pre-processing, in-processing, post-processing and deployment monitoring.
- Choosing a fairness notion is contextual and normative; examine intersections.
:::

=== POST ===
slug: fairness-metrics
title: "Fairness Metrics: Demographic Parity, Equalised Odds and Calibration"
category: ethics
level: Advanced
tags: fairness metrics, demographic parity, equalized odds, calibration, impossibility
summary: We formalise group fairness — demographic parity, equal opportunity, equalised odds, predictive parity and calibration — compute them with Fairlearn, and prove why several cannot hold simultaneously except in special cases.
---
To manage fairness, we must measure it. Researchers have proposed many formal definitions, most of which compare statistics of a model's predictions across groups defined by a protected attribute $A$. Understanding these definitions — and the tensions between them — is essential for anyone evaluating models that affect people.

## Notation

- $Y \in \{0, 1\}$: true outcome (e.g. 1 = needs urgent support).
- $\hat{Y} \in \{0, 1\}$: model decision; $S$: model score.
- $A$: group membership (e.g. $a$ and $b$).

## Independence: demographic (statistical) parity

$$
P(\hat{Y} = 1 \mid A = a) = P(\hat{Y} = 1 \mid A = b)
$$

Equal **selection rates** across groups. The **disparate impact ratio** $\frac{P(\hat{Y}=1 \mid A=a)}{P(\hat{Y}=1 \mid A=b)}$ is often compared with the US "four-fifths rule" (ratio below 0.8 flags possible adverse impact in employment contexts).

Appropriate when the outcome variable is itself biased or when equal allocation is a goal. Problematic when base rates genuinely differ — forcing equal selection can mean accepting less qualified candidates in one group or rejecting needy people in another.

## Separation: error-rate parity

**Equal opportunity** (Hardt, Price & Srebro, 2016): equal **true positive rates** —

$$
P(\hat{Y} = 1 \mid Y = 1, A = a) = P(\hat{Y} = 1 \mid Y = 1, A = b)
$$

Those who truly need support have the same chance of being identified, whatever their group.

**Equalised odds**: equal true positive rates **and** equal false positive rates:

$$
P(\hat{Y} = 1 \mid Y = y, A = a) = P(\hat{Y} = 1 \mid Y = y, A = b), \quad y \in \{0, 1\}
$$

Appropriate when labels are trustworthy and the harms are the errors themselves.

## Sufficiency: predictive parity and calibration

**Predictive parity**: equal precision (positive predictive value) —

$$
P(Y = 1 \mid \hat{Y} = 1, A = a) = P(Y = 1 \mid \hat{Y} = 1, A = b)
$$

**Calibration within groups**: for every score $s$,

$$
P(Y = 1 \mid S = s, A = a) = P(Y = 1 \mid S = s, A = b) = s
$$

A score of 0.7 means a 70% chance for everyone. Important when scores are communicated as risks and used by humans.

## The impossibility results

Kleinberg, Mullainathan and Raghavan (2016) and Chouldechova (2017) proved that, when **base rates differ** between groups ($P(Y = 1 \mid A = a) \ne P(Y = 1 \mid A = b)$), a classifier generally **cannot** simultaneously satisfy calibration (or predictive parity) **and** equal false positive **and** false negative rates — except in degenerate cases (perfect prediction).

Chouldechova's identity makes the tension concrete. For each group with prevalence $p$:

$$
\text{FPR} = \frac{p}{1 - p}\cdot\frac{1 - \text{PPV}}{\text{PPV}}\cdot(1 - \text{FNR})
$$

If PPV and FNR are equal across groups but prevalences $p$ differ, the FPRs **must** differ. This is exactly the COMPAS controversy: the tool's developers pointed to predictive parity; ProPublica pointed to unequal false positive rates. Both were measuring real properties; they could not both be equalised.

:::note
The impossibility theorems do not say fairness is hopeless. They say you must **choose** which property matters most for a specific decision, justify that choice with the stakeholders, and be transparent about the trade-offs — including the option of changing the decision process itself (e.g. adding human review, collecting better data, or not automating).
:::

## Computing fairness metrics with Fairlearn

```python
import numpy as np
from fairlearn.metrics import (MetricFrame, selection_rate, true_positive_rate, false_positive_rate,
                               demographic_parity_difference, equalized_odds_difference)
from sklearn.metrics import precision_score

rng = np.random.default_rng(1)
n = 4000
A = rng.choice(["group_a", "group_b"], n, p=[0.7, 0.3])
y = (rng.random(n) < np.where(A == "group_a", 0.30, 0.20)).astype(int)     # different base rates
score = np.clip(0.5 * y + rng.normal(0.25, 0.2, n) + np.where(A == "group_b", -0.05, 0), 0, 1)
y_pred = (score >= 0.5).astype(int)

mf = MetricFrame(metrics={"selection_rate": selection_rate, "TPR": true_positive_rate,
                          "FPR": false_positive_rate, "precision": precision_score},
                 y_true=y, y_pred=y_pred, sensitive_features=A)
print(mf.by_group.round(3))
print("demographic parity difference:", round(demographic_parity_difference(y, y_pred, sensitive_features=A), 3))
print("equalized odds difference:   ", round(equalized_odds_difference(y, y_pred, sensitive_features=A), 3))
```

Fairlearn also provides mitigation algorithms (`ExponentiatedGradient` with constraints such as `EqualizedOdds`, and `ThresholdOptimizer`); AIF360 is another comprehensive toolkit.

## Beyond group metrics

- **Individual fairness** (Dwork et al., 2012): similar individuals should be treated similarly — requires a task-appropriate similarity measure.
- **Counterfactual fairness** (Kusner et al., 2017): a decision should not change if the protected attribute had been different, with causally downstream features changed accordingly — requires a causal model.
- **Uncertainty**: with small groups, metric differences may be noise — report confidence intervals (bootstrap).

:::exercise
1. Using the code, compute all metrics, then apply `ThresholdOptimizer` with an equalised-odds constraint. What happens to overall accuracy and to precision by group?
2. Verify Chouldechova's identity numerically on the simulated data for each group.
3. For a decision in your field (e.g. prioritising home visits), argue which fairness metric should take priority and what trade-offs you accept.
:::

:::takeaway
- Independence (demographic parity), separation (equal opportunity/equalised odds) and sufficiency (predictive parity/calibration) are the main group-fairness families.
- With different base rates, calibration/predictive parity and equal error rates cannot all hold — a proven impossibility.
- Fairness metric choice is contextual and must be justified with stakeholders.
- Use toolkits like Fairlearn, report uncertainty, and consider individual and counterfactual notions.
:::

=== POST ===
slug: explainable-ai-lime-shap
title: "Explainable AI: LIME, SHAP and Interpretable Models"
category: ethics
level: Intermediate
tags: explainability, xai, shap, lime, interpretability, feature importance
summary: Why did the model decide that? We distinguish interpretable models from post-hoc explanations, derive Shapley values and SHAP, explain LIME, cover global vs local explanations and counterfactuals, and discuss the limits of explanations.
---
A caseworker is told that a household's priority score is low. A doctor sees a model flag an X-ray. A loan applicant is rejected. Each reasonably asks: **why?** Explanations help people trust (or appropriately distrust) models, detect errors and bias, comply with regulations that require meaningful information about automated decisions, and contest outcomes. **Explainable AI (XAI)** provides tools for this — with important limitations.

## Interpretable by design vs post-hoc explanation

- **Intrinsically interpretable models**: linear/logistic regression with few features, small decision trees, rule lists, generalised additive models (GAMs, e.g. Explainable Boosting Machines). Their structure *is* the explanation.
- **Post-hoc explanations**: methods that explain a trained black-box model (gradient boosting, neural networks) from the outside.

Cynthia Rudin (2019) argued forcefully that for **high-stakes decisions**, we should use interpretable models whenever they perform comparably — which, for many tabular problems, they do — rather than explaining black boxes with approximations that may be unfaithful.

## Global vs local explanations

- **Global**: how does the model behave overall? (Which features matter most? What is the shape of the relationship?)
- **Local**: why did the model make **this** prediction for **this** case?

Global tools: permutation importance, partial dependence plots (PDP), accumulated local effects (ALE), global SHAP summaries. Local tools: SHAP values, LIME, counterfactual explanations.

## Shapley values and SHAP

From cooperative game theory (Shapley, 1953): how should a "payout" (the prediction) be fairly divided among "players" (the features)? The Shapley value of feature $i$ is its average marginal contribution over all possible orders of adding features:

$$
\phi_i = \sum_{S \subseteq F \setminus \{i\}}\frac{|S|!\,(|F| - |S| - 1)!}{|F|!}\Big[v(S \cup \{i\}) - v(S)\Big]
$$

where $v(S)$ is the model's expected output when only features in $S$ are known. Shapley values uniquely satisfy desirable axioms, including **efficiency** (local accuracy):

$$
f(\mathbf{x}) = \phi_0 + \sum_{i=1}^{M}\phi_i
$$

— the prediction equals the baseline (average prediction) plus the sum of feature contributions.

**SHAP** (Lundberg & Lee, 2017) made Shapley values practical: **TreeSHAP** computes exact values for tree ensembles in polynomial time; **KernelSHAP** approximates them for any model; DeepSHAP and GradientSHAP for neural networks.

```python
import shap
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import HistGradientBoostingRegressor

X, y = fetch_california_housing(return_X_y=True, as_frame=True)
model = HistGradientBoostingRegressor(random_state=0).fit(X, y)

explainer = shap.Explainer(model, X.sample(200, random_state=0))   # background data defines the baseline
sv = explainer(X.iloc[:500])
shap.plots.beeswarm(sv)                   # global: feature impact across many predictions
shap.plots.waterfall(sv[0])               # local: why this particular prediction?
print("baseline + contributions = prediction:",
      round(sv[0].base_values + sv[0].values.sum(), 4), round(model.predict(X.iloc[[0]])[0], 4))
```

## LIME

**LIME** (Ribeiro, Singh & Guestrin, 2016) explains a single prediction by fitting a simple, interpretable **surrogate model** (e.g. sparse linear regression) on perturbed samples around the instance, weighted by proximity. For text, it removes words; for images, it hides superpixels. It is model-agnostic and intuitive, but explanations can be **unstable** (different runs or kernel widths give different explanations) and depend on how "neighbourhood" is defined.

## Counterfactual explanations

"Your application would have been prioritised if your household had two more dependants" — a **counterfactual explanation** (Wachter, Mittelstadt & Russell, 2017) states the smallest change that would alter the outcome. They are actionable and intuitive, but must respect **feasibility** (you cannot change your age) and can reveal how to game a system; tools like DiCE generate diverse, constrained counterfactuals.

## Limits and pitfalls

:::warning
- **Explanations are models of models**: post-hoc methods approximate behaviour and can be unfaithful; different methods can disagree on the same prediction.
- **Correlated features**: attributions split or shift unpredictably between correlated features; interventional vs observational SHAP answer different questions.
- **Not causal**: a high SHAP value for "postcode" says the model relies on it, not that postcode causes the outcome.
- **Manipulable**: studies showed models can be engineered to hide biased behaviour from LIME and SHAP (Slack et al., 2020).
- **Explanation ≠ justification**: a clear explanation of an unfair decision is still an unfair decision.
- **Audience matters**: data scientists, caseworkers and affected people need different explanations.
:::

## Explanations for people affected

Legal frameworks (e.g. GDPR provisions on automated decision-making) and good practice call for **meaningful information** about the logic involved and ways to contest decisions. Effective explanations for affected people are short, in plain language (and their own language), focused on the main reasons and on what they can do — with a clear route to a human reviewer.

:::exercise
1. Train an Explainable Boosting Machine (`interpret` package) and a gradient-boosting model on the same tabular data; compare accuracy and inspect the EBM's shape functions.
2. Compute SHAP and LIME explanations for the same five predictions. Where do they disagree?
3. Write a plain-language explanation template for a priority score, suitable for a caseworker to share with a family.
:::

:::takeaway
- Prefer interpretable models for high-stakes tabular decisions when they perform comparably.
- SHAP attributes predictions with Shapley values, satisfying local accuracy; TreeSHAP is exact for tree ensembles.
- LIME fits local surrogate models; counterfactuals show actionable changes.
- Explanations can be unfaithful, unstable, non-causal and manipulable — tailor them to the audience and keep contestability.
:::

=== POST ===
slug: privacy-and-differential-privacy
title: "Privacy in Machine Learning and Differential Privacy"
category: ethics
level: Advanced
tags: privacy, differential privacy, dp-sgd, membership inference, anonymization
summary: Models can leak the data they were trained on. We examine re-identification and model attacks, why anonymisation often fails, the mathematics of differential privacy, DP-SGD, and practical privacy-by-design.
---
Machine learning feeds on data, and much valuable data is personal: health records, locations, financial information, case files of displaced people. Privacy failures can expose people to discrimination, persecution, fraud or violence. Protecting privacy requires more than removing names — models themselves can leak their training data. This lecture covers the threats and the strongest formal defence we have: **differential privacy**.

## Why "anonymised" data often is not

- **Linkage attacks**: Latanya Sweeney showed that ZIP code, birth date and sex alone uniquely identify a large majority of Americans, and re-identified a governor's medical records by linking "anonymised" hospital data with a public voter list.
- The **Netflix Prize** dataset was de-anonymised by linking movie ratings with public IMDb reviews (Narayanan & Shmatikov, 2008).
- Location traces are highly unique: a few spatio-temporal points can identify most individuals in mobile-phone datasets (de Montjoye et al., 2013).

**k-anonymity** and related techniques (generalising or suppressing quasi-identifiers) help but remain vulnerable to background knowledge and composition of releases.

## Attacks on models

- **Membership inference** (Shokri et al., 2017): determine whether a specific person's record was in the training set — itself sensitive (e.g. membership in a dataset of patients with a particular disease). Overfit models are especially vulnerable.
- **Model inversion / attribute inference**: reconstruct sensitive attributes or representative inputs from a model.
- **Training data extraction**: large language models have been shown to regurgitate memorised training text verbatim, including personal information (Carlini et al., 2021).
- **Leakage through features and embeddings**.

## Differential privacy (DP)

Dwork, McSherry, Nissim and Smith (2006) proposed a rigorous definition. A randomised algorithm $M$ is **$(\varepsilon, \delta)$-differentially private** if for all neighbouring datasets $D$ and $D'$ differing in one individual's data, and all sets of outputs $S$:

$$
P[M(D) \in S] \le e^{\varepsilon}\,P[M(D') \in S] + \delta
$$

Intuition: the output is almost equally likely whether or not any single person participated, so an observer learns very little about any individual. Smaller $\varepsilon$ means stronger privacy; $\delta$ is a small failure probability (much smaller than $1/n$).

Key properties:

- **Robust to auxiliary information**: guarantees hold whatever the attacker knows.
- **Composition**: running several DP analyses adds up privacy loss (the "privacy budget").
- **Post-processing**: anything computed from a DP output remains DP.

## The Laplace mechanism

To release a numeric query $f(D)$ (e.g. a count) with **sensitivity** $\Delta f$ (the maximum change from one person), add Laplace noise:

$$
M(D) = f(D) + \text{Lap}\!\left(\frac{\Delta f}{\varepsilon}\right)
$$

```python
import numpy as np

def dp_count(values, predicate, epsilon, rng=np.random.default_rng()):
    true = sum(predicate(v) for v in values)
    return true + rng.laplace(0, 1.0 / epsilon)          # sensitivity of a count is 1

ages = np.random.default_rng(0).integers(0, 90, 10_000)
for eps in [0.1, 1.0, 10.0]:
    print(f"epsilon={eps:>4}: noisy count of children under 5 = {dp_count(ages, lambda a: a < 5, eps):.1f}")
print("true count:", int((ages < 5).sum()))
```

Large aggregate counts remain accurate while individual contributions are masked. National statistics offices (e.g. the US Census Bureau for the 2020 Census) have adopted DP for official data releases.

## DP-SGD: training models with differential privacy

Abadi et al. (2016) made deep learning differentially private:

1. Compute **per-example gradients**.
2. **Clip** each gradient to a maximum norm $C$ (bounding any individual's influence).
3. Add **Gaussian noise** with standard deviation $\sigma C$ to the sum.
4. Track cumulative privacy loss with an accountant (moments accountant / Rényi DP).

$$
\tilde{\mathbf{g}} = \frac{1}{B}\left(\sum_{i \in \mathcal{B}}\text{clip}(\mathbf{g}_i, C) + \mathcal{N}(0, \sigma^2C^2\mathbf{I})\right)
$$

Libraries: **Opacus** (PyTorch), TensorFlow Privacy. Costs: lower accuracy (especially for small datasets and minority groups), slower training, careful hyperparameter tuning. Pretraining on public data and fine-tuning privately helps considerably.

```python
# Sketch with Opacus
from opacus import PrivacyEngine
# model, optimizer, train_loader defined as usual
privacy_engine = PrivacyEngine()
# model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
#     module=model, optimizer=optimizer, data_loader=train_loader,
#     target_epsilon=3.0, target_delta=1e-5, epochs=10, max_grad_norm=1.0)
# ... train normally; privacy_engine.get_epsilon(1e-5) reports the budget spent
```

:::note
Differential privacy can **worsen accuracy disproportionately for under-represented groups** (their signal is small relative to the added noise). Privacy and fairness must be evaluated together.
:::

## Privacy by design in practice

1. **Data minimisation**: collect only what you need; delete when no longer needed.
2. **Purpose limitation** and a clear legal basis (e.g. under GDPR or national data-protection laws); informed consent where applicable.
3. **Pseudonymisation**, access control, encryption at rest and in transit, audit logs.
4. **Aggregation and DP** for published statistics and dashboards.
5. **Federated learning** or on-device processing to keep raw data local (next lecture).
6. **Privacy impact assessments** before new processing.
7. **Test models for memorisation** (membership inference, canary insertion) before release.

:::warning
For displaced and vulnerable populations, data can be dangerous: information about identity, ethnicity, religion or location, if leaked or shared, could expose people to persecution. Humanitarian data-protection guidance (e.g. from the ICRC and UN agencies) emphasises data minimisation, careful sharing agreements and assessing whether collecting data is necessary at all. "Can we build it?" must follow "should we collect it?".
:::

:::exercise
1. Run a simple membership-inference attack: compare a model's confidence on training vs held-out examples for an overfit and a well-regularised model.
2. Release a histogram of ages with the Laplace mechanism at ε = 0.5, 1 and 5, and measure the error per bin.
3. Train a small classifier with Opacus at ε ≈ 1, 3 and 8; plot accuracy versus ε, overall and for a minority subgroup.
:::

:::takeaway
- Removing names is not anonymisation: linkage and model attacks (membership inference, extraction) can reveal individuals.
- Differential privacy bounds how much any individual's data can affect outputs, via the parameters $(\varepsilon, \delta)$.
- The Laplace mechanism privatises statistics; DP-SGD clips per-example gradients and adds noise.
- Privacy by design — minimisation, security, DP, federated learning, impact assessments — is essential, especially for vulnerable populations.
:::

=== POST ===
slug: federated-learning
title: "Federated Learning: Training Without Centralising Data"
category: ethics
level: Advanced
tags: federated learning, privacy, fedavg, decentralized, secure aggregation
summary: Federated learning trains a shared model across many devices or institutions while raw data stays local. We derive FedAvg, discuss non-IID data, communication costs, secure aggregation, privacy limits and real applications.
---
Hospitals want to build better diagnostic models, but cannot share patient records. Phones could improve keyboard predictions, but users' messages should never leave their devices. Organisations in different countries hold data under different legal regimes. **Federated learning (FL)** offers a way to train a shared model **without collecting raw data in one place**: the model travels to the data, not the other way round.

## The setting

- A **server** coordinates training of a global model.
- Many **clients** (phones, hospitals, organisations) each hold private local data.
- Clients train locally and send **model updates** (weights or gradients), not data, back to the server.

**Cross-device FL**: millions of phones, each with little data, intermittently available (e.g. Google's Gboard next-word prediction, one of the first large deployments). **Cross-silo FL**: a handful of institutions (hospitals, banks) with large datasets and reliable connections.

## FedAvg

McMahan et al. (2017) proposed **Federated Averaging**, the standard algorithm:

1. The server sends the current global weights $\mathbf{w}^t$ to a sample of clients.
2. Each selected client $k$ runs several epochs of SGD on its local data, producing $\mathbf{w}_k^{t+1}$.
3. The server averages the updates, weighted by client dataset size $n_k$:

$$
\mathbf{w}^{t+1} = \sum_{k \in S_t}\frac{n_k}{\sum_{j \in S_t}n_j}\,\mathbf{w}_k^{t+1}
$$

4. Repeat for many rounds.

Doing multiple local steps before communicating dramatically reduces communication compared with sending every gradient.

```python
import copy
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

def local_train(model, X, y, epochs=2, lr=0.1):
    model = copy.deepcopy(model); opt = torch.optim.SGD(model.parameters(), lr=lr)
    for _ in range(epochs):
        for i in range(0, len(X), 32):
            loss = F.cross_entropy(model(X[i:i + 32]), y[i:i + 32])
            opt.zero_grad(); loss.backward(); opt.step()
    return model.state_dict(), len(X)

def fedavg(states_sizes):
    total = sum(n for _, n in states_sizes)
    return {k: sum(s[k] * (n / total) for s, n in states_sizes) for k in states_sizes[0][0]}

torch.manual_seed(0); rng = np.random.default_rng(0)
global_model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 2))
# Five clients with non-IID data: each has a different class balance
clients = []
for k in range(5):
    X = torch.randn(400, 10); w_true = torch.ones(10)
    y = ((X @ w_true + rng.normal(0, 1, 400).astype(np.float32) + (k - 2)) > 0).long()
    clients.append((X, y))
for rnd in range(20):
    chosen = rng.choice(5, 3, replace=False)                     # partial participation
    updates = [local_train(global_model, *clients[k]) for k in chosen]
    global_model.load_state_dict(fedavg(updates))
Xt = torch.cat([c[0] for c in clients]); yt = torch.cat([c[1] for c in clients])
print("global accuracy:", (global_model(Xt).argmax(1) == yt).float().mean().item())
```

## Challenges

1. **Non-IID data**: clients' data distributions differ (different languages, regions, patient populations). Local models drift apart ("client drift"), slowing or degrading convergence. Remedies: FedProx (a proximal term keeping local models near the global model), SCAFFOLD (control variates), and **personalisation** (fine-tuning the global model per client, or mixing global and local components).
2. **Communication cost**: models can be large; networks slow. Use fewer rounds with more local work, update compression (quantisation, sparsification), and smaller models.
3. **Systems heterogeneity**: devices differ in compute, battery and connectivity; clients drop out.
4. **Fairness**: the global model may serve large or typical clients well and small or unusual ones poorly; fairness-aware aggregation can help.
5. **Robustness**: malicious clients can send poisoned updates (backdoors); robust aggregation (median, trimmed mean, Krum) mitigates some attacks.

## Privacy: FL is not automatically private

:::warning
Keeping data local is a significant improvement, but model updates can still **leak information**: research has shown that training data (even images) can sometimes be reconstructed from shared gradients ("gradient inversion"), and membership can be inferred. Strong privacy requires combining FL with:
- **Secure aggregation** (Bonawitz et al., 2017): cryptographic protocols so the server sees only the **sum** of client updates, not individual updates;
- **Differential privacy**: clip client updates and add noise (at the client — local DP — or at the server — central DP);
- secure infrastructure and governance agreements among participants.
:::

## Applications

- **Mobile**: keyboard prediction, voice-assistant improvements, on-device personalisation.
- **Healthcare**: multi-hospital models for medical imaging and outcome prediction; studies such as a multi-institution COVID-19 outcome model (EXAM, 2021) trained across many hospitals worldwide.
- **Finance**: fraud detection across institutions.
- **Cross-organisational and cross-border collaboration**, where legal constraints prevent data pooling.

Frameworks: Flower, TensorFlow Federated, NVIDIA FLARE, PySyft, FedML.

:::exercise
1. Modify the FedAvg example so each client has only one class (extreme non-IID). How does global accuracy change? Implement FedProx and compare.
2. Add Gaussian noise to client updates (after clipping their norm) and plot accuracy against noise level.
3. Design a cross-silo federated learning setup for three organisations: what governance agreements, security measures and evaluation plan would you need?
:::

:::takeaway
- Federated learning trains a shared model while raw data stays with clients; FedAvg averages locally trained weights.
- Non-IID data, communication costs, heterogeneity, fairness and poisoning are core challenges.
- Updates can leak information — combine FL with secure aggregation and differential privacy.
- FL enables collaboration in mobile, healthcare and cross-organisational settings where data cannot be pooled.
:::
