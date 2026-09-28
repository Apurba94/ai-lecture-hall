=== POST ===
slug: ai-and-the-future-of-work
title: "AI and the Future of Work"
category: ethics
level: Beginner
tags: future of work, automation, labour, skills, economics
summary: Will AI take our jobs? We look at evidence on automation and augmentation, task-based analysis of exposure, early productivity studies of generative AI, distributional effects, and how individuals, organisations and policy can respond.
---
Every wave of automation has raised fears of mass unemployment — from the Luddites facing mechanised looms to economists debating computers in the 1960s. Generative AI renews the question with new force, because for the first time machines perform many **cognitive** and **creative** tasks previously done only by educated professionals. This lecture examines what we know, what we do not, and how to prepare.

## Tasks, not jobs

Economists find it more useful to analyse **tasks** than whole jobs (Autor, Levy and Murnane's task framework). A job is a bundle of tasks; technology automates some, augments others and creates new ones.

- **Automation**: AI performs a task instead of a person (transcribing audio, routing tickets).
- **Augmentation**: AI helps a person perform a task better or faster (drafting, coding assistance, decision support).
- **New tasks**: work that did not exist before (prompt design, AI evaluation and oversight, data annotation, AI governance).

Studies estimating exposure to large language models (e.g. Eloundou et al., 2023) suggest that a large share of workers in advanced economies have at least some tasks that LLMs could affect, with **higher exposure among higher-wage, higher-education occupations** — the reverse of earlier automation waves that mainly affected routine manual and clerical work. Exposure does not mean replacement: it indicates where work may change.

## Early evidence on productivity

Controlled studies of generative AI in real work have found meaningful gains on specific tasks:

- **Customer support**: a study of thousands of agents (Brynjolfsson, Li & Raymond, 2023) found an AI assistant increased resolutions per hour on average by around 14%, with the largest gains for **novice and lower-skilled** workers — the AI spread best practices.
- **Writing tasks**: an experiment (Noy & Zhang, 2023) found professionals completed writing tasks faster with higher rated quality when using ChatGPT.
- **Consulting tasks**: a field experiment with management consultants (Dell'Acqua et al., 2023) found large gains on tasks within the AI's capability — but **worse** performance on a task outside its "jagged frontier", where consultants over-relied on confidently wrong AI output.
- **Software development**: controlled experiments found faster completion of some coding tasks with AI assistants; effects in complex real-world settings vary.

Lessons: gains are real but uneven; AI can **narrow skill gaps** within tasks; and **knowing when not to trust AI** is itself a crucial skill.

## Distributional effects

Aggregate productivity gains do not guarantee broadly shared benefits. Concerns include:

- **Wage and employment pressure** in highly exposed occupations (translation, some writing, customer service, entry-level coding).
- **Entry-level jobs** that traditionally trained newcomers may shrink, disrupting career ladders.
- **Concentration** of gains among firms with data, compute and capital.
- **Global effects**: outsourced digital work (call centres, content writing, data entry) in lower-income countries may be affected — while AI tools may also open new opportunities for workers there.
- **Invisible labour**: AI depends on data annotators and content moderators, often poorly paid (see the data-labelling lecture).

Daron Acemoglu and others argue that the **direction** of technology is a choice: we can prioritise AI that augments workers and creates new tasks rather than AI that merely replaces labour.

## Implications for students

:::tip
- **Build durable foundations**: mathematics, statistics, programming, domain knowledge and clear writing remain valuable because they let you direct and verify AI.
- **Develop judgement**: evaluating outputs, spotting errors and understanding limits are the skills that distinguished successful AI users in experiments.
- **Combine AI with a domain**: health, agriculture, education, humanitarian work, law — domain experts who can use AI are in demand.
- **Learn continuously**: tools change monthly; learning how to learn is the meta-skill.
- **Human strengths**: empathy, trust-building, negotiation, leadership and ethical judgement remain central.
:::

## What organisations and policymakers can do

- **Organisations**: involve workers in redesigning workflows; invest in training; use AI to augment and upskill; be transparent about monitoring; share productivity gains.
- **Policy**: education and reskilling programmes, social safety nets, portable benefits, labour protections for platform and data workers, competition policy, and support for AI that serves public needs.

:::note
Predictions about AI and work have a poor track record in both directions. Be sceptical of confident forecasts of either mass unemployment or effortless prosperity. Watch the evidence — controlled studies, labour-market data — and remember that outcomes depend on choices made by companies, governments and workers.
:::

:::exercise
1. Choose a job you know well. Break it into ten tasks and classify each as likely automated, augmented or unaffected by current AI, with reasons.
2. Run a mini experiment: complete two similar tasks with and without an AI assistant, timing yourself and asking a peer to rate quality blind.
3. Write a one-page proposal for how an organisation you know could adopt AI in a way that augments staff and protects jobs.
:::

:::takeaway
- Analyse AI's impact by tasks: automation, augmentation and new tasks.
- LLM exposure is higher for many higher-wage, cognitive occupations; exposure is not replacement.
- Early studies show real but uneven productivity gains, often largest for less-experienced workers — and harm when AI is over-trusted outside its competence.
- Outcomes depend on choices: skills, organisational design and policy shape who benefits.
:::

=== POST ===
slug: responsible-ai-in-education
title: "Responsible AI in Education: Learning With, Not Instead Of, AI"
category: ethics
level: Beginner
tags: education, generative ai, academic integrity, learning, teaching
summary: Generative AI is transforming how students learn and teachers teach. We discuss benefits like personalised tutoring, risks to learning and integrity, why AI-text detectors fail, and practical guidelines for students and educators.
---
As an educator, I see generative AI changing classrooms faster than any technology before it. Students can ask an AI tutor to explain backpropagation at midnight, practise conversation in a new language, or get feedback on code. They can also outsource their thinking entirely — and learn nothing. How we use these tools will shape a generation's education. This lecture offers guidance for students and teachers alike.

## The opportunity: personalised learning at scale

Benjamin Bloom's famous "2 sigma problem" (1984) reported that students tutored one-to-one performed about two standard deviations better than students in conventional classrooms — but individual tutoring is too expensive to provide for everyone. AI tutors raise the possibility of affordable, personalised support:

- **Explanations on demand**, at the right level, in the learner's own language.
- **Practice and feedback**: generating exercises, checking reasoning, giving hints rather than answers.
- **Accessibility**: support for learners with disabilities (reading aids, speech interfaces) and for those studying in a second language.
- **Teacher support**: drafting lesson materials, differentiated exercises and rubrics, freeing time for human interaction.

Early studies of carefully designed AI tutoring have shown promising learning gains in some settings, particularly when the tutor is designed to guide rather than give answers. Evidence is still developing, and design matters enormously.

## The risks

1. **Outsourcing thinking**: learning requires effortful practice ("desirable difficulties"). If AI does the struggle, students may produce good answers while learning little — an illusion of competence. A randomised study with high-school maths students (Bastani et al., 2024) found that unrestricted access to a chatbot improved practice performance but **reduced** later performance on exams without AI, while a tutor-style version with guardrails largely mitigated the harm.
2. **Hallucinations**: confident errors in explanations, citations or code can mislead learners who cannot yet spot them.
3. **Academic integrity**: submitting AI-generated work as one's own.
4. **Equity**: unequal access to devices, connectivity and paid tools can widen gaps; models often perform worse in lower-resource languages.
5. **Privacy**: student data entered into AI tools may be stored or used for training.
6. **Homogenisation**: essays and ideas converging towards generic AI styles.

## Why AI-text detectors are not the answer

:::warning
AI-generated text detectors are unreliable: they produce **false positives** (flagging human writing), can be evaded by light paraphrasing, and studies have found they disproportionately flag writing by **non-native English speakers** (Liang et al., 2023). Accusing a student of misconduct based on a detector score is unjust. Assessment design, transparency and dialogue work better than detection.
:::

## Guidelines for students

:::tip
1. **Try first, then ask**: attempt problems yourself before consulting AI; use it to check and extend your thinking.
2. **Ask for hints and explanations, not final answers**: "Explain the idea behind this step" rather than "Solve this".
3. **Verify**: check facts, run the code, re-derive equations — treat AI output as a draft from a knowledgeable but fallible peer.
4. **Explain it back**: if you cannot explain a solution in your own words without AI, you have not learned it.
5. **Be transparent**: follow your course's AI policy and disclose how you used AI.
6. **Protect privacy**: do not paste personal data, confidential documents or others' work into AI tools.
7. **Use AI to go further**: generate practice questions, get feedback on drafts, explore alternative explanations, practise languages.
:::

## Guidelines for educators

- **Write a clear AI policy** for each course: what is allowed, what must be disclosed, and why.
- **Design assessments for learning**: in-class work, oral explanations, drafts and reflections, projects grounded in local contexts and personal experience, and tasks that require critique of AI outputs.
- **Teach AI literacy**: how models work, where they fail, how to verify, ethical and privacy issues.
- **Use AI to support, not replace, feedback and relationships** — students still need human mentors.
- **Choose tools responsibly**: data protection, accessibility, language coverage and cost for students.
- **Model good practice** by being transparent about your own use.

UNESCO has published guidance on generative AI in education and research, recommending, among other things, age-appropriate use, data protection and building educators' capacity.

## An example: a tutor-style prompt

```text
You are a patient tutor for a university machine learning course.
- Never give the final answer to an exercise directly.
- Ask me what I have tried, then give one hint at a time.
- When I make an error, ask a question that helps me find it.
- Use simple language and check my understanding with a short question at the end.
- If you are not sure about a fact, say so.
```

This shifts the AI from answer machine to Socratic partner — the design principle behind the most promising educational uses.

:::exercise
1. Use an AI tutor with the prompt above to work through an exercise from this course. Reflect: did it help you learn more than reading a solution?
2. Draft an AI-use policy for a course you are taking or teaching, including disclosure rules and rationale.
3. Design an assessment for a machine-learning topic that remains meaningful when students have access to AI tools.
:::

:::takeaway
- AI can personalise learning, broaden access and support teachers — when designed to guide, not replace, thinking.
- Risks include outsourced thinking, hallucinations, integrity, inequity, privacy and homogenisation.
- AI-text detectors are unreliable and biased; rely on assessment design and transparency.
- Students should try first, ask for hints, verify and explain; educators should set clear policies and teach AI literacy.
:::

=== POST ===
slug: careers-in-ai-roadmap
title: "Careers in AI: A Roadmap for Students"
category: ethics
level: Beginner
tags: careers, roadmap, portfolio, skills, job roles
summary: A practical guide to AI careers — the main roles, the skills each needs, a staged learning roadmap, how to build a portfolio that stands out, finding opportunities, and growing throughout your career.
---
Students often ask me: "Sir, how do I get a job in AI?" The field is broad, fast-moving and full of opportunity — in technology companies, research labs, startups, government, healthcare, finance, and in international, humanitarian and development organisations. This lecture offers a practical roadmap, based on what employers look for and what I have seen work for students.

## The main roles

| Role | Focus | Key skills |
|---|---|---|
| Data analyst | Insight from data, dashboards, reporting | SQL, spreadsheets, statistics, visualisation, communication |
| Data scientist | Modelling, experimentation, decision support | Statistics, ML, Python, experimentation, domain knowledge |
| ML engineer | Building and deploying ML systems | Software engineering, ML, MLOps, cloud, APIs |
| AI/LLM application engineer | Building products with foundation models | Python/TypeScript, APIs, RAG, evaluation, prompt design, security |
| Data engineer | Pipelines and data infrastructure | SQL, distributed processing, orchestration, data modelling |
| Research scientist / engineer | New methods and models | Deep mathematics, ML theory, experiments, paper writing |
| MLOps / platform engineer | Infrastructure for ML at scale | DevOps, containers, Kubernetes, monitoring |
| Responsible AI / governance specialist | Fairness, safety, policy, compliance | ML literacy, ethics, law and policy, evaluation |
| Domain expert + AI | AI applied within a field | Deep domain knowledge plus practical ML |

You do not need to choose forever — careers evolve — but choosing a direction helps you focus.

## A staged roadmap

**Stage 1 — Foundations (months 0–6)**
- Python programming, Git, the command line.
- Mathematics: linear algebra, calculus, probability and statistics (the Mathematics track of this site).
- Data handling with pandas and SQL; visualisation.

**Stage 2 — Core machine learning (months 4–10)**
- Supervised and unsupervised learning with scikit-learn; evaluation and validation; feature engineering.
- Complete two or three end-to-end projects on real data.

**Stage 3 — Deep learning (months 8–14)**
- PyTorch; CNNs, sequence models, transformers; transfer learning and fine-tuning.
- One project in vision or NLP, ideally in a language or domain you know well.

**Stage 4 — Specialise and ship (months 12+)**
- Choose depth: NLP/LLMs, computer vision, MLOps, reinforcement learning, data engineering, responsible AI.
- Deploy something real: an API, a small web app, a dashboard used by actual people.
- Learn cloud basics, Docker and monitoring.

Throughout: read papers, write about what you learn, and practise communication.

## Building a portfolio that stands out

Employers see many portfolios with the same Titanic, MNIST and house-price projects. Stand out by:

1. **Solving a real problem** — ideally one from your community, workplace or field, with real (appropriately anonymised and permitted) data.
2. **End-to-end work**: data collection and cleaning → modelling → evaluation → deployment → monitoring and lessons learned.
3. **Clear documentation**: a README explaining the problem, approach, results with honest limitations, and how to run it; a model card for your model.
4. **Rigour**: proper validation, baselines, error analysis, subgroup evaluation.
5. **Communication**: a short blog post or video explaining the project to a non-technical audience.
6. **Local-language or low-resource contributions**: datasets, models or benchmarks for under-served languages are valuable and distinctive.
7. **Open-source contributions**: fixing documentation or bugs in libraries you use.

```text
Strong project README outline
1. Problem & who it helps          5. Results (with baselines & confidence intervals)
2. Data (source, permission, ethics)   6. Error analysis & limitations
3. Approach & why                   7. How to run / demo link
4. Evaluation design                8. What I would do next
```

## Finding opportunities

- **Internships and graduate programmes** — apply broadly and early.
- **Competitions** (e.g. Kaggle) — good for practice; a strong finish is a credible signal.
- **Research assistantships** with faculty; co-authoring papers.
- **Hackathons and open-source communities**; local AI and data science meetups.
- **International and humanitarian organisations**: data, innovation and information-management roles, internships and fellowships; national and regional volunteering programmes.
- **Freelancing** on small projects to build experience.
- **Networking** with genuine curiosity: share your work, ask thoughtful questions, help others.

## Interview preparation

- Coding (data structures, Python, SQL).
- ML fundamentals: bias–variance, regularisation, evaluation metrics, overfitting, trees vs neural networks, how transformers work.
- Case studies: "How would you build a system to prioritise support requests?" — structure your answer: problem framing, data, baseline, model, evaluation, deployment, monitoring, risks.
- Your projects: be ready to explain every decision and what went wrong.

:::tip
The best predictor of success I have seen is not the first language a student learns or the prestige of their first job, but **consistent practice plus visible, well-explained work**. Build one thing properly every few months, write about it, and share it. In two years, your portfolio will speak for you.
:::

## Growing through your career

- Keep learning deliberately — the field changes quickly.
- Develop **communication and collaboration**; many projects fail from misunderstanding, not bad models.
- Seek mentors and, later, mentor others.
- Hold on to your values: the most respected AI professionals combine technical excellence with integrity about limitations and impact.

:::exercise
1. Choose a target role and write a six-month learning plan with monthly milestones and one portfolio project.
2. Rewrite the README of one of your projects using the outline above; ask a non-technical friend whether they understand what it does.
3. Prepare a two-minute explanation of your best project, including one thing that went wrong and what you learned.
:::

:::takeaway
- AI careers span analysis, data science, ML engineering, LLM applications, data engineering, research, MLOps and governance.
- Follow a staged roadmap: foundations → core ML → deep learning → specialisation and deployment.
- Stand out with real, end-to-end, well-documented, rigorous projects — ideally serving your community or language.
- Find opportunities through internships, research, competitions, open source and international organisations; keep learning.
:::

=== POST ===
slug: how-to-read-an-ml-research-paper
title: "How to Read a Machine Learning Research Paper"
category: ethics
level: Beginner
tags: research, papers, reading, critical thinking, arxiv
summary: Research papers are how new ideas enter the field. We present a three-pass reading method, a guide to each section, questions for critical reading, common red flags in ML papers, and tools for keeping up.
---
To stay current in AI — and to eventually contribute — you must read research papers. Beginners often find them intimidating: dense notation, unfamiliar terms, assumed knowledge. The good news is that reading papers is a skill, and it improves quickly with the right method. This lecture gives you a practical approach, based on S. Keshav's widely used "How to Read a Paper" method, adapted for machine learning.

## Anatomy of an ML paper

- **Title and abstract** — the claim in brief.
- **Introduction** — the problem, why it matters, the gap in prior work, the contributions (often a bulleted list — read it carefully).
- **Related work** — context and positioning.
- **Method** — the proposed approach: model, objective, algorithm.
- **Experiments** — datasets, baselines, metrics, main results, **ablations** (removing components to show which matter).
- **Discussion / limitations** — where it fails; often the most honest section.
- **Conclusion** — summary and future work.
- **Appendix** — hyperparameters, proofs, extra results; essential for reproduction.

## The three-pass method

### Pass 1 — the bird's-eye view (5–10 minutes)
Read the title, abstract, introduction, section headings, figures and tables (especially Figure 1 and the main results table), and the conclusion. Skip the maths.

Answer the **five Cs**: **Category** (what type of paper?), **Context** (which prior work does it build on?), **Correctness** (do the assumptions look valid?), **Contributions** (what is new?), **Clarity** (is it well written?). Decide whether to continue.

### Pass 2 — understanding the content (about an hour)
Read the whole paper carefully but skip detailed proofs. Study figures and tables closely: axes, error bars, which baselines are compared. Mark unfamiliar terms and references to read later. At the end, you should be able to **summarise the main idea and evidence** to a classmate.

### Pass 3 — deep understanding (several hours)
Re-derive the key equations; mentally (or actually) re-implement the method; question every assumption; identify what you would do differently. For papers central to your work, **reproduce a result** with the authors' code — the best way to truly understand it.

## Reading the method section

- Identify **inputs, outputs and the objective** (what is being optimised).
- Map notation to concepts; write your own notation table.
- Draw the architecture or pipeline.
- Ask: what is the **one key idea**? Most good papers have one.

## Critical reading: questions to ask

1. **Baselines**: are they strong, recent and **equally tuned**? Weak baselines inflate improvements.
2. **Statistical rigour**: multiple seeds? confidence intervals or standard deviations? Is the improvement larger than the noise?
3. **Datasets**: are they appropriate and diverse? Could there be leakage or contamination?
4. **Ablations**: do they show that each component matters?
5. **Compute**: how much was used? Would baselines improve with the same budget?
6. **Generalisation**: does it work beyond the benchmark? Out-of-distribution?
7. **Claims vs evidence**: do the words in the abstract match the numbers in the tables?
8. **Limitations and societal impact**: are they discussed honestly?

:::warning
Common red flags: comparisons only against old or weak baselines; results on a single seed; cherry-picked qualitative examples; missing hyperparameter details; test sets used for model selection; claims of "state of the art" by a tiny margin without error bars; evaluation only on the authors' own new benchmark; and broad claims ("understands", "reasons") supported by narrow benchmarks.
:::

## A reading template

```markdown
# Paper notes: <title> (<authors>, <venue/year>)

**One-sentence summary:**
**Problem & motivation:**
**Key idea:**
**Method (my words + diagram):**
**Evidence:** datasets / baselines / main numbers (with variance?)
**Ablations — what matters most:**
**Strengths:**
**Weaknesses / questions / red flags:**
**Relevance to my work:**
**Follow-up papers to read:**
```

Keeping notes in a consistent format builds a personal knowledge base you will use for years.

## Finding and keeping up with papers

- **arXiv** (cs.LG, cs.CL, cs.CV, cs.AI) — preprints, not yet peer-reviewed; read with appropriate caution.
- Major venues: NeurIPS, ICML, ICLR (general ML); ACL, EMNLP, NAACL (NLP); CVPR, ICCV, ECCV (vision); AAAI, IJCAI; plus domain venues (e.g. health, humanitarian technology, ACM COMPASS for computing and sustainable societies).
- **Semantic Scholar** and **Google Scholar** — search, citation graphs, alerts.
- **Papers with Code**-style resources and GitHub repositories for implementations.
- **Survey papers** — the best entry point to a new area.
- Reading groups — discussing a paper weekly with peers accelerates learning enormously.

:::tip
Start with **survey papers** and landmark papers (e.g. AlexNet, "Attention Is All You Need", BERT, DDPM, DQN — all covered in this course) before chasing the newest preprints. Foundational papers teach the ideas that later work builds on.
:::

:::exercise
1. Apply the first pass to three recent papers in an area you like and decide which one deserves a second pass.
2. Complete the reading template for "Attention Is All You Need" and compare your notes with the Transformer lecture on this site.
3. Choose one paper and critique its experimental evaluation using the eight critical-reading questions.
:::

:::takeaway
- Papers follow a standard structure; the contribution list, main figure and results table carry the core message.
- Use three passes: overview (5 Cs), content understanding, deep re-derivation/reproduction.
- Read critically: baselines, statistics, data, ablations, compute, generalisation, claims vs evidence, limitations.
- Keep structured notes, start with surveys and landmark papers, and read with peers.
:::

=== POST ===
slug: doing-ml-research-thesis-guide
title: "Doing Machine Learning Research: A Guide for Your Thesis"
category: ethics
level: Intermediate
tags: research, thesis, research questions, experiments, academic writing
summary: A practical guide to undergraduate and master's research in ML — choosing a question, reviewing literature, designing rigorous experiments, avoiding common pitfalls, writing clearly, and conducting research ethically.
---
For many students, the final-year project or thesis is the first real research experience: an open question, no answer key, and months to find something new. It can be the most rewarding part of your degree — or the most stressful. This closing lecture distils advice I give to the students I supervise, so that your research is rigorous, ethical and genuinely useful.

## Step 1: choose a good question

A good research question is:

- **Specific**: "Does continued pretraining on 50 MB of Bangla news improve NER F1 over XLM-R on dataset X?" rather than "Improve Bangla NLP".
- **Feasible**: achievable with your data, compute and time (be honest about GPU access).
- **Relevant**: someone cares about the answer — a community, an organisation, other researchers.
- **Testable**: you can design an experiment whose outcome could prove you wrong.

Good sources of questions: limitations sections of recent papers, failures you observe when applying existing methods to your language or domain, needs of local organisations, and replication of important results under new conditions. **Replication and careful evaluation studies are legitimate, valuable research**, especially for low-resource languages and new contexts.

## Step 2: review the literature efficiently

- Start from 2–3 **surveys** and follow citations backwards and forwards (Semantic Scholar's citation graph helps).
- Build a table: paper, method, data, metric, results, limitations.
- Identify the **gap** your work addresses, and the strongest existing baselines.
- Do not wait until you have read everything — alternate reading with early experiments.

## Step 3: design rigorous experiments

1. **Define metrics and evaluation data before running experiments.** Keep a test set untouched until the end.
2. **Start with strong, simple baselines** (e.g. TF-IDF + logistic regression, a fine-tuned small transformer) and reproduce published numbers where possible.
3. **Change one thing at a time**; plan **ablations** from the start.
4. **Multiple seeds** and confidence intervals; statistical tests for key comparisons.
5. **Fair comparisons**: equal tuning budgets for baselines and your method.
6. **Error analysis**: qualitative examination of failures often yields the most interesting insights.
7. **Track everything** (experiment tracking, versioned data and code) so you can reproduce every number in your thesis.

```text
Experiment plan template
Question:            Does X improve Y on Z?
Hypothesis:          X improves macro-F1 by >= 2 points over baseline B.
Data & splits:       dataset D (v1.2), fixed train/val/test, test used once.
Baselines:           B1 (simple), B2 (strong, published), each tuned with 20 trials.
Metrics:             macro-F1 (primary), per-class F1, calibration (secondary).
Seeds:               5 per configuration; report mean ± std and 95% CI.
Ablations:           remove component a; remove component b.
Compute budget:      ~40 GPU-hours.
Risks & fallback:    if D too small, use cross-validation; if no GPU, use smaller models.
```

## Step 4: avoid common pitfalls

:::warning
- **Data leakage** — preprocessing on all data, duplicates across splits, test-set tuning.
- **Weak or untuned baselines** that make any method look good.
- **Single-run results** reported as definitive.
- **Scope creep** — trying to solve everything; narrow the question instead.
- **Starting to write at the end** — write continuously.
- **Hiding negative results** — a well-analysed negative result is a contribution.
- **Over-claiming** — "our model understands Bangla" versus "improves F1 on two datasets".
:::

## Step 5: write clearly

Structure: abstract, introduction (problem, gap, contributions), background and related work, method, experimental setup, results, analysis/discussion, limitations, conclusion.

Writing advice:

- Write the **contribution list** early and let it guide everything.
- One idea per paragraph; topic sentence first.
- Every figure and table must be referenced and explained in the text; captions should be self-contained.
- Define notation once and use it consistently.
- Report numbers with appropriate precision and uncertainty.
- Be honest about limitations — reviewers and examiners respect it.
- Revise: the first draft is for you; later drafts are for readers. Ask peers to read it.

## Step 6: research ethics

- **Data ethics**: use data you are permitted to use; obtain ethics approval when working with human participants or personal data; anonymise and secure data; respect licences.
- **Community involvement**: when research concerns a community (e.g. a language community or displaced population), involve its members as partners, not just subjects; share results back.
- **Honesty**: never fabricate or selectively report results; disclose AI assistance in writing according to your institution's policy; cite properly — plagiarism includes paraphrasing without attribution.
- **Impact**: consider potential misuse and harms; include an impact statement.
- **Credit**: acknowledge everyone who contributed, including annotators.

## Step 7: share your work

Release code and (where permitted) data with documentation; write a blog post; present at local meetups, student conferences and workshops (many major conferences host workshops for specific languages, regions and application areas, often welcoming student work). Shared, reproducible work builds your reputation and helps others build on it.

:::note
Research is mostly learning to be comfortable with uncertainty: most ideas fail, and that is normal. Keep a research journal of what you tried, what happened and what you think it means. Meet your supervisor regularly with concrete results and specific questions. And remember why you started — curiosity and the wish to solve a problem that matters.
:::

:::exercise
1. Write three candidate research questions for a thesis, and evaluate each against the four criteria (specific, feasible, relevant, testable).
2. Fill in the experiment plan template for your favourite question.
3. Write a one-page introduction for your planned thesis: problem, gap, and a bulleted list of intended contributions.
:::

:::takeaway
- Choose specific, feasible, relevant, testable questions; replication and evaluation studies are valuable.
- Review literature efficiently, identify the gap and strongest baselines.
- Design rigorous experiments: fixed evaluation, strong baselines, ablations, multiple seeds, error analysis, tracking.
- Avoid leakage, weak baselines, scope creep and over-claiming; write continuously; follow research ethics and share your work.
:::
