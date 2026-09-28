=== POST ===
slug: topic-modeling-lda
title: "Topic Modelling: LDA, NMF and Neural Topic Models"
category: nlp
level: Intermediate
tags: topic modeling, lda, nmf, bertopic, unsupervised
summary: Topic models discover themes in large document collections without labels. We derive Latent Dirichlet Allocation's generative story, compare it with NMF and embedding-based BERTopic, and discuss evaluating and interpreting topics.
---
An organisation receives 50,000 free-text survey responses, or a researcher collects a decade of news articles. What themes do they contain? Reading everything is impossible, and there are no labels. **Topic models** discover recurring themes — **topics** — automatically, and describe each document as a mixture of them. They are a staple of computational social science, digital humanities and feedback analysis.

## What is a topic?

A topic is a **probability distribution over words**. A "health" topic puts high probability on *clinic, doctor, medicine, vaccine*; a "shelter" topic on *tent, roof, rain, repair*. A document is a **mixture of topics**: a complaint may be 70% "shelter" and 30% "water".

## Latent Dirichlet Allocation (LDA)

Blei, Ng and Jordan (2003) proposed LDA as a **generative probabilistic model** of how documents are written:

1. For each topic $k = 1, \dots, K$: draw a word distribution $\boldsymbol{\phi}_k \sim \text{Dirichlet}(\boldsymbol{\beta})$.
2. For each document $d$:
   1. draw topic proportions $\boldsymbol{\theta}_d \sim \text{Dirichlet}(\boldsymbol{\alpha})$;
   2. for each word position $n$: draw a topic $z_{dn} \sim \text{Categorical}(\boldsymbol{\theta}_d)$, then a word $w_{dn} \sim \text{Categorical}(\boldsymbol{\phi}_{z_{dn}})$.

We observe only the words; topics and proportions are **latent**. Inference inverts the story: find the $\boldsymbol{\phi}$, $\boldsymbol{\theta}$ and $z$ that best explain the corpus, using **collapsed Gibbs sampling** or **variational inference** (the approach in the original paper and in scikit-learn and gensim).

The **Dirichlet** priors control sparsity. A small $\alpha$ (< 1) encourages each document to use few topics; a small $\beta$ encourages each topic to concentrate on few words.

LDA is a **bag-of-words** model: word order is ignored, which is why preprocessing (stop-word removal, lemmatisation, removing very rare and very common words) matters a lot for it.

```python
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

docs = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"),
                          categories=["sci.med", "sci.space", "rec.autos", "talk.politics.guns"]).data
vec = CountVectorizer(stop_words="english", max_df=0.5, min_df=10, token_pattern=r"(?u)\b[a-zA-Z]{3,}\b")
X = vec.fit_transform(docs)
lda = LatentDirichletAllocation(n_components=4, learning_method="online", random_state=0,
                                doc_topic_prior=0.1, topic_word_prior=0.01).fit(X)
vocab = vec.get_feature_names_out()
for k, comp in enumerate(lda.components_):
    print(f"topic {k}:", ", ".join(vocab[comp.argsort()[-10:][::-1]]))
print("document 0 topic mixture:", lda.transform(X[:1]).round(2))
```

## Non-negative Matrix Factorisation (NMF)

Factorise the document–term matrix (often TF-IDF) into two non-negative matrices:

$$
\mathbf{X} \approx \mathbf{W}\mathbf{H}, \qquad \mathbf{W}, \mathbf{H} \ge 0
$$

Rows of $\mathbf{H}$ are topics (weights over words); rows of $\mathbf{W}$ are document–topic weights. Non-negativity yields additive, parts-based, interpretable components. NMF is fast, deterministic given initialisation, and often gives crisper topics than LDA on short texts.

## Embedding-based topic models: BERTopic

Short texts (tweets, survey answers, chat messages) contain too few words for bag-of-words models to estimate topic mixtures well. **BERTopic** (Grootendorst, 2022) takes a different route:

1. Embed each document with a sentence-transformer (captures meaning, handles synonyms).
2. Reduce dimensionality with **UMAP**.
3. Cluster with **HDBSCAN** (which also flags outliers).
4. Describe each cluster with a **class-based TF-IDF** (c-TF-IDF) — words distinctive of that cluster compared with others.
5. Optionally label topics with an LLM.

It works well on short and multilingual texts (with multilingual embeddings) and supports dynamic (over-time) topics. Its trade-off: each document is typically assigned one topic rather than a mixture.

```python
# pip install bertopic
from bertopic import BERTopic
topic_model = BERTopic(language="multilingual", min_topic_size=15)
topics, probs = topic_model.fit_transform(docs[:2000])
print(topic_model.get_topic_info().head(8))
```

## Evaluating topics

There is no ground truth, so evaluation combines:

- **Topic coherence**: do a topic's top words co-occur in a reference corpus? Metrics such as NPMI or $C_V$ correlate moderately with human judgements.
- **Topic diversity**: fraction of unique words across topics' top lists.
- **Held-out perplexity** (for LDA) — notably, Chang et al. (2009) found perplexity can be **negatively** correlated with human interpretability.
- **Human evaluation**: **word intrusion** tests (can people spot a random word inserted into a topic's top words?) and expert review.

## Choosing the number of topics

Try several $K$, compare coherence and diversity, and — above all — inspect topics with domain experts. Too few topics merge distinct themes; too many split them into near-duplicates.

:::warning
Topic models describe **patterns of word co-occurrence**, not ground truth about what people meant. Topics can be unstable across random seeds, and labels given to topics are human interpretations. Read representative documents for each topic before drawing conclusions, report uncertainty, and never let a topic model silently decide which voices are counted.
:::

:::exercise
1. Fit LDA with $K = 5, 10, 20$ on a corpus of your choice and compute coherence for each. Which do experts find most useful?
2. Compare LDA, NMF and BERTopic on 2,000 short survey-style texts; which topics are most interpretable?
3. Run LDA with three different random seeds and measure how stable the topics are.
:::

:::takeaway
- Topics are word distributions; documents are topic mixtures.
- LDA is a generative model with Dirichlet priors, fitted by Gibbs sampling or variational inference.
- NMF gives fast, additive topics; BERTopic clusters sentence embeddings and suits short texts.
- Evaluate with coherence, diversity and human judgement; interpret topics cautiously.
:::

=== POST ===
slug: multilingual-low-resource-nlp
title: "Multilingual and Low-Resource NLP (with a Focus on Bangla)"
category: nlp
level: Intermediate
tags: multilingual, low-resource, bangla, cross-lingual transfer, fairness
summary: Most of the world's languages have little digital data. We examine why this matters, how multilingual models enable cross-lingual transfer, the specific challenges of languages like Bangla, and practical strategies for building NLP in low-resource settings.
---
There are roughly 7,000 languages in the world, but most NLP research and data focus on a handful — above all English. Joshi et al. (2020) grouped languages by data availability and found that the vast majority of languages, spoken by well over a billion people in total, have almost no labelled data and little unlabelled text. Bangla, with more than 250 million speakers, is one of the most spoken languages in the world, yet it has historically been under-resourced in NLP. For people who need information and services in their own language — including refugees and migrants — this gap matters.

## What makes a language "low-resource"?

- Little **unlabelled text** online.
- Few **labelled datasets** for tasks (NER, sentiment, QA).
- Missing **tools**: tokenisers, morphological analysers, spell checkers.
- Few **evaluation benchmarks**, so progress cannot be measured.
- **Script and encoding issues**, non-standard spelling, dialectal variation and code-mixing.

## Challenges specific to Bangla (and similar languages)

- **Script complexity**: Bengali script uses conjunct consonants (যুক্তাক্ষর), vowel signs that attach before or after consonants, and multiple Unicode representations of visually identical text — Unicode normalisation is essential.
- **Rich morphology**: inflections for case, number, tense and person, plus postpositions attached to words, create many surface forms per lemma.
- **Romanised Bangla ("Banglish")** and **code-mixing** with English on social media: "ami office e jacchi, meeting ta late hobe".
- **Dialects**: Sylheti, Chittagonian and others differ substantially from standard Bangla.
- **Tokeniser inefficiency**: tokenisers trained mainly on English split Bangla into many more tokens, increasing cost and shortening effective context for multilingual LLMs.

## Strategy 1: multilingual pretrained models

Models pretrained on text in many languages share a single vocabulary and parameters:

- **mBERT** (104 languages), **XLM-RoBERTa** (100 languages, CommonCrawl), **mT5**, **NLLB-200** for translation, multilingual sentence encoders (LaBSE, multilingual E5), and multilingual LLMs.
- Language-specific models such as **BanglaBERT** (trained on large Bangla corpora) often outperform multilingual models on Bangla tasks.

## Strategy 2: cross-lingual transfer

Remarkably, a multilingual model fine-tuned on labelled data in one language (e.g. English NER) can perform the task in another language **zero-shot**. Representations of translations tend to align in the shared space. Transfer works best between **related** languages and scripts, and degrades for distant, under-represented ones. Benchmarks such as XTREME and XGLUE measure this.

```python
from transformers import pipeline
# A multilingual NLI model can classify text in many languages without language-specific training
clf = pipeline("zero-shot-classification", model="joeddav/xlm-roberta-large-xnli")
text = "আমাদের এলাকায় বিশুদ্ধ পানির খুব অভাব, শিশুরা অসুস্থ হয়ে পড়ছে।"   # "Our area lacks clean water; children are falling ill."
print(clf(text, candidate_labels=["water and sanitation", "education", "shelter", "health"], multi_label=True))
```

## Strategy 3: create data efficiently

- **Translate-train / translate-test**: machine-translate English training data into the target language (or test data into English). Cheap, but translation errors and "translationese" limit quality.
- **Annotation projection**: project labels (e.g. entities) through word alignments of parallel text.
- **Active learning** to prioritise the most informative examples for native-speaker annotators.
- **LLM-assisted annotation**: let an LLM pre-label, then have native speakers correct — verify quality carefully, since LLMs are weaker in low-resource languages.
- **Community-driven datasets**: initiatives such as Masakhane (African languages) show the power of participatory, community-led NLP with local speakers as researchers, not only annotators.

## Strategy 4: adapt the model

- **Continued pretraining** on monolingual target-language text.
- **Vocabulary extension**: add target-language tokens to the tokeniser and train their embeddings — reduces fertility and cost.
- **Adapters** per language (e.g. MAD-X) to add languages without retraining everything.

## Evaluate fairly

:::warning
Never assume a multilingual model works equally well in every language. Evaluate **per language and per dialect**, with test data written by native speakers (not only translated from English), and report results for each. Aggregate multilingual scores can hide very poor performance in exactly the languages that most need support.
:::

## Why it matters

Language technology determines who can access information, services and opportunities online. Poor support for a language means worse search results, worse translation, weaker content moderation (both over- and under-moderation), and LLM assistants that are less helpful or less safe. Building NLP for low-resource languages is both a fascinating research problem and an act of inclusion. It is also a field where students from those language communities have a unique advantage: native-speaker insight.

:::exercise
1. Measure tokens-per-word for the same 20 sentences in English and Bangla with three tokenisers (BERT, XLM-R, a GPT-style tokeniser).
2. Fine-tune XLM-R on an English sentiment dataset and evaluate zero-shot on a Bangla sentiment dataset; then fine-tune on 500 Bangla examples and compare.
3. Collect 30 code-mixed or romanised social-media sentences and test how a multilingual classifier handles them.
:::

:::takeaway
- Most languages lack data, tools and benchmarks; this creates real inequities.
- Bangla presents script, morphology, code-mixing, dialect and tokenisation challenges.
- Multilingual and language-specific pretrained models enable cross-lingual transfer and adaptation.
- Create data efficiently with native speakers, and always evaluate per language and dialect.
:::

=== POST ===
slug: dialogue-systems-chatbots
title: "Dialogue Systems and Chatbots: From Rules to LLM Assistants"
category: nlp
level: Intermediate
tags: chatbots, dialogue systems, intent detection, slot filling, conversational ai
summary: We compare rule-based, task-oriented and open-domain dialogue systems; cover intent detection, slot filling and dialogue state tracking; and show how LLM-based assistants with retrieval and tools are designed, evaluated and deployed safely.
---
From ELIZA in 1966 to today's AI assistants, conversation has been a grand challenge of AI. Chatbots now answer customer questions, help people book appointments, guide users through complex procedures and provide information in many languages. Designing a good conversational system requires more than a powerful model: it requires understanding the users, the tasks, the failure modes and the right hand-offs to humans.

## Types of dialogue systems

| Type | Goal | Example | Typical technology |
|---|---|---|---|
| Rule-based / scripted | Follow predefined flows | Menu-driven SMS bot | Decision trees, keyword rules |
| Task-oriented | Complete a specific task | Book an appointment, check case status | NLU + state tracking + policy + templates; or LLM with tools |
| Question answering | Answer information requests | FAQ bot | Retrieval + reader/LLM (RAG) |
| Open-domain / social | Engaging general conversation | Companion chatbots | Large language models |

## The classic task-oriented pipeline

1. **Natural Language Understanding (NLU)**:
   - **Intent detection** — what the user wants (`book_appointment`, `check_status`) — a text classification problem.
   - **Slot filling** — extract parameters (`date=Tuesday`, `clinic=Camp 4`) — a sequence-labelling problem, often solved jointly with intent detection.
2. **Dialogue State Tracking (DST)**: maintain a structured record of what is known so far, updated every turn (the user may change their mind).
3. **Dialogue Policy**: decide the next action — ask for a missing slot, confirm, query a database, hand over to a human.
4. **Natural Language Generation (NLG)**: turn the action into text (templates or a generator).

```text
User:   I need to see a doctor for my son on Tuesday.
NLU:    intent=book_appointment, slots={patient: son, date: Tuesday}
State:  {intent: book_appointment, patient: son, date: Tuesday, clinic: ?}
Policy: request(clinic)
NLG:    "Which clinic would you like to visit? Camp 4 or Camp 7?"
```

This modular design is controllable, testable and auditable — valuable when conversations have real consequences — but brittle to phrasings and flows its designers did not anticipate.

## End-to-end and LLM-based assistants

Large language models can conduct fluent, flexible conversation directly. Production assistants combine an LLM with:

- a **system prompt** defining role, tone, scope and rules;
- **retrieval** over approved documents (RAG) for accurate, current information;
- **tools/function calling** to query databases or take actions (check a case status, book a slot) through well-defined, permission-checked APIs;
- **memory** of the conversation (and, with consent, of user preferences);
- **guardrails**: input/output filters, topic restrictions, refusal of unsafe requests, and escalation to humans.

```python
# A minimal intent + slot baseline for a task-oriented bot
import re

INTENTS = {
    "book_appointment": ["appointment", "see a doctor", "book", "schedule"],
    "check_status": ["status", "my case", "application", "update"],
    "opening_hours": ["open", "hours", "close", "time"],
}
DAYS = r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday)"

def nlu(text):
    t = text.lower()
    intent = max(INTENTS, key=lambda k: sum(kw in t for kw in INTENTS[k]))
    slots = {}
    if m := re.search(DAYS, t):
        slots["date"] = m.group(1)
    if m := re.search(r"camp\s*(\d+)", t):
        slots["clinic"] = f"Camp {m.group(1)}"
    return intent, slots

print(nlu("Can I book a doctor appointment at camp 4 on Tuesday?"))
```

In practice, replace the keyword rules with a fine-tuned classifier or an LLM with structured (JSON) outputs — but keep the explicit state and policy where reliability matters.

## Designing good conversations

- **Scope clearly**: tell users what the bot can and cannot do.
- **Handle failure gracefully**: when unsure, ask a clarifying question or offer a human hand-off rather than guessing.
- **Confirm** consequential actions before executing them.
- **Accessibility and language**: support the languages, literacy levels and channels (SMS, WhatsApp, voice) your users actually use.
- **Privacy**: collect the minimum data needed; explain how it is used; avoid storing sensitive details unnecessarily.
- **Transparency**: users should know they are talking to an AI.

## Evaluation

- **Task-oriented**: task success rate, turns to completion, slot accuracy, joint goal accuracy for state tracking.
- **QA bots**: answer correctness and faithfulness against sources, containment rate (resolved without human) *balanced against* correct escalation.
- **Conversation quality**: human ratings of helpfulness, coherence, empathy and safety; A/B tests with real users.
- **Red-teaming**: deliberately attempt to make the bot give harmful, false or out-of-scope answers, including prompt-injection attacks hidden in retrieved documents.

:::warning
Conversational systems can cause harm through confident misinformation, inappropriate responses to users in distress, privacy leaks, or manipulation. For services used by vulnerable people — health, legal status, protection, mental health — design escalation paths to trained humans, restrict the bot to vetted information, test extensively with representative users, and monitor conversations (with appropriate consent and safeguards) after launch.
:::

:::exercise
1. Design the intents, slots and dialogue flow for a bot that helps families book vaccination appointments. Include error and hand-off paths.
2. Build a small intent classifier with 20 examples per intent using sentence embeddings, and evaluate on paraphrases written by classmates.
3. Red-team an LLM-based FAQ bot with 20 adversarial inputs (out-of-scope, misleading, prompt-injection). Document failures and propose guardrails.
:::

:::takeaway
- Dialogue systems range from scripted flows to task-oriented pipelines and open-domain LLM assistants.
- Task-oriented pipelines use intent detection, slot filling, state tracking, policy and generation — controllable and auditable.
- LLM assistants combine system prompts, retrieval, tools, memory and guardrails.
- Design for scope, graceful failure, confirmation, accessibility and privacy; evaluate task success and safety, and red-team.
:::

=== POST ===
slug: automatic-speech-recognition
title: "Automatic Speech Recognition: From HMMs to Whisper"
category: nlp
level: Intermediate
tags: speech recognition, asr, ctc, whisper, spectrogram, wer
summary: Speech recognition converts audio into text. We cover audio features and spectrograms, the classical HMM–GMM pipeline, end-to-end neural models with CTC and attention, self-supervised wav2vec 2.0, Whisper, and evaluation with word error rate.
---
Speech is the most natural human interface, and for people who cannot read or type easily, often the only practical one. **Automatic Speech Recognition (ASR)** converts spoken audio into text, powering voice assistants, captions for deaf and hard-of-hearing people, transcription of interviews and meetings, and voice interfaces for services in many languages. In the last decade ASR accuracy improved dramatically — but performance still varies widely across languages, accents and recording conditions.

## From waveform to features

Audio is a 1-D signal sampled at, for example, 16,000 samples per second. Speech information lives in **frequencies** that change over time, so we compute a **spectrogram**:

1. Split the signal into short overlapping frames (e.g. 25 ms windows every 10 ms).
2. Apply a Fourier transform to each frame (Short-Time Fourier Transform).
3. Map frequencies to the **mel scale**, which mimics human pitch perception (finer resolution at low frequencies), and take logarithms of the energies → **log-mel spectrogram** (typically 80 mel bins).

Classical systems further computed **MFCCs** (mel-frequency cepstral coefficients). Modern neural models consume log-mel spectrograms or even raw waveforms.

```python
import torch, torchaudio

waveform, sr = torchaudio.load("speech.wav")                       # (channels, samples)
waveform = torchaudio.functional.resample(waveform, sr, 16000).mean(0, keepdim=True)
mel = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=400, hop_length=160, n_mels=80)(waveform)
log_mel = torch.log(mel + 1e-6)
print(log_mel.shape)                    # (1, 80, frames) — about 100 frames per second
```

## The classical pipeline (until ~2015)

$$
\hat{W} = \arg\max_W P(W \mid X) = \arg\max_W \underbrace{P(X \mid W)}_{\text{acoustic model}}\;\underbrace{P(W)}_{\text{language model}}
$$

- **Acoustic model**: HMMs over **phonemes** (context-dependent triphones), with Gaussian mixture models (later deep neural networks — the "hybrid" DNN-HMM systems) modelling acoustic features.
- **Pronunciation lexicon**: maps words to phoneme sequences.
- **Language model**: n-gram model over words.
- **Decoder**: searches the combined space with weighted finite-state transducers and beam search.

Powerful but complex, requiring expert-built lexicons and many separately trained components.

## End-to-end neural ASR

Neural models map audio features directly to characters or subword tokens:

- **CTC-based** models (e.g. DeepSpeech, Jasper, QuartzNet): an encoder (RNN, CNN or transformer) outputs a distribution per frame; **CTC loss** (the same algorithm used in OCR) handles the unknown alignment between many audio frames and fewer output tokens.
- **Attention encoder–decoder** models (Listen, Attend and Spell): a decoder generates tokens while attending over encoder frames.
- **RNN-Transducer (RNN-T)**: combines an audio encoder, a prediction network over previous tokens and a joint network; naturally **streaming** — dominant in on-device assistants.
- **Conformer** encoders combine convolution (local acoustic patterns) and self-attention (global context) and are widely used.

External language models can still be fused during decoding to improve domain vocabulary.

## Self-supervised speech representations

**wav2vec 2.0** (Baevski et al., 2020) learns from **unlabelled audio**: a CNN encodes the raw waveform, spans of the latent sequence are masked, and a transformer is trained with a contrastive loss to identify the correct quantised latent for each masked position among distractors. Fine-tuned with CTC on very little labelled speech — in the paper, as little as ten minutes — it produced usable recognisers, and with an hour it achieved strong results on standard benchmarks. HuBERT and XLS-R (multilingual, 128 languages) extended the approach — a breakthrough for low-resource languages.

## Whisper: weakly supervised at scale

OpenAI's **Whisper** (2022) trained an encoder–decoder transformer on about **680,000 hours** of audio paired with (noisy) transcripts from the web, covering many languages, with multitask tokens for transcription, translation into English, language identification and timestamps. It is robust to accents, noise and domains without fine-tuning, and became a popular open model.

```python
from transformers import pipeline
asr = pipeline("automatic-speech-recognition", model="openai/whisper-small", chunk_length_s=30)
print(asr("interview_bn.wav", generate_kwargs={"language": "bengali", "task": "transcribe"})["text"])
```

## Evaluation: word error rate

$$
\text{WER} = \frac{S + D + I}{N}
$$

where $S$, $D$, $I$ are substitutions, deletions and insertions in the minimum edit-distance alignment and $N$ is the number of reference words. **Character error rate (CER)** is preferred for languages without clear word boundaries or with rich morphology. Normalise text consistently (numbers, punctuation, casing) before scoring.

```python
# pip install jiwer
import jiwer
print(jiwer.wer("the clinic opens at nine tomorrow", "the clinic open at nine to morrow"))
```

## Challenges

- **Accents, dialects and code-switching**; studies have found substantially higher error rates for some speaker groups (e.g. Koenecke et al., 2020, found commercial systems made roughly twice as many errors for Black American speakers as for white speakers).
- **Noise, reverberation, far-field microphones**, overlapping speakers (**diarisation** — who spoke when).
- **Low-resource languages** and domain-specific vocabulary (names, medical terms).
- **Hallucination** in large sequence-to-sequence models — generating text during silence or noise — a known issue for Whisper-style models that matters in medical and legal transcription.

:::warning
When ASR transcripts are used for records that affect people (medical notes, legal statements, case interviews), have humans review them, keep the original audio (with consent and security) for verification, and measure error rates for the specific languages and accents of your users.
:::

:::exercise
1. Record ten sentences in two languages or accents and compute WER/CER for Whisper-small and Whisper-medium.
2. Plot a log-mel spectrogram of your own voice saying a vowel and a fricative ("aaa" vs "sss"). Explain the differences.
3. Fine-tune a wav2vec 2.0 or XLS-R checkpoint with CTC on a small labelled dataset in a low-resource language and report CER.
:::

:::takeaway
- ASR converts log-mel spectrogram features into text.
- Classical systems combined HMM acoustic models, lexicons and n-gram LMs; end-to-end models use CTC, attention or transducers.
- Self-supervised wav2vec 2.0 enables ASR with little labelled data; Whisper scales weak supervision for robustness.
- Evaluate with WER/CER per accent and language; review transcripts used for consequential records.
:::

=== POST ===
slug: text-to-speech-synthesis
title: "Text-to-Speech: From Concatenation to Neural Voices"
category: nlp
level: Intermediate
tags: text-to-speech, tts, vocoder, tacotron, speech synthesis
summary: Text-to-speech systems turn written text into natural-sounding speech. We cover the TTS pipeline, text normalisation and phonemes, acoustic models like Tacotron and FastSpeech, neural vocoders, end-to-end and zero-shot voice models, evaluation, and voice-cloning ethics.
---
Text-to-speech (TTS) gives machines a voice. It reads screens aloud for blind and visually impaired users, delivers health information to people with low literacy, powers voice assistants and navigation, and can present information in local languages over a simple phone call. Neural methods made synthetic speech remarkably natural — which also created new risks from voice cloning.

## The TTS pipeline

1. **Text analysis (front-end)**
   - **Text normalisation**: expand numbers, dates, abbreviations and symbols into words ("Dr. Rahman, 12/05, 3.5 kg" → "Doctor Rahman, twelfth of May, three point five kilograms"). Language- and context-dependent — and a common source of errors.
   - **Grapheme-to-phoneme (G2P) conversion**: map spelling to pronunciation, handling exceptions and homographs ("read" present vs past).
   - **Prosody prediction**: phrasing, stress and intonation (questions rise in pitch).
2. **Acoustic model**: map phonemes/characters to an intermediate acoustic representation, usually a **mel spectrogram**.
3. **Vocoder**: convert the mel spectrogram into a waveform.

## Historical approaches

- **Formant synthesis**: rule-based models of the vocal tract — intelligible but robotic.
- **Concatenative synthesis (unit selection)**: record hours of speech from one speaker, cut into units, and select and join the best-matching units. Natural in-domain, but glitchy at joins and inflexible.
- **Statistical parametric synthesis (HMM-based)**: model acoustic parameters with HMMs — flexible, but muffled ("buzzy") speech.

## Neural TTS

**WaveNet** (DeepMind, 2016) generated raw audio sample by sample with dilated causal convolutions, producing strikingly natural speech — but slowly, one of 16,000–24,000 samples per second at a time.

**Tacotron 2** (2018) combined an attention-based sequence-to-sequence model (characters → mel spectrogram) with a WaveNet vocoder, reaching naturalness ratings close to recorded human speech on its evaluation.

**FastSpeech / FastSpeech 2** (2019–2020) made acoustic models **non-autoregressive**: a duration predictor decides how many frames each phoneme lasts, and the whole spectrogram is generated in parallel — much faster and more robust (no skipped or repeated words from attention failures). FastSpeech 2 also predicts pitch and energy for controllable prosody.

**Neural vocoders**: WaveRNN, WaveGlow (flow-based), **HiFi-GAN** (GAN-based, fast and high quality) made real-time high-quality synthesis possible on modest hardware.

**End-to-end models** such as **VITS** (2021) combine acoustic model and vocoder into one model trained with variational inference and adversarial losses.

## Codec language models and zero-shot voices

A newer paradigm treats speech as sequences of discrete tokens from a **neural audio codec** (e.g. EnCodec, SoundStream) and models them with language-model techniques. **VALL-E** (2023) showed that a model conditioned on text and a **three-second recording** of an unseen speaker could synthesise speech in that speaker's voice (zero-shot voice cloning). Many modern systems use codec tokens, diffusion or flow matching for highly natural, expressive, multilingual speech.

```python
# A lightweight open multilingual TTS example (Meta's MMS-TTS, VITS-based)
from transformers import VitsModel, AutoTokenizer
import torch, scipy.io.wavfile

model = VitsModel.from_pretrained("facebook/mms-tts-eng")
tok = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")
inputs = tok("The health centre is open from nine in the morning until four.", return_tensors="pt")
with torch.no_grad():
    wav = model(**inputs).waveform[0].numpy()
scipy.io.wavfile.write("announcement.wav", model.config.sampling_rate, wav)
```

(Meta's Massively Multilingual Speech project released TTS models for over a thousand languages, including many low-resource ones.)

## Evaluation

- **Mean Opinion Score (MOS)**: listeners rate naturalness from 1 to 5; report confidence intervals.
- **Intelligibility**: transcribe the synthetic speech with ASR (or humans) and compute WER.
- **Speaker similarity** for voice cloning (embedding similarity + listener tests).
- **Prosody and pronunciation errors**: especially names, numbers and loanwords.
- **Real-time factor** and latency for interactive systems.

Test with native speakers of each target language and dialect; pronunciation errors in local names and places undermine trust quickly.

## Ethics of synthetic voices

:::warning
Voice cloning enables fraud (impersonating relatives or officials on the phone), misinformation (fabricated audio of public figures), and violations of personal rights over one's voice. Responsible practice includes:
- **Consent**: only clone voices with explicit, informed permission.
- **Disclosure**: tell listeners when speech is synthetic.
- **Watermarking and detection** of synthetic audio where possible.
- **Access controls** on high-fidelity cloning features.
- **Awareness**: organisations should warn communities about voice-cloning scams, and never rely on voice alone to verify identity.
:::

## Applications for inclusion

TTS in local languages over basic phones (interactive voice response) can reach people with limited literacy or no smartphone; screen readers depend on high-quality TTS; and educational content can be delivered in learners' mother tongues. Building good voices for under-served languages requires recorded speech from native speakers, careful text normalisation rules and community evaluation.

:::exercise
1. Write text-normalisation rules for dates, currency amounts and phone numbers in your language, and test them on 20 sentences.
2. Synthesise the same paragraph with two TTS systems and run a small MOS test with five listeners.
3. Use ASR to transcribe synthetic speech and compute WER as an intelligibility measure.
:::

:::takeaway
- TTS = text normalisation + G2P + prosody → acoustic model (mel spectrogram) → vocoder.
- Neural models (Tacotron 2, FastSpeech 2, HiFi-GAN, VITS) made speech natural and fast.
- Codec language models enable zero-shot voice cloning from seconds of audio.
- Evaluate with MOS, intelligibility and native-speaker review; cloning demands consent, disclosure and safeguards.
:::

=== POST ===
slug: efficient-transformers-long-context
title: "Efficient Transformers: Sparse Attention, Linear Attention and FlashAttention"
category: nlp
level: Advanced
tags: efficient transformers, flashattention, sparse attention, linear attention, long context, state space models
summary: Self-attention's quadratic cost limits context length. We survey sparse and local attention, low-rank and kernel-based linear attention, IO-aware FlashAttention, and alternatives such as state-space models — with their trade-offs.
---
Self-attention compares every token with every other token: its time and memory grow **quadratically** with sequence length $T$. Doubling the context from 8K to 16K tokens quadruples attention cost. Yet many applications need long contexts — entire books, legal case files, codebases, long conversations, genomic sequences. A large research effort has produced ways to make transformers efficient. Some change the mathematics; the most influential simply changed **how** exact attention is computed.

## Where the cost comes from

For each head, computing $\mathbf{Q}\mathbf{K}^\top$ costs $O(T^2d)$ time and materialising the $T \times T$ score matrix costs $O(T^2)$ memory. For $T = 32{,}768$, a single fp16 attention matrix per head is about 2 GB.

## Family 1: sparse and local attention

Restrict each token to attend to a subset of positions:

- **Sliding-window (local) attention**: each token attends to $w$ neighbours — cost $O(Tw)$. Stacking layers expands the receptive field, like convolutions. Used (sometimes alternating with full attention) in several modern LLMs.
- **Dilated windows**: skip positions to cover longer ranges.
- **Global tokens**: a few tokens (e.g. [CLS] or task tokens) attend to and are attended by everything.
- **Longformer** (2020) combined sliding windows, dilation and global attention; **BigBird** (2020) added random attention and proved such sparse patterns retain theoretical expressivity.
- **Block-sparse** patterns (Sparse Transformer, 2019) for images, audio and long text.

Trade-off: efficient, but fixed sparsity can miss important long-range interactions.

## Family 2: low-rank and kernel (linear) attention

- **Linformer** (2020): project keys and values along the sequence dimension to a fixed size $k$, giving $O(Tk)$ — assumes the attention matrix is approximately low rank.
- **Kernel-based linear attention**: replace the softmax kernel $\exp(\mathbf{q}^\top\mathbf{k})$ with a feature map $\phi(\mathbf{q})^\top\phi(\mathbf{k})$. Then, by associativity,

$$
\text{Attn}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) \approx \frac{\phi(\mathbf{Q})\big(\phi(\mathbf{K})^\top\mathbf{V}\big)}{\phi(\mathbf{Q})\big(\phi(\mathbf{K})^\top\mathbf{1}\big)}
$$

  which costs $O(Td^2)$ — **linear** in $T$. **Performer** (2020) approximated softmax attention with random features. In causal form, linear attention becomes a **recurrence** with a fixed-size state, enabling constant-memory generation.

Trade-off: approximations have generally underperformed exact softmax attention on language modelling quality at equal scale, especially for tasks requiring precise retrieval from context.

## Family 3: exact attention, computed smarter — FlashAttention

Dao et al. (2022) observed that attention on GPUs is bottlenecked not by arithmetic but by **memory traffic** between slow high-bandwidth memory (HBM) and fast on-chip SRAM. **FlashAttention**:

- splits $\mathbf{Q}$, $\mathbf{K}$, $\mathbf{V}$ into **tiles** that fit in SRAM;
- computes attention tile by tile using an **online softmax** (maintaining running maxima and normalisers, so the softmax can be computed incrementally without seeing the whole row);
- never writes the full $T \times T$ matrix to HBM, and recomputes what is needed in the backward pass.

The result is **exact** attention with memory **linear** in $T$ and substantial wall-clock speed-ups. FlashAttention-2 and -3 improved parallelism and hardware utilisation further. It is now standard; in PyTorch, `F.scaled_dot_product_attention` dispatches to fused kernels automatically.

```python
import torch, torch.nn.functional as F, time

def naive_attention(q, k, v):
    s = q @ k.transpose(-2, -1) / q.size(-1) ** 0.5
    mask = torch.triu(torch.ones(s.shape[-2:], dtype=torch.bool, device=q.device), 1)
    return s.masked_fill(mask, float("-inf")).softmax(-1) @ v

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32
for T in [1024, 4096]:
    q = k = v = torch.randn(1, 8, T, 64, device=device, dtype=dtype)
    for name, fn in [("naive", naive_attention),
                     ("fused", lambda q, k, v: F.scaled_dot_product_attention(q, k, v, is_causal=True))]:
        t0 = time.time(); fn(q, k, v)
        if device == "cuda":
            torch.cuda.synchronize()
        print(f"T={T:>5} {name:<6} {1000 * (time.time() - t0):7.1f} ms")
```

## The online softmax trick

To compute $\text{softmax}(\mathbf{x})$ over a row seen in chunks, keep a running maximum $m$ and running sum $\ell$. When a new chunk with maximum $m'$ arrives, rescale: $\ell \leftarrow \ell e^{m - m_{\text{new}}} + \sum e^{x_i - m_{\text{new}}}$. The same rescaling applies to the running weighted sum of values. This algebraic trick, combined with tiling, is the heart of FlashAttention.

## Family 4: alternatives to attention

- **State-space models (SSMs)**: S4 (2021) and **Mamba** (2023) model sequences with structured linear recurrences that can be computed in parallel (as convolutions or scans) during training and as constant-memory recurrences during generation. Mamba adds **input-dependent (selective)** parameters, closing much of the quality gap with transformers on language.
- **Linear RNNs** (RWKV, RetNet, xLSTM variants, gated linear attention) pursue similar goals.
- **Hybrids** interleave a few attention layers with many SSM or linear layers, aiming for transformer-level recall with lower cost.

## Other practical techniques for long context

- **KV-cache optimisations**: grouped-query attention, cache quantisation, paged memory (vLLM's PagedAttention), evicting less important tokens.
- **Ring/sequence parallelism**: distribute very long sequences across GPUs.
- **Position-encoding scaling** (RoPE interpolation, YaRN) to extend context.
- **Retrieval instead of length**: often it is cheaper and more reliable to retrieve relevant passages than to put everything in the context.

:::note
Longer context windows do not guarantee the model **uses** the information well. Evaluate long-context models with tasks that require finding and combining information at different positions ("needle-in-a-haystack" tests are a start, but multi-hop and aggregation tests are more demanding).
:::

:::exercise
1. Compute the memory of the full attention matrix for $T = 128{,}000$ tokens, 32 heads, in fp16. Why is FlashAttention essential?
2. Implement the online softmax for a vector processed in chunks of 4, and verify it matches `torch.softmax`.
3. Implement causal linear attention with $\phi(x) = \text{elu}(x) + 1$ as a recurrence, and compare its outputs and speed with exact attention.
:::

:::takeaway
- Standard attention is $O(T^2)$ in time and memory.
- Sparse/local attention (Longformer, BigBird) and linear/low-rank attention (Linformer, Performer) reduce cost, often with quality trade-offs.
- FlashAttention computes exact attention with tiling and online softmax, cutting memory traffic — now the standard.
- State-space models (Mamba) and hybrids are efficient alternatives; long context still needs careful evaluation.
:::
