=== POST ===
slug: sentiment-analysis
title: "Sentiment Analysis: Opinions, Aspects and Nuance"
category: nlp
level: Beginner
tags: sentiment analysis, opinion mining, aspect-based, lexicons, classification
summary: Sentiment analysis detects opinions and emotions in text. We compare lexicon-based and learned approaches, tackle negation and sarcasm, extend to aspect-based sentiment and emotions, and discuss evaluation and misuse.
---
"The food was delicious but the service was painfully slow." Is this review positive or negative? Both — about different things. **Sentiment analysis** (opinion mining) identifies the attitudes expressed in text. It is used to understand customer feedback, monitor public opinion, analyse survey responses and track reactions to services. It is also a perfect case study in how simple-looking NLP tasks hide deep linguistic complexity.

## Levels of granularity

- **Document-level**: overall polarity of a review.
- **Sentence-level**: polarity of each sentence.
- **Aspect-based**: sentiment towards specific aspects ("food: positive; service: negative").
- **Emotion detection**: joy, anger, fear, sadness, surprise, disgust — beyond positive/negative.
- **Intensity**: how strongly positive or negative (a regression problem).

## Lexicon-based approaches

Use a dictionary of words with sentiment scores and aggregate them, with rules for:

- **Negation**: "not good" flips polarity.
- **Intensifiers**: "very good" > "good"; "slightly good" < "good".
- **Contrast**: "but" shifts emphasis to the clause that follows.
- **Punctuation, capitalisation and emojis**: "GREAT!!!" is stronger than "great".

**VADER** (Hutto & Gilbert, 2014) encodes such rules for social-media English and works without any training data.

```python
# pip install vaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
an = SentimentIntensityAnalyzer()
for s in ["The service was good.", "The service was not good.", "The service was VERY good!!!",
          "The food was delicious but the service was painfully slow."]:
    print(f"{an.polarity_scores(s)['compound']:+.3f}  {s}")
```

Lexicons are transparent and fast, but they miss domain-specific meaning ("unpredictable" is good for a film plot, bad for a car's brakes) and most context.

## Machine-learning approaches

- **TF-IDF n-grams + logistic regression**: bigrams like "not good" help with negation; strong baseline.
- **Fine-tuned transformers** (BERT, RoBERTa, multilingual XLM-R): the state of the art, capturing context, negation scope and many subtle cues.
- **LLMs zero-/few-shot**: good general sentiment judgement without training data; useful for bootstrapping labels.

## Hard phenomena

| Phenomenon | Example | Why it is hard |
|---|---|---|
| Negation scope | "I don't think it's bad at all" | Double negation; scope beyond adjacent words |
| Sarcasm/irony | "Great, another two-hour queue." | Literal words are positive |
| Comparatives | "Better than the old clinic, worse than expected" | Sentiment is relative |
| Implicit sentiment | "The battery lasted three hours." | Facts that imply opinion via world knowledge |
| Domain dependence | "long" (battery life vs waiting time) | Polarity varies by domain |
| Code-mixing | "Service ta khub bhalo chilo, but slow" | Multiple languages in one sentence |
| Mixed sentiment | Pros and cons in one text | Single label is inadequate |

## Aspect-based sentiment analysis (ABSA)

ABSA extracts **(aspect, opinion, polarity)** triples:

"The **doctor** was kind but the **waiting room** was dirty."
→ (doctor, kind, positive), (waiting room, dirty, negative).

Approaches: sequence labelling to find aspect terms, then classification of each aspect's polarity with the aspect as additional input (e.g. "[CLS] sentence [SEP] waiting room [SEP]" to a BERT classifier), or generative models that output the triples directly. For service improvement, ABSA is far more actionable than overall polarity.

```python
from transformers import pipeline
clf = pipeline("sentiment-analysis", model="cardiffnlp/twitter-xlm-roberta-base-sentiment")
print(clf(["The registration process was quick and staff were helpful.",
           "Waited 5 hours and nobody explained anything."]))
```

## Evaluation

Use macro-F1 (neutral is often the hardest and most frequent class), per-aspect metrics for ABSA, and correlation for intensity. Most importantly, **evaluate on your domain** — a model trained on movie reviews can perform poorly on health-service feedback or on another dialect.

## Responsible use

:::warning
Sentiment scores are noisy summaries, not measurements of people. Risks include:
- **Bias**: models have been shown to assign different sentiment to otherwise identical sentences mentioning different demographic groups or written in different dialects.
- **Misinterpretation**: dashboards of "average sentiment" can hide important minority complaints; always read representative examples.
- **Surveillance**: monitoring individuals' emotions (e.g. employees or protesters) raises serious ethical and legal concerns; some regulations restrict emotion recognition in workplaces and schools.
Use sentiment analysis to find patterns worth reading, not to judge individuals.
:::

:::exercise
1. Collect 50 feedback messages from a domain you know, label them, and compare VADER, TF-IDF + logistic regression and a pretrained transformer.
2. Build test pairs that differ only in a group identity term (e.g. names from different communities) and check whether sentiment predictions change.
3. Design an aspect list for feedback about a public service and annotate 30 messages with aspect-level sentiment.
:::

:::takeaway
- Sentiment can be analysed at document, sentence, aspect and emotion levels.
- Lexicons (VADER) are transparent; fine-tuned transformers handle context best.
- Negation, sarcasm, implicit sentiment, domain shift and code-mixing make the task hard.
- Aspect-based analysis is more actionable; audit for bias and avoid judging individuals.
:::

=== POST ===
slug: named-entity-recognition
title: "Named Entity Recognition: Finding People, Places and Organisations"
category: nlp
level: Intermediate
tags: ner, sequence labeling, bio tagging, information extraction, spacy
summary: NER locates and classifies entity mentions in text. We cover BIO tagging, feature-based and neural approaches, transformer token classification with subword alignment, entity-level evaluation, and domain adaptation.
---
"Janin A Apurba, an ICT officer at CNRS-UNHCR, delivered a workshop in Dhaka on 12 March." A **Named Entity Recognition (NER)** system should identify *Janin A Apurba* as a PERSON, *CNRS-UNHCR* as an ORGANISATION, *Dhaka* as a LOCATION and *12 March* as a DATE. NER is the foundation of **information extraction**: turning unstructured text into structured records for search, knowledge graphs, analytics and redaction of personal data.

## Entity types

Standard schemes (e.g. CoNLL-2003): PER, ORG, LOC, MISC. Richer schemes (OntoNotes) add DATE, TIME, MONEY, PERCENT, GPE (geo-political entity), PRODUCT, EVENT and more. Domains define their own: diseases, drugs and dosages in clinical text; case numbers and document types in administrative records.

## Sequence labelling with BIO tags

NER is framed as **token classification** with the **BIO** (or IOB2) scheme:

- **B-TYPE**: beginning of an entity;
- **I-TYPE**: inside (continuation) of an entity;
- **O**: outside any entity.

| Token | Janin | A | Apurba | works | at | CNRS-UNHCR | in | Dhaka |
|---|---|---|---|---|---|---|---|---|
| Tag | B-PER | I-PER | I-PER | O | O | B-ORG | O | B-LOC |

BIO tags let adjacent entities of the same type be separated. **BIOES/BILOU** adds explicit end and single-token tags and sometimes improves accuracy.

## Approaches through history

1. **Rules and gazetteers** — lists of names and patterns ("Mr. X", capitalised words after "in"). High precision on known names, poor coverage.
2. **Feature-based statistical models** — HMMs, then **Conditional Random Fields (CRFs)** with features such as the word, its shape ("Xxxx", "dd-dd"), prefixes/suffixes, capitalisation, neighbouring words and gazetteer membership.
3. **BiLSTM-CRF** (Lample et al., 2016) — word and character embeddings, a bidirectional LSTM, and a CRF layer enforcing valid tag transitions (e.g. I-PER cannot follow B-ORG).
4. **Fine-tuned transformers** — BERT-style encoders with a token-classification head; the current standard.
5. **LLM-based extraction** — prompting generative models to list entities; flexible for new entity types but needs careful evaluation.

## Transformers and subword alignment

Subword tokenisers split words: "Apurba" might become "Ap", "##ur", "##ba". Labels are per **word**, so we must align them: typically, give the label to the **first** subword of each word and ignore (label −100) the remaining subwords when computing the loss.

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

tok = AutoTokenizer.from_pretrained("bert-base-cased")
words = ["Janin", "A", "Apurba", "works", "at", "CNRS-UNHCR", "in", "Dhaka"]
tags  = ["B-PER", "I-PER", "I-PER", "O", "O", "B-ORG", "O", "B-LOC"]
label2id = {l: i for i, l in enumerate(["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"])}

enc = tok(words, is_split_into_words=True)
aligned, prev = [], None
for wid in enc.word_ids():
    if wid is None:
        aligned.append(-100)                     # special tokens
    elif wid != prev:
        aligned.append(label2id[tags[wid]])      # first subword gets the label
    else:
        aligned.append(-100)                     # other subwords ignored in the loss
    prev = wid
print(list(zip(tok.convert_ids_to_tokens(enc["input_ids"]), aligned)))

ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
print(ner("Janin A Apurba presented the results in Dhaka for UNHCR."))
```

Fine-tune with `AutoModelForTokenClassification` exactly like sequence classification, using the aligned labels.

## Evaluation: entity-level F1

Token-level accuracy is misleading (most tokens are "O"). The standard is **entity-level (exact-match) precision, recall and F1**: a predicted entity counts only if both its span and its type exactly match the gold entity (as implemented in `seqeval`). Partial-match metrics give additional insight for boundary errors.

```python
from seqeval.metrics import classification_report
gold = [["B-PER", "I-PER", "O", "B-LOC"]]
pred = [["B-PER", "O", "O", "B-LOC"]]            # truncated person span -> counts as an error
print(classification_report(gold, pred))
```

## Challenges

- **Ambiguity**: "Jordan" (person, country, river); "Apple" (fruit, company).
- **Nested entities**: "[[Dhaka] University]" — a location inside an organisation; flat BIO cannot represent nesting.
- **Domain shift**: news-trained models perform poorly on medical notes, social media or legal text.
- **Low-resource languages and scripts**: capitalisation, a key English cue, does not exist in Bangla, Arabic or Hindi.
- **Name diversity**: models trained mostly on Western names may underperform on names from other cultures — an important fairness issue, e.g. when NER drives record matching.

## Applications

Knowledge-graph construction, search and question answering, news analytics, clinical information extraction, **de-identification/anonymisation** (finding and masking names, phone numbers and ID numbers before data sharing), resume parsing and automatically populating case-management forms.

:::tip
For de-identification, **recall** is what matters — a missed name is a privacy breach. Combine a learned NER model with rules for structured identifiers (phone numbers, ID formats), and manually audit samples before releasing data.
:::

:::exercise
1. Convert five sentences you write into BIO format, including one with two adjacent entities of the same type.
2. Fine-tune a multilingual transformer (e.g. XLM-R) for NER on a dataset in a language other than English and report entity-level F1.
3. Evaluate a pretrained English NER model on 30 sentences containing names from diverse cultures. Where does it fail?
:::

:::takeaway
- NER is token classification with BIO tags; spans and types must both be correct.
- Approaches evolved from rules and CRFs to BiLSTM-CRF and fine-tuned transformers.
- Align word labels to the first subword; evaluate with entity-level F1.
- Ambiguity, nesting, domain shift, scripts and name diversity are the key challenges.
:::

=== POST ===
slug: pos-tagging-and-crf
title: "Part-of-Speech Tagging and Conditional Random Fields"
category: nlp
level: Advanced
tags: pos tagging, crf, sequence labeling, hmm, viterbi
summary: POS tagging assigns grammatical categories to words. We compare HMM taggers with discriminative Conditional Random Fields, derive the CRF likelihood and Viterbi decoding, and see why CRF layers still appear on top of neural encoders.
---
Is "book" a noun ("a book") or a verb ("book a ticket")? **Part-of-speech (POS) tagging** assigns each word its grammatical category — noun, verb, adjective, preposition and so on. It supports parsing, lemmatisation, information extraction and text-to-speech (pronunciation can depend on POS: "record" as noun vs verb). It is also the classic testbed for **structured prediction**, where outputs are interdependent. This lecture introduces the **Conditional Random Field (CRF)**, one of the most important models for such problems.

## Tag sets

The Penn Treebank uses 45 fine-grained English tags (NN, NNS, VB, VBD, JJ, …). **Universal Dependencies** defines 17 universal POS tags (NOUN, VERB, ADJ, ADP, PRON, DET, …) used consistently across more than 100 languages — ideal for multilingual work.

## Why context matters

A tagger that assigns each word its most frequent tag already achieves about 90% accuracy on English news, but context resolves the rest: "can" after "I" is an auxiliary verb; after "the" it is a noun. Tags also constrain each other: a determiner is usually followed by an adjective or noun.

## Generative approach: HMM taggers

An HMM models tags as hidden states and words as emissions:

$$
P(\mathbf{w}, \mathbf{t}) = \prod_i P(t_i \mid t_{i-1})\,P(w_i \mid t_i)
$$

Decoding uses the Viterbi algorithm (see the HMM lecture). HMMs struggle to use rich, overlapping features (suffixes, capitalisation, neighbouring words) because they must model how words are generated.

## Discriminative approach: linear-chain CRFs

Lafferty, McCallum and Pereira (2001) proposed modelling the conditional distribution of the whole tag sequence directly:

$$
P(\mathbf{y} \mid \mathbf{x}) = \frac{1}{Z(\mathbf{x})}\exp\left(\sum_{i=1}^{n}\Big[\underbrace{\psi(y_i, \mathbf{x}, i)}_{\text{emission score}} + \underbrace{A_{y_{i-1}, y_i}}_{\text{transition score}}\Big]\right)
$$

- **Emission scores** $\psi$ can use **any features of the entire input** — the current word, its suffix, neighbouring words, capitalisation, gazetteers — combined linearly (classical CRF) or produced by a neural network (BiLSTM-CRF, BERT-CRF).
- **Transition scores** $A$ capture which tag sequences are plausible.
- The **partition function** $Z(\mathbf{x}) = \sum_{\mathbf{y}'}\exp(\text{score}(\mathbf{x}, \mathbf{y}'))$ sums over all $K^n$ tag sequences — computed efficiently in $O(nK^2)$ with the **forward algorithm**.

Because it normalises over **whole sequences** (globally), a CRF avoids the **label bias problem** of locally normalised models like maximum-entropy Markov models.

## Training and decoding

**Training** maximises the conditional log-likelihood:

$$
\log P(\mathbf{y} \mid \mathbf{x}) = \text{score}(\mathbf{x}, \mathbf{y}) - \log Z(\mathbf{x})
$$

Its gradient equals observed feature counts minus expected feature counts under the model — computed with the forward–backward algorithm.

**Decoding** finds $\arg\max_{\mathbf{y}}\text{score}(\mathbf{x}, \mathbf{y})$ with **Viterbi** in $O(nK^2)$.

```python
import torch

def crf_log_partition(emissions, transitions):
    """emissions: (n, K) scores; transitions: (K, K) with transitions[i, j] = score(i -> j)."""
    alpha = emissions[0]
    for t in range(1, len(emissions)):
        alpha = torch.logsumexp(alpha[:, None] + transitions, dim=0) + emissions[t]
    return torch.logsumexp(alpha, dim=0)

def crf_score(emissions, transitions, tags):
    s = emissions[0, tags[0]]
    for t in range(1, len(tags)):
        s = s + transitions[tags[t - 1], tags[t]] + emissions[t, tags[t]]
    return s

def viterbi(emissions, transitions):
    score, back = emissions[0], []
    for t in range(1, len(emissions)):
        total = score[:, None] + transitions          # (K_prev, K_cur)
        score, idx = total.max(dim=0)
        score = score + emissions[t]; back.append(idx)
    best = [int(score.argmax())]
    for idx in reversed(back):
        best.append(int(idx[best[-1]]))
    return best[::-1]

n, K = 6, 4
em, tr = torch.randn(n, K), torch.randn(K, K)
gold = [0, 1, 1, 2, 3, 0]
nll = crf_log_partition(em, tr) - crf_score(em, tr, gold)     # negative log-likelihood
print("NLL:", nll.item(), " Viterbi path:", viterbi(em, tr))
```

## A feature-based CRF tagger in practice

```python
# pip install sklearn-crfsuite
import sklearn_crfsuite

def word2features(sent, i):
    w = sent[i]
    f = {"lower": w.lower(), "suffix3": w[-3:], "suffix2": w[-2:], "is_title": w.istitle(),
         "is_digit": w.isdigit(), "shape": "".join("X" if c.isupper() else "x" if c.islower() else "d"
                                                  if c.isdigit() else c for c in w)[:6]}
    f["prev"] = sent[i - 1].lower() if i > 0 else "<BOS>"
    f["next"] = sent[i + 1].lower() if i < len(sent) - 1 else "<EOS>"
    return f

train_sents = [["I", "can", "book", "a", "flight"], ["The", "book", "is", "on", "the", "table"]]
train_tags = [["PRON", "AUX", "VERB", "DET", "NOUN"], ["DET", "NOUN", "AUX", "ADP", "DET", "NOUN"]]
X = [[word2features(s, i) for i in range(len(s))] for s in train_sents]
crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.1, c2=0.1, max_iterations=100).fit(X, train_tags)
test = ["They", "book", "the", "room"]
print(crf.predict([[word2features(test, i) for i in range(len(test))]]))
```

(With a real treebank such as a Universal Dependencies corpus, such taggers reach well above 95% accuracy on English.)

## CRFs in the neural era

Modern taggers use a transformer encoder to produce emission scores. Adding a CRF layer on top still helps when **label dependencies** are strong and data is limited — e.g. NER with BIO constraints, where the CRF prevents invalid sequences like O → I-PER. With large pretrained encoders and plenty of data, the gain is often small, and a simple softmax per token is common.

:::note
POS tagging accuracy on well-resourced languages is now very high (around 97–98% for English), close to the inter-annotator ceiling. For low-resource languages, cross-lingual transfer from multilingual models and Universal Dependencies treebanks makes useful taggers possible with little labelled data.
:::

:::exercise
1. Explain the label bias problem and why global normalisation in CRFs avoids it.
2. Verify the forward algorithm by brute-force enumeration of all $K^n$ tag sequences for $n = 4$, $K = 3$.
3. Train a CRF POS tagger on a Universal Dependencies treebank for a language of your choice and compare with a fine-tuned multilingual transformer.
:::

:::takeaway
- POS tagging assigns grammatical categories; context resolves ambiguity.
- HMMs are generative; linear-chain CRFs model $P(\mathbf{y} \mid \mathbf{x})$ with arbitrary features and transition scores.
- The forward algorithm computes the partition function; Viterbi finds the best sequence, both in $O(nK^2)$.
- CRF layers on neural encoders enforce label consistency, useful for NER with limited data.
:::

=== POST ===
slug: neural-machine-translation
title: "Neural Machine Translation: From Seq2Seq to Multilingual Transformers"
category: nlp
level: Intermediate
tags: machine translation, nmt, bleu, multilingual, low-resource
summary: Machine translation is one of NLP's oldest and most impactful tasks. We trace its evolution to neural systems, cover training data, subword vocabularies, back-translation, evaluation with BLEU and COMET, and the challenges of low-resource languages.
---
Machine translation (MT) was one of the first proposed applications of computers, and today it lets billions of people read content in languages they do not speak. For displaced people navigating services in a new country, good translation can be the difference between understanding one's rights and not. This lecture follows MT from its statistical roots to modern neural and massively multilingual systems — and examines where it still fails.

## A brief history

1. **Rule-based MT** (1950s–1980s): dictionaries and hand-written grammar transfer rules. Brittle and expensive to build.
2. **Statistical MT (SMT)** (1990s–2016): IBM alignment models and **phrase-based** systems learned translation probabilities from parallel corpora, combined in a log-linear model with an n-gram language model. Google Translate used phrase-based SMT for years.
3. **Neural MT (NMT)** (2014–): sequence-to-sequence models with attention (Bahdanau et al., 2015) quickly surpassed SMT. In 2016, Google's GNMT reported large reductions in translation errors for major language pairs.
4. **Transformer NMT** (2017–): the Transformer was introduced *for* translation and became the standard.
5. **Massively multilingual and LLM-based MT** (2019–): single models translating between hundreds of languages; large language models translating via prompting.

## The NMT model

An encoder reads the source sentence; a decoder generates the target autoregressively, attending to the encoder states:

$$
P(\mathbf{y} \mid \mathbf{x}) = \prod_{t=1}^{T}P(y_t \mid y_{<t}, \mathbf{x})
$$

Training maximises the likelihood of reference translations (cross-entropy with teacher forcing, often with label smoothing). Decoding uses **beam search** with length normalisation.

## Data

- **Parallel corpora**: sentence-aligned translations — parliamentary proceedings, UN documents, subtitles, religious texts, web-mined pairs.
- **Quality filtering** matters: web-mined data contains misalignments and machine-translated text.
- **Shared subword vocabularies** (BPE/SentencePiece) across source and target languages let rare words and names be copied or transliterated.

## Back-translation

Monolingual target-language text is far more abundant than parallel text. **Back-translation** (Sennrich et al., 2016): train a reverse model (target → source), translate monolingual target sentences into synthetic sources, and add the synthetic pairs to training. The decoder learns from real, fluent target text. It is one of the most effective techniques, especially for low-resource pairs.

## Multilingual NMT

A single model can translate among many languages by prepending a **target-language tag** (e.g. `<2bn>`). Benefits: **transfer** from high-resource to related low-resource languages, and even **zero-shot** translation between pairs never seen together. Meta's **NLLB-200** ("No Language Left Behind", 2022) supports 200 languages, trained with mined data and evaluated on the FLORES-200 benchmark covering many under-served languages.

```python
from transformers import pipeline
translator = pipeline("translation", model="facebook/nllb-200-distilled-600M",
                      src_lang="eng_Latn", tgt_lang="ben_Beng")
print(translator("Registration is free. Please bring an identity document if you have one.",
                 max_length=100)[0]["translation_text"])
```

## Evaluation

**BLEU** (Papineni et al., 2002) measures n-gram overlap between the output and reference translations:

$$
\text{BLEU} = \text{BP}\cdot\exp\left(\sum_{n=1}^{4}\frac{1}{4}\log p_n\right), \qquad \text{BP} = \min\left(1, e^{1 - r/c}\right)
$$

where $p_n$ is modified n-gram precision and BP penalises translations shorter than the reference ($r$ = reference length, $c$ = candidate length). BLEU is cheap and reproducible (use **sacreBLEU** for standardised scores) but correlates imperfectly with human judgement: valid paraphrases score poorly.

**chrF** uses character n-grams — better for morphologically rich languages. **Learned metrics** such as **COMET** and BLEURT, trained to predict human quality judgements from embeddings, correlate substantially better with humans and are now standard alongside BLEU. **Human evaluation** (e.g. direct assessment, MQM error annotation) remains the gold standard.

```python
import sacrebleu
refs = [["The clinic opens at nine in the morning."]]
hyp = ["The clinic opens at 9 am."]
print(sacrebleu.corpus_bleu(hyp, refs).score, sacrebleu.corpus_chrf(hyp, refs).score)
```

## Persistent challenges

- **Low-resource languages**: little parallel data, poor tokenisation, few evaluation sets.
- **Domain shift**: medical, legal and humanitarian terminology.
- **Hallucinations**: fluent output unrelated to the source, especially under domain shift or with noisy input.
- **Gender and bias**: translating from gender-neutral languages (e.g. Bangla's "সে", Turkish "o") into English often defaults to stereotypes ("he is a doctor, she is a nurse").
- **Formality, culture and idioms**.
- **Document-level context**: pronouns and terminology consistency across sentences.

:::warning
In high-stakes settings — medical consent, legal proceedings, asylum interviews — raw machine translation errors can seriously harm people. Use MT to assist, not replace, qualified human interpreters and translators; provide glossaries, post-editing workflows and clear disclosure that text was machine-translated.
:::

:::exercise
1. Compute BLEU by hand (unigram and bigram precision with brevity penalty) for a short candidate and reference.
2. Translate 20 sentences from a domain you know with two MT systems; score with chrF and COMET, and rate them yourself. Do the metrics agree with you?
3. Test gender bias by translating gender-neutral sentences about various professions from Bangla or Turkish into English.
:::

:::takeaway
- MT evolved from rules to phrase-based statistics to neural encoder–decoders and transformers.
- Back-translation exploits monolingual data; multilingual models transfer to low-resource languages.
- BLEU and chrF measure overlap; learned metrics like COMET correlate better with humans.
- Low-resource languages, domain shift, hallucination and bias remain; keep humans in high-stakes loops.
:::

=== POST ===
slug: transformer-architecture-explained
title: "The Transformer Architecture Explained, Block by Block"
category: nlp
level: Intermediate
tags: transformer, attention, encoder-decoder, multi-head attention, deep learning
summary: The 2017 Transformer replaced recurrence with attention and became the foundation of modern AI. We walk through embeddings, positional encoding, multi-head self-attention, feed-forward layers, residuals, normalisation, masking and the encoder–decoder design.
---
In June 2017, eight researchers at Google published "Attention Is All You Need". Their **Transformer** dispensed with recurrence entirely, relying on attention to relate every position in a sequence to every other. It trained faster than RNNs, achieved state-of-the-art translation, and — within a few years — became the architecture behind BERT, GPT, T5, Vision Transformers, speech models, protein models and today's large language models. This is the most important architecture in modern AI, so we will study it carefully.

## The big picture

The original Transformer is an **encoder–decoder** model for translation:

- The **encoder** maps the source tokens to contextual representations using a stack of $N = 6$ identical layers.
- The **decoder** generates the target one token at a time using a stack of $N = 6$ layers that attend both to previously generated tokens and to the encoder output.

Base model dimensions: $d_{\text{model}} = 512$, 8 attention heads, feed-forward inner size 2048.

## Step 1: token embeddings

Tokens (subwords) are mapped to vectors of dimension $d_{\text{model}}$ by a learned embedding table (scaled by $\sqrt{d_{\text{model}}}$ in the original paper).

## Step 2: positional encoding

Self-attention treats its input as a **set** — it has no notion of order. The Transformer adds a **positional encoding** to each embedding. The original used fixed sinusoids:

$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right), \qquad PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)
$$

Different dimensions oscillate at different frequencies, giving each position a unique signature, and relative offsets correspond to linear transformations. (Modern models often use learned or rotary encodings — see the positional-encoding lecture.)

## Step 3: scaled dot-product attention

Each token's vector is projected into a **query** $\mathbf{q}$, **key** $\mathbf{k}$ and **value** $\mathbf{v}$. For all tokens at once, with matrices $\mathbf{Q}, \mathbf{K}, \mathbf{V}$:

$$
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}
$$

Every token computes a weighted average of all tokens' values, with weights determined by query–key similarity. "The animal didn't cross the street because **it** was too tired": the representation of "it" can draw heavily on "animal".

## Step 4: multi-head attention

One attention pattern is limiting. **Multi-head attention** runs $h$ attention operations in parallel, each with its own learned projections into a smaller subspace ($d_k = d_{\text{model}}/h$), then concatenates and projects the results:

$$
\text{MultiHead}(\mathbf{X}) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)\,\mathbf{W}^O, \qquad \text{head}_i = \text{Attention}(\mathbf{X}\mathbf{W}_i^Q, \mathbf{X}\mathbf{W}_i^K, \mathbf{X}\mathbf{W}_i^V)
$$

Different heads can specialise — some track syntax, some coreference, some adjacent tokens.

## Step 5: position-wise feed-forward network

After attention, each position passes independently through the same two-layer MLP:

$$
\text{FFN}(\mathbf{x}) = \max(0, \mathbf{x}\mathbf{W}_1 + \mathbf{b}_1)\mathbf{W}_2 + \mathbf{b}_2
$$

Attention **mixes information across positions**; the FFN **processes each position**. The FFN holds about two-thirds of a layer's parameters.

## Step 6: residual connections and layer normalisation

Each sub-layer (attention, FFN) is wrapped with a residual connection and LayerNorm. The original used post-norm, $\text{LN}(\mathbf{x} + \text{Sublayer}(\mathbf{x}))$; most modern models use **pre-norm**, $\mathbf{x} + \text{Sublayer}(\text{LN}(\mathbf{x}))$, which trains more stably.

## Step 7: the decoder and masking

Each decoder layer has three sub-layers:

1. **Masked self-attention** over the target prefix. A **causal mask** sets scores for future positions to $-\infty$ before the softmax, so position $t$ cannot peek at tokens $> t$ — preserving the autoregressive property while allowing all positions to be trained in parallel.
2. **Cross-attention**: queries come from the decoder; keys and values come from the **encoder output** — this is how the decoder "looks at" the source sentence.
3. **Feed-forward network**.

A final linear layer and softmax produce next-token probabilities. **Padding masks** also prevent attention to padding tokens in batches of different-length sentences.

## A compact implementation

```python
import math, torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d, h, dropout=0.1):
        super().__init__()
        self.h, self.dk = h, d // h
        self.qkv, self.out, self.drop = nn.Linear(d, 3 * d), nn.Linear(d, d), nn.Dropout(dropout)
    def forward(self, x, mask=None):                        # x: (B, T, d)
        B, T, d = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)  # each (B, h, T, dk)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.dk)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        att = self.drop(F.softmax(scores, dim=-1))
        return self.out((att @ v).transpose(1, 2).reshape(B, T, d))

class Block(nn.Module):                                      # pre-norm transformer block
    def __init__(self, d=256, h=8, ff=1024, dropout=0.1):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = MultiHeadAttention(d, h, dropout)
        self.ffn = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d), nn.Dropout(dropout))
    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask)
        return x + self.ffn(self.ln2(x))

T = 10
causal = torch.tril(torch.ones(T, T)).view(1, 1, T, T)      # lower-triangular mask
x = torch.randn(2, T, 256)
print(Block()(x, causal).shape)                              # (2, 10, 256)
```

## Why the Transformer won

1. **Parallelism**: all positions are processed simultaneously during training — ideal for GPUs — unlike sequential RNNs.
2. **Short paths**: any two positions interact in **one** layer, making long-range dependencies easier to learn.
3. **Scalability**: performance keeps improving with more data, parameters and compute (scaling laws).
4. **Generality**: the same block works for text, images (patches), audio, proteins and more.

The main cost: self-attention is $O(T^2)$ in sequence length in time and memory — addressed by efficient attention methods covered later.

## Three families

| Family | Architecture | Pretraining | Examples | Best for |
|---|---|---|---|---|
| Encoder-only | Bidirectional encoder | Masked language modelling | BERT, RoBERTa | Classification, extraction, embeddings |
| Decoder-only | Causal decoder | Next-token prediction | GPT family, LLaMA | Generation, general-purpose LLMs |
| Encoder–decoder | Both + cross-attention | Denoising / span corruption | T5, BART, original Transformer | Translation, summarisation |

:::exercise
1. Count the parameters of one encoder layer with $d = 512$, $h = 8$, FFN size 2048 (include biases and LayerNorm).
2. Show that without positional encodings, permuting the input tokens simply permutes the encoder outputs.
3. Train a tiny decoder-only transformer on character-level text (e.g. a public-domain book) and sample from it.
:::

:::takeaway
- The Transformer relies on attention, not recurrence: embeddings + positional encodings → stacked blocks of multi-head attention and feed-forward layers with residuals and normalisation.
- Scaled dot-product attention computes softmax$(\mathbf{Q}\mathbf{K}^\top/\sqrt{d_k})\mathbf{V}$; multiple heads capture different relations.
- The decoder uses causal masking and cross-attention to the encoder.
- Parallelism, short paths and scalability made it the foundation of modern AI.
:::

=== POST ===
slug: self-attention-in-depth
title: "Self-Attention in Depth: Intuition, Complexity and Variants"
category: nlp
level: Advanced
tags: self-attention, multi-head attention, kv cache, grouped query attention, complexity
summary: A deeper look at self-attention — what attention heads learn, the geometry of queries and keys, computational complexity, causal masking, KV caching, and efficient variants such as multi-query and grouped-query attention.
---
In the previous lecture we assembled the Transformer. Now we zoom into its core operation. Self-attention is deceptively simple — three matrix multiplications and a softmax — but understanding its behaviour, costs and modern variants is essential for anyone building or deploying transformer models.

## Self-attention as learned, dynamic routing

For an input sequence $\mathbf{X} \in \mathbb{R}^{T \times d}$:

$$
\mathbf{Q} = \mathbf{X}\mathbf{W}^Q, \quad \mathbf{K} = \mathbf{X}\mathbf{W}^K, \quad \mathbf{V} = \mathbf{X}\mathbf{W}^V, \qquad \mathbf{A} = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right), \quad \mathbf{Y} = \mathbf{A}\mathbf{V}
$$

Contrast with other layers:

- A **fully connected layer** mixes features with **fixed** weights.
- A **convolution** mixes **neighbouring** positions with fixed weights.
- **Self-attention** mixes **all** positions with weights $\mathbf{A}$ that are **computed from the input itself** — the connectivity pattern changes for every sentence.

Row $t$ of $\mathbf{A}$ is a probability distribution saying how much token $t$ reads from each other token.

## Why separate queries, keys and values?

Using the raw embeddings for everything would make attention symmetric ($\mathbf{x}_i^\top\mathbf{x}_j = \mathbf{x}_j^\top\mathbf{x}_i$) and force each token to attend most to itself. Separate projections let a token **ask** for one kind of information (query) while **advertising** another (key) and **providing** a third (value). A verb's query might seek its subject; a noun's key might advertise "I am a plausible subject".

## Why scale by $\sqrt{d_k}$?

If query and key components are independent with zero mean and unit variance, $\mathbf{q}^\top\mathbf{k}$ has variance $d_k$. Large scores push the softmax into saturated regions where gradients vanish. Dividing by $\sqrt{d_k}$ restores unit variance.

## What do attention heads learn?

Analyses of trained models (e.g. Clark et al., 2019 on BERT; mechanistic studies of GPT-style models) found heads that:

- attend to the **previous** or **next** token;
- track **syntactic relations** (a verb attending to its direct object, determiners to their nouns);
- resolve **coreference** (pronouns to their antecedents);
- attend heavily to special tokens like [SEP] or the first token — often acting as a "no-op" or **attention sink**;
- form **induction heads** in decoder models — circuits that look for a previous occurrence of the current token and copy what followed it, a mechanism linked to in-context learning.

Many heads are redundant; studies have pruned a large fraction of heads with little loss.

## Complexity

For sequence length $T$ and dimension $d$:

| Operation | Time | Memory |
|---|---|---|
| Projections ($\mathbf{Q}, \mathbf{K}, \mathbf{V}$) | $O(Td^2)$ | $O(Td)$ |
| Scores $\mathbf{Q}\mathbf{K}^\top$ | $O(T^2d)$ | $O(T^2)$ per head |
| Weighted sum $\mathbf{A}\mathbf{V}$ | $O(T^2d)$ | — |

For short sequences the $Td^2$ projections dominate; for long sequences the **quadratic** $T^2$ term dominates. Doubling context length quadruples attention cost — the main obstacle to long-context models. **FlashAttention** computes exact attention in tiles without materialising the $T \times T$ matrix in slow GPU memory, greatly reducing memory and wall-clock time (next lectures cover efficient transformers).

## Causal attention and the KV cache

In decoder-only models, token $t$ attends only to positions $\le t$. During **generation**, we produce one token at a time. Recomputing keys and values for the whole prefix at each step would be wasteful, since they do not change. The **KV cache** stores each layer's keys and values for previous tokens; each new step computes $\mathbf{q}, \mathbf{k}, \mathbf{v}$ only for the new token and attends over the cache.

The cache grows linearly with context length, and its memory can dominate inference:

$$
\text{KV memory} = 2 \times L \times T \times n_{\text{kv}} \times d_{\text{head}} \times \text{bytes per value} \times \text{batch}
$$

(factor 2 for keys and values, $L$ layers, $n_{\text{kv}}$ key/value heads).

## Multi-query and grouped-query attention

To shrink the KV cache:

- **Multi-Query Attention (MQA)** (Shazeer, 2019): all query heads share a **single** key and value head — the cache shrinks by a factor of $h$, speeding decoding, with some quality loss.
- **Grouped-Query Attention (GQA)** (Ainslie et al., 2023): query heads are divided into $g$ groups, each sharing one key/value head — a middle ground adopted by many modern LLMs.
- **Multi-head Latent Attention (MLA)**: compresses keys and values into a low-dimensional latent vector that is cached, further reducing memory.

```python
import torch

def kv_cache_gb(layers, context, kv_heads, head_dim, batch=1, bytes_per=2):
    return 2 * layers * context * kv_heads * head_dim * batch * bytes_per / 1e9

# A hypothetical 32-layer model, head_dim 128, 32 query heads, 32K context, fp16
print("MHA (32 KV heads):", round(kv_cache_gb(32, 32_768, 32, 128), 1), "GB per sequence")
print("GQA (8 KV heads): ", round(kv_cache_gb(32, 32_768, 8, 128), 1), "GB per sequence")
print("MQA (1 KV head):  ", round(kv_cache_gb(32, 32_768, 1, 128), 2), "GB per sequence")
```

## Cross-attention

In encoder–decoder models (and in multimodal models that attend to image features), queries come from one sequence and keys/values from another. The same equation, a different source of $\mathbf{K}$ and $\mathbf{V}$.

:::note
Attention weights are tempting to interpret, but as discussed in the explainability lectures, high attention weight is not the same as causal importance: values can be small, later layers can override earlier ones, and information is mixed across heads and layers. Interpretability research uses more rigorous tools (activation patching, attribution methods, sparse autoencoders).
:::

:::exercise
1. Show that without separate query/key projections ($\mathbf{W}^Q = \mathbf{W}^K$), the unnormalised score matrix is symmetric.
2. Measure wall-clock time and memory of naive attention in PyTorch for $T = 512, 1024, 2048, 4096$, and compare with `torch.nn.functional.scaled_dot_product_attention`.
3. Implement a KV cache for the tiny decoder from the previous lecture and measure generation speed with and without it.
:::

:::takeaway
- Self-attention mixes all positions with input-dependent weights; separate Q/K/V projections allow asymmetric, specialised interactions.
- Heads learn positional, syntactic, coreference and induction patterns; many are redundant.
- Attention costs $O(T^2)$ in sequence length; FlashAttention reduces memory traffic.
- The KV cache speeds generation; MQA, GQA and MLA shrink it.
:::

=== POST ===
slug: positional-encodings-rope-alibi
title: "Positional Encodings: Sinusoidal, Learned, RoPE and ALiBi"
category: nlp
level: Advanced
tags: positional encoding, rope, alibi, length generalization, transformers
summary: Attention is order-blind, so transformers need positional information. We compare absolute sinusoidal and learned encodings with relative methods — rotary embeddings (RoPE) and ALiBi — and discuss extending context length.
---
Self-attention is **permutation-equivariant**: shuffle the input tokens and the outputs are shuffled the same way, with no other change. Without positional information, "dog bites man" and "man bites dog" would look identical to a transformer. How we inject position has a large effect on how well models handle long contexts and whether they can extrapolate to sequences longer than those seen in training.

## Absolute positional encodings

### Sinusoidal (original Transformer)

$$
PE_{(p, 2i)} = \sin\left(\frac{p}{10000^{2i/d}}\right), \qquad PE_{(p, 2i+1)} = \cos\left(\frac{p}{10000^{2i/d}}\right)
$$

added to token embeddings. Each pair of dimensions is a rotating 2-D vector with a different frequency — like the hands of a clock running at many speeds. Key property: $PE_{p+k}$ is a **linear function** (a rotation) of $PE_p$, which in principle lets the model reason about relative offsets. No parameters, and defined for any position.

### Learned absolute embeddings

A trainable vector per position (BERT, GPT-2). Flexible, but undefined beyond the maximum trained length (e.g. 512 or 1024 positions) — the model cannot extrapolate.

## Relative position methods

What usually matters linguistically is **relative** distance ("the word two positions back"), not absolute index. Shaw et al. (2018) added learned relative-position terms to attention scores; **T5** uses a learned scalar **bias** per relative-distance bucket added to attention logits.

## Rotary Position Embedding (RoPE)

**RoPE** (Su et al., 2021) is used by LLaMA, Mistral, Qwen, and many other modern LLMs. Instead of adding a vector to embeddings, it **rotates** query and key vectors by an angle proportional to their position. Split a $d$-dimensional vector into $d/2$ pairs; rotate pair $i$ at position $p$ by angle $p\theta_i$ with $\theta_i = 10000^{-2i/d}$:

$$
\begin{bmatrix} q'_{2i} \\ q'_{2i+1} \end{bmatrix} = \begin{bmatrix} \cos p\theta_i & -\sin p\theta_i \\ \sin p\theta_i & \cos p\theta_i \end{bmatrix}\begin{bmatrix} q_{2i} \\ q_{2i+1} \end{bmatrix}
$$

Because rotations compose, the dot product between a query at position $m$ and a key at position $n$ depends only on their **relative offset** $m - n$:

$$
\langle R_m\mathbf{q},\, R_n\mathbf{k}\rangle = \langle\mathbf{q},\, R_{n-m}\mathbf{k}\rangle
$$

RoPE thus encodes absolute position in the representation while making attention **relative** — with no extra parameters, compatible with efficient attention kernels and the KV cache.

```python
import torch

def rope(x, base=10000.0):
    """x: (..., T, d) with even d. Rotates pairs of dimensions by position-dependent angles."""
    T, d = x.shape[-2], x.shape[-1]
    theta = base ** (-torch.arange(0, d, 2, dtype=torch.float32) / d)        # (d/2,)
    angles = torch.arange(T, dtype=torch.float32)[:, None] * theta[None]      # (T, d/2)
    cos, sin = angles.cos(), angles.sin()
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out

# Check the relative property: score depends only on the offset
q, k = torch.randn(1, 64), torch.randn(1, 64)
seq_q = rope(q.repeat(50, 1)); seq_k = rope(k.repeat(50, 1))
print((seq_q[10] @ seq_k[7]).item(), (seq_q[30] @ seq_k[27]).item())   # equal: both offset 3
```

## ALiBi: attention with linear biases

**ALiBi** (Press, Smith & Lewis, 2022) uses no positional embeddings at all. It adds a fixed, head-specific **linear penalty** to attention scores proportional to distance:

$$
\text{score}(i, j) = \mathbf{q}_i^\top\mathbf{k}_j - m_h\cdot(i - j)
$$

with slopes $m_h$ forming a geometric sequence across heads (e.g. $1/2, 1/4, \dots$). Distant tokens are penalised more; different heads attend at different ranges. ALiBi showed strong **length extrapolation**: models trained on short sequences performed well on longer ones.

## Comparison

| Method | Type | Parameters | Extrapolation | Used in |
|---|---|---|---|---|
| Sinusoidal | Absolute, added | None | Limited | Original Transformer |
| Learned | Absolute, added | $T_{\max} \times d$ | None | BERT, GPT-2 |
| T5 relative bias | Relative, score bias | Few | Moderate | T5 |
| RoPE | Relative via rotation | None | Moderate; extendable | LLaMA, many LLMs |
| ALiBi | Relative, linear score bias | None | Good | BLOOM, MPT |

## Extending context length

Training on very long sequences is expensive, so models are often trained at moderate length and then **extended**:

- **Position interpolation** (Chen et al., 2023): rescale positions so a longer sequence maps into the trained range (e.g. divide positions by 4), then fine-tune briefly.
- **NTK-aware scaling** and **YaRN**: rescale RoPE frequencies non-uniformly — preserving high-frequency (local) information while stretching low-frequency (long-range) components.
- Continued pretraining on long documents with the adjusted encoding.

Even with long context windows, models may use information in the middle of long inputs less reliably than at the beginning or end ("lost in the middle", Liu et al., 2023). Evaluate long-context behaviour with retrieval-style tests, not only perplexity.

## NoPE: no positional encoding?

Causal decoder models can partially infer position without explicit encodings, because the causal mask itself breaks symmetry (a token can count how many tokens precede it). Some studies found such models generalise reasonably to longer lengths, though explicit encodings remain standard.

:::exercise
1. Prove that $\langle R_m\mathbf{q}, R_n\mathbf{k}\rangle$ depends only on $m - n$ using properties of 2-D rotation matrices.
2. Plot the ALiBi bias matrix for 8 heads and sequence length 16. How do heads differ?
3. Train tiny transformers with learned, sinusoidal and RoPE encodings on sequences of length 64, and evaluate them on length 256.
:::

:::takeaway
- Attention is order-blind; positional encodings inject order.
- Absolute encodings (sinusoidal, learned) add position vectors; learned ones cannot extrapolate.
- RoPE rotates queries and keys so attention depends on relative offsets; ALiBi adds distance-proportional biases.
- Context can be extended with position interpolation or RoPE frequency scaling; test long-context use directly.
:::

=== POST ===
slug: bert-explained
title: "BERT: Bidirectional Encoder Representations from Transformers"
category: nlp
level: Intermediate
tags: bert, pretraining, masked language modeling, fine-tuning, encoders
summary: BERT showed that a bidirectional transformer pretrained on unlabelled text could be fine-tuned to beat task-specific models across NLP. We cover masked language modelling, input format, fine-tuning patterns, and successors such as RoBERTa, DeBERTa and multilingual encoders.
---
In October 2018, Google researchers released **BERT** (Devlin et al.). Fine-tuned with a single extra layer, it set new state-of-the-art results on eleven NLP benchmarks at once, including question answering and natural-language inference. BERT established the **pretrain-then-fine-tune** paradigm that still dominates NLP, and encoder models descended from it remain the workhorses for classification, extraction and embedding tasks in industry.

## The key idea: deep bidirectionality

Earlier pretrained language models were **unidirectional**: GPT-1 read left-to-right; ELMo concatenated separate left-to-right and right-to-left LSTMs. But understanding a word often requires context on **both** sides: "The **bank** of the river flooded" vs "The **bank** approved the loan". A standard left-to-right language-model objective cannot use right context, and naively conditioning on both sides would let each word "see itself" indirectly through the layers.

BERT's solution: **masked language modelling (MLM)**.

## Pretraining objectives

**1. Masked Language Modelling.** Randomly select 15% of input tokens. Of these:

- 80% are replaced with `[MASK]`;
- 10% are replaced with a random token;
- 10% are left unchanged.

The model predicts the original tokens at the selected positions using the full bidirectional context. The 80/10/10 mix reduces the mismatch between pretraining (which sees `[MASK]`) and fine-tuning (which never does), and forces the model to keep good representations of every token.

**2. Next Sentence Prediction (NSP).** Given sentence pairs A and B, predict whether B actually follows A (50%) or is random (50%). Intended to help tasks involving sentence pairs — later work (RoBERTa) found NSP unnecessary or even harmful.

Pretraining data: BooksCorpus and English Wikipedia (about 3.3 billion words). **BERT-Base**: 12 layers, hidden size 768, 12 heads, 110M parameters. **BERT-Large**: 24 layers, 1024 hidden, 16 heads, 340M parameters.

## Input representation

```text
[CLS] the clinic opens at nine [SEP] where is it located ? [SEP]
```

Each token's input = **token embedding** (WordPiece, 30,000 vocabulary) + **segment embedding** (A or B) + **learned position embedding**. The final hidden state of `[CLS]` serves as an aggregate representation for classification.

## Fine-tuning patterns

| Task | Input | Output layer |
|---|---|---|
| Single-sentence classification (sentiment) | `[CLS] text [SEP]` | Linear on `[CLS]` |
| Sentence-pair classification (entailment, paraphrase) | `[CLS] A [SEP] B [SEP]` | Linear on `[CLS]` |
| Token classification (NER) | `[CLS] text [SEP]` | Linear on every token |
| Extractive QA (SQuAD) | `[CLS] question [SEP] passage [SEP]` | Start/end logits over passage tokens |

All parameters are fine-tuned for a few epochs with a small learning rate (2e-5 to 5e-5).

```python
from transformers import pipeline, AutoTokenizer, AutoModelForMaskedLM
import torch

fill = pipeline("fill-mask", model="bert-base-uncased")
for r in fill("The doctor told the patient to take the [MASK] twice a day.")[:3]:
    print(round(r["score"], 3), r["token_str"])

tok = AutoTokenizer.from_pretrained("bert-base-uncased")
enc = tok("The clinic opens at nine.", "Where is it located?")
print(tok.convert_ids_to_tokens(enc["input_ids"]))
print(enc["token_type_ids"])                 # segment ids: 0 for sentence A, 1 for sentence B
```

## Why BERT worked

1. **Transfer**: general linguistic knowledge from billions of words transfers to tasks with few labels.
2. **Bidirectional context** at every layer.
3. **Minimal task-specific architecture** — one pretrained model, many tasks.

Probing studies found BERT's layers encode a rough hierarchy: surface features in lower layers, syntax in middle layers, and more semantic, task-relevant information in upper layers.

## Successors and improvements

- **RoBERTa** (2019): same architecture, better training — more data (160 GB), longer training, larger batches, dynamic masking, no NSP. Significantly better, showing BERT was **undertrained**.
- **ALBERT** (2019): parameter sharing across layers and factorised embeddings for smaller models.
- **DistilBERT** (2019): knowledge-distilled, 40% smaller and 60% faster while retaining most of BERT's accuracy.
- **ELECTRA** (2020): **replaced token detection** — a small generator replaces some tokens and the main model classifies every token as original or replaced. Learning from all tokens (not only 15%) makes pretraining much more sample-efficient.
- **DeBERTa** (2020–): disentangled attention (separate content and relative-position vectors) and an enhanced mask decoder; strong results on understanding benchmarks.
- **SpanBERT**: masks contiguous spans — better for QA and coreference.
- **Multilingual**: mBERT (104 languages) and **XLM-RoBERTa** (100 languages, 2.5 TB of filtered web data) enable cross-lingual transfer — fine-tune on English labels and apply to other languages. Language-specific models exist for many languages, including **BanglaBERT** for Bangla.
- **Domain-specific**: BioBERT, ClinicalBERT, SciBERT, LegalBERT, FinBERT — continued pretraining on domain text.
- **ModernBERT** (2024): updated encoder with RoPE, longer context and efficient attention.

## Limitations

- **Not generative**: MLM encoders cannot naturally generate fluent long text.
- **Context length**: 512 tokens for original BERT.
- **Pretrain–fine-tune mismatch** from `[MASK]` tokens.
- **Biases** from pretraining text.

:::note
In the LLM era, why use BERT-style encoders? They are **small, fast and cheap**, excellent for classification, NER, retrieval and embeddings, and easy to fine-tune on a single GPU. Many production NLP systems still run encoder models, sometimes with labels bootstrapped by an LLM.
:::

:::exercise
1. Explain why a bidirectional model cannot be trained with a standard next-word objective.
2. Fine-tune BERT and XLM-R on an English classification dataset and evaluate zero-shot on a translated test set in another language.
3. Probe BERT's layers: train logistic regressions on each layer's representations to predict POS tags, and plot accuracy by layer.
:::

:::takeaway
- BERT pretrains a bidirectional transformer encoder with masked language modelling (and originally NSP).
- Inputs combine token, segment and position embeddings; `[CLS]` aggregates for classification.
- Fine-tuning adds a small head for classification, tagging or span extraction.
- RoBERTa, ELECTRA, DeBERTa and multilingual/domain variants improved on it; encoders remain practical workhorses.
:::
