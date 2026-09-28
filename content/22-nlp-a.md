=== POST ===
slug: introduction-to-nlp
title: "Introduction to Natural Language Processing"
category: nlp
level: Beginner
tags: nlp, language, ambiguity, tasks, overview
summary: We open the NLP track with the question of why human language is so hard for machines — ambiguity, context, compositionality — and map the tasks, eras and methods of natural language processing.
---
Welcome to the NLP & Transformers track. Language is humanity's most powerful tool: we use it to share knowledge, coordinate, negotiate, teach and comfort. **Natural Language Processing (NLP)** aims to give machines the ability to understand and generate it. In the last decade, NLP went from a specialised field to the engine of AI's most visible systems. Let us begin by understanding why the problem is hard.

## Why language is hard

**Ambiguity at every level:**
- **Lexical**: "bank" (river or money), "bat" (animal or cricket equipment).
- **Syntactic**: "I saw the man with the telescope" — who has the telescope?
- **Semantic**: "Every student read a book" — the same book or different ones?
- **Pragmatic**: "Can you pass the salt?" is a request, not a question about ability.
- **Referential**: "The trophy didn't fit in the suitcase because *it* was too big."

**Other challenges:**
- **Context and world knowledge** — understanding requires facts not stated in the text.
- **Compositionality** — meanings combine systematically, but idioms ("kick the bucket") break the rules.
- **Variation** — dialects, slang, code-mixing ("Ami office e late hobo"), spelling errors, informal social-media text.
- **Long-tail vocabulary** — names, new words, technical terms.
- **Linguistic diversity** — about 7,000 languages, most with little digital data.

## Levels of linguistic analysis

| Level | Studies | NLP task example |
|---|---|---|
| Phonetics/phonology | Sounds | Speech recognition |
| Morphology | Word structure | Stemming, lemmatisation |
| Syntax | Sentence structure | Part-of-speech tagging, parsing |
| Semantics | Meaning | Word sense disambiguation, semantic role labelling |
| Pragmatics | Meaning in context | Intent detection, dialogue |
| Discourse | Multi-sentence structure | Coreference resolution, summarisation |

## Common NLP tasks

- **Classification**: sentiment, topic, spam, intent, toxicity.
- **Sequence labelling**: part-of-speech tags, named entities.
- **Structured prediction**: parsing, relation extraction.
- **Sequence-to-sequence**: translation, summarisation, paraphrasing.
- **Question answering and retrieval**.
- **Dialogue and conversational agents**.
- **Generation**: stories, code, reports.
- **Speech**: recognition and synthesis.

## Three eras of NLP

1. **Rule-based (1950s–1980s)**: hand-written grammars and dictionaries. The 1954 Georgetown–IBM experiment translated about 60 Russian sentences and inspired over-optimistic predictions. ELIZA (1966) mimicked conversation with patterns.
2. **Statistical (1990s–2010s)**: probabilistic models learned from corpora — n-gram language models, HMM taggers, statistical machine translation, conditional random fields. Frederick Jelinek's quip captured the mood: "Every time I fire a linguist, the performance of the speech recogniser goes up."
3. **Neural (2013–present)**: word embeddings (2013), RNN sequence-to-sequence models and attention (2014–2016), **transformers** (2017), pretrained models like BERT and GPT (2018–), and large language models that perform many tasks from instructions.

## A tiny NLP pipeline

```python
import re
from collections import Counter

text = """Machine learning helps organisations respond faster.
Machine translation helps people read information in their own language."""

tokens = re.findall(r"[a-zA-Z']+", text.lower())
print(tokens[:8])
print(Counter(tokens).most_common(5))

sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
print(len(sentences), "sentences")
```

Even this naive tokenisation hides decisions: what about "don't", hyphens, numbers, emojis, or scripts without spaces between words? We study these in the next lecture.

## NLP and responsibility

Language technology shapes access to information. Speakers of low-resource languages are underserved by many systems; toxic-content filters can disproportionately flag dialects; translation errors in medical, legal or asylum contexts can have serious consequences. Throughout this track we will pay attention to multilingual performance, bias and appropriate human oversight.

:::exercise
1. Find one example of each type of ambiguity (lexical, syntactic, pragmatic) in your own language.
2. List five NLP tasks that could help a public service organisation and state which task type each is.
3. Tokenise a paragraph of Bangla or another language with the regex above. What goes wrong?
:::

:::takeaway
- Language is ambiguous at every level and depends on context and world knowledge.
- NLP spans morphology to discourse and tasks from classification to generation.
- The field moved from rules to statistics to neural networks and pretrained transformers.
- Multilingual coverage and fairness are central concerns.
:::

=== POST ===
slug: text-preprocessing-tokenization
title: "Text Preprocessing: Tokenisation, Normalisation, Stemming and Lemmatisation"
category: nlp
level: Beginner
tags: preprocessing, tokenization, stemming, lemmatization, stopwords
summary: Raw text must be converted into units a model can process. We cover normalisation, word and sentence tokenisation, stop words, stemming versus lemmatisation, and how preprocessing needs differ for classical and neural models.
---
Before any model sees text, we must decide what the basic units are and how to clean them. These choices look mundane but significantly affect results. A preprocessing step that helps a bag-of-words classifier can hurt a transformer; a tokeniser designed for English may fail on Bangla or Chinese. Today we learn the standard toolkit and when to use each tool.

## Normalisation

- **Unicode normalisation** (NFC/NFKC) — the same visible character can have several byte encodings; normalise so they compare equal. Critical for scripts with combining marks (Bangla, Devanagari, Arabic diacritics).
- **Case folding** — lowercasing reduces vocabulary size but loses information ("US" vs "us", "Apple" the company).
- **Removing or replacing** URLs, email addresses, numbers, HTML tags — replace with placeholders (`<URL>`, `<NUM>`) rather than deleting, if they carry signal.
- **Spelling normalisation** for social media ("sooo gooood" → "so good") and transliteration handling for romanised text.

## Tokenisation

**Tokenisation** splits text into units (tokens).

- **Whitespace/rule-based word tokenisation**: split on spaces and punctuation, with rules for contractions ("don't" → "do", "n't"), abbreviations ("Dr."), hyphens, numbers ("3.14"), emails and emoticons.
- **Sentence segmentation**: split at ".", "?", "!", but not in "Dr." or "3.5"; Bangla uses the dari "।" as a full stop.
- **Languages without spaces** (Chinese, Japanese, Thai) require word-segmentation models.
- **Subword tokenisation** (BPE, WordPiece, SentencePiece) — used by all modern neural models; covered in a dedicated lecture.

```python
import nltk
nltk.download("punkt", quiet=True); nltk.download("punkt_tab", quiet=True)
from nltk.tokenize import word_tokenize, sent_tokenize

text = "Dr. Rahman can't attend on 3.5.2025. He'll join online — see https://example.org!"
print(sent_tokenize(text))
print(word_tokenize(text))

import spacy                       # python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")
doc = nlp(text)
print([(t.text, t.lemma_, t.pos_, t.is_stop) for t in doc][:10])
```

## Stop words

**Stop words** are very frequent function words ("the", "is", "of"). Removing them shrinks bag-of-words representations and can help topic modelling or keyword search.

:::warning
Stop-word removal can destroy meaning: "not good" → "good" flips sentiment; "to be or not to be" disappears entirely. For sentiment analysis, question answering and any neural model, **do not** remove stop words by default.
:::

## Stemming vs lemmatisation

Both reduce inflected words to a common base, so "running", "runs" and "ran" can be treated as related.

**Stemming** chops affixes with heuristic rules. The **Porter stemmer** (1980) turns "studies" → "studi", "university" → "univers". Fast, language-specific, and produces non-words; can over-stem ("university" and "universe" → "univers") or under-stem.

**Lemmatisation** uses vocabulary and morphological analysis to return the dictionary form (**lemma**): "studies" → "study", "better" → "good" (as adjective), "was" → "be". More accurate but slower, and it needs part-of-speech information.

| Word | Porter stem | Lemma |
|---|---|---|
| studies | studi | study |
| running | run | run |
| better | better | good (adj.) |
| was | wa | be |

For morphologically rich languages (Bangla, Turkish, Finnish, Arabic), where one root yields many surface forms, morphological analysis matters much more than in English.

## Other classical steps

- **n-grams**: contiguous sequences of $n$ tokens ("machine learning" as a bigram) capture local word order.
- **Part-of-speech tagging** and **named-entity recognition** as features.
- **Handling negation**: marking tokens after "not" ("not_good") in bag-of-words models.

## Classical vs neural preprocessing

| Step | Bag-of-words / TF-IDF models | Pretrained transformers |
|---|---|---|
| Lowercasing | Often helpful | Use the model's own convention (cased/uncased) |
| Stop-word removal | Sometimes | No |
| Stemming/lemmatisation | Often helpful | No — subword tokeniser handles morphology |
| Punctuation removal | Sometimes | No |
| Tokeniser | Word-level | **Must** use the model's own tokeniser |

:::tip
With pretrained transformers, apply only light normalisation (Unicode, whitespace, obvious noise) and then use the model's tokeniser exactly as in pretraining. Heavy "cleaning" removes information the model was trained to use.
:::

## Building a reusable preprocessing function

```python
import re, unicodedata

def normalise(text, lower=True):
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"https?://\S+", " <URL> ", text)
    text = re.sub(r"\S+@\S+\.\S+", " <EMAIL> ", text)
    text = re.sub(r"\d+([.,]\d+)?", " <NUM> ", text)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)          # "sooo" -> "soo"
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower() if lower else text

print(normalise("Sooo happy!!! Contact me at a.b@mail.com or visit https://x.org, costs 1,500"))
```

:::exercise
1. Compare Porter stemming and spaCy lemmatisation on 20 words, noting over- and under-stemming.
2. Train a TF-IDF sentiment classifier with and without stop-word removal. How does "not" affect results?
3. Write a sentence tokeniser that handles the Bangla dari (।) as well as English punctuation.
:::

:::takeaway
- Normalise Unicode and noise carefully; replace rather than delete informative tokens.
- Tokenisation defines the units of text; rules must handle contractions, abbreviations and scripts.
- Stemming chops heuristically; lemmatisation returns dictionary forms using morphology.
- Classical models benefit from cleaning; transformers need light normalisation and their own tokeniser.
:::

=== POST ===
slug: bag-of-words-tf-idf
title: "Bag of Words and TF-IDF: Classical Text Representation"
category: nlp
level: Beginner
tags: bag of words, tf-idf, vector space model, text classification, information retrieval
summary: The simplest way to turn documents into vectors is to count words. We build bag-of-words and TF-IDF representations, derive the IDF formula, use cosine similarity for retrieval, and train strong linear text classifiers.
---
Before embeddings and transformers, the workhorse of NLP and search was the **vector space model**: represent each document by the words it contains. It sounds crude — word order is discarded entirely — yet TF-IDF with a linear classifier remains a surprisingly strong, fast and interpretable baseline, and TF-IDF-style scoring (BM25) still powers many search engines.

## Bag of words (BoW)

Build a vocabulary $V$ of all words in the corpus. Represent each document $d$ as a vector of length $|V|$, where entry $t$ is the **count** of term $t$ in $d$.

| | machine | learning | helps | refugees | read |
|---|---|---|---|---|---|
| "machine learning helps" | 1 | 1 | 1 | 0 | 0 |
| "learning helps refugees read" | 0 | 1 | 1 | 1 | 1 |

Properties: very high-dimensional (tens of thousands of columns), extremely **sparse**, and **order-free** ("dog bites man" = "man bites dog"). Adding **n-grams** (bigrams like "not good") recovers some local order.

## The problem with raw counts

Common words ("the", "is", "and") dominate counts but carry little meaning. Words appearing in every document do not help distinguish documents. We want to weight terms by **how characteristic** they are.

## TF-IDF

**Term frequency** $\text{tf}(t, d)$ measures importance within a document — the raw count, or a dampened version such as $1 + \log(\text{count})$ so that the 50th occurrence matters less than the first.

**Inverse document frequency** measures rarity across the corpus of $N$ documents, where $\text{df}(t)$ is the number of documents containing $t$:

$$
\text{idf}(t) = \log\frac{N}{\text{df}(t)}
$$

(Scikit-learn uses a smoothed version $\log\frac{1 + N}{1 + \text{df}(t)} + 1$.) A term in every document gets IDF near its minimum; a term in one document gets a high IDF.

$$
\text{tfidf}(t, d) = \text{tf}(t, d)\times\text{idf}(t)
$$

Finally, **L2-normalise** each document vector so that long documents do not dominate.

:::note
IDF has an information-theoretic flavour: $\log\frac{N}{\text{df}}$ is the surprisal of observing the term in a randomly chosen document. Rare terms are more informative — the same intuition behind Shannon's information content.
:::

## Document similarity and retrieval

With L2-normalised TF-IDF vectors, **cosine similarity** is just the dot product. To search, embed the query as a TF-IDF vector and rank documents by cosine similarity.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

docs = ["How to register the birth of a newborn child",
        "Vaccination schedule for babies and children",
        "Renewing an expired passport or travel document",
        "Eligibility for monthly cash assistance",
        "Birth certificate replacement if lost"]
vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
D = vec.fit_transform(docs)
q = vec.transform(["lost birth certificate"])
for i in cosine_similarity(q, D).ravel().argsort()[::-1][:3]:
    print(round(cosine_similarity(q, D[i])[0, 0], 3), docs[i])
```

Note the limitation: a query "newborn documents" would not match "birth certificate" at all — TF-IDF only matches **exact terms**. This vocabulary-mismatch problem motivates embeddings and semantic search.

## BM25: TF-IDF refined for search

**Okapi BM25**, the long-time standard ranking function in search engines, refines TF-IDF with term-frequency **saturation** and **document-length normalisation**:

$$
\text{BM25}(q, d) = \sum_{t \in q}\text{idf}(t)\cdot\frac{f(t, d)\,(k_1 + 1)}{f(t, d) + k_1\left(1 - b + b\frac{|d|}{\text{avgdl}}\right)}
$$

with typical $k_1 \in [1.2, 2]$ and $b = 0.75$. BM25 remains a strong baseline and a component of modern **hybrid search** (lexical + semantic) used in retrieval-augmented generation.

## Text classification with TF-IDF

TF-IDF + a linear model (logistic regression or linear SVM) is fast, strong and interpretable:

```python
from sklearn.datasets import fetch_20newsgroups
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
import numpy as np

train = fetch_20newsgroups(subset="train", categories=["sci.med", "sci.space", "rec.autos"],
                           remove=("headers", "footers", "quotes"))
test = fetch_20newsgroups(subset="test", categories=train.target_names,
                          remove=("headers", "footers", "quotes"))
clf = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                    LogisticRegression(C=10, max_iter=2000))
clf.fit(train.data, train.target)
print("accuracy:", round(clf.score(test.data, test.target), 3))

vocab = clf[0].get_feature_names_out()
for i, c in enumerate(train.target_names):                 # most indicative n-grams per class
    print(c, "->", ", ".join(vocab[np.argsort(clf[1].coef_[i])[-8:]]))
```

## Strengths and limitations

**Strengths:** fast, cheap, no GPU; interpretable weights; works well with limited data; strong for topic-like tasks.

**Limitations:** ignores word order (beyond n-grams) and meaning; no notion that "car" and "automobile" are similar; huge sparse vectors; poor at tasks needing deeper understanding (sarcasm, reasoning, negation scope).

:::tip
Always build a TF-IDF + logistic regression baseline before training a transformer. If it achieves 92% and your fine-tuned transformer 93%, the cheaper model may be the right production choice.
:::

:::exercise
1. Compute TF-IDF by hand for a three-document corpus and verify with scikit-learn (set `smooth_idf=False, norm=None`).
2. Implement BM25 ranking for the small document collection above and compare rankings with TF-IDF cosine.
3. Compare unigram vs unigram+bigram TF-IDF features for sentiment classification. Which bigrams receive the largest weights?
:::

:::takeaway
- Bag of words counts terms; order is lost except through n-grams.
- TF-IDF weights terms by within-document frequency and corpus rarity; normalise and compare with cosine similarity.
- BM25 adds saturation and length normalisation and remains a strong search baseline.
- TF-IDF + linear models are fast, interpretable baselines; they cannot capture synonymy or deeper meaning.
:::

=== POST ===
slug: n-gram-language-models
title: "N-gram Language Models, Smoothing and Perplexity"
category: nlp
level: Intermediate
tags: language models, n-grams, smoothing, perplexity, markov assumption
summary: A language model assigns probabilities to sequences of words. We derive n-gram models from the chain rule and Markov assumption, fix zero probabilities with smoothing, generate text, and evaluate with perplexity.
---
A **language model (LM)** assigns a probability to any sequence of words. That simple capability powers spelling correction, speech recognition, machine translation, autocomplete — and, scaled up enormously, today's large language models. Before neural networks, the dominant language models were **n-gram models**. They introduce every core concept we need: the chain rule, the Markov assumption, sparsity, smoothing and perplexity.

## The chain rule

The probability of a sentence $w_1, \dots, w_n$ decomposes exactly:

$$
P(w_1, \dots, w_n) = \prod_{i=1}^{n}P(w_i \mid w_1, \dots, w_{i-1})
$$

Every language model — from bigrams to GPT — estimates these conditional next-word probabilities.

## The Markov assumption

Full histories are too varied to estimate from data. An **n-gram model** assumes each word depends only on the previous $n - 1$ words:

$$
P(w_i \mid w_1, \dots, w_{i-1}) \approx P(w_i \mid w_{i-n+1}, \dots, w_{i-1})
$$

Bigram ($n = 2$): $P(w_i \mid w_{i-1})$. Trigram: $P(w_i \mid w_{i-2}, w_{i-1})$.

## Maximum likelihood estimation

Estimate probabilities from counts in a corpus:

$$
P_{\text{MLE}}(w_i \mid w_{i-1}) = \frac{C(w_{i-1}, w_i)}{C(w_{i-1})}
$$

Special tokens `<s>` and `</s>` mark sentence boundaries so we can model how sentences begin and end.

## The sparsity problem

Language is productive: most plausible n-grams never appear in any finite corpus. If the test sentence contains a bigram with zero count, MLE assigns the whole sentence probability **zero**. The larger $n$, the worse the sparsity — a trigram model on a vocabulary of 50,000 words has $1.25 \times 10^{14}$ possible trigrams.

## Smoothing

**Add-one (Laplace) smoothing** pretends every n-gram occurred once more:

$$
P_{\text{Laplace}}(w_i \mid w_{i-1}) = \frac{C(w_{i-1}, w_i) + 1}{C(w_{i-1}) + |V|}
$$

Simple, but it moves far too much probability mass to unseen events for large vocabularies. **Add-k** uses $k < 1$.

**Backoff and interpolation** combine models of different orders:

$$
P_{\text{interp}}(w_i \mid w_{i-2}, w_{i-1}) = \lambda_3 P(w_i \mid w_{i-2}, w_{i-1}) + \lambda_2 P(w_i \mid w_{i-1}) + \lambda_1 P(w_i)
$$

with $\sum\lambda = 1$, tuned on held-out data.

**Kneser–Ney smoothing**, the best classical method, subtracts a fixed discount from observed counts and redistributes it using a clever **continuation probability**: how many **different** contexts a word appears in. "Francisco" is frequent but almost always follows "San", so it should get little probability after an unfamiliar context; "glasses" follows many different words.

## A bigram model from scratch

```python
import math, random
from collections import Counter, defaultdict

corpus = """the model learns from data . the model predicts the next word .
students learn from teachers . teachers learn from students .
the data helps the model learn .""".split(" . ")
sents = [["<s>"] + s.replace(".", "").split() + ["</s>"] for s in corpus]

unigrams, bigrams = Counter(), Counter()
for s in sents:
    unigrams.update(s[:-1])
    bigrams.update(zip(s[:-1], s[1:]))
V = len(set(w for s in sents for w in s))

def p(w, prev, k=0.1):                       # add-k smoothing
    return (bigrams[(prev, w)] + k) / (unigrams[prev] + k * V)

def perplexity(sentence):
    toks = ["<s>"] + sentence.split() + ["</s>"]
    logp = sum(math.log(p(w, prev)) for prev, w in zip(toks[:-1], toks[1:]))
    return math.exp(-logp / (len(toks) - 1))

print(perplexity("the model learns from students"))
print(perplexity("students predicts data the"))      # less fluent -> higher perplexity

def generate(max_len=12):
    w, out = "<s>", []
    nxt = defaultdict(list)
    for (a, b), c in bigrams.items():
        nxt[a] += [b] * c
    for _ in range(max_len):
        w = random.choice(nxt[w])
        if w == "</s>":
            break
        out.append(w)
    return " ".join(out)
print(generate())
```

## Evaluating language models: perplexity

**Perplexity** is the exponentiated average negative log-likelihood per word on held-out text:

$$
\text{PPL}(W) = P(w_1, \dots, w_N)^{-1/N} = \exp\left(-\frac{1}{N}\sum_{i=1}^{N}\log P(w_i \mid w_{<i})\right)
$$

Lower is better. Intuitively, perplexity is the **effective branching factor**: a perplexity of 100 means the model is as uncertain as choosing uniformly among 100 words at each step. On the Wall Street Journal, classic results showed perplexity falling from roughly 960 (unigram) to 170 (bigram) to 110 (trigram). Perplexities are only comparable across models with the **same vocabulary and tokenisation**.

## Limitations of n-grams

- **No generalisation across similar words**: seeing "the cat sat" says nothing about "the dog sat".
- **Short context**: dependencies beyond $n - 1$ words are invisible ("The students who attended the lecture on Tuesday **were** tired").
- **Storage**: large n-gram tables require gigabytes.

Neural language models (Bengio et al., 2003) addressed the first problem with **word embeddings**; RNNs and then transformers addressed the second. But n-gram models remain useful: in speech recognition decoders, for fast spelling correction, for data deduplication and for detecting benchmark contamination.

:::exercise
1. Derive the MLE bigram estimate by maximising the likelihood of a corpus subject to probabilities summing to one.
2. Extend the code to trigrams with linear interpolation, tuning $\lambda$ on held-out sentences.
3. Train bigram models on two different text genres and compare their perplexities on each genre's test set.
:::

:::takeaway
- Language models use the chain rule; n-gram models add the Markov assumption.
- MLE from counts suffers from zero probabilities; smoothing (add-k, interpolation, Kneser–Ney) fixes it.
- Perplexity measures predictive quality — the effective branching factor.
- n-grams cannot generalise across similar words or long contexts — motivating neural LMs.
:::

=== POST ===
slug: word2vec-word-embeddings
title: "Word2Vec: Learning Word Embeddings from Context"
category: nlp
level: Intermediate
tags: word2vec, embeddings, skip-gram, negative sampling, distributional semantics
summary: Word2Vec learns dense word vectors by predicting context words. We derive the skip-gram and CBOW objectives, negative sampling, and explore analogies, similarity and the limitations of static embeddings.
---
In 2013 Tomas Mikolov and colleagues at Google released **word2vec**, a simple and fast method for learning word vectors from raw text. Its vectors captured semantic and syntactic relationships so strikingly — the famous "king − man + woman ≈ queen" — that word embeddings became the standard input representation for neural NLP for the next five years. The ideas behind it, especially contrastive learning with negative sampling, remain central today.

## The distributional hypothesis

"You shall know a word by the company it keeps" (J.R. Firth, 1957). Words appearing in similar contexts tend to have similar meanings: "doctor" and "nurse" both appear near "hospital", "patient", "treatment". Word2vec turns this idea into a prediction task.

## Two architectures

For a corpus of words, with a context window of size $c$ (e.g. 5 words either side):

- **Skip-gram**: given a centre word, predict each surrounding context word.
- **CBOW (Continuous Bag of Words)**: given the (averaged) context words, predict the centre word.

Skip-gram works better for rare words and smaller corpora; CBOW trains faster.

## The skip-gram objective

Each word $w$ has two vectors: $\mathbf{v}_w$ (as centre word) and $\mathbf{u}_w$ (as context word). The probability of context word $o$ given centre word $c$ is a softmax:

$$
P(o \mid c) = \frac{\exp(\mathbf{u}_o^\top\mathbf{v}_c)}{\sum_{w \in V}\exp(\mathbf{u}_w^\top\mathbf{v}_c)}
$$

and we maximise the average log-probability of observed (centre, context) pairs. The problem: the denominator sums over the whole vocabulary (hundreds of thousands of words) for every training pair — far too expensive.

## Negative sampling

Instead of a full softmax, train a **binary classifier**: is this (centre, context) pair real, or a random fake? For a real pair $(c, o)$ and $k$ **negative** words $n_1, \dots, n_k$ sampled from a noise distribution:

$$
\mathcal{L} = -\log\sigma(\mathbf{u}_o^\top\mathbf{v}_c) - \sum_{i=1}^{k}\log\sigma(-\mathbf{u}_{n_i}^\top\mathbf{v}_c)
$$

Only $k + 1$ dot products per pair ($k$ = 5–20 for small data, 2–5 for large data). Negatives are drawn from the unigram distribution raised to the $3/4$ power, $P_n(w) \propto f(w)^{3/4}$, which boosts rare words relative to raw frequency. This is an early instance of the **contrastive learning** idea used by CLIP and SimCLR.

**Subsampling** of very frequent words (discarding "the" with high probability) speeds training and improves rare-word vectors.

## Implementation sketch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SkipGramNS(nn.Module):
    def __init__(self, vocab_size, dim=100):
        super().__init__()
        self.center = nn.Embedding(vocab_size, dim)
        self.context = nn.Embedding(vocab_size, dim)
        nn.init.uniform_(self.center.weight, -0.5 / dim, 0.5 / dim)
        nn.init.zeros_(self.context.weight)
    def forward(self, c, o, neg):               # c, o: (B,), neg: (B, k)
        vc = self.center(c)                                    # (B, d)
        pos = F.logsigmoid((self.context(o) * vc).sum(-1))     # (B,)
        negs = F.logsigmoid(-(self.context(neg) @ vc.unsqueeze(-1)).squeeze(-1)).sum(-1)
        return -(pos + negs).mean()

model = SkipGramNS(vocab_size=10_000)
c, o = torch.randint(0, 10_000, (32,)), torch.randint(0, 10_000, (32,))
neg = torch.randint(0, 10_000, (32, 5))
print(model(c, o, neg))
```

In practice, use the optimised **gensim** implementation:

```python
from gensim.models import Word2Vec
sentences = [line.lower().split() for line in open("corpus.txt", encoding="utf-8")]
w2v = Word2Vec(sentences, vector_size=100, window=5, min_count=5, sg=1, negative=10, epochs=5)
print(w2v.wv.most_similar("doctor", topn=5))
print(w2v.wv.most_similar(positive=["king", "woman"], negative=["man"], topn=3))
```

## What the vectors capture

- **Similarity**: nearest neighbours are semantically related words.
- **Analogies**: vector offsets encode relations — gender (man → woman), country–capital (France → Paris), verb tense (walk → walked), comparatives (good → better). Solve "a is to b as c is to ?" by finding the word nearest $\mathbf{v}_b - \mathbf{v}_a + \mathbf{v}_c$.
- **Clusters** of topics, and some interpretable directions.

:::note
Levy and Goldberg (2014) showed that skip-gram with negative sampling implicitly factorises a matrix of **pointwise mutual information** (PMI) between words and contexts, shifted by $\log k$. So word2vec connects to classical count-based distributional semantics (and to GloVe, next lecture).
:::

## Limitations

- **One vector per word type**: "bank" gets a single vector blending riverbank and financial meanings. **Contextual embeddings** (ELMo, BERT) solve this.
- **Out-of-vocabulary words** get no vector (FastText addresses this with subwords).
- **Bias**: embeddings reproduce stereotypes present in the training text; analogy tests have revealed associations such as occupations with genders.
- **Analogy results are overstated**: many famous analogies only work when the input words are excluded from the candidates, and performance varies widely across relation types.

:::exercise
1. Train word2vec on a corpus of news or Wikipedia text in your language and inspect nearest neighbours of ten words.
2. Derive the gradient of the negative-sampling loss with respect to $\mathbf{v}_c$.
3. Evaluate your embeddings on an analogy set, and test whether excluding the query words changes accuracy.
:::

:::takeaway
- Word2vec learns word vectors by predicting context (skip-gram) or centre words (CBOW).
- Negative sampling replaces the expensive softmax with a contrastive binary task.
- Vectors capture similarity and relational offsets; they implicitly factorise shifted PMI.
- Static embeddings cannot handle polysemy or unknown words and inherit corpus biases.
:::

=== POST ===
slug: glove-fasttext-embeddings
title: "GloVe and FastText: Global Statistics and Subword Embeddings"
category: nlp
level: Intermediate
tags: glove, fasttext, embeddings, subword, co-occurrence
summary: GloVe learns embeddings from a global co-occurrence matrix with a weighted least-squares objective; FastText represents words as bags of character n-grams, handling rare and unseen words. We compare them with word2vec and discuss evaluation.
---
Word2vec learns from local context windows one pair at a time. Two influential follow-ups improved on it in complementary ways. **GloVe** (Stanford, 2014) exploits **global** co-occurrence statistics directly. **FastText** (Facebook AI Research, 2016–2017) builds word vectors from **character n-grams**, so it can represent rare, misspelled and never-seen words — invaluable for morphologically rich languages.

## GloVe: Global Vectors

Pennington, Socher and Manning started from an observation about **ratios** of co-occurrence probabilities. Let $P_{ij} = P(j \mid i)$ be the probability that word $j$ appears in the context of word $i$. Consider $i$ = "ice", $j$ = "steam":

| Probe word $k$ | $P(k \mid \text{ice})/P(k \mid \text{steam})$ |
|---|---|
| solid | large (≫ 1) |
| gas | small (≪ 1) |
| water | ≈ 1 (related to both) |
| fashion | ≈ 1 (related to neither) |

Ratios distinguish relevant from irrelevant words and encode meaning. GloVe designs vectors so that dot products reflect **log co-occurrence counts**:

$$
\mathbf{w}_i^\top\tilde{\mathbf{w}}_j + b_i + \tilde{b}_j \approx \log X_{ij}
$$

where $X_{ij}$ counts how often $j$ occurs in $i$'s context. This is solved as a **weighted least-squares** problem over non-zero entries:

$$
J = \sum_{i,j:\,X_{ij} > 0}f(X_{ij})\left(\mathbf{w}_i^\top\tilde{\mathbf{w}}_j + b_i + \tilde{b}_j - \log X_{ij}\right)^2
$$

The weighting function

$$
f(x) = \begin{cases} (x/x_{\max})^{\alpha} & x < x_{\max} \\ 1 & \text{otherwise} \end{cases}, \qquad \alpha = 3/4,\; x_{\max} = 100
$$

prevents very frequent pairs from dominating and down-weights rare, noisy pairs. The final embedding is typically $\mathbf{w}_i + \tilde{\mathbf{w}}_i$.

GloVe combines the strengths of **count-based** methods (using global statistics efficiently, like LSA/SVD) and **prediction-based** methods (good linear structure, like word2vec). Pretrained GloVe vectors (trained on Wikipedia, web crawl and Twitter data) were widely used.

## FastText: subword information

Word2vec and GloVe treat "teach", "teacher", "teaching" and "teachers" as unrelated symbols, and have **no vector** for a word absent from training. Bojanowski et al. (2017) represented each word as a **bag of character n-grams** (typically lengths 3–6), with boundary markers. For "where" with $n = 3$:

```text
<wh, whe, her, ere, re>   plus the whole word <where>
```

A word's vector is the **sum** of its n-gram vectors:

$$
\mathbf{v}_w = \sum_{g \in \mathcal{G}_w}\mathbf{z}_g
$$

and training uses the skip-gram negative-sampling objective. Consequences:

- **Unseen words** get vectors from their n-grams ("teachable" shares n-grams with "teach" and "table").
- **Morphology** is captured: words sharing roots and affixes are close.
- **Misspellings** and informal spellings remain near their correct forms.
- Especially beneficial for **morphologically rich languages** — FastText released pretrained vectors for 157 languages, including Bangla.

N-gram vectors are stored in a fixed number of hash buckets (e.g. 2 million) to bound memory.

```python
from gensim.models import FastText

sentences = [s.lower().split() for s in open("corpus.txt", encoding="utf-8")]
ft = FastText(sentences, vector_size=100, window=5, min_count=3, min_n=3, max_n=6, epochs=5)
print(ft.wv.most_similar("teacher", topn=5))
print(ft.wv["teachingly"][:5])                   # works even if never seen in training
print(ft.wv.similarity("organisation", "organization"))   # spelling variants stay close
```

FastText also offers a very fast **text classifier**: average word and n-gram embeddings, then a linear softmax — trains on millions of documents in minutes on a CPU and is a strong baseline for tasks such as language identification.

## Comparing static embeddings

| | Word2vec | GloVe | FastText |
|---|---|---|---|
| Training signal | Local windows (prediction) | Global co-occurrence matrix (regression) | Local windows over subwords |
| OOV words | No vector | No vector | Yes, from n-grams |
| Morphology | Not captured | Not captured | Captured |
| Rare words | Weak | Weak | Better |
| Memory | Vocabulary × dim | Vocabulary × dim | + n-gram buckets |

## Evaluating embeddings

- **Intrinsic**: word similarity datasets (correlate cosine similarity with human ratings, e.g. SimLex-999, WordSim-353) and analogy tasks.
- **Extrinsic**: performance of downstream tasks (NER, classification) using the embeddings as features — what ultimately matters.

Intrinsic and extrinsic results do not always agree.

:::note
Static embeddings have largely been superseded by contextual models (BERT and beyond) for accuracy. But they remain valuable: tiny, fast, CPU-friendly and easy to deploy; useful for low-resource settings, lexicon expansion, quick similarity search and studying linguistic change and bias over time.
:::

:::exercise
1. List all character 3- to 5-grams of "<learning>" as FastText would, and explain how an unseen word "relearning" obtains a vector.
2. Compare nearest neighbours of a rare word in word2vec and FastText models trained on the same corpus.
3. Evaluate pretrained GloVe and FastText vectors on a word-similarity dataset using Spearman correlation.
:::

:::takeaway
- GloVe fits dot products to log co-occurrence counts with a weighted least-squares objective.
- FastText represents words as sums of character n-gram vectors, handling unseen words and morphology.
- FastText is especially useful for morphologically rich and low-resource languages.
- Evaluate embeddings extrinsically; static embeddings remain useful for lightweight applications.
:::

=== POST ===
slug: subword-tokenization-bpe-wordpiece
title: "Subword Tokenisation: BPE, WordPiece, Unigram and SentencePiece"
category: nlp
level: Intermediate
tags: tokenization, bpe, wordpiece, sentencepiece, subword
summary: Modern language models split text into subword units. We derive byte-pair encoding step by step, compare WordPiece and Unigram LM tokenisation, discuss byte-level BPE, and examine how tokenisation affects multilingual fairness and cost.
---
Word-level vocabularies face a dilemma: a vocabulary large enough to cover rare words becomes enormous, while any fixed vocabulary still meets unknown words. Character-level models avoid unknown words but produce very long sequences. **Subword tokenisation** is the practical middle ground: frequent words become single tokens, while rare words are split into meaningful pieces ("unbelievable" → "un", "believ", "able"). Every modern language model uses it, and the tokeniser quietly influences cost, context length and even fairness across languages.

## Byte-Pair Encoding (BPE)

Originally a data-compression algorithm, BPE was adapted for NLP by Sennrich, Haddow and Birch (2016). **Training**:

1. Start with a vocabulary of individual characters; represent every word in the corpus as a sequence of characters (with an end-of-word marker).
2. Count all adjacent symbol **pairs**, weighted by word frequency.
3. **Merge** the most frequent pair into a new symbol; add it to the vocabulary.
4. Repeat until the vocabulary reaches the desired size (e.g. 32,000–100,000+).

**Tokenising** new text applies the learned merges in order.

```python
from collections import Counter

def learn_bpe(word_freqs, num_merges):
    vocab = {tuple(w) + ("</w>",): f for w, f in word_freqs.items()}
    merges = []
    for _ in range(num_merges):
        pairs = Counter()
        for sym, f in vocab.items():
            for a, b in zip(sym, sym[1:]):
                pairs[(a, b)] += f
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        merges.append(best)
        new_vocab = {}
        for sym, f in vocab.items():
            out, i = [], 0
            while i < len(sym):
                if i < len(sym) - 1 and (sym[i], sym[i + 1]) == best:
                    out.append(sym[i] + sym[i + 1]); i += 2
                else:
                    out.append(sym[i]); i += 1
            new_vocab[tuple(out)] = f
        vocab = new_vocab
    return merges, vocab

merges, vocab = learn_bpe({"low": 5, "lower": 2, "newest": 6, "widest": 3}, 10)
print(merges[:6])          # e.g. ('e','s'), ('es','t'), ('est','</w>'), ('l','o'), ...
print(list(vocab)[:4])
```

This classic example from the BPE paper learns merges such as "es" → "est" → "est</w>", so "newest" and "widest" share the suffix token "est</w>".

## Byte-level BPE

GPT-2 introduced **byte-level BPE**: start from the 256 possible **bytes** of UTF-8 text instead of characters. Every possible string can be encoded — no unknown tokens ever, including emojis and any script — while merges still create efficient tokens for common sequences. Most GPT-style models use byte-level BPE (e.g. OpenAI's tiktoken encodings).

## WordPiece

Used by BERT. Similar to BPE, but instead of merging the most **frequent** pair, it merges the pair that most increases the **likelihood** of the training data under a unigram model — approximately the pair maximising

$$
\text{score}(a, b) = \frac{\text{freq}(ab)}{\text{freq}(a)\times\text{freq}(b)}
$$

favouring pairs whose parts are rarely seen apart. Continuation pieces are marked with "##": "playing" → "play", "##ing".

## Unigram language model tokenisation

Kudo (2018) took the opposite approach: start with a **large** candidate vocabulary and iteratively **remove** tokens whose removal least reduces the corpus likelihood under a unigram model, $P(\mathbf{x}) = \prod_i p(x_i)$. A word can have several segmentations; the tokeniser chooses the most probable (Viterbi) or **samples** segmentations during training (**subword regularisation**), which improves robustness.

## SentencePiece

**SentencePiece** (Kudo & Richardson, 2018) is a toolkit implementing BPE and Unigram that treats the input as a **raw stream** including spaces (represented as "▁"). It needs no language-specific pre-tokenisation — essential for languages without spaces between words — and makes tokenisation fully reversible. T5, LLaMA-family and many multilingual models use SentencePiece.

```python
# pip install sentencepiece transformers
import sentencepiece as spm
spm.SentencePieceTrainer.train(input="corpus.txt", model_prefix="mytok",
                               vocab_size=8000, model_type="unigram", character_coverage=0.9995)
sp = spm.SentencePieceProcessor(model_file="mytok.model")
print(sp.encode("Machine learning is transforming education.", out_type=str))

from transformers import AutoTokenizer
for name in ["bert-base-uncased", "gpt2"]:
    tok = AutoTokenizer.from_pretrained(name)
    print(name, tok.tokenize("Tokenisation influences multilingual fairness!"))
```

## Why tokenisation matters

1. **Sequence length and cost**: models have fixed context windows, and APIs charge per token. Text that splits into more tokens costs more and fits less content.
2. **Multilingual fairness**: tokenisers trained mostly on English split other languages — especially non-Latin scripts — into many more tokens per word. Studies (e.g. Petrov et al., 2023) found the same content can require several times more tokens in some languages than in English, meaning higher cost, shorter effective context and often lower quality for those speakers.
3. **Arithmetic and spelling**: numbers split inconsistently ("12345" → "123", "45") make arithmetic harder; models see tokens, not letters, which explains difficulty with tasks like counting letters in a word.
4. **Glitch tokens**: rare tokens seen seldom in training can trigger strange behaviour.

| Algorithm | Direction | Criterion | Used by |
|---|---|---|---|
| BPE | Bottom-up merges | Pair frequency | GPT family (byte-level), many LLMs |
| WordPiece | Bottom-up merges | Likelihood gain | BERT |
| Unigram LM | Top-down pruning | Likelihood loss | T5, ALBERT, via SentencePiece |

:::tip
When building a model for a specific language or domain (e.g. Bangla health texts), check the **fertility** (average tokens per word) of candidate tokenisers on your data. Training or extending a tokeniser with in-domain text can substantially improve efficiency and quality.
:::

:::exercise
1. Run BPE by hand for three merges on {"hug": 10, "pug": 5, "pun": 12, "bun": 4, "hugs": 5}.
2. Measure tokens per word for the same paragraph in English and Bangla (or another language) using two pretrained tokenisers.
3. Train SentencePiece BPE and Unigram models with vocabulary 8,000 on the same corpus and compare segmentations of rare words.
:::

:::takeaway
- Subword tokenisation balances vocabulary size against sequence length and eliminates unknown words.
- BPE merges frequent pairs; WordPiece merges by likelihood gain; Unigram prunes a large vocabulary.
- Byte-level BPE covers every string; SentencePiece handles raw text without pre-tokenisation.
- Tokenisation affects cost, context, arithmetic and fairness across languages.
:::

=== POST ===
slug: text-classification
title: "Text Classification: From Linear Models to Fine-Tuned Transformers"
category: nlp
level: Beginner
tags: text classification, fine-tuning, bert, pipelines, evaluation
summary: Text classification is the most widely deployed NLP task. We compare TF-IDF baselines, CNN and RNN classifiers, and fine-tuned transformers, and cover label design, imbalance, multilingual data and evaluation.
---
Routing support tickets, flagging urgent messages, detecting spam and hate speech, categorising news, identifying the topic of citizen feedback, triaging requests for assistance — **text classification** is the most common NLP task in production. Today we build classifiers of increasing sophistication and learn to choose among them.

## Task variants

- **Binary**: spam vs not spam.
- **Multiclass**: one of $K$ categories (topic).
- **Multi-label**: several labels at once (a message can be both "health" and "urgent").
- **Hierarchical**: categories with sub-categories.

## Step 1: design the labels

Clear, mutually exclusive (or explicitly multi-label) categories with written **annotation guidelines** and examples. Measure **inter-annotator agreement** (Cohen's or Fleiss' kappa) on a sample; low agreement means the task is ill-defined, and no model can do better than the labels.

## Step 2: strong classical baseline

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

baseline = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
    LogisticRegression(max_iter=2000, class_weight="balanced"))
# scores = cross_val_score(baseline, texts, labels, cv=5, scoring="f1_macro")
```

Character n-grams (`analyzer="char_wb", ngram_range=(2, 5)`) help with misspellings, informal text and morphologically rich languages.

## Step 3: neural classifiers (historical context)

- **CNN for text** (Kim, 2014): convolutions of widths 3–5 over word embeddings, max-pooled over time — detects key phrases regardless of position.
- **BiLSTM with attention**: reads the sequence in both directions and attends to the most informative words.

These improved on linear models but have been largely superseded by pretrained transformers.

## Step 4: fine-tune a pretrained transformer

Add a classification head on the [CLS] (or pooled) representation of a pretrained encoder and fine-tune the whole model:

```python
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding)
import numpy as np, evaluate

ds = load_dataset("ag_news")                                   # 4 news topics
ckpt = "distilbert-base-uncased"
tok = AutoTokenizer.from_pretrained(ckpt)
ds = ds.map(lambda b: tok(b["text"], truncation=True, max_length=128), batched=True)
model = AutoModelForSequenceClassification.from_pretrained(ckpt, num_labels=4)

f1 = evaluate.load("f1")
def metrics(p):
    return f1.compute(predictions=np.argmax(p.predictions, -1), references=p.label_ids, average="macro")

args = TrainingArguments("out", learning_rate=2e-5, per_device_train_batch_size=32, num_train_epochs=2,
                         weight_decay=0.01, eval_strategy="epoch", warmup_ratio=0.06, fp16=True)
trainer = Trainer(model=model, args=args, train_dataset=ds["train"].shuffle(seed=0).select(range(20000)),
                  eval_dataset=ds["test"], data_collator=DataCollatorWithPadding(tok), compute_metrics=metrics)
trainer.train()
```

Typical hyperparameters: learning rate $1\times10^{-5}$ to $5\times10^{-5}$, 2–4 epochs, warm-up, weight decay 0.01. Fine-tuning usually gives the best accuracy when a few thousand labelled examples are available.

## Other options

- **Sentence embeddings + logistic regression**: embed texts with a sentence-transformer model and train a linear classifier — fast, strong with small data, easy to update.
- **Few-shot / zero-shot with LLMs**: describe categories in a prompt and let a large language model classify. Excellent for bootstrapping and rare categories, but more expensive per prediction and must be evaluated carefully for consistency.
- **SetFit**: contrastive fine-tuning of sentence transformers with very few labels (e.g. 8 per class).

## Choosing an approach

| Labelled data | Latency/cost constraints | Recommended |
|---|---|---|
| None | Moderate | Zero-shot LLM, then collect labels |
| < 100 per class | Any | Sentence embeddings + linear model, SetFit, or few-shot LLM |
| Thousands | Low latency | Fine-tuned small transformer (DistilBERT, MiniLM) |
| Thousands | Very low resources | TF-IDF + linear model |
| Many languages | — | Multilingual encoder (e.g. XLM-R) or multilingual embeddings |

## Practical concerns

- **Imbalance**: class weights, threshold tuning, macro-F1 evaluation.
- **Long documents**: truncate wisely (beginning + end), chunk and aggregate, or use long-context models.
- **Multilingual and code-mixed text**: use multilingual models and evaluate per language.
- **Label drift**: new topics appear (a new disease outbreak, a new policy); monitor and retrain.
- **Error analysis**: read misclassified examples; many will be ambiguous or mislabelled.
- **Harm**: classifiers used for moderation or prioritisation can encode bias against dialects or groups — audit per subgroup.

:::exercise
1. Compare TF-IDF + logistic regression, sentence embeddings + logistic regression, and fine-tuned DistilBERT on the same dataset with 500, 2,000 and 10,000 training examples.
2. Measure inter-annotator agreement on 100 examples you and a classmate label independently.
3. Build a multi-label classifier with sigmoid outputs and tune per-label thresholds on validation data.
:::

:::takeaway
- Define labels carefully and measure annotator agreement.
- Start with TF-IDF + logistic regression; try sentence embeddings for small data.
- Fine-tuned transformers are usually most accurate with thousands of examples; LLMs help with zero/few-shot.
- Handle imbalance, long texts, languages, drift and fairness explicitly.
:::
