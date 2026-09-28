=== POST ===
slug: gpt-family-autoregressive-models
title: "The GPT Family: Autoregressive Language Models from GPT-1 to Today"
category: nlp
level: Intermediate
tags: gpt, autoregressive, language models, decoder-only, in-context learning
summary: GPT models are decoder-only transformers trained to predict the next token. We trace GPT-1 through GPT-3's few-shot learning to instruction-tuned assistants, and explain why next-token prediction at scale produces broad capabilities.
---
While BERT showed the power of bidirectional encoders for understanding, OpenAI pursued a different path: **generative pretraining** with a **decoder-only** transformer trained simply to predict the next token. Each generation of the **GPT** (Generative Pre-trained Transformer) family scaled this recipe up, and the results reshaped the field — culminating in conversational assistants used by hundreds of millions of people.

## The objective: next-token prediction

Given text $x_1, \dots, x_T$, a GPT model maximises

$$
\mathcal{L}(\theta) = \sum_{t=1}^{T}\log P_\theta(x_t \mid x_1, \dots, x_{t-1})
$$

using a stack of transformer blocks with **causal (masked) self-attention**, so each position attends only to earlier positions. Every position in every training sequence provides a training signal, and no labels are required — any text will do.

## GPT-1 (2018): pretrain, then fine-tune

Radford et al. pretrained a 12-layer, ~117M-parameter decoder on BooksCorpus, then fine-tuned it on downstream tasks with task-specific input formatting (e.g. concatenating premise and hypothesis with delimiters). It improved the state of the art on many benchmarks, establishing generative pretraining as effective for transfer — just months before BERT.

## GPT-2 (2019): zero-shot task transfer

GPT-2 scaled to **1.5 billion** parameters trained on WebText (about 40 GB of text from outbound links on Reddit with some engagement). The paper's thesis: a sufficiently large language model trained on diverse text learns to perform tasks **without explicit supervision**, because the tasks appear naturally in text. Writing "TL;DR:" after an article elicits a summary; "English: … French:" elicits translation. GPT-2's fluent generations led OpenAI to stage its release, citing misuse concerns — an early public debate about responsible release of language models.

## GPT-3 (2020): in-context learning

GPT-3 scaled to **175 billion** parameters trained on hundreds of billions of tokens. Its headline finding was **in-context learning**: without any gradient updates, the model performs a new task from a description and a few examples placed in the prompt.

```text
Translate English to French:
sea otter => loutre de mer
cheese => fromage
peppermint =>
```

Performance improved dramatically with scale and with the number of examples (**zero-shot < one-shot < few-shot**). Some abilities appeared only in the largest models. GPT-3 also demonstrated limitations: factual errors, inconsistent reasoning, sensitivity to prompt wording, and biases absorbed from web data.

## From language model to assistant

A raw pretrained model continues text; it does not reliably follow instructions or behave helpfully and safely. **InstructGPT** (Ouyang et al., 2022) added:

1. **Supervised fine-tuning (SFT)** on demonstrations of good responses to instructions;
2. **Reinforcement learning from human feedback (RLHF)**: a reward model trained on human preference comparisons, then policy optimisation (PPO) against it.

Human evaluators preferred outputs of a 1.3B-parameter InstructGPT model over those of the 175B GPT-3 base model — alignment training mattered more than raw size for usefulness. ChatGPT (late 2022) applied this recipe to dialogue and brought LLMs to the mainstream. Later GPT-4-class models added multimodal inputs and much stronger reasoning, alongside many competing model families (Claude, Gemini, LLaMA, Mistral, Qwen, DeepSeek and others). (The Generative AI track covers alignment and modern LLMs in depth.)

## Why does next-token prediction yield broad capabilities?

To predict the next token well across the whole internet, a model benefits from modelling grammar, facts, the structure of arguments, the behaviour of code, the conventions of dialogue, and patterns of reasoning found in text. Prediction is a form of **compression**, and good compression requires modelling the regularities that generated the data. Scale provides the capacity; diverse data provides the curriculum.

:::warning
"Predicting text that looks like reasoning" and "reasoning correctly" are not the same thing. Language models can produce fluent, confident, wrong answers (**hallucinations**), reproduce biases, and be manipulated by adversarial prompts. Treat outputs as drafts to verify, especially for factual, medical, legal or safety-critical content.
:::

## Generating text

At each step the model outputs a distribution over the vocabulary; a **decoding strategy** picks the next token (greedy, beam, temperature sampling, top-k, nucleus). Covered in detail in the Generative AI track.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tok = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2").eval()
prompt = "Machine learning can help teachers by"
ids = tok(prompt, return_tensors="pt").input_ids
with torch.no_grad():
    out = model.generate(ids, max_new_tokens=40, do_sample=True, temperature=0.8, top_p=0.9,
                         pad_token_id=tok.eos_token_id)
print(tok.decode(out[0], skip_special_tokens=True))

# Next-token probabilities
with torch.no_grad():
    probs = model(ids).logits[0, -1].softmax(-1)
top = probs.topk(5)
print([(tok.decode(int(i)), round(float(p), 3)) for p, i in zip(top.values, top.indices)])
```

## Encoder vs decoder, revisited

| | BERT-style encoders | GPT-style decoders |
|---|---|---|
| Attention | Bidirectional | Causal |
| Objective | Masked token prediction | Next-token prediction |
| Strength | Understanding, embeddings, efficient fine-tuning | Generation, in-context learning, general assistants |
| Typical size in practice | 100M–1B | 1B to hundreds of billions |

Decoder-only models became the dominant architecture for general-purpose LLMs because generation subsumes many tasks and the objective scales cleanly.

:::exercise
1. Compute GPT-2's perplexity on a paragraph of your own writing and on a shuffled version of the same words.
2. Try zero-shot, one-shot and three-shot prompts for a simple classification task with a small open model. How does accuracy change?
3. Explain why a causal mask lets all positions of a sequence be trained in parallel even though generation is sequential.
:::

:::takeaway
- GPT models are decoder-only transformers trained on next-token prediction with causal attention.
- GPT-1 showed generative pretraining transfers; GPT-2 zero-shot task transfer; GPT-3 in-context learning at scale.
- Instruction tuning and RLHF turned language models into helpful assistants.
- Next-token prediction at scale yields broad capabilities — and fluent errors that must be verified.
:::

=== POST ===
slug: t5-encoder-decoder-models
title: "T5 and BART: Encoder–Decoder Pretraining and Text-to-Text Learning"
category: nlp
level: Intermediate
tags: t5, bart, encoder-decoder, text-to-text, span corruption
summary: T5 casts every NLP task as text in, text out; BART pretrains as a denoising autoencoder. We cover span corruption, the text-to-text framework, the lessons of T5's systematic study, and when encoder–decoders are the right choice.
---
Between encoder-only BERT and decoder-only GPT sits a third family: **encoder–decoder** transformers pretrained for sequence-to-sequence tasks. Two landmark models — Google's **T5** and Facebook's **BART** (both 2019) — showed how to pretrain them effectively. They remain strong choices for translation, summarisation and other tasks where an input is transformed into an output.

## T5: the text-to-text framework

Raffel et al.'s "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" framed **every task as text-to-text**: the input is text with a task prefix, the output is text.

```text
"translate English to German: That is good."        → "Das ist gut."
"summarize: <article>"                              → "<summary>"
"cola sentence: The course is jumping well."        → "unacceptable"
"stsb sentence1: ... sentence2: ..."                → "3.8"
"question: Who wrote Hamlet? context: ..."          → "Shakespeare"
```

Even classification and regression outputs are strings. One model, one loss (cross-entropy on output tokens), one decoding procedure — for everything. This uniformity made large-scale multi-task learning and systematic comparison straightforward.

## Span corruption

T5's pretraining objective replaces random **spans** of the input (about 15% of tokens, average span length 3) with **sentinel tokens**, and trains the decoder to output the missing spans:

```text
Input:  Thank you <X> me to your party <Y> week.
Target: <X> for inviting <Y> last <Z>
```

Targets are short (only the dropped spans), which makes pretraining efficient.

## The C4 dataset and the systematic study

T5 introduced **C4** (Colossal Clean Crawled Corpus), about 750 GB of English web text from Common Crawl, filtered heuristically (removing pages with offensive words, code, very short lines, duplicates). Later audits found that such filtering also removed a disproportionate amount of text from and about some minority groups and dialects — a reminder that data-cleaning choices embed values.

The paper systematically compared architectures, objectives, datasets, transfer strategies and scaling. Key findings:

- An **encoder–decoder** with a denoising objective performed best in their setting.
- **Span corruption** was a strong, efficient objective.
- **Scale** (more parameters, more data, longer training) consistently helped; the largest T5 had 11B parameters.
- Multi-task pretraining followed by fine-tuning worked well.

**mT5** extended T5 to 101 languages; **Flan-T5** (2022) fine-tuned T5 on more than 1,800 tasks phrased as instructions, markedly improving zero-shot instruction following — an early demonstration of **instruction tuning**.

## BART: a denoising autoencoder

Lewis et al.'s **BART** combined a bidirectional encoder (like BERT) with an autoregressive decoder (like GPT). Pretraining corrupts documents and trains the model to reconstruct the **original**. They compared corruptions:

- token masking, token deletion;
- **text infilling** (replace spans, including empty spans, with a single mask token — the model must infer how many tokens are missing);
- sentence permutation, document rotation.

Text infilling plus sentence shuffling worked best. BART excelled at **abstractive summarisation** (e.g. CNN/DailyMail, XSum) and generation tasks, while matching RoBERTa on understanding benchmarks. **mBART** extended denoising pretraining to many languages for translation.

## Using T5 and BART

```python
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

t5 = pipeline("text2text-generation", model="google/flan-t5-base")
print(t5("Translate English to French: The school opens next Monday.")[0]["generated_text"])
print(t5("Is this review positive or negative? The staff were patient and kind.")[0]["generated_text"])

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
article = ("Heavy monsoon rains caused flooding in several districts. Local authorities opened "
           "temporary shelters in schools and distributed clean water. Health workers warned of "
           "waterborne diseases and urged families to boil drinking water.")
print(summarizer(article, max_length=40, min_length=10)[0]["summary_text"])
```

Fine-tuning follows the familiar pattern with `AutoModelForSeq2SeqLM` and `Seq2SeqTrainer`: tokenise inputs and targets, train with teacher forcing, generate with beam search for evaluation.

## When to choose an encoder–decoder

| Situation | Good choice |
|---|---|
| Input and output are both substantial text (translation, summarisation, rewriting) | Encoder–decoder (T5, BART, mT5, NLLB) |
| Classification, extraction, embeddings | Encoder (BERT family) |
| Open-ended generation, chat, general assistant | Decoder-only LLM |
| Limited compute, specific seq2seq task, need fine-tuning | Small T5/BART fine-tuned — often very cost-effective |

The encoder processes the input bidirectionally once; the decoder attends to it via cross-attention. For tasks with long inputs and short outputs (summarisation, QA), this can be efficient.

:::note
Decoder-only LLMs now perform most seq2seq tasks via prompting, and dominate at large scale. But fine-tuned encoder–decoders of a few hundred million parameters remain competitive, cheap and fast for focused production tasks — and Flan-T5 remains a popular, permissively licensed baseline for research.
:::

:::exercise
1. Write the span-corruption input and target for a sentence of your choice with two masked spans.
2. Fine-tune Flan-T5-small to rewrite complex sentences into plain language on a small dataset and evaluate with ROUGE and human judgement.
3. Compare BART-large-CNN and an instruction-tuned decoder LLM on summarising five news articles. Note factual errors in each.
:::

:::takeaway
- T5 casts every task as text-to-text with task prefixes and pretrains with span corruption on C4.
- BART pretrains as a denoising autoencoder; text infilling works well, especially for summarisation.
- Flan-T5 showed the power of instruction tuning across many tasks.
- Encoder–decoders suit transformation tasks and are cost-effective when fine-tuned.
:::

=== POST ===
slug: fine-tuning-pretrained-language-models
title: "Fine-Tuning Pretrained Language Models: A Practical Guide"
category: nlp
level: Intermediate
tags: fine-tuning, transfer learning, hugging face, hyperparameters, catastrophic forgetting
summary: How to adapt a pretrained language model to your task reliably — choosing a model, preparing data, hyperparameters, handling small data and instability, continued pretraining, and evaluating properly.
---
Pretrained language models have made strong NLP accessible to anyone with a few thousand labelled examples and a GPU. But fine-tuning has pitfalls: unstable runs, overfitting on small data, forgetting, and evaluation mistakes. This lecture collects practical guidance from research and experience into a reliable workflow.

## Step 1: choose the right base model

Consider:

- **Architecture**: encoder (classification, extraction), encoder–decoder (seq2seq), decoder (generation).
- **Language coverage**: English-only vs multilingual (XLM-R, mT5) vs language-specific models (e.g. BanglaBERT).
- **Domain**: general vs biomedical, legal, scientific, social media.
- **Size vs latency**: DistilBERT/MiniLM for speed; larger models for accuracy.
- **Licence**: some weights restrict commercial use.
- **Tokeniser fertility** on your text.

## Step 2: prepare data

- Clean labels; remove duplicates across splits.
- Keep a **held-out test set** that you touch once.
- Stratify splits; for time-dependent data, split by time.
- Check lengths: truncate or chunk long texts; set `max_length` to cover most examples.

## Step 3: sensible hyperparameters

| Hyperparameter | Typical range |
|---|---|
| Learning rate | $1\times10^{-5}$ – $5\times10^{-5}$ (encoders); lower for large models |
| Batch size | 16–64 (use gradient accumulation if memory-limited) |
| Epochs | 2–5 (more for tiny datasets, with early stopping) |
| Warm-up | 5–10% of steps |
| Weight decay | 0.01 |
| Schedule | Linear or cosine decay |
| Max sequence length | Enough for ~95% of examples |

## Step 4: training loop with Hugging Face

```python
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments,
                          Trainer, DataCollatorWithPadding, EarlyStoppingCallback)
import numpy as np
from sklearn.metrics import f1_score

ckpt = "xlm-roberta-base"                                     # multilingual encoder
ds = load_dataset("csv", data_files={"train": "train.csv", "validation": "val.csv"})
labels = sorted(set(ds["train"]["label"]))
l2i = {l: i for i, l in enumerate(labels)}
tok = AutoTokenizer.from_pretrained(ckpt)

def prep(batch):
    enc = tok(batch["text"], truncation=True, max_length=256)
    enc["labels"] = [l2i[l] for l in batch["label"]]
    return enc
ds = ds.map(prep, batched=True, remove_columns=ds["train"].column_names)

model = AutoModelForSequenceClassification.from_pretrained(
    ckpt, num_labels=len(labels), id2label=dict(enumerate(labels)), label2id=l2i)

def compute_metrics(p):
    return {"macro_f1": f1_score(p.label_ids, p.predictions.argmax(-1), average="macro")}

args = TrainingArguments("ft-out", learning_rate=2e-5, per_device_train_batch_size=16,
                         gradient_accumulation_steps=2, num_train_epochs=5, warmup_ratio=0.06,
                         weight_decay=0.01, eval_strategy="epoch", save_strategy="epoch",
                         load_best_model_at_end=True, metric_for_best_model="macro_f1",
                         bf16=True, seed=42)
trainer = Trainer(model=model, args=args, train_dataset=ds["train"], eval_dataset=ds["validation"],
                  data_collator=DataCollatorWithPadding(tok), compute_metrics=compute_metrics,
                  callbacks=[EarlyStoppingCallback(early_stopping_patience=2)])
trainer.train()
trainer.save_model("final-model")
```

## Handling small datasets and instability

Fine-tuning large models on small datasets (a few hundred to a few thousand examples) can be **unstable**: different random seeds give very different results, and some runs fail entirely (Mosbach et al., 2021; Dodge et al., 2020). Remedies:

- **Run several seeds** and report mean ± standard deviation; pick the best seed on validation, never on test.
- Use **bias-corrected Adam** (standard now), warm-up and **more training steps** — many "failed" runs were simply under-trained.
- Lower the learning rate for larger models.
- **Re-initialise the top layer(s)** of the encoder, which are most specialised to the pretraining objective.
- Consider **SetFit**, sentence-embedding classifiers or **parameter-efficient fine-tuning** (LoRA, adapters), which are often more stable.
- Use **intermediate-task transfer**: first fine-tune on a related larger dataset (e.g. natural-language inference), then on your task.

## Domain- and task-adaptive pretraining

If your text differs from the pretraining data (clinical notes, humanitarian reports, informal social media), **continue pretraining** the model with its original objective (MLM) on unlabelled in-domain text before fine-tuning. Gururangan et al. (2020), "Don't Stop Pretraining", showed consistent gains from domain-adaptive (DAPT) and task-adaptive (TAPT) pretraining.

## Catastrophic forgetting

Fine-tuning can erase general capabilities. For single-task deployment this may not matter; for multi-task or continual settings, use lower learning rates, fewer epochs, parameter-efficient methods, or mix in general data.

## Evaluate properly

- Use task-appropriate metrics (macro-F1 for imbalance, entity-level F1 for NER).
- Report variance across seeds.
- Evaluate on **slices**: languages, dialects, text lengths, sources.
- Compare against simple baselines (TF-IDF + logistic regression) and against zero-shot LLM prompting.
- Check calibration if probabilities drive decisions.
- Error analysis: read the mistakes.

:::tip
Log the exact model checkpoint, tokeniser version, data version, hyperparameters and seed for every run (Weights & Biases, MLflow). Reproducibility problems in NLP fine-tuning are common and entirely preventable.
:::

:::exercise
1. Fine-tune the same model on the same data with five seeds and report the spread of validation scores.
2. Continue MLM pretraining for one epoch on unlabelled in-domain text, then fine-tune; compare with direct fine-tuning.
3. Compare full fine-tuning with LoRA on the same task in terms of accuracy, trainable parameters and training time.
:::

:::takeaway
- Choose base models by architecture, language, domain, size and licence.
- Use small learning rates, warm-up, weight decay and early stopping; run multiple seeds.
- Small-data instability is common; more steps, re-initialising top layers and PEFT help.
- Continued in-domain pretraining often boosts results; evaluate by slices against simple baselines.
:::

=== POST ===
slug: question-answering-systems
title: "Question Answering: Extractive, Open-Domain and Generative"
category: nlp
level: Intermediate
tags: question answering, squad, retrieval, reading comprehension, rag
summary: Question answering systems return answers, not documents. We cover extractive reading comprehension with span prediction, open-domain retriever–reader pipelines, generative QA, evaluation metrics and the problem of unanswerable questions.
---
Search engines return lists of documents; people usually want **answers**. "What documents do I need to register a birth?" "When does the vaccination clinic open?" **Question answering (QA)** systems read text and respond directly. QA is a central NLP task, a benchmark of machine reading comprehension, and — through retrieval-augmented generation — the basis of many modern assistant systems.

## Types of QA

| Type | Setting | Output |
|---|---|---|
| Extractive (reading comprehension) | Question + a given passage | A span copied from the passage |
| Open-domain | Question only; search a large corpus | Span or generated answer |
| Generative / abstractive | Question (+ context) | Free-form text |
| Knowledge-base QA | Question over a structured knowledge graph/database | Entity, value or query result |
| Multi-hop | Answer requires combining facts from several documents | Span or text |
| Conversational | Questions depend on earlier turns | Span or text |

## Extractive QA

**SQuAD** (Stanford Question Answering Dataset, 2016) contains 100,000+ questions on Wikipedia paragraphs, each answered by a span. BERT-style models solve it by encoding `[CLS] question [SEP] passage [SEP]` and predicting, for each passage token, the probability of being the answer **start** and **end**:

$$
P_{\text{start}}(i) = \text{softmax}_i(\mathbf{w}_s^\top\mathbf{h}_i), \qquad P_{\text{end}}(j) = \text{softmax}_j(\mathbf{w}_e^\top\mathbf{h}_j)
$$

The predicted span maximises $P_{\text{start}}(i)\,P_{\text{end}}(j)$ subject to $i \le j$ and a maximum length. Within a couple of years of SQuAD's release, models exceeded the reported human performance on it — though this reflected the dataset's limits as much as true comprehension.

### Unanswerable questions

**SQuAD 2.0** added questions whose answer is **not** in the passage. The model must abstain (predict the `[CLS]` position as a "no answer" span) when its best span score is below a threshold. Knowing when not to answer is essential for trustworthy systems.

```python
from transformers import pipeline
qa = pipeline("question-answering", model="deepset/roberta-base-squad2")
context = ("Birth registration is free of charge at the civil registry office. Parents should bring "
           "the hospital birth notification and their identity documents. The office is open "
           "Sunday to Thursday from 9 am to 4 pm.")
for q in ["What should parents bring?", "When is the office open?", "How much does a passport cost?"]:
    r = qa(question=q, context=context, handle_impossible_answer=True)
    print(f"{q} -> {r['answer']!r} ({r['score']:.2f})")
```

The last question should yield an empty answer — the passage does not contain it.

## Evaluation

- **Exact Match (EM)**: prediction equals a gold answer after normalisation (lowercase, remove punctuation and articles).
- **Token F1**: overlap between predicted and gold answer tokens — partial credit.
- For generative QA: human judgement, reference-based metrics, and increasingly LLM-based judges (with caution); **faithfulness** to sources is critical.

## Open-domain QA: retriever + reader

When no passage is given, the system must first **find** relevant text in a large corpus:

1. **Retriever**: select the top-$k$ passages. Classical sparse retrieval (BM25) or **dense retrieval** — encode questions and passages into vectors with dual encoders (e.g. DPR, Karpukhin et al. 2020) and retrieve by maximum inner product.
2. **Reader**: extract or generate the answer from retrieved passages.

Dense retrievers are trained contrastively: a question should be closer to its gold passage than to other passages in the batch (in-batch negatives) and to "hard negatives" retrieved by BM25. Combining sparse and dense retrieval (**hybrid search**) and re-ranking the top results with a **cross-encoder** (which reads question and passage together) further improves accuracy.

## Generative QA and RAG

Sequence-to-sequence models (T5, BART) and LLMs generate answers conditioned on retrieved passages. **Fusion-in-Decoder** encodes each passage separately and lets the decoder attend to all of them. **Retrieval-Augmented Generation (RAG)** with LLMs is now the dominant pattern for question answering over organisational documents (covered in depth in the Generative AI track).

:::warning
Generative QA can **hallucinate** plausible answers not supported by any source. For consequential domains (health, legal rights, eligibility for assistance), require citations to retrieved passages, prefer abstention over guessing, keep answers grounded in approved documents, and route uncertain questions to humans.
:::

## Challenges

- **Lexical gap** between question and answer text (addressed by dense retrieval).
- **Multi-hop reasoning** across documents.
- **Temporal questions** — answers change over time; the knowledge source must be current.
- **Ambiguous questions** — clarify rather than guess.
- **Multilingual QA** — questions and documents in different languages; multilingual encoders enable cross-lingual retrieval.
- **Dataset artefacts** — models can exploit superficial cues (answer type, lexical overlap) rather than truly reading; adversarial evaluation reveals this.

:::exercise
1. Build an extractive QA demo over a set of 20 FAQ passages using BM25 retrieval plus a SQuAD2 reader. Measure EM and F1 on 30 questions you write.
2. Replace BM25 with a dense sentence-embedding retriever. Which questions improve and why?
3. Write 10 unanswerable questions for your corpus and measure how often the system correctly abstains.
:::

:::takeaway
- Extractive QA predicts start and end positions of an answer span; SQuAD 2.0 adds abstention.
- Evaluate with Exact Match and token F1; generative QA needs faithfulness checks.
- Open-domain QA pipelines retrieve (sparse, dense, hybrid, re-ranked) and then read or generate.
- Grounding, citations and abstention are essential to avoid confident hallucinations.
:::

=== POST ===
slug: text-summarization
title: "Text Summarisation: Extractive and Abstractive Methods"
category: nlp
level: Intermediate
tags: summarization, extractive, abstractive, rouge, faithfulness
summary: Summarisation condenses documents while preserving key information. We compare extractive methods (TextRank) with abstractive neural models, evaluate with ROUGE and factual-consistency checks, and discuss long documents and hallucination.
---
Analysts face more reports, articles, meeting transcripts and case notes than anyone can read. **Automatic summarisation** produces a shorter version that preserves the most important information. It is among the most useful — and most error-prone — NLP applications, because a fluent summary that states something false can be worse than no summary at all.

## Two paradigms

- **Extractive**: select the most important sentences from the source and concatenate them. Faithful by construction (every sentence appears in the source), but can be choppy and redundant.
- **Abstractive**: generate new text that paraphrases and condenses. More fluent and concise, but risks **hallucination** — content not supported by the source.

Summaries can also be **single-document** or **multi-document**, **generic** or **query-focused** ("summarise what this report says about water supply"), and targeted to different lengths and audiences.

## Extractive methods

**Frequency-based** (Luhn, 1958): score sentences by the frequency of their important words.

**TextRank** (Mihalcea & Tarau, 2004): build a graph whose nodes are sentences and whose edge weights are sentence similarities; run PageRank; pick the top-ranked sentences. Sentences similar to many others are "central".

```python
import numpy as np, re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def textrank_summary(text, k=2, d=0.85, iters=50):
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    X = TfidfVectorizer().fit_transform(sents)
    S = cosine_similarity(X); np.fill_diagonal(S, 0)
    S = S / (S.sum(1, keepdims=True) + 1e-9)          # row-normalise into transition probabilities
    r = np.ones(len(sents)) / len(sents)
    for _ in range(iters):
        r = (1 - d) / len(sents) + d * S.T @ r       # PageRank iteration
    top = sorted(np.argsort(-r)[:k])                  # keep original order
    return " ".join(sents[i] for i in top)

doc = ("Floods affected five districts this week. Around 20,000 families were displaced. "
       "Authorities opened 40 temporary shelters in schools. Clean water is the most urgent need. "
       "Health workers warned of possible cholera outbreaks. Schools will remain closed until Sunday.")
print(textrank_summary(doc))
```

Supervised extractive models (e.g. BERTSum) classify each sentence as "include" or not, trained on labels derived from reference summaries.

## Abstractive methods

Encoder–decoder transformers fine-tuned on (document, summary) pairs: **BART**, **PEGASUS** (pretrained with "gap sentence generation" — masking whole important sentences and generating them, closely matching the summarisation task), T5, and — increasingly — instruction-tuned LLMs prompted to summarise.

Common datasets: CNN/DailyMail (news highlights, fairly extractive), XSum (single-sentence, highly abstractive BBC summaries), arXiv/PubMed (long scientific papers), and meeting and dialogue summarisation sets.

## Evaluation

**ROUGE** (Lin, 2004) measures n-gram overlap with reference summaries, emphasising recall:

- **ROUGE-1 / ROUGE-2**: unigram / bigram overlap.
- **ROUGE-L**: longest common subsequence.

$$
\text{ROUGE-N recall} = \frac{\sum_{\text{gram}_n \in \text{ref}}\text{Count}_{\text{match}}(\text{gram}_n)}{\sum_{\text{gram}_n \in \text{ref}}\text{Count}(\text{gram}_n)}
$$

ROUGE is cheap but rewards word overlap, not correctness: a summary can score well while stating something false, or score poorly while being an excellent paraphrase.

```python
# pip install rouge-score
from rouge_score import rouge_scorer
sc = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
ref = "About 20,000 families were displaced by floods; clean water is the most urgent need."
hyp = "Floods displaced 20,000 families, and clean water is urgently needed."
print({k: round(v.fmeasure, 3) for k, v in sc.score(ref, hyp).items()})
```

**Factual consistency (faithfulness)** requires separate evaluation:

- **Entailment-based** checks: does the source entail each summary sentence (using an NLI model)? (e.g. SummaC)
- **QA-based** checks: generate questions from the summary, answer them from the source, and compare (QAGS, QuestEval).
- **LLM-as-judge** evaluations with explicit rubrics — useful at scale, but validate against human judgements.
- **Human evaluation**: coverage, faithfulness, fluency, conciseness.

Studies found that a substantial fraction of abstractive summaries from earlier neural models contained unsupported information (Maynez et al., 2020). Modern LLMs are better, but not immune.

## Long documents

Transformers have limited context windows. Strategies:

- **Truncation** (loses information);
- **Chunk-and-merge / map-reduce**: summarise chunks, then summarise the summaries;
- **Extract-then-abstract**: select important passages first;
- **Long-context models** (Longformer/LED, long-context LLMs) — but check whether information from the middle of long inputs is used.

## Responsible summarisation

:::warning
In decision-making contexts — case files, medical records, incident reports — a summary may be the only thing a busy reader sees. Omitted caveats and hallucinated details can distort decisions about real people. Show source links for each summary sentence, highlight uncertainty, keep the original accessible, and have domain experts review summaries that inform consequential decisions.
:::

:::exercise
1. Compare TextRank, BART-large-CNN and an instruction-tuned LLM on five reports; score with ROUGE and with your own faithfulness check.
2. Implement an NLI-based faithfulness checker: split a summary into sentences and test whether each is entailed by the source.
3. Summarise a 30-page document with a map-reduce strategy and evaluate what important facts are lost.
:::

:::takeaway
- Extractive summarisation selects sentences (e.g. TextRank); abstractive summarisation generates new text.
- BART, PEGASUS, T5 and LLMs dominate abstractive summarisation.
- ROUGE measures overlap, not truth; evaluate factual consistency separately.
- Long documents need chunking or long-context models; show sources and review consequential summaries.
:::

=== POST ===
slug: nlp-evaluation-metrics
title: "Evaluating NLP Systems: Perplexity, BLEU, ROUGE, BERTScore and Human Judgement"
category: nlp
level: Intermediate
tags: evaluation, perplexity, bleu, rouge, bertscore, llm-as-judge, benchmarks
summary: How do we know if a language system is good? We survey intrinsic and extrinsic evaluation, overlap metrics, embedding-based metrics, learned metrics, LLM judges, human evaluation, benchmarks and their pitfalls.
---
Evaluation is the compass of NLP research and engineering. A model can only improve on what we measure — and it will happily exploit any weakness in the measurement. Generated text is especially hard to evaluate: there are many correct translations of a sentence and many good summaries of a document. This lecture surveys the toolbox and its limits.

## Intrinsic vs extrinsic evaluation

- **Intrinsic**: measures a component in isolation (perplexity of a language model, word-similarity correlation of embeddings).
- **Extrinsic**: measures impact on a downstream task or real user outcomes (does better perplexity improve speech recognition? do users resolve their question faster?).

Intrinsic metrics are cheap and fast; extrinsic ones are what ultimately matter.

## Classification-style tasks

Accuracy, precision, recall, F1 (macro for imbalance), entity-level F1 for NER, exact match and token F1 for extractive QA — covered in earlier lectures. Always report confidence intervals or variance across seeds.

## Perplexity

For language models:

$$
\text{PPL} = \exp\left(-\frac{1}{N}\sum_{i=1}^{N}\log P(w_i \mid w_{<i})\right)
$$

Lower is better. Comparable only for models with the **same tokeniser** (or when normalised per character/byte, e.g. bits per byte). Perplexity measures predictive fit, not helpfulness, truthfulness or safety.

## Overlap-based generation metrics

| Metric | Measures | Typical use |
|---|---|---|
| BLEU | Modified n-gram precision + brevity penalty | Machine translation |
| chrF | Character n-gram F-score | MT, morphologically rich languages |
| ROUGE-1/2/L | n-gram / LCS overlap (recall-oriented) | Summarisation |
| METEOR | Unigram matching with stems and synonyms | MT, captioning |
| CIDEr | TF-IDF-weighted n-gram similarity to many references | Image captioning |

They are cheap and reproducible, but reward **surface overlap**: they penalise valid paraphrases and cannot detect factual errors. They correlate reasonably with human judgement when comparing systems of very different quality, and poorly when comparing strong systems.

## Embedding-based and learned metrics

- **BERTScore** (Zhang et al., 2020): match each token of the candidate to its most similar token in the reference using contextual embeddings; compute precision, recall and F1 of these similarities. Captures paraphrases.
- **COMET, BLEURT**: neural models trained to predict human quality ratings — substantially better correlation with human judgements in machine translation.
- **Faithfulness metrics**: NLI-based and QA-based consistency checks for summarisation.

```python
# pip install bert-score sacrebleu
from bert_score import score
import sacrebleu
cands = ["The clinic will open at 9 am tomorrow."]
refs = ["Tomorrow the health centre opens at nine in the morning."]
P, R, F = score(cands, refs, lang="en")
print("BERTScore F1:", round(F.item(), 3), " BLEU:", round(sacrebleu.corpus_bleu(cands, [refs]).score, 1))
```

BLEU is low (little exact overlap) while BERTScore is high (the meaning matches).

## LLM-as-a-judge

Large language models can grade outputs against rubrics or compare two responses pairwise. This scales evaluation of open-ended generation and correlates well with human preferences on many tasks (e.g. studies accompanying MT-Bench and Chatbot Arena). Known biases:

- **Position bias** — preferring the first (or second) answer; mitigate by swapping order.
- **Verbosity bias** — preferring longer answers.
- **Self-preference** — favouring outputs from the same model family.
- Limited reliability on specialised or factual content the judge itself gets wrong.

Validate LLM judges against human labels on a sample before trusting them.

## Human evaluation

The gold standard, but it must be designed carefully:

- Clear criteria (adequacy, fluency, faithfulness, helpfulness, harmlessness) with rubrics.
- Trained annotators; measure **inter-annotator agreement**.
- Pairwise comparisons are often more reliable than absolute scores.
- Blind the evaluators to which system produced which output; randomise order.
- Sufficient sample size for statistical significance.
- Domain experts for specialised content; native speakers for each language.

## Benchmarks and their pitfalls

Benchmarks (GLUE, SuperGLUE, SQuAD, MMLU, HellaSwag, BIG-bench, HELM and many more) drive progress but have recurring problems:

- **Saturation**: models reach or exceed human baselines quickly, often before the underlying capability is solved.
- **Annotation artefacts and shortcuts**: e.g. in natural-language inference datasets, hypothesis-only models performed far above chance because words like "not" correlated with contradiction.
- **Contamination**: test items appear in web-scale training data, inflating scores.
- **Narrowness**: benchmarks cover a small slice of real use, often English-centric.
- **Goodhart's law**: "When a measure becomes a target, it ceases to be a good measure."

Responses include adversarial and dynamic benchmarks (Adversarial NLI, Dynabench), held-out private test sets, contamination checks, holistic evaluations across many metrics (HELM), and — most important for practitioners — **custom evaluation sets built from your own use case**.

:::tip
For any production NLP system, build a **golden evaluation set**: a few hundred real, representative inputs with expert-verified outputs or rubrics, covering important edge cases and every language you serve. Re-run it on every model or prompt change, like a unit test suite.
:::

:::exercise
1. Compute BLEU, chrF, ROUGE-L and BERTScore for five paraphrases of one reference sentence, including one that changes a key fact. Which metrics notice the error?
2. Use an LLM to judge 20 pairs of answers in both orders. How often does the verdict flip?
3. Design a 50-item golden evaluation set and rubric for a chatbot that answers questions about a public service.
:::

:::takeaway
- Distinguish intrinsic metrics (perplexity) from extrinsic impact.
- Overlap metrics (BLEU, ROUGE, chrF) are cheap but blind to meaning and truth; BERTScore and COMET correlate better with humans.
- LLM judges scale evaluation but have position, length and self-preference biases.
- Human evaluation and custom golden sets remain essential; beware benchmark saturation and contamination.
:::

=== POST ===
slug: semantic-search-information-retrieval
title: "Information Retrieval and Semantic Search"
category: nlp
level: Intermediate
tags: information retrieval, semantic search, dense retrieval, bm25, reranking, vector search
summary: Search is the most used NLP application. We cover indexing, BM25, dense bi-encoder retrieval, cross-encoder re-ranking, hybrid search, approximate nearest neighbours, and evaluation with recall@k, MRR and nDCG.
---
Every day, billions of searches are answered by information retrieval (IR) systems. Inside organisations, good search over policies, reports and FAQs saves staff hours and helps people find accurate information. And retrieval is the foundation of retrieval-augmented generation (RAG): an LLM can only answer from documents that retrieval finds. This lecture covers modern retrieval end to end.

## The retrieval problem

Given a query $q$ and a collection of documents (or passages) $\mathcal{D}$, rank documents by relevance. Because collections may contain millions of items, systems use a **two-stage** design:

1. **First-stage retrieval** — fast, recall-oriented: retrieve the top 100–1,000 candidates.
2. **Re-ranking** — slower, precise: reorder the candidates with a more expensive model.

## Lexical retrieval: the inverted index and BM25

An **inverted index** maps each term to the list of documents containing it, enabling fast lookup of candidate documents for query terms. Documents are scored with **BM25** (see the TF-IDF lecture). Lexical retrieval excels at exact matches: names, codes, rare technical terms, ID numbers. It fails on **vocabulary mismatch** — "newborn paperwork" vs "birth registration".

## Dense retrieval with bi-encoders

A **bi-encoder** embeds queries and documents independently into the same vector space; relevance is the dot product or cosine similarity:

$$
s(q, d) = E_Q(q)^\top E_D(d)
$$

Document embeddings are computed **once** offline; at query time only the query is encoded, followed by nearest-neighbour search. Bi-encoders (DPR, sentence-transformers, E5, BGE, GTE and multilingual variants) are trained contrastively with in-batch and hard negatives, and capture meaning beyond exact words.

```python
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np

docs = ["Birth registration is free at the civil registry office.",
        "Vaccinations for children under five are given every Tuesday at the health post.",
        "To renew a passport, submit the old passport and two photographs.",
        "Cash assistance eligibility depends on household size and income.",
        "Lost birth certificates can be replaced by applying with a parent's ID."]
bi = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
D = bi.encode(docs, normalize_embeddings=True)

query = "my baby's birth paper got lost"
q = bi.encode([query], normalize_embeddings=True)[0]
cand = np.argsort(-(D @ q))[:3]                         # first stage: dense retrieval
print([docs[i] for i in cand])

ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")        # second stage: re-ranking
scores = ce.predict([(query, docs[i]) for i in cand])
print(docs[cand[int(np.argmax(scores))]])
```

## Cross-encoder re-ranking

A **cross-encoder** feeds query and document **together** into a transformer, letting every query token attend to every document token, and outputs a relevance score. It is far more accurate than a bi-encoder but too slow to run over the whole collection — perfect for re-ranking the top candidates. **Late-interaction** models such as ColBERT keep per-token embeddings and compute fine-grained similarity efficiently, a middle ground.

## Hybrid search

Lexical and dense retrieval make different errors. **Hybrid search** combines them, for example with **Reciprocal Rank Fusion (RRF)**:

$$
\text{RRF}(d) = \sum_{r \in \text{rankers}}\frac{1}{k + \text{rank}_r(d)}, \qquad k \approx 60
$$

Hybrid retrieval is a robust default for real-world search and RAG, especially when queries contain names or codes.

## Approximate nearest neighbour (ANN) search

Exact search over millions of vectors is slow. ANN indexes trade a little recall for large speed-ups:

- **HNSW** (Hierarchical Navigable Small World graphs): navigate a multi-layer proximity graph; excellent recall/speed trade-off.
- **IVF** (inverted file): cluster vectors and search only the nearest clusters.
- **Product quantisation (PQ)**: compress vectors into short codes for memory efficiency.

Libraries (FAISS, hnswlib) and vector databases (e.g. pgvector, Qdrant, Weaviate, Milvus, Elasticsearch/OpenSearch vector fields) implement these.

## Chunking documents

Long documents are split into **passages** (e.g. 200–500 tokens, with overlap) before embedding. Chunk boundaries matter: splitting mid-table or mid-procedure harms retrieval. Keep metadata (title, section, date, language, source URL) with each chunk for filtering and citation.

## Evaluation

With relevance judgements for a set of test queries:

- **Recall@k**: fraction of relevant documents in the top $k$ — key for first-stage retrieval and RAG.
- **MRR** (mean reciprocal rank): $\frac{1}{|Q|}\sum_q\frac{1}{\text{rank of first relevant}}$.
- **nDCG@k**: rewards relevant results near the top, supports graded relevance:

$$
\text{DCG@}k = \sum_{i=1}^{k}\frac{2^{\text{rel}_i} - 1}{\log_2(i + 1)}, \qquad \text{nDCG@}k = \frac{\text{DCG@}k}{\text{IDCG@}k}
$$

Public benchmarks include MS MARCO, BEIR (zero-shot across domains) and MTEB for embeddings. Always also evaluate on **your own queries** — log real user queries (with privacy safeguards) and label relevance.

:::note
Retrieval quality usually limits RAG quality. If the right passage is not in the top results, even the best LLM cannot answer correctly. Measure recall@k of your retriever before tuning prompts.
:::

:::exercise
1. Build BM25, dense and hybrid (RRF) retrieval over 200 passages from your domain; evaluate recall@5 and MRR on 40 labelled queries.
2. Add a cross-encoder re-ranker and measure the change in nDCG@5 and latency.
3. Compare three chunk sizes (100, 300, 800 tokens) for retrieval recall on the same queries.
:::

:::takeaway
- Retrieval is two-stage: recall-oriented candidate generation, then precise re-ranking.
- BM25 matches exact terms; bi-encoders match meaning; cross-encoders re-rank accurately.
- Hybrid search (e.g. RRF) is a robust default; ANN indexes (HNSW, IVF, PQ) make vector search fast.
- Chunk carefully, keep metadata, and evaluate with recall@k, MRR and nDCG on real queries.
:::
