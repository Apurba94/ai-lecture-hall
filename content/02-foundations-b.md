=== POST ===
slug: propositional-logic-for-ai
title: Propositional Logic for AI: Syntax, Semantics and Inference
category: foundations
level: Beginner
tags: logic, propositional logic, inference, sat, resolution
summary: Logic gives an agent a language for knowledge and a mechanical way to draw conclusions. We cover syntax, truth tables, entailment, resolution and the SAT problem that powers modern solvers.
---
A knowledge-based agent keeps a **knowledge base (KB)** of sentences about the world and uses **inference** to derive new sentences. The simplest formal language that supports this is propositional logic. Although simple, it is the foundation of hardware verification, planning-as-satisfiability and many industrial solvers. Let us learn it properly.

## Syntax

Propositional logic has **atomic propositions** — symbols such as $P$, $Q$, `Raining` — each of which is either true or false. Complex sentences are built with connectives:

| Connective | Symbol | Read as |
|---|---|---|
| Negation | $\neg P$ | not P |
| Conjunction | $P \land Q$ | P and Q |
| Disjunction | $P \lor Q$ | P or Q |
| Implication | $P \Rightarrow Q$ | if P then Q |
| Biconditional | $P \Leftrightarrow Q$ | P if and only if Q |

Precedence from highest to lowest is $\neg, \land, \lor, \Rightarrow, \Leftrightarrow$.

## Semantics: models and truth

A **model** assigns true or false to every symbol. The truth of a complex sentence is computed recursively. The only connective that surprises students is implication: $P \Rightarrow Q$ is false *only* when $P$ is true and $Q$ is false. "If the moon is made of cheese, then 2 + 2 = 5" is *true*, because the premise is false.

:::definition Entailment
$KB \models \alpha$ ("the KB entails alpha") means that $\alpha$ is true in **every** model in which $KB$ is true. Entailment is the formal notion of "follows logically".
:::

A sentence is **valid** (a tautology) if it is true in all models, and **satisfiable** if it is true in at least one. These are linked by a fundamental result:

$$
KB \models \alpha \quad \Longleftrightarrow \quad KB \land \neg\alpha \text{ is unsatisfiable}
$$

This is proof by contradiction, and it is how most automated reasoners work.

## Inference by model checking

The simplest inference algorithm enumerates all $2^n$ models and checks that $\alpha$ holds wherever $KB$ holds. It is **sound** (never derives false conclusions) and **complete** (derives every entailed sentence), but exponential.

```python
from itertools import product

def tt_entails(kb, alpha, symbols):
    """kb and alpha are functions from a model dict to bool."""
    for values in product([True, False], repeat=len(symbols)):
        model = dict(zip(symbols, values))
        if kb(model) and not alpha(model):
            return False
    return True

# KB: (Rain => WetGrass) and Rain.  Query: WetGrass
kb = lambda m: ((not m["Rain"]) or m["Wet"]) and m["Rain"]
print(tt_entails(kb, lambda m: m["Wet"], ["Rain", "Wet"]))   # True
```

## Inference rules and theorem proving

Instead of enumerating models, we can apply **inference rules** that are known to be sound:

- **Modus Ponens:** from $\alpha \Rightarrow \beta$ and $\alpha$, infer $\beta$.
- **And-Elimination:** from $\alpha \land \beta$, infer $\alpha$.
- **Logical equivalences:** De Morgan's laws, contraposition $(\alpha \Rightarrow \beta) \equiv (\neg\beta \Rightarrow \neg\alpha)$, and so on.

### Resolution

One rule alone — **resolution** — is complete for refutation when sentences are in **Conjunctive Normal Form (CNF)**, a conjunction of clauses where each clause is a disjunction of literals:

$$
\frac{\ell_1 \lor \dots \lor \ell_k, \qquad \neg\ell_i \lor m_1 \lor \dots \lor m_n}{\ell_1 \lor \dots \lor \ell_{i-1} \lor \ell_{i+1} \lor \dots \lor \ell_k \lor m_1 \lor \dots \lor m_n}
$$

To prove $KB \models \alpha$, convert $KB \land \neg\alpha$ to CNF and repeatedly resolve pairs of clauses. If you derive the **empty clause**, you have a contradiction, and the entailment is proved.

:::example
KB: $P \Rightarrow Q$, $Q \Rightarrow R$, $P$. Prove $R$.
CNF clauses: $(\neg P \lor Q)$, $(\neg Q \lor R)$, $(P)$, and the negated goal $(\neg R)$.
Resolve $(\neg P \lor Q)$ with $(P)$ → $(Q)$. Resolve $(Q)$ with $(\neg Q \lor R)$ → $(R)$. Resolve $(R)$ with $(\neg R)$ → empty clause. ✔
:::

## Horn clauses and efficient inference

A **Horn clause** has at most one positive literal, e.g. $\neg A \lor \neg B \lor C$, which is equivalent to $A \land B \Rightarrow C$. Knowledge bases of Horn clauses support **forward chaining** (data-driven: fire rules whose premises are known) and **backward chaining** (goal-driven: work backwards from the query) in time *linear* in the size of the KB. The Prolog language is built on backward chaining over Horn clauses.

## SAT solvers: logic at industrial scale

Deciding satisfiability of a CNF formula (SAT) was the first problem proved **NP-complete** (Cook, 1971). Yet modern **CDCL** (Conflict-Driven Clause Learning) solvers routinely handle formulas with millions of variables, thanks to clever ideas: unit propagation, learning new clauses from conflicts, non-chronological backjumping and restarts. SAT solvers verify microprocessors, schedule airlines and solve planning problems.

:::note
The gap between worst-case complexity (NP-complete) and practical performance is one of the most important lessons in computer science. Real problem instances have structure that clever algorithms exploit. Never dismiss a method solely because the worst case is exponential.
:::

## Limitations

Propositional logic cannot express general statements compactly. To say "every student who passes the exam gets a certificate" for 500 students, you need 500 separate sentences. **First-order logic**, our next lecture, fixes this by adding objects, relations and quantifiers.

:::exercise
1. Build the truth table for $(P \Rightarrow Q) \Leftrightarrow (\neg P \lor Q)$ and confirm it is valid.
2. Convert $(A \lor B) \Rightarrow (C \land D)$ to CNF.
3. Using resolution, prove $\neg P$ from $P \Rightarrow Q$ and $\neg Q$ (modus tollens).
:::

:::takeaway
- Propositional logic combines true/false atoms with $\neg, \land, \lor, \Rightarrow, \Leftrightarrow$.
- $KB \models \alpha$ iff $KB \land \neg\alpha$ is unsatisfiable.
- Resolution on CNF is refutation-complete; Horn clauses allow linear-time chaining.
- SAT is NP-complete in theory but highly practical with CDCL solvers.
:::

=== POST ===
slug: first-order-logic-and-inference
title: First-Order Logic: Objects, Relations, Quantifiers and Unification
category: foundations
level: Intermediate
tags: logic, first-order logic, unification, prolog, knowledge
summary: First-order logic lets us talk about objects and relations with quantifiers. We study its syntax and semantics, unification, generalised modus ponens, and resolution-based theorem proving.
---
Propositional logic treats the world as a list of facts. But the world has *structure*: it contains **objects** (students, courses, cities), **relations** between them (enrolled-in, adjacent-to) and **functions** (the lecturer *of* a course). First-order logic (FOL), also called predicate logic, captures that structure and is the most important knowledge-representation language in classical AI.

## Syntax of FOL

FOL adds the following building blocks:

- **Constants** name specific objects: `Janin`, `AUST`, `CSE101`.
- **Predicates** express properties and relations: `Student(x)`, `Enrolled(x, c)`.
- **Functions** map objects to objects: `Lecturer(CSE101)`.
- **Variables**: $x, y, z$.
- **Quantifiers**: universal $\forall$ ("for all") and existential $\exists$ ("there exists").
- **Equality**: $=$.

Examples:

$$
\forall x\; \text{Student}(x) \land \text{Passes}(x, \text{Exam}) \Rightarrow \text{Certified}(x)
$$

$$
\exists c\; \text{Course}(c) \land \text{Enrolled}(\text{Rafi}, c)
$$

## Two classic mistakes

:::warning
1. **Using $\land$ with $\forall$.** $\forall x\; \text{Student}(x) \land \text{Smart}(x)$ says *everything in the universe* is a smart student. You almost always want $\Rightarrow$ with $\forall$.
2. **Using $\Rightarrow$ with $\exists$.** $\exists x\; \text{Student}(x) \Rightarrow \text{Smart}(x)$ is true if there is *any* non-student at all. You almost always want $\land$ with $\exists$.
:::

Quantifier order matters too: $\forall x\, \exists y\; \text{Loves}(x, y)$ ("everyone loves someone") is very different from $\exists y\, \forall x\; \text{Loves}(x, y)$ ("there is someone whom everyone loves").

The quantifiers are dual: $\forall x\; \neg P(x) \equiv \neg \exists x\; P(x)$.

## Semantics

A model in FOL contains a **domain** of objects and an **interpretation** mapping constants to objects, predicates to relations and functions to functions. Because domains may be infinite, we can no longer check entailment by enumerating models — we need proof procedures.

## Knowledge engineering

Writing a good knowledge base is a craft. The process is:

1. Identify the task and the questions the KB must answer.
2. Assemble relevant knowledge (interview experts).
3. Decide on a **vocabulary** of predicates, functions and constants — an *ontology*.
4. Encode general knowledge as axioms.
5. Encode the specific problem instance.
6. Pose queries and debug.

## Unification

Inference in FOL requires matching sentences that contain variables. **Unification** finds a substitution $\theta$ that makes two expressions identical:

$$
\text{Unify}(\text{Knows}(\text{John}, x),\; \text{Knows}(y, \text{Mother}(y))) = \{y/\text{John},\; x/\text{Mother}(\text{John})\}
$$

We always want the **most general unifier (MGU)** — the one that commits to the fewest bindings.

```python
def unify(x, y, theta):
    """Terms: variables are strings starting with '?', compound terms are tuples."""
    if theta is None:
        return None
    if x == y:
        return theta
    if isinstance(x, str) and x.startswith("?"):
        return unify_var(x, y, theta)
    if isinstance(y, str) and y.startswith("?"):
        return unify_var(y, x, theta)
    if isinstance(x, tuple) and isinstance(y, tuple) and len(x) == len(y):
        for a, b in zip(x, y):
            theta = unify(a, b, theta)
        return theta
    return None

def unify_var(v, t, theta):
    if v in theta:
        return unify(theta[v], t, theta)
    if isinstance(t, str) and t in theta:
        return unify(v, theta[t], theta)
    if occurs(v, t, theta):
        return None                       # occurs check prevents x = f(x)
    return {**theta, v: t}

def occurs(v, t, theta):
    if v == t:
        return True
    if isinstance(t, str) and t in theta:
        return occurs(v, theta[t], theta)
    return isinstance(t, tuple) and any(occurs(v, a, theta) for a in t)

print(unify(("Knows", "John", "?x"), ("Knows", "?y", ("Mother", "?y")), {}))
```

## Generalised Modus Ponens

With unification, modus ponens lifts to FOL: from $p_1', \dots, p_n'$ and $(p_1 \land \dots \land p_n \Rightarrow q)$, if a substitution $\theta$ makes each $p_i'\theta = p_i\theta$, infer $q\theta$. This powers **forward chaining** (used in production systems and databases) and **backward chaining** (used in Prolog).

```prolog
parent(tom, bob).
parent(bob, ann).
grandparent(X, Z) :- parent(X, Y), parent(Y, Z).
% ?- grandparent(tom, Who).   ->  Who = ann
```

## Resolution for FOL

For full FOL, we convert sentences to clausal form. Two new steps appear:

- **Skolemisation** removes existential quantifiers: $\forall x\, \exists y\; \text{Loves}(x, y)$ becomes $\text{Loves}(x, F(x))$, where $F$ is a new *Skolem function*.
- **Dropping universal quantifiers**, since all remaining variables are implicitly universal.

Resolution with unification is **refutation-complete**: if $KB \models \alpha$, it will eventually derive a contradiction from $KB \land \neg\alpha$. However, FOL entailment is only **semi-decidable** — if $\alpha$ is *not* entailed, the procedure may run forever. This is a deep consequence of Gödel's and Turing's work.

:::note
Pure logic-based AI struggles with uncertainty, exceptions ("birds fly — except penguins") and learning from data. That is why modern AI blends logic with probability and learning. But knowledge graphs, database query languages, program verification and the "reasoning" evaluations of large language models all descend directly from FOL. Mastering it makes you a clearer thinker.
:::

:::exercise
1. Translate into FOL: "Every CSE student takes at least one AI course", "No course is taught by every lecturer".
2. Find the MGU of $P(x, f(y), z)$ and $P(g(a), f(b), x)$.
3. Explain why the occurs check is necessary with an example.
:::

:::takeaway
- FOL represents objects, relations, functions and quantified statements.
- Use $\Rightarrow$ with $\forall$, and $\land$ with $\exists$.
- Unification finds the most general substitution making terms identical.
- Resolution is refutation-complete but FOL entailment is only semi-decidable.
:::

=== POST ===
slug: knowledge-representation-ontologies
title: Knowledge Representation: Semantic Networks, Frames, Ontologies and Knowledge Graphs
category: foundations
level: Intermediate
tags: knowledge representation, ontologies, knowledge graphs, semantic web
summary: How should an intelligent system store what it knows? We compare semantic networks, frames, description logics and modern knowledge graphs, and discuss the trade-off between expressiveness and tractability.
---
An agent's intelligence depends not only on its reasoning algorithm but on *how its knowledge is represented*. The same fact can be easy or impossible to use depending on its form. Today we survey the major representation schemes and connect them to the knowledge graphs that power today's search engines and assistants.

## What a representation must do

A good knowledge representation (KR) scheme should offer:

- **Representational adequacy** — it can express what we need.
- **Inferential adequacy** — it supports deriving new knowledge.
- **Inferential efficiency** — inference is fast enough to be useful.
- **Acquisitional efficiency** — knowledge can be added easily.

There is an unavoidable tension: the more expressive a language, the harder inference becomes. Full first-order logic is expressive but undecidable; simple databases are fast but inflexible.

## Categories and objects

Much of human knowledge is organised in **categories** (Dog, Mammal, Animal) with **inheritance**: properties of a category are inherited by its members and subcategories. In FOL we might write:

$$
\forall x\; \text{Dog}(x) \Rightarrow \text{Mammal}(x)
$$

Taxonomies of this kind exist in biology, medicine (SNOMED CT has hundreds of thousands of concepts) and library science.

## Semantic networks

A **semantic network** represents knowledge as a graph: nodes are concepts or objects, and labelled edges are relations such as `is-a`, `part-of` or `has-colour`. For example: `Tweety —is-a→ Bird —is-a→ Animal`, `Bird —can→ Fly`.

Semantic networks make inheritance natural: to find whether Tweety can fly, follow `is-a` links upward. They also support **default reasoning**: `Penguin —can→ ¬Fly` overrides the inherited default because it is more specific.

## Frames

Marvin Minsky's **frames** (1974) organise knowledge into structured records with **slots** and **fillers**, much like objects in object-oriented programming:

```text
Frame: Lecture
  is-a:        Event
  location:    <Room>          default: Main Building
  lecturer:    <Person>
  duration:    default 90 minutes
  if-needed:   compute end_time from start_time + duration
```

Slots can have defaults, constraints and attached procedures ("demons") that run when values are needed or changed. Frames influenced OOP and modern schema design.

## Description logics and OWL

**Description logics (DLs)** are decidable fragments of FOL designed for defining and reasoning about categories. A DL definition looks like:

$$
\text{Parent} \equiv \text{Person} \sqcap \exists\, \text{hasChild}.\text{Person}
$$

DL reasoners answer questions such as **subsumption** (is every Parent a Person?) and **classification** (place a new concept correctly in the hierarchy). The Web Ontology Language **OWL**, a W3C standard, is based on description logics and underlies biomedical ontologies such as the Gene Ontology.

## Knowledge graphs

A **knowledge graph** stores facts as triples $(\text{subject}, \text{predicate}, \text{object})$:

```text
(Dhaka, capitalOf, Bangladesh)
(AUST, locatedIn, Dhaka)
(Janin_A_Apurba, alumnusOf, AUST)
```

The **RDF** standard encodes triples; **SPARQL** queries them. Large public knowledge graphs such as Wikidata contain over a hundred million items. Search engines use knowledge graphs to answer factual queries directly.

```python
triples = {
    ("Dhaka", "capitalOf", "Bangladesh"),
    ("AUST", "locatedIn", "Dhaka"),
    ("Bangladesh", "locatedIn", "SouthAsia"),
}

def located_in(x, triples):
    """Transitive closure of 'locatedIn' and 'capitalOf' treated as containment."""
    found, frontier = set(), {x}
    while frontier:
        nxt = {o for (s, p, o) in triples if s in frontier and p in {"locatedIn", "capitalOf"}}
        frontier = nxt - found
        found |= nxt
    return found

print(located_in("AUST", triples))   # {'Dhaka', 'Bangladesh', 'SouthAsia'}
```

### Knowledge graph embeddings

Modern ML represents entities and relations as vectors. The **TransE** model, for example, learns embeddings such that $\mathbf{e}_{\text{head}} + \mathbf{r} \approx \mathbf{e}_{\text{tail}}$ for true triples. This enables **link prediction** — inferring missing facts — and bridges symbolic knowledge with neural learning.

## Reasoning with defaults and change

Real knowledge is messy. Two classic difficulties:

- **Non-monotonic reasoning** — conclusions may be withdrawn given new information ("Tweety flies" until we learn Tweety is a penguin). Circumscription and default logic formalise this.
- **The frame problem** — when an action occurs, how do we avoid stating everything that did *not* change? Successor-state axioms offer a solution in the situation calculus.

:::note
Large language models store an enormous amount of knowledge implicitly in their weights, but that knowledge is hard to inspect, update or verify. A very active research area combines LLMs with explicit knowledge graphs — using the graph for precise, updatable facts and the model for language. Retrieval-augmented generation, which we study later, is one practical form of this marriage.
:::

:::exercise
1. Draw a semantic network for your university: departments, courses, lecturers and students.
2. Write a description-logic definition for "a Course taught only by Professors".
3. Represent ten facts about your city as triples and write a query that finds everything located in it.
:::

:::takeaway
- KR schemes trade expressiveness against efficient inference.
- Semantic networks and frames organise knowledge with inheritance and defaults.
- Description logics (OWL) are decidable fragments of FOL for ontologies.
- Knowledge graphs store triples at web scale and can be embedded as vectors for ML.
:::

=== POST ===
slug: expert-systems-rule-based-ai
title: Expert Systems and Rule-Based AI: Rise, Fall and Legacy
category: foundations
level: Beginner
tags: expert systems, rules, forward chaining, mycin, history
summary: Expert systems were AI's first commercial success. We build a small rule engine, examine certainty factors from MYCIN, and ask why rule-based systems remain useful — and where they fail.
---
In the 1980s, "AI" in industry meant one thing: **expert systems**. These programs captured the knowledge of human specialists as if–then rules and used it to diagnose diseases, configure computers and approve loans. Studying them teaches us both the power of explicit knowledge and the reasons the field moved towards learning.

## Architecture of an expert system

A classical expert system has five parts:

1. **Knowledge base** — the domain rules, e.g. *IF the infection is meningitis AND the patient is a child THEN consider organism X*.
2. **Working memory** — the facts about the current case.
3. **Inference engine** — applies rules to facts (forward or backward chaining).
4. **Explanation facility** — answers "why?" and "how?" questions by showing the chain of rules.
5. **Knowledge acquisition interface** — tools for knowledge engineers to add and edit rules.

The separation of *knowledge* from *inference* was a key insight: the same engine (an "expert system shell") could be reused across domains.

## Forward chaining with a tiny rule engine

```python
RULES = [
    ({"fever", "cough"}, "respiratory_infection"),
    ({"respiratory_infection", "chest_pain"}, "suspect_pneumonia"),
    ({"suspect_pneumonia"}, "recommend_xray"),
    ({"rash", "fever"}, "suspect_measles"),
]

def forward_chain(facts, rules):
    facts = set(facts)
    trace = []
    changed = True
    while changed:
        changed = False
        for premises, conclusion in rules:
            if premises <= facts and conclusion not in facts:
                facts.add(conclusion)
                trace.append(f"{' & '.join(sorted(premises))} => {conclusion}")
                changed = True
    return facts, trace

facts, why = forward_chain({"fever", "cough", "chest_pain"}, RULES)
print(facts)
print("\n".join(why))      # the explanation facility for free
```

Real engines such as **CLIPS** and **Drools** use the **Rete algorithm**, which avoids re-checking every rule on every cycle by caching partial matches in a network. This makes rule systems with thousands of rules run efficiently.

### Conflict resolution

When several rules can fire, the engine needs a policy: prefer the most specific rule, the most recently added fact, or a rule with higher priority ("salience"). The choice can change the system's behaviour significantly.

## Handling uncertainty: MYCIN's certainty factors

**MYCIN** (Stanford, 1970s) diagnosed bacterial blood infections with around 600 rules. Because medical evidence is uncertain, each rule carried a **certainty factor (CF)** in $[-1, 1]$. When two rules support the same conclusion with positive factors $CF_1$ and $CF_2$, MYCIN combined them as

$$
CF = CF_1 + CF_2 (1 - CF_1)
$$

In evaluations, MYCIN's recommendations were judged comparable to those of infectious-disease specialists — impressive for the era — though it was never used routinely in hospitals, due to legal, ethical and integration concerns.

:::note
Certainty factors were later shown to be inconsistent with probability theory in some situations — for example, they can double-count correlated evidence. This motivated the adoption of **Bayesian networks**, which we will study soon. It is a textbook case of engineering heuristics being replaced by principled mathematics.
:::

## Famous systems

| System | Domain | Notable fact |
|---|---|---|
| DENDRAL (1965) | Inferring molecular structure from mass spectra | Often called the first expert system |
| MYCIN (1970s) | Blood infection diagnosis | Introduced certainty factors and explanations |
| XCON / R1 (1980) | Configuring DEC VAX computers | Reported large annual savings for DEC |
| PROSPECTOR | Mineral exploration | Credited with helping locate a molybdenum deposit |

## Why expert systems declined

1. **The knowledge acquisition bottleneck.** Experts find it hard to articulate what they know; interviewing them is slow and expensive.
2. **Brittleness.** Systems failed abruptly outside their narrow domain and had no common sense.
3. **Maintenance.** Thousands of interacting rules became impossible to update safely.
4. **No learning.** They could not improve from experience or new data.

Machine learning addressed the first and fourth problems directly: instead of asking experts for rules, learn the mapping from examples.

## The legacy is alive

Rule-based systems never disappeared. They run in **business rules engines** for insurance and banking, **clinical decision support** alerts, **compliance** checks and **fraud detection**, where auditability matters. Many modern production ML systems are hybrids: a learned model produces a score, and explicit rules enforce hard policy constraints. Moreover, the explanation facility of expert systems foreshadowed today's field of **Explainable AI**.

:::tip
When a problem has clear, stable, legally mandated logic — "applicants under 18 cannot open an account" — write a rule, not a model. Use ML where patterns are too complex or too variable to write down. Knowing which tool to use is part of engineering maturity.
:::

:::exercise
1. Extend the rule engine to support backward chaining: given a goal, find which facts would need to be true.
2. Compute the combined CF when two rules support a hypothesis with CF 0.6 and 0.5.
3. List three situations in your country's public services where a transparent rule-based system would be preferable to a black-box model.
:::

:::takeaway
- Expert systems separate a rule-based knowledge base from a generic inference engine.
- Forward chaining (data-driven) and backward chaining (goal-driven) are the core inference modes; Rete makes them fast.
- They declined due to the knowledge acquisition bottleneck, brittleness and inability to learn.
- Rules remain valuable for auditable, policy-driven decisions and hybrid ML systems.
:::

=== POST ===
slug: bayesian-networks
title: Bayesian Networks: Reasoning Under Uncertainty
category: foundations
level: Intermediate
tags: probability, bayesian networks, graphical models, inference
summary: A Bayesian network encodes a joint probability distribution compactly using conditional independence. We learn the semantics, d-separation, exact inference by enumeration and variable elimination, and approximate sampling.
---
Real agents never have complete certainty. Symptoms suggest diseases but do not prove them; sensors are noisy. Probability theory is the correct calculus for this situation, but a naïve joint distribution over $n$ binary variables needs $2^n - 1$ numbers. **Bayesian networks**, introduced by Judea Pearl in the 1980s, solve this by exploiting conditional independence. Pearl received the Turing Award for this work.

## Definition

A Bayesian network (BN) is a **directed acyclic graph (DAG)** in which:

- each node is a random variable $X_i$;
- each node has a **conditional probability table (CPT)** $P(X_i \mid \text{Parents}(X_i))$.

The network represents the joint distribution as a product of local factors:

$$
P(X_1, \dots, X_n) = \prod_{i=1}^{n} P\big(X_i \mid \text{Parents}(X_i)\big)
$$

If each variable has at most $k$ parents, the network needs $O(n\,2^k)$ numbers instead of $O(2^n)$ — an exponential saving.

## The classic burglary example

Pearl's example: your house has an alarm that can be triggered by a **Burglary** (B) or an **Earthquake** (E). Two neighbours, **John** (J) and **Mary** (M), may call you when they hear the alarm (A).

The structure is B → A ← E, A → J, A → M. Plausible CPTs:

| | |
|---|---|
| $P(B) = 0.001$ | $P(E) = 0.002$ |
| $P(A \mid B, E) = 0.95$ | $P(A \mid B, \neg E) = 0.94$ |
| $P(A \mid \neg B, E) = 0.29$ | $P(A \mid \neg B, \neg E) = 0.001$ |
| $P(J \mid A) = 0.90$, $P(J \mid \neg A) = 0.05$ | $P(M \mid A) = 0.70$, $P(M \mid \neg A) = 0.01$ |

Ten numbers specify a joint distribution over 32 outcomes.

## Conditional independence and d-separation

The graph encodes independence assumptions. Each node is conditionally independent of its non-descendants given its parents. More generally, **d-separation** tells us when two sets of variables are independent given evidence. Three patterns matter:

1. **Chain** $X \to Y \to Z$: $X$ and $Z$ are dependent, but independent given $Y$.
2. **Fork** $X \leftarrow Y \to Z$: common cause; independent given $Y$.
3. **Collider** $X \to Y \leftarrow Z$: $X$ and $Z$ are *independent*, but become **dependent** once $Y$ (or a descendant) is observed.

:::note
The collider case produces **explaining away**. Burglary and earthquake are independent. But if the alarm rings and you learn there was an earthquake, the probability of a burglary *drops* — the earthquake explains the alarm. Humans reason this way intuitively; rule-based systems with certainty factors could not.
:::

## Exact inference by enumeration

Query: what is $P(B \mid j, m)$, the probability of a burglary given both neighbours call? We sum out the hidden variables $E$ and $A$:

$$
P(B \mid j, m) = \alpha \sum_{e} \sum_{a} P(B)\,P(e)\,P(a \mid B, e)\,P(j \mid a)\,P(m \mid a)
$$

```python
from itertools import product

P_B, P_E = 0.001, 0.002
P_A = {(True, True): .95, (True, False): .94, (False, True): .29, (False, False): .001}
P_J = {True: .90, False: .05}
P_M = {True: .70, False: .01}

def pr(p, v):  # probability that a Boolean variable with P(true)=p has value v
    return p if v else 1 - p

def joint(b, e, a, j, m):
    return pr(P_B, b) * pr(P_E, e) * pr(P_A[(b, e)], a) * pr(P_J[a], j) * pr(P_M[a], m)

unnorm = {b: sum(joint(b, e, a, True, True) for e, a in product([True, False], repeat=2))
          for b in [True, False]}
z = sum(unnorm.values())
print({b: round(v / z, 4) for b, v in unnorm.items()})   # {True: 0.2842, False: 0.7158}
```

Even when both neighbours call, the probability of a burglary is only about 28%, because burglaries are rare and the neighbours are unreliable. This is **base-rate reasoning**, and humans are notoriously bad at it.

## Variable elimination

Enumeration recomputes the same products many times. **Variable elimination** stores intermediate results as *factors* and sums out variables one at a time, like dynamic programming. Its cost depends on the **elimination order**; finding the best order is NP-hard, but good heuristics exist. For **polytrees** (singly connected networks) inference is linear in network size; in general, exact inference is #P-hard.

## Approximate inference by sampling

For large networks we sample:

- **Prior (direct) sampling** — sample each variable in topological order from its CPT.
- **Rejection sampling** — discard samples inconsistent with the evidence (wasteful when evidence is unlikely).
- **Likelihood weighting** — fix evidence variables and weight each sample by the likelihood of the evidence.
- **Gibbs sampling (MCMC)** — repeatedly resample one non-evidence variable given its **Markov blanket** (parents, children and children's parents).

## Learning Bayesian networks

Parameters (CPTs) can be learned from data by counting (maximum likelihood) or with Bayesian priors (e.g. adding pseudo-counts). Learning the **structure** is harder — a search over DAGs guided by a score such as BIC. Structure learning connects to the field of **causal inference**, where arrows are interpreted as causes and we ask what happens under *interventions* — another of Pearl's major contributions.

:::exercise
1. Compute $P(A)$ — the prior probability that the alarm rings — from the CPTs.
2. Using the code, compute $P(B \mid j, \neg m)$. Why is it lower?
3. Draw a network for "Rain → WetGrass ← Sprinkler" and explain explaining-away in words.
:::

:::takeaway
- A Bayesian network factorises a joint distribution along a DAG: $\prod_i P(X_i \mid \text{Parents}(X_i))$.
- d-separation reads independences from the graph; colliders cause explaining away.
- Exact inference: enumeration and variable elimination (hard in general, easy on polytrees).
- Approximate inference: likelihood weighting and Gibbs sampling.
:::

=== POST ===
slug: hidden-markov-models
title: Hidden Markov Models: Filtering, Smoothing and the Viterbi Algorithm
category: foundations
level: Intermediate
tags: hmm, probability, viterbi, forward algorithm, time series
summary: When the world changes over time and we only see noisy observations, Hidden Markov Models let us infer what is really happening. We derive the forward algorithm, Viterbi decoding and Baum–Welch learning.
---
A security guard works underground and never sees the sky. Each day the director comes in either with or without an umbrella. From this sequence of umbrella observations, can the guard infer whether it is raining? This is the classic setting of a **Hidden Markov Model (HMM)**: a hidden state evolves over time and emits noisy observations. HMMs powered speech recognition for three decades and remain important in bioinformatics and signal processing.

## The model

An HMM has:

- hidden states $X_t \in \{1, \dots, S\}$ (e.g. Rain / NoRain);
- observations $E_t$ (e.g. Umbrella / NoUmbrella);
- an **initial distribution** $\pi_i = P(X_1 = i)$;
- a **transition model** $T_{ij} = P(X_{t+1} = j \mid X_t = i)$;
- a **sensor (emission) model** $O_{j}(e) = P(E_t = e \mid X_t = j)$.

Two assumptions make it tractable:

1. **Markov assumption**: $P(X_{t+1} \mid X_{1:t}) = P(X_{t+1} \mid X_t)$.
2. **Sensor Markov assumption**: $P(E_t \mid X_{1:t}, E_{1:t-1}) = P(E_t \mid X_t)$.

## The four inference tasks

| Task | Question | Algorithm |
|---|---|---|
| Filtering | $P(X_t \mid e_{1:t})$ — where am I now? | Forward algorithm |
| Prediction | $P(X_{t+k} \mid e_{1:t})$ — where will I be? | Forward + transitions |
| Smoothing | $P(X_k \mid e_{1:t})$, $k < t$ — where was I? | Forward–backward |
| Most likely explanation | $\arg\max_{x_{1:t}} P(x_{1:t} \mid e_{1:t})$ | Viterbi |

## Filtering: the forward algorithm

Filtering is a recursive update: predict with the transition model, then correct with the new observation.

$$
f_{t+1}(j) = \alpha \; O_j(e_{t+1}) \sum_{i} f_t(i)\, T_{ij}
$$

where $\alpha$ normalises the vector to sum to one. Each step costs $O(S^2)$, independent of $t$ — the agent can run forever with constant memory. This predict–update pattern is exactly the structure of the **Kalman filter** (for continuous linear-Gaussian models) and the **particle filter** (for general models) used in robotics.

## Smoothing: forward–backward

To estimate a past state using *all* evidence, combine a forward message with a backward message:

$$
b_k(i) = \sum_j T_{ij}\, O_j(e_{k+1})\, b_{k+1}(j), \qquad P(X_k = i \mid e_{1:t}) \propto f_k(i)\, b_k(i)
$$

## Decoding: the Viterbi algorithm

Often we want the single most likely *sequence* of hidden states — e.g. the most likely sequence of words given audio. Viterbi is dynamic programming over the trellis of states, replacing the sum in the forward algorithm with a max:

$$
m_{t+1}(j) = O_j(e_{t+1}) \max_i \big[ m_t(i)\, T_{ij} \big]
$$

and keeping back-pointers to recover the path.

```python
import numpy as np

states = ["Rain", "NoRain"]
pi = np.array([0.5, 0.5])
T = np.array([[0.7, 0.3],        # from Rain
              [0.3, 0.7]])       # from NoRain
O = {"U":  np.array([0.9, 0.2]), # P(umbrella | state)
     "~U": np.array([0.1, 0.8])}

def forward(obs):
    f = pi * O[obs[0]]; f /= f.sum()
    out = [f]
    for e in obs[1:]:
        f = O[e] * (f @ T); f /= f.sum()
        out.append(f)
    return out

def viterbi(obs):
    logT = np.log(T)
    m = np.log(pi) + np.log(O[obs[0]])
    back = []
    for e in obs[1:]:
        scores = m[:, None] + logT            # scores[i, j]
        back.append(scores.argmax(axis=0))
        m = scores.max(axis=0) + np.log(O[e])
    path = [int(m.argmax())]
    for bp in reversed(back):
        path.append(int(bp[path[-1]]))
    return [states[s] for s in reversed(path)]

obs = ["U", "U", "~U", "U", "U"]
print([f.round(3) for f in forward(obs)])
print(viterbi(obs))
```

:::tip
Always work in **log space** for Viterbi and normalise at each step for the forward algorithm. Multiplying hundreds of probabilities smaller than one underflows to zero in floating point — a bug that silently ruins results.
:::

## Learning HMM parameters: Baum–Welch

If we only have observation sequences, we can learn $\pi$, $T$ and $O$ with the **Baum–Welch algorithm**, a special case of **Expectation–Maximisation (EM)**:

- **E-step**: use forward–backward to compute expected counts of each transition and emission.
- **M-step**: re-estimate parameters from those expected counts.

EM increases the likelihood at every iteration but may converge to a local optimum, so multiple random restarts are common.

## Applications and limitations

HMMs were the backbone of speech recognition (hidden phonemes, observed acoustic features) until deep learning took over around 2012. They remain widely used for **gene finding** and **protein family modelling** in bioinformatics, **part-of-speech tagging**, activity recognition from wearable sensors, and finance regime detection.

Their limitation is the **first-order Markov assumption**: the next state depends only on the current one, and each observation depends only on the current state. Recurrent neural networks and transformers relax these assumptions and can capture long-range dependencies — at the cost of interpretability and data hunger.

:::exercise
1. Run the forward algorithm by hand for two days of umbrella observations.
2. Modify the code to implement smoothing with the forward–backward algorithm.
3. Explain why the most likely *sequence* (Viterbi) may differ from the sequence of individually most likely states (smoothing).
:::

:::takeaway
- An HMM has hidden Markov states that emit observations.
- The forward algorithm filters in $O(S^2)$ per step; forward–backward smooths.
- Viterbi finds the single most likely state sequence by dynamic programming in log space.
- Baum–Welch (EM) learns parameters from unlabelled sequences.
:::

=== POST ===
slug: classical-planning-strips
title: Classical Planning: STRIPS, PDDL and Planning Graphs
category: foundations
level: Intermediate
tags: planning, strips, pddl, graphplan, heuristics
summary: Planning is search with structured, factored states. We represent actions with preconditions and effects, compare forward and backward planning, and see how domain-independent heuristics are derived automatically.
---
Search algorithms treat states as black boxes. But in many problems the state has an obvious internal structure: *the robot is in room A, the box is on the table, the door is closed*. **Classical planning** exploits this structure to solve problems far larger than blind search could handle, using **domain-independent** heuristics that are computed automatically from the problem description.

## Representing planning problems

The **STRIPS** representation (Stanford Research Institute Problem Solver, 1971), used to control the robot Shakey, describes:

- **States** as conjunctions of ground, positive facts (the closed-world assumption: anything not mentioned is false).
- **Goals** as conjunctions of facts that must hold.
- **Actions** as schemas with **preconditions**, an **add list** and a **delete list**.

Modern planners use **PDDL** (Planning Domain Definition Language), a standardised descendant used in the International Planning Competition:

```lisp
(:action move
  :parameters (?r - robot ?from ?to - room)
  :precondition (and (at ?r ?from) (connected ?from ?to))
  :effect (and (at ?r ?to) (not (at ?r ?from))))

(:action pick-up
  :parameters (?r - robot ?b - box ?loc - room)
  :precondition (and (at ?r ?loc) (at ?b ?loc) (hand-empty ?r))
  :effect (and (holding ?r ?b) (not (at ?b ?loc)) (not (hand-empty ?r))))
```

Applying an action to a state $s$ gives $s' = (s \setminus \text{Del}(a)) \cup \text{Add}(a)$.

## Forward (progression) planning

Start at the initial state and apply applicable actions — ordinary search over states. Naively this looks hopeless because the number of ground actions is huge. It became the dominant approach only once good heuristics were discovered.

## Backward (regression) planning

Start from the goal and work backwards, considering only **relevant** actions — actions that achieve some goal fact and delete none. Regressing goal $g$ through action $a$ gives

$$
g' = (g \setminus \text{Add}(a)) \cup \text{Pre}(a)
$$

Regression has a lower branching factor, but works with *sets* of states, which makes heuristics harder to design.

## Domain-independent heuristics

The magic of planning is that heuristics can be derived automatically from the action descriptions by relaxation.

- **Ignore-preconditions heuristic**: every action is applicable everywhere — the number of steps becomes roughly the number of unsatisfied goals (a set-cover problem).
- **Ignore-delete-lists heuristic**: actions never undo progress. The resulting relaxed problem is monotone and can be solved (approximately) in polynomial time. The famous **FF planner** uses the length of a relaxed plan as its heuristic, $h_{FF}$.
- **Landmarks**: facts that *must* be true at some point in every valid plan. Counting unachieved landmarks gives powerful heuristics (used by the LAMA planner).

:::note
This is the same relaxation principle we met in A* for the 8-puzzle — but now applied *automatically* to any domain written in PDDL. The planner designer never needs to know anything about warehouses, satellites or logistics.
:::

## Planning graphs and GraphPlan

A **planning graph** is a layered structure alternating fact levels and action levels. Level $S_0$ holds the initial facts, $A_0$ all applicable actions, $S_1$ all facts they might produce, and so on. **Mutex** (mutual exclusion) links record pairs of actions or facts that cannot co-occur.

- The level at which all goal facts first appear, non-mutex, is an admissible estimate of plan length.
- **GraphPlan** (1995) extends the graph until goals appear and then searches backwards for a valid plan.

## Other approaches

- **Planning as satisfiability (SATPlan)**: encode "is there a plan of length $T$?" as a SAT formula and hand it to a SAT solver, increasing $T$ as needed.
- **Partial-order planning**: build plans as partially ordered sets of actions, committing to orderings only when necessary.
- **Hierarchical Task Network (HTN) planning**: decompose high-level tasks ("build house") into subtasks using methods written by experts — widely used in games and industrial systems.

## Beyond classical assumptions

Classical planning assumes a fully observable, deterministic, static world with instantaneous actions. Real applications relax these:

- **Temporal planning** handles durations and concurrency.
- **Probabilistic planning** uses Markov Decision Processes (see the reinforcement-learning track).
- **Contingent and conformant planning** handle partial observability.
- **Replanning / execution monitoring**: when the world deviates from the plan, detect it and plan again.

## A tiny forward planner

```python
from collections import deque

def plan(init, goal, actions):
    """actions: list of (name, pre, add, delete) with frozensets of facts."""
    start = frozenset(init)
    frontier, parent = deque([start]), {start: None}
    while frontier:
        s = frontier.popleft()
        if goal <= s:
            steps = []
            while parent[s]:
                s, name = parent[s]
                steps.append(name)
            return steps[::-1]
        for name, pre, add, delete in actions:
            if pre <= s:
                t = (s - delete) | add
                if t not in parent:
                    parent[t] = (s, name)
                    frontier.append(t)
    return None

A = [("move A->B", {"at A"}, {"at B"}, {"at A"}),
     ("move B->A", {"at B"}, {"at A"}, {"at B"}),
     ("pick box", {"at B", "box B", "empty"}, {"holding"}, {"box B", "empty"}),
     ("drop box", {"at A", "holding"}, {"box A", "empty"}, {"holding"})]
A = [(n, frozenset(p), frozenset(a), frozenset(d)) for n, p, a, d in A]
print(plan({"at A", "box B", "empty"}, frozenset({"box A"}), A))
```

:::exercise
1. Write a PDDL domain for the blocks world with `stack`, `unstack`, `pickup` and `putdown`.
2. Compute the ignore-delete-lists heuristic by hand for a simple blocks-world state.
3. Replace BFS in the code above with A* using the number of unsatisfied goal facts as a heuristic. Is it admissible?
:::

:::takeaway
- Planning represents states as sets of facts and actions by preconditions, add and delete lists (STRIPS/PDDL).
- Forward search with automatically derived heuristics (e.g. $h_{FF}$, landmarks) dominates modern planning.
- Planning graphs, SAT encodings and HTNs are important alternative techniques.
- Real-world planning must handle time, uncertainty and replanning.
:::

=== POST ===
slug: symbolic-vs-connectionist-ai
title: Symbolic vs Connectionist AI — and the Neuro-Symbolic Synthesis
category: foundations
level: Beginner
tags: symbolic ai, neural networks, neuro-symbolic, philosophy
summary: For decades AI was split between those who manipulate symbols and those who train networks. We compare the two paradigms honestly and examine how modern research tries to combine their strengths.
---
Every field has its great debate. In AI, it is between **symbolic** approaches — intelligence as manipulation of explicit symbols with rules — and **connectionist** approaches — intelligence as the emergent behaviour of large networks of simple units trained on data. Understanding this debate will help you read research papers critically and choose the right tool for a problem.

## The symbolic tradition

Symbolic AI, sometimes called **GOFAI** ("Good Old-Fashioned AI"), rests on Newell and Simon's **Physical Symbol System Hypothesis** (1976): *a physical symbol system has the necessary and sufficient means for general intelligent action.* Knowledge is represented explicitly — logic, rules, frames, plans — and intelligence is search and inference over those representations.

**Strengths:**
- **Interpretability** — you can read the rules and trace the reasoning.
- **Precision and guarantees** — a theorem prover's proof is correct.
- **Data efficiency** — one rule can cover infinitely many cases.
- **Compositionality** — complex concepts are built from simple ones systematically.

**Weaknesses:**
- **Brittleness** — failure outside anticipated cases.
- **The knowledge acquisition bottleneck** — rules must be written by hand.
- **The symbol grounding problem** — how do symbols like `Cat` connect to raw pixels?
- **Poor handling of noise and perception.**

## The connectionist tradition

Connectionism models cognition as parallel computation in networks of neuron-like units, with knowledge stored in **distributed** connection weights and acquired by **learning**. Its lineage runs from McCulloch–Pitts neurons and Rosenblatt's perceptron through the 1986 *Parallel Distributed Processing* volumes to modern deep learning.

**Strengths:**
- **Learning from raw data** — images, audio, text.
- **Robustness to noise** and graceful degradation.
- **Generalisation by similarity** — nearby inputs produce nearby outputs.
- **Scalability** — performance improves with data and compute.

**Weaknesses:**
- **Opacity** — billions of weights are hard to interpret.
- **Data hunger** — many examples are needed.
- **Unreliable systematic generalisation** — models may fail on novel combinations of familiar parts or long chains of reasoning.
- **No guarantees** — a network can be confidently wrong.

## A side-by-side comparison

| Aspect | Symbolic | Connectionist |
|---|---|---|
| Knowledge | Explicit rules and facts | Distributed weights |
| Acquisition | Hand-engineered | Learned from data |
| Reasoning | Search, logical inference | Pattern completion, function approximation |
| Perception | Weak | Strong |
| Explanation | Natural | Difficult |
| Typical failure | Brittle at edges | Confident errors, spurious correlations |

## The famous arguments

In 1988 **Fodor and Pylyshyn** argued that human thought is **systematic**: anyone who can think "John loves Mary" can think "Mary loves John". They claimed connectionist networks could not explain this unless they implemented a symbol system. Connectionists responded that networks can *learn* systematic behaviour from data. The debate is very much alive: modern studies of whether large language models generalise compositionally are its direct descendants.

**Gary Marcus** has argued for decades that deep learning needs symbolic components for abstraction and reasoning. **Geoffrey Hinton**, **Yann LeCun** and **Yoshua Bengio** have generally argued that neural systems can learn to reason given the right architectures and objectives. Serious scientists disagree — which makes it a fertile area for your own research.

## Neuro-symbolic AI

Rather than choosing sides, many researchers now combine the paradigms:

1. **Neural perception + symbolic reasoning.** A network detects objects in an image; a symbolic program answers questions about them (e.g. the Neuro-Symbolic Concept Learner).
2. **Differentiable logic.** Logical rules are relaxed into continuous functions so they can be trained with gradient descent (Logic Tensor Networks, ∂ILP).
3. **Neural-guided search.** AlphaGo and AlphaZero combine neural evaluation with symbolic tree search; AlphaGeometry pairs a language model with a symbolic geometry engine to solve olympiad problems.
4. **Language models with tools.** An LLM writes code or calls a calculator, solver or database — delegating exact computation to symbolic systems.
5. **Knowledge graphs + embeddings**, which we met in the knowledge representation lecture.

```python
# A toy illustration: a "neural" scorer proposes, a symbolic checker verifies.
import random

def neural_guess(a, b):
    """Stand-in for a learned model: usually right, sometimes wrong."""
    return a * b + random.choice([0, 0, 0, 1, -1])

def symbolic_verify(a, b, answer):
    return answer == a * b          # exact, trustworthy

def solve(a, b, tries=5):
    for _ in range(tries):
        guess = neural_guess(a, b)
        if symbolic_verify(a, b, guess):
            return guess
    return a * b                    # fall back to exact computation

print(solve(37, 41))
```

This "generate with a network, verify with a symbolic tool" pattern is increasingly common in reasoning systems for mathematics and code.

:::note
When you read a claim that a model "reasons", ask: does it generalise to longer problems than it was trained on? To novel combinations? Does performance collapse under small irrelevant changes to the problem? These questions come directly from the symbolic–connectionist debate, and they are the right questions to ask.
:::

:::exercise
1. Choose one application (medical diagnosis, legal document analysis, or robot navigation) and argue which paradigm — or which hybrid — is most appropriate.
2. Read about the Neuro-Symbolic Concept Learner or AlphaGeometry and summarise how the neural and symbolic parts interact.
3. Design a small experiment to test compositional generalisation in a neural model.
:::

:::takeaway
- Symbolic AI offers interpretability, precision and data efficiency but is brittle and hand-built.
- Connectionist AI offers learning, robustness and perception but is opaque and data-hungry.
- The systematicity debate continues in modern studies of LLM generalisation.
- Neuro-symbolic methods combine neural perception and learning with symbolic reasoning and verification.
:::
