=== POST ===
slug: what-is-artificial-intelligence
title: What Is Artificial Intelligence? A Rigorous Introduction
category: foundations
level: Beginner
tags: definitions, history, agents, overview
summary: We open the course by asking the hardest question first — what do we actually mean by "intelligence" in a machine? Four classical definitions, one working definition, and a map of the field.
---
Good morning, everyone, and welcome to the first lecture. Before we write a single line of code or a single equation, we must agree on what we are studying. This sounds trivial. It is not. Researchers have argued about the definition of Artificial Intelligence (AI) for more than seventy years, and the definition you adopt quietly shapes every design decision you will make later.

## Four ways to define AI

In their classic textbook, Russell and Norvig organise definitions of AI along two axes. The first axis asks whether we care about **thinking** or **acting**. The second asks whether we measure success against **human performance** or against an ideal standard of **rationality**. Crossing the two gives four schools of thought.

| | Human-like | Rational |
|---|---|---|
| **Thinking** | Cognitive modelling: build systems that think like people | Laws of thought: build systems that reason correctly using logic |
| **Acting** | The Turing Test approach: build systems that behave like people | Rational agents: build systems that act to achieve the best expected outcome |

1. **Thinking humanly.** Here AI is a branch of cognitive science. We want programs whose internal steps mirror human mental processes. To validate such a system you need psychological experiments or brain imaging, not just correct answers.
2. **Acting humanly.** Alan Turing's 1950 proposal: if a machine's conversational behaviour cannot be distinguished from a human's, we should call it intelligent. This view is operational — it ignores what happens inside.
3. **Thinking rationally.** Descended from Aristotle's syllogisms and modern formal logic. If we can write down correct rules of inference, a machine that follows them thinks "correctly". The difficulty is that most real-world knowledge is uncertain and informal.
4. **Acting rationally.** An agent is rational if it chooses the action that maximises its expected performance given what it has perceived. This is the definition most modern researchers adopt, because it is mathematically precise and does not require us to copy human quirks.

:::definition Working definition for this course
**Artificial Intelligence** is the study and construction of *agents* that perceive their environment and take actions that maximise their chance of achieving their goals, where the quality of the actions is measured by a well-defined performance measure.
:::

## Why the rational-agent view wins

Consider a self-driving car. Do we want it to drive *like a human*? Humans text while driving, get tired and misjudge distances. We want it to drive *well* — to minimise accidents, travel time and discomfort. The rational-agent view lets us write that goal down as a performance measure and then engineer towards it.

The rational-agent view also unifies the field. A chess engine, a spam filter, a language model and a warehouse robot all look different, but each maps a history of observations to an action:

$$
f : \mathcal{P}^* \rightarrow \mathcal{A}
$$

where $\mathcal{P}^*$ is the set of all percept sequences and $\mathcal{A}$ is the set of actions. The whole of AI can be viewed as the search for good functions $f$ — and machine learning is the branch that *learns* $f$ from data rather than programming it by hand.

:::note
Perfect rationality is usually impossible: computing the truly optimal action may take longer than the age of the universe. Researchers therefore speak of **bounded rationality** — doing the best you can with limited time and memory. Keep this idea in mind; it will reappear in search, planning and reinforcement learning.
:::

## The main subfields

Artificial Intelligence is an umbrella. Beneath it you will meet the following areas in this course:

- **Search and planning** — finding sequences of actions that reach a goal.
- **Knowledge representation and reasoning** — encoding facts and drawing conclusions with logic.
- **Reasoning under uncertainty** — probability, Bayesian networks and decision theory.
- **Machine Learning (ML)** — improving performance at a task from experience (data).
- **Deep Learning (DL)** — machine learning with many-layered neural networks.
- **Natural Language Processing (NLP)** — understanding and generating human language.
- **Computer Vision** — understanding images and video.
- **Robotics** — perception and action in the physical world.
- **Reinforcement Learning** — learning to act from rewards.

A useful mental picture is a set of nested circles: Deep Learning sits inside Machine Learning, which sits inside Artificial Intelligence. Not all AI is learning (a classical chess engine uses search, not learning), and not all machine learning is deep (a decision tree is not a neural network).

## A tiny rational agent in code

Let us make this concrete. Below is a reflex agent for a two-square vacuum world. It perceives its location and whether the square is dirty, and chooses an action.

```python
def reflex_vacuum_agent(percept):
    """percept = (location, status) where location in {'A','B'}."""
    location, status = percept
    if status == "Dirty":
        return "Suck"
    return "Right" if location == "A" else "Left"

# Simulate a few steps
world = {"A": "Dirty", "B": "Dirty"}
loc = "A"
for step in range(4):
    action = reflex_vacuum_agent((loc, world[loc]))
    print(step, loc, world[loc], "->", action)
    if action == "Suck":
        world[loc] = "Clean"
    else:
        loc = "B" if action == "Right" else "A"
```

This agent is trivially simple, yet it fits our definition: it perceives, it acts, and its performance can be measured (for example, one point per clean square per time step). Everything else in this course — neural networks, transformers, AlphaZero — is a vastly more sophisticated answer to the same question: *given what I have seen, what should I do?*

## Strong AI, weak AI and hype

You will hear the phrases **weak (narrow) AI** — systems that perform a specific task — and **strong AI** or **Artificial General Intelligence (AGI)** — systems with flexible, human-level competence across tasks. Every deployed system today, including very large language models, is best understood through careful measurement of what it can and cannot do rather than through labels. As scientists, we measure; we do not merely marvel.

:::exercise
1. Classify each system into one of the four quadrants: (a) a program that proves geometry theorems, (b) a chatbot designed to pass as a human, (c) a thermostat, (d) a model of human memory recall.
2. Write a performance measure for an automated email spam filter. What trade-off must your measure capture?
3. Modify the vacuum agent so that it stops (`NoOp`) when both squares are clean. What extra information does the agent now need to remember?
:::

:::takeaway
- AI can be defined along two axes: thinking vs acting, and human-like vs rational.
- This course adopts the **rational agent** definition: act to maximise expected performance.
- Machine Learning is the subfield where the agent's behaviour is *learned from data*; Deep Learning is ML with deep neural networks.
- Perfect rationality is usually infeasible — real systems are **boundedly rational**.
:::

=== POST ===
slug: history-of-ai
title: A History of AI: From Turing to Transformers
category: foundations
level: Beginner
tags: history, turing, dartmouth, deep learning, ai winter
summary: Seventy years of ambition, disappointment and breakthroughs. Understanding the history of AI teaches you why today's methods look the way they do — and why humility is a scientific virtue.
---
A scientist who does not know the history of their field is condemned to rediscover old ideas and repeat old mistakes. Today we take a guided tour through the history of Artificial Intelligence. Pay attention to a recurring pattern: bold promises, brilliant early results, a collision with reality, and then a new idea that changes the game.

## Prehistory: the dream of mechanical reasoning (before 1950)

The idea that reasoning could be mechanised is ancient. Aristotle formalised syllogisms. In the 17th century Leibniz dreamed of a *calculus ratiocinator* that would settle arguments by calculation. George Boole (1854) showed that logic could be written as algebra, and Gottlob Frege created predicate logic. In 1936 Alan Turing defined the Turing machine, giving a precise meaning to "computation" itself.

Two more ingredients arrived in the 1940s. **Warren McCulloch and Walter Pitts (1943)** proposed a mathematical model of a neuron as a threshold logic unit and showed that networks of such units could compute logical functions. **Norbert Wiener** founded cybernetics, the study of feedback and control. Both ideas — logic and neurons — would compete and cooperate for the next eighty years.

## Birth of the field (1950–1956)

In 1950 Turing published *Computing Machinery and Intelligence*, asking "Can machines think?" and proposing the imitation game. In 1956 John McCarthy, Marvin Minsky, Claude Shannon and Nathaniel Rochester organised the **Dartmouth Summer Research Project**, where the phrase "Artificial Intelligence" was adopted. Their proposal famously conjectured that "every aspect of learning or any other feature of intelligence can in principle be so precisely described that a machine can be made to simulate it."

## The golden years (1956–1974)

Early successes were dazzling:

- **Logic Theorist** (Newell and Simon, 1956) proved theorems from *Principia Mathematica*.
- **General Problem Solver** tried to capture human-like means–ends reasoning.
- **Arthur Samuel's checkers program** (1959) learned to beat its creator — and Samuel coined the term *machine learning*.
- **Frank Rosenblatt's Perceptron** (1958) learned to classify simple patterns in hardware.
- **ELIZA** (Weizenbaum, 1966) simulated a psychotherapist with simple pattern matching and surprised people with how human it felt.
- **Shakey the robot** combined perception, planning and action.

Researchers predicted human-level AI within a generation.

## The first AI winter (1974–1980)

Reality intervened. Programs that worked on toy problems failed to scale due to **combinatorial explosion**. Machine translation performed poorly. In 1969 Minsky and Papert's book *Perceptrons* proved that a single-layer perceptron cannot represent functions such as XOR, which dampened neural network research. The 1973 **Lighthill Report** in the UK criticised the field, and funding was cut on both sides of the Atlantic.

## Expert systems boom and bust (1980–1993)

AI revived commercially through **expert systems** — programs encoding the if–then rules of human specialists. MYCIN diagnosed blood infections; XCON configured computer orders and saved its company millions. Japan launched the Fifth Generation Computer project. But expert systems were brittle, expensive to maintain and could not learn. When specialised Lisp machine hardware collapsed commercially in the late 1980s, a **second AI winter** followed.

Meanwhile, something important happened quietly: in 1986 **Rumelhart, Hinton and Williams** popularised **backpropagation** for training multi-layer networks, solving the problem Minsky and Papert had highlighted.

## The statistical turn (1990s–2000s)

The field matured by embracing probability and statistics. Judea Pearl's **Bayesian networks** gave a principled way to reason under uncertainty. **Support Vector Machines**, boosting and random forests achieved excellent results with strong theory. **IBM Deep Blue defeated Garry Kasparov in 1997** using massive search, not learning. Yann LeCun's convolutional networks read handwritten cheques. Researchers increasingly measured progress on shared benchmark datasets.

## The deep learning revolution (2006–2017)

Three forces converged: **large datasets** (ImageNet, 2009), **GPU computing**, and **better training techniques** (ReLU activations, dropout, good initialisation). In 2012, **AlexNet** won the ImageNet challenge by a huge margin, and the field changed almost overnight. Milestones followed quickly:

| Year | Milestone |
|---|---|
| 2013 | Word2Vec learns word meanings as vectors; DQN learns Atari games from pixels |
| 2014 | Generative Adversarial Networks; sequence-to-sequence translation |
| 2015 | ResNet trains networks with 150+ layers |
| 2016 | AlphaGo defeats Lee Sedol at Go |
| 2017 | "Attention Is All You Need" introduces the Transformer |

## The era of foundation models (2018–present)

The Transformer enabled models pretrained on enormous corpora: **BERT** (2018) for understanding and the **GPT** series for generation. Scaling laws showed that performance improves predictably with more data, parameters and compute. Diffusion models transformed image generation. AlphaFold 2 (2020) effectively solved protein structure prediction for many proteins. Conversational assistants built on large language models reached hundreds of millions of users. Today the research frontier includes reasoning, multimodality, efficiency, agents that use tools, and — crucially — safety and alignment.

:::note
Notice what the successful paradigms share: they **learn from data** and they **scale with computation**. Richard Sutton called this "the bitter lesson" — general methods that leverage computation eventually beat hand-crafted human knowledge. Whether that lesson holds forever is an open question, and a good research topic.
:::

## Lessons for a young researcher

1. **Beware of over-promising.** Both AI winters followed inflated expectations.
2. **Old ideas return.** Neural networks were declared dead twice. Backpropagation, attention and reinforcement learning all have deep roots.
3. **Benchmarks drive progress — and can mislead.** Always ask what a benchmark does not measure.
4. **Compute and data matter as much as cleverness.**

:::exercise
Pick one milestone from the table above and write a one-page summary: what problem did it solve, what was the key idea, and what limitation did it leave for the next generation?
:::

:::takeaway
- AI began formally at Dartmouth in 1956, built on logic, computation and neuron models.
- Two AI winters followed periods of over-promising.
- The statistical and deep learning turns shifted the field from hand-written rules to learning from data.
- Transformers and scaling produced today's foundation models.
:::

=== POST ===
slug: turing-test-and-its-critics
title: The Turing Test and Its Critics
category: foundations
level: Beginner
tags: turing test, philosophy, chinese room, evaluation
summary: Turing replaced "Can machines think?" with a game. We examine the imitation game, the Chinese Room argument, and what modern AI evaluation has learned from seventy years of debate.
---
In 1950 Alan Turing sidestepped a philosophical swamp. Instead of defining "thinking" — a word with no agreed meaning — he proposed a test that anyone could run. Today we study that test, its strongest objections, and why the debate still matters for how we evaluate modern AI systems.

## The imitation game

Turing's setup has three participants: a human interrogator, a human respondent and a machine. The interrogator communicates with both by text only and must decide which is the machine. If the interrogator cannot reliably tell them apart, Turing argued, we have as much reason to call the machine intelligent as we have for attributing minds to other people — whose inner lives we also only infer from behaviour.

The genius of the test is that it is **behavioural** and **domain-general**. The interrogator may ask about poetry, arithmetic, jokes or personal history. To pass, a machine needs language understanding, knowledge, reasoning and some model of human conversation.

## Turing's own replies to objections

Turing anticipated many objections in his paper. A few remain instructive:

- **The theological objection** — thinking is a function of the soul. Turing replied that this constrains the Creator's power.
- **The mathematical objection** — Gödel's theorem shows that formal systems have limits. Turing noted that humans have limits too.
- **Lady Lovelace's objection** — machines can only do what we tell them. Turing argued that machines can surprise us and, importantly, can *learn*. He even sketched the idea of a "child machine" educated by experience — a remarkably prescient description of machine learning.
- **The argument from consciousness** — a machine may behave intelligently without experiencing anything. Turing pointed out that we cannot verify consciousness in other humans either.

## The Chinese Room

The most famous critique came from philosopher **John Searle (1980)**. Imagine a person who speaks no Chinese locked in a room with a huge rule book. Chinese questions are passed in; by following the rules mechanically, the person produces fluent Chinese answers. To outsiders, the room "understands" Chinese. Yet the person inside understands nothing.

Searle's conclusion: **syntax is not sufficient for semantics**. Running the right program does not by itself produce understanding.

Common replies include:

1. **The systems reply** — the person does not understand, but the whole system (person + rules + room) does.
2. **The robot reply** — ground the symbols in perception and action and understanding may emerge.
3. **The brain simulator reply** — if the program simulated neurons exactly, would Searle still deny understanding?

:::note
You are not required to settle the Chinese Room debate to be an excellent AI engineer. But you *are* required to be precise about claims. "The model understands" is a philosophical claim; "the model answers 87% of held-out questions correctly" is an empirical one. In papers and in industry, prefer the empirical claim.
:::

## Weaknesses of the Turing Test as a benchmark

Even ignoring philosophy, the test has practical problems:

- **It rewards deception.** Early chatbots "passed" short tests by pretending to be a teenager with poor English, deflecting hard questions.
- **It is anthropocentric.** A system could be superhumanly useful yet obviously non-human (it multiplies 20-digit numbers instantly).
- **It depends on the judge.** Naive interrogators are easily fooled; experts are not.
- **It is binary.** Science needs graded, reproducible measurements.

## What replaced it

Modern AI evaluation is a battery of specific benchmarks: reading comprehension, mathematical reasoning, coding tasks, image recognition, commonsense reasoning, and more. Alternatives in the spirit of Turing include the **Winograd Schema Challenge**, which tests commonsense pronoun resolution:

> The trophy doesn't fit in the brown suitcase because *it* is too big. What is too big?

Changing "big" to "small" flips the answer — a test that superficial statistics were once expected to fail. Large language models now handle many such examples, which in turn forced researchers to build harder, adversarial benchmarks. This arms race between models and benchmarks is one of the defining dynamics of modern AI.

## A small experiment

The following toy "ELIZA-style" responder shows how pattern matching can seem conversational without any understanding.

```python
import re, random

RULES = [
    (r"i feel (.*)", ["Why do you feel {0}?", "How long have you felt {0}?"]),
    (r"i am (.*)",   ["Why do you say you are {0}?", "Do you enjoy being {0}?"]),
    (r"(.*) mother(.*)", ["Tell me more about your family."]),
    (r"(.*)",        ["Please go on.", "I see. Can you elaborate?"]),
]

def respond(text):
    text = text.lower().strip(".!?")
    for pattern, answers in RULES:
        m = re.match(pattern, text)
        if m:
            return random.choice(answers).format(*m.groups())

print(respond("I feel tired of exams"))
print(respond("I am a student at AUST"))
```

Run it and chat with it for a few minutes. Notice how quickly it breaks when you ask a factual or multi-step question.

:::exercise
1. Design three interrogator questions you believe would expose the ELIZA program above within one turn. Explain why.
2. Write a short argument (half a page) for or against the systems reply to the Chinese Room.
3. Propose a graded (non-binary) version of the Turing Test. How would you score it?
:::

:::takeaway
- The Turing Test operationalises intelligence as indistinguishable conversational behaviour.
- Searle's Chinese Room argues that symbol manipulation alone does not produce understanding.
- As a benchmark the test is judge-dependent, binary and rewards deception.
- Modern evaluation uses many targeted, reproducible benchmarks — and keeps making them harder.
:::

=== POST ===
slug: intelligent-agents-and-environments
title: Intelligent Agents, Environments and the PEAS Framework
category: foundations
level: Beginner
tags: agents, environments, peas, rationality
summary: The agent is the central abstraction of AI. We define agents formally, characterise environments along six dimensions, and study the architectures from simple reflex agents to learning agents.
---
In the first lecture we defined AI as the construction of rational agents. Today we make that definition precise. By the end you should be able to take any AI problem — a trading bot, a medical assistant, a drone — and describe it in the vocabulary of agents and environments. This vocabulary is the lingua franca of the entire field.

## Agents, percepts and actions

An **agent** is anything that perceives its environment through **sensors** and acts upon it through **actuators**. A human has eyes and hands; a robot has cameras and motors; a software agent receives keystrokes or network packets and outputs text or API calls.

- A **percept** is the agent's input at a single instant.
- The **percept sequence** is the complete history of everything the agent has perceived.
- The **agent function** maps percept sequences to actions.
- The **agent program** is the concrete implementation running on physical hardware.

The distinction between function and program matters: the function is a mathematical object (possibly infinite), while the program must be finite and run in bounded time.

## Rationality, formally

A rational agent selects, for every possible percept sequence, the action expected to maximise its **performance measure**, given the evidence of the percept sequence and whatever built-in knowledge it has.

$$
a^* = \arg\max_{a \in \mathcal{A}} \; \mathbb{E}\left[ U \mid a, \; p_{1:t}, \; K \right]
$$

Here $U$ is the performance (utility), $p_{1:t}$ is the percept history and $K$ is prior knowledge. Note three subtleties:

1. **Rational is not omniscient.** An agent that crosses the street after looking carefully and is hit by a falling satellite was still rational.
2. **Rationality includes information gathering.** Looking before crossing is a rational action because it improves future decisions.
3. **Rationality requires autonomy.** An agent relying only on its designer's prior knowledge, never learning, is fragile.

:::warning
Designing the performance measure is harder than it looks. Reward a cleaning robot for "amount of dirt sucked up" and a clever agent may learn to dump dirt back on the floor and suck it up again. **Measure what you actually want in the environment, not how you think the agent should behave.** This is an early glimpse of the alignment problem.
:::

## The PEAS description

To specify a task environment, list its **P**erformance measure, **E**nvironment, **A**ctuators and **S**ensors.

| Agent | Performance | Environment | Actuators | Sensors |
|---|---|---|---|---|
| Automated taxi | Safety, speed, legality, comfort, profit | Roads, traffic, pedestrians, weather | Steering, accelerator, brake, horn, display | Cameras, LiDAR, GPS, speedometer |
| Medical diagnosis assistant | Patient health, cost, lawsuits avoided | Patient, hospital staff | Questions, test orders, diagnoses | Symptoms, test results, patient answers |
| Spam filter | Accuracy, low false positives | Email stream, users | Label as spam / not spam | Email text, headers, metadata |
| Refugee registration chatbot | Correct answers, accessibility, privacy | Users in many languages, case database | Text replies, referrals | Typed or spoken messages |

## Properties of environments

Environments differ along several dimensions, and each dimension determines which algorithms are appropriate.

- **Fully vs partially observable.** Can the sensors see the complete relevant state? Chess: fully. Poker: partially.
- **Single vs multi-agent.** Are other agents optimising their own goals? Multi-agent environments may be competitive or cooperative.
- **Deterministic vs stochastic.** Is the next state completely determined by the current state and action?
- **Episodic vs sequential.** Is each decision independent (classifying images) or do actions affect the future (driving)?
- **Static vs dynamic.** Does the world change while the agent deliberates?
- **Discrete vs continuous.** Are states, time and actions countable?
- **Known vs unknown.** Does the agent know the rules (the transition model)?

The hardest case — partially observable, multi-agent, stochastic, sequential, dynamic, continuous and unknown — describes much of real life, including driving a taxi.

## Agent architectures

We now study four agent designs of increasing sophistication.

### 1. Simple reflex agents
Act only on the current percept using condition–action rules. Fast and simple, but they fail when the environment is partially observable.

### 2. Model-based reflex agents
Maintain an **internal state** that tracks aspects of the world the agent cannot currently see, updated using a model of how the world evolves and how actions affect it.

### 3. Goal-based agents
Know what states are desirable and use **search** or **planning** to find action sequences that reach them. More flexible: change the goal and behaviour changes without rewriting rules.

### 4. Utility-based agents
Replace a binary goal with a **utility function** that scores how desirable each state is, allowing trade-offs (fast vs safe) and decisions under uncertainty via expected utility.

### Learning agents
Any of these can be made into a **learning agent** with four components: a *performance element* that chooses actions, a *critic* that evaluates outcomes against a standard, a *learning element* that improves the performance element, and a *problem generator* that suggests exploratory actions. Reinforcement learning, which we study later, is a precise mathematical realisation of this picture.

```python
class ModelBasedAgent:
    def __init__(self, rules, update_state):
        self.state = {}
        self.last_action = None
        self.rules = rules              # list of (condition_fn, action)
        self.update_state = update_state

    def __call__(self, percept):
        self.state = self.update_state(self.state, self.last_action, percept)
        for condition, action in self.rules:
            if condition(self.state):
                self.last_action = action
                return action
        self.last_action = "NoOp"
        return "NoOp"
```

:::exercise
1. Write a PEAS description for (a) an online English tutor, (b) a crop-disease detector on a farmer's phone.
2. Classify both environments along all seven dimensions.
3. Explain why a simple reflex agent could get stuck in an infinite loop in a partially observable environment, and how randomisation could help.
:::

:::takeaway
- An agent maps percept sequences to actions; rational agents maximise expected performance.
- **PEAS** specifies a task environment: Performance, Environment, Actuators, Sensors.
- Environment properties (observability, determinism, dynamics, etc.) dictate algorithm choice.
- Architectures progress from reflex → model-based → goal-based → utility-based → learning agents.
:::

=== POST ===
slug: uninformed-search-strategies
title: Uninformed Search: BFS, DFS, Uniform-Cost and Iterative Deepening
category: foundations
level: Beginner
tags: search, bfs, dfs, uniform cost, algorithms
summary: Many AI problems reduce to finding a path in a huge graph. We formalise search problems and analyse the classic blind strategies for completeness, optimality, time and space.
---
Suppose you are in Dhaka and want to reach Chattogram using a road map, but you have no idea which direction Chattogram lies. You can only look at which cities are connected. How do you systematically find a route? This is **uninformed** (or blind) search, and it is the foundation on which all smarter search algorithms are built.

## Formulating a search problem

A search problem consists of:

1. A set of **states** $S$ and an **initial state** $s_0$.
2. A set of **actions** $A(s)$ available in each state.
3. A **transition model** $\text{Result}(s, a)$ giving the next state.
4. A **goal test** $\text{IsGoal}(s)$.
5. An **action cost** function $c(s, a, s')$.

A **solution** is a sequence of actions from $s_0$ to a goal; an **optimal solution** has the lowest total cost. The states and transitions implicitly define a graph — often astronomically large (the 15-puzzle has about $10^{13}$ reachable states), so we generate it lazily.

## The general algorithm

All search algorithms maintain a **frontier** of nodes waiting to be expanded and (in graph search) a **reached** set to avoid repeated states. They differ only in *which node to expand next*.

```python
from collections import deque
import heapq

def breadth_first_search(start, goal, neighbors):
    frontier = deque([start])
    parent = {start: None}
    while frontier:
        s = frontier.popleft()
        if s == goal:
            return reconstruct(parent, s)
        for t in neighbors(s):
            if t not in parent:          # early goal test variant is also common
                parent[t] = s
                frontier.append(t)
    return None

def uniform_cost_search(start, goal, neighbors_with_cost):
    frontier = [(0, start)]
    best = {start: 0}
    parent = {start: None}
    while frontier:
        g, s = heapq.heappop(frontier)
        if s == goal:
            return g, reconstruct(parent, s)
        if g > best[s]:
            continue                      # stale queue entry
        for t, c in neighbors_with_cost(s):
            if g + c < best.get(t, float("inf")):
                best[t] = g + c
                parent[t] = s
                heapq.heappush(frontier, (g + c, t))
    return None

def reconstruct(parent, s):
    path = []
    while s is not None:
        path.append(s)
        s = parent[s]
    return path[::-1]
```

## Measuring search algorithms

We judge a strategy on four criteria, using $b$ = branching factor, $d$ = depth of the shallowest goal, $m$ = maximum depth, $C^*$ = optimal cost, and $\epsilon$ = the minimum action cost.

| Strategy | Complete? | Optimal? | Time | Space |
|---|---|---|---|---|
| Breadth-first (BFS) | Yes (finite $b$) | Yes if costs equal | $O(b^d)$ | $O(b^d)$ |
| Uniform-cost (UCS) | Yes if $\epsilon > 0$ | Yes | $O(b^{1+\lfloor C^*/\epsilon \rfloor})$ | same |
| Depth-first (DFS) | No (infinite paths) | No | $O(b^m)$ | $O(bm)$ |
| Depth-limited | No | No | $O(b^\ell)$ | $O(b\ell)$ |
| Iterative deepening (IDS) | Yes | Yes if costs equal | $O(b^d)$ | $O(bd)$ |
| Bidirectional | Yes | Yes (with care) | $O(b^{d/2})$ | $O(b^{d/2})$ |

### Breadth-first search
Expands the shallowest node first using a FIFO queue. It finds the shortest path in number of steps. Its weakness is **memory**: with $b = 10$ and $d = 10$, BFS stores roughly $10^{10}$ nodes — far beyond typical RAM.

### Uniform-cost search
Expands the node with the smallest path cost $g(n)$ using a priority queue. It is exactly **Dijkstra's algorithm**, adapted to stop at the goal. It is optimal for any non-negative costs.

### Depth-first search
Expands the deepest node first (a LIFO stack or recursion). Its memory is only linear in depth, which is its great virtue, but it can wander down infinite branches and returns whatever solution it finds first.

### Iterative deepening search
Runs depth-limited DFS with limits $0, 1, 2, \dots$ until a goal is found. It seems wasteful to regenerate the upper levels repeatedly, but the total cost is dominated by the last level:

$$
N_{IDS} = (d)b + (d-1)b^2 + \dots + (1)b^d = O(b^d)
$$

For $b = 10, d = 5$, IDS generates 123,450 nodes versus 111,110 for BFS — only about 11% more — while using linear memory.

:::note
Iterative deepening is the preferred uninformed method when the search space is large and the solution depth is unknown. Remember this pattern — "repeat a cheap bounded search with increasing bounds" — because it reappears in IDA*, in game-tree search, and in anytime algorithms.
:::

### Bidirectional search
Searches forward from the start and backward from the goal simultaneously, stopping when the frontiers meet. Since $2b^{d/2} \ll b^d$, the savings are enormous, but you must be able to compute predecessors and the goal must be explicit.

## A worked example

Consider the graph A–B (1), A–C (5), B–C (1), C–D (1). From A to D:

- **BFS** finds A→C→D (2 steps, cost 6).
- **UCS** finds A→B→C→D (3 steps, cost 3) — the cheaper route.

This illustrates the key lesson: BFS optimises the *number of steps*; UCS optimises *total cost*.

:::exercise
1. Implement iterative deepening DFS and test it on the 8-puzzle from a start state three moves from the goal.
2. Show that DFS is not optimal by constructing a small graph where it returns a longer path.
3. Why does UCS need $\epsilon > 0$ for completeness? Construct a counterexample with zero-cost actions.
:::

:::takeaway
- A search problem is defined by states, actions, transitions, goal test and costs.
- BFS is optimal for unit costs but memory-hungry; UCS (Dijkstra) is optimal for general costs.
- DFS uses little memory but is neither complete nor optimal.
- Iterative deepening combines BFS's guarantees with DFS's memory footprint.
:::

=== POST ===
slug: informed-search-a-star
title: Informed Search: Greedy Best-First and A* with Admissible Heuristics
category: foundations
level: Intermediate
tags: search, a-star, heuristics, admissibility, consistency
summary: Knowledge about where the goal lies transforms search. We derive A*, prove its optimality with admissible heuristics, and learn how to invent good heuristics by relaxing problems.
---
Blind search is honest but slow. A human planning a route from Dhaka to Chattogram does not consider roads heading towards Rajshahi, because they *know roughly where Chattogram is*. Today we give our search algorithms that kind of knowledge through a **heuristic function**, and we study the most celebrated algorithm in classical AI: **A\***.

## Heuristic functions

A heuristic $h(n)$ estimates the cost of the cheapest path from node $n$ to a goal. For route finding, the straight-line distance is a natural choice. For the 8-puzzle, two classic heuristics are:

- $h_1$ = number of misplaced tiles;
- $h_2$ = sum of **Manhattan distances** of each tile from its goal position.

## Greedy best-first search

Greedy search expands the node with smallest $h(n)$ — whatever *looks* closest to the goal. It is often fast but neither optimal nor (in tree search) complete: it can be lured down a path that looks promising but is expensive or a dead end.

## A* search

A* combines the cost already paid with the estimated cost remaining:

$$
f(n) = g(n) + h(n)
$$

where $g(n)$ is the path cost from the start to $n$. A* expands the node with the lowest $f$ — the estimated cost of the cheapest solution *through* $n$.

```python
import heapq, itertools

def a_star(start, is_goal, neighbors, h):
    counter = itertools.count()          # tie-breaker so states need not be comparable
    frontier = [(h(start), next(counter), 0, start)]
    g_best = {start: 0}
    parent = {start: None}
    while frontier:
        f, _, g, s = heapq.heappop(frontier)
        if is_goal(s):
            path = []
            while s is not None:
                path.append(s); s = parent[s]
            return g, path[::-1]
        if g > g_best[s]:
            continue
        for t, cost in neighbors(s):
            g2 = g + cost
            if g2 < g_best.get(t, float("inf")):
                g_best[t] = g2
                parent[t] = s
                heapq.heappush(frontier, (g2 + h(t), next(counter), g2, t))
    return None
```

## Admissibility and consistency

:::definition
A heuristic is **admissible** if it never overestimates: $h(n) \le h^*(n)$ for every $n$, where $h^*$ is the true optimal cost-to-go.

A heuristic is **consistent** (monotone) if for every node $n$ and successor $n'$ reached by action $a$: $h(n) \le c(n, a, n') + h(n')$ — a form of the triangle inequality.
:::

Every consistent heuristic is admissible (with $h(\text{goal}) = 0$), but not vice versa.

### Why A* is optimal

**Theorem.** With an admissible heuristic, A* tree search returns an optimal solution.

*Proof sketch.* Suppose A* selects a suboptimal goal $G_2$ for expansion, with $g(G_2) > C^*$. Let $n$ be a node on an optimal path that is still on the frontier (one always exists). Then

$$
f(n) = g(n) + h(n) \le g(n) + h^*(n) = C^* < g(G_2) = f(G_2)
$$

so $n$ has a strictly smaller $f$ than $G_2$ and would have been expanded first — a contradiction. $\blacksquare$

With a *consistent* heuristic, $f$ values are non-decreasing along any path, which means that the first time A* expands a state it has found the optimal path to it — so graph search with a closed set remains optimal.

:::note
A* is also **optimally efficient**: among all optimal algorithms that extend paths from the start using the same heuristic, none is guaranteed to expand fewer nodes. That is a strong statement — it means improvements must come from *better heuristics*, not better bookkeeping.
:::

## The quality of a heuristic

If $h_2(n) \ge h_1(n)$ for all $n$ and both are admissible, we say $h_2$ **dominates** $h_1$, and A* with $h_2$ never expands more nodes. For the 8-puzzle, Manhattan distance dominates misplaced tiles. Typical results at solution depth 20:

| Heuristic | Nodes generated (approx.) | Effective branching factor |
|---|---|---|
| None (IDS) | too many to run | ~2.8 |
| $h_1$ misplaced tiles | ~ 39,000 | ~1.47 |
| $h_2$ Manhattan | ~ 1,600 | ~1.28 |

## Inventing heuristics by relaxation

Where do good heuristics come from? The most powerful technique is **problem relaxation**: remove constraints from the problem, then solve the easier problem exactly. The cost of an optimal solution to a relaxed problem is an admissible heuristic for the original.

For the 8-puzzle, the rule is "a tile can move from A to B if A is adjacent to B **and** B is blank."

- Drop both conditions → a tile can jump anywhere → $h_1$.
- Drop "B is blank" → tiles slide through each other → $h_2$.

Other techniques include **pattern databases** (store exact costs for sub-problems) and **learned heuristics**, where a neural network predicts cost-to-go — the idea behind modern systems that combine search with learning.

## Variants you should know

- **Weighted A\***: $f = g + w\,h$ with $w > 1$ trades optimality (bounded by factor $w$) for speed.
- **IDA\***: iterative deepening on $f$-cost, using linear memory.
- **SMA\***: uses all available memory and drops the worst nodes when full.

:::exercise
1. Prove that Manhattan distance is admissible for the 8-puzzle.
2. Is $\max(h_1, h_2)$ admissible? Is $h_1 + h_2$? Justify.
3. Implement A* on a 2-D grid with obstacles using Manhattan distance, and count expanded nodes compared with UCS.
:::

:::takeaway
- A* expands nodes by $f(n) = g(n) + h(n)$.
- With an **admissible** heuristic A* is optimal; **consistency** makes graph search optimal and efficient.
- Better (dominating) heuristics expand fewer nodes.
- Design heuristics by **relaxing** the problem's constraints.
:::

=== POST ===
slug: adversarial-search-minimax-alpha-beta
title: Adversarial Search: Minimax and Alpha–Beta Pruning
category: foundations
level: Intermediate
tags: games, minimax, alpha-beta, search, game theory
summary: When an opponent is trying to defeat you, search must account for their choices. We derive minimax, prove alpha–beta pruning correct, and see how real game engines evaluate positions.
---
So far our agents have been alone in the world. Today an adversary enters the room. In chess, draughts or tic-tac-toe, the opponent's moves are chosen to *minimise* your outcome. Planning under such opposition is called **adversarial search**, and it produced some of AI's most famous triumphs — from Deep Blue to AlphaZero.

## Games as search problems

We model a two-player, zero-sum, perfect-information game with:

- $s_0$: the initial position;
- $\text{ToMove}(s)$: whose turn it is (MAX or MIN);
- $\text{Actions}(s)$ and $\text{Result}(s, a)$;
- $\text{IsTerminal}(s)$;
- $\text{Utility}(s, p)$: the final payoff for player $p$ (e.g. +1 win, 0 draw, −1 loss).

"Zero-sum" means one player's gain is the other's loss, so a single number describes the outcome.

## The minimax value

Assume both players play optimally. The **minimax value** of a state is

$$
\text{MM}(s) =
\begin{cases}
\text{Utility}(s) & \text{if terminal} \\
\max_{a} \text{MM}(\text{Result}(s,a)) & \text{if MAX to move} \\
\min_{a} \text{MM}(\text{Result}(s,a)) & \text{if MIN to move}
\end{cases}
$$

MAX chooses the move leading to the highest minimax value. This recursion is a depth-first exploration of the whole game tree, with time $O(b^m)$ and space $O(bm)$. For chess ($b \approx 35$, $m \approx 80$ plies) that is about $10^{123}$ nodes — utterly infeasible. We need two ideas: **pruning** and **evaluation functions**.

## Alpha–beta pruning

Alpha–beta computes exactly the same decision as minimax while ignoring branches that cannot influence it. It carries two bounds:

- $\alpha$: the best value MAX can already guarantee on the current path;
- $\beta$: the best value MIN can already guarantee on the current path.

Whenever $\alpha \ge \beta$, the current node cannot affect the final decision, so we stop exploring it.

```python
import math

def alphabeta(state, depth, alpha, beta, maximizing, game):
    if depth == 0 or game.is_terminal(state):
        return game.evaluate(state)
    if maximizing:
        value = -math.inf
        for move in game.ordered_moves(state):
            value = max(value, alphabeta(game.result(state, move), depth - 1,
                                         alpha, beta, False, game))
            alpha = max(alpha, value)
            if alpha >= beta:
                break           # beta cut-off: MIN will never allow this line
        return value
    else:
        value = math.inf
        for move in game.ordered_moves(state):
            value = min(value, alphabeta(game.result(state, move), depth - 1,
                                         alpha, beta, True, game))
            beta = min(beta, value)
            if alpha >= beta:
                break           # alpha cut-off: MAX already has a better option
        return value
```

### Why pruning is safe — an intuition

Suppose MAX has already found a move worth 5. Exploring a second move, MIN's first reply gives 3. MIN will choose *at most* 3 at that node, which is already worse for MAX than 5. Whatever the remaining replies are, MAX will not choose this move, so we prune them.

### How much does it help?

The effectiveness depends on **move ordering**. With perfect ordering alpha–beta examines about $O(b^{m/2})$ nodes — the effective branching factor drops from $b$ to $\sqrt{b}$, letting you search roughly **twice as deep** in the same time. With random ordering it is about $O(b^{3m/4})$.

:::tip
Practical move-ordering tricks: try captures first, try the best move from the previous iteration first (iterative deepening provides this), and use "killer moves" that caused cut-offs at the same depth elsewhere.
:::

## Imperfect real-time decisions

Since we cannot reach terminal states, we cut off search at a depth limit and apply a **heuristic evaluation function** $\text{Eval}(s)$ estimating the expected utility. Classical chess engines used weighted linear features:

$$
\text{Eval}(s) = w_1 f_1(s) + w_2 f_2(s) + \dots + w_n f_n(s)
$$

where $f_1$ might be material balance (queen = 9, rook = 5, …), $f_2$ mobility, $f_3$ king safety, and so on. Modern engines replace this hand-crafted function with a **neural network** trained on millions of positions — a beautiful meeting of search and learning.

Two pitfalls arise with cut-offs:

- **The horizon effect**: a bad event (losing a queen) can be pushed just beyond the search depth by delaying moves.
- **Non-quiescent positions**: evaluating in the middle of a capture sequence is misleading. **Quiescence search** extends the search on such positions until they are "quiet".

## Beyond two-player deterministic games

- **Stochastic games** (backgammon) use **expectiminimax**, with chance nodes averaging over dice outcomes.
- **Imperfect information** (poker) requires reasoning over belief states and game-theoretic equilibria.
- **Monte Carlo Tree Search**, covered in a later lecture, replaces exhaustive search with random simulations and powers Go programs.

:::exercise
1. Draw a depth-2 game tree with branching factor 3 and leaf values 3, 12, 8, 2, 4, 6, 14, 5, 2. Compute the minimax value and mark which leaves alpha–beta prunes with left-to-right ordering.
2. Implement tic-tac-toe with alpha–beta and count nodes visited with and without pruning.
3. Explain why alpha–beta cannot be directly applied to expectiminimax without bounds on utility.
:::

:::takeaway
- Minimax computes optimal play assuming an optimal opponent.
- Alpha–beta returns the same decision while pruning irrelevant branches; perfect ordering yields $O(b^{m/2})$.
- Real engines cut off search and use evaluation functions — increasingly learned by neural networks.
- Watch for the horizon effect; use quiescence search.
:::

=== POST ===
slug: constraint-satisfaction-problems
title: Constraint Satisfaction Problems: Backtracking, Propagation and Heuristics
category: foundations
level: Intermediate
tags: csp, backtracking, arc consistency, scheduling
summary: Timetabling, map colouring, Sudoku and circuit layout share a structure. CSPs exploit that structure with backtracking, variable-ordering heuristics and constraint propagation such as AC-3.
---
Every semester the university must schedule hundreds of exams into rooms and time slots without clashes. This is not a path-finding problem — we do not care about the *sequence* of steps, only about the final assignment. Problems of this kind are **Constraint Satisfaction Problems (CSPs)**, and they have a factored structure that makes them far more efficient to solve than generic search.

## Definition

A CSP consists of three components:

1. A set of **variables** $X = \{X_1, \dots, X_n\}$.
2. A **domain** $D_i$ of possible values for each variable.
3. A set of **constraints** $C$, each specifying allowed combinations of values for a subset of variables.

A **complete, consistent assignment** is a solution.

:::example Map colouring
Colour the seven divisions of a map with {red, green, blue} so that no two neighbouring regions share a colour. Variables are regions; domains are colours; each constraint says $X_i \ne X_j$ for adjacent regions. The constraint graph has an edge for every neighbouring pair.
:::

Other classic CSPs: **Sudoku** (81 variables, domain 1–9, all-different constraints on rows, columns and boxes), **N-queens**, **job-shop scheduling** and **exam timetabling**.

## Types of constraints

- **Unary**: restrict a single variable ($X_1 \ne \text{red}$).
- **Binary**: relate two variables ($X_1 \ne X_2$).
- **Global / higher-order**: involve many variables (AllDifferent).
- **Soft constraints / preferences**: "Prof. Rahman prefers morning slots" — these make it a **constraint optimisation problem**.

## Backtracking search

The basic algorithm is depth-first search that assigns one variable at a time and backtracks when a constraint is violated.

```python
def backtrack(assignment, variables, domains, consistent):
    if len(assignment) == len(variables):
        return assignment
    var = select_unassigned(variables, assignment, domains)
    for value in order_values(var, domains):
        if consistent(var, value, assignment):
            assignment[var] = value
            result = backtrack(assignment, variables, domains, consistent)
            if result is not None:
                return result
            del assignment[var]
    return None
```

Because assignments are commutative (order does not matter), we only branch on one variable per level. That alone reduces the tree from $n!\,d^n$ to $d^n$ leaves. Still exponential — so we add intelligence.

## Heuristics that make backtracking fast

1. **Minimum Remaining Values (MRV)** — choose the variable with the fewest legal values left. This "fail-first" heuristic prunes dead ends early.
2. **Degree heuristic** — to break ties, choose the variable involved in the most constraints with unassigned variables.
3. **Least Constraining Value (LCV)** — for the chosen variable, try values that rule out the fewest choices for neighbours ("succeed-first").

:::note
Notice the asymmetry: we choose variables to *fail first* but values to *succeed first*. Every variable must eventually be assigned, so discovering failure early saves time; but we only need one value to work, so we try the most promising first.
:::

## Constraint propagation

Instead of waiting to discover a conflict, we can **infer** reductions in domains.

### Forward checking
After assigning $X$, delete from each unassigned neighbour's domain any value inconsistent with $X$. If some domain becomes empty, backtrack immediately.

### Arc consistency and AC-3
A variable $X_i$ is **arc-consistent** with respect to $X_j$ if for every value in $D_i$ there exists some value in $D_j$ satisfying the constraint. The AC-3 algorithm enforces this across the whole network:

```python
from collections import deque

def ac3(domains, neighbors, satisfies):
    queue = deque((xi, xj) for xi in domains for xj in neighbors[xi])
    while queue:
        xi, xj = queue.popleft()
        if revise(domains, xi, xj, satisfies):
            if not domains[xi]:
                return False                # inconsistency detected
            for xk in neighbors[xi]:
                if xk != xj:
                    queue.append((xk, xi))
    return True

def revise(domains, xi, xj, satisfies):
    removed = False
    for x in list(domains[xi]):
        if not any(satisfies(xi, x, xj, y) for y in domains[xj]):
            domains[xi].remove(x)
            removed = True
    return removed
```

AC-3 runs in $O(c\,d^3)$ time for $c$ binary constraints and domain size $d$. On easy Sudoku puzzles, arc consistency alone solves the whole grid without any search! Combining search with propagation at every node is called **MAC** (Maintaining Arc Consistency).

## Local search for CSPs

For very large problems, a different approach often wins: start with a complete (possibly inconsistent) assignment and repeatedly change the value of a conflicted variable to the value that minimises the number of conflicts — the **min-conflicts** heuristic. It solves the million-queens problem in a few dozen steps on average and is widely used in real scheduling systems.

## Exploiting structure

If the constraint graph is a **tree**, the CSP can be solved in $O(n d^2)$ time — linear in the number of variables — by ordering variables topologically and enforcing directional arc consistency. Many real problems can be made tree-like by removing a small **cycle cutset** or by **tree decomposition**. This is a recurring theme in AI: *structure makes hard problems easy*.

:::exercise
1. Formulate your department's exam timetable as a CSP. What are the variables, domains, hard constraints and soft constraints?
2. Run AC-3 by hand on the map-colouring problem after assigning one region red.
3. Implement MRV + forward checking for Sudoku and count backtracks on a "hard" puzzle, compared with plain backtracking.
:::

:::takeaway
- A CSP is variables + domains + constraints; a solution is a complete consistent assignment.
- Backtracking with MRV, degree and LCV heuristics is the workhorse solver.
- Constraint propagation (forward checking, AC-3, MAC) prunes domains before failure occurs.
- Min-conflicts local search and tree-structured decompositions scale to huge problems.
:::
