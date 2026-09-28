=== POST ===
slug: local-search-hill-climbing-simulated-annealing
title: Local Search: Hill Climbing, Simulated Annealing and Beam Search
category: foundations
level: Beginner
tags: search, optimization, hill climbing, simulated annealing
summary: When only the final configuration matters, we can abandon paths and move through the space of complete solutions. Local search is simple, memory-light and the conceptual ancestor of gradient descent.
---
In many problems we do not care about the path to the goal — only the goal itself. The 8-queens puzzle, timetabling, chip layout and hyperparameter tuning all ask for a *configuration* that is good. **Local search** algorithms keep a single current state (or a few), and repeatedly move to neighbouring states. They use constant memory and can find reasonable solutions in enormous or continuous spaces. They are also your first step towards understanding gradient-based learning.

## The state-space landscape

Imagine every state placed on a landscape whose height is its **objective value** (or negative cost). Local search tries to find the **global maximum**. The terrain contains:

- **Local maxima** — peaks higher than their neighbours but lower than the global maximum.
- **Plateaus** — flat regions where no neighbour is better.
- **Ridges** — sequences of local maxima that are hard to navigate with axis-aligned moves.

## Hill climbing

Steepest-ascent hill climbing moves to the best neighbour and stops when no neighbour is better.

```python
import random

def conflicts(board):
    """board[c] = row of the queen in column c. Count attacking pairs."""
    n, total = len(board), 0
    for i in range(n):
        for j in range(i + 1, n):
            if board[i] == board[j] or abs(board[i] - board[j]) == j - i:
                total += 1
    return total

def hill_climb(n=8, max_steps=1000):
    board = [random.randrange(n) for _ in range(n)]
    for _ in range(max_steps):
        current = conflicts(board)
        if current == 0:
            return board
        best, best_move = current, None
        for col in range(n):
            for row in range(n):
                if row != board[col]:
                    old = board[col]; board[col] = row
                    c = conflicts(board)
                    if c < best:
                        best, best_move = c, (col, row)
                    board[col] = old
        if best_move is None:
            return None               # stuck in a local minimum
        board[best_move[0]] = best_move[1]
    return None

print(hill_climb())
```

From a random start, steepest-ascent hill climbing solves 8-queens only about **14%** of the time — it gets stuck in local optima. But it is very fast, averaging about 4 steps when it succeeds.

### Fixes for hill climbing

- **Sideways moves** allow moving across plateaus (with a cap on consecutive moves). This raises the success rate for 8-queens to roughly 94%.
- **Stochastic hill climbing** picks randomly among uphill moves.
- **First-choice hill climbing** generates random neighbours until one is better — good when neighbourhoods are huge.
- **Random-restart hill climbing** repeats from new random states until success. If each attempt succeeds with probability $p$, the expected number of restarts is $1/p$. It is surprisingly effective.

## Simulated annealing

In metallurgy, annealing cools a metal slowly so its atoms settle into a low-energy crystal. **Simulated annealing** mimics this: it picks a random move; if it improves, accept it; if it worsens the objective by $\Delta E < 0$, accept it with probability

$$
P(\text{accept}) = e^{\Delta E / T}
$$

where the **temperature** $T$ decreases over time according to a *schedule*. At high temperature the search explores widely, even going downhill; as $T \to 0$ it behaves like hill climbing.

```python
import math, random

def simulated_annealing(state, energy, neighbor, T0=10.0, alpha=0.995, steps=20000):
    current, e_cur = state, energy(state)
    best, e_best = current, e_cur
    T = T0
    for _ in range(steps):
        cand = neighbor(current)
        e_new = energy(cand)
        delta = e_cur - e_new             # positive means improvement (minimising energy)
        if delta > 0 or random.random() < math.exp(delta / T):
            current, e_cur = cand, e_new
            if e_cur < e_best:
                best, e_best = current, e_cur
        T = max(T * alpha, 1e-8)
    return best, e_best
```

:::note
Theory says that if the temperature is lowered *slowly enough* (logarithmically), simulated annealing finds the global optimum with probability approaching one. In practice that schedule is far too slow, so we use geometric cooling ($T \leftarrow \alpha T$) and accept good-but-not-optimal answers. This tension between theoretical guarantees and practical schedules will reappear with learning-rate schedules in deep learning.
:::

## Local beam search

Keep $k$ states instead of one. At each step generate all successors of all $k$ states and keep the best $k$. Unlike $k$ independent restarts, information is shared — states that are doing well attract the search. **Stochastic beam search** chooses successors with probability proportional to their value, avoiding a collapse into one region. (Beam search reappears in NLP for decoding sequences.)

## Continuous spaces and gradients

When states are real-valued vectors $\mathbf{x}$ and the objective $f$ is differentiable, the best local move is in the direction of the gradient:

$$
\mathbf{x} \leftarrow \mathbf{x} + \eta\, \nabla f(\mathbf{x})
$$

This is **gradient ascent** — hill climbing with infinitesimal steps. Replace ascent with descent on a loss function and you have the engine that trains every neural network. Local optima, plateaus and ridges all have analogues in the loss landscapes of deep networks (saddle points, flat regions, narrow valleys).

:::exercise
1. Add sideways moves (limit 100) to the 8-queens hill climber and measure the success rate over 1,000 runs.
2. Use simulated annealing to solve a 20-city travelling salesperson problem where a neighbour reverses a random segment of the tour (2-opt).
3. Explain why local beam search with $k$ states is different from $k$ parallel random restarts.
:::

:::takeaway
- Local search works on complete configurations with constant memory.
- Hill climbing is fast but gets stuck in local optima, plateaus and ridges; random restarts help.
- Simulated annealing accepts worse moves with probability $e^{\Delta E/T}$ to escape local optima.
- Gradient descent is the continuous cousin of hill climbing — the foundation of deep learning.
:::

=== POST ===
slug: genetic-algorithms-evolutionary-computation
title: Genetic Algorithms and Evolutionary Computation
category: foundations
level: Beginner
tags: genetic algorithms, evolution, optimization, neuroevolution
summary: Nature optimises through selection, crossover and mutation. We implement a genetic algorithm from scratch, discuss the schema theorem and survey evolution strategies and neuroevolution.
---
Evolution has produced eyes, wings and brains without a designer. **Evolutionary computation** borrows its principles — variation and selection — to search for solutions to hard optimisation problems. Genetic algorithms (GAs) are especially useful when the objective is not differentiable, the search space is discrete or strange, and we can evaluate candidate solutions but not reason about them analytically.

## The core loop

A genetic algorithm maintains a **population** of candidate solutions called **individuals**, each encoded as a **chromosome** (often a bit string or a vector). Each generation:

1. **Evaluate** each individual with a **fitness function**.
2. **Select** parents, favouring fitter individuals.
3. **Crossover** (recombination): combine two parents to produce offspring.
4. **Mutate**: randomly perturb offspring with small probability.
5. **Replace** the old population (often keeping the best few — **elitism**).

## Selection schemes

- **Fitness-proportionate (roulette wheel)**: probability of selection $\propto$ fitness. Sensitive to the scale of fitness values.
- **Tournament selection**: pick $k$ individuals at random, choose the best. Simple, robust and tunable via $k$.
- **Rank selection**: select based on rank rather than raw fitness.

Selection pressure controls the balance between **exploitation** (converge fast on good solutions) and **exploration** (maintain diversity). Too much pressure causes *premature convergence*.

## Crossover and mutation

For bit strings, **one-point crossover** picks a cut point and swaps tails:

```text
Parent A: 11010 | 011        Child 1: 11010 | 100
Parent B: 00111 | 100   →    Child 2: 00111 | 011
```

**Uniform crossover** chooses each gene from either parent independently. For permutations (like travelling salesperson tours) we need special operators such as **order crossover (OX)** that preserve validity. **Mutation** flips bits (probability typically $\approx 1/L$ for length $L$) or adds Gaussian noise to real-valued genes.

## A complete GA in Python

We maximise the number of ones in a 40-bit string (the "OneMax" problem) — trivial, but it shows every component.

```python
import random

L, POP, GENS, PMUT = 40, 60, 80, 1 / 40

def fitness(ind):
    return sum(ind)

def tournament(pop, k=3):
    return max(random.sample(pop, k), key=fitness)

def crossover(a, b):
    cut = random.randint(1, L - 1)
    return a[:cut] + b[cut:], b[:cut] + a[cut:]

def mutate(ind):
    return [1 - g if random.random() < PMUT else g for g in ind]

pop = [[random.randint(0, 1) for _ in range(L)] for _ in range(POP)]
for gen in range(GENS):
    elite = max(pop, key=fitness)
    children = [elite]                         # elitism
    while len(children) < POP:
        c1, c2 = crossover(tournament(pop), tournament(pop))
        children += [mutate(c1), mutate(c2)]
    pop = children[:POP]
    if gen % 10 == 0:
        print(gen, fitness(max(pop, key=fitness)))
print("best:", fitness(max(pop, key=fitness)))
```

## Why does it work? The schema theorem

John Holland, who invented GAs in the 1970s, analysed them using **schemata** — templates like `1**0*` where `*` is a wildcard. His **schema theorem** states that short, low-order schemata with above-average fitness receive exponentially increasing numbers of trials in successive generations:

$$
\mathbb{E}[m(H, t+1)] \ge m(H, t)\,\frac{f(H)}{\bar f}\left[1 - p_c \frac{\delta(H)}{L - 1} - o(H)\,p_m\right]
$$

where $m(H,t)$ counts instances of schema $H$, $f(H)$ is its average fitness, $\delta(H)$ its defining length and $o(H)$ its order. The associated **building-block hypothesis** suggests GAs work by combining good short building blocks. It gives useful intuition, though it is not a full convergence proof.

:::warning
The encoding makes or breaks a GA. If good solutions cannot be assembled from good parts under your crossover operator, crossover becomes random destruction. Spend most of your design effort on representation and fitness function, not on tuning rates.
:::

## Relatives in the evolutionary family

- **Evolution Strategies (ES)**: real-valued vectors with self-adapting mutation strengths. **CMA-ES** adapts a full covariance matrix and is a state-of-the-art black-box optimiser for continuous problems.
- **Genetic Programming**: evolves programs or expression trees — useful for symbolic regression.
- **Differential Evolution**: creates mutants from differences between population members.
- **Neuroevolution**: evolves neural network weights and/or architectures (NEAT). Large-scale ES has been shown to be competitive with reinforcement learning on some control tasks because it parallelises extremely well.

## When to use evolutionary methods

Use them when gradients are unavailable, the landscape is rugged or discontinuous, evaluation can be parallelised, or you need a *set* of diverse solutions (multi-objective optimisation with NSGA-II produces a Pareto front). Avoid them when a gradient or a specialised solver exists — those are usually far more sample-efficient.

:::exercise
1. Modify the GA to solve the 8-queens problem using a permutation encoding and order crossover.
2. Plot best and average fitness per generation with and without elitism.
3. Try tournament sizes 2, 5 and 10. How does selection pressure affect convergence speed and final quality?
:::

:::takeaway
- GAs evolve a population through selection, crossover and mutation.
- Representation and fitness design matter more than parameter tuning.
- The schema theorem explains how good building blocks spread.
- CMA-ES, genetic programming and neuroevolution extend the idea to continuous, program and network search.
:::

=== POST ===
slug: fuzzy-logic-systems
title: Fuzzy Logic: Reasoning with Degrees of Truth
category: foundations
level: Beginner
tags: fuzzy logic, control systems, membership functions, reasoning
summary: Is 29°C "hot"? Fuzzy logic replaces true/false with degrees of membership. We build a Mamdani fuzzy controller step by step: fuzzification, rule evaluation, aggregation and defuzzification.
---
Classical logic insists that every statement is true or false. But human concepts are vague: "tall", "hot", "fast", "high risk". Is a 29°C day hot? Somewhat. **Fuzzy logic**, introduced by Lotfi Zadeh in 1965, formalises such graded concepts. It became famous through control systems — from washing machines and rice cookers to the Sendai subway in Japan, whose fuzzy controller gave a smoother ride.

## Fuzzy sets

A classical (crisp) set $A$ has a characteristic function $\chi_A(x) \in \{0, 1\}$. A **fuzzy set** has a **membership function**

$$
\mu_A : X \rightarrow [0, 1]
$$

where $\mu_A(x)$ is the degree to which $x$ belongs to $A$. Common shapes are triangular, trapezoidal and Gaussian:

$$
\mu_{\text{tri}}(x; a, b, c) = \max\left(0,\; \min\left(\frac{x - a}{b - a},\; \frac{c - x}{c - b}\right)\right)
$$

:::warning
Fuzzy membership is **not** probability. $\mu_{\text{Hot}}(29) = 0.7$ does not mean "there is a 70% chance it is hot". It means 29°C is hot *to degree 0.7*. Probability describes uncertainty about crisp facts; fuzziness describes vagueness of concepts.
:::

## Fuzzy operators

Zadeh's standard operators generalise logic:

| Operation | Definition |
|---|---|
| AND (intersection) | $\mu_{A \cap B}(x) = \min(\mu_A(x), \mu_B(x))$ |
| OR (union) | $\mu_{A \cup B}(x) = \max(\mu_A(x), \mu_B(x))$ |
| NOT (complement) | $\mu_{\neg A}(x) = 1 - \mu_A(x)$ |

More generally, AND can be any **t-norm** (e.g. product $ab$) and OR any **t-conorm** (e.g. probabilistic sum $a + b - ab$). Note that fuzzy logic violates the law of excluded middle: $\max(\mu, 1 - \mu)$ can be less than 1.

## A fuzzy controller, step by step

Let us design a fan-speed controller. Input: temperature (°C). Output: fan speed (%).

**Linguistic variables:**
- Temperature: *Cold*, *Warm*, *Hot*.
- Fan speed: *Slow*, *Medium*, *Fast*.

**Rules:**
1. IF temperature is Cold THEN speed is Slow.
2. IF temperature is Warm THEN speed is Medium.
3. IF temperature is Hot THEN speed is Fast.

The **Mamdani** inference process has four stages:

1. **Fuzzification** — compute membership degrees of the crisp input.
2. **Rule evaluation** — compute each rule's firing strength and clip its output set.
3. **Aggregation** — combine clipped output sets with max.
4. **Defuzzification** — convert the aggregated fuzzy set into a crisp number, usually with the **centroid**:

$$
y^* = \frac{\int y\, \mu(y)\, dy}{\int \mu(y)\, dy}
$$

```python
import numpy as np

def tri(x, a, b, c):
    return np.maximum(0, np.minimum((x - a) / (b - a + 1e-9), (c - x) / (c - b + 1e-9)))

temp_sets = {"cold": (0, 10, 22), "warm": (18, 25, 32), "hot": (28, 38, 50)}
speed = np.linspace(0, 100, 501)
speed_sets = {"slow": tri(speed, 0, 15, 45), "medium": tri(speed, 30, 50, 70), "fast": tri(speed, 55, 85, 100)}
rules = [("cold", "slow"), ("warm", "medium"), ("hot", "fast")]

def fan_speed(t):
    agg = np.zeros_like(speed)
    for tin, sout in rules:
        strength = tri(np.array(t, float), *temp_sets[tin])       # fuzzification
        agg = np.maximum(agg, np.minimum(strength, speed_sets[sout]))  # clip + aggregate
    return float((speed * agg).sum() / (agg.sum() + 1e-9))      # centroid

for t in [8, 20, 26, 30, 40]:
    print(t, "°C ->", round(fan_speed(t), 1), "%")
```

Notice how the output changes *smoothly* with temperature — there is no abrupt jump at a threshold. That smoothness is exactly why fuzzy controllers feel natural.

## Sugeno models

The **Takagi–Sugeno–Kang (TSK)** model uses crisp functions as rule outputs, e.g. "IF temperature is Hot THEN speed $= 2t + 10$". The final output is the weighted average of rule outputs by firing strength. TSK models are computationally cheaper and easier to optimise.

## Neuro-fuzzy systems

**ANFIS** (Adaptive Neuro-Fuzzy Inference System, 1993) represents a Sugeno fuzzy system as a network and learns membership-function parameters by gradient descent. This combines the interpretability of linguistic rules with the adaptivity of learning — an early instance of the neuro-symbolic idea.

## Where fuzzy logic stands today

Fuzzy control remains common in consumer appliances, automotive subsystems, industrial process control and decision-support tools where experts can describe behaviour in words. In mainstream machine learning, probabilistic and neural methods dominate, but fuzzy ideas persist in soft membership (e.g. fuzzy c-means clustering) and in interpretable rule-based models.

:::exercise
1. Add a second input, humidity (Low/High), and write six rules for the fan controller.
2. Replace min with product for AND. How does the output curve change?
3. Implement fuzzy c-means clustering and compare its soft assignments with k-means.
:::

:::takeaway
- Fuzzy sets assign degrees of membership in $[0,1]$; fuzziness models vagueness, not randomness.
- AND/OR/NOT generalise to min/max/complement (or other t-norms).
- Mamdani inference: fuzzify → evaluate rules → aggregate → defuzzify.
- Neuro-fuzzy systems like ANFIS learn fuzzy parameters from data.
:::

=== POST ===
slug: monte-carlo-tree-search
title: Monte Carlo Tree Search: The Algorithm Behind Superhuman Go
category: foundations
level: Intermediate
tags: mcts, games, uct, alphago, bandits
summary: When the game tree is too vast and positions too hard to evaluate, MCTS builds an asymmetric tree guided by random simulations and the UCB1 bandit formula. We implement it and connect it to AlphaZero.
---
For decades, Go resisted the alpha–beta methods that conquered chess. Its branching factor is around 250, and — worse — nobody could write a good evaluation function for Go positions. The breakthrough came in 2006 with **Monte Carlo Tree Search (MCTS)**, which estimates position values from random play-outs and focuses effort on the most promising moves. A decade later MCTS combined with deep networks produced AlphaGo.

## The key idea

Instead of evaluating a position with hand-written heuristics, **play many games to the end** from that position, choosing moves randomly (or with a cheap policy), and record the win rate. A position from which random play often wins is probably good. Then grow a search tree *selectively* — spending more simulations on promising moves.

## The four phases

Each MCTS iteration consists of:

1. **Selection** — starting at the root, descend the tree choosing children by a *tree policy* that balances exploration and exploitation, until reaching a node with unexpanded children.
2. **Expansion** — add one (or more) child nodes.
3. **Simulation (roll-out)** — play a random game from the new node to a terminal state.
4. **Backpropagation** — propagate the result up the path, updating visit counts $N$ and total rewards $W$ at every node.

After the time budget is exhausted, play the move at the root with the **most visits** (more robust than highest average).

## UCT: the tree policy

Choosing a child is a **multi-armed bandit** problem. The **UCB1** formula, applied to trees (UCT, Kocsis & Szepesvári 2006), picks the child maximising

$$
\text{UCT}(i) = \frac{W_i}{N_i} + c\sqrt{\frac{\ln N}{N_i}}
$$

The first term favours children with high average reward (exploitation); the second favours rarely visited children (exploration). The constant $c$ (theoretically $\sqrt{2}$ for rewards in $[0,1]$) tunes the balance. Under UCT, the estimated values converge to the minimax values as simulations grow.

## Implementation

```python
import math, random

class Node:
    def __init__(self, state, parent=None, move=None):
        self.state, self.parent, self.move = state, parent, move
        self.children = []
        self.untried = list(state.legal_moves())
        self.N, self.W = 0, 0.0

    def uct_child(self, c=1.4):
        return max(self.children,
                   key=lambda ch: ch.W / ch.N + c * math.sqrt(math.log(self.N) / ch.N))

def mcts(root_state, iterations=2000):
    root = Node(root_state)
    for _ in range(iterations):
        node, state = root, root_state.copy()
        # 1. Selection
        while not node.untried and node.children:
            node = node.uct_child()
            state.play(node.move)
        # 2. Expansion
        if node.untried:
            move = node.untried.pop(random.randrange(len(node.untried)))
            state.play(move)
            child = Node(state.copy(), node, move)
            node.children.append(child)
            node = child
        # 3. Simulation
        while not state.is_over():
            state.play(random.choice(state.legal_moves()))
        # 4. Backpropagation (reward from the perspective of the player who moved into node)
        while node is not None:
            node.N += 1
            node.W += state.reward_for(node.state.player_who_just_moved())
            node = node.parent
    return max(root.children, key=lambda ch: ch.N).move
```

The `state` object needs `legal_moves`, `play`, `copy`, `is_over`, `reward_for` and `player_who_just_moved` methods. Implement them for tic-tac-toe or Connect Four and you will have a strong player with no game knowledge beyond the rules.

## Why MCTS is attractive

- **Anytime**: stop whenever you like and return the best move so far.
- **Asymmetric**: the tree grows deep along promising lines and stays shallow elsewhere.
- **Domain-independent**: only needs a simulator of the rules — no evaluation function.
- **Handles stochasticity and imperfect information** with variants (e.g. Information Set MCTS).

## From MCTS to AlphaZero

Random roll-outs are noisy. **AlphaGo** (2016) and **AlphaZero** (2017) replaced them with deep neural networks:

- A **policy network** $p(a \mid s)$ provides priors that focus selection on plausible moves.
- A **value network** $v(s)$ estimates the outcome directly — AlphaZero dropped roll-outs entirely.

The selection rule becomes **PUCT**:

$$
a^* = \arg\max_a \left[ Q(s,a) + c\, P(s,a) \frac{\sqrt{\sum_b N(s,b)}}{1 + N(s,a)} \right]
$$

AlphaZero trains both networks from self-play, using MCTS as a **policy improvement operator**: the visit distribution produced by search is a better policy than the raw network, so the network is trained to imitate it. Starting from random play and given only the rules, AlphaZero reached superhuman strength in chess, shogi and Go. **MuZero** (2019) went further, learning a model of the rules themselves.

:::note
MCTS + learned networks is a template far beyond board games. It has been used for chemical synthesis planning, discovering faster matrix-multiplication algorithms (AlphaTensor) and sorting routines (AlphaDev), and it inspires "search at inference time" methods for language-model reasoning.
:::

:::exercise
1. Implement the `state` interface for tic-tac-toe and verify that MCTS with 1,000 iterations never loses.
2. Vary the exploration constant $c$ from 0.1 to 5. What happens to playing strength?
3. Explain why choosing the most-visited child is more robust than choosing the highest average value.
:::

:::takeaway
- MCTS estimates values by simulation and grows an asymmetric tree.
- Its four phases are selection, expansion, simulation and backpropagation.
- UCT uses the UCB1 bandit formula to balance exploration and exploitation.
- AlphaZero replaces roll-outs with policy and value networks trained by self-play.
:::

=== POST ===
slug: decision-theory-and-utility
title: Decision Theory: Utility, Expected Value and the Value of Information
category: foundations
level: Intermediate
tags: decision theory, utility, expected utility, value of information
summary: Rational agents must act under uncertainty. We develop utility theory from axioms, the principle of maximum expected utility, decision networks and the value of perfect information.
---
Probability tells an agent what it should *believe*. Decision theory tells it what it should *do*. Combining the two gives the principle at the heart of rational agency: **choose the action with maximum expected utility (MEU)**. Today we build this theory carefully, because it underlies reinforcement learning, Bayesian optimisation and responsible deployment of any ML system that makes decisions.

## Maximum expected utility

Suppose action $a$ leads to outcome $s'$ with probability $P(s' \mid a, e)$ given evidence $e$, and $U(s')$ measures how desirable $s'$ is. The expected utility is

$$
EU(a \mid e) = \sum_{s'} P(s' \mid a, e)\, U(s')
$$

and a rational agent chooses

$$
a^* = \arg\max_a EU(a \mid e)
$$

## Why utilities? The von Neumann–Morgenstern axioms

Why should preferences be representable by a number, and why should we maximise its *expectation*? Von Neumann and Morgenstern (1944) showed that if an agent's preferences over **lotteries** (probability distributions over outcomes) satisfy a few reasonable axioms, then a utility function exists and the agent behaves as a maximiser of expected utility. The axioms are:

1. **Orderability** — any two lotteries can be compared.
2. **Transitivity** — if $A \succ B$ and $B \succ C$ then $A \succ C$.
3. **Continuity** — if $A \succ B \succ C$, some mixture of $A$ and $C$ is equivalent to $B$.
4. **Substitutability** — indifferent lotteries can be swapped inside more complex lotteries.
5. **Monotonicity** — a higher chance of a preferred outcome is preferred.
6. **Decomposability** — compound lotteries reduce to simple ones.

:::note
Violating transitivity makes an agent exploitable. If you prefer A to B, B to C and C to A, a trader can charge you a small fee to swap C for B, then B for A, then A for C — forever. This "money pump" argument is why rationality is defined by consistency, not by any particular goal.
:::

## Utility of money and risk attitudes

Utility is not the same as money. Most people prefer a guaranteed 1 million to a 50% chance of 3 million, even though the expected *money* of the gamble is higher. This is captured by a **concave** utility function, such as $U(x) = \log x$, which models **risk aversion**. The difference between the expected monetary value of a lottery and its **certainty equivalent** is the **risk premium** — the basis of the insurance industry.

- Concave $U$: risk-averse.
- Linear $U$: risk-neutral.
- Convex $U$: risk-seeking.

## Human irrationality

Kahneman and Tversky's experiments showed humans systematically deviate from expected-utility theory: we weigh losses more than gains (**loss aversion**), are influenced by how options are **framed**, and overweight small probabilities. Their **prospect theory** describes these behaviours. AI systems that interact with humans must account for them, and designers must beware of building systems that exploit them.

## Decision networks

A **decision network** (influence diagram) extends a Bayesian network with:

- **chance nodes** (ovals) — random variables;
- **decision nodes** (rectangles) — choices;
- **utility nodes** (diamonds) — the utility as a function of parents.

To evaluate: for each value of the decision node, set it, compute posterior probabilities of the utility node's parents, compute expected utility, and pick the best.

## The value of information

Should a doctor order an expensive test before treating? Should an oil company buy a seismic survey? Decision theory answers precisely. The **value of perfect information (VPI)** about variable $E_j$ is

$$
VPI_e(E_j) = \left( \sum_{k} P(E_j = e_{jk} \mid e)\; EU(a^*_{e_{jk}} \mid e, E_j = e_{jk}) \right) - EU(a^* \mid e)
$$

— the expected utility of deciding *after* learning $E_j$, minus the expected utility of deciding now. Key properties:

- VPI is **never negative** (in expectation, information cannot hurt a rational agent).
- VPI is **zero** if the information would not change the optimal decision.
- VPI is **not additive** — two tests may be redundant.

A **myopic information-gathering agent** requests the observation with the highest VPI minus cost, as long as it is positive.

```python
def expected_utility(action, p_state, utility):
    return sum(p * utility[action][s] for s, p in p_state.items())

# Should we deploy a new model? States: model is good / bad.
p = {"good": 0.6, "bad": 0.4}
U = {"deploy": {"good": 100, "bad": -150}, "keep_old": {"good": 0, "bad": 0}}

eu_now = max(expected_utility(a, p, U) for a in U)
# Perfect information: we would learn the true state first, then choose the best action.
eu_informed = sum(p[s] * max(U[a][s] for a in U) for s in p)
print("EU now:", eu_now, " EU with info:", eu_informed, " VPI:", eu_informed - eu_now)
```

Here deploying now has expected utility $0.6 \times 100 - 0.4 \times 150 = 0$, the same as keeping the old model. With perfect information we deploy only when good: $0.6 \times 100 = 60$. So a perfect evaluation is worth up to 60 utility units — a quantitative argument for investing in a thorough offline and A/B evaluation before deployment.

:::exercise
1. With $U(x) = \sqrt{x}$, find the certainty equivalent of a 50/50 lottery between 0 and 10,000.
2. Show that VPI is zero when the optimal action is the same for every value of $E_j$.
3. Draw a decision network for deciding whether to evacuate a camp before a forecast cyclone.
:::

:::takeaway
- Rational action = maximise expected utility.
- The vNM axioms justify utility functions; violating them makes agents exploitable.
- Concave utility captures risk aversion; humans deviate systematically (prospect theory).
- The value of information quantifies when gathering evidence is worth the cost.
:::

=== POST ===
slug: swarm-intelligence-pso-aco
title: Swarm Intelligence: Particle Swarm Optimisation and Ant Colony Optimisation
category: foundations
level: Beginner
tags: swarm intelligence, pso, aco, optimization, nature-inspired
summary: Ants find shortest paths and birds flock without a leader. We study how simple local rules produce intelligent collective behaviour and implement PSO and ACO for optimisation problems.
---
No single ant knows the shortest route to food, yet the colony finds it. No single starling directs the flock, yet it moves as one. **Swarm intelligence** studies how collective intelligence emerges from many simple agents following local rules. From this, AI researchers derived powerful optimisation algorithms that are simple to implement and embarrassingly parallel.

## Principles of swarm systems

Swarm systems share a few properties:

- **Decentralisation** — no central controller.
- **Local interaction** — agents sense only neighbours or their immediate environment.
- **Stigmergy** — indirect communication by modifying the environment (e.g. ants leaving pheromone).
- **Positive and negative feedback** — good solutions are reinforced; evaporation or inertia prevents lock-in.
- **Emergence** — global patterns arise that no individual intends.

Craig Reynolds' **Boids** (1987) showed that three rules — *separation*, *alignment* and *cohesion* — produce realistic flocking animation. It has been used in films and games ever since.

## Particle Swarm Optimisation (PSO)

Kennedy and Eberhart (1995) introduced PSO to optimise continuous functions. Each particle $i$ has a position $\mathbf{x}_i$ (a candidate solution) and velocity $\mathbf{v}_i$. It remembers its **personal best** $\mathbf{p}_i$, and the swarm knows the **global best** $\mathbf{g}$. At each step:

$$
\mathbf{v}_i \leftarrow w\,\mathbf{v}_i + c_1 r_1 (\mathbf{p}_i - \mathbf{x}_i) + c_2 r_2 (\mathbf{g} - \mathbf{x}_i)
$$

$$
\mathbf{x}_i \leftarrow \mathbf{x}_i + \mathbf{v}_i
$$

where $r_1, r_2 \sim U(0,1)$ are random, $w$ is the **inertia weight**, $c_1$ the **cognitive** coefficient (trust in self) and $c_2$ the **social** coefficient (trust in the swarm). Typical values: $w \approx 0.7$, $c_1 = c_2 \approx 1.5$.

```python
import numpy as np

def rastrigin(x):                  # many local minima; global min 0 at the origin
    return 10 * x.shape[-1] + (x**2 - 10 * np.cos(2 * np.pi * x)).sum(axis=-1)

def pso(f, dim=5, n=40, iters=300, w=0.72, c1=1.49, c2=1.49, bound=5.12, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-bound, bound, (n, dim))
    v = rng.uniform(-1, 1, (n, dim))
    pbest, pval = x.copy(), f(x)
    g = pbest[pval.argmin()].copy()
    for _ in range(iters):
        r1, r2 = rng.random((n, dim)), rng.random((n, dim))
        v = w * v + c1 * r1 * (pbest - x) + c2 * r2 * (g - x)
        x = np.clip(x + v, -bound, bound)
        val = f(x)
        better = val < pval
        pbest[better], pval[better] = x[better], val[better]
        g = pbest[pval.argmin()].copy()
    return g, pval.min()

best, value = pso(rastrigin)
print(best.round(3), round(value, 4))
```

:::tip
If the swarm converges too quickly to a poor solution, use a **local-best topology** (each particle only knows the best of its neighbours in a ring) instead of the global best. Information spreads more slowly, which preserves diversity — the same exploration–exploitation trade-off we met in genetic algorithms and MCTS.
:::

## Ant Colony Optimisation (ACO)

Marco Dorigo's ACO (1992) mimics how ants find short paths. Ants deposit pheromone as they walk; shorter paths are completed faster and more often, so they accumulate more pheromone, attracting more ants — positive feedback. Pheromone evaporates, preventing the colony from locking into early choices.

For the **travelling salesperson problem**, ant $k$ at city $i$ chooses the next city $j$ with probability

$$
p_{ij}^k = \frac{\tau_{ij}^{\alpha}\, \eta_{ij}^{\beta}}{\sum_{l \in \text{allowed}} \tau_{il}^{\alpha}\, \eta_{il}^{\beta}}
$$

where $\tau_{ij}$ is the pheromone on edge $(i,j)$, $\eta_{ij} = 1/d_{ij}$ is the heuristic desirability, and $\alpha, \beta$ weight the two. After all ants finish their tours, pheromone updates as

$$
\tau_{ij} \leftarrow (1 - \rho)\,\tau_{ij} + \sum_k \Delta\tau_{ij}^k, \qquad \Delta\tau_{ij}^k = \frac{Q}{L_k} \text{ if ant } k \text{ used } (i,j)
$$

with evaporation rate $\rho$ and tour length $L_k$.

ACO works especially well on **dynamic** graph problems — for example routing in communication networks where link costs change, because the pheromone trail continuously adapts.

## Other swarm algorithms

- **Artificial Bee Colony** — employed, onlooker and scout bees balance exploitation and exploration.
- **Firefly algorithm** — attractiveness decays with distance.
- **Swarm robotics** — many simple robots cooperate for search-and-rescue, mapping or agriculture, relying on local communication for robustness.

:::warning
The literature contains hundreds of "novel" metaphor-based algorithms (inspired by wolves, whales, bats…). Many are minor variations of PSO or evolutionary strategies with new vocabulary. As a scientist, judge an algorithm by rigorous benchmarks and clear mechanisms, not by the charm of its metaphor.
:::

:::exercise
1. Run the PSO code on the sphere function $f(\mathbf{x}) = \sum x_i^2$ and on Rastrigin. Plot the best value per iteration.
2. Implement ACO for a 15-city TSP and compare it with simulated annealing.
3. Explain the role of evaporation in ACO. What happens if $\rho = 0$? If $\rho = 1$?
:::

:::takeaway
- Swarm intelligence produces global intelligence from simple local rules and stigmergy.
- PSO moves particles towards personal and global bests with inertia.
- ACO builds solutions probabilistically guided by pheromone and heuristics, with evaporation.
- Evaluate nature-inspired algorithms by evidence, not metaphors.
:::

=== POST ===
slug: narrow-general-and-super-intelligence
title: Narrow AI, General AI and Superintelligence: Separating Science from Speculation
category: foundations
level: Beginner
tags: agi, narrow ai, superintelligence, evaluation, philosophy
summary: What would it mean for AI to be "general"? We define narrow and general intelligence, examine how to measure generality, and discuss the arguments about superintelligence with scientific care.
---
Few topics generate more headlines — and more confusion — than "Artificial General Intelligence". As future scientists and engineers, you will be asked about it by journalists, policymakers and your own families. Today we build a careful vocabulary so you can discuss it with precision rather than hype.

## Narrow AI

**Narrow (or weak) AI** refers to systems designed or trained for a specific task or a bounded family of tasks: recognising faces, translating text, recommending videos, playing Go. Narrow systems can be superhuman within their domain — no human beats AlphaZero at chess — yet useless outside it. AlphaZero cannot tell you what a chess piece is made of.

## General intelligence

**Artificial General Intelligence (AGI)** usually refers to a system that can perform a wide range of cognitive tasks at or above human level, adapt to *new* tasks efficiently, and transfer knowledge across domains. The key word is **generality** — not peak performance on any single task.

Several definitions coexist:

- **Human-comparison definitions**: AGI can do most economically valuable cognitive work that humans can do.
- **Capability-breadth definitions**: performance across a broad battery of tasks.
- **Learning-efficiency definitions**: François Chollet argues that intelligence is *skill-acquisition efficiency* — how quickly a system learns new skills from little data, relative to its priors and experience. His **ARC** (Abstraction and Reasoning Corpus) benchmark tests this with novel visual puzzles.
- **Levels frameworks**: some researchers propose graded levels (emerging, competent, expert, virtuoso, superhuman) crossed with breadth (narrow vs general), treating AGI as a spectrum rather than a single threshold.

:::note
Notice that these definitions disagree on what to measure: *outputs* (tasks completed), *breadth*, or *learning efficiency*. When someone claims a system "is" or "is not" AGI, the first question to ask is: **by which definition, measured how?**
:::

## Where current systems stand

Large language and multimodal models are the most general AI systems built so far: one model writes code, summarises law, explains physics and describes images. At the same time, careful evaluations reveal characteristic weaknesses:

- sensitivity to phrasing and irrelevant details;
- difficulty with problems requiring long chains of reliable reasoning or genuinely novel abstractions;
- factual errors stated confidently (hallucination);
- limited ability to learn continually from experience after training.

The honest scientific position is that generality has increased dramatically, measurement is hard, and experts disagree about how far current approaches will go.

## Measuring generality

A good evaluation of general capability should be:

1. **Broad** — covering many domains and skill types.
2. **Novel** — tasks the system could not have memorised from training data (**data contamination** is a serious problem when models are trained on much of the internet).
3. **Efficiency-aware** — accounting for how much data and compute the system needed.
4. **Robust** — performance should survive rephrasing and adversarial variation.

```python
# A tiny illustration of contamination checking: n-gram overlap between
# a benchmark question and a training corpus.
def ngrams(text, n=8):
    words = text.lower().split()
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}

def overlap_ratio(test_item, corpus_ngrams, n=8):
    grams = ngrams(test_item, n)
    return len(grams & corpus_ngrams) / max(1, len(grams))
```

## Superintelligence

**Superintelligence** refers to intellect that greatly exceeds human cognitive performance in virtually all domains. Arguments about it include:

- **The intelligence explosion hypothesis** (I. J. Good, 1965): a machine that can improve its own design could trigger recursive self-improvement.
- **Orthogonality thesis** (Bostrom): intelligence and final goals are independent — a highly capable system need not share human values.
- **Instrumental convergence**: many goals imply similar sub-goals (acquire resources, avoid being switched off), which could create risks if goals are misspecified.

Sceptics respond that intelligence is not a single scalar, that real-world progress is bottlenecked by experiments, data and physical resources, and that capabilities have historically grown gradually rather than explosively.

## A responsible stance

As a researcher, you can hold two ideas at once:

1. Present-day harms — bias, misinformation, privacy violations, labour disruption — are real and need solving **now**.
2. Longer-term risks from increasingly capable systems deserve serious technical research in **alignment, interpretability and evaluation**.

These are not competing priorities; both demand rigorous engineering and good governance. We return to both in the ethics track.

:::exercise
1. Choose one AGI definition above and design three tasks that would test it without contamination risk.
2. Try a few ARC-style puzzles (available online). Describe the reasoning you used. Could a pattern matcher solve them?
3. Write a 300-word response to a newspaper headline claiming "AI is now smarter than humans". What would you ask the journalist?
:::

:::takeaway
- Narrow AI excels at bounded tasks; AGI emphasises generality and adaptation.
- Definitions of AGI differ — always ask which definition and which measurement.
- Good evaluations of generality are broad, novel, efficiency-aware and robust.
- Superintelligence arguments (explosion, orthogonality, instrumental convergence) are debated; present and future risks both merit serious work.
:::

=== POST ===
slug: landscape-of-modern-ai
title: The Landscape of Modern AI: A Map for Students
category: foundations
level: Beginner
tags: overview, roadmap, careers, research areas
summary: A guided map of today's AI ecosystem — the paradigms, the tools, the research frontiers and how the tracks of this course fit together — so you always know where you are in the journey.
---
We end the Foundations track by stepping back and looking at the whole map. When students first enter AI, they often feel lost among acronyms — CNN, RL, LLM, RAG, MLOps. This lecture gives you a mental atlas. Return to it whenever you feel lost in later tracks.

## Three paradigms, one goal

Every AI system you meet uses some mixture of three paradigms:

1. **Search and reasoning** — explore possibilities explicitly (A*, alpha–beta, MCTS, planners, SAT solvers).
2. **Probabilistic modelling** — represent and update uncertainty (Bayesian networks, HMMs, Gaussian processes).
3. **Learning from data** — fit functions from examples (regression, trees, neural networks).

Modern breakthroughs often *combine* them: AlphaZero combines search and learning; a self-driving stack combines learned perception, probabilistic tracking and planning; an LLM agent combines a learned language model with tool calls and search.

## The machine learning taxonomy

| Paradigm | Data | Goal | Example |
|---|---|---|---|
| Supervised learning | Inputs with labels | Predict labels | Spam detection |
| Unsupervised learning | Inputs only | Discover structure | Customer segmentation |
| Self-supervised learning | Inputs; labels derived from data | Learn representations | Predicting masked words |
| Reinforcement learning | Interaction and rewards | Learn a policy | Game playing, robotics |
| Semi-supervised / weak supervision | Few labels, many unlabelled | Leverage both | Medical imaging |

## The deep learning stack

Deep learning dominates perception and language. Its components:

- **Architectures** — MLPs, CNNs (images), RNNs (sequences), Transformers (almost everything now), GNNs (graphs), diffusion models (generation).
- **Training** — backpropagation, stochastic gradient descent and Adam, normalisation, regularisation.
- **Scale** — GPUs/TPUs, distributed training, mixed precision.
- **Paradigms** — pretraining on huge unlabelled data, then fine-tuning or prompting for tasks.

## Application domains

- **Computer vision** — classification, detection, segmentation, medical imaging, remote sensing.
- **Natural language processing** — translation, question answering, summarisation, chat assistants.
- **Speech** — recognition, synthesis, speaker identification.
- **Science** — protein structure, weather forecasting, materials discovery.
- **Recommendation and search** — the most economically significant ML systems in the world.
- **Robotics and control** — manipulation, navigation, autonomous vehicles.
- **Humanitarian and development work** — mapping settlements from satellite imagery, forecasting displacement, multilingual information services for people in need.

## The engineering reality

A research notebook is not a product. Deployed ML requires **data pipelines**, **evaluation**, **monitoring**, **versioning** and **governance**. Studies of production systems have repeatedly found that the model code is a small fraction of the whole system. That is why this course includes an MLOps track.

## Research frontiers (as of today)

1. **Reasoning and planning** in learned models.
2. **Efficiency** — smaller, faster, cheaper models; on-device AI.
3. **Multimodality** — joint understanding of text, images, audio, video and action.
4. **Agents** — models that use tools, browse, write code and act over long horizons.
5. **Alignment, interpretability and safety.**
6. **Data** — curation, synthetic data, and the limits of available human-generated text.
7. **AI for science** and **AI for social good**.

## How to use this course

The tracks form a deliberate sequence:

```text
AI Foundations ──► Mathematics for ML ──► Machine Learning ──► Deep Learning
                                                                   │
            ┌───────────────┬──────────────────┬───────────────────┤
            ▼               ▼                  ▼                   ▼
     Computer Vision   NLP & Transformers   Generative AI   Reinforcement Learning
            └───────────────┴────────┬─────────┴───────────────────┘
                                     ▼
                     MLOps & Engineering ──► Ethics, Society & Careers
```

:::tip
**Study method that works:** for every lecture, (1) read once for intuition, (2) re-derive the key equation on paper without looking, (3) run and modify the code, (4) attempt the exercises, and (5) explain the idea to a friend in two minutes. If you cannot do step 5, return to step 1. This is how I recommend my own students study.
:::

## Tools you should install

- **Python 3** with `numpy`, `pandas`, `matplotlib`, `scikit-learn`.
- **PyTorch** (or TensorFlow/Keras) for deep learning.
- **Jupyter** or VS Code notebooks for experiments.
- **Git** for version control — from your very first project.
- A free cloud GPU notebook environment for heavier experiments.

```bash
python -m venv ai-course
source ai-course/bin/activate        # on Windows: ai-course\Scripts\activate
pip install numpy pandas matplotlib scikit-learn torch jupyter
```

:::exercise
1. Choose one application domain above and list which tracks of this course it needs.
2. Install the tools listed and run a "hello world" that trains a scikit-learn classifier on the Iris dataset.
3. Pick one research frontier and find a recent survey paper on it. Write a half-page summary.
:::

:::takeaway
- AI combines search/reasoning, probabilistic modelling and learning from data.
- ML paradigms: supervised, unsupervised, self-supervised, reinforcement and semi-supervised.
- Deep learning with transformers underlies most modern perception and language systems.
- Production AI is mostly engineering: data, evaluation, monitoring and governance.
:::
