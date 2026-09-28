=== POST ===
slug: model-based-reinforcement-learning
title: "Model-Based Reinforcement Learning: Learning and Planning with World Models"
category: reinforcement-learning
level: Advanced
tags: model-based rl, world models, dyna, planning, mbpo, dreamer
summary: Model-based agents learn a model of the environment and use it to plan or generate imagined experience. We cover Dyna, model-predictive control, model errors and ensembles, MBPO, and latent world models like Dreamer and MuZero.
---
Model-free agents learn purely from trial and error and often need millions of interactions — acceptable in a fast simulator, unacceptable for a real robot or a system affecting people. **Model-based RL** learns a **model** of how the world works — how states change and what rewards follow — and uses it to plan or to generate imagined experience. Done well, it is dramatically more **sample-efficient**. Done badly, the agent exploits the model's mistakes.

## What is a model?

A learned model predicts dynamics and rewards:

$$
\hat{s}_{t+1} \sim \hat{p}_\psi(s_{t+1} \mid s_t, a_t), \qquad \hat{r}_{t+1} = \hat{r}_\psi(s_t, a_t)
$$

It is trained by **supervised learning** on observed transitions — regression for deterministic dynamics, or probabilistic models (e.g. networks outputting Gaussian means and variances) to capture stochasticity.

## Using the model

### 1. Dyna: learning from imagined experience
Sutton's **Dyna** architecture (1990) interleaves:
- real experience → update the value function (direct RL) and update the model;
- **planning**: sample simulated transitions from the model and apply the same Q-learning update.

Each real step can drive many imagined updates.

```python
import random
import numpy as np

def dyna_q(env, episodes=50, n_planning=20, alpha=0.1, gamma=0.95, eps=0.1, seed=0):
    rng = random.Random(seed)
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    model = {}                                            # (s, a) -> (r, s2, terminal)
    for ep in range(episodes):
        s, _ = env.reset(seed=seed + ep); done = False
        while not done:
            a = env.action_space.sample() if rng.random() < eps else int(np.argmax(Q[s]))
            s2, r, term, trunc, _ = env.step(a); done = term or trunc
            Q[s, a] += alpha * (r + gamma * (0 if term else Q[s2].max()) - Q[s, a])   # direct RL
            model[(s, a)] = (r, s2, term)                                          # model learning
            for _ in range(n_planning):                                            # planning
                (ps, pa), (pr, ps2, pterm) = rng.choice(list(model.items()))
                Q[ps, pa] += alpha * (pr + gamma * (0 if pterm else Q[ps2].max()) - Q[ps, pa])
            s = s2
    return Q
# e.g. dyna_q(gym.make("FrozenLake-v1", is_slippery=False))
```

With planning steps, Dyna typically reaches good policies in far fewer real episodes than plain Q-learning.

### 2. Planning at decision time: model-predictive control (MPC)
At each step, use the model to simulate candidate action sequences over a short horizon, choose the best, execute **only the first action**, then replan. Sampling-based optimisers such as the **cross-entropy method** make this practical. MPC re-plans constantly, correcting for model errors — widely used in robotics and industrial control.

### 3. Tree search with a model
Games provide perfect models (the rules); search methods such as MCTS use them (next lecture).

## The central problem: model error

Learned models are imperfect, especially far from the training data. A planner optimising against the model will **exploit its errors**, finding actions that look great in imagination but fail in reality. Errors also **compound** over long imagined rollouts. Remedies:

- **Probabilistic ensembles**: train several models; disagreement measures uncertainty (epistemic). PETS (Chua et al., 2018) combined probabilistic ensembles with MPC and matched model-free performance with far fewer samples.
- **Short rollouts**: **MBPO** (Janner et al., 2019) generates only short model rollouts branching from real states and trains SAC on the mix of real and imagined data — trusting the model only near data.
- **Penalise uncertainty** in planning, or return to real data frequently.

## Latent world models

For high-dimensional observations like images, predicting future **pixels** is hard and wasteful. **World models** learn a compact **latent** state and dynamics in that space:

- **World Models** (Ha & Schmidhuber, 2018): a VAE for frames, an RNN for latent dynamics, and a small controller trained "in the dream".
- **Dreamer** (Hafner et al., 2020–2023): learns a recurrent state-space model from pixels and trains an actor–critic entirely on imagined latent trajectories; DreamerV3 used a single set of hyperparameters across diverse domains, including collecting diamonds in Minecraft from scratch.
- **MuZero** (Schrittwieser et al., 2020): learns a model that predicts only what matters for planning — rewards, values and policies — **not** observations, and plans with MCTS in the learned latent space; it matched AlphaZero in Go, chess and shogi without being given the rules, and set strong results on Atari.

## Model-based vs model-free

| | Model-free | Model-based |
|---|---|---|
| Sample efficiency | Lower | Higher |
| Asymptotic performance | Often higher (no model bias) | Can be limited by model errors |
| Computation per step | Lower | Higher (planning) |
| Transfer to new goals | Retrain | Model can be reused for new rewards |
| Interpretability | Low | Model predictions can be inspected |

:::note
Humans are strongly model-based: we imagine consequences before acting. Much current research — including large "foundation" world models trained on video — pursues general-purpose models of how the world evolves, which agents could use for planning and safe exploration.
:::

:::exercise
1. Compare Q-learning and Dyna-Q with 5, 20 and 50 planning steps on a deterministic maze; plot steps per episode.
2. Train an ensemble of 5 neural dynamics models on CartPole transitions and plot their prediction disagreement for in-distribution and unusual states.
3. Implement random-shooting MPC with a learned CartPole model and compare its performance with the true simulator as the model.
:::

:::takeaway
- Model-based RL learns dynamics and reward models and uses them to plan or imagine experience.
- Dyna mixes real and simulated updates; MPC replans at each step; tree search uses models in games.
- Planners exploit model errors — use ensembles, uncertainty and short rollouts (PETS, MBPO).
- Latent world models (Dreamer, MuZero) plan in learned compact spaces with high sample efficiency.
:::

=== POST ===
slug: alphago-alphazero
title: "AlphaGo, AlphaZero and MuZero: Search Meets Deep Learning"
category: reinforcement-learning
level: Advanced
tags: alphago, alphazero, muzero, mcts, self-play, games
summary: DeepMind's Go programs combined deep neural networks with Monte Carlo tree search and self-play. We trace AlphaGo's supervised and RL training, AlphaZero's tabula-rasa self-play, MuZero's learned model, and the lessons for AI.
---
In March 2016, DeepMind's **AlphaGo** defeated Lee Sedol, one of the world's strongest Go players, 4–1 in Seoul — a milestone many experts had expected to be at least a decade away. Go's enormous branching factor (about 250 legal moves per position) and the difficulty of evaluating positions had defeated classical game-tree search. AlphaGo and its successors combined **deep learning**, **reinforcement learning** and **Monte Carlo tree search (MCTS)**. Their design ideas — learned intuition guiding explicit search, and self-improvement through self-play — have influenced areas far beyond games.

## Why Go was hard

- Chess engines search deep trees with hand-crafted evaluation functions. In Go, the branching factor makes deep exhaustive search impossible, and nobody could write a good evaluation function: whether a group of stones is safe depends on subtle, global patterns.
- Humans rely on **intuition** (which moves look promising) and **judgement** (who is winning). AlphaGo learned both with neural networks.

## AlphaGo (2016)

Silver et al. (*Nature*, 2016) trained several networks operating on the 19×19 board:

1. **Supervised policy network** $p_\sigma$: a 13-layer CNN trained to predict expert human moves from about 30 million positions — reaching about 57% move-prediction accuracy.
2. **RL policy network** $p_\rho$: initialised from $p_\sigma$ and improved by **policy-gradient self-play** against earlier versions of itself.
3. **Value network** $v_\theta$: trained by regression to predict the game's winner from positions sampled from self-play games (one position per game to avoid overfitting correlated positions).
4. **Fast rollout policy**: a small, quick policy for simulating games to the end.

During play, **MCTS** used the policy network to prioritise promising moves and combined the value network with rollout outcomes to evaluate leaf positions. The famous move 37 in game 2 — a move human experts initially considered a mistake — showed the system finding strategies outside human convention.

## AlphaGo Zero (2017): no human data

AlphaGo Zero learned **tabula rasa** — from random play, knowing only the rules:

- A single **residual network** with two heads: a policy $\mathbf{p}$ and a value $v$, $(\mathbf{p}, v) = f_\theta(s)$.
- **No rollouts**: leaf evaluation used the value head only.
- **Self-play with MCTS as a policy-improvement operator**: at each move, MCTS guided by the current network produces visit counts $\boldsymbol{\pi}$ — a **stronger** policy than the raw network. The network is then trained to match those search probabilities and to predict the game outcome $z$:

$$
\mathcal{L} = (z - v)^2 - \boldsymbol{\pi}^\top\log\mathbf{p} + c\|\theta\|^2
$$

This loop — search improves the policy, the network learns from search, better networks make search stronger — is a form of **policy iteration** with MCTS as the improvement step. After about three days of training, AlphaGo Zero defeated the version that beat Lee Sedol by 100 games to 0.

## The search: PUCT

MCTS selects actions within the tree by

$$
a^* = \arg\max_a\left[Q(s, a) + c_{\text{puct}}\,P(s, a)\frac{\sqrt{\sum_bN(s, b)}}{1 + N(s, a)}\right]
$$

balancing the estimated value $Q$ with an exploration bonus weighted by the network's prior $P$. (See the MCTS lecture in the Foundations track.) Dirichlet noise at the root encourages exploration during self-play.

```python
import math

def puct_select(children, c_puct=1.5):
    """children: dict action -> {'N': visits, 'W': total value, 'P': prior}."""
    total = sum(ch["N"] for ch in children.values())
    def score(ch):
        q = ch["W"] / ch["N"] if ch["N"] else 0.0
        return q + c_puct * ch["P"] * math.sqrt(total + 1e-8) / (1 + ch["N"])
    return max(children, key=lambda a: score(children[a]))

print(puct_select({"a": {"N": 10, "W": 6.0, "P": 0.5}, "b": {"N": 1, "W": 0.2, "P": 0.3},
                   "c": {"N": 0, "W": 0.0, "P": 0.2}}))
```

## AlphaZero (2018): one algorithm, three games

The same algorithm, with minimal game-specific changes, learned **chess, shogi and Go** from scratch and defeated the strongest existing programs in each (Stockfish in chess, Elmo in shogi, AlphaGo Zero in Go) in the published matches. Its chess style — dynamic piece sacrifices, long-term positional play — attracted wide interest from grandmasters.

## MuZero (2020): learning the rules too

**MuZero** removed the need for a known simulator. It learns three functions: a **representation** mapping observations to a latent state, a **dynamics** function predicting the next latent state and reward, and a **prediction** function giving policy and value. MCTS runs entirely in the learned latent space. MuZero matched AlphaZero in board games and achieved strong results on Atari — bridging model-based planning and model-free learning.

## Beyond games

The AlphaZero/MuZero recipe has been applied to discovering faster matrix-multiplication algorithms (AlphaTensor), faster sorting routines adopted into a standard C++ library (AlphaDev), video-compression rate control, and chip floor-planning research. Its central lesson — **combine learned intuition with explicit search, and let the system improve by playing against itself** — also informs current work on reasoning in language models, where sampling and search at test time improve results.

:::note
These systems relied on perfect simulators (the game rules), clear win/loss rewards and enormous compute (thousands of TPUs for self-play in some runs). Real-world problems rarely offer all three — one reason why transferring AlphaZero's successes beyond games has required significant adaptation.
:::

:::exercise
1. Explain why MCTS visit counts are a better training target for the policy than the raw network output.
2. Implement AlphaZero-style self-play for tic-tac-toe or Connect Four with a small network (a simplified version is a classic course project).
3. Compare the roles of rollouts in AlphaGo and the value head in AlphaGo Zero. Why could Zero drop rollouts?
:::

:::takeaway
- AlphaGo combined supervised and RL policy networks, a value network and MCTS to defeat a top Go professional.
- AlphaGo Zero learned from scratch by self-play, using MCTS as a policy-improvement operator and a single policy–value network.
- AlphaZero generalised to chess and shogi; MuZero learned its own latent model for planning.
- The recipe — learned intuition + search + self-play — has found uses beyond games, given good simulators and compute.
:::

=== POST ===
slug: multi-agent-reinforcement-learning
title: "Multi-Agent Reinforcement Learning: Cooperation, Competition and Equilibria"
category: reinforcement-learning
level: Advanced
tags: multi-agent, marl, game theory, self-play, cooperation, ctde
summary: When many learning agents share an environment, each faces a moving target. We introduce Markov games, Nash equilibria, independent learners, centralised training with decentralised execution, self-play and emergent behaviour.
---
Traffic intersections with many autonomous vehicles, teams of warehouse robots, trading agents in markets, players in a strategy game, LLM agents negotiating on behalf of users — many real problems involve **multiple agents** acting simultaneously. **Multi-agent reinforcement learning (MARL)** studies how agents learn when their outcomes depend on each other's behaviour. It brings together RL and **game theory**, and introduces challenges absent in single-agent RL.

## Markov games

A **Markov game** (stochastic game) generalises the MDP to $N$ agents:

- a shared state $s$;
- each agent $i$ chooses an action $a^i$; the **joint action** $\mathbf{a} = (a^1, \dots, a^N)$ determines the transition $P(s' \mid s, \mathbf{a})$;
- each agent receives its own reward $r^i(s, \mathbf{a})$.

Settings:

| Setting | Rewards | Example |
|---|---|---|
| Fully cooperative | Shared reward | Robot team moving a heavy object |
| Fully competitive (zero-sum) | $r^1 = -r^2$ | Chess, Go, poker duels |
| Mixed / general-sum | Different, partly aligned | Traffic, markets, negotiation, social dilemmas |

With partial observability (each agent sees only local observations) we get **Dec-POMDPs**.

## Solution concepts

What is "optimal" when outcomes depend on others? Game theory offers **equilibria**:

- A **Nash equilibrium** is a joint policy where no agent can improve its expected return by changing its own policy unilaterally.
- In two-player **zero-sum** games, Nash equilibria correspond to **minimax** strategies — well defined and computable in principle.
- In general-sum games, there may be many equilibria, some much better for everyone than others (e.g. mutual cooperation vs mutual defection in the prisoner's dilemma).

**Social dilemmas** — situations where individually rational behaviour leads to collectively poor outcomes (overusing a shared resource) — are a major research theme, relevant to understanding cooperation among AI agents and humans.

## Challenges

1. **Non-stationarity**: from each agent's perspective, the environment changes as other agents learn — violating the stationary MDP assumption and destabilising learning.
2. **Credit assignment**: with a shared team reward, which agent's action caused success?
3. **Scalability**: the joint action space grows exponentially with the number of agents.
4. **Partial observability and communication**.
5. **Equilibrium selection** and **coordination**: agents must converge on compatible conventions (drive on the left or the right?).

## Approaches

### Independent learners
Each agent runs its own RL algorithm (e.g. independent Q-learning or independent PPO), treating others as part of the environment. Simple and scalable; surprisingly competitive in some benchmarks, but without convergence guarantees due to non-stationarity.

### Centralised training, decentralised execution (CTDE)
During training (often in simulation), use **global information**; at execution, each agent acts on local observations only.

- **MADDPG** (Lowe et al., 2017): each agent's critic sees all agents' observations and actions; actors use local observations.
- **Value decomposition** for cooperative tasks: **VDN** sums per-agent Q-values; **QMIX** combines them with a monotonic mixing network, so each agent's greedy local action is consistent with the joint greedy action.
- **MAPPO**: PPO with a centralised value function — a strong, simple baseline in cooperative benchmarks.

### Self-play and populations
For competitive games, agents train against copies of themselves (AlphaZero). To avoid cycles and overfitting to one opponent, train against **populations** or leagues of past and diverse agents — the approach behind DeepMind's **AlphaStar** (StarCraft II, reaching Grandmaster level) and OpenAI Five (Dota 2).

### Game-theoretic methods
**Counterfactual regret minimisation (CFR)** converges to equilibria in imperfect-information games; combined with search and learning it produced superhuman poker agents (Libratus, and Pluribus for six-player no-limit Texas hold'em).

## A tiny example: independent learners in a coordination game

```python
import numpy as np

# Two agents choose A or B. Payoff 1 each if they match, 0 otherwise (a coordination game).
payoff = np.array([[1, 0], [0, 1]])
rng = np.random.default_rng(0)
Q1, Q2, alpha, eps = np.zeros(2), np.zeros(2), 0.1, 0.1

for t in range(2000):
    a1 = rng.integers(2) if rng.random() < eps else int(np.argmax(Q1))
    a2 = rng.integers(2) if rng.random() < eps else int(np.argmax(Q2))
    r = payoff[a1, a2]
    Q1[a1] += alpha * (r - Q1[a1]); Q2[a2] += alpha * (r - Q2[a2])
print("agent 1 prefers", "AB"[int(np.argmax(Q1))], "| agent 2 prefers", "AB"[int(np.argmax(Q2))], Q1.round(2), Q2.round(2))
```

Run it with different seeds: the agents usually settle on a shared **convention** — sometimes A, sometimes B — illustrating equilibrium selection.

## Emergent behaviour

MARL can produce surprising emergent strategies. In OpenAI's hide-and-seek environment (Baker et al., 2020), teams discovered a sequence of strategies and counter-strategies — building shelters, using ramps, "box surfing" — none explicitly rewarded. Emergent communication, tool use and social conventions are active research areas.

:::note
As LLM-based agents increasingly interact with each other — negotiating, trading, coordinating — multi-agent dynamics become a practical safety concern: collusion, escalation, and exploitation between agents. Concepts from MARL and game theory are essential for designing systems whose collective behaviour remains beneficial.
:::

:::exercise
1. Change the payoff matrix to a prisoner's dilemma and run independent learners. What outcome emerges?
2. Implement QMIX's monotonic mixing idea for two agents with a small cooperative gridworld (a simplified version).
3. Explain why independent Q-learning violates the Markov assumption from each agent's perspective.
:::

:::takeaway
- Markov games extend MDPs to many agents; settings are cooperative, competitive or mixed.
- Nash equilibria define stable joint policies; social dilemmas show individual and collective interests can conflict.
- Non-stationarity, credit assignment and scale are key challenges.
- Approaches include independent learners, CTDE (MADDPG, QMIX, MAPPO), self-play leagues and CFR.
:::

=== POST ===
slug: imitation-learning-inverse-rl
title: "Imitation Learning and Inverse Reinforcement Learning"
category: reinforcement-learning
level: Advanced
tags: imitation learning, behavioral cloning, dagger, inverse rl, gail
summary: When rewards are hard to specify but demonstrations are available, agents can learn by imitation. We cover behavioural cloning and its compounding errors, DAgger, inverse RL, maximum-entropy IRL and adversarial imitation (GAIL).
---
Writing a reward function for "drive safely and courteously" or "fold laundry neatly" is extremely difficult. Showing someone how to do it is easy. **Imitation learning** trains agents from **expert demonstrations** rather than hand-designed rewards. It underpins autonomous-driving research, robot manipulation, game agents — and, in a sense, supervised fine-tuning of language models on human-written responses.

## Behavioural cloning (BC)

The simplest approach: treat demonstrations as a supervised dataset of (state, action) pairs and train a policy by classification or regression:

$$
\min_\theta\;\mathbb{E}_{(s, a) \sim \mathcal{D}_{\text{expert}}}\big[-\log\pi_\theta(a \mid s)\big]
$$

It is simple, needs no environment interaction, and works well when demonstrations are plentiful and cover the situations the agent will face. ALVINN (1989) steered a vehicle with a neural network trained on human driving; modern robot learning uses BC with expressive policies (e.g. diffusion policies or transformers) and large demonstration datasets.

## The compounding-error problem

BC violates the i.i.d. assumption. The expert rarely makes mistakes, so demonstrations contain few examples of **recovering** from errors. When the learned policy makes a small mistake, it drifts into states **unlike** the training data, where it makes bigger mistakes, drifting further — errors compound. Ross and Bagnell showed that the expected cost of BC can grow **quadratically** with the task horizon $T$ (as $O(\epsilon T^2)$ for per-step error rate $\epsilon$), whereas an ideal learner's cost grows linearly.

## DAgger: dataset aggregation

Ross, Gordon and Bagnell (2011) proposed **DAgger**:

1. Train a policy on expert data.
2. Run **the learner's** policy to collect the states it actually visits.
3. Ask the **expert** to label the correct action for those states.
4. Aggregate into the dataset and retrain; repeat.

DAgger trains on the learner's own state distribution, fixing compounding errors and achieving linear-in-horizon error bounds. The cost: an expert available to label on demand (possible in simulation or with a scripted planner; harder with humans).

```python
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def dagger(env, expert, n_iters=5, rollouts=10):
    X, y = [], []
    policy = None
    for it in range(n_iters):
        for _ in range(rollouts):
            s, _ = env.reset(); done = False
            while not done:
                X.append(s); y.append(expert(s))                     # expert labels every visited state
                a = expert(s) if policy is None else int(policy.predict([s])[0])   # learner acts
                s, _, term, trunc, _ = env.step(a); done = term or trunc
        policy = KNeighborsClassifier(5).fit(np.array(X), np.array(y))  # retrain on aggregated data
    return policy
```

## Inverse reinforcement learning (IRL)

Instead of copying actions, **infer the reward function** that the expert appears to optimise, then find a policy for that reward with RL (Ng & Russell, 2000; Abbeel & Ng, 2004). Motivations:

- A reward is a compact, **transferable** description of the task — it generalises to new situations and dynamics better than copied actions.
- Understanding **why** the expert acts, not just **what** they do.

The problem is **ill-posed**: many rewards explain the same behaviour (a zero reward makes every policy optimal). **Maximum-entropy IRL** (Ziebart et al., 2008) resolves ambiguity by assuming the expert chooses trajectories with probability

$$
P(\tau) \propto \exp\big(R_\psi(\tau)\big)
$$

and fits $\psi$ by maximum likelihood — preferring the least committed explanation consistent with demonstrations. It was used, for example, to model taxi drivers' route choices.

## Adversarial imitation: GAIL

**Generative Adversarial Imitation Learning** (Ho & Ermon, 2016) links IRL with GANs. A **discriminator** $D(s, a)$ learns to distinguish expert state–action pairs from the agent's; the agent (trained with RL, e.g. TRPO/PPO) receives reward for fooling the discriminator, such as $-\log(1 - D(s, a))$. GAIL matches the expert's **state–action distribution** directly, avoiding compounding errors, with far fewer demonstrations than BC — though it requires environment interaction.

## Comparing approaches

| Method | Needs environment interaction | Needs interactive expert | Handles compounding errors | Recovers reward |
|---|---|---|---|---|
| Behavioural cloning | No | No | No | No |
| DAgger | Yes | Yes | Yes | No |
| IRL (e.g. MaxEnt) | Yes (RL inner loop) | No | Yes | Yes |
| GAIL | Yes | No | Yes | Implicitly |

## Connections and cautions

- **RLHF** is related: rather than demonstrations, it learns a reward from **preferences** — another way of inferring human intent.
- **Offline RL** (next lecture) learns from logged data that may include non-expert behaviour, using rewards.

:::warning
Imitation copies the demonstrator — including their mistakes and biases. A model imitating historical human decisions (e.g. case prioritisation or hiring) will reproduce past discrimination. Before imitating human decisions in consequential domains, examine whether the demonstrated behaviour is actually what we want.
:::

:::exercise
1. Train behavioural cloning on 5, 20 and 100 expert CartPole episodes (use a trained PPO agent as the expert) and measure performance.
2. Implement DAgger with the trained agent as expert and compare with BC at equal numbers of expert labels.
3. Explain why IRL is ill-posed and how the maximum-entropy principle resolves the ambiguity.
:::

:::takeaway
- Imitation learning learns from demonstrations when rewards are hard to specify.
- Behavioural cloning is simple supervised learning but suffers compounding errors ($O(\epsilon T^2)$).
- DAgger queries the expert on the learner's own states to fix distribution shift.
- IRL infers rewards (MaxEnt IRL); GAIL matches expert behaviour adversarially; imitation inherits the demonstrator's biases.
:::

=== POST ===
slug: offline-reinforcement-learning
title: "Offline Reinforcement Learning: Learning from Logged Data"
category: reinforcement-learning
level: Advanced
tags: offline rl, batch rl, distribution shift, cql, iql, off-policy evaluation
summary: In many domains, trial-and-error exploration is unsafe or impossible, but logged data exists. Offline RL learns policies from fixed datasets. We explain the distributional shift problem and methods such as BCQ, CQL and IQL, plus off-policy evaluation.
---
A hospital cannot let an RL agent experiment with treatments on patients to see what happens. A public agency cannot randomly vary how it allocates assistance just to explore. Yet both have years of **logged data** — decisions taken and outcomes observed. **Offline RL** (also called batch RL) aims to learn good policies purely from such fixed datasets, without any new interaction. It promises to bring RL to healthcare, education, logistics and recommendation — but it is fundamentally harder than it first appears.

## The setting

Given a dataset $\mathcal{D} = \{(s_i, a_i, r_i, s'_i)\}$ collected by some **behaviour policy** $\pi_\beta$ (humans, an old system, a mix), learn a policy $\pi$ that performs well when deployed — without collecting more data.

## Why not just run off-policy RL on the dataset?

Q-learning is off-policy, so in principle it can learn from any data. In practice, naive offline Q-learning or DQN on a fixed dataset **fails badly**. The reason is **distributional shift** combined with the max in the Bellman target:

$$
y = r + \gamma\max_{a'}Q(s', a')
$$

The max considers **all** actions, including actions that never appear in the data for state $s'$. Their Q-values are pure extrapolation by the function approximator — often wildly overestimated. The policy then prefers exactly these **out-of-distribution (OOD) actions**, and because no new data is collected, the errors are never corrected. In online RL, trying an overestimated action reveals the truth; offline, the agent can never check.

## Approach 1: constrain the policy to the data

Keep the learned policy close to the behaviour policy:

- **BCQ** (Batch-Constrained Q-learning, Fujimoto et al., 2019): a generative model proposes actions similar to those in the data; the policy chooses only among them.
- **TD3+BC** (Fujimoto & Gu, 2021): add a behavioural-cloning term to the TD3 actor loss, $\max_\pi\big[\lambda Q(s, \pi(s)) - (\pi(s) - a)^2\big]$ — remarkably simple and strong.
- **BRAC / BEAR**: penalise divergence (KL, MMD) from the behaviour policy.

## Approach 2: pessimistic values

Make OOD actions look **bad** rather than good:

- **Conservative Q-Learning (CQL)** (Kumar et al., 2020) adds a regulariser that pushes down Q-values on actions sampled from the learned policy (or all actions via log-sum-exp) and pushes up Q-values on dataset actions:

$$
\min_Q\;\alpha\Big(\mathbb{E}_{s \sim \mathcal{D}}\Big[\log\sum_a\exp Q(s, a)\Big] - \mathbb{E}_{(s,a) \sim \mathcal{D}}\big[Q(s, a)\big]\Big) + \tfrac{1}{2}\,\mathbb{E}_{\mathcal{D}}\big[(Q - \mathcal{B}Q)^2\big]
$$

  The learned Q-function lower-bounds the true value of the policy — a principled form of **pessimism**.

## Approach 3: avoid querying OOD actions at all

- **Implicit Q-Learning (IQL)** (Kostrikov et al., 2022) never evaluates actions outside the dataset. It fits a value function with **expectile regression** on dataset Q-values (approximating a max over **in-distribution** actions), and extracts a policy by **advantage-weighted regression**: behavioural cloning weighted by $\exp(\beta A(s, a))$ — imitate the good actions in the data more strongly.

```python
import torch

def expectile_loss(diff, tau=0.7):
    """Asymmetric L2: weight positive errors by tau, negative by (1 - tau)."""
    weight = torch.where(diff > 0, tau, 1 - tau)
    return (weight * diff.pow(2)).mean()

def awr_policy_loss(log_prob_data_actions, advantages, beta=3.0, max_weight=100.0):
    """Advantage-weighted regression: clone dataset actions, weighted by exp(beta * A)."""
    w = torch.exp(beta * advantages).clamp(max=max_weight)
    return -(w.detach() * log_prob_data_actions).mean()

diff = torch.randn(256)                     # Q(s, a) - V(s) on dataset pairs
print(expectile_loss(diff).item(), awr_policy_loss(torch.randn(256), torch.randn(256) * 0.1).item())
```

## Off-policy evaluation (OPE)

Before deploying a learned policy, estimate its value from logged data:

- **Importance sampling** estimators (high variance for long horizons);
- **Direct methods** (fit a model or Q-function and evaluate);
- **Doubly robust** estimators combining both.

OPE is hard and estimates can be badly wrong; in high-stakes settings, follow OPE with carefully monitored small-scale pilots.

## Data matters most

Offline RL cannot find good actions that the data never tried. Dataset **coverage** and **quality** determine what is achievable:

- Expert-only data → behavioural cloning may do as well.
- Diverse, "medium" data with varied behaviour → offline RL can outperform the behaviour policy by stitching together good parts of different trajectories.
- Confounded data (decisions based on information not recorded) → estimates can be biased, a problem shared with causal inference.

:::warning
In domains such as healthcare, offline RL policies can recommend actions that look optimal because of **confounding** or rare, unrepresentative data — for example, recommending an intervention that historically was only given to less severe patients. Treat offline-RL recommendations as hypotheses for expert review and prospective evaluation, not as decisions.
:::

:::exercise
1. Collect a dataset of 50,000 CartPole transitions from a mediocre policy, train DQN offline, and compare its predicted Q-values with its actual returns.
2. Add a CQL-style regulariser and compare performance and Q-value estimates.
3. Use the D4RL-style benchmarks (or Minari datasets) to compare behavioural cloning, TD3+BC and IQL on one task.
:::

:::takeaway
- Offline RL learns from fixed logged data without further interaction — essential where exploration is unsafe.
- Naive off-policy RL fails because the max over actions exploits overestimated out-of-distribution actions.
- Remedies: constrain the policy to the data (BCQ, TD3+BC), pessimistic values (CQL), or avoid OOD queries (IQL).
- Off-policy evaluation is difficult; data coverage and confounding limit what offline RL can achieve.
:::

=== POST ===
slug: rl-in-the-real-world-robotics
title: "Reinforcement Learning in the Real World: Robotics, Sim-to-Real and Safety"
category: reinforcement-learning
level: Advanced
tags: robotics, sim-to-real, domain randomization, safe rl, real-world rl
summary: Games are forgiving; the real world is not. We examine the challenges of deploying RL — sample efficiency, safety, reward design, partial observability — and the techniques that make it work: simulation, domain randomisation, safe RL and human oversight.
---
Reinforcement learning's most famous successes happened in games and simulators, where an agent can fail millions of times at no cost. The physical and social world is different: robots break, experiments are slow, rewards are unclear and mistakes can hurt people. Yet RL has achieved remarkable real-world results — dexterous robot hands, agile legged locomotion, data-centre cooling, and aligning language models. This closing lecture of the track examines what it takes to make RL work outside the simulator.

## Challenges of real-world RL

Dulac-Arnold et al. (2019) catalogued the main challenges:

1. **Sample efficiency**: real interactions are slow and expensive.
2. **Safety**: exploration must not cause damage or harm.
3. **Reward specification**: rewards are hard to define and easy to game.
4. **Partial observability and noise**: sensors are imperfect; delays exist.
5. **High-dimensional, continuous** state and action spaces.
6. **Non-stationarity**: the world changes (wear, seasons, user behaviour).
7. **Constraints**: physical limits, regulations, budgets.
8. **Explainability** and the need for operators to trust the system.
9. **Learning from limited logged data** (offline RL).

## Simulation and sim-to-real transfer

The dominant strategy is to train in **simulation** and transfer to reality. The difference between simulator and world — the **reality gap** — causes policies to fail on hardware. Techniques:

- **Domain randomisation**: randomise physical parameters (masses, friction, motor strength, delays), visual appearance (textures, lighting, camera position) and noise during training, so the real world looks like just another variation. OpenAI's Dactyl (2018–2019) used massive domain randomisation to train a robot hand to manipulate a cube and solve a Rubik's Cube.
- **System identification**: measure real parameters and calibrate the simulator.
- **Domain adaptation**: adapt the policy or its perception with a small amount of real data.
- **Teacher–student training**: a "teacher" policy with privileged simulator information (exact terrain, friction) trains a "student" that uses only real sensors — used for robust quadruped locomotion over rough terrain.
- **Massively parallel GPU simulation** (e.g. Isaac Gym-style simulators) lets legged robots learn to walk in minutes of wall-clock time.

```python
import numpy as np

class RandomizedPendulumParams:
    """Sample physics parameters each episode for domain randomisation."""
    def __init__(self, rng=None):
        self.rng = rng or np.random.default_rng()
    def sample(self):
        return {
            "mass": self.rng.uniform(0.8, 1.2),           # ±20% around nominal
            "length": self.rng.uniform(0.9, 1.1),
            "friction": self.rng.uniform(0.0, 0.1),
            "action_delay_steps": int(self.rng.integers(0, 3)),
            "obs_noise_std": self.rng.uniform(0.0, 0.02),
        }

params = RandomizedPendulumParams(np.random.default_rng(0))
print([params.sample()["mass"].round(3) for _ in range(5)])
# At every env.reset(), apply params.sample() to the simulator before the episode.
```

## Safe reinforcement learning

Safety must be designed in:

- **Constrained MDPs**: maximise reward subject to expected cost limits, $\mathbb{E}[\sum_t\gamma^tc_t] \le d$, solved with Lagrangian methods or constrained policy optimisation (CPO).
- **Safety layers and shields**: a verified filter overrides actions that would violate constraints (e.g. keep a drone within a geofence).
- **Conservative exploration**: explore only within a known-safe region; expand it cautiously.
- **Learning from demonstrations and offline data** to avoid dangerous random exploration.
- **Human oversight**: emergency stops, approval of new policies, gradual rollout.

## Reward design in practice

- Start from the **true objective** and measurable outcomes; avoid rewarding proxies that can be gamed.
- Combine sparse task rewards with carefully designed **shaping** (potential-based shaping preserves optimal policies — Ng et al., 1999).
- Add explicit penalties for unsafe or undesirable side effects, and test for specification gaming in simulation before deployment.
- When a reward is too hard to write, learn it from **demonstrations** (IRL) or **preferences** (RLHF).

## Success stories

- **Robotics**: dexterous in-hand manipulation (Dactyl), robust quadruped and humanoid locomotion trained in simulation, robot learning from large demonstration datasets combined with RL fine-tuning.
- **Industrial control**: DeepMind reported substantial energy savings in data-centre cooling using RL-based recommendations and later autonomous control with safety constraints; RL has been explored for controlling plasma shapes in a tokamak fusion reactor (trained in simulation).
- **Recommendation and operations**: contextual bandits and RL for personalisation, inventory and logistics.
- **Language models**: RLHF and RL with verifiable rewards — perhaps RL's most widespread deployment today.

## Responsible deployment

:::warning
RL systems optimise relentlessly for their reward, adapt to users and environments, and can behave unexpectedly in novel situations. In domains affecting people — healthcare, education, social services, finance — prefer:
- decision support over full autonomy;
- conservative, constrained policies with human override;
- extensive offline evaluation, then small monitored pilots;
- rewards reviewed by domain experts and affected communities;
- continuous monitoring for drift, gaming and unintended harm.
:::

:::exercise
1. Train PPO on Pendulum with fixed physics, then evaluate it on a pendulum with 30% heavier mass. Retrain with domain randomisation and compare robustness.
2. Formulate a constrained MDP for a delivery drone (reward: deliveries; cost: battery depletion below a threshold, no-fly zones).
3. Choose a real-world process you know (e.g. irrigation scheduling) and write a risk assessment for applying RL to it, including reward-gaming scenarios.
:::

:::takeaway
- Real-world RL faces sample-efficiency, safety, reward, observability and non-stationarity challenges.
- Simulation with domain randomisation, system identification and teacher–student training bridges the reality gap.
- Safe RL uses constraints, shields, conservative exploration, demonstrations and human oversight.
- Successes span robotics, industrial control and LLM alignment; deploy conservatively where people are affected.
:::
