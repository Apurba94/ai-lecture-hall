=== POST ===
slug: convolutional-neural-networks-intro
title: Convolutional Neural Networks: The Core Ideas
category: deep-learning
level: Beginner
tags: cnn, convolution, weight sharing, feature maps, computer vision
summary: Convolutions exploit the structure of images through local connectivity, weight sharing and translation equivariance. We define the convolution operation, count parameters, and build a CNN that learns hierarchical features.
---
An image of $224 \times 224$ pixels with three colour channels has 150,528 input values. A fully connected layer with just 1,000 hidden units would need 150 million weights — and it would treat a cat in the top-left corner as unrelated to the same cat in the bottom-right. **Convolutional neural networks (CNNs)** solve both problems by building knowledge about images directly into the architecture. They were the engine of the deep learning revolution in computer vision.

## Three structural priors

1. **Local connectivity** — each unit looks only at a small neighbourhood (its **receptive field**), because nearby pixels are strongly related and meaningful patterns (edges, textures) are local.
2. **Weight sharing** — the same small filter is applied at every position, because a pattern useful in one location is useful everywhere.
3. **Translation equivariance** — shifting the input shifts the feature map by the same amount. Combined with pooling, this gives approximate **translation invariance**.

These priors massively reduce parameters and improve generalisation.

## The convolution operation

For a 2-D input $X$ and a $k \times k$ kernel $K$, the output feature map is

$$
Y[i, j] = \sum_{u=0}^{k-1}\sum_{v=0}^{k-1}K[u, v]\,X[i + u,\, j + v] + b
$$

(Strictly speaking, this is **cross-correlation**; deep learning libraries call it convolution because the kernel is learned, so flipping is irrelevant.)

With $C_{\text{in}}$ input channels and $C_{\text{out}}$ output channels, each output channel has its own kernel spanning all input channels:

$$
Y_c[i, j] = \sum_{c'=1}^{C_{\text{in}}}\sum_{u, v}K_{c, c'}[u, v]\,X_{c'}[i + u, j + v] + b_c
$$

**Parameter count** of a conv layer: $C_{\text{out}} \times C_{\text{in}} \times k \times k + C_{\text{out}}$. A $3 \times 3$ conv from 64 to 128 channels has $128 \times 64 \times 9 + 128 = 73{,}856$ parameters — independent of image size.

## What filters learn

Hand-designed filters illustrate the idea. The kernel

$$
\begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}
$$

(the Sobel operator) responds to vertical edges. A CNN **learns** its kernels by backpropagation. Visualisations of trained networks consistently show:

- **first layer**: oriented edges, colour blobs, Gabor-like patterns;
- **middle layers**: textures, corners, repeated motifs;
- **deep layers**: object parts — eyes, wheels, windows — and whole objects.

This hierarchy emerges because each layer's receptive field grows: stacking two $3 \times 3$ convs gives a $5 \times 5$ receptive field; three give $7 \times 7$.

## A typical CNN

```text
Input image
→ [Conv → BatchNorm → ReLU] × 2 → Pool     (low-level features, high resolution)
→ [Conv → BatchNorm → ReLU] × 2 → Pool     (more channels, lower resolution)
→ [Conv → BatchNorm → ReLU] × 2 → Global average pool
→ Linear → class logits
```

As depth increases, spatial resolution decreases and the number of channels increases — trading "where" information for "what" information.

## Building one in PyTorch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

class SmallCNN(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        def block(cin, cout):
            return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                                 nn.BatchNorm2d(cout), nn.ReLU(inplace=True))
        self.features = nn.Sequential(
            block(1, 32), block(32, 32), nn.MaxPool2d(2),      # 28 -> 14
            block(32, 64), block(64, 64), nn.MaxPool2d(2),     # 14 -> 7
            block(64, 128), nn.AdaptiveAvgPool2d(1))           # 7 -> 1
        self.head = nn.Linear(128, n_classes)
    def forward(self, x):
        return self.head(self.features(x).flatten(1))

model = SmallCNN()
print("parameters:", sum(p.numel() for p in model.parameters()))

tf = transforms.ToTensor()
train = datasets.FashionMNIST(".", train=True, download=True, transform=tf)
loader = torch.utils.data.DataLoader(train, batch_size=128, shuffle=True)
opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
model.train()
for epoch in range(2):
    for x, y in loader:
        opt.zero_grad(); loss = F.cross_entropy(model(x), y); loss.backward(); opt.step()
    print(f"epoch {epoch}: last batch loss {loss.item():.3f}")
```

With about 100,000 parameters, this network typically exceeds 90% accuracy on Fashion-MNIST after a couple of epochs — far better than an MLP of similar size.

## Convolution variants you will meet

- **1×1 convolutions** — mix channels at each position without spatial extent; used to change channel counts cheaply (bottlenecks).
- **Dilated (atrous) convolutions** — spread kernel taps apart to enlarge the receptive field without more parameters; used in segmentation.
- **Depthwise separable convolutions** — a per-channel spatial conv followed by a 1×1 conv; dramatically cheaper (MobileNet).
- **Transposed convolutions** — learnable upsampling for segmentation and generative models.
- **1-D and 3-D convolutions** — for audio/time series and video/volumetric medical scans.

:::note
Convolution is not only for images. Any data with a grid structure and local correlations — audio spectrograms, genomic sequences, time series, even text — can benefit. And transformers applied to images (Vision Transformers) learn similar local patterns in early layers, suggesting CNNs' priors capture something real about visual data.
:::

:::exercise
1. Compute the output size and parameter count of a conv layer with 3 input channels, 16 output channels, $5 \times 5$ kernels, applied to a $32 \times 32$ image with no padding.
2. Apply the Sobel kernel manually to a grayscale image with `F.conv2d` and display the result.
3. Replace the CNN above with an MLP of similar parameter count and compare accuracy on Fashion-MNIST.
:::

:::takeaway
- CNNs encode local connectivity, weight sharing and translation equivariance.
- Parameters depend on kernel size and channels, not image size.
- Stacked convolutions learn hierarchical features with growing receptive fields.
- 1×1, dilated, depthwise-separable and transposed convolutions extend the toolkit.
:::

=== POST ===
slug: pooling-stride-padding
title: Padding, Stride, Pooling and Receptive Fields
category: deep-learning
level: Beginner
tags: cnn, padding, stride, pooling, receptive field
summary: The geometry of convolutional layers determines output sizes, computational cost and what each unit can see. We derive the output-size formula, compare pooling types, and compute receptive fields.
---
Designing a CNN requires fluency with a few geometric knobs — padding, stride, pooling and dilation. They determine the shape of every feature map, the computational cost, and the **receptive field** of each unit (how much of the input it can see). Shape mismatches are among the most common errors in CNN code, so let us master the arithmetic.

## The output-size formula

For input size $n$ (height or width), kernel size $k$, padding $p$, stride $s$ and dilation $d$:

$$
n_{\text{out}} = \left\lfloor\frac{n + 2p - d(k - 1) - 1}{s}\right\rfloor + 1
$$

With $d = 1$ this simplifies to $\left\lfloor\frac{n + 2p - k}{s}\right\rfloor + 1$.

| Input | Kernel | Padding | Stride | Output |
|---|---|---|---|---|
| 32 | 3 | 0 | 1 | 30 |
| 32 | 3 | 1 | 1 | 32 ("same") |
| 32 | 3 | 1 | 2 | 16 |
| 224 | 7 | 3 | 2 | 112 (ResNet stem) |
| 28 | 5 | 0 | 1 | 24 |

## Padding

Without padding ("valid" convolution), each layer shrinks the feature map by $k - 1$ and border pixels are used by fewer output units. **Zero padding** with $p = (k - 1)/2$ for odd $k$ keeps the size unchanged ("same" padding), making it easy to stack many layers. Alternatives such as reflection padding reduce border artefacts in image generation.

## Stride

A stride of $s$ moves the kernel $s$ pixels at a time, downsampling the output by roughly $s$ in each dimension. **Strided convolutions** are a learnable alternative to pooling and are common in modern architectures. Downsampling reduces computation in later layers and enlarges receptive fields.

## Pooling

Pooling summarises each local window with a fixed function:

- **Max pooling** — keeps the strongest activation; provides a little translation invariance ("the feature is present somewhere in this window").
- **Average pooling** — smooths.
- **Global average pooling (GAP)** — averages each channel over the entire map, producing one number per channel. Introduced in Network-in-Network and used in GoogLeNet and ResNet, it replaces huge fully connected layers, drastically reducing parameters and allowing variable input sizes.

Pooling has no parameters. A $2 \times 2$ max pool with stride 2 halves height and width.

```python
import torch
import torch.nn as nn

x = torch.randn(1, 3, 224, 224)
layers = [
    ("conv7 s2 p3", nn.Conv2d(3, 64, 7, stride=2, padding=3)),
    ("maxpool3 s2 p1", nn.MaxPool2d(3, stride=2, padding=1)),
    ("conv3 p1", nn.Conv2d(64, 64, 3, padding=1)),
    ("conv3 s2 p1", nn.Conv2d(64, 128, 3, stride=2, padding=1)),
    ("dilated conv3 d2 p2", nn.Conv2d(128, 128, 3, padding=2, dilation=2)),
    ("global avg pool", nn.AdaptiveAvgPool2d(1)),
]
for name, layer in layers:
    x = layer(x)
    print(f"{name:<20} -> {tuple(x.shape)}")
```

## Receptive fields

The receptive field is the region of the input that can influence a unit. For a stack of layers with kernel sizes $k_l$ and strides $s_l$, it grows as

$$
r_l = r_{l-1} + (k_l - 1)\prod_{i=1}^{l-1}s_i, \qquad r_0 = 1
$$

Strides early in the network multiply the growth of all later layers. Examples:

- Three stacked $3 \times 3$ convs with stride 1: $r = 1 + 2 + 2 + 2 = 7$.
- A $3 \times 3$ conv after a stride-2 layer adds $2 \times 2 = 4$ instead of 2.

:::note
Why do VGG and ResNet use stacks of $3 \times 3$ convolutions instead of large kernels? Two stacked $3 \times 3$ layers have the receptive field of one $5 \times 5$ layer but use $2 \times 9 = 18$ weights per channel pair instead of 25, and add an extra non-linearity. Three $3 \times 3$ layers match a $7 \times 7$ with 27 instead of 49 weights.
:::

The **effective** receptive field is smaller than the theoretical one: Luo et al. (2016) showed the influence of input pixels is roughly Gaussian-shaped, concentrated in the centre. Tasks that need global context (segmentation of large objects, scene understanding) therefore benefit from dilated convolutions, pyramid pooling or attention.

## Dilation

A dilated kernel inserts gaps between its taps. A $3 \times 3$ kernel with dilation 2 covers a $5 \times 5$ area with 9 weights. Stacking dilations 1, 2, 4, 8 grows the receptive field exponentially while keeping resolution — the design of WaveNet for audio and DeepLab for segmentation. Watch for **gridding artefacts** if dilation rates share common factors.

## Trade-offs in downsampling

Downsampling saves computation and increases receptive field but discards spatial detail. Classification tolerates it well; dense prediction (segmentation, keypoints) needs detail back, which is why those architectures use **skip connections** from high-resolution layers (U-Net, FPN).

Standard downsampling can also break shift invariance through **aliasing**: a one-pixel input shift can change the output noticeably. Blurring before subsampling ("anti-aliased CNNs") improves consistency.

:::exercise
1. Compute every feature-map size in VGG-16 for a $224 \times 224$ input.
2. Compute the theoretical receptive field of the network defined in the code above.
3. Replace max pooling with stride-2 convolutions in a small CNN and compare accuracy and parameter count.
:::

:::takeaway
- Output size $= \lfloor(n + 2p - d(k-1) - 1)/s\rfloor + 1$ — memorise it.
- "Same" padding keeps size; stride and pooling downsample; global average pooling replaces big dense layers.
- Receptive fields grow with depth, multiplied by earlier strides; stacked $3\times3$ convs are efficient.
- Dilation enlarges receptive fields without losing resolution.
:::

=== POST ===
slug: recurrent-neural-networks
title: Recurrent Neural Networks: Modelling Sequences
category: deep-learning
level: Intermediate
tags: rnn, sequences, backpropagation through time, language modeling
summary: Sequences need memory. RNNs carry a hidden state through time with shared weights. We define the vanilla RNN, unroll it, derive backpropagation through time, and see why long dependencies are hard.
---
Text, speech, sensor streams, stock prices, DNA — much of the world's data comes as **sequences** where order matters and inputs have variable length. A feedforward network takes a fixed-size input and has no memory. **Recurrent neural networks (RNNs)** process one element at a time while maintaining a **hidden state** that summarises everything seen so far. Although transformers now dominate many sequence tasks, RNN concepts — hidden state, unrolling, BPTT, gating — are foundational and are experiencing a revival in efficient modern architectures.

## The vanilla RNN

At each time step $t$, with input $\mathbf{x}_t$ and previous hidden state $\mathbf{h}_{t-1}$:

$$
\mathbf{h}_t = \tanh\big(\mathbf{W}_{hh}\mathbf{h}_{t-1} + \mathbf{W}_{xh}\mathbf{x}_t + \mathbf{b}_h\big), \qquad \hat{\mathbf{y}}_t = \mathbf{W}_{hy}\mathbf{h}_t + \mathbf{b}_y
$$

The **same weights** are used at every time step — weight sharing across time, analogous to convolution's sharing across space. This lets the RNN handle sequences of any length with a fixed number of parameters.

## Unrolling

Conceptually, we "unroll" the loop into a deep feedforward network with one layer per time step, all layers sharing weights:

```text
x1 → [RNN] → h1 → [RNN] → h2 → [RNN] → h3 → ...
             ↓             ↓             ↓
             y1            y2            y3
```

An RNN processing a sequence of length 100 is a 100-layer-deep network — which is why the vanishing-gradient problem hits RNNs especially hard.

## Sequence task patterns

| Pattern | Example | Output used |
|---|---|---|
| Many-to-one | Sentiment classification | Final hidden state |
| One-to-many | Image captioning | Sequence generated from one input |
| Many-to-many (aligned) | Part-of-speech tagging | Output at every step |
| Many-to-many (unaligned) | Translation | Encoder–decoder (next lectures) |

## Backpropagation through time (BPTT)

Training applies backpropagation to the unrolled graph. The total loss is the sum over time steps, and because weights are shared, the gradient for $\mathbf{W}_{hh}$ sums contributions from every step:

$$
\frac{\partial L}{\partial\mathbf{W}_{hh}} = \sum_{t}\sum_{k \le t}\frac{\partial L_t}{\partial\mathbf{h}_t}\left(\prod_{j=k+1}^{t}\frac{\partial\mathbf{h}_j}{\partial\mathbf{h}_{j-1}}\right)\frac{\partial^+\mathbf{h}_k}{\partial\mathbf{W}_{hh}}
$$

where $\partial^+$ denotes the immediate (direct) partial derivative. The crucial term is the product of Jacobians

$$
\frac{\partial\mathbf{h}_j}{\partial\mathbf{h}_{j-1}} = \text{diag}\big(1 - \mathbf{h}_j^2\big)\,\mathbf{W}_{hh}
$$

If the largest singular value of these Jacobians is below 1, contributions from distant steps **vanish** exponentially; if above 1, they **explode**. Consequently vanilla RNNs struggle to learn dependencies more than roughly 10–20 steps apart — for instance, agreeing a verb with a subject far back in a long sentence.

**Truncated BPTT** limits backpropagation to a window of (say) 100 steps for efficiency, carrying the hidden state forward but not gradients beyond the window.

## A character-level language model

```python
import torch
import torch.nn as nn

text = "machine learning is the study of algorithms that improve through experience. " * 50
chars = sorted(set(text)); stoi = {c: i for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text])
V, H, T = len(chars), 128, 64

class CharRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, 32)
        self.rnn = nn.RNN(32, H, batch_first=True)          # vanilla tanh RNN
        self.out = nn.Linear(H, V)
    def forward(self, x, h=None):
        o, h = self.rnn(self.emb(x), h)
        return self.out(o), h

model = CharRNN(); opt = torch.optim.Adam(model.parameters(), lr=3e-3)
for step in range(600):
    i = torch.randint(0, len(data) - T - 1, (32,))
    x = torch.stack([data[j:j + T] for j in i]); y = torch.stack([data[j + 1:j + T + 1] for j in i])
    logits, _ = model(x)
    loss = nn.functional.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
    opt.zero_grad(); loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)       # essential for RNNs
    opt.step()
print("loss:", round(loss.item(), 3))

# Sample text
x, h, out = torch.tensor([[stoi["m"]]]), None, "m"
for _ in range(80):
    logits, h = model(x, h)
    x = torch.multinomial(logits[0, -1].softmax(-1), 1)[None]
    out += chars[x.item()]
print(out)
```

This model predicts the next character — the same objective, in miniature, as modern large language models.

## Bidirectional and deep RNNs

- **Bidirectional RNNs** run one RNN forwards and another backwards and concatenate their states, so each position sees both past and future context. Useful for tagging and classification (not for generation, where the future is unknown).
- **Stacked (deep) RNNs** feed the hidden-state sequence of one RNN layer into another.

## Strengths and limitations

**Strengths:** constant memory per step at inference; natural streaming; handles arbitrary lengths.

**Limitations:** sequential computation cannot be parallelised across time during training (slow on GPUs); vanishing gradients limit memory; a fixed-size hidden state is an information bottleneck.

The fixes came in stages: **gating** (LSTM, GRU) for memory, **attention** for the bottleneck, and **transformers** for parallelism. Recently, **state-space models** and linear recurrent architectures (e.g. Mamba, RWKV) revisit recurrence with parallelisable training, aiming to combine transformer-like quality with RNN-like inference efficiency.

:::exercise
1. Count the parameters of a vanilla RNN with input size 32 and hidden size 128.
2. Train the character model without gradient clipping and with a higher learning rate. What happens?
3. Build a task where the label depends on the first token of a length-50 sequence, and show that a vanilla RNN struggles to learn it.
:::

:::takeaway
- RNNs carry a hidden state across time with shared weights, handling variable-length sequences.
- Training unrolls the network and applies backpropagation through time.
- Products of Jacobians make long-range gradients vanish or explode; clip gradients.
- Gating, attention and transformers address RNN limitations; modern recurrent models revisit them efficiently.
:::

=== POST ===
slug: lstm-networks
title: "Long Short-Term Memory (LSTM): Gated Memory Explained"
category: deep-learning
level: Intermediate
tags: lstm, gates, rnn, sequences, memory
summary: LSTMs add a protected cell state and three gates that decide what to forget, write and reveal. We walk through the equations, explain why they preserve gradients, and apply them to sequence tasks.
---
In 1997 Sepp Hochreiter and Jürgen Schmidhuber published the **Long Short-Term Memory** network, designed specifically to overcome the vanishing-gradient problem they had analysed. For nearly two decades, LSTMs were the dominant architecture for speech recognition, machine translation, handwriting recognition and language modelling. Understanding them teaches the powerful idea of **gating** — learned, multiplicative control over information flow — which reappears in GRUs, gated linear units and modern state-space models.

## The key idea: a protected memory lane

The LSTM maintains two states:

- the **cell state** $\mathbf{c}_t$ — a long-term memory "conveyor belt" updated mostly by **addition**;
- the **hidden state** $\mathbf{h}_t$ — the short-term output exposed to the rest of the network.

Three **gates**, each a sigmoid layer producing values in $(0, 1)$, control the cell.

## The equations

Given input $\mathbf{x}_t$ and previous hidden state $\mathbf{h}_{t-1}$:

$$
\begin{aligned}
\mathbf{f}_t &= \sigma(\mathbf{W}_f[\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f) && \text{forget gate} \\
\mathbf{i}_t &= \sigma(\mathbf{W}_i[\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i) && \text{input gate} \\
\tilde{\mathbf{c}}_t &= \tanh(\mathbf{W}_c[\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_c) && \text{candidate memory} \\
\mathbf{c}_t &= \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t && \text{cell update} \\
\mathbf{o}_t &= \sigma(\mathbf{W}_o[\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o) && \text{output gate} \\
\mathbf{h}_t &= \mathbf{o}_t \odot \tanh(\mathbf{c}_t) && \text{hidden state}
\end{aligned}
$$

## Reading the gates

- **Forget gate** $\mathbf{f}_t$: what fraction of each memory component to keep. In a language model, it might clear the stored grammatical number of the subject when a new sentence begins.
- **Input gate** $\mathbf{i}_t$: how much of the new candidate to write.
- **Candidate** $\tilde{\mathbf{c}}_t$: the new content that could be stored.
- **Output gate** $\mathbf{o}_t$: which parts of the memory to reveal as the hidden state now.

## Why LSTMs preserve gradients

Look at the cell update: $\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \dots$. Its Jacobian with respect to the previous cell (along the direct path) is

$$
\frac{\partial\mathbf{c}_t}{\partial\mathbf{c}_{t-1}} = \text{diag}(\mathbf{f}_t)
$$

No repeated multiplication by a weight matrix and no squashing derivative — just the forget gate. When the network learns to keep $\mathbf{f}_t \approx 1$ for a component, the gradient flows back through many steps almost unchanged. Hochreiter and Schmidhuber called this the **constant error carousel**. It is the same idea as a residual connection, applied through time — two decades before ResNets.

:::tip
Initialise the forget-gate bias to a positive value (e.g. 1.0) so the LSTM begins by remembering rather than forgetting. Jozefowicz et al. (2015) found this simple trick substantially improves LSTM training.
:::

## Parameters and cost

Each of the four transformations maps $[\mathbf{h}_{t-1}, \mathbf{x}_t]$ to the hidden size $H$, so an LSTM has

$$
4\big(H(H + D) + H\big)
$$

parameters for input size $D$ — four times a vanilla RNN.

## Using LSTMs in PyTorch

```python
import torch
import torch.nn as nn

class SentimentLSTM(nn.Module):
    def __init__(self, vocab, emb=100, hidden=128, layers=2, classes=2):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=0)
        self.lstm = nn.LSTM(emb, hidden, num_layers=layers, batch_first=True,
                            bidirectional=True, dropout=0.3)
        self.head = nn.Linear(2 * hidden, classes)
    def forward(self, tokens, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(self.emb(tokens), lengths.cpu(),
                                                   batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        final = torch.cat([h[-2], h[-1]], dim=1)          # last layer, forward + backward
        return self.head(final)

model = SentimentLSTM(vocab=20000)
tokens = torch.randint(1, 20000, (4, 30)); lengths = torch.tensor([30, 22, 15, 8])
print(model(tokens, lengths).shape)                        # (4, 2)
print("parameters:", sum(p.numel() for p in model.parameters()))
```

Note **packed sequences**: they let the LSTM skip padding tokens so the final state reflects each sequence's true end.

## A memory test

A classic diagnostic is the **adding problem**: given a long sequence of random numbers with two marked positions, output the sum of the two marked numbers. A vanilla RNN fails once sequences exceed a few dozen steps; an LSTM solves it for hundreds of steps — direct evidence of long-term memory.

## Variants

- **Peephole connections** let gates look at the cell state.
- **Coupled forget/input gates**: $\mathbf{i}_t = 1 - \mathbf{f}_t$.
- **GRU** — a simpler two-gate design (next lecture).
- **ConvLSTM** — convolutions instead of matrix multiplications, for spatio-temporal data like weather radar.
- **xLSTM** (2024) revisits the LSTM with exponential gating and matrix memories to compete with transformers at scale.

## Where LSTMs stand today

Transformers replaced LSTMs for most large-scale NLP because they parallelise over sequence length and model long-range interactions directly. But LSTMs remain practical for **streaming** and **on-device** applications, small datasets, time-series forecasting and control, where their constant per-step cost and compactness are advantages.

:::exercise
1. Derive the parameter count formula and verify it for $D = 100$, $H = 128$.
2. Implement an LSTM cell from the equations using `nn.Linear` layers and check it against `nn.LSTMCell` with copied weights.
3. Train a vanilla RNN and an LSTM on the adding problem with lengths 20, 100 and 300. Plot test error.
:::

:::takeaway
- LSTMs add a cell state updated additively and three sigmoid gates (forget, input, output).
- The cell's self-Jacobian is just the forget gate, so gradients can flow over long ranges.
- Initialise forget-gate biases positive; use packed sequences for padded batches.
- Transformers dominate large-scale NLP, but LSTMs remain useful for streaming, small data and time series.
:::

=== POST ===
slug: gru-gated-recurrent-units
title: Gated Recurrent Units (GRU) and Choosing a Recurrent Cell
category: deep-learning
level: Intermediate
tags: gru, rnn, lstm, gates, sequences
summary: The GRU simplifies the LSTM to two gates and one state while keeping long-term memory. We derive its equations, compare it with LSTMs empirically, and give practical guidance for recurrent models.
---
Seventeen years after the LSTM, Kyunghyun Cho and colleagues (2014) introduced a streamlined alternative while developing neural machine translation: the **Gated Recurrent Unit (GRU)**. It merges the cell and hidden states and uses two gates instead of three. It trains faster, has fewer parameters, and in many tasks matches the LSTM's accuracy.

## The equations

$$
\begin{aligned}
\mathbf{z}_t &= \sigma(\mathbf{W}_z[\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_z) && \text{update gate} \\
\mathbf{r}_t &= \sigma(\mathbf{W}_r[\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_r) && \text{reset gate} \\
\tilde{\mathbf{h}}_t &= \tanh(\mathbf{W}_h[\mathbf{r}_t \odot \mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_h) && \text{candidate state} \\
\mathbf{h}_t &= (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t && \text{interpolation}
\end{aligned}
$$

(Some references and libraries swap the roles of $\mathbf{z}_t$ and $1 - \mathbf{z}_t$; the meaning is the same.)

## Interpreting the gates

- **Update gate** $\mathbf{z}_t$ decides how much of the state to replace. When $\mathbf{z}_t \approx 0$, the state is copied forward unchanged — preserving memory and gradients, like the LSTM's forget gate set to 1. It effectively couples the LSTM's forget and input gates: whatever is written replaces an equal fraction of what is kept.
- **Reset gate** $\mathbf{r}_t$ decides how much of the previous state to use when computing the candidate. When $\mathbf{r}_t \approx 0$, the unit ignores the past and behaves like a feedforward layer on the current input — useful at boundaries, such as the start of a new phrase.

The gradient along the direct path is $\partial\mathbf{h}_t/\partial\mathbf{h}_{t-1} \supseteq \text{diag}(1 - \mathbf{z}_t)$, providing the same highway for long-range gradient flow as the LSTM's cell.

## GRU vs LSTM

| | LSTM | GRU |
|---|---|---|
| States | Cell $\mathbf{c}_t$ + hidden $\mathbf{h}_t$ | Hidden $\mathbf{h}_t$ only |
| Gates | Forget, input, output | Update, reset |
| Parameters (per layer) | $4(H(H + D) + H)$ | $3(H(H + D) + H)$ |
| Speed | Slower | ~25% fewer operations |
| Output gating | Yes — can hide memory | No — full state exposed |

Empirical comparisons (Chung et al., 2014; Jozefowicz et al., 2015; Greff et al., 2017) found **no consistent winner**: GRUs often match LSTMs, sometimes win on smaller datasets, while LSTMs can be slightly stronger on tasks requiring counting or very precise memory — the output gate and separate cell give extra control. A large study of LSTM variants concluded that the forget gate and output activation are the most critical components.

## Comparing them in code

```python
import torch
import torch.nn as nn

def make_copy_task(batch, T, n_symbols=8, delay=50):
    """Remember a short sequence of symbols, output it after a long delay."""
    seq = torch.randint(1, n_symbols, (batch, 5))
    x = torch.zeros(batch, T, dtype=torch.long); x[:, :5] = seq; x[:, 5 + delay] = n_symbols  # "go" marker
    y = torch.zeros(batch, T, dtype=torch.long); y[:, 6 + delay:11 + delay] = seq
    return x, y

class Seq(nn.Module):
    def __init__(self, cell, H=64, V=10):
        super().__init__()
        self.emb = nn.Embedding(V, 16)
        self.rnn = {"lstm": nn.LSTM, "gru": nn.GRU, "rnn": nn.RNN}[cell](16, H, batch_first=True)
        self.out = nn.Linear(H, V)
    def forward(self, x):
        return self.out(self.rnn(self.emb(x))[0])

T = 70
for cell in ["rnn", "gru", "lstm"]:
    torch.manual_seed(0); m = Seq(cell); opt = torch.optim.Adam(m.parameters(), 3e-3)
    for step in range(1500):
        x, y = make_copy_task(64, T)
        logits = m(x)
        loss = nn.functional.cross_entropy(logits[:, 56:61].reshape(-1, 10), y[:, 56:61].reshape(-1))
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    x, y = make_copy_task(512, T)
    acc = (m(x)[:, 56:61].argmax(-1) == y[:, 56:61]).float().mean().item()
    print(f"{cell:<5} copy accuracy after a 50-step delay: {acc:.2f}")
```

On such long-delay memory tasks, the gated cells typically succeed where the vanilla RNN stays near chance.

## Practical guidance for recurrent models

1. **Start with a GRU** for speed; try an LSTM if accuracy matters and compute allows.
2. **Clip gradients** (norm 1–5) — always.
3. Use **2–3 layers** with dropout between them; beyond that, gains diminish.
4. Use **bidirectional** layers when the whole sequence is available at prediction time.
5. **Pack padded sequences** and sort or bucket by length for efficiency.
6. Consider **1-D CNNs** or **temporal convolutional networks** — they parallelise well and often compete with RNNs on sequence classification.
7. For long-context language tasks, prefer **transformers** or modern **state-space models**.

:::note
The GRU's interpolation $\mathbf{h}_t = (1 - \mathbf{z}_t)\mathbf{h}_{t-1} + \mathbf{z}_t\tilde{\mathbf{h}}_t$ is an input-dependent exponential moving average. Many recent efficient sequence models — including gated linear RNNs and selective state-space models like Mamba — are built around similar data-dependent decay mechanisms, but designed so that training can be parallelised across time.
:::

:::exercise
1. Show that a GRU with $\mathbf{r}_t = \mathbf{1}$ and $\mathbf{z}_t = \mathbf{1}$ reduces to a vanilla tanh RNN.
2. Count the parameters of a GRU and an LSTM with $D = 64$, $H = 256$.
3. Train GRU and LSTM sentiment classifiers on a movie-review dataset and compare accuracy and training time per epoch.
:::

:::takeaway
- The GRU uses an update gate (how much to replace) and a reset gate (how much past to use), with a single state.
- Its interpolation provides a gradient highway like the LSTM's cell.
- GRUs have ~25% fewer parameters; accuracy is usually comparable to LSTMs.
- Clip gradients, use packing, and consider CNNs, transformers or state-space models for long sequences.
:::

=== POST ===
slug: sequence-to-sequence-models
title: Sequence-to-Sequence Models and the Encoder–Decoder Framework
category: deep-learning
level: Intermediate
tags: seq2seq, encoder-decoder, machine translation, teacher forcing, beam search
summary: Translation maps a sequence to another of different length. We build the encoder–decoder architecture, train it with teacher forcing, decode with greedy and beam search, and expose the bottleneck that motivated attention.
---
How do you translate "আমি ভাত খাই" into "I eat rice"? The input and output are both sequences, of different lengths, with different word orders. In 2014 two groups — Sutskever, Vinyals and Le at Google, and Cho et al. — showed that neural networks could learn such mappings end to end with the **sequence-to-sequence (seq2seq)** encoder–decoder architecture. It transformed machine translation and became a template for summarisation, speech recognition, question answering and code generation.

## The architecture

**Encoder:** an RNN reads the source sequence $x_1, \dots, x_S$ and produces hidden states; its final state $\mathbf{c} = \mathbf{h}_S$ is a fixed-size **context vector** summarising the whole input.

**Decoder:** another RNN, initialised with $\mathbf{c}$, generates the target one token at a time. At step $t$ it takes the previous output token $y_{t-1}$ and its state, and predicts a distribution over the next token:

$$
P(y_1, \dots, y_T \mid x_{1:S}) = \prod_{t=1}^{T}P(y_t \mid y_{<t}, \mathbf{c})
$$

Generation begins with a special start token `<sos>` and stops when the decoder emits `<eos>`.

## Training with teacher forcing

The loss is the sum of cross-entropies over target positions. During training we feed the decoder the **ground-truth** previous token rather than its own prediction — **teacher forcing**. This makes training stable and parallelisable across time steps (for the decoder inputs), but creates **exposure bias**: at inference the decoder sees its own, possibly wrong, predictions, a situation it never encountered during training. **Scheduled sampling** gradually replaces ground-truth inputs with model predictions to reduce this mismatch.

## Decoding strategies

Finding the most probable output sequence exactly is intractable (the space is exponential). Approximations:

- **Greedy decoding** — pick the most probable token each step. Fast, but an early mistake cannot be undone.
- **Beam search** — keep the $k$ best partial sequences (the beam) at each step, ranked by total log-probability; expand each by every token and keep the top $k$ again. Beam widths of 4–10 are typical in translation.

Beam search favours short sequences (each extra token adds a negative log-probability), so scores are **length-normalised**, e.g. divided by $T^\alpha$ with $\alpha \approx 0.6$–$1$. For open-ended generation, sampling methods (top-k, nucleus) are preferred — covered in the LLM lectures.

## A compact implementation

```python
import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, V, E=64, H=128):
        super().__init__()
        self.emb, self.rnn = nn.Embedding(V, E), nn.GRU(E, H, batch_first=True)
    def forward(self, src):
        outputs, h = self.rnn(self.emb(src))
        return outputs, h                          # outputs kept for attention later

class Decoder(nn.Module):
    def __init__(self, V, E=64, H=128):
        super().__init__()
        self.emb, self.rnn, self.out = nn.Embedding(V, E), nn.GRU(E, H, batch_first=True), nn.Linear(H, V)
    def forward(self, tgt_in, h):
        o, h = self.rnn(self.emb(tgt_in), h)
        return self.out(o), h

class Seq2Seq(nn.Module):
    def __init__(self, V_src, V_tgt):
        super().__init__()
        self.enc, self.dec = Encoder(V_src), Decoder(V_tgt)
    def forward(self, src, tgt_in):                # teacher forcing
        _, h = self.enc(src)
        logits, _ = self.dec(tgt_in, h)
        return logits
    @torch.no_grad()
    def greedy(self, src, sos, eos, max_len=30):
        _, h = self.enc(src)
        y = torch.full((src.size(0), 1), sos)
        out = []
        for _ in range(max_len):
            logits, h = self.dec(y, h)
            y = logits[:, -1].argmax(-1, keepdim=True)
            out.append(y)
            if (y == eos).all():
                break
        return torch.cat(out, 1)

# Toy task: reverse a sequence of digits (tokens 3..12); 0=pad, 1=sos, 2=eos
V = 13
model = Seq2Seq(V, V); opt = torch.optim.Adam(model.parameters(), 2e-3)
for step in range(3000):
    src = torch.randint(3, V, (64, 8))
    tgt = torch.cat([src.flip(1), torch.full((64, 1), 2)], 1)
    tgt_in = torch.cat([torch.full((64, 1), 1), tgt[:, :-1]], 1)
    loss = nn.functional.cross_entropy(model(src, tgt_in).reshape(-1, V), tgt.reshape(-1))
    opt.zero_grad(); loss.backward(); opt.step()
test = torch.randint(3, V, (3, 8))
print(test.tolist()); print(model.greedy(test, 1, 2).tolist())
```

## The bottleneck problem

The entire source sentence must be compressed into a single fixed-size vector $\mathbf{c}$. For short sentences this works; for long ones, information is lost. Cho et al. (2014) observed that translation quality **degraded sharply with sentence length**. Sutskever et al. found a curious trick helped: **reversing the source sentence**, which placed the first source words close to the first target words and shortened the dependencies the network had to bridge.

The principled fix was to let the decoder look back at **all** encoder states, choosing which to focus on at each step. That is **attention** — the subject of the next lecture, and the seed from which the Transformer grew.

:::note
The encoder–decoder idea is far more general than RNNs. Transformer encoder–decoders (T5, Whisper for speech recognition, translation models), image-captioning systems (CNN or ViT encoder, text decoder) and many multimodal models follow the same pattern: encode one modality or sequence into representations, then decode another conditioned on them.
:::

:::exercise
1. Implement beam search for the `Seq2Seq` model with beam width 3 and length normalisation.
2. Train the reversal task with source lengths 8 and 30. How does accuracy change, and why?
3. Explain exposure bias with a concrete example of how one wrong token can derail subsequent predictions.
:::

:::takeaway
- Seq2seq encodes the source into a context and decodes the target autoregressively.
- Teacher forcing makes training efficient but causes exposure bias.
- Decode with greedy or length-normalised beam search.
- The fixed-size context vector is a bottleneck for long inputs — solved by attention.
:::

=== POST ===
slug: attention-mechanism
title: The Attention Mechanism: Learning Where to Look
category: deep-learning
level: Intermediate
tags: attention, bahdanau, luong, query key value, seq2seq
summary: Attention lets a model compute a weighted focus over all input positions for each output. We derive Bahdanau and Luong attention, generalise to queries, keys and values, and see why it became the foundation of transformers.
---
When a human translator writes each word of a translation, they glance back at the relevant words of the source sentence rather than recalling a compressed memory of the whole thing. In 2014 Dzmitry Bahdanau, Kyunghyun Cho and Yoshua Bengio gave neural networks the same ability. Their **attention mechanism** removed the seq2seq bottleneck, dramatically improved translation of long sentences — and planted the idea that grew into the Transformer.

## The idea

Keep **all** encoder hidden states $\mathbf{h}_1, \dots, \mathbf{h}_S$. At each decoder step $t$, compute a separate context vector as a **weighted average** of them, with weights reflecting how relevant each source position is to the current output:

$$
\mathbf{c}_t = \sum_{j=1}^{S}\alpha_{tj}\,\mathbf{h}_j, \qquad \alpha_{tj} = \frac{\exp(e_{tj})}{\sum_{k=1}^{S}\exp(e_{tk})}
$$

The **alignment scores** $e_{tj}$ measure the compatibility of decoder state $\mathbf{s}_{t-1}$ (or $\mathbf{s}_t$) with encoder state $\mathbf{h}_j$. The softmax turns scores into a probability distribution — a soft, differentiable "pointer" to the relevant input positions.

## Scoring functions

**Additive (Bahdanau) attention:**

$$
e_{tj} = \mathbf{v}^\top\tanh(\mathbf{W}_1\mathbf{s}_{t-1} + \mathbf{W}_2\mathbf{h}_j)
$$

a small feedforward network scoring each pair.

**Multiplicative (Luong) attention** (Luong et al., 2015):

- dot: $e_{tj} = \mathbf{s}_t^\top\mathbf{h}_j$
- general: $e_{tj} = \mathbf{s}_t^\top\mathbf{W}\mathbf{h}_j$

Dot-product scoring is cheaper and can be computed for all positions at once with matrix multiplication.

**Scaled dot-product** (Transformer): $e = \mathbf{q}^\top\mathbf{k}/\sqrt{d_k}$. Scaling by $\sqrt{d_k}$ keeps the variance of scores around 1 when vectors have $d_k$ components of unit variance; without it, large dimensions produce huge scores, a saturated softmax and vanishing gradients.

## Queries, keys and values

The general abstraction, introduced explicitly in the Transformer:

- a **query** $\mathbf{q}$ — what I am looking for;
- **keys** $\mathbf{k}_j$ — what each position offers, used for matching;
- **values** $\mathbf{v}_j$ — the content returned if selected.

$$
\text{Attention}(\mathbf{q}, \mathbf{K}, \mathbf{V}) = \sum_j\text{softmax}_j\left(\frac{\mathbf{q}^\top\mathbf{k}_j}{\sqrt{d_k}}\right)\mathbf{v}_j
$$

Think of a soft dictionary lookup: instead of retrieving the single value whose key exactly matches, retrieve a blend of values weighted by key similarity. In seq2seq attention, the query is the decoder state and keys and values are both the encoder states.

## Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V, mask=None):
    """Q: (B, Tq, d)  K: (B, Tk, d)  V: (B, Tk, dv)"""
    scores = Q @ K.transpose(-2, -1) / K.size(-1) ** 0.5       # (B, Tq, Tk)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))   # ignore padding / future
    weights = F.softmax(scores, dim=-1)
    return weights @ V, weights                                 # (B, Tq, dv)

class AdditiveAttention(nn.Module):
    def __init__(self, d_dec, d_enc, d_att=64):
        super().__init__()
        self.W1, self.W2 = nn.Linear(d_dec, d_att, bias=False), nn.Linear(d_enc, d_att, bias=False)
        self.v = nn.Linear(d_att, 1, bias=False)
    def forward(self, s, H, mask=None):          # s: (B, d_dec), H: (B, S, d_enc)
        e = self.v(torch.tanh(self.W1(s)[:, None] + self.W2(H))).squeeze(-1)   # (B, S)
        if mask is not None:
            e = e.masked_fill(mask == 0, float("-inf"))
        a = F.softmax(e, dim=-1)
        return (a[:, :, None] * H).sum(1), a    # context (B, d_enc), weights (B, S)

B, S, d = 2, 6, 16
H = torch.randn(B, S, d); s = torch.randn(B, d)
ctx, w = AdditiveAttention(d, d)(s, H)
print(ctx.shape, w.sum(-1))                        # weights sum to 1 for each example
out, w2 = scaled_dot_product_attention(torch.randn(B, 3, d), H, H)
print(out.shape, w2.shape)
```

In an attentional seq2seq decoder, the context vector $\mathbf{c}_t$ is concatenated with the decoder state (or input) to predict the next token.

## Interpretability: alignment maps

Plotting the attention weights $\alpha_{tj}$ as a matrix (target positions × source positions) reveals learned **alignments**: roughly diagonal for similar word orders, with swaps where languages reorder (e.g. adjective–noun order between English and French). These plots were a striking early glimpse into what neural translators learned.

:::warning
Attention weights are not automatically faithful **explanations** of a model's decision. Research (e.g. "Attention is not Explanation", Jain & Wallace 2019, and responses) showed that different attention patterns can yield the same predictions. Use attention maps as a diagnostic, not proof of reasoning.
:::

## Why attention was revolutionary

1. **No bottleneck** — the decoder accesses the full source at every step; long-sentence performance stopped collapsing.
2. **Short gradient paths** — any output connects to any input through one attention step, easing learning of long-range dependencies.
3. **Content-based addressing** — the model retrieves information by what it is, not where it is.
4. **Generality** — attention works between any two sets of vectors: text and images (captioning), questions and documents (QA), a sequence and itself (**self-attention**).

In 2017, Vaswani et al. asked: if attention is so powerful, do we need recurrence at all? Their answer — "Attention Is All You Need" — introduced the Transformer, which we study in the NLP track.

:::exercise
1. Show that if keys and queries have independent components with zero mean and unit variance, $\text{Var}(\mathbf{q}^\top\mathbf{k}) = d_k$, justifying the $\sqrt{d_k}$ scaling.
2. Add additive attention to the seq2seq reversal model from the previous lecture and compare accuracy on length-30 sequences.
3. Plot the attention matrix for a trained reversal model. What pattern do you expect, and do you see it?
:::

:::takeaway
- Attention computes a context as a softmax-weighted average of encoder states, recomputed for every output.
- Scores can be additive (Bahdanau) or (scaled) dot-product (Luong, Transformer).
- The query–key–value view is a soft dictionary lookup.
- Attention removed the seq2seq bottleneck and led directly to the Transformer.
:::

=== POST ===
slug: residual-connections-resnet-idea
title: Residual Connections: Why Very Deep Networks Became Trainable
category: deep-learning
level: Intermediate
tags: residual connections, resnet, skip connections, deep networks, degradation
summary: Deeper plain networks can train worse than shallower ones. Residual connections fix this by learning corrections to the identity. We explain the degradation problem, the gradient highway, and variants from ResNet to transformers.
---
In 2015, researchers at Microsoft noticed something paradoxical. A 56-layer plain convolutional network had **higher training error** than a 20-layer one. This was not overfitting — the deeper network was worse even on the training data. A deeper network should be able to do at least as well, since the extra layers could simply learn the identity function. Yet optimisation failed to find that solution. Kaiming He, Xiangyu Zhang, Shaoqing Ren and Jian Sun's solution — **residual learning** — enabled networks with more than 150 layers, won the 2015 ImageNet competition, and became one of the most important ideas in deep learning.

## The degradation problem

As depth increases in plain networks (even with BatchNorm and good initialisation), accuracy saturates and then **degrades**. Normalisation had largely addressed vanishing gradients in magnitude, so the issue is subtler: deep compositions of non-linear layers are hard to optimise, and learning even an identity mapping through a stack of non-linear layers is surprisingly difficult for SGD.

## Residual learning

Instead of asking a block of layers to learn a desired mapping $H(\mathbf{x})$ directly, let it learn the **residual** $F(\mathbf{x}) = H(\mathbf{x}) - \mathbf{x}$, and add the input back through a **skip (shortcut) connection**:

$$
\mathbf{y} = F(\mathbf{x}; \{\mathbf{W}_i\}) + \mathbf{x}
$$

If the optimal mapping is close to the identity, the block only needs to push $F$ towards zero — which is easy (e.g. small weights). Each block learns a **correction** to its input rather than a whole new representation.

When dimensions differ (e.g. after downsampling), the shortcut uses a projection: $\mathbf{y} = F(\mathbf{x}) + \mathbf{W}_s\mathbf{x}$, typically a $1 \times 1$ convolution with stride.

## The gradient highway

For a stack of residual blocks, $\mathbf{x}_{L} = \mathbf{x}_l + \sum_{i=l}^{L-1}F(\mathbf{x}_i)$. The gradient is

$$
\frac{\partial\mathcal{L}}{\partial\mathbf{x}_l} = \frac{\partial\mathcal{L}}{\partial\mathbf{x}_L}\left(\mathbf{I} + \frac{\partial}{\partial\mathbf{x}_l}\sum_{i=l}^{L-1}F(\mathbf{x}_i)\right)
$$

The identity term means the gradient from the loss reaches every earlier layer **directly**, without passing through a product of many weight matrices. It cannot vanish merely because of depth (He et al., "Identity Mappings in Deep Residual Networks", 2016).

## Other perspectives

- **Ensembles of paths:** Veit et al. (2016) showed a ResNet behaves like an ensemble of exponentially many shallower paths; deleting a single block from a trained ResNet barely hurts, unlike a plain network.
- **Iterative refinement:** each block refines the representation incrementally.
- **Dynamical systems:** $\mathbf{x}_{l+1} = \mathbf{x}_l + F(\mathbf{x}_l)$ is an Euler step of an ordinary differential equation — the insight behind **Neural ODEs**.
- **Smoother loss landscapes:** visualisations show skip connections turn chaotic loss surfaces into much smoother, more convex-looking ones.

## Residual blocks in code

```python
import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout))
        self.shortcut = nn.Identity() if stride == 1 and cin == cout else nn.Sequential(
            nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))
        nn.init.zeros_(self.f[-1].weight)          # zero-init: each block starts as identity
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.f(x) + self.shortcut(x))

class PreNormResidual(nn.Module):              # the transformer-style residual
    def __init__(self, dim, sublayer):
        super().__init__()
        self.norm, self.sublayer = nn.LayerNorm(dim), sublayer
    def forward(self, x):
        return x + self.sublayer(self.norm(x))

x = torch.randn(2, 64, 32, 32)
print(BasicBlock(64, 128, stride=2)(x).shape)       # (2, 128, 16, 16)
```

**Zero-initialising** the last normalisation weight in each residual branch makes every block start as an identity function — training begins as a shallow network and gradually "grows" depth. Goyal et al. found this improves large-batch training.

## Variants and descendants

| Architecture | Skip-connection idea |
|---|---|
| ResNet (2015) | Additive identity shortcuts |
| Pre-activation ResNet (2016) | BN and ReLU inside the residual branch; clean identity path |
| Highway Networks (2015) | Gated shortcuts (learned mixing) — a precursor |
| DenseNet (2017) | Concatenate all previous feature maps |
| U-Net (2015) | Long skips from encoder to decoder for segmentation |
| Transformer (2017) | Residual around every attention and MLP sub-layer |
| Stochastic depth | Randomly drop residual branches during training |

:::note
Residual connections are now nearly universal — in vision, language, speech, reinforcement learning and diffusion models. A transformer layer is essentially "residual stream + attention correction + MLP correction". Interpretability researchers even describe transformers in terms of a **residual stream** that every layer reads from and writes to.
:::

:::exercise
1. Train plain and residual CNNs with 8, 20 and 56 layers on CIFAR-10 for a few epochs and compare **training** error.
2. Remove individual blocks from a trained ResNet at test time and measure the accuracy drop; repeat for a plain network.
3. Show that $\mathbf{x}_{l+1} = \mathbf{x}_l + hF(\mathbf{x}_l)$ is the Euler discretisation of $d\mathbf{x}/dt = F(\mathbf{x})$.
:::

:::takeaway
- Plain deep networks suffer from degradation: deeper can train worse.
- Residual blocks learn corrections $F(\mathbf{x})$ added to the identity: $\mathbf{y} = F(\mathbf{x}) + \mathbf{x}$.
- The identity path gives gradients a direct highway, enabling hundreds of layers.
- Skip connections underpin ResNets, U-Nets, DenseNets and every transformer.
:::
