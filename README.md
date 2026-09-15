Tiny GPT From Scratch

A compact decoder-only Transformer language model implemented from first principles with Python and PyTorch. The project covers the complete pipeline from raw text and byte-pair encoding (BPE) to causal self-attention, training, resumable checkpoints, autoregressive generation, and controlled experiments.

The model is trained on the Tiny Shakespeare corpus and intentionally kept small so that every stage can be inspected and understood.

Highlights

Byte-level BPE tokenizer implemented without an external tokenizer library

Token and learned positional embeddings

Causal scaled dot-product self-attention

Multi-head attention with output projection

Pre-normalized Transformer blocks with residual connections

GELU feed-forward networks

Next-token cross-entropy training with AdamW

Training and validation loss estimation

Resumable model and optimizer checkpoints

Greedy, temperature, and top-k generation

Context-length and model-size experiments with loss, perplexity, throughput, and text-quality comparisons

Architecture

flowchart TD
    A["BPE token IDs (B,T)"] --> B["Token + position embeddings (B,T,C)"]
    B --> C["Pre-norm Transformer blocks"]
    C --> D["Final LayerNorm"]
    D --> E["Vocabulary projection (B,T,V)"]
    E --> F["Loss during training or sampling during generation"]

Each Transformer block contains:

x = x + MultiHeadAttention(LayerNorm(x))
x = x + FeedForward(LayerNorm(x))

The causal mask prevents a token from attending to future positions, making the model suitable for next-token generation.

End-to-end pipeline

Raw Shakespeare text
        ↓
Train/validation split
        ↓
Train byte-level BPE tokenizer
        ↓
Encode text into token-ID tensors
        ↓
Create input and shifted-target windows
        ↓
TinyGPT forward pass
        ↓
Cross-entropy loss and AdamW updates
        ↓
Checkpointed trained model
        ↓
Autoregressive text generation

Project structure

Tiny_GPT_From_Scratch/
├── artifacts/
│   ├── checkpoints/           # Saved model and optimizer states
│   └── tokenizer/             # BPE vocabulary and merge rules
├── data/
│   ├── raw/                   # Original Tiny Shakespeare text
│   └── processed/             # Splits and encoded token tensors
├── model/
│   ├── attention.py           # Causal attention head and multi-head attention
│   ├── embeddings.py          # Token and positional embeddings
│   ├── feed_forward.py        # Per-token GELU feed-forward network
│   ├── transformer_block.py   # Pre-norm block and residual connections
│   ├── transformer.py         # Sequential Transformer stack
│   └── tiny_gpt.py            # Complete model, loss, and generation
├── tokenizer/
│   └── bpe.py                 # BPE training, encoding, decoding, save/load
├── checkpoint.py              # Checkpoint serialization and restoration
├── compare_contexts.py        # Context-length comparison
├── compare_models.py          # Model-size comparison
├── dataset.py                 # Context windows and random batches
├── experiments.py             # Named experiment configurations
├── generate.py                # Sampling experiments
├── prepare_data.py            # Train/validation text split
├── prepare_tokens.py          # Encode and save corpus tensors
├── train.py                   # Training, evaluation, throughput, checkpoints
└── train_tokenizer.py         # Train and validate the BPE tokenizer

Core components

Byte-level BPE tokenizer

The tokenizer starts with the 256 possible byte values and learns frequently occurring adjacent pairs. With a configured vocabulary size of 300, it learns 44 merged tokens in addition to the base byte vocabulary.

Important functions in tokenizer/bpe.py:

get_pair_counts() counts adjacent token pairs.

merge_pair() replaces occurrences of a selected pair.

train_bpe() repeatedly merges the most frequent pair.

build_vocabulary() maps token IDs back to byte sequences.

encode() and decode() convert between text and token IDs.

save_tokenizer() and load_tokenizer() persist tokenizer artifacts.

Training examples

The corpus is stored as a one-dimensional tensor of token IDs. dataset.get_batch() samples random windows and creates shifted targets:

Input:  [t0, t1, t2, t3]
Target: [t1, t2, t3, t4]

Every sequence position therefore acts as a next-token classification example.

Decoder-only Transformer

For the baseline configuration:

Setting

Value

Vocabulary size

300

Context length

16

Embedding size

32

Attention heads

4

Features per head

8

Transformer blocks

3

Parameters

57,900

Tensor shapes through the model:

Token IDs:                 (B,T)
Token + position vectors:  (B,T,C)
Transformer output:        (B,T,C)
Vocabulary logits:         (B,T,V)
Targets:                   (B,T)
Cross-entropy loss:        scalar

Setup

Clone the repository and create a virtual environment:

git clone https://github.com/awichauhan/Tiny_GPT_From_Scratch.git
cd Tiny_GPT_From_Scratch

python3 -m venv venv
source venv/bin/activate

python -m pip install --upgrade pip
pip install torch numpy

The project was developed with Python 3.11.

Prepare the data and tokenizer

Place the corpus at:

data/raw/tiny_shakespeare.txt

Create the train/validation split:

python prepare_data.py

Train and verify the BPE tokenizer:

python train_tokenizer.py

Encode both corpus splits and save PyTorch tensors:

python prepare_tokens.py

Generated artifacts include:

artifacts/tokenizer/merges.json
artifacts/tokenizer/vocabulary.json
data/processed/train_tokens.pt
data/processed/validation_tokens.pt

Train the model

Available experiments are defined in experiments.py.

python train.py --experiment context_8
python train.py --experiment context_16
python train.py --experiment context_32
python train.py --experiment medium_model

Training performs:

Random batch sampling

Forward propagation

Next-token cross-entropy calculation

Backpropagation

AdamW parameter updates

Periodic train/validation evaluation

Throughput reporting

Checkpoint saving

Each experiment writes to its own checkpoint:

artifacts/checkpoints/context_8.pt
artifacts/checkpoints/context_16.pt
artifacts/checkpoints/context_32.pt
artifacts/checkpoints/medium_model.pt

If a checkpoint exists, training restores the model, optimizer, step, configuration, metrics, and random-number-generator state. A completed experiment is not silently trained past its configured target step.

Generate text

Run the sampling comparison:

python generate.py

Generation supports:

Greedy/argmax decoding

Temperature-controlled sampling

Top-k filtering

Fixed random seeds for reproducible comparisons

At each generation step, the model crops to its latest context window, takes the final position's vocabulary logits, samples one token, appends it, and repeats.

The tokenizer uses safe replacement decoding for generated byte sequences. This prevents incomplete UTF-8 byte predictions from crashing generation.

Run the comparisons

Compare context lengths under identical sampling settings:

python compare_contexts.py

Compare small and medium model capacity:

python compare_models.py

Results

All context experiments used 2,000 training steps and 256 next-token targets per step. Throughput values were measured locally on CPU and should be treated as environment-specific.

Context-length comparison

Context

Batch

Parameters

Validation loss

Perplexity

Tokens/s

8

32

57,644

3.2092

24.76

39,398

16

16

57,900

3.2765

26.48

43,717

32

8

58,412

3.2922

26.90

46,183

Observations:

Context 8 achieved the lowest validation loss under the limited training budget.

Context 32 produced the strongest dialogue and punctuation structure in the inspected sample.

Longer context supplies more potential information, but a small model may require more capacity or training to use it effectively.

Higher measured throughput for the longer contexts reflects hardware efficiency at this tiny scale; it does not remove attention's quadratic scaling with context length.

Model-size comparison

Metric

Small model

Medium model

Context length

16

16

Embedding size

32

64

Transformer blocks

3

4

Parameters

57,900

239,020

Validation loss

3.2765

2.9750

Perplexity

26.48

19.59

Training throughput

43,717 tokens/s

27,836 tokens/s

The medium model used 4.13 times as many parameters, reduced perplexity by approximately 26%, and processed approximately 36% fewer tokens per second.

Sampling comparison

Strategy

Observed behavior

Greedy

Deterministic but entered a repetitive I I I... loop

Temperature 0.5

More conservative and repetitive

Temperature 0.8

Better diversity/coherence balance

Temperature 1.2

Noisier output and occasional invalid byte predictions

Temperature 0.8 + top-k 20

Best overall balance for these checkpoints

Example medium-model generation

Prompt:

ROMEO:

Generated excerpt:

ROMEO: bre:
Sids many, bly subo's your grawh abe
As that with ry mive;
As atte nose.

QUCENTETHES:
Ioth gee that at they mentrend you hounce.

DUCOMIO:
Now, I have a could you eew a ploager.

The generated text remains imperfect, as expected for a 239K-parameter model trained for only 2,000 steps, but it demonstrates learned speaker formatting, punctuation, line structure, and Shakespeare-like token patterns.

What this project demonstrates

How text becomes byte-level BPE token IDs

Why inputs and shifted targets create next-token supervision

How query, key, and value projections produce contextual representations

Why causal masking is required for decoder-only generation

How residual paths and LayerNorm stabilize Transformer blocks

How vocabulary projection turns contextual features into token logits

How autograd, cross-entropy, and AdamW train the network

Why validation loss is necessary alongside training loss

How complete training state is saved and resumed

How decoding strategy changes repetition, diversity, and text quality

How model capacity and context length affect loss, throughput, and generation

