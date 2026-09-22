# Tiny GPT — Cybersecurity Domain Fine-Tuning

An educational implementation of **domain-adaptive fine-tuning** on a small decoder-only Transformer built from scratch with Python and PyTorch.

This project starts from a previously trained **Tiny GPT** language model and adapts it from its original Shakespeare distribution to a cybersecurity domain using data derived from:

- MITRE ATT&CK Enterprise
- NIST Cybersecurity Glossary

The purpose of the project is not to build a production-quality security LLM. The model contains only **57,900 parameters** and has a **16-token context window**.

Instead, the project is designed to study the mechanics of:

- continued pretraining / domain adaptation
- tokenizer reuse during fine-tuning
- domain shift
- validation methodology
- catastrophic forgetting
- controlled model comparison
- decoding strategies
- limitations caused by model capacity, tokenization, and context length

---

## Relationship to Tiny GPT From Scratch

This repository builds on:

**Tiny GPT From Scratch**

The original project implemented the complete language-model pipeline:

```text
Raw text
   ↓
Byte-level BPE tokenizer
   ↓
Token IDs
   ↓
Context windows
   ↓
Decoder-only Transformer
   ↓
Cross-entropy loss
   ↓
Backpropagation
   ↓
AdamW optimization
   ↓
Autoregressive generation
```

The original model was trained on **Tiny Shakespeare**.

This project asks the next question:

> Can an already-trained language model be adapted to a completely different domain without training it from scratch again?

The target domain chosen for the experiment is **cybersecurity**.

---

# Experiment Overview

The complete experiment is:

```text
Shakespeare-trained Tiny GPT
            ↓
Evaluate on cybersecurity text
            ↓
Prepare MITRE + NIST security corpus
            ↓
Reuse original BPE tokenizer
            ↓
Tokenize security corpus
            ↓
Measure tokenizer/domain mismatch
            ↓
Fine-tune all pretrained parameters
            ↓
Save security-adapted checkpoint
            ↓
Evaluate on security validation data
            ↓
Evaluate again on Shakespeare data
            ↓
Measure catastrophic forgetting
            ↓
Compare generation before vs after
```

---

# Base Model

The pretrained Tiny GPT used for this experiment has:

| Setting | Value |
|---|---:|
| Vocabulary size | 300 |
| Embedding size | 32 |
| Context length | 16 |
| Attention heads | 4 |
| Transformer blocks | 3 |
| Parameters | 57,900 |

Architecture:

```text
Token IDs
   ↓
Token Embeddings
   +
Position Embeddings
   ↓
Transformer Block × 3
   ↓
Final LayerNorm
   ↓
Vocabulary Projection
   ↓
300 vocabulary logits
   ↓
Next-token prediction
```

The model was already trained on Tiny Shakespeare before beginning this project.

Fine-tuning therefore starts from **learned parameters**, not random initialization.

---

# Cybersecurity Dataset

Two public cybersecurity sources were used.

## MITRE ATT&CK Enterprise

The MITRE ATT&CK Enterprise STIX dataset contains multiple cybersecurity object types.

The following object categories were selected:

```text
attack-pattern
malware
tool
intrusion-set
campaign
course-of-action
```

Objects marked as revoked or deprecated were excluded.

Extracted MITRE records:

```text
1,798
```

---

## NIST Cybersecurity Glossary

The NIST glossary provides cybersecurity terminology and definitions.

Example structure:

```text
Term
   ↓
Definition
   ↓
Source
```

Extracted NIST definitions:

```text
7,747
```

---

# Data Preparation

The extracted records were:

```text
MITRE records
      +
NIST definitions
      ↓
Text extraction
      ↓
Cleaning
      ↓
Markdown/citation removal
      ↓
Whitespace normalization
      ↓
Deduplication
      ↓
Deterministic shuffle
      ↓
90/10 train-validation split
```

Final corpus:

| Dataset statistic | Value |
|---|---:|
| MITRE records | 1,798 |
| NIST records | 7,747 |
| Unique combined records | 9,441 |
| Training records | 8,496 |
| Validation records | 945 |

The deterministic shuffle uses a fixed random seed so that the split can be reproduced.

---

# Why the Original Tokenizer Was Reused

The original Tiny GPT tokenizer was trained on Shakespeare and contains:

```text
300 tokens
```

It might seem natural to train a new tokenizer for cybersecurity.

That was deliberately **not done**.

The pretrained model has already learned embeddings associated with the existing token IDs:

```text
token ID 263
      ↓
embedding[263]
```

Changing the tokenizer could assign a completely different meaning to token ID `263`, while the model would still contain the embedding learned for the old token.

Therefore the experiment preserves:

```text
same tokenizer
same vocabulary
same token IDs
same model architecture
```

and changes only the model parameters through fine-tuning.

---

# Tokenizer Domain-Shift Analysis

The existing BPE tokenizer can represent cybersecurity terms because it begins from UTF-8 bytes, but domain-specific words are often split inefficiently.

Examples:

| Word | Bytes | Tokens | Bytes / Token |
|---|---:|---:|---:|
| firewall | 8 | 7 | 1.14 |
| malware | 7 | 6 | 1.17 |
| authentication | 14 | 11 | 1.27 |
| credential | 10 | 9 | 1.11 |
| cryptography | 12 | 12 | 1.00 |
| vulnerability | 13 | 12 | 1.08 |

Example:

```text
authentication

a | u | th | en | t | i | c | a | t | i | on
```

The tokenizer therefore remains compatible with the pretrained model but is not optimized for the cybersecurity domain.

This becomes important later because the model has only a **16-token context window**.

---

# Security Tokenization

The cleaned corpus was encoded with the original BPE tokenizer.

Generated token tensors:

```text
data/processed/security_train_tokens.pt
data/processed/security_validation_tokens.pt
```

Token counts:

| Split | Tokens |
|---|---:|
| Security training | 1,990,177 |
| Security validation | 233,575 |

---

# Baseline Evaluation

Before fine-tuning, the pretrained model was evaluated on both:

```text
Shakespeare validation data
Security validation data
```

No parameters were changed during this stage.

The evaluation used:

```python
model.eval()
torch.no_grad()
```

and did not call:

```python
loss.backward()
optimizer.step()
```

This established the model's performance before cybersecurity adaptation.

---

# Full-Parameter Fine-Tuning

The pretrained model weights were loaded from the original checkpoint.

A **new optimizer** was created instead of restoring the old Shakespeare-training optimizer state.

Fine-tuning configuration:

| Setting | Value |
|---|---:|
| Training steps | 500 |
| Batch size | 16 |
| Learning rate | `1e-4` |
| Context length | 16 |
| Optimizer | AdamW |
| Trainable parameters | 57,900 |
| Fine-tuning type | Full parameter |

Training loop:

```text
security_train_tokens
        ↓
random context windows
        ↓
Tiny GPT
        ↓
next-token logits
        ↓
cross-entropy loss
        ↓
loss.backward()
        ↓
gradients
        ↓
optimizer.step()
        ↓
updated model parameters
```

All model parameters were allowed to change.

This is a form of **domain-adaptive continued pretraining** because the training objective remains next-token prediction.

---

# Fine-Tuning Progress

During training, both security and Shakespeare validation loss were monitored.

| Step | Train Loss | Security Val Loss | Shakespeare Val Loss |
|---:|---:|---:|---:|
| Before FT | — | 3.7613 | 3.2625 |
| 1 | 3.6760 | 3.7756 | 3.2677 |
| 100 | 3.7317 | 3.4844 | 3.3773 |
| 200 | 3.6915 | 3.4848 | 3.4366 |
| 300 | 3.3334 | 3.3665 | 3.4825 |
| 400 | 3.4568 | 3.3549 | 3.5083 |
| 500 | 3.3312 | 3.2852 | 3.5563 |

The trend already suggests two simultaneous effects:

```text
Security loss ↓
Shakespeare loss ↑
```

---

# Controlled Base vs Fine-Tuned Evaluation

A separate controlled evaluation was then performed.

Both models received:

- identical validation datasets
- identical batch size
- identical context length
- identical random seed
- identical sampled validation windows
- 100 evaluation batches

Configuration:

```text
Evaluation batches: 100
Batch size:         32
Context length:     16
Random seed:        42
```

Results:

| Metric | Base Model | Security Fine-Tuned | Change |
|---|---:|---:|---:|
| Security validation loss | 3.7467 | **3.3058** | **-11.77%** |
| Shakespeare validation loss | **3.2650** | 3.5439 | **+8.54%** |

---

# Main Experimental Result

The security validation loss decreased:

```text
3.7467
   ↓
3.3058
```

Relative reduction:

```text
11.77%
```

At the same time, Shakespeare validation loss increased:

```text
3.2650
   ↓
3.5439
```

Relative degradation:

```text
8.54%
```

The experiment therefore demonstrates both:

```text
DOMAIN ADAPTATION
       +
CATASTROPHIC FORGETTING
```

The model became better at predicting cybersecurity text while becoming worse at its original Shakespeare distribution.

---

# Generation Experiments

After quantitative evaluation, the base and fine-tuned models were compared qualitatively using identical prompts.

Generation strategies tested:

```text
Greedy decoding
Temperature sampling
Top-k sampling
Top-p / nucleus sampling
```

Top-p sampling was added to the original Tiny GPT generation implementation.

---

## Greedy Decoding

Always selects:

```text
argmax(next-token logits)
```

It is deterministic but caused strong repetition in this very small model.

Example behavior:

```text
...tititititititition...
```

---

## Temperature Sampling

Logits are scaled using:

```text
logits / temperature
```

Lower temperature sharpens the distribution.

Higher temperature increases diversity.

---

## Top-k Sampling

Only the `k` highest-scoring candidate tokens remain available for sampling.

Example:

```text
top_k = 20
```

means that only the 20 strongest candidate tokens can be sampled.

---

## Top-p / Nucleus Sampling

Instead of selecting a fixed number of candidate tokens, top-p keeps the smallest set of tokens whose cumulative probability reaches a chosen threshold.

Example:

```text
top_p = 0.9
```

Conceptually:

```text
sort probabilities
       ↓
cumulative sum
       ↓
keep tokens until probability mass ≈ 0.9
       ↓
sample
```

This creates an adaptive candidate set.

---

# Generation Result

Changing the decoding algorithm did **not** solve the generation-quality problem.

The base model produced mostly Shakespeare-like fragments.

The security-fine-tuned model produced different patterns and more technical-looking fragments, but still failed to generate coherent cybersecurity explanations.

This showed an important distinction:

```text
Fine-tuning changed the probability distribution
                    ≠
Fine-tuning created a capable language model
```

Sampling strategies can change how tokens are selected from the model's probability distribution.

They cannot compensate for a weak underlying probability distribution.

---

# Training-Format Sanity Test

A final generation experiment used prompts that more closely matched the fine-tuning corpus:

```text
Term: firewall
Definition:

Term: malware
Definition:

Term: authentication
Definition:

Name: Malware
Description:
```

This experiment revealed another major limitation.

Prompt token counts were:

| Prompt | Token count |
|---|---:|
| `Term: firewall\nDefinition:` | 22 |
| `Term: malware\nDefinition:` | 20 |
| `Term: authentication\nDefinition:` | 25 |
| `Name: Malware\nDescription:` | 23 |

But the model context length is:

```text
16
```

Generation uses a sliding context:

```python
current_context = token_ids[:, -self.context_length:]
```

Therefore the model was already discarding part of every structured prompt before generating its first new token.

For example:

```text
25-token prompt
       ↓
context length = 16
       ↓
first 9 tokens removed
```

The combination of:

```text
small vocabulary
        +
inefficient security tokenization
        +
16-token context
```

severely limits useful conditioning.

---

# Why Generation Remained Poor

The final experiments identified three primary bottlenecks.

## 1. Model Capacity

The model contains only:

```text
57,900 parameters
```

It was intentionally designed for learning Transformer mechanics rather than producing production-quality language.

Fine-tuning cannot turn a weak base language model into a strong foundation model.

---

## 2. Context Length

The model sees only:

```text
16 tokens
```

This is too small for realistic cybersecurity prompts, especially when technical words are fragmented into many BPE tokens.

---

## 3. Tokenizer Domain Mismatch

The tokenizer was trained on Shakespeare.

It can represent cybersecurity text, but often requires many small token pieces.

That increases sequence length and consumes the already-small context window rapidly.

---

# Final Conclusion

The experiment successfully demonstrates that **domain-adaptive fine-tuning works even on a model built from scratch**.

Quantitatively:

```text
Security validation loss:
3.7467 → 3.3058
11.77% relative improvement
```

while:

```text
Shakespeare validation loss:
3.2650 → 3.5439
8.54% relative degradation
```

This provides direct experimental evidence of:

```text
domain adaptation
        +
catastrophic forgetting
```

However, qualitative generation remained poor because of fundamental limitations in:

```text
model capacity
context length
tokenizer efficiency
original language-model quality
```

The experiment therefore also demonstrates an important lesson:

> Fine-tuning can move a model's learned probability distribution toward a new domain, but it cannot create capabilities that the underlying model does not have sufficient capacity to represent.

---

# Project Structure

Important files introduced or used by the fine-tuning experiment:

```text
Tiny_GPT_Fine_Tuning/
│
├── artifacts/
│   ├── checkpoints/
│   │   ├── tiny_gpt.pt
│   │   └── security_finetuned.pt
│   └── tokenizer/
│
├── data/
│   ├── raw/
│   │   ├── tiny_shakespeare.txt
│   │   ├── mitre_attack/
│   │   └── nist/
│   │
│   └── processed/
│       ├── train_tokens.pt
│       ├── validation_tokens.pt
│       ├── security_corpus.txt
│       ├── security_train.txt
│       ├── security_validation.txt
│       ├── security_train_tokens.pt
│       └── security_validation_tokens.pt
│
├── model/
│   ├── attention.py
│   ├── embeddings.py
│   ├── feed_forward.py
│   ├── transformer_block.py
│   ├── transformer.py
│   └── tiny_gpt.py
│
├── tokenizer/
│   └── bpe.py
│
├── inspect_security_data.py
├── prepare_finetune_data.py
├── prepare_finetune_tokens.py
├── inspect_security_tokenization.py
├── evaluate_base_model.py
├── finetune.py
├── compare_finetuned_model.py
├── compare_generations.py
├── checkpoint.py
└── dataset.py
```

---

# Running the Fine-Tuning Experiment

The original Tiny GPT checkpoint and tokenizer must already exist.

## 1. Inspect the raw security datasets

```bash
python inspect_security_data.py
```

---

## 2. Prepare the security corpus

```bash
python prepare_finetune_data.py
```

This performs:

```text
schema extraction
cleaning
deduplication
shuffle
train-validation split
```

---

## 3. Tokenize the security corpus

```bash
python prepare_finetune_tokens.py
```

---

## 4. Inspect tokenizer behavior

```bash
python inspect_security_tokenization.py
```

---

## 5. Evaluate the original pretrained model

```bash
python evaluate_base_model.py
```

---

## 6. Fine-tune on cybersecurity data

```bash
python finetune.py
```

The resulting checkpoint is saved as:

```text
artifacts/checkpoints/security_finetuned.pt
```

---

## 7. Run the controlled evaluation

```bash
python compare_finetuned_model.py
```

This compares:

```text
base model
vs
security fine-tuned model
```

on both security and Shakespeare validation data.

---

## 8. Compare generation

```bash
python compare_generations.py
```

Generation experiments include:

```text
Greedy
Temperature
Top-k
Top-p
```

---

# Concepts Practiced

## Machine Learning / LLM Concepts

- Transfer learning
- Continued pretraining
- Domain adaptation
- Fine-tuning
- Next-token prediction
- Cross-entropy loss
- Train/validation separation
- Controlled evaluation
- Domain shift
- Catastrophic forgetting
- Autoregressive generation
- Temperature sampling
- Top-k sampling
- Top-p / nucleus sampling
- Context-window limitations
- Tokenizer/model compatibility

---

## Python / PyTorch Concepts

- Dictionaries and lists
- JSON parsing
- File I/O
- Regular expressions
- Functions and reusable modules
- Deterministic random seeds
- Tensor slicing
- Tensor reshaping
- `torch.no_grad()`
- `model.eval()`
- `loss.backward()`
- `optimizer.step()`
- `torch.sort()`
- `torch.cumsum()`
- Boolean masking
- `torch.multinomial()`
- `torch.gather()`
- Checkpoint serialization

---

## DSA Concepts

- Hash maps / dictionaries
- Sets and hash-based deduplication
- Arrays / lists
- Sequential traversal — `O(N)`
- Random indexing
- Sliding windows
- Sorting — approximately `O(V log V)`
- Prefix / cumulative sums — `O(V)`
- Threshold filtering
- Sequence concatenation
- Fixed-size context windows

---

# What Comes Next

This project intentionally stops after demonstrating domain adaptation on Tiny GPT.

The next stage moves the same experiment to a **real pretrained language model**.

The progression becomes:

```text
Tiny GPT From Scratch
        ↓
understand Transformer mechanics

Tiny GPT Security Fine-Tuning
        ↓
understand domain adaptation
and catastrophic forgetting

Real Pretrained Model Fine-Tuning
        ↓
apply the same concepts
at realistic model scale
```

The real-model project will investigate:

```text
baseline generation
tokenizer efficiency
security-domain continued pretraining
before/after evaluation
instruction tuning
response-only loss
LoRA / PEFT
full fine-tuning vs parameter-efficient fine-tuning
```

---

# Project Status

- [x] Start from pretrained Tiny GPT checkpoint
- [x] Acquire MITRE ATT&CK data
- [x] Acquire NIST glossary data
- [x] Inspect JSON schemas
- [x] Extract cybersecurity text
- [x] Clean corpus
- [x] Deduplicate corpus
- [x] Create train/validation split
- [x] Reuse pretrained tokenizer
- [x] Tokenize security corpus
- [x] Analyze tokenizer domain mismatch
- [x] Establish base-model security baseline
- [x] Full-parameter cybersecurity fine-tuning
- [x] Save fine-tuned checkpoint
- [x] Controlled base-vs-fine-tuned evaluation
- [x] Measure catastrophic forgetting
- [x] Compare generation
- [x] Implement Top-p sampling
- [x] Test training-format prompts
- [x] Identify context-window bottleneck
- [x] Complete Tiny GPT domain-adaptation experiment

---

## Educational Scope

This repository is intentionally experimental.

The generated text should **not** be interpreted as reliable cybersecurity guidance.

The project exists to study how pretrained language models behave when adapted to a new domain and to expose the mechanics behind modern LLM fine-tuning workflows.