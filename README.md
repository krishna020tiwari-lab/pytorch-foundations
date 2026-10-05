# PyTorch Foundations: From Autograd Mechanics to Production Pipelines

A hands-on, code-first repository tracking my deep learning implementation journey using PyTorch 2.x and CUDA acceleration.

## Progression

1. **`01_cuda_check.py`**: CUDA environment verification and tensor device allocation.
2. **`02_autograd_basics.py`**: Manual computational graphs and backpropagation mechanics (`requires_grad`, `.backward()`).
3. **`03_linear_regression_scratch.py`**: Linear regression built completely from scratch without `torch.nn` modules.
4. **`04_linear_regression_nn.py`**: Refactoring manual linear regression into professional PyTorch abstractions (`nn.Module`, `nn.Linear`, `optim.SGD`).
5. **`05_multi_layer_perceptron.py`**: Non-linear function approximation ($y = x^2$) using Multi-Layer Perceptrons and `ReLU` activation functions.
6. **`06_dataset_dataloader.py`**: Custom `Dataset` and `DataLoader` pipelines for mini-batch stochastic gradient descent.
7. 07_convolutional_neural_network.py: Built a Convolutional Neural Network (CNN) using nn.Conv2d and nn.MaxPool2d for spatial feature extraction on MNIST images, achieving 98.54% test accuracy with a dedicated evaluation loop (model.eval(), torch.no_grad())
8. Transfer Learning with ResNet-18
- **File:** `08_transfer_learning.py`
- **Dataset:** CIFAR-10
- **Model:** Pre-trained ResNet-18 (Feature Extractor frozen, replaced `model.fc`)
- **Test Accuracy:** 76.91%
- ## 8b. Advanced Fine-Tuning & Data Augmentation
- **File:** `08b_fine_tuning_augmentation.py`
- **Dataset:** CIFAR-10
- **Techniques:** Unfrozen `layer4`, Differential Learning Rates, Random Horizontal Flips & Crops
- **Test Accuracy:** 97.09% (Jump from 76.91%)
- ## 9. Recurrent Neural Networks (LSTM)
- **File:** `09_recurrent_neural_network.py`
- **Dataset:** MNIST (Treated as $28 \times 28$ sequential rows)
- **Architecture:** 2-Layer LSTM + Linear Classification Head
- **Test Accuracy:** 97.96%
- ## 10. Text Classification with Word Embeddings
- **File:** `10_text_classification_embeddings.py`
- **Dataset:** Synthetic Sentiment Analysis Corpus
- **Architecture:** `nn.Embedding` + LSTM + Linear Classifier
- **Key Concept:** Learned dense semantic vector representations (`embed_dim`) from discrete token IDs to feed sequential LSTM states.
- ## 10b 🎯 Multi-Class Architecture:
- Scaled from 2 to 3 classes using nn.CrossEntropyLoss().
🔄 Bidirectional Dynamics: Processed text in both directions and concatenated dual states: (batch_size, hidden_dim * 2).
## Setup & Environment
- PyTorch 2.14.0+cu130
- Python 3.10+
- CUDA Enabled GPU

- # 🤖 Stage 5: Transformers & Contextual Embeddings (Hugging Face)

This section covers the transition from word-level static lookup tables (`nn.Embedding`) to subword tokenization and contextual representations using pretrained Transformer models (BERT).

---

## 📌 Key Concepts Covered

1. ✂️ **Subword Tokenization (WordPiece / BPE):**
   * Eliminates the Out-Of-Vocabulary (`<UNK>`) problem by splitting rare or complex words into known subword units (e.g., `"unfriendly"` $\rightarrow$ `['un', '##fr', '##ien', '##dly']`).
   * The `##` prefix denotes a continuation subword token.

2. 🏷️ **Special Control Tokens:**
   * `[CLS]` (ID: `101`): Sequence-level classification token added at the beginning.
   * `[SEP]` (ID: `102`): Sentence boundary separation token added at the end.

3. 🧠 **Static vs. Contextual Embeddings:**
   * **Static (`nn.Embedding`):** Maps every unique word to a fixed, unchanging vector.
   * **Contextual (BERT):** Dynamically alters a token's vector based on its surrounding context via self-attention (e.g., `"bank"` in financial vs. river context yielded a cosine similarity of `0.6987`).

---

## 📐 Tensor Shape & Architecture

| Script | Model / Tokenizer | Input Shape | Output / Result |
| :--- | :--- | :--- | :--- |
| `11_subword_tokenization.py` | `bert-base-uncased` | Raw string | Subword tokens & Special Token IDs |
| `12_bert_embeddings.py` | `BertModel` | `(batch_size=2, seq_len=8)` | `last_hidden_state`: `(2, 8, 768)` |

---

## 🛠️ Scripts Overview

* 📄 `11_subword_tokenization.py`: Inspects subword token splitting, prefix continuation symbols (`##`), and automatic control token encoding.
* 📄 `12_bert_embeddings.py`: Loads `bert-base-uncased` from Hugging Face, extracts 768-dimensional token hidden states, and calculates contextual cosine similarity across different sentence contexts.

---

## 🚀 How to Run

```bash
# Run Subword Tokenization Demo
python Pytorch_Foundation/11_subword_tokenization.py

# Run BERT Contextual Embedding Extraction
python Pytorch_Foundation/12_bert_embeddings.py
