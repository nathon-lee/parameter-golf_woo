# 11L PartialRoPE + LNScale + EMA + SWA + TTT (non-record)

Non-record submission for the Parameter Golf challenge. This run was tested on **1×H100 for 80 minutes** (approximately equivalent to 8×H100 for 10 minutes).

## Architecture

- **11 transformer layers**, d=512, 8 heads / 4 KV heads, 3× ReluSquared MLP
- **U-Net skip connections**: encoder-decoder style with learnable skip weights
- **Partial RoPE**: rotary on 16 of 64 head dims for position-free generalization
- **LN Scale**: RMSNorm damped by 1/sqrt(layer+1) for deep gradient stability
- **SmearGate**: per-dim gate blending current + previous token embeddings
- **BigramHash(4096, dim=128→512)**: hash-based bigram context embeddings
- Tied input/output embeddings

## Training

- Muon optimizer (Newton-Schulz) for 2D weights, momentum warmup 0.85→0.99
- Adam (beta1=0.8, beta2=0.95) for scalars/embeddings, WD=0.04
- Wallclock-aware cosine warmdown over last ~3000 steps
- Orthogonal init with muP output-projection scaling
- EMA (decay=0.997) + SWA (last 40% of training)

## Compression

- Mixed int5 (MLP) / int6 (attn) per-row quantization + int8 fallback
- zstd-22 compression
- **⚠️ Artifact size: 17.4MB (exceeds 16MB limit — needs further compression)**

## Evaluation

- Sliding window with stride=64 for near-max context scoring
- Full-model SGD TTT: 3 epochs over val, first 2 blocks frozen

## Configuration

```bash
MICRO_BATCH_SEQS=64 \
MAX_WALLCLOCK_SECONDS=4800 \
VAL_LOSS_EVERY=500 \
TRAIN_LOG_EVERY=100 \
WARMUP_STEPS=0 \
RUN_ID=full_1gpu \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

## Key Metrics

| Metric | Value |
|---|---|
| val_loss (pre-TTT) | 2.0444 |
| val_bpb (pre-TTT) | 1.2108 |
| Training steps | 3806 |
| Training time | 4800906ms |
| SWA count | 1185 |
| Model params | 27,092,057 |
| Artifact bytes | 17,347,056 |
| Code bytes | 49,495 |
| Total bytes | 17,396,551 |

## Known Issues

1. **Artifact exceeds 16MB** — needs more aggressive quantization or smaller bigram_vocab_size
2. **TTT final BPB pending** — TTT was still running at time of submission

## Included Files

- `train_gpt.py` — training script
- `train.log` — training log
- `submission.json` — submission metadata
- `README.md` — this file
