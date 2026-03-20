# SmearGate + Seq2048 + NorMuon Weight Decay + NTK RoPE

**Mean val_bpb: 1.1537** (pending 3-seed verification)

Builds on Int6 STE + NorMuon + SWA + Sliding Window baseline (1.1602) with three new techniques.

## New Techniques (over baseline)

1. **SmearGate**: Learned gate that blends each token's embedding with its predecessor. Adds ~512 parameters. Captures local bigram-like context at the embedding level before any attention layers.

2. **Sequence Length 2048**: Training at 2x context length (up from 1024). More context per optimization step improves representation quality. Eval also uses seq_len=2048 with NTK-aware RoPE extrapolation.

3. **NorMuon Weight Decay (0.02)**: Decoupled weight decay applied to NorMuon optimizer parameters. Improves generalization and quantization robustness.

4. **NTK-Aware RoPE Extrapolation**: Adjusts RoPE base frequency when eval_seq_len differs from train_seq_len, preserving positional encoding quality at extended lengths.

## Retained Techniques (from baseline)

- Int6 STE (fake quantization with straight-through estimator)
- NorMuon optimizer (row-normalized Newton-Schulz)
- 3x MLP width (1536 hidden)
- FP16 tied embedding passthrough
- Sliding window eval (stride=64)
- SWA (stochastic weight averaging during warmdown)
- Zstd-22 compression
- U-Net skip connections

## Results

| Seed | val_bpb | Steps | ms/step | Artifact |
|------|---------|-------|---------|----------|
| 1337 | 1.1537 (single seed) | TBD | TBD | TBD |
| 42 | TBD | TBD | TBD | TBD |
| 7 | TBD | TBD | TBD | TBD |

## Architecture

- 9 layers, 512 dim, 8 heads, 4 KV heads (GQA)
- Vocab 1024 (SentencePiece BPE), seq len 2048, tied embeddings
- relu² activation, RoPE, logit softcapping (30.0)
- SmearGate on embeddings

## Dependencies

Standard PyTorch + zstandard for zstd compression.
