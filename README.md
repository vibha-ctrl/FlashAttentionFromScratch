# FlashAttentionFromScratch
Read this paper: https://arxiv.org/abs/2205.14135. My attempt at implementing from scratch.

tldr

Transformers rely heavily on scaled dot-product attention, but the standard implementation is memory-inefficient: it constructs the full N×N attention matrix and performs huge, repeated reads/writes between GPU HBM (global memory) and the compute cores.

Modern GPUs are memory-bandwidth bound, not compute-bound, so the bottleneck is moving data, not doing math. This becomes especially expensive because GPUs only have a few megabytes of on-chip SRAM (registers, shared memory, and cache), which is far too small to hold the full attention matrix or all of Q, K, and V at once. As a result, naive attention repeatedly streams large blocks of data from slow HBM, making it both memory-hungry and slow.

FlashAttention solves this by using a tiling algorithm that keeps small blocks of Q, K, and V in fast on-chip SRAM and never materializes the full N×N attention matrix. By minimizing HBM traffic and maximizing on-chip reuse, it dramatically reduces memory usage and significantly improves speed.

In this notebook, we’ll implement both naive attention and FlashAttention to compare their performance. We’ll use randomly initialized tensors for Q, K, and V, since the goal is to understand algorithmic differences and memory behavior not semantic output. If your local machine doesn’t have a GPU like mine, you can run everything on an NVIDIA A100 GPU in Google Colab.
