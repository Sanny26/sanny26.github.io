---
title: "Patch-wise Mixed Precision Attention for Vision Models"
date: 2025-03-05
hiddenInHomeList: true
ShowToc: true
tags:
  - Worklog
---

> Note: This is a active worklog of an ongoing project.

## Why
Vision and 3D foundation models (e.g., Dust3R, VGGT) rely on transformer attention over hundreds of image patches. At inference time, every patch is processed with the same precision (FP16/FP32), even when many patches are visually uninformative (background). This wastes compute and memory, limiting on-device deployment.

## Hypothesis
Not all patches need equal numerical precision. By routing low-saliency patches through a lower-precision path (e.g., FP8/INT8) while keeping high-saliency patches at higher precision, we can reduce latency and memory without degrading geometric fidelity where it matters.

## Why This Should Work
- Empirical precedent: LLM quantization ([SmoothQuant](https://arxiv.org/abs/2211.10438), [AWQ](https://arxiv.org/abs/2306.00978)) and [FlashAttention-3](https://github.com/Dao-AILab/flash-attention) show mixed-precision matmuls can be fast and accurate when the kernel is designed for it.
- Vision evidence: Patch-level quantization retains accuracy on ImageNet; 3D ViT backbones tolerate post-training quantization.
- Intuition: Background patches contribute less to reconstruction/decision quality; concentrating precision on salient tokens should preserve quality while cutting cost.

## Plan and Milestones
1) Baseline kernels
- Implement/profiling: fused FP16 attention; add FP8 path with FP32 accumulation. Measure throughput, SRAM usage, tensor-core utilization.

2) Precision routing prototype
- Random routing to sweep high:low precision ratios; establish performance ceilings and correctness checks.
- Data-driven routing with saliency scores (entropy, Q–K similarity, feature norm). Precompute masks; launch per-tier kernels.

3) Fused mixed-precision kernel
- Single launch with uniform branches per tile using precomputed masks; verify no warp divergence and stable occupancy.

4) Integration + eval
- Drop-in to Dust3R/VGGT-style blocks; run depth and multi-view consistency subsets. Track latency, peak memory, and geometric metrics.

<!-- ## Evaluation Checklist
- Performance: end-to-end latency, peak VRAM, kernel-level metrics (achieved occupancy, bytes/FLOP).
- Accuracy: depth error, reprojection consistency, pose quality; visual regressions on representative scenes.
- Ablations: routing signal choice; precision ratio; layers enabled; tile sizes. -->

## References & Resources
- FlashAttention-3: fast attention kernels — https://github.com/Dao-AILab/flash-attention
- NVIDIA Transformer Engine (FP8 kernels) — https://github.com/NVIDIA/TransformerEngine
- SmoothQuant: Accurate and Efficient Post-Training Quantization — https://arxiv.org/abs/2211.10438
- AWQ: Activation-aware Weight Quantization — https://arxiv.org/abs/2306.00978
- CLIP: Contrastive Language-Image Pretraining — https://arxiv.org/abs/2103.00020
- DINO: Self-Distillation with No Labels — https://arxiv.org/abs/2104.14294
- Triton language for custom kernels — https://triton-lang.org

## Risks
- Attention sinks: “uninformative” tokens can carry global context. Mitigation: per-layer saliency refresh or restrict routing to mid/deep layers.
- Routing overhead: scoring + shuffling can erase gains. Mitigation: lightweight metrics, batched prefix-sums, minimize reorders.
- Hardware limits: FP8 depends on GPU support. Mitigation: fall back to INT8 or packed FP16 emulation; keep interfaces modular.
