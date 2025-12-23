---
title: "Speeding up Multi-view Transformers on H100s!"
date: 2025-12-15
hiddenInHomeList: false
weight: 1
ShowToc: false
UseHugoToc: true
tags:
  - 3D Vision
  - Efficient Deep Learning
  - GPU Programming
---

> Note: This is a active worklog of an ongoing project. In this project, I explore training free approaches to speed up alternating attention in multiview transformers

Let's set some context before we dive into the project.

3D foundational models are feed-forward transformer architectures that take a set of images of a scene as input and predict 3D scene properties such as camera parameters, depth, and point maps. The multi-view transformer backbone consumes {{< hl >}}N{{< /hl >}} patch tokens per image across {{< hl >}}M{{< /hl >}} images and produces a corresponding set of 3D-aware tokens. These tokens are then decoded into the required 3D outputs using task-specific heads. Intuitively, the backbone learns correspondence and triangulation-like reasoning inside its attention layers to recover geometric context.

A common building block in SOTA multi-view transformers like VGGT, MapAnything, and π³ is the alternating-attention layer, which switches between view-wise self-attention (S.A) and global self-attention (S.A).
1. In view-wise attention, S.A is applied within a single view’s patch tokens, so the attention cost scales as {{< hl >}}$O(N^2 \cdot d)${{< /hl >}}.
2. In global attention, S.A is applied over patch tokens from all views, so the attention cost scales as {{< hl >}}$O((MN)^2 \cdot d)${{< /hl >}}.

When global attention is repeated for {{< hl >}}L{{< /hl >}} layers in the multi-view transformer, it quickly becomes compute-heavy as the number of images {{< hl >}}M{{< /hl >}} grows.

Here is the key question.
### Is there any way to reduce the cost of global self-attention?

In classic COLMAP-style reconstructions, a scene graph is built to choose which image pairs should be matched. The nodes are images and the edges connect image pairs that have strong overlap (for example, many co-visible points). This graph is data-dependent and parameter-dependent, and it is often far from complete.

Now compare that to the implicit scene graph formed by global attention in 3D foundational models. Because every image attends to every other image, a complete graph is induced over all image pairs. However, a complete graph is not always needed to estimate the correct 3D scene parameters. In the example below, the edges highlighted in yellow correspond to weak image pairs with low co-visibility.

<img src="/images/posts/mpavit/example.png" alt="Example Scene graph" height="400">

This motivates a simple idea: use a data-dependent matching score to prune low-similarity image pairs during inference, reducing global-attention compute.

### Training-free saliency-based global attention

Let’s revisit the 3D foundational model pipeline. Given a set of images, they are patchified and embedded into patch tokens (for example, using a ViT backbone such as DINOv2 to get local descriptors). These per-view tokens are then fed into the multi-view transformer backbone to produce 3D-aware tokens.

Here is a generic training-free algorithm to perform saliency-based global attention.
1. Compute image-pair cosine similarity to form a saliency matrix {{< hl >}}$K${{< /hl >}} of shape {{< hl >}}$M \times M${{< /hl >}}.
2. Choose a threshold {{< hl >}}$\tau${{< /hl >}} and build a binary mask $B$ where {{< hl >}}$B[i, j] = 1$ if $K[i, j] \ge \tau${{< /hl >}}.
3. If a reference view is required, force connectivity by setting {{< hl >}}$B[0, j] = 1$ and $B[i, 0] = 1${{< /hl >}}.
4.Some models (for example VGGT and MapAnything) append special learnable tokens. If {{< hl >}}$G${{< /hl >}} tokens are appended to the {{< hl >}}$N${{< /hl >}} patch tokens, the effective token count is {{< hl >}}$N' = N + G${{< /hl >}}.

The binary mask can be used in different ways.
- Regime 1: Use the mask to perform {{< hl >}}sparse attention{{< /hl >}} across {{< hl >}}L{{< /hl >}} layers, skipping image pairs with low overlap.
- Regime 2: Use the mask to perform {{< hl >}}mixed-precision attention{{< /hl >}} by processing high-saliency pairs in higher precision (FP16 or BF16) and low-saliency pairs in lower precision (NVFP4, INT4, INT8).

The initial implementation follows Regime 1, and the results are discussed below. First, here are some implementation details.

To implement sparse attention efficiently, block or tile masks are needed to control which token tiles are processed. FlashAttention-style kernels compute attention in tiles (for example 64 x 64 or 128 x 128) to avoid materializing the full attention matrix while evaluating $Q \cdot K^T$. Here, {{< hl >}}the goal is to skip tiles that correspond to low-overlap image pairs, using the frame-level mask $B${{< /hl >}}.

A tile-level mask can be derived from $B$ by mapping each tile’s token range back to its frame index. If a tile starts at token index $t$, its frame is {{< hl >}}$\lfloor t / N' \rfloor${{< /hl >}} (and similarly for the key side). This lets the kernel skip tiles whose (query-frame, key-frame) pair is masked out in the implicit scene graph.

> Note: this works cleanly when $N'$ is a multiple of the kernel tile size (for example 64 or 128). If not, some tiles can straddle two frames. Padding each frame to a tile multiple keeps boundaries aligned. In such cases, we computing the straddling tiles and then zeroing tokens can preserve correctness, but it will not save as much compute.

## Qualitative and Quantative Results

Two things matter here: whether reconstruction quality stays stable under sparse global attention, and whether inference gets faster. All experiments below were run on [VGGT](https://vgg-t.github.io/).

### Where this plugs into VGGT-style aggregation
In a VGGT-like pipeline, patch tokens are extracted per image and fed into the alternating-attention backbone. The saliency procedure produces a frame-pair mask that is applied only inside the global-attention blocks, so attention remains dense within any allowed frame pair while compute is removed for disallowed pairs.

### Evaluation setup
Two evaluations were run:
1. A **qualitative reconstruction example** (Gerard Hall) with $N = 50$ input images.
2. **Quantitative reconstruction metrics** on the CO3D benchmark.

Two knobs were used:
- A similarity threshold $\tau$ for frame-pair pruning.
- A layer cutoff $L$ so that sparse global attention is applied only up to layer $L$ (out of 24 layers).

### How much global-attention compute gets removed by pruning?
Let $|E|$ be the number of valid frame pairs that remain after pruning (ordered pairs, including self-pairs, to match the baseline count). Dense all-pairs attention corresponds to $|E| = M^2$.

With a frame-pair mask, {{< hl >}}the amount of global-attention work is roughly proportional to the number of unmasked (frame, frame) blocks, so the per-layer cost scales like $|E| \cdot N^2$ {{< /hl >}}rather than $M^2 \cdot N^2$. This makes the relative global-attention block compute approximately $|E| / N^2$.

For the $N = 50$ qualitative example, dense all-pairs uses $N^2 = 2500$ pairs.

| Threshold $\tau$ | Valid pairs $\vert E \vert$ | Relative global-attn blocks ($\vert E \vert / 2500$) |
|---:|---:|---:|
| 0.00 | 2500 | 1.000 |
| 0.90 | 314 | 0.126 ($\approx 12.6\%$) |
| 0.95 | 154 | 0.062 ($\approx 6.2\%$) |

This is the cleanest "theoretical" lever. Wall-clock speedups depend on how much of global attention dominates runtime, and how efficiently the underlying kernel can skip masked blocks.

### Qualitative reconstruction results (Gerard Hall, $N = 50$)
Reconstructions stayed visually close to the baseline when sparse global attention was limited to early layers (e.g., $L=10$) for both $\tau = 0.90$ and $\tau = 0.95$.

<figure class="align-center">
    <video width="33%" controls autoplay loop muted playsinline>
        <source src="/images/posts/mpavit/baseline.mov" type="video/mp4">
        Your browser does not support the video tag.
    </video>
    <video width="33%" controls autoplay loop muted playsinline>
        <source src="/images/posts/mpavit/L10T0.9.mov" type="video/mp4">
        Your browser does not support the video tag.
    </video>
    <video width="33%" controls autoplay loop muted playsinline>
        <source src="/images/posts/mpavit/L10T0.95.mov" type="video/mp4">
        Your browser does not support the video tag.
    </video>
    <figcaption style="display: flex; justify-content: space-between; text-align: center; margin-top: 0.5rem; font-size: 0.9em;">
        <div style="width: 33%;">FlashAttention <br>(Baseline)</div>
        <div style="width: 33%;">Sparse Global Attn <br>($L=10, \tau=0.9$)</div>
        <div style="width: 33%;">Sparse Global Attn <br>($L=10, \tau=0.95$)</div>
    </figcaption>
</figure> 


When sparsification was pushed deeper (e.g., $L=15$), reconstruction quality degraded noticeably even though more compute was removed. The failure mode looks consistent with deeper layers needing longer-range interactions to maintain global 3D consistency.


<figure class="align-center">
    <video width="33%" controls autoplay loop muted playsinline>
        <source src="/images/posts/mpavit/baseline.mov" type="video/mp4">
        Your browser does not support the video tag.
    </video>
    <video width="33%" controls autoplay loop muted playsinline>
        <source src="/images/posts/mpavit/L15T0.9.mov" type="video/mp4">
        Your browser does not support the video tag.
    </video>
    <video width="33%" controls autoplay loop muted playsinline>
        <source src="/images/posts/mpavit/L15T0.95.mov" type="video/mp4">
        Your browser does not support the video tag.
    </video>
    <figcaption style="display: flex; justify-content: space-between; text-align: center; margin-top: 0.5rem; font-size: 0.9em;">
        <div style="width: 33%;">FlashAttention <br>(Baseline)</div>
        <div style="width: 33%;">Sparse Global Attn <br>($L=15, \tau=0.9$)</div>
        <div style="width: 33%;">Sparse Global Attn <br>($L=15, \tau=0.95$)</div>
    </figcaption>
</figure> 

### Runtime and memory (GPU inference)
Compute and memory were measured with input tensor shape `[B, N, 3, H, W] = [1, N, 3, 518, 518]`.
<!-- 
| #Images | Method | Peak Mem (GB) | Inference Time (s) |
|:---:|:---|---:|---:|
| <span class="text-blue">100</span> | Baseline | 13.50 | 7.18 |
| <span class="text-blue">100</span> | Proposed ($L = 10$) | 14.10 | 5.87 |
| <span class="text-blue">100</span> | Proposed ($L = 15$) | 13.97 | 5.33 |
| <span class="text-orange">500</span> | Baseline | 48.59 | 157.58 |
| <span class="text-orange">500</span> | Proposed ($L = 10$) | 56.89 | 105.87 |
| <span class="text-orange">500</span> | Proposed ($L = 15$) | 56.85 | 75.52 | -->


| #Images | Method | Inference Time (s) |
|:---:|:---|---:|
| <span class="text-blue">100</span> | Baseline | 7.18 |
| <span class="text-blue">100</span> | Proposed ($L = 10$) | 5.87 |
| <span class="text-blue">100</span> | Proposed ($L = 15$) | 5.33 |
| <span class="text-orange">500</span> | Baseline | 157.58 |
| <span class="text-orange">500</span> | Proposed ($L = 10$) | 105.87 |
| <span class="text-orange">500</span> | Proposed ($L = 15$) | 75.52 |

**Observation:** We see consistent performance gains, ranging from {{<hl>}}1.22×–1.35× speedups on small scenes ($N=100$) to significant **1.49×–2.09×** improvements on larger scenes ($N=500$){{</hl>}} where **global attention** computation dominates.

### CO3D quantitative reconstruction metrics
[CO3D](https://ai.meta.com/datasets/co3d-dataset/) performance is reported using AUC at pose error thresholds of 30/15/5 degrees (higher is better). Here, $L$ denotes the highest transformer layer where sparse global attention is applied (masking is used up to layer $L$). Results are averaged over 5 CO3D classes. We observe that for $L=10$, the reconstruction metrics remain {{<hl>}}within 0.5% of the baseline{{</hl>}}, effectively preserving scene understanding capabilities while reducing compute.

| Method | AUC@30 | AUC@15 | AUC@5 |
|:---|---:|---:|---:|
| Baseline (VGGT) | 0.8769 | 0.8112 | 0.6356 |
| Proposed ($L = 10$) | 0.8759 | 0.8100 | 0.6322 |
| Proposed ($L = 15$) | 0.8614 | 0.7843 | 0.5809 |

### Takeaway
{{<hl>}}Frame-pair pruning can remove a large fraction of global-attention blocks, but sparsification should be limited to early layers to avoid quality collapse{{</hl>}}. In these runs, $L = 10$ stayed close to the baseline on CO3D while still providing meaningful inference speedups, whereas $L = 15$ traded away reconstruction quality for additional compute savings.



<!-- # --
HCI
- Software engineering position - 
- What percentage 

- what is the project: HCI conferences (UIST conference), prototype - interactive system
- sequence of phots or videos, create a 3d asset/models (sam3d)
- how to align 3d models with the dense point cloud 

- correspondence 
- polishing the user interface  -->




