---
title: "Language-guided Image Tokenization"
date: 2025-03-05
hiddenInHomeList: true
tags:
  - GenAI
  - Multimodal Alignment
  - Model Efficiency
---

#### [Github](https://github.com/Sanny26/TexTok-DiT) · [Report](https://sanny26.github.io/pdfs/posts/textok/11_777_Report1.pdf)

{{< figure src="/images/posts/textok/arch.png" alt="Language-guided Image Tokenization Results" width="1000"
caption="Top image shows the 1D VAE based image tokenizer implemented in this project. Bottom image shows the application of frozen tokenizer to train a super-resolution Diffusion Transformer.">}}


**1D-Tokenizer based visual generation scales up!**

Super-resolution (SR) is a compelling testbed for 1D, text-aware tokenizers because it combines strict spatial fidelity with the potential benefits of language guidance. Unlike free-form text-to-image generation, SR must refine an existing low-resolution (LR) image without violating its structure, making it a stress test for multimodal alignment.

Generating 2D token grids for large images is compute-intensive, so 1D tokenization offers an efficient alternative: images are encoded into a compact latent sequence, refined in latent space with text guidance, and detokenized back into image tokens.

We develop TexTok-VAE, a 1D continuous, text-aware tokenizer attending to captions during both encoding and decoding, compressing images into 32 Gaussian-distributed tokens conditioned on CLIP text embeddings. Experiments were conducted at limited scale due to compute constraints, but still provide meaningful insights.

> **Note**: This work is inspired by [TexTok](https://arxiv.org/pdf/2412.05796) and [TA-TiTok](https://tacju.github.io/projects/maskgen.html). Limited scaling experiments were conducted to understand this space and reproduce some of the published results.

## Understanding effectiveness of 1D-tokenizers.
Text-aware tokenizers offer a great balance between compute and context needed for generation.

| **Tokenizer** | **Tokens** | **Inference Time** | **FID** | **Status** |
|---------------|------------|-------------------|---------|------------|
| 2D VAE (Open-source) | 256 | 5.10 | <span style="background-color: #28a745; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold;">2.27</span> | Baseline |
| 1D Image Tokenizer | 32 | <span style="background-color: #28a745; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold;">0.24</span> | 2.77 | 95% faster |
| **Text-aware Tokenizer** | **32 + 77(text)** | <span style="background-color:rgb(191, 227, 59); color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold;">**0.326**</span> | <span style="background-color: rgb(191, 227, 59); color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold;">**2.75**</span> | **Best balance** |


## Results
{{< figure src="/images/posts/textok/tok.png" alt="Language-guided Image Tokenization Results" width="1000" caption="Left: Input image to the tokenizer. Middle: Text-Aware Tokenizer Reconstructed Image for an empty caption as input. Right: Text-aware Tokenizer Reconstructed Image for a meaningful caption with context of how to reconstruct certain parts of the image." >}}

### Key Insights

- **Computational Efficiency**: Text-aware tokenization reduces inference time by 94% compared to VAE-based approaches
- **Quality Preservation**: FID scores remain competitive (2.75 vs 2.27 baseline) despite dramatic speed improvements
- **Scalability**: Demonstrates that 1D tokenization can effectively scale for visual generation tasks
- **Sharper reconstructions**: Continuous Gaussian latents outperform discrete tokens, avoiding blurriness and artifacts
- **Enhanced detail recovery**: Text guidance improves fine-grained reconstruction (e.g., instruments, watch faces), though complex scenes remain challenging.
- **Limitations**: High text guidance scales can induce hallucinations; some fine details (faces, hands) remain difficult

{{< figure src="/images/posts/textok/textokvae_results.png" alt="Language-guided Image Tokenization Results" width="1000" caption="**Effect of text guidance on image tokenization**. *The Detokenized column displays reconstructed images with and without text guidance. The Difference columns highlight the texture details missed in the reconstruction when no caption is provided, emphasizing the improvements text guidance brings to capturing finer image details.*" >}}


<!-- 
{{< figure src="/images/posts/textok/sr_results.png" alt="Language-guided Image Tokenization Results" width="1000" caption="Qualitative results from our super-resolution models, compared to the original, low-resolution, and detokenized images, along with their respective captions." >}}

<style>
/* Add background color to the figure for transparent images */
.post-content figure img {
    background-color: #d5d7d6 !important;
    padding: 20px;
    border-radius: 8px;
}
</style> -->

