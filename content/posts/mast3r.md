---
title: "Geometry Driven Imitation Learning"
date: 2025-03-05
hiddenInHomeList: true
ShowToc: true
tags:
  - Worklog
---

<!-- {{< figure src="/images/robomast3r.png" alt="Robo 3D Teaser" width="1000" >}} -->

> Note: This is a active worklog of an ongoing project.

<video src="/images/posts/robovla/sample_video.mp4" autoplay loop controls></video>
<p align="center"><em>OpenVLA model's predicted action trajectory for a pick and place task in LIBERO simulation environment</em></p>

Multi-view correspondences are useful priors for robot action planning and execution. Gemini Robotics report also points to the same direction when referring to  

## Early work: RGB-Only 3D Diffusion Policy with implicit VGGT representations

![alt text](/images/posts/robovla/architecture.png)

The project addresses the limitations of current manipulation policies that rely on costly and unreliable depth sensors by exploring an RGB-only alternative. It integrates the Visual Geometry Grounded Transformer (VGGT) with a 3D Diffusion Policy (DP3) to enable robot manipulation using only standard RGB inputs. VGGT offers robust 3D scene understanding from RGB images, eliminating the need for depth sensors and their calibration overhead. The approach extracts intermediate geometric-semantic features (from VGGT’s 6th attention layer), compresses them through a 64D MLP bottleneck, and feeds them into the unchanged DP3 architecture. This design aims to achieve stable, efficient manipulation while reducing hardware dependency and improving real-world deployability.

***Performance Results***

Initial testing on the **Adroit Hammer** dexterous manipulation task shows promising results:

| Method | Sensor Input | Success Rate | Hardware Cost |
|--------|-------------|--------------|---------------|
| DP3 + Point Cloud | RGB-D Camera | 100% | $200+ sensor |
| **Our: DP3 + VGGT** | **RGB Only** | **91%** | **$20 webcam** |
| DP3 + DINO | RGB Only | 87% | $20 webcam |

The results suggest that RGB-only approaches might be viable for manipulation tasks, though more extensive evaluation is needed to confirm generalization.

### Motivation: From Task-Specific to Generalist Robot Policies
Our DP3 + VGGT work achieved 91% success on Adroit Hammer, but revealed a fundamental limitation: each new task requires hundreds of expert demonstrations and dedicated training. This O(N) scaling is economically impossible—household robots need 1000+ skills, but we can't collect 500 demos per skill.
The industry solution: Vision-Language-Action (VLA) models that leverage internet-scale pretraining.

## Current work: Enhancing VLA Spatial Reasoning with Object Completion Priors

**Research Question**: Can object completion priors improve VLA manipulation 
performance beyond basic 3D reconstruction?

**Approach**:
- VLA Backbone: OpenVLA
- 3D Encoder: Rayst3r (vs. VLM-3R's CUT3R)
- Key Difference: Object completion priors for occluded geometry

**Expected Benefits**:
1. Better grasp affordance prediction
2. Occlusion-robust planning
3. Understanding of object physics/stability

### Comparison to VLM-3R

| Aspect | VLM-3R | Our Approach |
|--------|--------|--------------|
| Task | Visual QA + spatial reasoning | Robot manipulation |
| 3D Model | CUT3R (basic reconstruction) | Rayst3r (with completion) |
| Object Understanding | Implicit from point clouds | Explicit completion priors |
| Focus | Language-grounded reasoning | Action prediction |

### Preliminary Results

Experiments in progress........

<!-- ### Open Questions
1. Does object completion improve manipulation vs. basic reconstruction?
2. Can we train end-to-end or need frozen 3D encoder?
3. Computational cost of Rayst3r at inference time? -->


## 📚 References and Related Work

- **3D Diffusion Policy (DP3)**: [Paper](https://arxiv.org/abs/2403.03954) | [Original Repo](https://github.com/YanjieZe/3D-Diffusion-Policy)
- **VGGT**: [Paper](https://arxiv.org/abs/2503.11651) | [Project Page](https://vgg-t.github.io/)
- **R3M**: [Paper](https://arxiv.org/abs/2203.12601) - Seminal work on pretrained visual representations for robotics