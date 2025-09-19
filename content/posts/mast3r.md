---
title: "Geometry Meets VLA"
date: 2025-03-05
hiddenInHomeList: true
ShowToc: true
tags:
  - Worklog
---

<!-- {{< figure src="/images/robomast3r.png" alt="Robo 3D Teaser" width="1000" >}} -->

<video src="/images/posts/robovla/sample_video.mp4" autoplay loop controls></video>

## Project Overview

Over the past year, a lot of progress has been made in connecting geometry-specialist models and generalist VLAs. Models like Dust3R and VGGT have shown that multi-view geometry can be recovered reliably in real time. On the other side, frameworks such as OpenVLA and π₀-FAST have demonstrated that vision-language-action pipelines can scale across tasks and robots with minimal finetuning.

The natural next step is to ask: can these two worlds reinforce each other? Geometry provides structure and robustness, while VLAs offer generalization and flexible grounding.

⸻

Current State of the Project

The project explores how geometry-aware representations can be integrated into VLA training pipelines. A key challenge is deciding how much 3D structure to inject:
-	Intermediate latent features from geometry models may enrich policies without committing to an explicit depth prior.
-	Reconstructed depth or pose priors could provide stronger grounding but risk narrowing the model’s ability to generalize across tasks.

The open question we are tackling now is where to strike this balance — injecting enough 3D structure to help manipulation and reasoning, without breaking the generalization power that makes VLAs so attractive in the first place.