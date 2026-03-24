# Latent Space Predictor

**Inspired by Prof. Kim's lectures on video processing and Yann LeCun's JEPA philosophy.**

A compact research-style demo for action-conditioned video prediction in latent space.

## Project Motivation

A core idea in modern video representation learning is that a model does not always need to generate every pixel of the next frame in order to understand dynamics. If a compact latent captures the right semantics, then predicting future states in latent space can be both faster and more meaningful.

This project explores that idea in a minimal robot-pushing setting:

- encode each frame into a semantic latent
- condition on the current action
- predict the next latent instead of directly generating the next image

The result is a small but concrete prototype that demonstrates predictive world-model thinking in a form simple enough to inspect end-to-end.

## What This Demo Shows

- A tiny latent transition model can be dramatically cheaper than direct pixel prediction.
- A compact latent can still preserve motion-relevant structure, especially the manipulated object's movement trend.
- Even a minimal project can communicate research taste: representation learning, action-conditioned dynamics, and JEPA-style abstraction.

## Key Result

Running the default script on the current machine produced:

| Metric | Result |
|---|---:|
| Latent predictor loss | `0.0025` |
| Object-motion trend accuracy | `0.920` |
| Latent predictor parameters | `415` |
| Pixel baseline parameters | `28,176` |
| Measured inference speedup | `~59x` |

Interpretation:

- the latent model is much smaller and faster than the pixel baseline
- the predicted latent still tracks the object's movement direction well enough to remain semantically meaningful

## Method Overview

### 1. Synthetic Push-Style Video Dataset

Each frame contains:

- a red robot end-effector
- a blue object
- an action `(dx, dy)`

When the end-effector reaches the object, the object is pushed. This creates a simple robot-video dynamics task with interpretable motion.

### 2. Simple Semantic Encoder

For transparency, the encoder is intentionally lightweight. It detects the centroids of the robot and object and builds a latent vector:

`[robot_x, robot_y, object_x, object_y, rel_x, rel_y, distance]`

This makes the representation easy to understand and easy to evaluate.

### 3. Latent Dynamics Predictor

A tiny MLP learns:

`[z_t, action_t] -> z_(t+1)`

The goal is not photorealism. The goal is to model motion-relevant structure efficiently.

### 4. Pixel-Space Baseline

A naive baseline learns:

`[image_t, action_t] -> image_(t+1)`

This gives a direct comparison for parameter count and inference cost, highlighting why latent prediction is attractive.

## Why This Is a Strong Lab-Application Project

This repository is intentionally small, but it still demonstrates several qualities that matter in research:

- translating a high-level idea into a runnable prototype
- designing a controlled experiment instead of only describing intuition
- comparing against a baseline rather than making a claim without evidence
- communicating both results and limitations clearly

In other words, it is not just “a coding project.” It is a small experimental argument.

## Repository Structure

```text
latent-space-predictor/
├── README.md
├── requirements.txt
├── scripts/
│   └── run_demo.py
└── src/
    └── latent_demo/
        ├── __init__.py
        ├── dataset.py
        ├── models.py
        └── pipeline.py
```

## Quickstart

This project uses only the Python standard library.

```bash
cd latent-space-predictor
python3 scripts/run_demo.py
```

Artifacts are written to:

- `outputs/report.txt`
- `outputs/metrics.json`
- `outputs/rollout_comparison.ppm`

## Portfolio Framing

If I were presenting this project as part of a lab application, I would frame it like this:

> I built this demo to explore a simple version of action-conditioned predictive modeling for video. Instead of generating the next frame directly, I encode the scene into a compact semantic latent and learn the transition in latent space. Even in a toy pushing environment, the model is much faster than pixel-space prediction while still preserving object-motion trends.

## Limitations and Next Steps

- The encoder is hand-crafted rather than learned.
- The dataset is synthetic rather than a real robot dataset such as PushT or BAIR.
- The current model is designed for clarity, not maximum accuracy.

Natural next steps would be:

- replace the encoder with a learned CNN
- train the latent model end-to-end in PyTorch
- evaluate on a real robot-video dataset
- compare against stronger baselines beyond a tiny pixel MLP

## Suggested GitHub Description

`A minimal JEPA-style latent dynamics demo for robot video prediction.`
