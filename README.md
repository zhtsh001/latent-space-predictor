# Latent Space Predictor

**Inspired by Prof. Kim's lectures on video processing and Yann LeCun's JEPA philosophy.**

A minimal, self-contained demo for learning robot-video dynamics in latent space.

This project builds a tiny push-style video dataset, encodes each frame into a semantic embedding, and trains a small predictor to map:

`current latent + action -> next latent`

The point is simple:

- predicting the next latent is much cheaper than directly generating the next image
- a compact latent can still preserve motion semantics such as object displacement trend

## What is in this repo?

- A synthetic push-style robot video dataset generator
- A simple image encoder based on colored-object centroids
- A tiny MLP latent predictor
- A pixel-space baseline for speed comparison
- An evaluation script that exports metrics and visualization frames

## Why this is a good lab-application demo

This repo is intentionally small enough to understand in one sitting, but it still demonstrates a research-style idea:

1. learn a compact state representation from video
2. predict dynamics in embedding space instead of image space
3. show that the learned transition model is faster and still semantically meaningful

## Repository structure

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

This demo is written in pure Python standard library, so it has no external runtime dependency.

```bash
cd latent-space-predictor
python3 scripts/run_demo.py
```

Outputs will be written to:

- `outputs/report.txt`
- `outputs/metrics.json`
- `outputs/rollout_comparison.ppm`

## Method

### 1. Synthetic push-style dataset

Each frame contains:

- a red robot end-effector
- a blue object
- an action `(dx, dy)`

When the end-effector reaches the object, the object gets pushed. This gives us a tiny but meaningful robot-video dynamics problem.

### 2. Encoder

For clarity, the encoder is deliberately simple:

- detect the centroid of the red robot
- detect the centroid of the blue object
- build a latent vector from their positions and relative geometry

The latent is:

`[robot_x, robot_y, object_x, object_y, rel_x, rel_y, distance]`

### 3. Predictor

A small MLP learns:

`[z_t, action_t] -> z_(t+1)`

### 4. Baseline

A naive pixel-space predictor tries to map:

`[image_t, action_t] -> image_(t+1)`

This is not meant to be state of the art. It is included to make one clean argument:

predicting a small semantic state is cheaper than predicting all pixels.

## Main claim

After running `python3 scripts/run_demo.py`, the script reports:

- latent prediction loss
- object-motion trend accuracy
- parameter count of the latent predictor vs the pixel baseline
- measured forward-pass speed ratio

## Example result from a local run

On the current machine, the default script produced:

- latent predictor loss: `0.0025`
- object-motion trend accuracy: `0.920`
- latent predictor parameters: `415`
- pixel baseline parameters: `28,176`
- measured inference speedup: `~59x`

So even in this tiny toy setting, the latent transition model is both:

- much cheaper than direct pixel prediction
- good enough to preserve the direction of object movement

## Expected takeaway

Even in this toy setup, the latent predictor should:

- run much faster than direct pixel prediction
- preserve the direction of object movement well enough to be semantically meaningful

That makes it a good “small but research-flavored” portfolio piece for a lab application.

## Notes

- The current encoder is hand-crafted for transparency.
- A natural next step would be replacing it with a learned CNN encoder and training end-to-end in PyTorch.
- The dataset is synthetic so the whole demo stays reproducible and dependency-free.

## Suggested GitHub pitch

If you send this repo to a professor, you can describe it like this:

> A minimal JEPA-style dynamics demo: instead of generating the next frame directly, I encode each frame into a compact semantic latent and learn a transition model in latent space conditioned on action. Even in a toy robot-pushing setting, the latent predictor is substantially faster than pixel-space prediction while preserving the motion trend of the manipulated object.

## Suggested application note

If you want, you can pair this repo with a short message like:

> I built this small project to show that I can turn ideas from video representation learning into a concrete, testable prototype. The project is intentionally minimal, but it reflects my interest in predictive world models, compact representations, and action-conditioned video dynamics.
