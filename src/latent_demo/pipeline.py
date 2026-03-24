"""End-to-end training and evaluation pipeline."""

from __future__ import annotations

import json
import math
import os
import random
import statistics
import time
from typing import Dict, List, Sequence, Tuple

from .dataset import (
    BLUE,
    RED,
    State,
    build_rollout_panel,
    flatten_image,
    generate_dataset,
    render_state,
    save_ppm,
)
from .models import TinyMLP


def centroid_from_color(image: Sequence[Sequence[Tuple[int, int, int]]], color: Tuple[int, int, int]) -> Tuple[float, float]:
    width = len(image[0])
    height = len(image)
    total_x = 0.0
    total_y = 0.0
    total = 0.0
    for y in range(height):
        for x in range(width):
            if image[y][x] == color:
                total_x += x
                total_y += y
                total += 1.0
    if total == 0.0:
        return 0.5, 0.5
    return total_x / total / (width - 1), total_y / total / (height - 1)


def encode_image(image: Sequence[Sequence[Tuple[int, int, int]]]) -> List[float]:
    robot_x, robot_y = centroid_from_color(image, RED)
    object_x, object_y = centroid_from_color(image, BLUE)
    rel_x = object_x - robot_x
    rel_y = object_y - robot_y
    dist = math.sqrt(rel_x ** 2 + rel_y ** 2)
    return [robot_x, robot_y, object_x, object_y, rel_x, rel_y, dist]


def vector_add(a: Sequence[float], b: Sequence[float]) -> List[float]:
    return [x + y for x, y in zip(a, b)]


def train_model(
    model: TinyMLP,
    inputs: Sequence[Sequence[float]],
    targets: Sequence[Sequence[float]],
    epochs: int,
    learning_rate: float,
    seed: int = 0,
) -> List[float]:
    rng = random.Random(seed)
    order = list(range(len(inputs)))
    history: List[float] = []
    for _ in range(epochs):
        rng.shuffle(order)
        losses = []
        for idx in order:
            losses.append(model.train_step(inputs[idx], targets[idx], learning_rate))
        history.append(statistics.fmean(losses))
    return history


def benchmark_forward(model: TinyMLP, sample: Sequence[float], repeats: int = 500) -> float:
    start = time.perf_counter()
    for _ in range(repeats):
        model.predict(sample)
    duration = time.perf_counter() - start
    return duration / repeats


def object_motion_trend_accuracy(pred_latents: Sequence[Sequence[float]], target_latents: Sequence[Sequence[float]]) -> float:
    matches = 0
    total = 0
    for pred, target in zip(pred_latents, target_latents):
        pred_dx = pred[2] - pred[0]
        pred_dy = pred[3] - pred[1]
        target_dx = target[2] - target[0]
        target_dy = target[3] - target[1]
        pred_norm = math.sqrt(pred_dx ** 2 + pred_dy ** 2)
        target_norm = math.sqrt(target_dx ** 2 + target_dy ** 2)
        if pred_norm < 1e-8 or target_norm < 1e-8:
            continue
        cosine = ((pred_dx * target_dx) + (pred_dy * target_dy)) / (pred_norm * target_norm)
        matches += 1 if cosine > 0.85 else 0
        total += 1
    return matches / total if total else 0.0


def latent_to_renderable_state(latent: Sequence[float]) -> State:
    return State(
        robot_x=max(0.08, min(0.92, latent[0])),
        robot_y=max(0.08, min(0.92, latent[1])),
        object_x=max(0.08, min(0.92, latent[2])),
        object_y=max(0.08, min(0.92, latent[3])),
    )


def rollout_prediction(
    predictor: TinyMLP,
    first_latent: Sequence[float],
    actions: Sequence[Sequence[float]],
) -> List[List[float]]:
    latents = [list(first_latent)]
    current = list(first_latent)
    for action in actions:
        current = predictor.predict(list(current) + list(action))
        latents.append(current)
    return latents


def write_report(path: str, metrics: Dict[str, float], latent_history: Sequence[float], pixel_history: Sequence[float]) -> None:
    lines = [
        "Latent Space Predictor Report",
        "",
        f"Final latent training loss: {metrics['latent_final_loss']:.6f}",
        f"Final pixel baseline loss: {metrics['pixel_final_loss']:.6f}",
        f"Object motion trend accuracy: {metrics['trend_accuracy']:.3f}",
        f"Latent predictor parameters: {int(metrics['latent_parameters'])}",
        f"Pixel baseline parameters: {int(metrics['pixel_parameters'])}",
        f"Average latent forward pass (sec): {metrics['latent_forward_sec']:.8f}",
        f"Average pixel forward pass (sec): {metrics['pixel_forward_sec']:.8f}",
        f"Measured speedup: {metrics['speedup_x']:.2f}x",
        "",
        "Loss history snapshot:",
        f"- latent start/end: {latent_history[0]:.6f} -> {latent_history[-1]:.6f}",
        f"- pixel start/end: {pixel_history[0]:.6f} -> {pixel_history[-1]:.6f}",
    ]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def run_demo(output_dir: str) -> Dict[str, float]:
    os.makedirs(output_dir, exist_ok=True)

    data = generate_dataset()
    latent_inputs: List[List[float]] = []
    latent_targets: List[List[float]] = []
    pixel_inputs: List[List[float]] = []
    pixel_targets: List[List[float]] = []
    encoded_frames: List[List[float]] = []
    encoded_next_frames: List[List[float]] = []

    for frame, next_frame, action in zip(data["frames"], data["next_frames"], data["actions"]):
        latent = encode_image(frame)
        next_latent = encode_image(next_frame)
        encoded_frames.append(latent)
        encoded_next_frames.append(next_latent)
        latent_inputs.append(latent + action)
        latent_targets.append(next_latent)
        pixel_inputs.append(flatten_image(frame) + action)
        pixel_targets.append(flatten_image(next_frame))

    latent_model = TinyMLP(input_dim=len(latent_inputs[0]), hidden_dim=24, output_dim=len(latent_targets[0]), seed=1)
    pixel_model = TinyMLP(input_dim=len(pixel_inputs[0]), hidden_dim=32, output_dim=len(pixel_targets[0]), seed=2)

    latent_history = train_model(latent_model, latent_inputs, latent_targets, epochs=40, learning_rate=0.03, seed=11)
    pixel_history = train_model(pixel_model, pixel_inputs[:160], pixel_targets[:160], epochs=4, learning_rate=0.01, seed=13)

    latent_predictions = [latent_model.predict(sample) for sample in latent_inputs]
    latent_loss = statistics.fmean(
        sum((pred[i] - target[i]) ** 2 for i in range(len(target))) / len(target)
        for pred, target in zip(latent_predictions, latent_targets)
    )

    pixel_predictions = [pixel_model.predict(sample) for sample in pixel_inputs[:120]]
    pixel_loss = statistics.fmean(
        sum((pred[i] - target[i]) ** 2 for i in range(len(target))) / len(target)
        for pred, target in zip(pixel_predictions, pixel_targets[:120])
    )

    trend_accuracy = object_motion_trend_accuracy(latent_predictions, latent_targets)
    latent_forward_sec = benchmark_forward(latent_model, latent_inputs[0], repeats=1200)
    pixel_forward_sec = benchmark_forward(pixel_model, pixel_inputs[0], repeats=300)
    speedup_x = pixel_forward_sec / max(latent_forward_sec, 1e-12)

    start_idx = data["episode_start_indices"][0]
    horizon = 6
    rollout_actions = data["actions"][start_idx : start_idx + horizon]
    true_frames = [data["frames"][start_idx]]
    true_frames.extend(data["next_frames"][start_idx : start_idx + horizon])
    rollout_latents = rollout_prediction(latent_model, encoded_frames[start_idx], rollout_actions)
    pred_frames = [render_state(latent_to_renderable_state(latent), size=12) for latent in rollout_latents]
    panel = build_rollout_panel(true_frames, pred_frames)
    save_ppm(os.path.join(output_dir, "rollout_comparison.ppm"), panel)

    metrics = {
        "latent_final_loss": latent_loss,
        "pixel_final_loss": pixel_loss,
        "trend_accuracy": trend_accuracy,
        "latent_parameters": float(latent_model.parameter_count()),
        "pixel_parameters": float(pixel_model.parameter_count()),
        "latent_forward_sec": latent_forward_sec,
        "pixel_forward_sec": pixel_forward_sec,
        "speedup_x": speedup_x,
    }

    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    write_report(os.path.join(output_dir, "report.txt"), metrics, latent_history, pixel_history)
    return metrics
