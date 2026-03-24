"""Synthetic push-style robot video dataset and tiny image utilities."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple


Color = Tuple[int, int, int]
Image = List[List[Color]]

WHITE: Color = (245, 245, 245)
RED: Color = (220, 60, 60)
BLUE: Color = (65, 105, 225)
GRAY: Color = (200, 200, 200)


@dataclass
class State:
    robot_x: float
    robot_y: float
    object_x: float
    object_y: float


def clamp(value: float, low: float = 0.08, high: float = 0.92) -> float:
    return max(low, min(high, value))


def distance(ax: float, ay: float, bx: float, by: float) -> float:
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def step_dynamics(state: State, action: Sequence[float]) -> State:
    dx, dy = action
    next_robot_x = clamp(state.robot_x + dx)
    next_robot_y = clamp(state.robot_y + dy)
    next_object_x = state.object_x
    next_object_y = state.object_y

    prev_gap = distance(state.robot_x, state.robot_y, state.object_x, state.object_y)
    new_gap = distance(next_robot_x, next_robot_y, state.object_x, state.object_y)
    near_contact = prev_gap < 0.14 or new_gap < 0.12
    moving_toward_object = new_gap < prev_gap

    if near_contact and moving_toward_object:
        next_object_x = clamp(state.object_x + 0.82 * dx)
        next_object_y = clamp(state.object_y + 0.82 * dy)

    return State(
        robot_x=next_robot_x,
        robot_y=next_robot_y,
        object_x=next_object_x,
        object_y=next_object_y,
    )


def blank_image(size: int) -> Image:
    return [[WHITE for _ in range(size)] for _ in range(size)]


def draw_square(image: Image, cx: float, cy: float, color: Color, half_size: int) -> None:
    size = len(image)
    px = int(round(cx * (size - 1)))
    py = int(round(cy * (size - 1)))
    for y in range(max(0, py - half_size), min(size, py + half_size + 1)):
        for x in range(max(0, px - half_size), min(size, px + half_size + 1)):
            image[y][x] = color


def render_state(state: State, size: int = 12) -> Image:
    image = blank_image(size)
    for i in range(size):
        image[0][i] = GRAY
        image[size - 1][i] = GRAY
        image[i][0] = GRAY
        image[i][size - 1] = GRAY
    draw_square(image, state.object_x, state.object_y, BLUE, half_size=1)
    draw_square(image, state.robot_x, state.robot_y, RED, half_size=1)
    return image


def flatten_image(image: Image) -> List[float]:
    flat: List[float] = []
    for row in image:
        for r, g, b in row:
            flat.extend([r / 255.0, g / 255.0, b / 255.0])
    return flat


def stitch_images(left: Image, right: Image, gap: int = 2) -> Image:
    height = len(left)
    filler = [WHITE for _ in range(gap)]
    stitched: Image = []
    for y in range(height):
        stitched.append(left[y] + filler + right[y])
    return stitched


def save_ppm(path: str, image: Image) -> None:
    height = len(image)
    width = len(image[0])
    with open(path, "w", encoding="ascii") as handle:
        handle.write(f"P3\n{width} {height}\n255\n")
        for row in image:
            values: List[str] = []
            for r, g, b in row:
                values.extend([str(r), str(g), str(b)])
            handle.write(" ".join(values))
            handle.write("\n")


def build_rollout_panel(true_frames: Sequence[Image], pred_frames: Sequence[Image], gap: int = 1) -> Image:
    panel_rows: Image = []
    for index in range(len(true_frames)):
        paired = stitch_images(true_frames[index], pred_frames[index], gap=2)
        panel_rows.extend(paired)
        if index != len(true_frames) - 1:
            separator = [WHITE for _ in range(len(paired[0]))]
            for _ in range(gap):
                panel_rows.append(separator[:])
    return panel_rows


def generate_dataset(
    num_episodes: int = 80,
    episode_length: int = 18,
    image_size: int = 12,
    seed: int = 7,
) -> Dict[str, List]:
    rng = random.Random(seed)
    frames: List[Image] = []
    next_frames: List[Image] = []
    actions: List[List[float]] = []
    states: List[State] = []
    next_states: List[State] = []
    episode_start_indices: List[int] = []

    for _ in range(num_episodes):
        state = State(
            robot_x=rng.uniform(0.12, 0.32),
            robot_y=rng.uniform(0.12, 0.32),
            object_x=rng.uniform(0.55, 0.78),
            object_y=rng.uniform(0.55, 0.78),
        )
        episode_start_indices.append(len(frames))
        for _ in range(episode_length):
            target_dx = state.object_x - state.robot_x
            target_dy = state.object_y - state.robot_y
            norm = max(1e-6, math.sqrt(target_dx ** 2 + target_dy ** 2))
            action_scale = rng.uniform(0.025, 0.055)
            action = [
                action_scale * target_dx / norm + rng.uniform(-0.01, 0.01),
                action_scale * target_dy / norm + rng.uniform(-0.01, 0.01),
            ]
            next_state = step_dynamics(state, action)
            frames.append(render_state(state, size=image_size))
            next_frames.append(render_state(next_state, size=image_size))
            actions.append(action)
            states.append(state)
            next_states.append(next_state)
            state = next_state

    return {
        "frames": frames,
        "next_frames": next_frames,
        "actions": actions,
        "states": states,
        "next_states": next_states,
        "episode_start_indices": episode_start_indices,
    }
