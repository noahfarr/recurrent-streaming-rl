import argparse
import glob
import json
import math
import os
import shutil
import subprocess
import sys
import threading
from collections import namedtuple
from pathlib import Path

import numpy as np
import pyray as pr
import torch
from carbs import ObservationInParam
from omegaconf import OmegaConf
from sklearn.decomposition import PCA

from src.utils.carbs import (
    build_carbs,
    from_unit,
    make_space,
    pareto_points,
    to_unit,
)

PANEL_WIDTH = 250
PANEL_TOGGLE = (12, 14, 28, 24)
PANEL_COLLAPSED_LEFT = 104
PADDING = 18
EPSILON = 1e-6
LEGEND_Y = 120
X_BUTTON_Y, Y_BUTTON_Y, BUTTON_HEIGHT = 50, 78, 22
MENU_WIDTH, MENU_ROW_HEIGHT = 380, 22
SURROGATE_GRID_SIZE = 48
PLOT_MARGINS = (64, 96, 38, 56)

BACKGROUND = pr.Color(8, 14, 18, 255)
PANEL = pr.Color(14, 22, 28, 255)
WHITE = pr.Color(235, 235, 235, 255)
GREY = pr.Color(130, 140, 150, 255)
ACCENT = pr.Color(230, 180, 90, 255)
GREEN = pr.Color(120, 200, 120, 255)
BLUE = pr.Color(120, 160, 230, 255)
CYAN = pr.Color(130, 224, 255, 255)
RED = pr.Color(220, 95, 80, 255)
SELECTED = pr.Color(255, 210, 70, 255)
HIGHLIGHT = pr.Color(255, 230, 120, 255)
FRAME = pr.Color(50, 60, 70, 255)
FRAME_LIGHT = pr.Color(60, 70, 80, 255)
BUTTON = pr.Color(30, 42, 52, 255)
MENU_BACKGROUND = pr.Color(20, 30, 38, 255)
MENU_BORDER = pr.Color(80, 90, 100, 255)
MENU_HIGHLIGHT = pr.Color(40, 55, 68, 255)

VIRIDIS_STOPS = [
    (0.0, (68, 1, 84)),
    (0.25, (59, 82, 139)),
    (0.5, (33, 145, 140)),
    (0.75, (94, 201, 98)),
    (1.0, (253, 231, 37)),
]
MAGMA_STOPS = [
    (0.0, (12, 10, 22)),
    (0.45, (95, 25, 95)),
    (0.72, (222, 92, 52)),
    (1.0, (252, 232, 170)),
]

FONT = None
DERIVED_KEYS = ("score", "cost")
Axis = namedtuple("Axis", "to_unit label from_unit")
Geometry = namedtuple("Geometry", "x y width height window_width window_height")


def draw_text(string, x, y, size, color):
    pr.draw_text_ex(
        FONT, string, pr.Vector2(float(x), float(y)), float(size), 1.0, color
    )


def text_width(string, size):
    return pr.measure_text_ex(FONT, string, float(size), 1.0).x


def draw_text_centered(string, center_x, y, size, color):
    draw_text(string, int(center_x - text_width(string, size) / 2), int(y), size, color)


def draw_text_right(string, right_x, y, size, color):
    draw_text(string, int(right_x - text_width(string, size)), int(y), size, color)


def draw_text_vertical(string, x, center_y, size, color):
    pr.draw_text_pro(
        FONT,
        string,
        pr.Vector2(float(x), float(center_y)),
        pr.Vector2(float(text_width(string, size) / 2), 8.0),
        -90.0,
        float(size),
        1.0,
        color,
    )


def gradient(fraction, stops):
    fraction = min(max(fraction, 0.0), 1.0)
    for i in range(len(stops) - 1):
        low_stop, low_color = stops[i]
        high_stop, high_color = stops[i + 1]
        if fraction <= high_stop:
            blend = (fraction - low_stop) / (high_stop - low_stop + EPSILON)
            return pr.Color(
                *(
                    int(low_color[c] + blend * (high_color[c] - low_color[c]))
                    for c in range(3)
                ),
                255,
            )
    return pr.Color(*stops[-1][1], 255)


def score_color(fraction):
    return gradient(fraction, VIRIDIS_STOPS)


def uncertainty_color(fraction):
    return gradient(fraction, MAGMA_STOPS)


def format_value(value):
    return f"{value:.4g}" if isinstance(value, (int, float)) else f"{value}"


def truncate(string, length=22):
    return string if len(string) <= length else string[: length - 2] + ".."


def short_label(key):
    leaf = key.split(".")[-1]
    if leaf == "learning_rate":
        return f"{key.split('.')[0].split('_')[0]}_lr"
    return leaf


def draw_tooltip(lines, x, y, window_width, window_height):
    width = int(max(text_width(line, 16) for line in lines)) + 16
    height = len(lines) * 18 + 12
    x = int(min(x, window_width - width))
    y = int(min(y, window_height - height))
    pr.draw_rectangle(x, y, width, height, pr.Color(0, 0, 0, 225))
    pr.draw_rectangle_lines(x, y, width, height, pr.Color(90, 100, 110, 255))
    for index, line in enumerate(lines):
        draw_text(line, x + 8, y + 6 + index * 18, 16, WHITE)


def draw_polyline(points, color, thickness):
    previous = None
    for point in points:
        if point is not None and previous is not None:
            pr.draw_line_ex(
                pr.Vector2(previous[0], previous[1]),
                pr.Vector2(point[0], point[1]),
                thickness,
                color,
            )
        previous = point


def nearest_point(points, x, y, radius):
    best_index, best_distance = None, radius
    for index, point in enumerate(points):
        if point is None:
            continue
        distance = math.hypot(point[0] - x, point[1] - y)
        if distance < best_distance:
            best_distance, best_index = distance, index
    return best_index


def resolve_paths(arguments):
    paths = []
    for argument in arguments:
        if "*" in argument:
            paths += sorted(glob.glob(argument))
        elif Path(argument).is_dir():
            paths.append(str(Path(argument) / "observations.jsonl"))
        else:
            paths.append(argument)
    return paths


def load_observations(path):
    trials = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            trial = json.loads(line)
        except json.JSONDecodeError:
            continue
        trial["_source"] = Path(path).parent.name
        trials.append(trial)
    return trials


def load_trials(paths):
    return [trial for path in paths for trial in load_observations(path)]


def dotted_get(config, key):
    value = config
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def load_wandb_trials(entity, project, sweep_id):
    import wandb

    api = wandb.Api()
    path = f"{entity}/{project}" if entity else project
    runs = api.runs(path, filters={"config.sweep": sweep_id})
    trials = []
    for run in runs:
        keys = run.config.get("hyperparameters") or []
        hyperparameters = {key: dotted_get(run.config, key) for key in keys}
        score = run.summary.get("score")
        cost = run.summary.get("cost")
        trials.append(
            {
                "hyperparameters": hyperparameters,
                "score": float(score) if score is not None else 0.0,
                "cost": float(cost) if cost is not None else 0.0,
                "is_failure": score is None or cost is None,
                "_source": run.name or run.id,
            }
        )
    return trials


def load_spaces(path):
    config = OmegaConf.load(Path(path).parent / ".hydra" / "config.yaml")
    return {name: make_space(param) for name, param in config.sweep.params.items()}


def value_of(trial, key):
    if key in DERIVED_KEYS:
        return trial.get(key)
    return trial["hyperparameters"].get(key)


def infer_scale(values):
    finite = np.array(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0 or finite.min() <= 0:
        return "linear"
    if np.all(np.abs(finite - 2 ** np.round(np.log2(finite))) / finite < 1e-6):
        return "pow2"
    if finite.max() / finite.min() > 50:
        return "log"
    return "linear"


def make_axis(key, trials, spaces):
    if key in spaces:
        space = spaces[key]
        return Axis(
            lambda value: to_unit(space, value),
            key,
            lambda fraction: from_unit(space, fraction),
        )
    values = [value_of(trial, key) for trial in trials]
    values = [v for v in values if v is not None and np.isfinite(v)]
    scale = "log" if key == "cost" and min(values) > 0 else "linear"
    if key not in DERIVED_KEYS:
        scale = infer_scale(values)
    low, high = min(values), max(values)
    if scale in ("log", "pow2"):
        base = 2.0 if scale == "pow2" else 10.0
        low, high = math.log(low, base), math.log(high, base)

        def to_unit_fraction(value):
            if value is None or value <= 0:
                return None
            return float(
                np.clip((math.log(value, base) - low) / (high - low + EPSILON), 0, 1)
            )

        def from_unit_fraction(fraction):
            return base ** (low + fraction * (high - low))

    else:

        def to_unit_fraction(value):
            if value is None:
                return None
            return float(np.clip((value - low) / (high - low + EPSILON), 0, 1))

        def from_unit_fraction(fraction):
            return low + fraction * (high - low)

    return Axis(to_unit_fraction, f"{key} [{scale}]", from_unit_fraction)


def compute_pareto(trials):
    non_failures = [(i, t) for i, t in enumerate(trials) if not t.get("is_failure")]
    if not non_failures:
        return set()
    observations = [{"output": t["score"], "cost": t["cost"]} for _, t in non_failures]
    _, indices = pareto_points(observations)
    return {non_failures[i][0] for i in indices}


def prepare(trials, spaces):
    axis_keys = ["score", "cost"] + sorted(
        {k for t in trials for k in t["hyperparameters"]}
    )
    hyper_keys = axis_keys[2:]
    axes = {key: make_axis(key, trials, spaces) for key in axis_keys}
    sources = sorted({trial["_source"] for trial in trials})
    return trials, axis_keys, hyper_keys, axes, compute_pareto(trials), sources


def parameter_importances(trials, axes, hyper_keys):
    results = []
    for key in hyper_keys:
        unit_values, scores = [], []
        for trial in trials:
            if trial.get("is_failure"):
                continue
            unit = axes[key].to_unit(value_of(trial, key))
            score = value_of(trial, "score")
            if unit is None or score is None or not np.isfinite(score):
                continue
            unit_values.append(unit)
            scores.append(score)
        unit_values, scores = np.array(unit_values), np.array(scores)
        if (
            len(unit_values) < 3
            or np.std(unit_values) < EPSILON
            or np.std(scores) < EPSILON
        ):
            correlation = 0.0
        else:
            correlation = float(np.corrcoef(unit_values, scores)[0, 1])
        results.append((key, correlation))
    results.sort(key=lambda item: abs(item[1]), reverse=True)
    return results


def trial_detail_lines(trial, multiple_sources):
    lines = [f"score: {trial['score']:.4g}", f"cost: {trial['cost']:.4g}"]
    if trial.get("is_failure"):
        lines.append("FAILURE")
    if multiple_sources:
        lines.append(f"src: {trial['_source']}")
    for key in sorted(trial["hyperparameters"]):
        lines.append(f"{key}: {format_value(trial['hyperparameters'][key])}")
    return lines


def fit_surrogate(paths, trials):
    config = OmegaConf.load(Path(paths[0]).parent / ".hydra" / "config.yaml")
    optimizer, _, _ = build_carbs(config)
    carbs = optimizer.carbs
    seen = set()
    for trial in trials:
        hyperparameters = dict(trial["hyperparameters"])
        signature = tuple(sorted(hyperparameters.items()))
        if signature in seen:
            continue
        seen.add(signature)
        carbs.observe(
            ObservationInParam(
                input=hyperparameters,
                output=float(trial["score"]),
                cost=float(trial["cost"]),
                is_failure=bool(trial.get("is_failure")),
            )
        )
    model = carbs.get_surrogate_model()
    model.fit_observations(carbs.success_observations)
    observations = (
        torch.stack([o.real_number_input for o in carbs.success_observations])
        .cpu()
        .numpy()
    )
    return {
        "model": model,
        "observations": observations,
        "names": list(carbs._real_number_space_by_name.keys()),
        "observation_count": len(carbs.success_observations),
        "failure_count": len(carbs.failure_observations),
        "radius": float(carbs.search_radius_in_basic),
    }


def query_surrogate(model, points):
    tensor = torch.as_tensor(points, dtype=torch.float32)
    with torch.no_grad():
        mean, variance = model.output_model(tensor.to(model.params.device))
        std = torch.sqrt(variance)
        estimate = model._surrogate_to_target(mean).cpu().numpy()
        high = model._surrogate_to_target(mean + std).cpu().numpy()
        low = model._surrogate_to_target(mean - std).cpu().numpy()
    return estimate, np.abs(high - low) / 2.0


def project_to_plane(fit, spaces, size):
    observations = fit["observations"]
    pca = PCA(n_components=2).fit(observations)
    embedding = pca.transform(observations)
    low = embedding.min(0) - 0.08 * (embedding.max(0) - embedding.min(0) + 1e-9)
    high = embedding.max(0) + 0.08 * (embedding.max(0) - embedding.min(0) + 1e-9)
    span_x, span_y = (high[0] - low[0]) or 1.0, (high[1] - low[1]) or 1.0
    grid_x = np.linspace(low[0], high[0], size)
    grid_y = np.linspace(low[1], high[1], size)
    grid_points = np.array(
        [[grid_x[i], grid_y[j]] for j in range(size) for i in range(size)]
    )
    reconstructed = pca.inverse_transform(grid_points)
    score_estimate, uncertainty = query_surrogate(fit["model"], reconstructed)

    fraction_x = (grid_points[:, 0] - low[0]) / span_x
    fraction_y = (grid_points[:, 1] - low[1]) / span_y
    coefficients, *_ = np.linalg.lstsq(
        np.column_stack([np.ones_like(fraction_x), fraction_x, fraction_y]),
        reconstructed,
        rcond=None,
    )
    fields = {}
    for index, name in enumerate(fit["names"]):
        offset, gradient_x, gradient_y = (float(v) for v in coefficients[:, index])
        space = spaces.get(name)
        fields[name] = {
            "offset": offset,
            "gradient_x": gradient_x,
            "gradient_y": gradient_y,
            "to_value": (
                lambda basic, space=space: (
                    space.param_from_basic(basic) if space else basic
                )
            ),
        }

    explained = pca.explained_variance_ratio_
    x_value_at = lambda fraction: f"{low[0] + fraction * span_x:.2g}"
    y_value_at = lambda fraction: f"{low[1] + fraction * span_y:.2g}"
    tick_fractions = (0.0, 0.5, 1.0)
    return {
        "score_grid": score_estimate.reshape(size, size),
        "uncertainty_grid": uncertainty.reshape(size, size),
        "size": size,
        "score_min": float(np.nanmin(score_estimate)),
        "score_max": float(np.nanmax(score_estimate)),
        "uncertainty_min": float(np.nanmin(uncertainty)),
        "uncertainty_max": float(np.nanmax(uncertainty)),
        "observation_count": fit["observation_count"],
        "failure_count": fit["failure_count"],
        "radius": fit["radius"],
        "x_label": f"PC1 ({explained[0] * 100:.0f}% var)",
        "y_label": f"PC2 ({explained[1] * 100:.0f}% var)",
        "x_name": "PC1",
        "y_name": "PC2",
        "x_value_at": x_value_at,
        "y_value_at": y_value_at,
        "x_ticks": [(f, x_value_at(f)) for f in tick_fractions],
        "y_ticks": [(f, y_value_at(f)) for f in tick_fractions],
        "trial_points": [
            ((e[0] - low[0]) / span_x, (e[1] - low[1]) / span_y) for e in embedding
        ],
        "fields": fields,
    }


class Surrogate:
    def __init__(self, paths, grid_size=SURROGATE_GRID_SIZE):
        self.paths, self.grid_size = paths, grid_size
        self.background = {
            "grid": None,
            "signature": None,
            "error": None,
            "running": False,
        }

    def start(self, trials, spaces):
        if self.background["running"]:
            return
        self.background.update(running=True, error=None, grid=None, signature=None)
        snapshot, signature = trials, len(trials)

        def work():
            try:
                fit = fit_surrogate(self.paths, snapshot)
                grid = project_to_plane(fit, spaces, self.grid_size)
                self.background["grid"] = grid
                self.background["signature"] = signature
            except Exception as error:
                self.background["error"] = str(error)
            finally:
                self.background["running"] = False

        threading.Thread(target=work, daemon=True).start()

    def invalidate(self):
        self.background.update(grid=None, signature=None)

    def display(self, trials):
        count = len(trials)
        if self.background["grid"] is not None and self.background["signature"] == count:
            return self.background["grid"]
        if self.background["error"]:
            return {"error": self.background["error"]}
        return None


def inner_plot(geometry):
    left, right, top, bottom = PLOT_MARGINS
    return (
        geometry.x + left,
        geometry.y + top,
        geometry.width - left - right,
        geometry.height - top - bottom,
    )


def draw_color_legend():
    draw_text("color = score", 14, LEGEND_Y, 16, ACCENT)
    for offset in range(PANEL_WIDTH - 28):
        pr.draw_rectangle(
            14 + offset, LEGEND_Y + 24, 1, 16, score_color(offset / (PANEL_WIDTH - 28))
        )
    draw_text("low", 14, LEGEND_Y + 42, 12, GREY)
    draw_text_right("high", PANEL_WIDTH - 14, LEGEND_Y + 42, 12, GREY)


def draw_selector_button(y, label, value, color, size=16):
    pr.draw_rectangle(14, y, PANEL_WIDTH - 28, BUTTON_HEIGHT, BUTTON)
    draw_text(f"{label}  {truncate(value, 18)}", 20, y + 3, size, color)


def selector_hit(mouse_x, mouse_y, y):
    return 14 <= mouse_x <= PANEL_WIDTH - 14 and y <= mouse_y < y + BUTTON_HEIGHT


def dropdown_index(mouse_y, base):
    return int((mouse_y - base) // MENU_ROW_HEIGHT)


def draw_dropdown(base, keys, current, mouse, label_of=str):
    height = len(keys) * MENU_ROW_HEIGHT
    pr.draw_rectangle(14, base, MENU_WIDTH, height, MENU_BACKGROUND)
    pr.draw_rectangle_lines(14, base, MENU_WIDTH, height, MENU_BORDER)
    for index, key in enumerate(keys):
        row_y = base + index * MENU_ROW_HEIGHT
        if (
            14 <= mouse.x <= 14 + MENU_WIDTH
            and row_y <= mouse.y < row_y + MENU_ROW_HEIGHT
        ):
            pr.draw_rectangle(14, row_y, MENU_WIDTH, MENU_ROW_HEIGHT, MENU_HIGHLIGHT)
        draw_text(
            label_of(key), 22, row_y + 3, 16, HIGHLIGHT if key == current else WHITE
        )


def draw_heatmap(grid, value_min, value_max, color_for, x, y, width, height):
    size = grid.shape[0]
    cell_width, cell_height = width / size, height / size
    value_span = (value_max - value_min) or 1.0
    for row in range(size):
        row_y = int(y + (size - 1 - row) * cell_height)
        for column in range(size):
            pr.draw_rectangle(
                int(x + column * cell_width),
                row_y,
                int(cell_width) + 1,
                int(cell_height) + 1,
                color_for((grid[row, column] - value_min) / value_span),
            )
    pr.draw_rectangle_lines(int(x), int(y), int(width), int(height), FRAME_LIGHT)


def draw_colorbar(color_for, value_min, value_max, x, y, width, height):
    for offset in range(int(height)):
        pr.draw_rectangle(
            x, int(y) + offset, width, 1, color_for(1.0 - offset / height)
        )
    pr.draw_rectangle_lines(x, int(y), width, int(height), FRAME_LIGHT)
    draw_text(f"{value_max:.3g}", x + width + 5, int(y) - 2, 13, GREY)
    draw_text(f"{value_min:.3g}", x + width + 5, int(y + height) - 12, 13, GREY)


def draw_loading_skeleton(geometry, trial_count):
    plot_x, plot_y, plot_width, plot_height = inner_plot(geometry)
    fill = pr.Color(26, 36, 44, 255)

    def block(x, y, width, height, rounded=0.6):
        pr.draw_rectangle_rounded(
            pr.Rectangle(float(x), float(y), float(width), float(height)),
            rounded,
            6,
            fill,
        )

    block(plot_x + plot_width / 2 - 100, plot_y - 32, 200, 18)
    pr.draw_rectangle(int(plot_x), int(plot_y), int(plot_width), int(plot_height), fill)

    band = plot_width * 0.16
    center = plot_x - band + ((pr.get_time() * 0.55) % 1.0) * (plot_width + 2 * band)
    step = plot_width / 64
    for column in range(64):
        x = plot_x + column * step
        distance = abs(x + step / 2 - center) / (band / 2)
        if distance < 1.0:
            pr.draw_rectangle(
                int(x),
                int(plot_y),
                int(step) + 1,
                int(plot_height),
                pr.Color(130, 150, 170, int(55 * (1.0 - distance))),
            )
    pr.draw_rectangle_lines(
        int(plot_x), int(plot_y), int(plot_width), int(plot_height), FRAME
    )

    block(plot_x + plot_width + 26, plot_y, 18, plot_height, 0.0)
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        block(plot_x + fraction * plot_width - 16, plot_y + plot_height + 6, 32, 10)
        block(plot_x - 46, plot_y + (1 - fraction) * plot_height - 5, 36, 10)
    block(plot_x + plot_width / 2 - 55, plot_y + plot_height + 26, 110, 12)
    block(geometry.x - 36, plot_y + plot_height / 2 - 55, 12, 110)
    draw_text(
        f"fitting GP surrogate to {trial_count} observations ...",
        geometry.x,
        geometry.window_height - 22,
        14,
        GREY,
    )


def draw_surrogate(app, grid):
    geometry = app.geometry
    if grid is None:
        draw_loading_skeleton(geometry, len(app.trials))
        return
    if "error" in grid:
        draw_text(
            f"surrogate failed: {grid['error']}",
            geometry.x + 20,
            geometry.y + 40,
            16,
            GREY,
        )
        return

    mouse = app.mouse
    plot_x, plot_y, plot_width, plot_height = inner_plot(geometry)
    size = grid["size"]
    if app.show_uncertainty:
        values, value_min, value_max = (
            grid["uncertainty_grid"],
            grid["uncertainty_min"],
            grid["uncertainty_max"],
        )
        color_for, title = uncertainty_color, "uncertainty (1 std)"
    else:
        values, value_min, value_max = (
            grid["score_grid"],
            grid["score_min"],
            grid["score_max"],
        )
        color_for, title = score_color, "predicted score"

    draw_heatmap(
        values, value_min, value_max, color_for, plot_x, plot_y, plot_width, plot_height
    )
    draw_text_centered(title, plot_x + plot_width / 2, plot_y - 30, 20, ACCENT)
    for fraction, label in grid["x_ticks"]:
        draw_text_centered(
            label, plot_x + fraction * plot_width, plot_y + plot_height + 5, 13, GREY
        )
    for fraction, label in grid["y_ticks"]:
        draw_text_right(
            label, plot_x - 8, plot_y + (1 - fraction) * plot_height - 7, 13, GREY
        )
    draw_text_centered(
        grid["x_label"], plot_x + plot_width / 2, plot_y + plot_height + 26, 15, GREY
    )
    draw_text_vertical(
        grid["y_label"], geometry.x - 30, plot_y + plot_height / 2, 15, GREY
    )
    draw_colorbar(
        color_for,
        value_min,
        value_max,
        int(plot_x + plot_width + 26),
        int(plot_y),
        18,
        plot_height,
    )

    for fraction_x, fraction_y in grid["trial_points"]:
        if fraction_x is not None and fraction_y is not None:
            pr.draw_circle_v(
                pr.Vector2(
                    plot_x + fraction_x * plot_width,
                    plot_y + (1 - fraction_y) * plot_height,
                ),
                2.5,
                pr.Color(255, 255, 255, 180),
            )

    draw_text(
        f"{grid['observation_count']} obs  {grid['failure_count']} fail   "
        f"search radius {grid['radius']:.2f}   [m] metric",
        geometry.x,
        geometry.window_height - 22,
        14,
        GREY,
    )

    if (
        plot_x <= mouse.x <= plot_x + plot_width
        and plot_y <= mouse.y <= plot_y + plot_height
    ):
        cell_width, cell_height = plot_width / size, plot_height / size
        column = min(max(int((mouse.x - plot_x) / cell_width), 0), size - 1)
        row = min(max(int((plot_y + plot_height - mouse.y) / cell_height), 0), size - 1)
        lines = [
            f"{grid['x_name']} = {grid['x_value_at'](column / (size - 1))}",
            f"{grid['y_name']} = {grid['y_value_at'](row / (size - 1))}",
            f"pred score: {grid['score_grid'][row, column]:.3g}",
            f"uncertainty: {grid['uncertainty_grid'][row, column]:.3g}",
        ]
        fraction_x = (mouse.x - plot_x) / plot_width
        fraction_y = 1 - (mouse.y - plot_y) / plot_height
        for name, field in grid["fields"].items():
            basic = (
                field["offset"]
                + field["gradient_x"] * fraction_x
                + field["gradient_y"] * fraction_y
            )
            lines.append(f"{short_label(name)}: {format_value(field['to_value'](basic))}")
        draw_tooltip(
            lines,
            mouse.x + 14,
            mouse.y + 14,
            geometry.window_width,
            geometry.window_height,
        )


class View:
    hotkey = None
    name = "?"

    def update(self, app):
        pass

    def handle_keys(self, app):
        pass

    def click(self, app):
        pass

    def menu(self, app):
        pass

    def panel(self, app):
        draw_text(f"view: {self.name}", 14, X_BUTTON_Y + 3, 18, ACCENT)

    def draw(self, app):
        pass


class ScatterView(View):
    hotkey, name = pr.KEY_ONE, "scatter"

    def update(self, app):
        geometry = app.geometry
        x_axis, y_axis = app.axes[app.x_key], app.axes[app.y_key]
        self.positions = [None] * len(app.trials)
        for index, trial in enumerate(app.trials):
            if trial.get("is_failure") or not app.is_visible(trial):
                continue
            unit_x = x_axis.to_unit(value_of(trial, app.x_key))
            unit_y = y_axis.to_unit(value_of(trial, app.y_key))
            if unit_x is None or unit_y is None:
                continue
            self.positions[index] = (
                geometry.x + PADDING + unit_x * (geometry.width - 2 * PADDING),
                geometry.y + PADDING + (1 - unit_y) * (geometry.height - 2 * PADDING),
            )

    def draw(self, app):
        geometry, mouse, positions = app.geometry, app.mouse, self.positions
        x_axis, y_axis = app.axes[app.x_key], app.axes[app.y_key]
        score_to_unit = app.axes["score"].to_unit
        for index, trial in enumerate(app.trials):
            if positions[index] is not None:
                pr.draw_circle_v(
                    pr.Vector2(*positions[index]),
                    4.0,
                    score_color(score_to_unit(value_of(trial, "score"))),
                )
        for index in app.pareto:
            if positions[index]:
                pr.draw_circle_lines_v(
                    pr.Vector2(positions[index][0], positions[index][1]), 7, WHITE
                )

        pr.draw_rectangle_lines(
            geometry.x, geometry.y, geometry.width, geometry.height, FRAME
        )
        for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
            x = int(geometry.x + PADDING + fraction * (geometry.width - 2 * PADDING))
            pr.draw_line(
                x,
                geometry.y + geometry.height,
                x,
                geometry.y + geometry.height + 5,
                GREY,
            )
            draw_text_centered(
                f"{x_axis.from_unit(fraction):.3g}",
                x,
                geometry.y + geometry.height + 8,
                16,
                GREY,
            )
            y = int(
                geometry.y + PADDING + (1 - fraction) * (geometry.height - 2 * PADDING)
            )
            pr.draw_line(geometry.x - 5, y, geometry.x, y, GREY)
            draw_text_right(
                f"{y_axis.from_unit(fraction):.3g}", geometry.x - 8, y - 8, 16, GREY
            )
        draw_text_centered(
            x_axis.label,
            geometry.x + geometry.width / 2,
            geometry.y + geometry.height + 26,
            16,
            GREY,
        )
        draw_text_vertical(
            y_axis.label, PANEL_WIDTH + 12, geometry.y + geometry.height / 2, 16, GREY
        )

        on_x_axis = (
            geometry.x <= mouse.x <= geometry.x + geometry.width
            and geometry.y + geometry.height
            <= mouse.y
            <= geometry.y + geometry.height + 30
        )
        on_y_axis = (
            geometry.x - 44 <= mouse.x <= geometry.x
            and geometry.y <= mouse.y <= geometry.y + geometry.height
        )
        if on_x_axis:
            fraction = np.clip(
                (mouse.x - geometry.x - PADDING) / (geometry.width - 2 * PADDING), 0, 1
            )
            pr.draw_line(
                int(mouse.x),
                geometry.y,
                int(mouse.x),
                geometry.y + geometry.height,
                pr.Color(120, 200, 120, 120),
            )
            draw_tooltip(
                [f"{app.x_key} = {format_value(x_axis.from_unit(fraction))}"],
                mouse.x + 14,
                mouse.y - 26,
                geometry.window_width,
                geometry.window_height,
            )
        elif on_y_axis:
            fraction = np.clip(
                1 - (mouse.y - geometry.y - PADDING) / (geometry.height - 2 * PADDING),
                0,
                1,
            )
            pr.draw_line(
                geometry.x,
                int(mouse.y),
                geometry.x + geometry.width,
                int(mouse.y),
                pr.Color(120, 160, 230, 120),
            )
            draw_tooltip(
                [f"{app.y_key} = {format_value(y_axis.from_unit(fraction))}"],
                mouse.x + 14,
                mouse.y - 26,
                geometry.window_width,
                geometry.window_height,
            )

        if app.selected is not None and positions[app.selected]:
            pr.draw_circle_lines_v(pr.Vector2(*positions[app.selected]), 9, SELECTED)
            draw_tooltip(
                trial_detail_lines(app.trials[app.selected], app.multiple_sources),
                geometry.x + 10,
                geometry.y + 10,
                geometry.window_width,
                geometry.window_height,
            )
        hovered = nearest_point(positions, mouse.x, mouse.y, 11.0)
        app.hovered_trial = hovered
        if hovered is not None:
            draw_tooltip(
                trial_detail_lines(app.trials[hovered], app.multiple_sources),
                mouse.x + 14,
                mouse.y + 14,
                geometry.window_width,
                geometry.window_height,
            )

    def panel(self, app):
        draw_selector_button(X_BUTTON_Y, "X", app.x_key, GREEN)
        draw_selector_button(Y_BUTTON_Y, "Y", app.y_key, BLUE)
        draw_color_legend()
        draw_text("white ring = Pareto", 14, LEGEND_Y + 64, 14, GREY)

    def click(self, app):
        mouse_x, mouse_y = app.mouse.x, app.mouse.y
        if selector_hit(mouse_x, mouse_y, X_BUTTON_Y):
            app.menu = None if app.menu == "x" else "x"
        elif selector_hit(mouse_x, mouse_y, Y_BUTTON_Y):
            app.menu = None if app.menu == "y" else "y"
        elif app.menu in ("x", "y"):
            base = (X_BUTTON_Y if app.menu == "x" else Y_BUTTON_Y) + BUTTON_HEIGHT
            index = dropdown_index(mouse_y, base)
            if 14 <= mouse_x <= 14 + MENU_WIDTH and 0 <= index < len(app.axis_keys):
                setattr(
                    app, "x_key" if app.menu == "x" else "y_key", app.axis_keys[index]
                )
            app.menu = None
        elif mouse_x > PANEL_WIDTH:
            app.selected = nearest_point(self.positions, mouse_x, mouse_y, 12.0)

    def menu(self, app):
        if app.menu in ("x", "y"):
            base = (X_BUTTON_Y if app.menu == "x" else Y_BUTTON_Y) + BUTTON_HEIGHT
            current = app.x_key if app.menu == "x" else app.y_key
            draw_dropdown(base, app.axis_keys, current, app.mouse)


class ParallelView(View):
    hotkey, name = pr.KEY_TWO, "parallel"

    def draw(self, app):
        geometry, mouse = app.geometry, app.mouse
        keys = app.hyper_keys + ["cost", "score"]
        score_to_unit = app.axes["score"].to_unit
        if len(keys) < 2:
            draw_text("need >= 2 axes", geometry.x + 20, geometry.y + 20, 16, GREY)
            return
        column_x = [
            geometry.x + PADDING + i * (geometry.width - 2 * PADDING) / (len(keys) - 1)
            for i in range(len(keys))
        ]
        labels = [short_label(key) for key in keys]
        label_size = 18.0
        label_height = min(
            int(max(text_width(label, label_size) for label in labels)) + 30,
            geometry.height // 2,
        )
        top, bottom = geometry.y + PADDING, geometry.y + geometry.height - label_height
        for index, key in enumerate(keys):
            pr.draw_line(
                int(column_x[index]),
                top,
                int(column_x[index]),
                bottom,
                pr.Color(60, 72, 84, 255),
            )
            draw_text_centered(
                f"{app.axes[key].from_unit(1):.3g}", column_x[index], top - 20, 16, GREY
            )
            draw_text_centered(
                f"{app.axes[key].from_unit(0):.3g}",
                column_x[index],
                bottom + 3,
                16,
                GREY,
            )
            pr.draw_text_pro(
                FONT,
                labels[index],
                pr.Vector2(float(column_x[index]), float(bottom + 20)),
                pr.Vector2(0.0, label_size / 2),
                90.0,
                label_size,
                1.0,
                ACCENT if key in DERIVED_KEYS else WHITE,
            )

        polylines = []
        for index, trial in enumerate(app.trials):
            if trial.get("is_failure") or not app.is_visible(trial):
                continue
            points = []
            for column, key in enumerate(keys):
                unit = app.axes[key].to_unit(value_of(trial, key))
                points.append(
                    None
                    if unit is None
                    else (column_x[column], bottom - unit * (bottom - top))
                )
            polylines.append((index, points, value_of(trial, "score")))

        hovered, hover_distance = None, 9.0
        for index, points, score in polylines:
            for point in points:
                if point is None:
                    continue
                distance = math.hypot(point[0] - mouse.x, point[1] - mouse.y)
                if distance < hover_distance:
                    hover_distance, hovered = distance, index
        app.hovered_trial = hovered
        for index, points, score in polylines:
            if index != hovered:
                color = score_color(score_to_unit(score))
                draw_polyline(points, pr.Color(color.r, color.g, color.b, 70), 1.0)
        if hovered is not None:
            hovered_points = next(
                points for index, points, _ in polylines if index == hovered
            )
            draw_polyline(hovered_points, HIGHLIGHT, 2.5)
            draw_tooltip(
                trial_detail_lines(app.trials[hovered], app.multiple_sources),
                mouse.x + 14,
                mouse.y + 14,
                geometry.window_width,
                geometry.window_height,
            )

    def panel(self, app):
        super().panel(app)
        draw_color_legend()


class ImportanceView(View):
    hotkey, name = pr.KEY_THREE, "importance"

    def draw(self, app):
        geometry = app.geometry
        importances = parameter_importances(app.trials, app.axes, app.hyper_keys)
        draw_text(
            "|correlation| with score", geometry.x + PADDING, geometry.y, 16, ACCENT
        )
        if not importances:
            return

        label_width = 150
        value_width = 56
        top = geometry.y + 44
        bottom = geometry.y + geometry.height - PADDING
        row_height = min(34.0, (bottom - top) / max(len(importances), 1))
        bar_x = geometry.x + PADDING + label_width
        bar_max = geometry.width - 2 * PADDING - label_width - value_width
        rows_bottom = top + len(importances) * row_height

        grid_color = pr.Color(34, 44, 52, 255)
        bar_height = max(int(row_height * 0.2), 4)
        for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
            grid_x = int(bar_x + fraction * bar_max)
            pr.draw_line(grid_x, int(top - 8), grid_x, int(rows_bottom), grid_color)
            draw_text_centered(f"{fraction:.2g}", grid_x, int(top - 24), 12, GREY)

        for index, (key, correlation) in enumerate(importances):
            center_y = top + index * row_height + row_height / 2
            draw_text(short_label(key), geometry.x + PADDING, int(center_y - 8), 15, WHITE)
            bar_width = max(abs(correlation) * bar_max, 2.0)
            pr.draw_rectangle(
                int(bar_x),
                int(center_y - bar_height / 2),
                int(bar_width),
                bar_height,
                GREEN if correlation >= 0 else RED,
            )
            draw_text(f"{correlation:+.2f}", int(bar_x + bar_max + 8), int(center_y - 8), 14, GREY)


class HistoryView(View):
    hotkey, name = pr.KEY_FOUR, "history"

    def draw(self, app):
        geometry, mouse = app.geometry, app.mouse
        if not app.trials:
            return
        score_to_unit = app.axes["score"].to_unit
        top, bottom = geometry.y + PADDING, geometry.y + geometry.height - PADDING
        left, right = geometry.x + PADDING, geometry.x + geometry.width - PADDING
        count = len(app.trials)

        def at_index(index):
            return left + (index / max(count - 1, 1)) * (right - left)

        def at_score(score):
            unit = score_to_unit(score)
            return bottom - (0.0 if unit is None else unit) * (bottom - top)

        pr.draw_rectangle_lines(
            geometry.x, geometry.y, geometry.width, geometry.height, FRAME
        )
        for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
            grid_y = int(bottom - fraction * (bottom - top))
            pr.draw_line(left, grid_y, right, grid_y, pr.Color(28, 38, 46, 255))
            draw_text_right(
                f"{app.axes['score'].from_unit(fraction):.3g}",
                geometry.x - 8,
                grid_y - 8,
                16,
                GREY,
            )

        best, best_points = -float("inf"), []
        for index, trial in enumerate(app.trials):
            score = value_of(trial, "score")
            if trial.get("is_failure") or score is None or not np.isfinite(score):
                continue
            best = max(best, score)
            best_points.append((at_index(index), at_score(best)))
        draw_polyline(best_points, GREEN, 2.0)

        hovered, hover_distance = None, 10.0
        for index, trial in enumerate(app.trials):
            score = value_of(trial, "score")
            if trial.get("is_failure") or score is None or not np.isfinite(score):
                pr.draw_circle_v(pr.Vector2(at_index(index), bottom + 8), 2.5, RED)
                continue
            x, y = at_index(index), at_score(score)
            pr.draw_circle_v(
                pr.Vector2(x, y), 3.0, score_color(score_to_unit(score) or 0.0)
            )
            distance = math.hypot(x - mouse.x, y - mouse.y)
            if distance < hover_distance:
                hover_distance, hovered = distance, index
        app.hovered_trial = hovered

        draw_text("best so far", left + 8, top - 2, 16, GREEN)
        draw_text(
            "trial index", geometry.x + geometry.width // 2 - 40, bottom + 18, 16, GREY
        )
        draw_text_vertical("score", geometry.x - 52, (top + bottom) / 2, 16, GREY)
        if hovered is not None:
            draw_tooltip(
                trial_detail_lines(app.trials[hovered], app.multiple_sources),
                mouse.x + 14,
                mouse.y + 14,
                geometry.window_width,
                geometry.window_height,
            )

    def panel(self, app):
        super().panel(app)
        draw_color_legend()


class ConstellationView(View):
    hotkey, name = pr.KEY_SIX, "constellation"
    STATIC_AXES = ("score", "cost")

    def __init__(self):
        self.axis_keys = None
        self.azimuth, self.elevation, self.distance = 0.7, 0.45, 3.6
        self.texture = None
        self.texture_size = (0, 0)
        self.dragging = False

    def axis_options(self, app):
        return list(self.STATIC_AXES) + app.hyper_keys

    def placements(self, app):
        result = {}
        for index, trial in enumerate(app.trials):
            if trial.get("is_failure") or not app.is_visible(trial):
                continue
            point, ok = [], True
            for key in self.axis_keys:
                unit = app.axes[key].to_unit(value_of(trial, key))
                if unit is None:
                    ok = False
                    break
                point.append(unit * 2.0 - 1.0)
            if ok:
                result[index] = (point[0], point[1], point[2])
        return result

    def update(self, app):
        if self.axis_keys is None:
            ranked = parameter_importances(app.trials, app.axes, app.hyper_keys)
            top = ranked[0][0] if ranked else (app.hyper_keys[0] if app.hyper_keys else "score")
            self.axis_keys = ["cost", "score", top]

    def camera(self):
        cam = pr.Camera3D()
        cam.target = pr.Vector3(0.0, 0.0, 0.0)
        cam.up = pr.Vector3(0.0, 1.0, 0.0)
        cam.fovy = 45.0
        cam.projection = pr.CAMERA_PERSPECTIVE
        cos_elevation = math.cos(self.elevation)
        cam.position = pr.Vector3(
            self.distance * cos_elevation * math.cos(self.azimuth),
            self.distance * math.sin(self.elevation),
            self.distance * cos_elevation * math.sin(self.azimuth),
        )
        return cam

    def ensure_texture(self, width, height):
        if self.texture_size != (width, height):
            if self.texture is not None:
                pr.unload_render_texture(self.texture)
            self.texture = pr.load_render_texture(width, height)
            self.texture_size = (width, height)
        return self.texture

    def draw(self, app):
        geometry, mouse = app.geometry, app.mouse
        placements = self.placements(app)
        over = (
            geometry.x <= mouse.x <= geometry.x + geometry.width
            and geometry.y <= mouse.y <= geometry.y + geometry.height
        )
        if over and pr.is_mouse_button_down(pr.MOUSE_BUTTON_LEFT):
            self.dragging = True
        if not pr.is_mouse_button_down(pr.MOUSE_BUTTON_LEFT):
            self.dragging = False
        if self.dragging:
            delta = pr.get_mouse_delta()
            self.azimuth += delta.x * 0.01
            self.elevation = float(np.clip(self.elevation - delta.y * 0.01, -1.4, 1.4))
        if over:
            self.distance = float(
                np.clip(self.distance - pr.get_mouse_wheel_move() * 0.3, 1.8, 8.0)
            )

        width, height = int(geometry.width), int(geometry.height)
        texture = self.ensure_texture(width, height)
        cam = self.camera()
        score_to_unit = app.axes["score"].to_unit

        pr.begin_texture_mode(texture)
        pr.clear_background(BACKGROUND)
        pr.begin_mode_3d(cam)
        pr.draw_cube_wires(pr.Vector3(0.0, 0.0, 0.0), 2.0, 2.0, 2.0, FRAME)
        origin = pr.Vector3(-1.0, -1.0, -1.0)
        pr.draw_line_3d(origin, pr.Vector3(1.0, -1.0, -1.0), GREEN)
        pr.draw_line_3d(origin, pr.Vector3(-1.0, 1.0, -1.0), BLUE)
        pr.draw_line_3d(origin, pr.Vector3(-1.0, -1.0, 1.0), CYAN)
        for index, point in placements.items():
            color = score_color(score_to_unit(value_of(app.trials[index], "score")))
            pr.draw_sphere_ex(pr.Vector3(*point), 0.02, 6, 6, color)
        pr.end_mode_3d()
        pr.end_texture_mode()

        pr.draw_texture_rec(
            texture.texture,
            pr.Rectangle(0.0, 0.0, float(width), float(-height)),
            pr.Vector2(float(geometry.x), float(geometry.y)),
            WHITE,
        )
        pr.draw_rectangle_lines(
            geometry.x, geometry.y, geometry.width, geometry.height, FRAME
        )
        draw_text(
            f"constellation: {' / '.join(short_label(k) for k in self.axis_keys)}   (color = score)",
            geometry.x + 10,
            geometry.y + 8,
            16,
            ACCENT,
        )
        draw_text(
            "drag to rotate   scroll to zoom   click point to copy   white ring = Pareto",
            geometry.x + 10,
            geometry.y + geometry.height - 22,
            13,
            GREY,
        )

        axis_ends = (
            (pr.Vector3(1.08, -1.0, -1.0), self.axis_keys[0], GREEN),
            (pr.Vector3(-1.0, 1.08, -1.0), self.axis_keys[1], BLUE),
            (pr.Vector3(-1.0, -1.0, 1.08), self.axis_keys[2], CYAN),
        )
        for end_point, key, color in axis_ends:
            screen = pr.get_world_to_screen_ex(end_point, cam, width, height)
            sx, sy = geometry.x + screen.x, geometry.y + screen.y
            if not (
                geometry.x <= sx <= geometry.x + geometry.width
                and geometry.y <= sy <= geometry.y + geometry.height
            ):
                continue
            axis = app.axes[key]
            draw_text_centered(short_label(key), sx, int(sy) - 16, 14, color)
            draw_text_centered(
                f"[{axis.from_unit(0.0):.2g}, {axis.from_unit(1.0):.2g}]",
                sx,
                int(sy),
                11,
                GREY,
            )

        for index in app.pareto:
            if index not in placements:
                continue
            screen = pr.get_world_to_screen_ex(
                pr.Vector3(*placements[index]), cam, width, height
            )
            sx, sy = geometry.x + screen.x, geometry.y + screen.y
            if (
                geometry.x <= sx <= geometry.x + geometry.width
                and geometry.y <= sy <= geometry.y + geometry.height
            ):
                pr.draw_circle_lines_v(pr.Vector2(sx, sy), 7, WHITE)

        hovered, hover_distance = None, 12.0
        for index, point in placements.items():
            screen = pr.get_world_to_screen_ex(pr.Vector3(*point), cam, width, height)
            sx, sy = geometry.x + screen.x, geometry.y + screen.y
            distance = math.hypot(sx - mouse.x, sy - mouse.y)
            if distance < hover_distance:
                hover_distance, hovered = distance, index
        app.hovered_trial = hovered
        if hovered is not None:
            draw_tooltip(
                trial_detail_lines(app.trials[hovered], app.multiple_sources),
                mouse.x + 14,
                mouse.y + 14,
                geometry.window_width,
                geometry.window_height,
            )

    def panel(self, app):
        for index, label in enumerate(("X", "Y", "Z")):
            draw_selector_button(
                X_BUTTON_Y + index * 28,
                label,
                short_label(self.axis_keys[index]),
                (GREEN, BLUE, CYAN)[index],
                14,
            )
        draw_text("axis - click to change", 14, X_BUTTON_Y + 3 * 28 + 6, 13, GREY)

    def click(self, app):
        mouse_x, mouse_y = app.mouse.x, app.mouse.y
        for index in range(3):
            if selector_hit(mouse_x, mouse_y, X_BUTTON_Y + index * 28):
                app.menu = None if app.menu == f"caxis{index}" else f"caxis{index}"
                return
        for index in range(3):
            if app.menu == f"caxis{index}":
                options = self.axis_options(app)
                hit = dropdown_index(mouse_y, X_BUTTON_Y + index * 28 + BUTTON_HEIGHT)
                if 14 <= mouse_x <= 14 + MENU_WIDTH and 0 <= hit < len(options):
                    self.axis_keys[index] = options[hit]
                app.menu = None
                return

    def menu(self, app):
        for index in range(3):
            if app.menu == f"caxis{index}":
                draw_dropdown(
                    X_BUTTON_Y + index * 28 + BUTTON_HEIGHT,
                    self.axis_options(app),
                    self.axis_keys[index],
                    app.mouse,
                    short_label,
                )


class ModelView(View):
    hotkey, name = pr.KEY_FIVE, "model"

    def update(self, app):
        self.grid = app.surrogate.display(app.trials)

    def handle_keys(self, app):
        if pr.is_key_pressed(pr.KEY_M):
            app.show_uncertainty = not app.show_uncertainty

    def draw(self, app):
        draw_surrogate(app, self.grid)

    def panel(self, app):
        super().panel(app)
        shown = "uncertainty" if app.show_uncertainty else "predicted score"
        draw_text("GP surrogate (PCA)", 14, LEGEND_Y, 16, ACCENT)
        draw_text(f"showing: {shown}", 14, LEGEND_Y + 22, 14, WHITE)
        draw_text("m metric", 14, LEGEND_Y + 42, 13, GREY)


def copy_text_to_clipboard(text):
    data = text.encode()
    try:
        if sys.platform == "darwin":
            subprocess.run(["pbcopy"], input=data, check=True)
            return True
        if os.environ.get("WAYLAND_DISPLAY") and shutil.which("wl-copy"):
            subprocess.run(["wl-copy"], input=data, check=True)
            return True
        if shutil.which("xclip"):
            subprocess.run(
                ["xclip", "-selection", "clipboard"], input=data, check=True
            )
            return True
    except Exception:
        return False
    return False


def copy_screenshot_to_clipboard():
    pr.take_screenshot("visualize_hyperparameters.png")
    path = os.path.abspath("visualize_hyperparameters.png")
    if not os.path.exists(path):
        return "screenshot failed"
    try:
        if sys.platform == "darwin":
            script = f'set the clipboard to (read (POSIX file "{path}") as «class PNGf»)'
            subprocess.run(["osascript", "-e", script], check=True)
            return "copied to clipboard"
        if os.environ.get("WAYLAND_DISPLAY") and shutil.which("wl-copy"):
            with open(path, "rb") as file:
                subprocess.run(
                    ["wl-copy", "--type", "image/png"], stdin=file, check=True
                )
            return "copied to clipboard"
        if shutil.which("xclip"):
            subprocess.run(
                ["xclip", "-selection", "clipboard", "-t", "image/png", "-i", path],
                check=True,
            )
            return "copied to clipboard"
        return "install xclip or wl-copy"
    except Exception:
        return "clipboard copy failed"
    finally:
        os.remove(path)


class App:
    def __init__(self, paths, window_width, window_height, wandb_spec=None):
        self.paths = paths
        self.wandb_spec = wandb_spec
        self.spaces = {} if wandb_spec else load_spaces(paths[0])
        self.multiple_sources = wandb_spec is not None or len(paths) > 1
        self.window_width, self.window_height = window_width, window_height
        self.load_data()
        self.x_key, self.y_key = "cost", "score"
        self.selected = None
        self.menu = None
        self.show_uncertainty = False
        self.toast, self.toast_frames, self.take_screenshot = "", 0, False
        self.views = [
            ScatterView(),
            ParallelView(),
            ImportanceView(),
            HistoryView(),
            ConstellationView(),
        ]
        if wandb_spec is None:
            self.surrogate = Surrogate(paths)
            self.surrogate.start(self.trials, self.spaces)
            self.views.append(ModelView())
        else:
            self.surrogate = None
        self.view_index = 0
        self.panel_collapsed = False
        self.panel_anim = 0.0
        self.top_k = None
        self.dragging_slider = False
        self.visible_total = 0
        self.visible_ids = None
        self.hovered_trial = None
        self.geometry = self.mouse = None

    def load_data(self):
        trials = (
            load_wandb_trials(*self.wandb_spec)
            if self.wandb_spec
            else load_trials(self.paths)
        )
        (
            self.trials,
            self.axis_keys,
            self.hyper_keys,
            self.axes,
            self.pareto,
            self.sources,
        ) = prepare(trials, self.spaces)

    def reload(self):
        self.load_data()
        self.selected = None
        if self.surrogate is not None:
            self.surrogate.invalidate()
            self.surrogate.start(self.trials, self.spaces)
        self.toast, self.toast_frames = f"reloaded {len(self.trials)} trials", 120

    @property
    def view(self):
        return self.views[self.view_index]

    def run(self):
        pr.init_window(
            self.window_width, self.window_height, "visualize_hyperparameters"
        )
        pr.set_target_fps(60)
        global FONT
        font_path = str(
            Path(__file__).resolve().parent.parent
            / "src"
            / "utils"
            / "JetBrainsMono-Regular.ttf"
        )
        FONT = pr.load_font_ex(font_path, 48, None, 0)
        pr.set_texture_filter(FONT.texture, pr.TEXTURE_FILTER_BILINEAR)
        while not pr.window_should_close():
            self.update()
            self.render()
        pr.close_window()

    def update(self):
        if pr.is_key_pressed(pr.KEY_R):
            try:
                self.reload()
            except Exception as error:
                self.toast, self.toast_frames = f"reload failed: {error}", 180
        for index, view in enumerate(self.views):
            if pr.is_key_pressed(view.hotkey):
                self.view_index, self.menu = index, None
        if pr.is_key_pressed(pr.KEY_TAB):
            self.view_index, self.menu = (self.view_index + 1) % len(self.views), None
        if pr.is_key_pressed(pr.KEY_S):
            self.take_screenshot = True

        target = 1.0 if self.panel_collapsed else 0.0
        step = pr.get_frame_time() / 0.18
        if self.panel_anim < target:
            self.panel_anim = min(target, self.panel_anim + step)
        else:
            self.panel_anim = max(target, self.panel_anim - step)
        expanded_left, collapsed_left = PANEL_WIDTH + 74, PANEL_COLLAPSED_LEFT
        x, y = int(expanded_left + (collapsed_left - expanded_left) * self.panel_eased()), 46
        self.geometry = Geometry(
            x,
            y,
            self.window_width - x - 16,
            self.window_height - y - 48,
            self.window_width,
            self.window_height,
        )
        self.mouse = pr.get_mouse_position()
        self.update_filter()
        self.handle_slider()
        self.view.handle_keys(self)
        self.view.update(self)
        if pr.is_mouse_button_pressed(pr.MOUSE_BUTTON_LEFT):
            if self.panel_toggle_hit():
                self.panel_collapsed, self.menu = not self.panel_collapsed, None
            else:
                if self.hovered_trial is not None:
                    self.copy_config(self.hovered_trial)
                if not self.panel_collapsed:
                    self.view.click(self)

    def panel_eased(self):
        a = self.panel_anim
        return a * a * (3.0 - 2.0 * a)

    def update_filter(self):
        non_failures = [t for t in self.trials if not t.get("is_failure")]
        self.visible_total = len(non_failures)
        if self.top_k is None or self.top_k >= self.visible_total:
            self.visible_ids = None
        else:
            ranked = sorted(non_failures, key=lambda t: t["score"], reverse=True)
            self.visible_ids = {id(t) for t in ranked[: self.top_k]}

    def is_visible(self, trial):
        return self.visible_ids is None or id(trial) in self.visible_ids

    def slider_geometry(self):
        return 16, self.window_height - 86, PANEL_WIDTH - 32

    def handle_slider(self):
        if self.panel_collapsed or self.panel_anim != 0.0 or self.visible_total <= 1:
            self.dragging_slider = False
            return
        x, y, width = self.slider_geometry()
        over = x - 6 <= self.mouse.x <= x + width + 6 and y - 10 <= self.mouse.y <= y + 14
        if pr.is_mouse_button_pressed(pr.MOUSE_BUTTON_LEFT) and over:
            self.dragging_slider = True
        if not pr.is_mouse_button_down(pr.MOUSE_BUTTON_LEFT):
            self.dragging_slider = False
        if self.dragging_slider:
            fraction = float(np.clip((self.mouse.x - x) / width, 0.0, 1.0))
            k = 1 + round(fraction * (self.visible_total - 1))
            self.top_k = None if k >= self.visible_total else k

    def draw_slider(self):
        if self.visible_total <= 1:
            return
        x, y, width = self.slider_geometry()
        k = self.visible_total if self.top_k is None else self.top_k
        fraction = (k - 1) / (self.visible_total - 1)
        knob_x = int(x + fraction * width)
        label = (
            f"showing all {self.visible_total}"
            if self.top_k is None
            else f"showing top {k} / {self.visible_total}"
        )
        draw_text(label, x, y - 22, 14, GREY)
        pr.draw_rectangle(x, y, width, 4, pr.Color(40, 52, 62, 255))
        pr.draw_rectangle(x, y, knob_x - x, 4, ACCENT)
        pr.draw_circle(knob_x, y + 2, 7.0, ACCENT)
        pr.draw_circle(knob_x, y + 2, 3.5, PANEL)

    def panel_toggle_hit(self):
        x, y, width, height = PANEL_TOGGLE
        return x <= self.mouse.x <= x + width and y <= self.mouse.y <= y + height

    def draw_panel_toggle(self):
        x, y, width, height = PANEL_TOGGLE
        hovered = self.panel_toggle_hit()
        if hovered:
            pr.draw_rectangle_rounded(
                pr.Rectangle(float(x), float(y), float(width), float(height)),
                0.3,
                6,
                pr.Color(255, 255, 255, 20),
            )
        color = WHITE if hovered else pr.Color(180, 192, 204, 255)
        inset_x, inset_y = x + 4, y + 5
        inset_w, inset_h = width - 8, height - 10
        divider = inset_x + int(inset_w * 0.36)
        pr.draw_rectangle(
            inset_x + 1,
            inset_y + 1,
            divider - inset_x - 1,
            inset_h - 2,
            pr.Color(color.r, color.g, color.b, 70),
        )
        pr.draw_rectangle_lines_ex(
            pr.Rectangle(float(inset_x), float(inset_y), float(inset_w), float(inset_h)),
            1.0,
            color,
        )
        pr.draw_rectangle(divider, inset_y + 1, 1, inset_h - 2, color)

    def copy_config(self, index):
        trial = self.trials[index]
        overrides = [
            f"{key}={value}" for key, value in trial["hyperparameters"].items()
        ]
        config = OmegaConf.from_dotlist(overrides)
        copied = copy_text_to_clipboard(OmegaConf.to_yaml(config))
        self.toast = "copied config to clipboard" if copied else "clipboard copy failed"
        self.toast_frames = 120

    def render(self):
        pr.begin_drawing()
        pr.clear_background(BACKGROUND)
        self.hovered_trial = None
        self.view.draw(self)

        eased = self.panel_eased()
        if eased < 0.999:
            camera = pr.Camera2D()
            camera.offset = pr.Vector2(-PANEL_WIDTH * eased, 0.0)
            camera.target = pr.Vector2(0.0, 0.0)
            camera.rotation = 0.0
            camera.zoom = 1.0
            pr.begin_mode_2d(camera)
            pr.draw_rectangle(0, 0, PANEL_WIDTH, self.window_height, PANEL)
            draw_text(f"{len(self.trials)} trials", 48, 16, 20, WHITE)
            self.view.panel(self)
            draw_text("1scat 2para 3imp 4hist 5mdl 6con", 14, self.window_height - 44, 13, GREY)
            draw_text("tab cycle  r reload  s shot", 14, self.window_height - 24, 13, GREY)
            if self.multiple_sources:
                draw_text("sources:", 14, LEGEND_Y + 90, 14, GREY)
                for index, source in enumerate(self.sources):
                    draw_text(source, 22, LEGEND_Y + 110 + index * 18, 14, WHITE)
            self.draw_slider()
            self.view.menu(self)
            pr.end_mode_2d()
        self.draw_panel_toggle()

        if self.toast_frames > 0:
            draw_tooltip(
                [self.toast],
                self.geometry.x + self.geometry.width // 2 - 80,
                self.geometry.y + 10,
                self.window_width,
                self.window_height,
            )
            self.toast_frames -= 1
        pr.end_drawing()

        if self.take_screenshot:
            self.toast = copy_screenshot_to_clipboard()
            self.toast_frames, self.take_screenshot = 120, False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*")
    parser.add_argument("--wandb-sweep", help="load trials with this config.sweep value")
    parser.add_argument("--wandb-project", default="recurrent-streaming-rl")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--width", type=int, default=1400)
    parser.add_argument("--height", type=int, default=900)
    arguments = parser.parse_args()
    if arguments.wandb_sweep:
        wandb_spec = (
            arguments.wandb_entity,
            arguments.wandb_project,
            arguments.wandb_sweep,
        )
        App(None, arguments.width, arguments.height, wandb_spec=wandb_spec).run()
    else:
        if not arguments.paths:
            parser.error("paths required unless --wandb-sweep is given")
        App(resolve_paths(arguments.paths), arguments.width, arguments.height).run()


if __name__ == "__main__":
    main()
