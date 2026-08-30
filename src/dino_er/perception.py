"""Current-frame visual perception for clean Chrome Dino Canvas frames.

The controller sensory vector is derived from exactly one rendered RGB frame.
The only temporal state in this module is a private previous-obstacle box used
for passed-obstacle diagnostics. It is never exposed to a controller, never
changes the sensory vector and is not part of the survival-time objective.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from PIL import Image, ImageDraw

Pose = Literal["running", "jumping", "ducking"]
ObstacleClass = Literal["small_cactus", "large_cactus", "pterodactyl"]
OBSERVATION_PROVENANCE = {"observation_source": "current_rendered_frame_pixels_only"}


@dataclass(frozen=True)
class SensorDefinition:
    """Auditable definition of one current-frame controller input."""

    name: str
    detection: str
    raw_unit: str
    normalization: str
    normalized_range: tuple[float, float]
    missing_value: float


SENSORY_SCHEMA = (
    SensorDefinition(
        "obstacle_relative_x",
        "nearest current-frame obstacle left edge minus Dino right edge",
        "pixels",
        "clip to [-100, 600], then affine-map to [0, 1]",
        (0.0, 1.0),
        0.0,
    ),
    SensorDefinition(
        "obstacle_relative_y",
        "nearest current-frame obstacle top minus Dino top",
        "pixels",
        "clip to [-150, 150], then affine-map to [0, 1]",
        (0.0, 1.0),
        0.0,
    ),
    SensorDefinition(
        "obstacle_width",
        "width of nearest current-frame obstacle bounding box",
        "pixels",
        "clip to [0, 100], then divide by 100",
        (0.0, 1.0),
        0.0,
    ),
    SensorDefinition(
        "obstacle_height",
        "height of nearest current-frame obstacle bounding box",
        "pixels",
        "clip to [0, 70], then divide by 70",
        (0.0, 1.0),
        0.0,
    ),
    SensorDefinition(
        "dino_y",
        "top edge of current-frame Dino bounding box",
        "pixels",
        "clip to [0, 150], then divide by 150",
        (0.0, 1.0),
        0.0,
    ),
)
SENSORY_NAMES = tuple(sensor.name for sensor in SENSORY_SCHEMA)


@dataclass(frozen=True)
class BoundingBox:
    """Inclusive pixel bounding box in the 600 by 150 clean game frame."""

    x: int
    y: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.x + self.width - 1

    @property
    def bottom(self) -> int:
        return self.y + self.height - 1


@dataclass(frozen=True)
class VisualEstimate:
    """Unnormalised quantities measured only in the current frame."""

    obstacle_relative_x: float
    obstacle_relative_y: float
    obstacle_width: float
    obstacle_height: float
    dino_y: float
    dino_pose: Pose | None
    obstacle_class: ObstacleClass | None


@dataclass(frozen=True)
class PerceptionResult:
    """Current-frame sensory result plus externally observed episode events."""

    sensory: np.ndarray
    estimate: VisualEstimate
    dino_box: BoundingBox | None
    obstacle_boxes: tuple[BoundingBox, ...]
    nearest_obstacle: BoundingBox | None
    validity: dict[str, bool]
    failure_reason: str | None
    reward: float
    terminated: bool
    processed_frame: np.ndarray | None


def _boxes_from_pixels(mask: np.ndarray) -> list[BoundingBox]:
    working = mask.copy()
    horizon_threshold = min(180, int(working.shape[1] * 0.8))
    working[np.count_nonzero(working, axis=1) > horizon_threshold] = False

    # Exact 8-connected components via horizontal runs. A Dino frame is
    # sparse; joining a few hundred runs is substantially cheaper than
    # storing and flood-filling every foreground pixel in Python.
    padded = np.pad(working, ((0, 0), (1, 1)), constant_values=False)
    transitions = np.diff(padded.astype(np.int8, copy=False), axis=1)
    starts = np.argwhere(transitions == 1)
    ends = np.argwhere(transitions == -1)
    runs: list[tuple[int, int, int]] = []
    rows: list[list[int]] = [[] for _ in range(working.shape[0])]
    parents: list[int] = []

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def join_runs(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for start, end in zip(starts, ends, strict=True):
        y = int(start[0])
        if y != int(end[0]):
            raise RuntimeError("Foreground run transition rows do not match")
        x_start = int(start[1])
        x_end = int(end[1]) - 1
        index = len(runs)
        runs.append((y, x_start, x_end))
        parents.append(index)
        rows[y].append(index)
        if y == 0:
            continue
        for previous in rows[y - 1]:
            _, previous_start, previous_end = runs[previous]
            if previous_start <= x_end + 1 and x_start <= previous_end + 1:
                join_runs(index, previous)

    aggregates: dict[int, list[int]] = {}
    for index, (y, x_start, x_end) in enumerate(runs):
        root = find(index)
        area = x_end - x_start + 1
        if root not in aggregates:
            aggregates[root] = [x_start, y, x_end, y, area]
        else:
            aggregate = aggregates[root]
            aggregate[0] = min(aggregate[0], x_start)
            aggregate[1] = min(aggregate[1], y)
            aggregate[2] = max(aggregate[2], x_end)
            aggregate[3] = max(aggregate[3], y)
            aggregate[4] += area

    components = [
        (
            BoundingBox(
                min_x,
                min_y,
                max_x - min_x + 1,
                max_y - min_y + 1,
            ),
            area,
        )
        for min_x, min_y, max_x, max_y, area in aggregates.values()
        if area >= 3
    ]

    merged = True
    while merged:
        merged = False
        for left_index, (left, left_area) in enumerate(components):
            for right_index in range(left_index + 1, len(components)):
                right, right_area = components[right_index]
                horizontal_overlap = left.x <= right.right and right.x <= left.right
                vertical_overlap = left.y <= right.bottom and right.y <= left.bottom
                vertical_gap = max(left.y - right.bottom - 1, right.y - left.bottom - 1)
                horizontal_gap = max(left.x - right.right - 1, right.x - left.right - 1)
                if not (
                    (horizontal_overlap and vertical_gap <= 4)
                    or (vertical_overlap and horizontal_gap <= 2)
                ):
                    continue
                union = BoundingBox(
                    min(left.x, right.x),
                    min(left.y, right.y),
                    max(left.right, right.right) - min(left.x, right.x) + 1,
                    max(left.bottom, right.bottom) - min(left.y, right.y) + 1,
                )
                components[left_index] = (union, left_area + right_area)
                del components[right_index]
                merged = True
                break
            if merged:
                break
    return [box for box, area in components if area >= 18 and box.width >= 4 and box.height >= 4]


class VisualPerception:
    """Extract the documented current-frame controller sensory vector."""

    sensory_low = np.array(
        [-100, -150, 0, 0, 0],
        dtype=np.float32,
    )
    sensory_high = np.array(
        [600, 150, 100, 70, 150],
        dtype=np.float32,
    )

    def __init__(self, *, create_overlay: bool = True) -> None:
        self._previous_obstacle_for_pass_detection: BoundingBox | None = None
        self._create_overlay = create_overlay

    def reset(self) -> None:
        self._previous_obstacle_for_pass_detection = None

    def process(self, frame: np.ndarray) -> PerceptionResult:
        if frame.shape != (150, 600, 3) or frame.dtype != np.uint8:
            raise ValueError(f"Expected uint8 RGB frame (150, 600, 3), received {frame.shape}")
        # Chromium's clean Dino canvas is monochrome RGB, so one channel is
        # exactly the same intensity as averaging all three. Sampling the
        # guaranteed background corner avoids a full-frame median on every
        # candidate and keeps 100-environment batches responsive.
        grayscale = frame[:, :, 0]
        background = int(grayscale[0, 0])
        foreground = (
            grayscale < background - 30 if background >= 128 else grayscale > background + 30
        )
        foreground[:26] = False
        boxes = _boxes_from_pixels(foreground)

        dino_candidates = [
            box
            for box in boxes
            if 28 <= box.x <= 110 and box.right <= 125 and box.height >= 18 and box.width >= 15
        ]
        dino_box = max(
            dino_candidates,
            key=lambda box: box.width * box.height,
            default=None,
        )
        obstacles = tuple(
            box
            for box in boxes
            if box != dino_box and box.width <= 100 and box.height >= 16 and box.y >= 35
        )
        terminated = self._detect_game_over(foreground)
        processed = (
            self._render_processed_frame(
                frame,
                dino_box,
                obstacles,
                min(
                    (box for box in obstacles if dino_box is None or box.right >= dino_box.x),
                    key=lambda box: box.x,
                    default=None,
                ),
            )
            if self._create_overlay
            else frame
        )
        return self._result(dino_box, obstacles, terminated, processed)

    def _result(
        self,
        dino: BoundingBox | None,
        obstacles: tuple[BoundingBox, ...],
        terminated: bool,
        processed: np.ndarray | None,
    ) -> PerceptionResult:
        nearest = min(
            (box for box in obstacles if dino is None or box.right >= dino.x),
            key=lambda box: box.x,
            default=None,
        )
        pose = self._pose(dino)
        obstacle_class = self._obstacle_class(nearest)
        estimate = VisualEstimate(
            float(nearest.x - dino.right) if nearest is not None and dino is not None else math.nan,
            float(nearest.y - dino.y) if nearest is not None and dino is not None else math.nan,
            float(nearest.width) if nearest is not None else math.nan,
            float(nearest.height) if nearest is not None else math.nan,
            float(dino.y) if dino is not None else math.nan,
            pose,
            obstacle_class,
        )
        validity = {
            "dino": dino is not None,
            "obstacle": nearest is not None,
            "pose": pose is not None,
            "class": obstacle_class is not None,
        }
        missing = [name for name, valid in validity.items() if not valid]
        reward = -1.0 if terminated else self._detect_pass(dino, nearest)
        self._previous_obstacle_for_pass_detection = nearest
        return PerceptionResult(
            sensory=self._normalise(estimate),
            estimate=estimate,
            dino_box=dino,
            obstacle_boxes=obstacles,
            nearest_obstacle=nearest,
            validity=validity,
            failure_reason=(
                "unavailable current-frame estimates: " + ", ".join(missing) if missing else None
            ),
            reward=reward,
            terminated=terminated,
            processed_frame=processed,
        )

    @staticmethod
    def _pose(dino: BoundingBox | None) -> Pose | None:
        if dino is None:
            return None
        if dino.width >= 50:
            return "ducking"
        if dino.bottom < 137:
            return "jumping"
        return "running"

    @staticmethod
    def _obstacle_class(obstacle: BoundingBox | None) -> ObstacleClass | None:
        if obstacle is None:
            return None
        if obstacle.bottom < 124:
            return "pterodactyl"
        if obstacle.height > 40:
            return "large_cactus"
        return "small_cactus"

    @staticmethod
    def _detect_game_over(foreground: np.ndarray) -> bool:
        text_pixels = int(np.count_nonzero(foreground[39:58, 198:402]))
        restart_pixels = int(np.count_nonzero(foreground[70:112, 276:324]))
        return text_pixels > 180 and restart_pixels > 80

    def _detect_pass(
        self,
        dino: BoundingBox | None,
        obstacle: BoundingBox | None,
    ) -> float:
        if dino is None or self._previous_obstacle_for_pass_detection is None:
            return 0.0
        previous_was_close = self._previous_obstacle_for_pass_detection.x <= dino.right + 25
        switched_to_new = (
            obstacle is None or obstacle.x > self._previous_obstacle_for_pass_detection.x + 100
        )
        return 1.0 if previous_was_close and switched_to_new else 0.0

    def _normalise(
        self,
        estimate: VisualEstimate,
    ) -> np.ndarray:
        raw = np.array(
            [
                estimate.obstacle_relative_x,
                estimate.obstacle_relative_y,
                estimate.obstacle_width,
                estimate.obstacle_height,
                estimate.dino_y,
            ],
            dtype=np.float32,
        )
        unavailable = ~np.isfinite(raw)
        clipped = np.clip(
            np.nan_to_num(raw, nan=0.0),
            self.sensory_low,
            self.sensory_high,
        )
        sensory = (clipped - self.sensory_low) / (self.sensory_high - self.sensory_low)
        sensory[unavailable] = 0.0
        return sensory.astype(np.float32, copy=False)

    @staticmethod
    def _render_processed_frame(
        frame: np.ndarray,
        dino: BoundingBox | None,
        obstacles: tuple[BoundingBox, ...],
        nearest: BoundingBox | None,
    ) -> np.ndarray:
        image = Image.fromarray(frame.copy(), mode="RGB")
        draw = ImageDraw.Draw(image)
        if dino:
            draw.rectangle(
                (dino.x, dino.y, dino.right, dino.bottom),
                outline=(0, 120, 255),
                width=1,
            )
        for obstacle in obstacles:
            colour = (255, 80, 0) if obstacle == nearest else (255, 180, 0)
            draw.rectangle(
                (obstacle.x, obstacle.y, obstacle.right, obstacle.bottom),
                outline=colour,
                width=1,
            )
        return np.asarray(image, dtype=np.uint8)


def process_population_pixel_components(
    perceptions: list[VisualPerception],
    shared_world: np.ndarray,
    dino_patches: np.ndarray,
    game_over_text_patches: np.ndarray,
    restart_patches: np.ndarray,
) -> list[PerceptionResult]:
    """Read compact rendered pixels without materialising 100 complete frames."""

    count = len(perceptions)
    if (
        shared_world.shape != (150, 600)
        or dino_patches.shape != (count, 150, 60)
        or game_over_text_patches.shape != (count, 11, 191)
        or restart_patches.shape != (count, 32, 36)
        or shared_world.dtype != np.uint8
        or dino_patches.dtype != np.uint8
        or game_over_text_patches.dtype != np.uint8
        or restart_patches.dtype != np.uint8
    ):
        raise ValueError("Invalid shared-world private-pixel component batch")

    shared_background = int(shared_world[0, 0])
    shared_foreground = (
        shared_world < shared_background - 30
        if shared_background >= 128
        else shared_world > shared_background + 30
    )
    shared_foreground[:26] = False
    obstacle_boxes = tuple(
        box
        for box in _boxes_from_pixels(shared_foreground)
        if box.width <= 100 and box.height >= 16 and box.y >= 35
    )

    # The browser sends final rendered pixel crops, not player coordinates.
    # Differencing each crop against the rendered shared world isolates the
    # candidate sprite without mistaking the horizon or a nearby obstacle for
    # the Dino. This remains a pixel-only visual operation.
    dino_foreground = dino_patches != shared_world[None, :, 50:110]
    dino_rows = np.any(dino_foreground, axis=2)
    dino_columns = np.any(dino_foreground, axis=1)
    dino_boxes: list[BoundingBox | None] = []
    for index in range(count):
        y_values = np.flatnonzero(dino_rows[index])
        x_values = np.flatnonzero(dino_columns[index])
        if y_values.size == 0 or x_values.size == 0:
            dino_boxes.append(None)
            continue
        min_y = int(y_values[0])
        max_y = int(y_values[-1])
        min_x = int(x_values[0]) + 50
        max_x = int(x_values[-1]) + 50
        dino_boxes.append(
            BoundingBox(
                x=min_x,
                y=min_y,
                width=max_x - min_x + 1,
                height=max_y - min_y + 1,
            )
        )

    game_over_text_foreground = game_over_text_patches != shared_world[None, 42:53, 205:396]
    restart_foreground = restart_patches != shared_world[None, 75:107, 284:320]
    text_pixels = np.count_nonzero(
        game_over_text_foreground,
        axis=(1, 2),
    )
    restart_pixels = np.count_nonzero(
        restart_foreground,
        axis=(1, 2),
    )
    terminated = (text_pixels > 180) & (restart_pixels > 80)

    results: list[PerceptionResult] = []
    for index, perception in enumerate(perceptions):
        results.append(
            perception._result(
                dino_boxes[index],
                obstacle_boxes,
                bool(terminated[index]),
                None,
            )
        )
    return results
