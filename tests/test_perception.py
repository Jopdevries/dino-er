from __future__ import annotations

import math

import numpy as np
import pytest

from dino_er.perception import (
    SENSORY_NAMES,
    SENSORY_SCHEMA,
    VisualPerception,
    process_population_pixel_components,
)


def _frame(
    *,
    dino_y: int = 95,
    obstacle_x: int | None = 300,
    game_over: bool = False,
) -> np.ndarray:
    frame = np.full((150, 600, 3), 247, dtype=np.uint8)
    frame[127:129, :] = 83
    frame[dino_y : dino_y + 43, 50:90] = 83
    if obstacle_x is not None:
        frame[105:140, obstacle_x : obstacle_x + 17] = 83
    if game_over:
        frame[40:55, 210:390] = 83
        frame[75:105, 282:318] = 83
    return frame


def test_sensory_vector_is_current_frame_only() -> None:
    current = _frame(dino_y=75, obstacle_x=288)
    perception_with_history = VisualPerception()
    perception_with_history.process(_frame(dino_y=95, obstacle_x=350))
    after_other_frame = perception_with_history.process(current)
    fresh = VisualPerception().process(current)
    np.testing.assert_allclose(after_other_frame.sensory, fresh.sensory)
    assert after_other_frame.estimate == fresh.estimate
    prohibited = ("speed", "velocity", "ttc", "previous", "flow", "stack")
    assert not any(token in name for name in SENSORY_NAMES for token in prohibited)


def test_schema_documents_every_normalised_feature_and_missing_value() -> None:
    assert tuple(sensor.name for sensor in SENSORY_SCHEMA) == SENSORY_NAMES
    assert len(SENSORY_SCHEMA) == 5
    for sensor in SENSORY_SCHEMA:
        assert sensor.detection
        assert sensor.raw_unit
        assert sensor.normalization
        assert sensor.normalized_range == (0.0, 1.0)
        assert sensor.missing_value == 0.0


def test_missing_pixels_use_zero_sentinels_not_fabricated_state() -> None:
    blank = np.full((150, 600, 3), 247, dtype=np.uint8)
    result = VisualPerception().process(blank)
    assert not any(result.validity.values())
    assert math.isnan(result.estimate.dino_y)
    assert math.isnan(result.estimate.obstacle_relative_x)
    assert np.isfinite(result.sensory).all()
    assert np.all(result.sensory == 0.0)


def test_visual_game_over_is_external() -> None:
    frame = _frame(game_over=True)
    original = frame.copy()
    result = VisualPerception().process(frame)
    assert result.terminated
    assert result.reward == pytest.approx(-1.0)
    np.testing.assert_array_equal(frame, original)


def test_inverted_night_frame_has_the_same_visual_estimates() -> None:
    day = _frame(dino_y=75, obstacle_x=288)
    night = 255 - day
    day_result = VisualPerception().process(day)
    night_result = VisualPerception().process(night)
    assert night_result.validity == day_result.validity
    assert night_result.dino_box == day_result.dino_box
    assert night_result.obstacle_boxes == day_result.obstacle_boxes
    np.testing.assert_allclose(night_result.sensory, day_result.sensory)


def test_invalid_frame_shape_is_rejected() -> None:
    with pytest.raises(ValueError):
        VisualPerception().process(np.zeros((10, 10, 3), dtype=np.uint8))


def test_compact_pixel_components_match_complete_private_frames() -> None:
    shared = np.full((150, 600, 3), 247, dtype=np.uint8)
    shared[127:129, :] = 83
    shared[105:140, 300:317] = 83
    first = shared.copy()
    first[70:113, 50:90] = 83
    second = shared.copy()
    second[80:123, 50:90] = 83
    frames = np.stack([first, second])

    complete = [VisualPerception(create_overlay=False).process(frame) for frame in frames]
    compact = process_population_pixel_components(
        [VisualPerception(create_overlay=False) for _ in range(2)],
        shared[:, :, 0],
        frames[:, :, 50:110, 0],
        frames[:, 42:53, 205:396, 0],
        frames[:, 75:107, 284:320, 0],
    )

    for complete_result, compact_result in zip(complete, compact, strict=True):
        np.testing.assert_allclose(compact_result.sensory, complete_result.sensory)
        assert compact_result.estimate == complete_result.estimate
        assert compact_result.dino_box == complete_result.dino_box
        assert compact_result.obstacle_boxes == complete_result.obstacle_boxes
        assert compact_result.terminated == complete_result.terminated
        assert compact_result.reward == complete_result.reward
