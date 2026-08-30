from __future__ import annotations

import numpy as np
import pytest

from dino_er.controllers import (
    ControllerSpec,
    ControllerType,
    CTRNNController,
    PopulationControllerRuntime,
    ReactiveController,
    action_from_scores,
    build_controller,
    default_controller_spec,
    resolve_accelerator,
    unflatten_genome,
    validate_genome,
)
from dino_er.perception import SENSORY_NAMES


def _spec(controller_type: ControllerType) -> ControllerSpec:
    return default_controller_spec(controller_type, len(SENSORY_NAMES))


def test_default_controller_parameter_counts_are_closely_matched() -> None:
    reactive = _spec("reactive")
    proactive = _spec("proactive")
    assert reactive.parameter_count == proactive.parameter_count == 93


def test_reactive_controller_is_deterministic_and_has_no_state() -> None:
    spec = _spec("reactive")
    genome = np.random.default_rng(4).normal(size=spec.parameter_count)
    controller = ReactiveController(genome, spec)
    sensory = np.linspace(0.0, 1.0, spec.input_size)
    first_action, first_scores = controller.act_with_scores(sensory)
    controller.reset(99)
    second_action, second_scores = controller.act_with_scores(sensory)
    assert controller.hidden_state is None
    assert first_action == second_action
    np.testing.assert_array_equal(first_scores, second_scores)


def test_proactive_state_evolves_and_resets_to_zero() -> None:
    spec = _spec("proactive")
    genome = np.random.default_rng(5).normal(size=spec.parameter_count)
    controller = CTRNNController(genome, spec)
    sensory = np.linspace(0.1, 0.9, spec.input_size)
    assert np.all(controller.hidden_state == 0.0)
    controller.act_with_scores(sensory)
    first_hidden = controller.hidden_state
    assert not np.all(first_hidden == 0.0)
    controller.act_with_scores(sensory)
    assert not np.array_equal(controller.hidden_state, first_hidden)
    controller.reset()
    assert np.all(controller.hidden_state == 0.0)


def test_proactive_candidates_do_not_share_parameters_or_hidden_state() -> None:
    spec = _spec("proactive")
    genome = np.random.default_rng(51).normal(size=spec.parameter_count)
    first = CTRNNController(genome, spec)
    second = CTRNNController(genome, spec)
    second_recurrent = second.parameters["w_rec"].copy()

    assert not np.shares_memory(
        first.parameters["w_rec"],
        second.parameters["w_rec"],
    )
    first.parameters["w_rec"][0, 0] += 1.0
    np.testing.assert_array_equal(second.parameters["w_rec"], second_recurrent)

    first.act_with_scores(np.ones(spec.input_size))
    assert not np.all(first.hidden_state == 0.0)
    assert np.all(second.hidden_state == 0.0)


@pytest.mark.parametrize("controller_type", ["reactive", "proactive"])
def test_genome_decoding_and_controller_action(controller_type: ControllerType) -> None:
    spec = _spec(controller_type)
    genome = np.random.default_rng(6).normal(size=spec.parameter_count)
    parameters = unflatten_genome(genome, spec)
    assert tuple(parameters) == tuple(name for name, _shape in spec.parameter_shapes)
    action, scores = build_controller(genome, spec).act_with_scores(np.zeros(spec.input_size))
    assert action in (0, 1, 2)
    assert scores.shape == (3,)


def test_invalid_genomes_are_rejected() -> None:
    spec = _spec("reactive")
    with pytest.raises(ValueError, match="expected"):
        validate_genome(np.zeros(spec.parameter_count - 1), spec)
    invalid = np.zeros(spec.parameter_count)
    invalid[0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        validate_genome(invalid, spec)


def test_action_selection_uses_argmax_and_keeps_no_key_for_ties() -> None:
    assert action_from_scores(np.asarray([0.0, 0.10, -0.2])) == 1
    assert action_from_scores(np.asarray([0.0, 0.16, -0.2])) == 1
    assert action_from_scores(np.asarray([0.0, -0.2, 0.16])) == 2
    assert action_from_scores(np.asarray([0.0, 0.0, -0.2])) == 0


@pytest.mark.parametrize("controller_type", ["reactive", "proactive"])
def test_batched_cpu_runtime_matches_individual_controllers(
    controller_type: ControllerType,
) -> None:
    spec = _spec(controller_type)
    rng = np.random.default_rng(71)
    genomes = [rng.normal(0.0, 0.3, spec.parameter_count) for _ in range(4)]
    individual = [build_controller(genome, spec) for genome in genomes]
    runtime = PopulationControllerRuntime(genomes, spec, "cpu")
    active = np.asarray([True, True, False, True])

    for _ in range(3):
        sensory = rng.random((4, spec.input_size))
        expected_actions = np.zeros(4, dtype=np.int64)
        expected_scores = np.zeros((4, 3))
        for index, controller in enumerate(individual):
            if active[index]:
                action, scores = controller.act_with_scores(sensory[index])
                expected_actions[index] = action
                expected_scores[index] = scores
        actions, scores, hidden = runtime.act_with_scores(sensory, active)
        np.testing.assert_array_equal(actions, expected_actions)
        np.testing.assert_array_equal(scores, expected_scores)
        if controller_type == "reactive":
            assert hidden is None
        else:
            assert hidden is not None
            for index, controller in enumerate(individual):
                np.testing.assert_array_equal(hidden[index], controller.hidden_state)
    assert runtime.completed_batches == 3


def test_cuda_runtime_matches_cpu_actions_when_cuda_is_available() -> None:
    try:
        torch = pytest.importorskip("torch")
    except OSError as error:
        pytest.skip(f"PyTorch native libraries are unavailable: {type(error).__name__}")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    spec = _spec("proactive")
    rng = np.random.default_rng(72)
    genomes = [rng.normal(0.0, 0.3, spec.parameter_count) for _ in range(6)]
    cpu = PopulationControllerRuntime(genomes, spec, "cpu")
    cuda = PopulationControllerRuntime(genomes, spec, "cuda")
    active = np.asarray([True, True, False, True, False, True])

    for _ in range(4):
        sensory = rng.random((6, spec.input_size))
        cpu_actions, cpu_scores, cpu_hidden = cpu.act_with_scores(sensory, active)
        cuda_actions, cuda_scores, cuda_hidden = cuda.act_with_scores(sensory, active)
        np.testing.assert_array_equal(cuda_actions, cpu_actions)
        np.testing.assert_allclose(cuda_scores, cpu_scores, rtol=1e-12, atol=1e-12)
        assert cpu_hidden is not None and cuda_hidden is not None
        np.testing.assert_allclose(cuda_hidden, cpu_hidden, rtol=1e-12, atol=1e-12)
    assert cpu.completed_batches == 4
    assert cuda.completed_batches == 4


def test_auto_accelerator_falls_back_when_pytorch_native_libraries_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_import(_name: str) -> None:
        raise OSError("localized operating-system error")

    monkeypatch.setattr("dino_er.controllers.importlib.import_module", fail_import)
    status = resolve_accelerator("auto")
    assert status.selected == "cpu"
    assert status.label == "cpu | PyTorch native libraries unavailable"
    assert status.diagnostic is not None
    assert "uv sync --extra dev --extra cuda" in status.diagnostic


def test_explicit_cuda_reports_an_english_native_library_remediation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_import(_name: str) -> None:
        raise OSError("localized operating-system error")

    monkeypatch.setattr("dino_er.controllers.importlib.import_module", fail_import)
    with pytest.raises(RuntimeError, match="PyTorch could not load its native libraries"):
        resolve_accelerator("cuda")
