"""Deterministic neural controllers and their flat evolved genomes."""

from __future__ import annotations

import hashlib
import importlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

ControllerType = Literal["reactive", "proactive"]
Accelerator = Literal["auto", "cpu", "cuda"]


def action_from_scores(scores: np.ndarray) -> int:
    """Map three motor scores to a key state using deterministic argmax."""

    values = np.asarray(scores, dtype=np.float64)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError("Action scores must be three finite values")
    return int(np.argmax(values))


@dataclass(frozen=True)
class ControllerSpec:
    """Architecture definition shared by genome and controller code."""

    controller_type: ControllerType
    input_size: int
    hidden_size: int
    action_size: int = 3
    # One current-frame pixel observation/action decision per fixed Chromium
    # physics frame, i.e. an explicit 60-Hz ER boundary.
    dt: float = 1.0 / 60.0
    tau: float = 0.25

    def __post_init__(self) -> None:
        if self.controller_type not in ("reactive", "proactive"):
            raise ValueError(f"Unsupported controller type: {self.controller_type}")
        if self.input_size <= 0 or self.hidden_size <= 0:
            raise ValueError("input_size and hidden_size must be positive")
        if self.action_size != 3:
            raise ValueError("Chrome Dino requires exactly three action scores")
        if self.dt <= 0 or self.tau <= 0 or self.dt > self.tau:
            raise ValueError("Require 0 < dt <= tau for the CTRNN Euler update")

    @property
    def parameter_shapes(self) -> tuple[tuple[str, tuple[int, ...]], ...]:
        common = (
            ("w_in", (self.hidden_size, self.input_size)),
            ("b_hidden", (self.hidden_size,)),
        )
        if self.controller_type == "reactive":
            return common + (
                ("w_out", (self.action_size, self.hidden_size)),
                ("b_out", (self.action_size,)),
            )
        return common + (
            ("w_rec", (self.hidden_size, self.hidden_size)),
            ("w_out", (self.action_size, self.hidden_size)),
            ("b_out", (self.action_size,)),
        )

    @property
    def parameter_count(self) -> int:
        return sum(int(np.prod(shape)) for _, shape in self.parameter_shapes)


def default_controller_spec(
    controller_type: ControllerType,
    input_size: int,
) -> ControllerSpec:
    """Return the parameter-matched architectures used in the experiment."""

    # With the five direct visual geometry inputs, 10 reactive units and six
    # recurrent units both produce exactly 93 neural parameters.
    hidden_size = 10 if controller_type == "reactive" else 6
    return ControllerSpec(controller_type, input_size, hidden_size)


def validate_genome(
    genome: np.ndarray | list[float],
    spec: ControllerSpec,
) -> np.ndarray:
    """Return a canonical finite float64 genome or raise a precise error."""

    canonical = np.asarray(genome, dtype=np.float64)
    if canonical.ndim != 1:
        raise ValueError(f"Genome must be one-dimensional, got {canonical.shape}")
    if canonical.size != spec.parameter_count:
        raise ValueError(
            f"{spec.controller_type} genome has {canonical.size} values; "
            f"expected {spec.parameter_count}"
        )
    if not np.isfinite(canonical).all():
        raise ValueError("Genome contains non-finite values")
    return np.ascontiguousarray(canonical)


def genome_hash(genome: np.ndarray | list[float], spec: ControllerSpec) -> str:
    """Hash the canonical architecture label and float64 genome bytes."""

    canonical = validate_genome(genome, spec)
    digest = hashlib.sha256()
    digest.update(spec.controller_type.encode("ascii"))
    digest.update(np.asarray(canonical.shape, dtype="<i8").tobytes())
    digest.update(canonical.astype("<f8", copy=False).tobytes())
    return digest.hexdigest()


def unflatten_genome(
    genome: np.ndarray | list[float],
    spec: ControllerSpec,
) -> dict[str, np.ndarray]:
    """Decode a genome in the single documented row-major field order."""

    canonical = validate_genome(genome, spec)
    parameters: dict[str, np.ndarray] = {}
    offset = 0
    for name, shape in spec.parameter_shapes:
        size = int(np.prod(shape))
        parameters[name] = canonical[offset : offset + size].reshape(shape).copy()
        offset += size
    if offset != canonical.size:  # defensive invariant
        raise AssertionError("Genome layout did not consume every value")
    return parameters


class ReactiveController:
    """Feedforward sensory-vector → tanh hidden → linear-score controller."""

    def __init__(self, genome: np.ndarray | list[float], spec: ControllerSpec) -> None:
        if spec.controller_type != "reactive":
            raise ValueError("ReactiveController requires a reactive spec")
        self.spec = spec
        self.parameters = unflatten_genome(genome, spec)

    def reset(self, seed: int | None = None) -> None:
        del seed

    @property
    def hidden_state(self) -> None:
        return None

    def act_with_scores(self, sensory: np.ndarray) -> tuple[int, np.ndarray]:
        inputs = _validate_sensory(sensory, self.spec.input_size)
        hidden = np.tanh(self.parameters["w_in"] @ inputs + self.parameters["b_hidden"])
        scores = self.parameters["w_out"] @ hidden + self.parameters["b_out"]
        return action_from_scores(scores), scores.copy()


class CTRNNController:
    """Proactive controller with one Euler-integrated recurrent state."""

    def __init__(self, genome: np.ndarray | list[float], spec: ControllerSpec) -> None:
        if spec.controller_type != "proactive":
            raise ValueError("CTRNNController requires a proactive spec")
        self.spec = spec
        self.parameters = unflatten_genome(genome, spec)
        self._hidden = np.zeros(spec.hidden_size, dtype=np.float64)

    def reset(self, seed: int | None = None) -> None:
        del seed
        self._hidden.fill(0.0)

    @property
    def hidden_state(self) -> np.ndarray:
        return self._hidden.copy()

    def act_with_scores(self, sensory: np.ndarray) -> tuple[int, np.ndarray]:
        inputs = _validate_sensory(sensory, self.spec.input_size)
        drive = np.tanh(
            self.parameters["w_in"] @ inputs
            + self.parameters["w_rec"] @ self._hidden
            + self.parameters["b_hidden"]
        )
        self._hidden += (self.spec.dt / self.spec.tau) * (-self._hidden + drive)
        scores = self.parameters["w_out"] @ self._hidden + self.parameters["b_out"]
        return action_from_scores(scores), scores.copy()


def build_controller(
    genome: np.ndarray | list[float],
    spec: ControllerSpec,
) -> ReactiveController | CTRNNController:
    """Build the controller selected by ``spec``."""

    if spec.controller_type == "reactive":
        return ReactiveController(genome, spec)
    return CTRNNController(genome, spec)


@dataclass(frozen=True)
class AcceleratorStatus:
    requested: Accelerator
    selected: Literal["cpu", "cuda"]
    label: str
    diagnostic: str | None = None


def resolve_accelerator(requested: Accelerator) -> AcceleratorStatus:
    """Resolve an honest controller-inference accelerator for this machine."""

    if requested not in ("auto", "cpu", "cuda"):
        raise ValueError("accelerator must be auto, cpu, or cuda")
    if requested == "cpu":
        return AcceleratorStatus(requested, "cpu", "cpu | NumPy controller inference")
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        label = "cpu | CUDA extra not installed"
        diagnostic = "The optional PyTorch CUDA dependency is not installed."
        cuda_error = (
            "CUDA controller inference requires the optional PyTorch extra. "
            "Run `uv sync --extra dev --extra cuda`."
        )
    except OSError:
        label = "cpu | PyTorch native libraries unavailable"
        cuda_error = diagnostic = (
            "PyTorch could not load its native libraries. Reinstall the optional "
            "CUDA environment with `uv sync --extra dev --extra cuda`."
        )
    else:
        try:
            cuda_available = bool(torch.cuda.is_available())
        except (OSError, RuntimeError):
            label = "cpu | PyTorch CUDA probe failed"
            cuda_error = diagnostic = (
                "PyTorch could not load its native libraries. Reinstall the optional "
                "CUDA environment with `uv sync --extra dev --extra cuda`."
            )
        else:
            if not cuda_available:
                label = "cpu | CUDA unavailable"
                diagnostic = cuda_error = (
                    "PyTorch is installed, but torch.cuda.is_available() is false."
                )
            else:
                try:
                    name = str(torch.cuda.get_device_name(0))
                except (OSError, RuntimeError):
                    label = "cpu | CUDA device query failed"
                    cuda_error = diagnostic = (
                        "PyTorch could not load its native libraries. Reinstall the optional "
                        "CUDA environment with `uv sync --extra dev --extra cuda`."
                    )
                else:
                    return AcceleratorStatus(
                        requested, "cuda", f"cuda:0 | {name} | controllers only"
                    )
    if requested == "cuda":
        raise RuntimeError(cuda_error)
    return AcceleratorStatus(requested, "cpu", label, diagnostic)


class PopulationControllerRuntime:
    """Batched population controller execution with optional honest CUDA use."""

    def __init__(
        self,
        genomes: Sequence[np.ndarray | list[float]] | np.ndarray,
        spec: ControllerSpec,
        accelerator: Accelerator = "auto",
    ) -> None:
        if len(genomes) == 0:
            raise ValueError("Population controller runtime requires genomes")
        self.spec = spec
        self.population_size = len(genomes)
        self.status = resolve_accelerator(accelerator)
        self._controllers: list[ReactiveController | CTRNNController] = []
        self._torch: Any = None
        self._parameters: dict[str, Any] = {}
        self._hidden: Any = None
        self.completed_batches = 0
        if self.status.selected == "cpu":
            self._controllers = [build_controller(genome, spec) for genome in genomes]
            return
        torch = importlib.import_module("torch")
        self._torch = torch
        device = torch.device("cuda:0")
        decoded = [unflatten_genome(genome, spec) for genome in genomes]
        for name, _shape in spec.parameter_shapes:
            stacked = np.stack([parameters[name] for parameters in decoded])
            self._parameters[name] = torch.as_tensor(
                stacked,
                dtype=torch.float64,
                device=device,
            )
        if spec.controller_type == "proactive":
            self._hidden = torch.zeros(
                (self.population_size, spec.hidden_size),
                dtype=torch.float64,
                device=device,
            )

    def reset(self, seed: int | None = None) -> None:
        if self.status.selected == "cpu":
            for controller in self._controllers:
                controller.reset(seed)
            return
        if self._hidden is not None:
            with self._torch.inference_mode():
                self._hidden.zero_()

    def act_with_scores(
        self,
        sensory: np.ndarray,
        active: np.ndarray,
        *,
        reset_state_each_step: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        inputs = np.asarray(sensory, dtype=np.float64)
        active_mask = np.asarray(active, dtype=np.bool_)
        if inputs.shape != (self.population_size, self.spec.input_size):
            raise ValueError("Population sensory matrix has the wrong shape")
        if active_mask.shape != (self.population_size,):
            raise ValueError("Population active mask has the wrong shape")
        if not np.isfinite(inputs).all():
            raise ValueError("Population sensory matrix contains non-finite values")
        if self.status.selected == "cpu":
            result = self._act_cpu(inputs, active_mask, reset_state_each_step)
        else:
            with self._torch.inference_mode():
                result = self._act_cuda(inputs, active_mask, reset_state_each_step)
        self.completed_batches += 1
        return result

    def _act_cpu(
        self,
        inputs: np.ndarray,
        active: np.ndarray,
        reset_state_each_step: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        actions = np.zeros(self.population_size, dtype=np.int64)
        scores = np.zeros((self.population_size, self.spec.action_size))
        hidden = (
            np.zeros((self.population_size, self.spec.hidden_size))
            if self.spec.controller_type == "proactive"
            else None
        )
        for index, controller in enumerate(self._controllers):
            if not active[index]:
                if hidden is not None:
                    hidden[index] = controller.hidden_state
                continue
            if reset_state_each_step:
                controller.reset()
            action, candidate_scores = controller.act_with_scores(inputs[index])
            actions[index] = action
            scores[index] = candidate_scores
            if hidden is not None:
                hidden[index] = controller.hidden_state
        return actions, scores, hidden

    def _act_cuda(
        self,
        inputs: np.ndarray,
        active: np.ndarray,
        reset_state_each_step: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        device_inputs = self._torch.as_tensor(
            inputs,
            dtype=self._torch.float64,
            device="cuda:0",
        )
        active_mask = self._torch.as_tensor(active, dtype=self._torch.bool, device="cuda:0")
        w_in = self._parameters["w_in"]
        b_hidden = self._parameters["b_hidden"]
        if self.spec.controller_type == "reactive":
            hidden = self._torch.tanh(
                self._torch.bmm(w_in, device_inputs.unsqueeze(-1)).squeeze(-1) + b_hidden
            )
        else:
            hidden_state = self._hidden
            if reset_state_each_step:
                hidden_state = self._torch.where(
                    active_mask.unsqueeze(-1),
                    self._torch.zeros_like(hidden_state),
                    hidden_state,
                )
            drive = self._torch.tanh(
                self._torch.bmm(w_in, device_inputs.unsqueeze(-1)).squeeze(-1)
                + self._torch.bmm(
                    self._parameters["w_rec"],
                    hidden_state.unsqueeze(-1),
                ).squeeze(-1)
                + b_hidden
            )
            proposed = hidden_state + (self.spec.dt / self.spec.tau) * (-hidden_state + drive)
            self._hidden = self._torch.where(
                active_mask.unsqueeze(-1),
                proposed,
                hidden_state,
            )
            hidden = self._hidden
        scores = (
            self._torch.bmm(
                self._parameters["w_out"],
                hidden.unsqueeze(-1),
            ).squeeze(-1)
            + self._parameters["b_out"]
        )
        scores = self._torch.where(
            active_mask.unsqueeze(-1),
            scores,
            self._torch.zeros_like(scores),
        )
        actions = self._torch.argmax(scores, dim=1)
        actions = self._torch.where(active_mask, actions, self._torch.zeros_like(actions))
        return (
            actions.cpu().numpy().astype(np.int64, copy=False),
            scores.cpu().numpy(),
            hidden.cpu().numpy() if self.spec.controller_type == "proactive" else None,
        )


def _validate_sensory(sensory: np.ndarray, input_size: int) -> np.ndarray:
    inputs = np.asarray(sensory, dtype=np.float64)
    if inputs.shape != (input_size,):
        raise ValueError(f"Sensory vector has shape {inputs.shape}; expected ({input_size},)")
    if not np.isfinite(inputs).all():
        raise ValueError("Sensory vector contains non-finite values")
    return inputs
