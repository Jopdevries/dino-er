"""Persistent Chrome transport restricted to rendered pixels and key actions."""

from __future__ import annotations

import contextlib
import functools
import json
import math
import queue
import threading
import urllib.parse
import zlib
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
from playwright.sync_api import Browser, BrowserContext, Page, Playwright, sync_playwright
from websockets.exceptions import ConnectionClosed
from websockets.sync.server import Server, ServerConnection, serve


class VisualTransportError(RuntimeError):
    """Raised when the persistent visual channel fails."""


@dataclass(frozen=True)
class PopulationPixelBatch:
    """Lossless rendered pixel components for private 600 by 150 frames."""

    shared_world: np.ndarray
    dino_patches: np.ndarray
    game_over_text_patches: np.ndarray
    restart_patches: np.ndarray

    def materialize(self) -> np.ndarray:
        """Reconstruct the exact ordered private frames without engine state."""

        count = int(self.dino_patches.shape[0])
        grayscale = np.broadcast_to(
            self.shared_world,
            (count, *self.shared_world.shape),
        ).copy()
        grayscale[:, :, 50:110] = self.dino_patches
        grayscale[:, 42:53, 205:396] = self.game_over_text_patches
        grayscale[:, 75:107, 284:320] = self.restart_patches
        return np.repeat(grayscale[:, :, :, None], 3, axis=3)


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        del format, args


class BrowserVisualTransport:
    """Launch one browser and exchange only Canvas frames and action IDs."""

    def __init__(
        self,
        *,
        distribution_dir: Path,
        headless: bool,
        instances: int = 1,
        simulation_speed: float = 1.0,
        timeout_seconds: float = 10.0,
    ) -> None:
        if not distribution_dir.is_dir():
            raise FileNotFoundError(
                f"Built browser game not found at {distribution_dir}. "
                "Run `npm.cmd --prefix game run build` first."
            )
        self.timeout_seconds = timeout_seconds
        self.instances = instances
        if not math.isfinite(simulation_speed) or not 0.25 <= simulation_speed <= 100:
            raise ValueError("simulation_speed must be between 0.25 and 100")
        self._messages: queue.Queue[dict[str, Any]] = queue.Queue()
        self._arena_controls: queue.Queue[dict[str, Any]] = queue.Queue()
        self._connection_ready = threading.Event()
        self._connection_lost = threading.Event()
        self._window_closed = threading.Event()
        self._connection: ServerConnection | None = None
        self._connection_failure: str | None = None
        self._closed = False
        self._crashed = False
        self._request_id = 0
        self._browser_log: deque[str] = deque(maxlen=50)
        self._previous_raw_atlas: np.ndarray | None = None
        self._dino_patches: np.ndarray | None = None
        self._game_over_text_patches: np.ndarray | None = None
        self._restart_patches: np.ndarray | None = None

        handler = functools.partial(_QuietHandler, directory=str(distribution_dir))
        self._http_server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self._http_thread = threading.Thread(
            target=self._http_server.serve_forever,
            name="dino-http",
            daemon=True,
        )
        self._http_thread.start()

        self._ws_server: Server = serve(
            self._handle_websocket,
            "127.0.0.1",
            0,
            max_size=3 * 1024 * 1024,
        )
        self._ws_thread = threading.Thread(
            target=self._ws_server.serve_forever,
            name="dino-websocket",
            daemon=True,
        )
        self._ws_thread.start()
        ws_port = int(self._ws_server.socket.getsockname()[1])
        http_port = int(self._http_server.server_address[1])
        query = urllib.parse.urlencode(
            {
                "bridge": f"ws://127.0.0.1:{ws_port}",
                "instances": instances,
                "realtime": int(not headless),
                "speed": simulation_speed,
            }
        )

        self._playwright: Playwright = sync_playwright().start()
        self._browser: Browser = self._playwright.chromium.launch(
            channel="chrome",
            headless=headless,
        )
        self._context: BrowserContext = self._browser.new_context(
            viewport={"width": 1100, "height": 800},
            device_scale_factor=1,
        )
        self._page: Page = self._context.new_page()
        self._page.on(
            "console",
            lambda message: self._browser_log.append(f"console[{message.type}]: {message.text}"),
        )
        self._page.on("pageerror", lambda error: self._browser_log.append(str(error)))
        self._page.on("crash", self._mark_crashed)
        self._page.on("close", self._mark_window_closed)
        self._page.goto(
            f"http://127.0.0.1:{http_port}/batch.html?{query}",
            wait_until="load",
            timeout=int(timeout_seconds * 1000),
        )
        if not self._connection_ready.wait(timeout_seconds):
            self.close()
            raise VisualTransportError("Browser did not open the pixel-only channel")
        ready = self._wait_message(expected_type="ready")
        if ready.get("protocol") != 6:
            self.close()
            raise VisualTransportError("Unsupported browser bridge protocol")
        if ready.get("instances") != instances:
            self.close()
            raise VisualTransportError("Browser created the wrong batch size")

    def configure_arena(
        self,
        *,
        seed: int,
        metadata: dict[str, Any],
    ) -> PopulationPixelBatch:
        self._previous_raw_atlas = None
        self._dino_patches = None
        self._game_over_text_patches = None
        self._restart_patches = None
        return self._frame_batch_request(
            {
                "type": "configure_arena",
                "seed": int(seed),
                "metadata": metadata,
            }
        )

    def act_batch(
        self,
        actions: list[int],
        *,
        active: list[bool] | None = None,
    ) -> PopulationPixelBatch:
        if len(actions) != self.instances or any(action not in (0, 1, 2) for action in actions):
            raise ValueError("Actions must contain one value in {0,1,2} per instance")
        active_flags = [True] * self.instances if active is None else active
        if len(active_flags) != self.instances:
            raise ValueError("Active flags must match the arena population")
        return self._frame_batch_request(
            {
                "type": "action_batch",
                "actions": actions,
                "active": [bool(value) for value in active_flags],
            }
        )

    def send_arena_diagnostics(self, payload: dict[str, Any]) -> None:
        self._send({"type": "arena_diagnostics", **payload})

    def send_arena_status(
        self,
        phase: str,
        message: str,
        **fields: Any,
    ) -> None:
        self._send(
            {
                "type": "arena_status",
                "phase": phase,
                "message": message,
                **fields,
            }
        )

    def poll_arena_controls(self) -> list[dict[str, Any]]:
        controls: list[dict[str, Any]] = []
        while True:
            try:
                controls.append(self._arena_controls.get_nowait())
            except queue.Empty:
                return controls

    def wait_until_window_closed(
        self,
        on_poll: Callable[[], None] | None = None,
    ) -> None:
        """Keep an owned visible browser available until the user closes it."""

        while not self._window_closed.wait(0.1):
            if self._closed:
                return
            if on_poll is not None:
                on_poll()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if hasattr(self, "_context"):
                with contextlib.suppress(Exception):
                    self._context.close()
            if hasattr(self, "_browser"):
                with contextlib.suppress(Exception):
                    self._browser.close()
            if hasattr(self, "_playwright"):
                with contextlib.suppress(Exception):
                    self._playwright.stop()
        finally:
            self._window_closed.set()
            if hasattr(self, "_ws_server"):
                with contextlib.suppress(Exception):
                    self._ws_server.shutdown()
            if hasattr(self, "_http_server"):
                with contextlib.suppress(Exception):
                    self._http_server.shutdown()
                with contextlib.suppress(Exception):
                    self._http_server.server_close()

    def _handle_websocket(self, connection: ServerConnection) -> None:
        if self._connection is not None:
            connection.close(1008, "only one game connection is allowed")
            return
        self._connection = connection
        self._connection_lost.clear()
        self._connection_failure = None
        self._connection_ready.set()
        try:
            for raw_message in connection:
                if isinstance(raw_message, bytes):
                    self._messages.put(
                        {
                            "type": "binary_frame_batch",
                            "data": raw_message,
                        }
                    )
                    continue
                try:
                    message = json.loads(raw_message)
                except (json.JSONDecodeError, TypeError):
                    self._messages.put(
                        {"type": "transport_error", "error": "invalid JSON response"}
                    )
                    continue
                if not isinstance(message, dict) or "type" not in message:
                    self._messages.put(
                        {
                            "type": "transport_error",
                            "error": "response must be a typed JSON object",
                        }
                    )
                    continue
                if message.get("type") == "arena_control":
                    self._arena_controls.put(message)
                else:
                    self._messages.put(message)
        except ConnectionClosed as error:
            if not self._closed:
                self._connection_failure = f"{type(error).__name__}: {error}"
                self._browser_log.append(
                    f"visual channel closed unexpectedly: {self._connection_failure}"
                )
        finally:
            if self._connection is connection:
                self._connection = None
                self._connection_lost.set()

    def _decode_sparse_xor_runs(self, payload: bytes, frame_size: int) -> np.ndarray:
        if self._previous_raw_atlas is None:
            raise VisualTransportError("Browser sent a delta frame without a key frame")
        decoded = self._previous_raw_atlas.copy()
        offset = 0
        previous_end = 0
        while offset < len(payload):
            if offset + 6 > len(payload):
                raise VisualTransportError("Browser sent a truncated sparse pixel run")
            start = int.from_bytes(payload[offset : offset + 4], "little")
            length = int.from_bytes(payload[offset + 4 : offset + 6], "little")
            offset += 6
            end = start + length
            if (
                length <= 0
                or start < previous_end
                or end > frame_size
                or offset + length > len(payload)
            ):
                raise VisualTransportError("Browser sent an invalid sparse pixel run")
            values = np.frombuffer(payload, dtype=np.uint8, count=length, offset=offset)
            np.bitwise_xor(decoded[start:end], values, out=decoded[start:end])
            offset += length
            previous_end = end
        return decoded

    def _frame_batch_request(self, message: dict[str, Any]) -> PopulationPixelBatch:
        self._request_id += 1
        request_id = self._request_id
        self._send({**message, "requestId": request_id})
        response = self._wait_message(
            expected_type="frame_batch",
            expected_request_id=request_id,
        )
        compact_count = response.get("count")
        candidate_ids = response.get("candidateIds")
        if (
            not isinstance(compact_count, int)
            or not 1 <= compact_count <= self.instances
            or not isinstance(candidate_ids, list)
            or len(candidate_ids) != compact_count
            or any(
                not isinstance(candidate_id, int) or not 0 <= candidate_id < self.instances
                for candidate_id in candidate_ids
            )
            or len(set(candidate_ids)) != compact_count
        ):
            raise VisualTransportError("Browser returned invalid active candidate IDs")
        dino_columns = min(20, compact_count)
        dino_rows = (compact_count + dino_columns - 1) // dino_columns
        game_over_text_columns = min(6, compact_count)
        game_over_text_patch_y = 150 + dino_rows * 150
        game_over_text_rows = (compact_count + game_over_text_columns - 1) // game_over_text_columns
        restart_columns = min(33, compact_count)
        restart_patch_y = game_over_text_patch_y + game_over_text_rows * 11
        restart_rows = (compact_count + restart_columns - 1) // restart_columns
        atlas_width = max(
            600,
            dino_columns * 60,
            game_over_text_columns * 191,
            restart_columns * 36,
        )
        atlas_height = restart_patch_y + restart_rows * 32
        if (
            response.get("width") != 600
            or response.get("height") != 150
            or response.get("populationSize") != self.instances
            or response.get("atlasLayout") != "shared_world_active_private_pixel_patches_v4"
            or response.get("atlasWidth") != atlas_width
            or response.get("atlasHeight") != atlas_height
            or response.get("dinoColumns") != dino_columns
            or response.get("gameOverTextColumns") != game_over_text_columns
            or response.get("gameOverTextPatchY") != game_over_text_patch_y
            or response.get("restartColumns") != restart_columns
            or response.get("restartPatchY") != restart_patch_y
            or response.get("observation") != "private_clean_candidate_frames"
            or response.get("encoding")
            not in {
                "zlib_grayscale_u8",
                "zlib_xor_grayscale_u8",
                "sparse_xor_runs_u8",
            }
            or response.get("rawByteLength") != atlas_width * atlas_height
            or not isinstance(response.get("byteLength"), int)
        ):
            raise VisualTransportError("Browser returned an invalid batch envelope")
        binary = self._wait_message(expected_type="binary_frame_batch")
        compressed = binary.get("data")
        if not isinstance(compressed, bytes):
            raise VisualTransportError("Browser returned an invalid binary frame batch")
        if len(compressed) != response["byteLength"]:
            raise VisualTransportError("Browser binary frame length does not match its envelope")
        frame_size = atlas_width * atlas_height
        if response["encoding"] == "sparse_xor_runs_u8":
            if self._previous_raw_atlas is None:
                raise VisualTransportError("Browser sent a delta frame without a key frame")
            encoded = self._decode_sparse_xor_runs(compressed, frame_size)
        else:
            try:
                raw = zlib.decompress(compressed)
            except zlib.error as error:
                raise VisualTransportError(
                    "Browser returned an invalid compressed grayscale atlas"
                ) from error
            if len(raw) != frame_size:
                raise VisualTransportError("Browser returned an invalid grayscale atlas size")
            encoded = np.frombuffer(raw, dtype=np.uint8)
            if response["encoding"] == "zlib_xor_grayscale_u8":
                if self._previous_raw_atlas is None:
                    raise VisualTransportError("Browser sent a delta frame without a key frame")
                encoded = np.bitwise_xor(self._previous_raw_atlas, encoded)
        self._previous_raw_atlas = encoded
        atlas = encoded.reshape(
            atlas_height,
            atlas_width,
        )

        shared_world = atlas[:150, :600].copy()
        dino_region = atlas[
            150 : 150 + dino_rows * 150,
            : dino_columns * 60,
        ]
        compact_dino_patches = (
            dino_region.reshape(dino_rows, 150, dino_columns, 60)
            .transpose(0, 2, 1, 3)
            .reshape(dino_rows * dino_columns, 150, 60)
        )[:compact_count].copy()
        game_over_text_region = atlas[
            game_over_text_patch_y : game_over_text_patch_y + game_over_text_rows * 11,
            : game_over_text_columns * 191,
        ]
        compact_game_over_text_patches = (
            game_over_text_region.reshape(
                game_over_text_rows,
                11,
                game_over_text_columns,
                191,
            )
            .transpose(0, 2, 1, 3)
            .reshape(
                game_over_text_rows * game_over_text_columns,
                11,
                191,
            )
        )[:compact_count].copy()
        restart_region = atlas[
            restart_patch_y : restart_patch_y + restart_rows * 32,
            : restart_columns * 36,
        ]
        compact_restart_patches = (
            restart_region.reshape(
                restart_rows,
                32,
                restart_columns,
                36,
            )
            .transpose(0, 2, 1, 3)
            .reshape(restart_rows * restart_columns, 32, 36)
        )[:compact_count].copy()
        if self._dino_patches is None:
            if candidate_ids != list(range(self.instances)):
                raise VisualTransportError(
                    "The first private pixel batch must contain the complete population"
                )
            dino_patches = compact_dino_patches
            game_over_text_patches = compact_game_over_text_patches
            restart_patches = compact_restart_patches
        else:
            if self._game_over_text_patches is None or self._restart_patches is None:
                raise VisualTransportError("Private pixel component cache is incomplete")
            dino_patches = self._dino_patches.copy()
            game_over_text_patches = self._game_over_text_patches.copy()
            restart_patches = self._restart_patches.copy()
            dino_patches[candidate_ids] = compact_dino_patches
            game_over_text_patches[candidate_ids] = compact_game_over_text_patches
            restart_patches[candidate_ids] = compact_restart_patches
        self._dino_patches = dino_patches
        self._game_over_text_patches = game_over_text_patches
        self._restart_patches = restart_patches
        return PopulationPixelBatch(
            shared_world=shared_world,
            dino_patches=dino_patches,
            game_over_text_patches=game_over_text_patches,
            restart_patches=restart_patches,
        )

    def _send(self, message: dict[str, Any]) -> None:
        if self._closed or self._crashed:
            raise VisualTransportError(self._failure_context("browser is unavailable"))
        connection = self._connection
        if connection is None:
            raise VisualTransportError(self._failure_context("visual channel is closed"))
        try:
            connection.send(json.dumps(message, separators=(",", ":"), allow_nan=False))
        except ConnectionClosed as error:
            self._connection_failure = f"{type(error).__name__}: {error}"
            self._browser_log.append(
                f"visual channel closed unexpectedly: {self._connection_failure}"
            )
            if self._connection is connection:
                self._connection = None
            self._connection_lost.set()
            raise VisualTransportError(
                self._failure_context(self._connection_lost_message("sending an arena request"))
            ) from error

    def _wait_message(
        self,
        *,
        expected_type: str,
        expected_request_id: int | None = None,
    ) -> dict[str, Any]:
        while True:
            if self._connection_lost.is_set():
                raise VisualTransportError(
                    self._failure_context(
                        self._connection_lost_message(f"waiting for {expected_type!r}")
                    )
                )
            try:
                message = self._messages.get(timeout=self.timeout_seconds)
            except queue.Empty as error:
                raise VisualTransportError(
                    self._failure_context(f"timed out waiting for {expected_type!r}")
                ) from error
            if message.get("type") == "transport_error":
                raise VisualTransportError(str(message.get("error", "transport error")))
            if message.get("type") == "arena_control":
                self._arena_controls.put(message)
                continue
            if message.get("type") != expected_type:
                continue
            if expected_request_id is not None and message.get("requestId") != expected_request_id:
                continue
            return message

    def _mark_crashed(self, *_: object) -> None:
        self._crashed = True
        self._window_closed.set()

    def _mark_window_closed(self, *_: object) -> None:
        self._window_closed.set()

    def _connection_lost_message(self, waiting_for: str) -> str:
        if self._crashed:
            return f"browser process crashed while {waiting_for}"
        if self._window_closed.is_set() and not self._closed:
            return f"browser window was closed while {waiting_for}"
        if self._connection_failure:
            return f"visual channel disconnected while {waiting_for}: {self._connection_failure}"
        return f"visual channel disconnected while {waiting_for}"

    def _failure_context(self, message: str) -> str:
        log = "\n".join(self._browser_log)
        return f"{message}\nBrowser diagnostics:\n{log}" if log else message
