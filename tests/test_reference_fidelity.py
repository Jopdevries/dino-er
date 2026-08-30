from __future__ import annotations

import hashlib
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CHROMIUM_COMMIT = "1ccb91e11f09fbbdec4f8f754d0e2f7d28246660"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_pinned_chromium_assets_have_audited_hashes() -> None:
    asset_root = REPOSITORY_ROOT / "game" / "public" / "assets" / "chromium"
    assert _sha256(asset_root / "100-offline-sprite.png") == (
        "04d05978fdb111358073ab0524e5c1fafc0826615c206987618416b8bd8a4747"
    )
    assert _sha256(asset_root / "200-offline-sprite.png") == (
        "e4222715b556e7d99622c83e620d2f8e090047e56adb07923047f95828d561f2"
    )
    engine_source = (REPOSITORY_ROOT / "game" / "src" / "engine.ts").read_text(encoding="utf-8")
    assert CHROMIUM_COMMIT in engine_source


def test_provenance_names_only_the_canonical_chromium_source() -> None:
    provenance = (REPOSITORY_ROOT / "third_party" / "CHROMIUM_DINO_SOURCES.txt").read_text(
        encoding="utf-8"
    )
    assert "https://github.com/chromium/chromium" in provenance
    assert CHROMIUM_COMMIT in provenance
    assert "wayou" not in provenance.lower()
