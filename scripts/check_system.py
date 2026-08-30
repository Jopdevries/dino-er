"""Report the local evolutionary runtime and honest controller accelerator."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys

import numpy as np

from dino_er.controllers import resolve_accelerator


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", dest="as_json")
    arguments = parser.parse_args()
    accelerator = resolve_accelerator("auto")
    information = {
        "operating_system": platform.system(),
        "python_version": sys.version.split()[0],
        "logical_cpu_count": os.cpu_count(),
        "numpy_version": np.__version__,
        "controller_compute": accelerator.label,
        "controller_compute_diagnostic": accelerator.diagnostic,
        "population_runtime": "one shared world with private candidate frames",
    }
    if arguments.as_json:
        print(json.dumps(information, indent=2))
    else:
        for key, value in information.items():
            print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
