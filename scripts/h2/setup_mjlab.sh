#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Keep the native MJWarp runtime separate from the Newton VBD training environment.
set -euo pipefail
lab_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
destination="$lab_root/.local/mjlab_h2"
repository="${MJLAB_REPOSITORY:-https://github.com/JuanaDd/mjlab.git}"
revision="ba3f5814a703e6823513aac4a44d0f80c29a2284"

if [[ -e "$destination" ]]; then
    echo "Refusing to overwrite existing mjlab checkout: $destination" >&2
    exit 1
fi

mkdir -p "$lab_root/.local"
git clone --no-checkout "$repository" "$destination"
git -C "$destination" checkout --detach "$revision"
uv sync --project "$destination" --python 3.13 --locked --extra cu128 --no-dev
uv pip install --python "$destination/.venv/bin/python" \
    --index https://pypi.nvidia.com/ warp-lang==1.16.0.dev20260716
echo "Prepared native mjlab with the reference MJWarp runtime. Use uv run --no-project with its Python interpreter."
