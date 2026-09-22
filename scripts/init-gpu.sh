#!/usr/bin/env bash
set -eou pipefail

CUDA_LIB_DIRS=$(
	find .venv/lib/python3.11/site-packages/nvidia \
		-type d -name lib -print |
	paste -sd:
)

export LD_LIBRARY_PATH="$CUDA_LIB_DIRS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec uv run --no-sync "$@"

