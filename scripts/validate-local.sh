#!/bin/bash
set -euo pipefail

echo "Checking Docker availability..."
docker --version

echo "Building DagFlow image..."
docker compose build

echo "Running R tests..."
docker compose run --rm --workdir /workspace dagflow test

echo "Validation complete."
