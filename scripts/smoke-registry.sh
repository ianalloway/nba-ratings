#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python3}"
version="$(sed -nE 's/^version = "([^"]+)"$/\1/p' "$repo_dir/pyproject.toml")"
if [[ -z "$version" ]]; then
  echo "Could not read project version from pyproject.toml" >&2
  exit 1
fi
spec="${1:-nba-edge==${version}}"
smoke_dir="$(mktemp -d)"
trap 'rm -rf "$smoke_dir"' EXIT
"$python_bin" -m venv "$smoke_dir/venv"

for attempt in 1 2 3 4 5; do
  if "$smoke_dir/venv/bin/python" -m pip install --no-cache-dir "$spec"; then
    break
  fi
  if [[ "$attempt" == 5 ]]; then
    exit 1
  fi
  sleep 5
done

# Running outside the checkout ensures imports come from the installed wheel.
cd "$smoke_dir"
EXPECTED_VERSION="$version" "$smoke_dir/venv/bin/python" <<'PY'
import os
from importlib.metadata import version

from nba_edge import (
    american_to_decimal,
    clv_edge,
    expected_margin,
    kelly_fraction,
    logistic_win_prob,
    update_elo,
)

assert version("nba-edge") == os.environ["EXPECTED_VERSION"]
assert 0 < logistic_win_prob(rating_diff=120) < 1
assert isinstance(update_elo(1600, 1580, 1.0), tuple)
assert expected_margin(rating_diff=120) > 0
assert american_to_decimal(-110) > 1
assert kelly_fraction(0.58, -110, fraction=0.25) > 0
assert clv_edge(-110, -120) > 0
print(f"Verified clean install of nba-edge=={version('nba-edge')}")
PY
