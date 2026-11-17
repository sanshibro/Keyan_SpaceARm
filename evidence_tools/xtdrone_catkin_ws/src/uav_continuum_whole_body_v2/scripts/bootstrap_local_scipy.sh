#!/usr/bin/env bash
set -euo pipefail

PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPOSITORY="$(cd "$PACKAGE_DIR/../../.." && pwd)"
DEB_DIR="$REPOSITORY/.v2_deps/debs"
ROOT_DIR="$REPOSITORY/.v2_deps/root"

mkdir -p "$DEB_DIR" "$ROOT_DIR"
cd "$DEB_DIR"
apt download python3-scipy liblbfgsb0 python3-decorator
for package in ./*.deb; do
  dpkg-deb -x "$package" "$ROOT_DIR"
done

echo "Local SciPy runtime prepared at $ROOT_DIR"
echo "Run: source $PACKAGE_DIR/env_v2_deps.sh"
