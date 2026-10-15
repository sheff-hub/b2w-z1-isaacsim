#!/bin/bash
# Pack robots_usd/ into a tarball for a GitHub release (the assets are too large for git).
#   tools/pack_assets.sh            -> dist/robots_usd.tar.gz
# Then attach it to a release and install with:  ASSETS_URL=<its download URL> ./setup.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "${ROOT}/dist"
tar -czf "${ROOT}/dist/robots_usd.tar.gz" -C "${ROOT}/robots_usd" --exclude README.md --exclude '*.bak*' .
ls -la "${ROOT}/dist/robots_usd.tar.gz"
