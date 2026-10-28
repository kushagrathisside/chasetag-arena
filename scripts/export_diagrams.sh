#!/usr/bin/env bash
# Export every docs/diagrams/*.drawio to an SVG next to it, for the documentation site.
#
# Edit a diagram in the draw.io desktop app (or app.diagrams.net), save the .drawio file,
# then run this script. It needs the draw.io desktop app on PATH as `drawio`. Without a
# display it runs under xvfb-run, when that is installed.
set -euo pipefail
cd "$(dirname "$0")/.."

runner=()
if [[ -z "${DISPLAY:-}" ]] && command -v xvfb-run >/dev/null; then
    runner=(xvfb-run -a)
fi

shopt -s nullglob
sources=(docs/diagrams/*.drawio)
if (( ${#sources[@]} == 0 )); then
    echo "no .drawio files in docs/diagrams" >&2
    exit 1
fi

for src in "${sources[@]}"; do
    out="${src%.drawio}.svg"
    timeout 180 "${runner[@]}" drawio --export --format svg --output "$out" "$src" >/dev/null 2>&1
    # draw.io adds a rasterised PNG copy of every HTML label, for viewers without
    # <foreignObject> support. Browsers render the real text, so the copies are removed.
    python3 - "$out" <<'PY'
import re, sys
path = sys.argv[1]
svg = open(path, encoding="utf-8").read()
svg = re.sub(r'<image [^>]*xlink:href="data:image/png;base64,[^"]*"[^>]*/>', "", svg)
open(path, "w", encoding="utf-8").write(svg.rstrip("\n") + "\n")
PY
    echo "exported $out"
done
