#!/usr/bin/env bash
# Milestone 0 spike: verify the prebuilt QuadWild + Bi-MDF binaries actually produce a
# valid, low-singularity quad mesh from the koala test asset, with zero Python pipeline
# code involved. This is the go/no-go gate for the whole architecture (see
# /Users/bhargavreddy/.claude/plans/users-bhargavreddy-downloads-3d-researc-cheerful-lemur.md).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QW_DIR="$HERE/../third_party/quadwild_bimdf"
WORK_DIR="$HERE/smoke_work"

mkdir -p "$WORK_DIR"
cp "$HERE/koala.obj" "$WORK_DIR/koala.obj"

cd "$QW_DIR"

echo "=== Step 1: quadwild (feature detection + pre-remesh) ==="
./quadwild "$WORK_DIR/koala.obj" 2 config/prep_config/basic_setup_Organic.txt

echo "=== Step 2: quad_from_patches (Bi-MDF quantization) ==="
./quad_from_patches "$WORK_DIR/koala_rem_p0.obj" 123 config/main_config/flow_noalign_lemon.txt

echo "=== Done. Outputs in $WORK_DIR ==="
ls -la "$WORK_DIR"
