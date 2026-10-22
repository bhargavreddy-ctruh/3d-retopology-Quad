# QuadWild + Bi-MDF binaries — provenance

- Source: https://github.com/cgg-bern/quadwild-bimdf
- Release: v0.0.2 ("Better windows support.")
- Asset: macos-binaries.zip (5.12 MB, SHA/commit eef4fc3)
- Downloaded: 2026-10-21
- License: GPL-3.0 (see upstream repo LICENSE)
- Contents: `quadwild`, `quad_from_patches` (universal Mach-O: arm64 + x86_64), plus sample
  `config/prep_config/`, `config/main_config/`, `config/satsuma/` files.
- Usage (from upstream README):
  ```
  ./quadwild path/to/mesh.obj 2 config/prep_config/basic_setup.txt
  ./quad_from_patches path/to/mesh_rem_p0.obj 123 config/main_config/flow_noalign_lemon.txt
  ```
- Note: `config/prep_config/basic_setup_Organic.txt` ships in this release and is a better starting
  point than `basic_setup.txt` for organic character meshes (e.g. the koala test asset).
- This directory is GPL-3.0 licensed third-party code, kept outside the Python venv and invoked only
  via subprocess from `src/retopo/crossfield_backend.py` / `src/retopo/quantization.py`.
