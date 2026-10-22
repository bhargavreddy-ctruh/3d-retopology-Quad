import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

KOALA_PATH = (
    "/Users/bhargavreddy/Downloads/3D/obj/"
    "pngtree-cute-cartoon-animal-character-clipart-colorful-koala-png-image_9200765_raw_1790k.glb.glb"
)


@pytest.fixture(scope="session")
def koala_glb_path():
    assert Path(KOALA_PATH).exists(), f"Test asset missing: {KOALA_PATH}"
    return KOALA_PATH
