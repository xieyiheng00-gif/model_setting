import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)  # tests reference configs/*.yaml relative to the repo root


@pytest.fixture(scope="session")
def data_prep():
    """llm_training/data_prep.loader (../llm_training, $LMARCH_DATA_PREP, or importable)."""
    pytest.importorskip("pyarrow")
    from lmarch.data.packed import import_data_prep
    try:
        return import_data_prep("")
    except ImportError as e:
        pytest.skip(str(e))


@pytest.fixture(scope="session")
def packed(tmp_path_factory, data_prep):
    """Synthetic packed dataset with the real layout (vocab 512)."""
    from lmarch.data.synthetic import write_synthetic_packed
    return write_synthetic_packed(tmp_path_factory.mktemp("packed"), vocab_size=512)
