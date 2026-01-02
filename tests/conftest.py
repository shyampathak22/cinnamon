import sys
from pathlib import Path

import pytest
import torch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import ModelConfig


@pytest.fixture
def cfg():
    """Default model configuration for tests."""
    return ModelConfig()


@pytest.fixture
def small_cfg():
    """Smaller configuration for faster tests."""
    cfg = ModelConfig()
    cfg.d_model = 256
    cfg.n_layers = 2
    cfg.n_heads = 4
    cfg.max_seq_len = 128
    cfg.d_ckv = 128
    cfg.d_cq = 128
    cfg.d_head = 32
    cfg.d_v = 32
    cfg.d_rope = 16
    return cfg


@pytest.fixture
def device():
    """Return available device (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "cuda: mark test as requiring CUDA")
    config.addinivalue_line("markers", "slow: mark test as slow running")


def pytest_collection_modifyitems(config, items):
    """Skip CUDA tests if CUDA is not available."""
    if not torch.cuda.is_available():
        skip_cuda = pytest.mark.skip(reason="CUDA not available")
        for item in items:
            if "cuda" in item.keywords:
                item.add_marker(skip_cuda)
