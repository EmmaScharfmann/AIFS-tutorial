import os
import ssl

import certifi
import torch


def configure_ssl():
    """Point OpenSSL to the certifi bundle (needed on some macOS setups)."""
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
    ssl._create_default_https_context = ssl.create_default_context


def get_device() -> str:
    """Return ``'cuda'``, ``'mps'``, or ``'cpu'`` in priority order."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def device_label() -> str:
    """Human-readable description of the active compute device."""
    device = get_device()
    if device == "cuda":
        return f"CUDA — {torch.cuda.get_device_name(0)}"
    if device == "mps":
        return "Apple MPS (Metal)"
    return "CPU (no GPU detected — inference will be slow)"
