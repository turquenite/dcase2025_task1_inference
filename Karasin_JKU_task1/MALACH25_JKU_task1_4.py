"""
DCASE 2025 full fine-tuned baseline model — Modular API for ASC inference.
"""

from typing import Optional, List, Tuple
import importlib.resources as pkg_resources

import torch
from torch import Tensor

import MALACH25_JKU_task1._common as _common
from MALACH25_JKU_task1 import ckpts

DEFAULT_CKPT = "sys4.ckpt"


def load_model(model_file_path: Optional[str] = None) -> _common.Baseline:
    """
    Load the full fine-tuned baseline model from a checkpoint.

    Args:
        model_file_path: Optional path to a .ckpt file. If None, uses the default packaged checkpoint.

    Returns:
        A Baseline model instance with loaded weights.
    """
    # Use default checkpoint from package resources if no path is given
    if model_file_path is None:
        with pkg_resources.path(ckpts, DEFAULT_CKPT) as ckpt_path:
            model_file_path = str(ckpt_path)

    model = _common.load_model(model_file_path)
    return model


def load_inputs(
    file_paths: List[str],
    device_ids: List[str],
    model: _common.Baseline,
    num_workers: int = 16,
    batch_size: int = 256
) -> List[Tensor]:
    """
    Load and preprocess audio files in parallel with batch-wise resampling and STFT.

    Args:
        file_paths: List of .wav file paths.
        device_ids: List of corresponding device IDs (same length as file_paths).
        model: Baseline model (used for preprocessing).
        num_workers: Number of threads used for parallel loading.
        batch_size: Number of waveforms per batch for STFT/mel processing.

    Returns:
        List of mel spectrogram tensors [1, 1, n_mels, T], in same order as file_paths.
    """
    inputs = _common.load_inputs(**locals())  # forward all args as keyword args
    return inputs


def get_model_for_device(
    model: _common.Baseline,
    device_id: str
) -> torch.nn.Module:
    """
    Extract the device model corresponding to a specific device ID.

    Args:
        model: Baseline model instance.
        device_id: Device identifier string (e.g., 's1').

    Returns:
        The device model (nn.Module) associated with the given device.
    """
    device_model = _common.get_model_for_device(**locals())  # forward all args as keyword args
    return device_model


def predict(
    file_paths: List[str],
    device_ids: List[str],
    model_file_path: Optional[str] = None,
    use_cuda: bool = True,
    batch_size: int = 64
) -> Tuple[List[torch.Tensor], List[str]]:
    """
    Run inference on a list of audio files using device-specific models.

    Files are grouped by device ID and processed in batches.

    Args:
        file_paths: List of audio file paths.
        device_ids: List of device IDs corresponding to each file.
        model_file_path: Optional path to a model checkpoint (.ckpt).
        use_cuda: Whether to use GPU (if available).
        batch_size: Number of examples per inference batch.

Returns:
    A tuple (logits, class_order)
    - logits: List of tensors, one per file, each of shape [n_classes]
    - class_order: List of class names corresponding to the output logits
    """
    model = load_model(model_file_path)
    result = _common.predict(
        file_paths=file_paths,
        device_ids=device_ids,
        model_or_path=model,
        use_cuda=use_cuda,
        batch_size=batch_size
    )
    return result
