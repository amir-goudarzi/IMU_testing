import torch
import torch.nn as nn

def normalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    normalized_tensor = nn.Softmax(dim=1)(tensor)
    return normalized_tensor

def cut_and_pad(signal: torch.Tensor, sampling_rate: int, seconds) -> torch.Tensor:
    bins = int(sampling_rate * seconds)
    padded_signal = torch.zeros((signal.shape[0], bins))
    if signal.shape[-1] < sampling_rate * seconds:
        padded_signal = torch.zeros((signal.shape[0], bins))
        padded_signal[:, :signal.shape[-1]] = signal
    else:
        padded_signal = signal[:bins]

    return padded_signal
