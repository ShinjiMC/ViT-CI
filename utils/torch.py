import os

import numpy as np
import torch
import torch.nn as nn


def tensor_to_array(tensor: torch.Tensor) -> np.ndarray:
    return tensor.cpu().detach().numpy()

def get_device(device_name: str) -> torch.device:
    try:
        device = torch.device(device_name)
    except RuntimeError as e:
        print(f"[Device name error] Use cpu device! Error details: {e}")
        device = torch.device("cpu")
    return device

def save_model(model: nn.Module, dir_name: str, file_name: str):
    os.makedirs(dir_name, exist_ok=True)
    dir_name = dir_name.strip()
    if dir_name[-1] != "/":
        dir_name += "/"
    save_path = dir_name + file_name
    if "best_model" in file_name:
        save_path += ".pth"
        torch.save(model.state_dict(), save_path)
    else:
        torch.save(model.state_dict(), save_path)

def load_model(model: nn.Module, path: str):
    model.load_state_dict(torch.load(path))
