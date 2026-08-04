import torch

from networks.infer_lgunet import infer_lgunet
from utils.accelerator import get_accelerator


_ACCELERATOR = get_accelerator()


class InferLGUNetModel:
    def __init__(self, logger, **params) -> None:
        self.logger = logger
        self.params = params
        self.device = _ACCELERATOR.device(0)

        network_params = params.get("network_params", {})
        self.model = {"infer_lgunet": infer_lgunet(**network_params).to(self.device)}

    def to(self, device):
        self.device = torch.device(device)
        for key in self.model:
            self.model[key].to(self.device)
