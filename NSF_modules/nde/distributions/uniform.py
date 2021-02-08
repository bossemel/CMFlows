from torch import distributions
import torch
import NSF_modules.utils as utils


class TweakedUniform(distributions.Uniform):
    def __init__(self, low, high, device):
        super().__init__(low, high)
        self.device = device

    def log_prob(self, value, context):
        log_prob = super().log_prob(value)
        assert torch.isfinite(log_prob.sum())
        return utils.sum_except_batch(log_prob, 1).to(self.device)
        # result = super().log_prob(value)
        # if len(result.shape) == 2 and result.shape[1] == 1:
        #     return result.reshape(-1)
        # else:
        #     return result

    def sample(self, num_samples, context):
        return super().sample((num_samples, )).to(self.device)
