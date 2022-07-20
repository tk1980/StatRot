import torch
import torch.nn as nn
import torch.nn.functional as F
import math

__all__ = [
    'StatRotLinear', 'LogitAdjust'
]

class StatRotLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False, alpha=45):
        super(StatRotLinear, self).__init__(in_features=in_features, out_features=out_features, bias=bias)
        self.alpha = alpha * math.pi / 180
    
    def forward(self, input):
        logit = F.linear(input, self.weight, None) # dot-product
        if self.training:
            with torch.no_grad():
                D = float(min(self.in_features, self.out_features))
                eps = torch.randn_like(logit).div(math.sqrt(D - 1.))
            norms = input.pow(2).sum(dim=1,keepdim=True) * self.weight.pow(2).sum(dim=1)
            logit = math.cos(self.alpha) * logit  + math.sin(self.alpha) * eps * torch.sqrt(norms - logit**2)
        
        if self.bias is not None:
            logit = logit + self.bias

        return logit
    
    def extra_repr(self):
        return super(StatRotLinear, self).extra_repr() + ', alpha={}'.format(self.alpha * 180 / math.pi)


class LogitAdjust(nn.Module):
    def __init__(self, class_counts):
        super(LogitAdjust, self).__init__()
        base_probs = torch.tensor(class_counts, dtype=torch.float32)
        base_probs = base_probs / base_probs.sum()
        self.register_buffer("base_probs_log", base_probs.log())

    def forward(self, input):
        if self.training:
            input = input + self.base_probs_log
        return input
