import torch.nn as nn
import torch
class ModuleAttrMixin(nn.Module):
    def __init__(self):
        super().__init__()
        #self._dummy_variable = nn.Parameter()
        assert "torch" in globals(), "Torch is not defined in the current context!"
        self._dummy_variable = nn.Parameter(torch.empty(0), requires_grad=False)
        #self.register_buffer('_dummy_variable', torch.zeros(1))
        #self._dummy_variable = nn.Parameter(torch.zeros(1, requires_grad=False))
    @property
    def device(self):
        return next(iter(self.parameters())).device
    
    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
