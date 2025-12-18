class BaseFlowMatcher():
    def compute_loss(self, model, target, **kwargs):
        raise NotImplementedError

    def sample(self, model, shape, device, num_steps, return_traces=False, **kwargs):
        raise NotImplementedError
