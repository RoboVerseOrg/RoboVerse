class LPFilter:
    """Low-pass filter for smoothing signals.

    Args:
        alpha (float): Filter coefficient between 0 and 1.
    """

    def __init__(self, alpha):
        self.alpha = alpha
        self.y = None
        self.is_init = False

    def next(self, x):
        """Apply the filter to the input signal.

        Args:
            x (float): The input signal.

        Returns:
            float: The filtered signal.
        """
        if not self.is_init:
            self.y = x
            self.is_init = True
            return self.y.copy()
        self.y = self.y + self.alpha * (x - self.y)
        return self.y.copy()

    def reset(self):
        """Reset the filter.

        This method is used to reset the filter to its initial state.
        """
        self.y = None
        self.is_init = False
