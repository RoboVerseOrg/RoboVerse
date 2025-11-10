from .base import EnvTypes


class SB3EnvWrapper:
    def __init__(self, env: EnvTypes):
        self.env = env

class RlLibEnvWrapper:
    def __init__(self, env: EnvTypes):
        self.env = env
