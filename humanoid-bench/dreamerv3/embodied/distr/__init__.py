import multiprocessing as mp

try:
    mp.set_start_method("spawn")
except RuntimeError:
    pass

from .client import Client
from .proc_server import ProcServer
from .process import Process, StoppableProcess
from .server import Server
from .sockets import NotAliveError, ProtocolError, RemoteError
from .thread import StoppableThread, Thread
from .utils import run, terminate_subprocesses, warn_remote_error
