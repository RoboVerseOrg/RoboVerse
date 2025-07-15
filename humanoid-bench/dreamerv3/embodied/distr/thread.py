import ctypes
import threading

from . import utils


class Thread:
    def __init__(self, fn, *args, name=None, start=False):
        self.fn = fn
        self._exitcode = None
        self.exception = None
        name = name or fn.__name__
        self.old_name = name[:]
        self.thread = threading.Thread(target=self._wrapper, args=args, name=name, daemon=True)
        self.started = False
        start and self.start()

    @property
    def name(self):
        return self.thread.name

    @property
    def ident(self):
        return self.thread.ident

    @property
    def alive(self):
        return self.thread.is_alive()

    @property
    def exitcode(self):
        return self._exitcode

    def start(self):
        assert not self.started
        self.started = True
        self.thread.start()

    def check(self):
        assert self.started
        if self.exception:
            e = self.exception
            self.exception = None
            raise e

    def join(self, timeout=None):
        self.check()
        self.thread.join(timeout)

    def terminate(self):
        if not self.alive:
            return
        thread = self.thread
        if hasattr(thread, "_thread_id"):
            thread_id = thread._thread_id
        else:
            thread_id = [k for k, v in threading._active.items() if v is thread][0]
        result = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(thread_id), ctypes.py_object(SystemExit))
        if result > 1:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(thread_id), None)

    def __repr__(self):
        attrs = ("name", "ident", "started", "exitcode")
        attrs = [f"{k}={getattr(self, k)}" for k in attrs]
        return f"{type(self).__name__}(" + ", ".join(attrs) + ")"

    def _wrapper(self, *args):
        try:
            self.fn(*args)
            self._exitcode = 0
        except SystemExit:
            return
        except Exception as e:
            self.exception = e
            self._exitcode = 1
            utils.warn_remote_error(e, self.name)


class StoppableThread(Thread):
    def __init__(self, fn, *args, name=None, start=False):
        self.runflag = None

        def fn2(*args):
            assert self.runflag is not None
            context = utils.Context(lambda: self.runflag)
            fn(context, *args)

        super().__init__(fn2, *args, name=name, start=start)

    def start(self):
        self.runflag = True
        super().start()

    def stop(self, wait=10):
        self.runflag = False
        self.check()
        if not self.alive:
            return
        if wait is True:
            self.join()
        elif wait:
            self.join(wait)
            if self.alive:
                print(f"Terminating thread '{self.name}' that did not want to stop.")
                self.terminate()
