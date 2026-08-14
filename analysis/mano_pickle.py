import pickle
import sys
import types


class _DummyCh:
    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self.state = state


class _DummySelect(_DummyCh):
    pass


def _install_chumpy_shim():
    existing = {name: sys.modules.get(name) for name in ("chumpy", "chumpy.ch", "chumpy.reordering")}

    chumpy = types.ModuleType("chumpy")
    ch = types.ModuleType("chumpy.ch")
    reordering = types.ModuleType("chumpy.reordering")
    ch.Ch = _DummyCh
    reordering.Select = _DummySelect
    chumpy.ch = ch
    chumpy.reordering = reordering

    sys.modules["chumpy"] = chumpy
    sys.modules["chumpy.ch"] = ch
    sys.modules["chumpy.reordering"] = reordering
    return existing


def _restore_modules(previous):
    for name, module in previous.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def load_mano_pickle(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f, encoding="latin1")
    except ModuleNotFoundError as exc:
        if exc.name != "chumpy":
            raise

    previous = _install_chumpy_shim()
    try:
        with open(path, "rb") as f:
            return pickle.load(f, encoding="latin1")
    finally:
        _restore_modules(previous)
