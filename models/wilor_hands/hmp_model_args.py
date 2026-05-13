import os

import numpy as np
import yaml


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


class Arguments:
    """Loads the minimal HMP runtime config used by STRIDE pose refinement."""

    def __init__(self, base_dir, config_path, filename="default.yaml", mano_dir=None):
        with open(os.path.join(config_path, "mano.yaml"), "r", encoding="utf-8") as handle:
            smpl = yaml.safe_load(handle)

        self.smpl = Struct(**smpl)
        self.smpl.offsets["right"] = np.array(self.smpl.offsets["right"])
        self.smpl.offsets["left"] = np.array(self.smpl.offsets["left"])
        self.smpl.parents = np.array(self.smpl.parents).astype(np.int32)
        self.smpl.joint_num = 16
        self.smpl.joints_to_use = np.array(self.smpl.joints_to_use)
        self.smpl.joints_to_use = np.arange(0, 63).reshape((-1, 3))[self.smpl.joints_to_use].reshape(-1)
        self.smpl.smpl_body_model = self._resolve_mano_dir(base_dir, mano_dir)

        self.filename = os.path.splitext(filename)[0]
        with open(os.path.join(config_path, filename), "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)

        for key, value in config.items():
            if isinstance(value, dict):
                setattr(self, key, Struct(**value))
            else:
                setattr(self, key, value)

        self.json = config

    @staticmethod
    def _resolve_mano_dir(base_dir, mano_dir):
        candidates = []
        if mano_dir is not None:
            candidates.append(os.fspath(mano_dir))
        candidates.extend(
            [
                os.path.join(base_dir, "mano_data"),
                os.path.join(base_dir, "mano_data", "mano_data"),
                os.path.join(base_dir, "WiLoR", "mano_data"),
                os.path.join(base_dir, "_DATA", "data", "mano"),
            ]
        )
        for candidate in candidates:
            if os.path.exists(os.path.join(candidate, "MANO_RIGHT.pkl")):
                return candidate
        raise FileNotFoundError(
            "Could not locate a MANO asset directory containing MANO_RIGHT.pkl. "
            f"Checked: {candidates}"
        )
