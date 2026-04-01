from pathlib import Path
import os


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def model_assets_root() -> Path:
    override = os.environ.get("MODEL_ASSETS_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return (repo_root() / "model_assets").resolve()


def require_model_asset(relative_path: str, description: str, env_var: str | None = None) -> Path:
    if env_var:
        override = os.environ.get(env_var)
        if override:
            path = Path(override).expanduser().resolve()
        else:
            path = model_assets_root() / relative_path
    else:
        path = model_assets_root() / relative_path

    if not path.exists():
        raise FileNotFoundError(f"{description} not found at: {path}")
    return path
