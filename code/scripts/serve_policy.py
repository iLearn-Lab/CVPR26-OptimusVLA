from __future__ import annotations

import dataclasses
import enum
import logging
from pathlib import Path
import socket

import torch
import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.task_head.local_consistency_memory import LocalConsistencyMemory
from openpi.task_head.memory_init import MemoryInitProvider
from openpi.task_head.task_head_mlp import TaskHeadMLP
from openpi.training import config as _config


_REPO_ROOT = Path(__file__).resolve().parents[1]


class EnvMode(enum.Enum):
    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    config: str
    dir: str


@dataclasses.dataclass
class Default:
    pass


@dataclasses.dataclass
class Args:
    env: EnvMode = EnvMode.LIBERO
    default_prompt: str | None = None
    port: int = 8000
    record: bool = False
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)

    use_memory: bool = True
    task_head_ckpt: str = str(_REPO_ROOT / "checkpoints" / "gpm_task_head.pt")
    memory_meta_path: str = str(_REPO_ROOT / "memory" / "gpm_memory_meta.pt")
    faiss_index_path: str = str(_REPO_ROOT / "memory" / "gpm_memory.index")
    memory_actions_path: str = str(_REPO_ROOT / "memory" / "gpm_memory_actions.npz")
    action_norm_stats_path: str = ""
    action_use_quantile_norm: bool = True

    align_mode: str = "hybrid"
    mixture_mode: str = "gaussian"
    temperature: float = 10.0
    sigma_min: float = 0.05
    noise_min: float = 0.20
    noise_max: float = 1.00
    nfe_min: int = 1
    nfe_max: int = 10
    memory_top_k: int = 8
    memory_refresh_every: int = 1
    memory_refresh_sim_threshold: float = 0.0
    progress_mode: str = "client"
    replan_steps: int = 10
    debug_memory: bool = False

    use_lcm: bool = False
    lcm_ckpt: str = str(_REPO_ROOT / "checkpoints" / "lcm.pt")
    lcm_hidden: int = 256
    lcm_layers: int = 1
    lcm_heads: int = 4
    lcm_dropout: float = 0.0
    lcm_mamba_impl: str = "auto"
    lcm_mamba_state: int = 16
    lcm_mamba_conv: int = 4
    lcm_mamba_expand: int = 2
    lcm_scale: float = 0.10
    lcm_debug: bool = False


DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
}


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    checkpoint = DEFAULT_CHECKPOINT.get(env)
    if checkpoint is None:
        raise ValueError(f"Unsupported environment mode: {env}")
    return _policy_config.create_trained_policy(
        _config.get_config(checkpoint.config),
        checkpoint.dir,
        default_prompt=default_prompt,
    )


def create_policy(args: Args) -> _policy.Policy:
    match args.policy:
        case Checkpoint():
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config),
                args.policy.dir,
                default_prompt=args.default_prompt,
            )
        case Default():
            return create_default_policy(args.env, default_prompt=args.default_prompt)


def _require_file(name: str, path: str) -> None:
    if not path or not Path(path).exists():
        raise FileNotFoundError(f"{name} not found: {path}")


def _torch_load_cpu(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_task_head(path: str, device: str) -> TaskHeadMLP:
    ckpt = _torch_load_cpu(path)
    head = TaskHeadMLP(
        in_dim=int(ckpt["in_dim"]),
        hidden=int(ckpt["hidden"]),
        out_dim=int(ckpt["out_dim"]),
    )
    state_dict = ckpt.get("state_dict") or ckpt.get("task_head")
    if state_dict is None:
        raise KeyError(f"Cannot find task head weights in {path}")
    head.load_state_dict(state_dict, strict=True)
    return head.eval().float().to(device)


def _clear_runtime_state(model) -> None:
    model.memory_session = None
    model._sample_call_count = 0
    model._external_progress = 0.0
    for attr in ("_last_action_chunk", "_lcm_prev_chunk", "_lcm_h"):
        if hasattr(model, attr):
            setattr(model, attr, None)


def _attach_runtime_helpers(model, args: Args) -> None:
    def set_progress(progress: float) -> None:
        model._external_progress = float(max(0.0, min(1.0, progress)))

    def reset_memory_session() -> None:
        _clear_runtime_state(model)

    model.set_progress = set_progress
    model.reset_memory_session = reset_memory_session
    model.progress_mode = args.progress_mode
    model.replan_steps_hint = int(args.replan_steps)


def _inject_gpm_lcm(model, args: Args) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.use_memory = bool(args.use_memory)
    _attach_runtime_helpers(model, args)
    _clear_runtime_state(model)

    if not args.use_memory:
        model.task_head = None
        model.memory_provider = None
        model.lcm = None
        model.use_lcm = False
        logging.info("GPM is disabled; serving the base policy.")
        return

    _require_file("task_head_ckpt", args.task_head_ckpt)
    _require_file("memory_meta_path", args.memory_meta_path)
    _require_file("faiss_index_path", args.faiss_index_path)
    _require_file("memory_actions_path", args.memory_actions_path)
    if args.action_norm_stats_path:
        _require_file("action_norm_stats_path", args.action_norm_stats_path)

    model.task_head = _load_task_head(args.task_head_ckpt, device)
    model.memory_provider = MemoryInitProvider(
        memory_meta_path=args.memory_meta_path,
        faiss_index_path=args.faiss_index_path,
        memory_actions_path=args.memory_actions_path,
        align_mode=args.align_mode,
        mixture_mode=args.mixture_mode,
        temperature=args.temperature,
        sigma_min=args.sigma_min,
        noise_scale_range=(args.noise_min, args.noise_max),
        nfe_range=(args.nfe_min, args.nfe_max),
        device=device,
        action_norm_stats_path=args.action_norm_stats_path or None,
        action_use_quantile_norm=args.action_use_quantile_norm,
    )
    model.memory_top_k = int(args.memory_top_k)
    model.memory_refresh_every = int(args.memory_refresh_every)
    model.memory_refresh_sim_threshold = float(args.memory_refresh_sim_threshold)
    model.debug_memory = bool(args.debug_memory)
    logging.info(
        "Loaded GPM: task_head=%s memory_items=%d top_k=%d refresh_every=%d",
        args.task_head_ckpt,
        int(model.memory_provider.num_items),
        model.memory_top_k,
        model.memory_refresh_every,
    )

    if args.use_lcm:
        _require_file("lcm_ckpt", args.lcm_ckpt)
        lcm_ckpt = _torch_load_cpu(args.lcm_ckpt)
        meta = lcm_ckpt.get("meta", {})
        horizon = int(model.config.action_horizon)
        action_dim = int(model.config.action_dim)
        if int(meta.get("H", horizon)) != horizon or int(meta.get("A", action_dim)) != action_dim:
            raise RuntimeError(
                f"LCM checkpoint shape mismatch: checkpoint H/A={meta.get('H')}/{meta.get('A')} "
                f"model H/A={horizon}/{action_dim}"
            )
        lcm = LocalConsistencyMemory(
            action_dim=action_dim,
            horizon=horizon,
            hidden=int(meta.get("hidden", args.lcm_hidden)),
            n_layers=int(meta.get("layers", args.lcm_layers)),
            n_heads=int(meta.get("heads", args.lcm_heads)),
            dropout=float(meta.get("dropout", args.lcm_dropout)),
            use_tanh=bool(meta.get("use_tanh", False)),
            mamba_impl=str(meta.get("mamba_impl", args.lcm_mamba_impl)),
            mamba_state=int(meta.get("mamba_state", args.lcm_mamba_state)),
            mamba_conv=int(meta.get("mamba_conv", args.lcm_mamba_conv)),
            mamba_expand=int(meta.get("mamba_expand", args.lcm_mamba_expand)),
        ).to(device).eval()
        key = "lcm" if "lcm" in lcm_ckpt else "state_dict"
        lcm.load_state_dict(lcm_ckpt[key], strict=True)
        model.lcm = lcm
        model.use_lcm = True
        model.lcm_scale = float(args.lcm_scale)
        model.debug_lcm = bool(args.lcm_debug)
        logging.info("Loaded LCM: ckpt=%s scale=%.3f", args.lcm_ckpt, model.lcm_scale)
    else:
        model.lcm = None
        model.use_lcm = False
        model.lcm_scale = 1.0
        model.debug_lcm = False


class ProgressAwarePolicy:
    """Submit per-episode progress to the underlying PyTorch model before inference."""

    def __init__(self, base_policy: _policy.Policy):
        self._base = base_policy

    @property
    def _model(self):
        return self._base._model

    @property
    def metadata(self):
        return self._base.metadata

    def infer(self, element: dict):
        reset_flag = element.pop("episode_reset", None)
        if reset_flag and hasattr(self._model, "reset_memory_session"):
            self._model.reset_memory_session()

        progress = element.pop("progress", None)
        if getattr(self._model, "progress_mode", "client") == "client" and progress is not None:
            if hasattr(self._model, "set_progress"):
                self._model.set_progress(float(progress))

        return self._base.infer(element)


def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = policy.metadata
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    _inject_gpm_lcm(policy._model, args)
    policy = ProgressAwarePolicy(policy)

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server on %s (%s), port %d", hostname, local_ip, args.port)
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
