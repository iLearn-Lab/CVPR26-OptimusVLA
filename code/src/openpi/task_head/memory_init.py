from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import faiss  # type: ignore
import numpy as np
import torch

try:
    from torch._dynamo import disable as dynamo_disable
except Exception:

    def dynamo_disable(fn):
        return fn


def load_action_normalizer(
    norm_stats_path: Optional[str],
    *,
    use_quantiles: bool = True,
) -> Optional[dict[str, np.ndarray]]:
    if not norm_stats_path:
        return None
    with open(norm_stats_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    stats_root = data.get("norm_stats", data)
    if "actions" not in stats_root:
        raise KeyError(f"Cannot find actions norm stats in {norm_stats_path}")

    stats = stats_root["actions"]
    out: dict[str, np.ndarray] = {"use_quantiles": np.asarray(bool(use_quantiles))}
    for key in ("mean", "std", "q01", "q99"):
        value = stats.get(key)
        if value is not None:
            out[key] = np.asarray(value, dtype=np.float32)

    if use_quantiles and ("q01" not in out or "q99" not in out):
        raise KeyError(f"actions q01/q99 are required for quantile normalization: {norm_stats_path}")
    if not use_quantiles and ("mean" not in out or "std" not in out):
        raise KeyError(f"actions mean/std are required for z-score normalization: {norm_stats_path}")
    return out


def normalize_actions_array(
    actions: np.ndarray,
    normalizer: Optional[dict[str, np.ndarray]],
) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    if normalizer is None:
        return actions
    out = actions.copy()
    if bool(normalizer["use_quantiles"]):
        q01 = normalizer["q01"]
        q99 = normalizer["q99"]
        dim = min(out.shape[-1], q01.shape[-1])
        out[..., :dim] = (out[..., :dim] - q01[:dim]) / (q99[:dim] - q01[:dim] + 1e-6) * 2.0 - 1.0
    else:
        mean = normalizer["mean"]
        std = normalizer["std"]
        dim = min(out.shape[-1], mean.shape[-1])
        out[..., :dim] = (out[..., :dim] - mean[:dim]) / (std[:dim] + 1e-6)
    return out.astype(np.float32, copy=False)


def _torch_load_cpu(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


@dynamo_disable
def _load_actions_npz(path: str) -> np.ndarray:
    with np.load(path) as npz:
        return np.asarray(npz["actions"], dtype=np.float32)


def _enumerate_chunks(actions: np.ndarray, chunk_len: int, stride: int) -> np.ndarray:
    total_steps, action_dim = actions.shape
    horizon = int(chunk_len)
    if total_steps < horizon:
        return np.empty((0, horizon, action_dim), dtype=np.float32)
    count = 1 + (total_steps - horizon) // int(stride)
    out = np.empty((count, horizon, action_dim), dtype=np.float32)
    start = 0
    for i in range(count):
        out[i] = actions[start : start + horizon]
        start += int(stride)
    return out


def _time_resample_to_horizon(actions: np.ndarray, horizon: int) -> np.ndarray:
    total_steps, action_dim = actions.shape
    if total_steps == horizon:
        return actions.copy()
    xs = np.linspace(0, total_steps - 1, num=total_steps, dtype=np.float32)
    qs = np.linspace(0, total_steps - 1, num=int(horizon), dtype=np.float32)
    out = np.empty((int(horizon), action_dim), dtype=np.float32)
    for i in range(action_dim):
        out[:, i] = np.interp(qs, xs, actions[:, i])
    return out


class PackedActionStore:
    """Read all memory trajectories from a single compressed NPZ file."""

    def __init__(self, path: str):
        self.path = Path(path)
        data = np.load(self.path, allow_pickle=False)
        self.actions = np.asarray(data["actions"], dtype=np.float32)
        self.offsets = np.asarray(data["offsets"], dtype=np.int64)
        if self.offsets.ndim != 1 or self.offsets.shape[0] < 2:
            raise ValueError(f"Invalid offsets array in {self.path}")
        if self.offsets[-1] != self.actions.shape[0]:
            raise ValueError(f"Packed actions and offsets do not match in {self.path}")
        self.ids: list[str] | None = None
        self.id_to_index: dict[str, int] = {}
        if "ids" in data:
            self.ids = [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in np.asarray(data["ids"]).tolist()]
            self.id_to_index = {value: i for i, value in enumerate(self.ids)}

    def __len__(self) -> int:
        return int(self.offsets.shape[0] - 1)

    def get(self, key: int | str) -> np.ndarray:
        if isinstance(key, str):
            if key not in self.id_to_index:
                raise KeyError(f"Unknown packed action id: {key}")
            index = self.id_to_index[key]
        else:
            index = int(key)
        if index < 0 or index >= len(self):
            raise IndexError(f"Packed action index out of range: {index}")
        start = int(self.offsets[index])
        end = int(self.offsets[index + 1])
        return np.asarray(self.actions[start:end], dtype=np.float32)


class MemoryInitProvider:
    """Retrieve task-conditioned action priors from GPM memory."""

    def __init__(
        self,
        memory_meta_path: str,
        faiss_index_path: str,
        *,
        memory_actions_path: str | None = None,
        align_mode: str = "hybrid",
        mixture_mode: str = "gaussian",
        temperature: float = 10.0,
        sigma_min: float = 0.05,
        noise_scale_range: tuple[float, float] = (0.2, 1.0),
        nfe_range: tuple[int, int] = (1, 10),
        device: str = "cuda",
        action_norm_stats_path: str | None = None,
        action_use_quantile_norm: bool = True,
    ):
        dev = device if device == "cuda" and torch.cuda.is_available() else "cpu"
        self.device = torch.device(dev)
        self.align_mode = str(align_mode)
        self.mixture_mode = str(mixture_mode)
        self.temperature = float(temperature)
        self.sigma_min = float(sigma_min)
        self.noise_low, self.noise_high = map(float, noise_scale_range)
        self.nfe_min, self.nfe_max = map(int, nfe_range)
        self.action_normalizer = load_action_normalizer(
            action_norm_stats_path,
            use_quantiles=bool(action_use_quantile_norm),
        )

        self.memory: list[dict] = _torch_load_cpu(memory_meta_path)
        if not self.memory:
            raise RuntimeError(f"Empty memory metadata: {memory_meta_path}")
        self.num_items = len(self.memory)
        self.index = faiss.read_index(faiss_index_path)
        if int(getattr(self.index, "ntotal", self.num_items)) != self.num_items:
            logging.warning(
                "FAISS index size and metadata length differ: index=%s metadata=%s",
                getattr(self.index, "ntotal", "unknown"),
                self.num_items,
            )
        self.action_store = PackedActionStore(memory_actions_path) if memory_actions_path else None
        if self.action_store is not None and len(self.action_store) != self.num_items:
            logging.warning(
                "Packed action count and metadata length differ: actions=%d metadata=%d",
                len(self.action_store),
                self.num_items,
            )

    def normalize_actions(self, actions: np.ndarray) -> np.ndarray:
        return normalize_actions_array(actions, self.action_normalizer)

    def _entry_action_key(self, memory_index: int) -> int | str:
        entry = self.memory[int(memory_index)]
        for key in ("action_id", "trajectory_id", "packed_id", "id"):
            if key in entry:
                return str(entry[key])
        return int(memory_index)

    def _load_actions_for_index(self, memory_index: int) -> np.ndarray:
        entry = self.memory[int(memory_index)]
        if self.action_store is not None:
            actions = self.action_store.get(self._entry_action_key(memory_index))
        elif "actions_path" in entry:
            actions = _load_actions_npz(str(entry["actions_path"]))
        else:
            raise KeyError(
                "Memory metadata must contain action_id for packed actions or actions_path for legacy loading."
            )
        return self.normalize_actions(actions)

    def _weights_from_scores(self, scores: np.ndarray) -> np.ndarray:
        x = (scores - scores.max()) * self.temperature
        weights = np.exp(x)
        weights = weights / (weights.sum() + 1e-12)
        return weights.astype(np.float32)

    def _lambda_from_similarity(self, similarity: float) -> float:
        s01 = (max(-1.0, min(1.0, float(similarity))) + 1.0) / 2.0
        return float(self.noise_high - s01 * (self.noise_high - self.noise_low))

    def _nfe_from_similarity(self, similarity: float) -> int:
        s01 = (max(-1.0, min(1.0, float(similarity))) + 1.0) / 2.0
        nfe = self.nfe_min + (1.0 - s01) * (self.nfe_max - self.nfe_min)
        return int(round(nfe))

    def _block_for_entry(self, memory_index: int, horizon: int, progress: float) -> np.ndarray | None:
        entry = self.memory[int(memory_index)]
        chunk_meta = entry.get("chunk_meta", {})
        chunk_len = int(chunk_meta.get("chunk_len", horizon))
        stride = int(chunk_meta.get("stride", 1))
        actions = self._load_actions_for_index(memory_index)
        block = None

        if self.align_mode in ("chunk", "hybrid"):
            chunks = _enumerate_chunks(actions, chunk_len, stride)
            if chunks.shape[0] > 0:
                chunk_index = int(np.clip(np.floor(progress * (chunks.shape[0] - 1)), 0, chunks.shape[0] - 1))
                block = chunks[chunk_index]

        if block is None and self.align_mode in ("resample", "hybrid"):
            block = _time_resample_to_horizon(actions, chunk_len)

        if block is not None and block.shape[0] != int(horizon):
            block = _time_resample_to_horizon(block, int(horizon))
        return block.astype(np.float32) if block is not None else None

    @dynamo_disable
    def query(
        self,
        query_emb: torch.Tensor,
        k: int,
        action_horizon: int,
        *,
        progress: float = 0.0,
    ):
        if query_emb.ndim != 1:
            raise ValueError(f"query_emb must be [D], got {tuple(query_emb.shape)}")

        query = query_emb.detach().cpu().numpy().astype("float32")[None, :]
        faiss.normalize_L2(query)
        scores_raw, idxs_raw = self.index.search(query, int(k))

        selected_scores: list[float] = []
        selected_idxs: list[int] = []
        for score, idx in zip(scores_raw[0].tolist(), idxs_raw[0].tolist()):
            if idx < 0:
                continue
            selected_scores.append(float(score))
            selected_idxs.append(int(idx))
            if len(selected_idxs) >= int(k):
                break
        if not selected_idxs:
            raise RuntimeError("FAISS returned no valid memory entries.")

        scores = np.asarray(selected_scores, dtype=np.float32)
        idxs = np.asarray(selected_idxs, dtype=np.int64)
        weights = self._weights_from_scores(scores)
        similarity = float((weights * scores).sum())

        blocks = []
        sources = []
        horizon = int(action_horizon)
        for rank, memory_index in enumerate(idxs.tolist()):
            block = self._block_for_entry(memory_index, horizon, float(progress))
            if block is None:
                continue
            blocks.append(block)
            sources.append(
                {
                    "idx": int(memory_index),
                    "action_id": self._entry_action_key(memory_index),
                    "score": float(scores[rank]),
                }
            )
        if not blocks:
            raise RuntimeError("No valid action blocks were found in memory.")

        stacked = np.stack(blocks, axis=0)
        weights = weights[: stacked.shape[0]].reshape(-1, 1, 1)
        weights = weights / (weights.sum() + 1e-12)
        sample, lam = self._sample_from_blocks(stacked, weights, similarity)
        nfe = self._nfe_from_similarity(similarity)
        x_init = torch.from_numpy(sample).to(torch.float32).to(self.device)
        info = {
            "k": int(stacked.shape[0]),
            "scores": scores[: stacked.shape[0]].tolist(),
            "weights": weights[:, 0, 0].tolist(),
            "similarity_global": similarity,
            "lambda_noise": float(lam),
            "nfe": int(nfe),
            "align_mode": self.align_mode,
            "mixture_mode": self.mixture_mode,
            "sources": sources,
            "prior_mean": float(x_init.mean().item()),
            "prior_std": float(x_init.std(unbiased=False).item()),
            "win_start": 0,
            "win_end": horizon,
        }
        return x_init, nfe, info

    def _sample_from_blocks(
        self,
        blocks: np.ndarray,
        weights: np.ndarray,
        similarity: float,
        *,
        mixture_mode: str | None = None,
    ) -> tuple[np.ndarray, float]:
        mode = mixture_mode or self.mixture_mode
        lam = self._lambda_from_similarity(similarity)
        if mode == "gaussian":
            mu = (weights * blocks).sum(axis=0)
            var = (weights * (blocks - mu[None]) ** 2).sum(axis=0)
            var = np.maximum(var, self.sigma_min**2)
            eps = np.random.randn(*mu.shape).astype(np.float32)
            sample = mu + lam * eps * np.sqrt(var).astype(np.float32)
            return sample.astype(np.float32), lam
        if mode == "mog":
            probs = weights[:, 0, 0] / (weights[:, 0, 0].sum() + 1e-12)
            choice = np.random.choice(blocks.shape[0], size=(blocks.shape[1],), p=probs)
            sample = np.empty((blocks.shape[1], blocks.shape[2]), dtype=np.float32)
            base_std = max(self.sigma_min, 0.03)
            for step in range(blocks.shape[1]):
                action = blocks[choice[step], step]
                sample[step] = action + lam * base_std * np.random.randn(*action.shape).astype(np.float32)
            return sample.astype(np.float32), lam
        raise ValueError(f"Unknown mixture_mode: {mode}")


class ActionMemorySession:
    """Cache a top-k memory query for one episode and sample action priors by progress."""

    def __init__(
        self,
        provider: MemoryInitProvider,
        init_task_emb: torch.Tensor,
        k: int,
        H: int,
        *,
        progress: float = 0.0,
    ):
        self.provider = provider
        self.H = int(H)
        self.init_emb_cpu = init_task_emb.detach().to("cpu", copy=True).contiguous()
        _, _, info = provider.query(init_task_emb, k=k, action_horizon=H, progress=progress)
        self.init_info = info
        self.nfe_adapt = int(info["nfe"])
        self.s_global = float(info["similarity_global"])
        self.blocks_all: list[np.ndarray] = []
        raw_scores: list[float] = []

        for source in info["sources"]:
            memory_index = int(source["idx"])
            entry = provider.memory[memory_index]
            chunk_meta = entry.get("chunk_meta", {})
            chunk_len = int(chunk_meta.get("chunk_len", self.H))
            stride = int(chunk_meta.get("stride", 1))
            actions = provider._load_actions_for_index(memory_index)

            if provider.align_mode in ("chunk", "hybrid"):
                chunks = _enumerate_chunks(actions, chunk_len, stride)
                if chunks.shape[0] > 0:
                    if chunk_len != self.H:
                        chunks = np.stack(
                            [_time_resample_to_horizon(chunk, self.H) for chunk in chunks],
                            axis=0,
                        )
                    self.blocks_all.append(chunks.astype(np.float32))
                    raw_scores.append(float(source["score"]))
                    continue

            if provider.align_mode in ("resample", "hybrid"):
                block = _time_resample_to_horizon(actions, self.H)[None, ...]
                self.blocks_all.append(block.astype(np.float32))
                raw_scores.append(float(source["score"]))

        if raw_scores:
            scores = np.asarray(raw_scores, dtype=np.float32)
            self.weights = provider._weights_from_scores(scores)
        else:
            self.weights = np.asarray([1.0], dtype=np.float32)

    def _chosen_blocks(self, progress: float) -> tuple[np.ndarray, np.ndarray, list[int]]:
        chosen = []
        win_starts = []
        for blocks in self.blocks_all:
            count = blocks.shape[0]
            if count <= 0:
                continue
            chunk_index = int(np.clip(np.floor(float(progress) * (count - 1)), 0, count - 1))
            chosen.append(blocks[chunk_index])
            win_starts.append(chunk_index)
        if not chosen:
            raise RuntimeError("No cached action blocks in memory session.")
        blocks = np.stack(chosen, axis=0)
        weights = self.weights[: blocks.shape[0]].reshape(-1, 1, 1)
        return blocks, weights, win_starts

    @dynamo_disable
    def prior_mean(self, progress: float) -> torch.Tensor:
        blocks, weights, _ = self._chosen_blocks(progress)
        mu = (weights * blocks).sum(axis=0).astype(np.float32)
        return torch.from_numpy(mu).to(torch.float32).to(self.provider.device)

    @dynamo_disable
    def estimate_progress(self, step_idx: int, replan_steps: int) -> float:
        lengths = []
        for source in self.init_info["sources"]:
            entry = self.provider.memory[int(source["idx"])]
            length = int(entry.get("length", 0))
            if length <= 0:
                length = int(entry.get("chunk_meta", {}).get("T", 0))
            lengths.append(length)
        if not lengths:
            return 0.0
        weights = self.weights[: len(lengths)]
        weights = weights / (weights.sum() + 1e-12)
        expected_length = float((weights * np.asarray(lengths, dtype=np.float32)).sum())
        executed_steps = float(int(step_idx) * max(1, int(replan_steps)))
        return float(np.clip(executed_steps / max(expected_length, 1.0), 0.0, 1.0))

    @dynamo_disable
    def sample_chunk(
        self,
        progress: float,
        *,
        mixture_mode: str | None = None,
        return_debug: bool = False,
    ):
        blocks, weights, win_starts = self._chosen_blocks(progress)
        sample, lam = self.provider._sample_from_blocks(
            blocks,
            weights,
            self.s_global,
            mixture_mode=mixture_mode,
        )
        x_t = torch.from_numpy(sample).to(torch.float32).to(self.provider.device)
        if not return_debug:
            return x_t

        top = np.argsort(-self.weights)[:3]
        debug = {
            "k": int(blocks.shape[0]),
            "win_start": int(win_starts[0]) if win_starts else -1,
            "win_end": int(win_starts[0] + self.H) if win_starts else -1,
            "top3_sims": [float(self.init_info["scores"][i]) for i in top[:3]],
            "top3_weights": [float(self.weights[i]) for i in top[:3]],
            "prior_mean": float(x_t.mean().item()),
            "prior_std": float(x_t.std(unbiased=False).item()),
            "noise_sigma": float(lam),
        }
        return x_t, debug

    @dynamo_disable
    def maybe_refresh(
        self,
        new_task_emb: Optional[torch.Tensor],
        *,
        step_idx: int,
        refresh_every: int = 0,
        sim_threshold: float = 0.0,
        k: Optional[int] = None,
    ) -> bool:
        should_refresh = False
        if refresh_every > 0 and step_idx > 0 and step_idx % refresh_every == 0:
            should_refresh = True
        if not should_refresh and sim_threshold > 0.0 and new_task_emb is not None:
            new_emb = new_task_emb.detach().to("cpu").contiguous().numpy().astype("float32")
            ref_emb = self.init_emb_cpu.detach().contiguous().numpy().astype("float32")
            cosine = float((new_emb * ref_emb).sum()) / float(
                np.linalg.norm(new_emb) * np.linalg.norm(ref_emb) + 1e-12
            )
            should_refresh = cosine < float(sim_threshold)

        if not should_refresh:
            return False

        query_emb = new_task_emb if new_task_emb is not None else self.init_emb_cpu.to(self.provider.device)
        self.__init__(
            provider=self.provider,
            init_task_emb=query_emb,
            k=int(k) if k is not None else max(1, len(self.blocks_all)),
            H=self.H,
            progress=0.0,
        )
        return True
