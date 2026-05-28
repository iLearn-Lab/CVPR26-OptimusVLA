from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def torch_load_cpu(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def resolve_actions_path(raw_path: str, actions_root: Path | None, marker: str) -> Path:
    path = Path(raw_path)
    if path.exists():
        return path
    if actions_root is not None:
        text = raw_path.replace("\\", "/")
        candidates = []
        if marker:
            marker_with_sep = marker.rstrip("/") + "/"
            if marker_with_sep in text:
                rel = text.split(marker_with_sep, 1)[1]
                candidates.extend([actions_root / marker / rel, actions_root / rel])
        candidates.extend([actions_root / text.lstrip("/"), actions_root / path.name])
        for candidate in candidates:
            if candidate.exists():
                return candidate
    raise FileNotFoundError(f"Cannot resolve action trajectory: {raw_path}")


def load_actions(path: Path) -> np.ndarray:
    with np.load(path) as npz:
        actions = np.asarray(npz["actions"], dtype=np.float32)
    if actions.ndim != 2:
        raise ValueError(f"Expected actions with shape [T, A], got {actions.shape} from {path}")
    return actions


def sanitize_entry(entry: dict, action_id: str, length: int) -> dict:
    clean = {}
    for key, value in entry.items():
        lowered = str(key).lower()
        if lowered == "actions_path" or lowered.endswith("_path") or lowered == "path":
            continue
        if isinstance(value, str) and (value.startswith("/") or value.startswith("~")):
            continue
        clean[key] = value
    clean["action_id"] = action_id
    clean["length"] = int(length)
    chunk_meta = dict(clean.get("chunk_meta", {}))
    chunk_meta.setdefault("T", int(length))
    clean["chunk_meta"] = chunk_meta
    return clean


def main() -> None:
    parser = argparse.ArgumentParser(description="Pack per-trajectory memory actions into one NPZ file.")
    parser.add_argument("--input-meta", required=True, help="Legacy memory metadata file.")
    parser.add_argument("--output-meta", required=True, help="Sanitized metadata path to write.")
    parser.add_argument("--output-actions", required=True, help="Packed actions NPZ path to write.")
    parser.add_argument(
        "--actions-root",
        default="",
        help="Optional root used to resolve legacy absolute action paths.",
    )
    parser.add_argument(
        "--path-marker",
        default="",
        help="Path segment used when remapping legacy action paths.",
    )
    args = parser.parse_args()

    actions_root = Path(args.actions_root).resolve() if args.actions_root else None
    memory = torch_load_cpu(args.input_meta)
    if not isinstance(memory, list) or not memory:
        raise ValueError(f"Expected a non-empty list of metadata entries in {args.input_meta}")

    arrays = []
    offsets = [0]
    ids = []
    sanitized = []

    for index, entry in enumerate(memory):
        if "actions_path" not in entry:
            raise KeyError(f"Entry {index} does not contain actions_path")
        path = resolve_actions_path(str(entry["actions_path"]), actions_root, args.path_marker)
        actions = load_actions(path)
        action_id = f"{index:06d}"
        arrays.append(actions)
        offsets.append(offsets[-1] + actions.shape[0])
        ids.append(action_id)
        sanitized.append(sanitize_entry(entry, action_id, actions.shape[0]))

    flat_actions = np.concatenate(arrays, axis=0).astype(np.float32, copy=False)
    offsets_array = np.asarray(offsets, dtype=np.int64)
    ids_array = np.asarray(ids, dtype="U16")

    output_actions = Path(args.output_actions)
    output_actions.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_actions, actions=flat_actions, offsets=offsets_array, ids=ids_array)

    output_meta = Path(args.output_meta)
    output_meta.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sanitized, output_meta)

    print(f"packed_entries={len(sanitized)}")
    print(f"total_steps={flat_actions.shape[0]}")
    print(f"action_dim={flat_actions.shape[1]}")
    print(f"output_meta={output_meta}")
    print(f"output_actions={output_actions}")


if __name__ == "__main__":
    main()
