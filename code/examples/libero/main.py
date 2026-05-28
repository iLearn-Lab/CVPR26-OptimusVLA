from __future__ import annotations

import collections
import dataclasses
import fcntl
import json
import logging
import math
import os
import pathlib
import time
from typing import Any

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
LOG_TS_FMT = "%Y-%m-%d %H:%M:%S"


def _now_str() -> str:
    return time.strftime(LOG_TS_FMT, time.localtime())


def safe_append(log_file: str, record: dict[str, Any]) -> None:
    path = pathlib.Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False)
    with open(path, "a", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 10
    log_file: str = "./libero_results.jsonl"
    task_suite_name: str = "libero_goal"
    num_steps_wait: int = 10
    num_trials_per_task: int = 50
    seed: int = 7


def eval_libero(args: Args) -> None:
    np.random.seed(args.seed)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    max_steps = _max_steps_for_suite(args.task_suite_name)
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    safe_append(
        args.log_file,
        {
            "ts": _now_str(),
            "event": "process_start",
            "pid": os.getpid(),
            "task_suite": args.task_suite_name,
            "port": args.port,
            "seed": args.seed,
            "replan_steps": args.replan_steps,
        },
    )

    total_episodes = 0
    total_successes = 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        task_episodes = 0
        task_successes = 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info("Task: %s", task_description)
            env.reset()
            action_plan = collections.deque()
            obs = env.set_init_state(initial_states[episode_idx])
            t = 0
            done = False
            episode_reset_pending = True

            while t < max_steps + args.num_steps_wait:
                try:
                    if t < args.num_steps_wait:
                        obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_image = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    image = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(image, args.resize_size, args.resize_size)
                    )
                    wrist_image = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_image, args.resize_size, args.resize_size)
                    )

                    if not action_plan:
                        executed_steps = max(0, t - args.num_steps_wait)
                        progress = float(np.clip(executed_steps / max_steps, 0.0, 1.0))
                        element = {
                            "observation/image": image,
                            "observation/wrist_image": wrist_image,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "prompt": str(task_description),
                            "progress": progress,
                            "episode_reset": episode_reset_pending,
                        }
                        action_chunk = client.infer(element)["actions"]
                        episode_reset_pending = False
                        if len(action_chunk) < args.replan_steps:
                            raise RuntimeError(
                                f"Policy returned {len(action_chunk)} actions, "
                                f"but replan_steps is {args.replan_steps}."
                            )
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()
                    obs, _, done, _ = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1
                except Exception as exc:
                    logging.exception("Episode failed: %s", exc)
                    break

            task_episodes += 1
            total_episodes += 1
            logging.info("Success: %s", done)
            logging.info("Episodes: %d Successes: %d Rate: %.1f%%", total_episodes, total_successes, 100.0 * total_successes / total_episodes)

        logging.info("Task success rate: %.4f", float(task_successes) / float(task_episodes))
        logging.info("Current total success rate: %.4f", float(total_successes) / float(total_episodes))

    safe_append(
        args.log_file,
        {
            "ts": _now_str(),
            "event": "run_summary",
            "pid": os.getpid(),
            "task_suite": args.task_suite_name,
            "port": args.port,
            "total_episodes": int(total_episodes),
            "total_successes": int(total_successes),
            "total_success_rate": float(total_successes / total_episodes),
            "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "unknown"),
        },
    )


def _max_steps_for_suite(task_suite_name: str) -> int:
    match task_suite_name:
        case "libero_spatial":
            return 220
        case "libero_object":
            return 280
        case "libero_goal":
            return 300
        case "libero_10":
            return 520
        case "libero_90":
            return 400
    raise ValueError(f"Unknown task suite: {task_suite_name}")


def _get_libero_env(task, resolution: int, seed: int):
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
