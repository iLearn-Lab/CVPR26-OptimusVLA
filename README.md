<div align="center">
<h2 align="center">
    <b>Global Prior Meets Local Consistency: Dual-Memory Augmented  
     <br />  Vision-Language-Action Model for Efficient Robotic Manipulation
   <br /> <font size=3>CVPR 2026 </font></b> 
</h2>
<div>
<a target="_blank" href="https://scholar.google.com/citations?user=TDBF2UoAAAAJ&hl=en&oi=ao">Zaijing&#160;Li</a><sup>1 2</sup>,
<a target="_blank" href="https://scholar.google.com/citations?user=rxaiRMUAAAAJ&hl=en">Bing&#160;Hu</a><sup>1</sup>,
<a target="_blank" href="https://scholar.google.com/citations?user=9Vc--XsAAAAJ&hl=en&oi=ao">Rui&#160;Shao</a><sup>1 3&#9993</sup>,
<a target="_blank" href="https://scholar.google.com/citations?user=Mpg0w3cAAAAJ&hl=en&oi=ao">Gongwei&#160;Chen</a><sup>1</sup>,
<a target="_blank" href="https://scholar.google.com/citations?hl=en&user=Awsue7sAAAAJ">Dongmei&#160;Jiang</a><sup>2&#9993</sup>,
    <br>
<a target="_blank" href="https://scholar.google.com/citations?hl=en&user=FCIpXqwAAAAJ">Pengwei&#160;Xie</a><sup>4</sup>,
<a target="_blank" href="https://scholar.google.com/citations?hl=en&user=FCJVUYgAAAAJ">Jianye&#160;HAO</a><sup>4</sup>,
 <a target="_blank" href="https://scholar.google.com/citations?hl=en&user=yywVMhUAAAAJ">Liqiang&#160;Nie</a><sup>1</sup>
</div>
<sup>1</sup>Harbin Institute of Technology, Shenzhen&#160&#160&#160</span>
<sup>2</sup>PengCheng Laboratory, Shenzhen&#160&#160&#160</span>
    <br>
<sup>3</sup>Shenzhen Loop Area Institute&#160&#160&#160</span>
<sup>4</sup>Huawei Noah's Ark Lab&#160&#160&#160</span>
<br />
<sup>&#9993&#160;</sup>Corresponding author&#160;&#160;</span>
<br/>
<div align="center">
    <a href="https://arxiv.org/abs/2602.20200" target="_blank">
    <img src="https://img.shields.io/badge/Paper-arXiv-deepgreen" alt="Paper arXiv"></a>
    <a href="https://cybertronagent.github.io/OptimusVLA.github.io/" target="_blank">
    <img src="https://img.shields.io/badge/Project-OptimusVLA-9cf" alt="Project Page"></a>
</div>
</div>



## :new: Updates
- [02/2026] :fire: OptimusVLA is accepted to **CVPR 2026**!
- [02/2026] :fire: [Project page](https://cybertronagent.github.io/OptimusVLA.github.io/) released.
- [02/2026] :fire: [Arxiv paper](https://arxiv.org/abs/2602.20200) released.




## :balloon: OptimusVLA Framework
Overview of OptimusVLA framework. Given a task and the current observation, the Vision–Language backbone first encodes the inputs into a multimodal representation. GPM then retrieves a task-level prior based on this representation, while LBM dynamically encodes the historical action sequence to produce a consistency constraint. Finally, the flow policy denoises the initialization with an adaptive NFEs schedule to generate the action chunk.
<img src="./assets/fig1.png" >

## :rocket: How to Run
**OptimusVLA is built on the [openpi](https://github.com/Physical-Intelligence/openpi) framework. Therefore, please first download and configure the openpi environment, and then download the pi_05 model weights**.

### Install openpi
1. Clone the official OpenPI repository

```bash
git clone --recurse-submodules git@github.com:Physical-Intelligence/openpi.git
cd openpi
git submodule update --init --recursive
```

2. Create the main OpenPI environment:

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

3. PyTorch Support
   
```bash
cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
```

4. Download pi05_libero model checkpoints and convert it to pytorch version

OptimusVLA does not release the pi0.5 policy checkpoint. Prepare a PyTorch
`pi05_libero` checkpoint yourself. The policy directory must contain:

```text
model.safetensors
assets/physical-intelligence/libero/norm_stats.json
```

If you start from a JAX OpenPI checkpoint, convert it with the upstream OpenPI
converter:

```bash
cd "${OPENPI_ROOT}"
uv run examples/convert_jax_model_to_pytorch.py \
  --checkpoint-dir /path/to/pi05_libero_jax_checkpoint \
  --config-name pi05_libero \
  --output-path /path/to/pi05_libero_pytorch
```

Set the policy path:

```bash
export POLICY_DIR=/path/to/pi05_libero_pytorch
```

5. Install the extra OptimusVLA inference dependency:

```bash
uv pip install faiss-cpu
uv pip install mamba-ssm
```

6. Create the LIBERO Client Environment
Create the LIBERO example environment using the official OpenPI instructions:

```bash
cd "${OPENPI_ROOT}"
uv venv --python 3.8 examples/libero/.venv
source examples/libero/.venv/bin/activate
uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu113 \
  --index-strategy=unsafe-best-match
uv pip install -e packages/openpi-client
uv pip install -e third_party/libero
deactivate
```

Use this environment only for the LIBERO client in
`examples/libero/main.py`. Run the policy server from the OpenPI server
environment created in step 2.

### Apply the OptimusVLA Code
Assume this package was cloned or extracted at `/path/to/optimusvla-release`.
Copy its contents into the OpenPI checkout:

```bash
export OPENPI_ROOT=/path/to/openpi
rsync -av \
  --exclude README.md \
  --exclude checkpoints \
  --exclude memory \
  /path/to/optimusvla-release/code/ "${OPENPI_ROOT}/"
```

### Download OptimusVLA Assets
Download the asset directories from the Hugging Face repository that hosts this
release. They must end up under the OpenPI root exactly as follows:

```text
${OPENPI_ROOT}/checkpoints/gpm_task_head.pt
${OPENPI_ROOT}/checkpoints/lcm.pt
${OPENPI_ROOT}/memory/gpm_memory_meta.pt
${OPENPI_ROOT}/memory/gpm_memory.index
${OPENPI_ROOT}/memory/gpm_memory_actions.npz
```

## Start the Policy Server Manually
With the default asset paths above, only the pi0.5 policy path and norm stats
path need to be provided explicitly:

```bash
cd "${OPENPI_ROOT}"
export OPENPI_TORCH_COMPILE=0
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py \
  --env LIBERO \
  --port 8000 \
  --use-memory \
  --action-norm-stats-path "${POLICY_DIR}/assets/physical-intelligence/libero/norm_stats.json" \
  --action-use-quantile-norm \
  --memory-top-k 8 \
  --memory-refresh-every 1 \
  --align-mode hybrid \
  --mixture-mode gaussian \
  --temperature 10.0 \
  --sigma-min 0.05 \
  --noise-min 0.20 \
  --noise-max 1.00 \
  --nfe-min 1 \
  --nfe-max 10 \
  --use-lcm \
  --lcm-scale 0.10 \
  policy:checkpoint \
  --policy.config pi05_libero \
  --policy.dir "${POLICY_DIR}"
```

The command above uses these defaults:

```text
--task-head-ckpt checkpoints/gpm_task_head.pt
--lcm-ckpt checkpoints/lcm.pt
--memory-meta-path memory/gpm_memory_meta.pt
--faiss-index-path memory/gpm_memory.index
--memory-actions-path memory/gpm_memory_actions.npz
```

### Run LIBERO Evaluation

```bash
cd "${OPENPI_ROOT}"
POLICY_DIR=/path/to/pi05_libero_pytorch bash scripts/run_libero_eval.sh
```
By default, the script runs `libero_spatial`, `libero_object`, `libero_goal`,
and `libero_10`. Logs are written to a timestamped directory under `logs/`,
and the final summary is saved as `results.txt` in that directory. The summary
contains the exit code, episode count, success count, and success rate for each
suite. 

Default model and server settings:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `POLICY_CONFIG` | `pi05_libero` | OpenPI policy config used for the base pi0.5 checkpoint. |
| `POLICY_DIR` | Required | Local PyTorch pi0.5 checkpoint directory. This release does not include it. |
| `ACTION_NORM_STATS_PATH` | `${POLICY_DIR}/assets/physical-intelligence/libero/norm_stats.json` | Action normalization stats used by GPM memory actions. |
| `HOST` / `PORT` | `127.0.0.1` / `8000` | Client connection target and server port. |
| `SERVER_CUDA_VISIBLE_DEVICES` | `${CUDA_VISIBLE_DEVICES}` or `0` | GPU used by the policy server. |
| `CLIENT_CUDA_VISIBLE_DEVICES` | `${CUDA_VISIBLE_DEVICES}` or `0` | GPU visible to the LIBERO clients. |
| `OPENPI_TORCH_COMPILE` | `0` | Disables Torch compile by default for easier first-run debugging. |

Default GPM settings:

| Parameter | Default | Meaning |
| --- | --- | --- |
| GPM memory | Enabled | The helper always passes `--use-memory`. |
| GPM assets | `checkpoints/gpm_task_head.pt`, `memory/gpm_memory_meta.pt`, `memory/gpm_memory.index`, `memory/gpm_memory_actions.npz` | Default task head, metadata, FAISS index, and packed action memory paths. |
| `--action-use-quantile-norm` | Enabled | Uses LIBERO action quantile statistics for memory action normalization. |
| `MEMORY_TOP_K` | `8` | Number of retrieved memory candidates per query. |
| `MEMORY_REFRESH_EVERY` | `1` | Refreshes memory retrieval every replan request. |
| `ALIGN_MODE` | `hybrid` | Time alignment mode for the retrieved memory trajectory. |
| `MIXTURE_MODE` | `gaussian` | Builds a Gaussian action prior from retrieved memories. |
| `TEMPERATURE` | `10.0` | Softmax temperature used when weighting retrieved memories. |
| `SIGMA_MIN` | `0.05` | Lower bound for the Gaussian prior standard deviation. |
| `NOISE_MIN` / `NOISE_MAX` | `0.20` / `1.00` | Noise schedule range used by the memory-guided sampler. |
| `NFE_MIN` / `NFE_MAX` | `1` / `10` | Sampling step range selected from memory confidence. |

Default LCM settings:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `USE_LCM` | `1` | Enables LCM refinement after GPM. Set `USE_LCM=0` for GPM-only inference. |
| `LCM_SCALE` | `0.10` | Strength of the LCM correction applied to the action chunk. |
| LCM checkpoint | `checkpoints/lcm.pt` | Default LCM checkpoint path. |
| LCM architecture fallback | hidden `256`, layers `1`, heads `4`, dropout `0.0`, `mamba_impl=auto` | Used only when these fields are absent from checkpoint metadata. |

Default evaluation and logging settings:

| Parameter | Default | Meaning |
| --- | --- | --- |
| Suites | `libero_spatial`, `libero_object`, `libero_goal`, `libero_10` | Four standard LIBERO suites run in parallel. |
| `NUM_TRIALS_PER_TASK` | `50` | Episodes evaluated per task. |
| `REPLAN_STEPS` | `10` | Number of actions executed before requesting a new chunk. |
| `NUM_STEPS_WAIT` | `10` | Initial dummy steps before policy control begins. |
| `SEED` | `7` | LIBERO environment seed. |
| `RESIZE_SIZE` | `224` | Image size sent to the policy client. |
| `MUJOCO_GL` | `egl` | Headless MuJoCo rendering backend. |
| `LOG_DIR` | `logs/libero_eval_<timestamp>` | Directory for server logs, client stdout logs, JSONL records, and `results.txt`. |
| `RESULTS_TXT` | `${LOG_DIR}/results.txt` | Final text summary with per-suite and overall success rates. |


## :smile_cat: Evaluation results on Real World
We evaluate OptimusVLA on Generalization Tasks and Long-horizon Tasks via GALAXEA R1 Lite robot.
<img src="./assets/fig2.png" >

## :hugs: Citation

If you find this work useful for your research, please kindly cite our paper:

```
@article{li2026optimusvla,
  title={Global Prior Meets Local Consistency: Dual-Memory Augmented Vision-Language-Action Model for Efficient Robotic Manipulation},
  author={Zaijing Li, Bing Hu, Rui Shao, Gongwei Chen, Dongmei Jiang, Pengwei Xie, Jianye Hao, Liqiang Nie},
  journal={arXiv preprint arXiv:2602.20200},
  year={2026}
}

