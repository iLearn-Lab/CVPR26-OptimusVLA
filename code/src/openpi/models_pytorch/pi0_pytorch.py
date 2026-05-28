import logging
import math
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812
import openpi.models.gemma as _gemma
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing
from openpi.task_head.memory_init import ActionMemorySession
import os


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


class PI0Pytorch(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pi05 = config.pi05

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True] if self.pi05 else [False, False],
            precision=config.dtype,
        )
        self._capture_prefix_tokens = False
        self._last_prefix_tokens = None
        
        self.action_in_proj = nn.Linear(32, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, 32)

        if self.pi05:
            self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
            self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        else:
            self.state_proj = nn.Linear(32, action_expert_config.width)
            self.action_time_mlp_in = nn.Linear(2 * action_expert_config.width, action_expert_config.width)
            self.action_time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        torch.set_float32_matmul_precision("high")
        compile_flag = os.environ.get("OPENPI_TORCH_COMPILE", "1").lower()
        if compile_flag not in {"0", "false", "no", "off"}:
            self.sample_actions = torch.compile(self.sample_actions, mode="max-autotune")
        else:
            logging.info("[PI0Pytorch] torch.compile disabled by OPENPI_TORCH_COMPILE=%s", compile_flag)

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

        msg = "transformers_replace is not installed correctly. Please install it with `uv pip install transformers==4.53.2` and `cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/`."
        try:
            from transformers.models.siglip import check

            if not check.check_whether_transformers_replace_is_installed_correctly():
                raise ValueError(msg)
        except ImportError:
            raise ValueError(msg) from None
    

    def enable_prefix_token_capture(self, flag: bool = True):
        """Enable or disable caching of VLM prefix tokens."""
        self._capture_prefix_tokens = bool(flag)
        if not flag:
            self._last_prefix_tokens = None

    @torch.no_grad()
    def get_last_prefix_tokens(self):
        """Return the latest cached VLM prefix tokens with shape [B, L, D]."""
        return self._last_prefix_tokens

    def _set_last_prefix_tokens(self, x: torch.Tensor | None):
        if x is None:
            self._last_prefix_tokens = None
            return
        if x.ndim == 2:
            x = x.unsqueeze(0)
        self._last_prefix_tokens = x.detach()

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True

        logging.info("Enabled gradient checkpointing for PI0Pytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False

        logging.info("Disabled gradient checkpointing for PI0Pytorch model")

    def is_gradient_checkpointing_enabled(self):
        """Check if gradient checkpointing is enabled."""
        return self.gradient_checkpointing_enabled

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

    def _preprocess_observation(self, observation, *, train=True):
        """Helper method to preprocess observation."""
        observation = _preprocessing.preprocess_observation_pytorch(observation, train=train)
        return (
            list(observation.images.values()),
            list(observation.image_masks.values()),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
        )

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for PaliGemma transformer processing.
        """
        embs = []
        pad_masks = []
        att_masks = []

        # Process images
        for img, img_mask in zip(images, img_masks, strict=True):

            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)

            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))

            # Create attention masks so that image tokens attend to each other
            att_masks += [0] * num_img_embs

        # Process language tokens
        def lang_embed_func(lang_tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # full attention between image and language inputs
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        # Get batch size from the first dimension of the concatenated tensors
        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_suffix(self, state, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        if not self.pi05:
            if self.state_proj.weight.dtype == torch.float32:
                state = state.to(torch.float32)

            # Embed state
            def state_proj_func(state):
                return self.state_proj(state)

            state_emb = self._apply_checkpoint(state_proj_func, state)

            embs.append(state_emb[:, None, :])
            bsize = state_emb.shape[0]
            device = state_emb.device

            state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
            pad_masks.append(state_mask)

            # Set attention masks so that image and language inputs do not attend to state or actions
            att_masks += [1]

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0, device=timestep.device
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Fuse timestep + action information using an MLP
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        if not self.pi05:
            time_emb = time_emb[:, None, :].expand_as(action_emb)
            action_time_emb = torch.cat([action_emb, time_emb], dim=2)

            # Apply MLP layers
            def mlp_func(action_time_emb):
                x = self.action_time_mlp_in(action_time_emb)
                x = F.silu(x)  # swish == silu
                return self.action_time_mlp_out(x)

            action_time_emb = self._apply_checkpoint(mlp_func, action_time_emb)
            adarms_cond = None
        else:
            # time MLP (for adaRMS)
            def time_mlp_func(time_emb):
                x = self.time_mlp_in(time_emb)
                x = F.silu(x)  # swish == silu
                x = self.time_mlp_out(x)
                return F.silu(x)

            time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
            action_time_emb = action_emb
            adarms_cond = time_emb

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.action_horizon - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def forward(self, observation, actions, noise=None, time=None) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=True)

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        # Prepare attention masks
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        # Apply gradient checkpointing if enabled
        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )

        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        # Apply gradient checkpointing to final action projection if enabled
        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)

        return F.mse_loss(u_t, v_t, reduction="none")

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = observation.state.shape[0]

        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"

        outputs_embeds, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        dbg = bool(getattr(self, "debug_memory", False))
        use_mem = bool(getattr(self, "use_memory", True))
        if not use_mem:
            if noise is None:
                H = self.config.action_horizon
                A = self.config.action_dim
                actions_shape = (bsize, H, A)
                noise = self.sample_noise(actions_shape, device)
            dt = -1.0 / num_steps
            dt = torch.tensor(dt, dtype=torch.float32, device=device)
            x_t = noise
            time = torch.tensor(1.0, dtype=torch.float32, device=device)
            while time >= -dt / 2:
                expanded_time = time.expand(bsize)
                v_t = self.denoise_step(state, prefix_pad_masks, past_key_values, x_t, expanded_time)
                x_t = x_t + dt * v_t
                time += dt
            return x_t

        prefix_for_head = outputs_embeds[0] if isinstance(outputs_embeds, (list, tuple)) else outputs_embeds
        if getattr(self, "_capture_prefix_tokens", False):
            self._set_last_prefix_tokens(prefix_for_head)

        task_emb = None
        if prefix_for_head is not None and getattr(self, "task_head", None) is not None:
            toks = prefix_for_head
            if toks.ndim == 2:
                toks = toks.unsqueeze(0)
            pooled = toks.mean(dim=1).to(dtype=torch.float32, device=device)
            task_emb = self.task_head(pooled).squeeze(0)
            task_emb = F.normalize(task_emb, dim=-1)

        if getattr(self, "memory_provider", None) is not None and task_emb is not None:
            memory_top_k = int(getattr(self, "memory_top_k", 8))
            if getattr(self, "memory_session", None) is None:
                if dbg:
                    logging.info("Starting a new GPM memory session.")
                self.memory_session = ActionMemorySession(
                    provider=self.memory_provider,
                    init_task_emb=task_emb,
                    k=memory_top_k,
                    H=self.config.action_horizon,
                    progress=0.0,
                )
                self._sample_call_count = 0
                if dbg:
                    logging.info(
                        "GPM memory session created: k=%d horizon=%d memory_size=%d",
                        memory_top_k,
                        int(self.config.action_horizon),
                        int(getattr(self.memory_provider, "num_items", -1)),
                    )
            else:
                step_idx = getattr(self, "_sample_call_count", 0)
                did = self.memory_session.maybe_refresh(
                    new_task_emb=task_emb,
                    step_idx=step_idx,
                    refresh_every=int(getattr(self, "memory_refresh_every", 1)),
                    sim_threshold=float(getattr(self, "memory_refresh_sim_threshold", 0.0)),
                    k=memory_top_k,
                )
                if dbg:
                    logging.info("GPM memory refresh: refreshed=%s step_idx=%d", bool(did), step_idx)

        memory_prior_used = False
        lcm_prior_chunk = None
        if noise is None:
            H = self.config.action_horizon
            A = self.config.action_dim
            if getattr(self, "memory_session", None) is not None:
                if getattr(self, "progress_mode", "client") == "memory" and getattr(self, "memory_session", None) is not None:
                    step_idx = getattr(self, "_sample_call_count", 0)
                    replan_hint = int(getattr(self, "replan_steps_hint", self.config.action_horizon))
                    progress_used = float(self.memory_session.estimate_progress(step_idx=step_idx, replan_steps=replan_hint))
                    src = "memory"
                else:
                    progress_used = float(getattr(self, "_external_progress", 0.0))
                    src = "client"

                X_init, debug_info = self.memory_session.sample_chunk(progress_used, return_debug=True)
                noise = X_init[None, :, :].expand(bsize, H, A).contiguous()
                try:
                    lcm_mu = self.memory_session.prior_mean(progress_used)
                    lcm_prior_chunk = lcm_mu[None, :, :].expand(bsize, H, A).contiguous()
                except Exception:
                    lcm_prior_chunk = noise
                memory_prior_used = True
                num_steps = self.memory_session.nfe_adapt
                self._sample_call_count = getattr(self, "_sample_call_count", 0) + 1
                if dbg:
                    logging.info(
                        "GPM prior: progress=%.3f source=%s nfe=%d k=%d win=[%d,%d) "
                        "top_sims=%s prior_mean=%.4f prior_std=%.4f noise_sigma=%.4f",
                        progress_used,
                        src,
                        int(num_steps),
                        debug_info.get("k", -1),
                        debug_info.get("win_start", -1),
                        debug_info.get("win_end", -1),
                        debug_info.get("top3_sims", []),
                        debug_info.get("prior_mean", float("nan")),
                        debug_info.get("prior_std", float("nan")),
                        debug_info.get("noise_sigma", float("nan")),
                    )
            else:
                actions_shape = (bsize, H, A)
                noise = self.sample_noise(actions_shape, device)
                if dbg:
                    reason = []
                    if getattr(self, "memory_provider", None) is None:
                        reason.append("no_provider")
                    if task_emb is None:
                        reason.append("no_task_emb")
                    if (
                        getattr(self, "memory_provider", None) is not None
                        and task_emb is not None
                        and getattr(self, "memory_session", None) is None
                    ):
                        reason.append("session_create_failed")
                    logging.info(
                        "Using Gaussian noise fallback: H=%d A=%d reason=%s",
                        H,
                        A,
                        "+".join(reason) or "unknown",
                    )

        use_lcm = (
            bool(getattr(self, "use_lcm", False))
            and getattr(self, "lcm", None) is not None
            and memory_prior_used
            and noise is not None
        )
        prev_lcm = getattr(self, "_lcm_prev_chunk", None)
        if use_lcm and prev_lcm is not None:
            prev_lcm = prev_lcm.to(device=device, dtype=torch.float32)
            if prev_lcm.ndim == 2:
                prev_lcm = prev_lcm.unsqueeze(0)
            if prev_lcm.shape[0] != bsize:
                if prev_lcm.shape[0] == 1:
                    prev_lcm = prev_lcm.expand(bsize, -1, -1).contiguous()
                else:
                    prev_lcm = None
            if prev_lcm is not None and prev_lcm.shape[1] != self.config.action_horizon:
                prev_lcm = F.interpolate(
                    prev_lcm.permute(0, 2, 1),
                    size=self.config.action_horizon,
                    mode="linear",
                    align_corners=False,
                ).permute(0, 2, 1)

            if prev_lcm is not None and prev_lcm.shape[-1] == self.config.action_dim:
                h_lcm = getattr(self, "_lcm_h", None)
                if h_lcm is not None:
                    h_lcm = h_lcm.to(device=device)
                context_lcm = lcm_prior_chunk if lcm_prior_chunk is not None else noise
                context_lcm = context_lcm.to(device=device, dtype=torch.float32)
                with torch.no_grad():
                    lcm_bias, h_new = self.lcm(prev_lcm, context_lcm, h_lcm)
                scale = float(getattr(self, "lcm_scale", 1.0))
                noise = noise + scale * lcm_bias.to(device=device, dtype=noise.dtype)
                self._lcm_h = h_new.detach() if h_new is not None else None
                if getattr(self, "debug_lcm", False):
                    logging.info(
                        "[LCM] applied action bias | scale=%.3f | norm(raw)=%.5f | norm(scaled)=%.5f",
                        scale,
                        float(lcm_bias.norm()),
                        float((scale * lcm_bias).norm()),
                    )
        elif getattr(self, "debug_lcm", False):
            reason = "no-prev" if prev_lcm is None else ("no-memory-prior" if not memory_prior_used else "disabled")
            logging.info("[LCM] skipped: %s", reason)

        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
            )
            x_t = x_t + dt * v_t
            time += dt

        self._lcm_prev_chunk = x_t.detach()
        return x_t

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        # Prepare attention masks
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)
