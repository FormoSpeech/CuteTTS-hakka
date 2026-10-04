# Copyright 2026 FormoSpeech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Batched autoregressive inference.

`naive_ar_infer` decodes one utterance at a time. This runs B utterances
together through the same per-patch loop:

- prefixes of different lengths are left-padded and masked, with per-row
  position ids, so every row's last prefix token sits in the final column;
- each decode step appends one masked-in column per row;
- LM-level CFG runs the B unconditional prefixes as their own batch, and the
  diffusion head samples all 2B (or B) rows in one call, laid out
  [conditional rows; unconditional rows] as the head expects;
- each row stops on its own stop prediction. Finished rows keep stepping
  (their attention is row-local, so they cannot affect the others) until
  every row has stopped, and are truncated afterwards.

Batching changes the noise draws and float reduction orders, so a batched
row is not bit-identical to the same utterance decoded alone; it is a draw
from the same model.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from transformers.cache_utils import DynamicCache

from cutetts.inference.generation import (
    NaiveInferConfig,
    _acoustic_connector_dtype,
    _add_audio_dit_cond_input,
    _add_distilled_cfg_strength_input,
    _add_sampling_condition_cache_input,
    _add_speaker_embedding_input,
    _add_sway_sampling_input,
    _cfg_enabled,
    _head_condition_dtype,
    _latent_sequence_for_lm,
    _lm_cfg_enabled,
    _new_sampling_condition_cache,
    _require_offline_decode_compatible,
)
from cutetts.modeling.model import CuteTTSModel


def _left_pad(embeds: list[torch.Tensor], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """[1, T_i, D] prefixes -> left-padded [B, T, D] embeddings and [B, T] mask."""
    lengths = [int(e.size(1)) for e in embeds]
    width, dim = max(lengths), int(embeds[0].size(-1))
    out = embeds[0].new_zeros((len(embeds), width, dim), device=device)
    mask = torch.zeros((len(embeds), width), dtype=torch.long, device=device)
    for row, (embed, length) in enumerate(zip(embeds, lengths)):
        out[row, width - length :] = embed[0].to(device)
        mask[row, width - length :] = 1
    return out, mask


class _Branch:
    """One LM branch (conditional or unconditional) for B rows."""

    def __init__(self, lm: CuteTTSModel, prefixes: list[torch.Tensor]):
        self.lm = lm
        embeds, self.mask = _left_pad(prefixes, lm.device)
        positions = (self.mask.cumsum(-1) - 1).clamp(min=0)
        self.next_position = self.mask.sum(-1, keepdim=True)
        self.cache = DynamicCache()
        outputs = lm.forward_lm(
            inputs_embeds=embeds,
            attention_mask=self.mask,
            position_ids=positions,
            past_key_values=self.cache,
            use_cache=True,
            output_attentions=False,
        )
        self.last_hidden = outputs.last_hidden_state[:, -1, :]

    def step(self, input_embeds: torch.Tensor) -> None:
        ones = self.mask.new_ones((self.mask.size(0), 1))
        self.mask = torch.cat([self.mask, ones], dim=1)
        outputs = self.lm.forward_lm(
            inputs_embeds=input_embeds,
            attention_mask=self.mask,
            position_ids=self.next_position,
            past_key_values=self.cache,
            use_cache=True,
            output_attentions=False,
        )
        self.next_position = self.next_position + 1
        self.last_hidden = outputs.last_hidden_state[:, -1, :]


@torch.inference_mode()
def batched_ar_infer(
    config: NaiveInferConfig,
    lm: CuteTTSModel,
    prefix_embeds: list[torch.Tensor],
    uncond_prefix_embeds: list[torch.Tensor] | None,
    speaker_embedding: torch.Tensor | None = None,
    on_patch: Callable[[int, torch.Tensor, torch.Tensor], None] | None = None,
) -> list[torch.Tensor]:
    """Decode B utterances together.

    Args:
        config: Same settings `naive_ar_infer` takes (batch_lm_cfg_decode,
            static_lm_cache and compile_lm_decode are not supported here).
        prefix_embeds: B conditional prefixes, each [1, T_i, D].
        uncond_prefix_embeds: B unconditional prefixes for LM-level CFG.
        speaker_embedding: DiT speaker condition, [B, C] or, with CFG,
            [2B, C] laid out [conditional rows; unconditional rows].
        on_patch: Called after every step with (step, latents in VAE scale
            [B, patch, C], active [B] bool). Rows are active up to and
            including their final patch; streaming callers decode and emit
            only active rows.

    Returns:
        Per row, the generated latents in VAE scale, [N_i, patch, C].
    """
    if config.batch_lm_cfg_decode or config.static_lm_cache or config.compile_lm_decode:
        raise ValueError("batched_ar_infer uses a dynamic cache and eager LM decode.")
    use_lm_cfg = _lm_cfg_enabled(config)
    if use_lm_cfg and uncond_prefix_embeds is None:
        raise ValueError("LM-level CFG needs the unconditional prefixes.")
    _require_offline_decode_compatible(lm)

    batch = len(prefix_embeds)
    cond = _Branch(lm, prefix_embeds)
    uncond = _Branch(lm, uncond_prefix_embeds) if use_lm_cfg else None

    head_dtype = _head_condition_dtype(lm)
    connector_dtype = _acoustic_connector_dtype(lm)
    use_cfg = _cfg_enabled(config)
    patch_size = int(getattr(lm.config, "diff_dit_patch_size", 1))
    shape = (batch, patch_size, lm.config.acoustic_latent_dim) if patch_size != 1 else (
        batch,
        lm.config.acoustic_latent_dim,
    )
    previous = torch.zeros(shape, device=lm.device, dtype=head_dtype)
    sampling_condition_cache = _new_sampling_condition_cache(lm)

    finished_at = torch.full((batch,), -1, dtype=torch.long)
    latents: list[torch.Tensor] = []
    for step in range(config.max_decode_length):
        stops = (torch.argmax(lm.stop_predictor(cond.last_hidden), dim=1) == 1).cpu()
        if lm.config.two_class_stop_predictor:
            newly = stops & (finished_at < 0)
            finished_at[newly] = step

        head_input = dict(
            num_sampling_steps=config.diffusion_steps,
            cfg=config.cfg_strength if use_cfg else 0.0,
        )
        _add_sway_sampling_input(head_input, config)
        _add_distilled_cfg_strength_input(head_input, config)
        _add_sampling_condition_cache_input(head_input, sampling_condition_cache)
        _add_audio_dit_cond_input(
            head_input, previous, use_cfg, previous if use_lm_cfg else None
        )
        _add_speaker_embedding_input(head_input, speaker_embedding)
        hidden = cond.last_hidden.to(dtype=head_dtype)
        if use_lm_cfg:
            hidden = torch.cat([hidden, uncond.last_hidden.to(dtype=head_dtype)], dim=0)
        pred = lm.head.sample(hidden, **head_input)
        latents.append(pred / lm.speech_scaling_factor - lm.speech_bias_factor)
        if on_patch is not None:
            on_patch(step, latents[-1], (finished_at < 0) | (finished_at >= step))

        if bool((finished_at >= 0).all()):
            break
        previous = pred.to(dtype=head_dtype)
        input_embeds = lm.embed_acoustic_latents(
            _latent_sequence_for_lm(pred).to(dtype=connector_dtype)
        )
        cond.step(input_embeds)
        if uncond is not None:
            uncond.step(input_embeds)

    stacked = torch.stack(latents, dim=1)  # [B, steps, patch, C]
    lengths = [
        int(finished_at[row]) + 1 if finished_at[row] >= 0 else stacked.size(1)
        for row in range(batch)
    ]
    return [stacked[row, : lengths[row]] for row in range(batch)]
