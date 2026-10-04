# Copyright 2026 OPPO and Fudan University
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

"""Public Python inference API."""

from __future__ import annotations

import hashlib
import math
import queue
import random
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from cutetts.inference.conditioning import (
    build_guidance_plan,
    build_prefix_segment,
    dit_speaker_for_plan,
    initial_previous_from_prefix,
    lm_speaker_for_branch,
)
from cutetts.hakka import dialect_clause
from cutetts.inference.batch_generation import batched_ar_infer
from cutetts.inference.generation import (
    NaiveInferConfig,
    _normalize_decoded_waveforms,
    _offline_decode_generated_latents,
    _vae_decode_autocast,
    naive_ar_infer,
)
from cutetts.modeling.sampling import set_sampler_compile_mode
from cutetts.runtime import RuntimeBundle, load_runtime, prepare_reference_audio


@dataclass(frozen=True)
class GenerationResult:
    waveform: torch.Tensor
    sample_rate: int


@dataclass(frozen=True)
class AudioChunk:
    """One decoded mono PCM chunk produced during streaming generation.

    ``waveform`` is a contiguous CPU float32 tensor with shape ``[1, samples]``.
    """

    waveform: torch.Tensor
    sample_rate: int


@dataclass(frozen=True)
class VoicePrompt:
    """A reference voice encoded once, for reuse across requests.

    Build it with ``CuteTTS.create_voice_prompt`` and pass it as
    ``reference_audio``; generation then skips decoding the file and running
    the VAE encoder and speaker encoder on it. ``reference_features`` are the
    VAE latents of the (cropped) reference, ``[frames, latent_dim]``;
    ``speaker_embedding`` is the speaker encoder output, ``[1, dim]``. Both
    come from components shared by every CuteTTS checkpoint.

    ``reference_embeds`` caches the language model's embedding of the
    reference (its local encoder output, ``[patches, hidden]``), the costliest
    per-request step once the reference is encoded. It belongs to the model
    that built it, identified by ``embeds_key``; a model with different local
    encoder weights ignores it and embeds ``reference_features`` itself.
    """

    reference_features: torch.Tensor
    speaker_embedding: torch.Tensor
    reference_embeds: torch.Tensor | None = None
    embeds_key: str | None = None
    # Per (model, dialect): the embedded prompt up to and including the
    # reference, which every request with this voice and dialect shares.
    # Kept in memory only.
    _prefix_cache: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    def save(self, path: str | Path) -> None:
        from safetensors.torch import save_file

        tensors = {
            "reference_features": self.reference_features,
            "speaker_embedding": self.speaker_embedding,
        }
        metadata = None
        if self.reference_embeds is not None and self.embeds_key is not None:
            tensors["reference_embeds"] = self.reference_embeds
            metadata = {"embeds_key": self.embeds_key}
        save_file(
            {name: tensor.detach().cpu().contiguous() for name, tensor in tensors.items()},
            str(path),
            metadata=metadata,
        )

    @classmethod
    def load(cls, path: str | Path) -> VoicePrompt:
        from safetensors import safe_open

        with safe_open(str(path), framework="pt") as handle:
            tensors = {name: handle.get_tensor(name) for name in handle.keys()}
            metadata = handle.metadata() or {}
        return cls(
            tensors["reference_features"],
            tensors["speaker_embedding"],
            tensors.get("reference_embeds"),
            metadata.get("embeds_key"),
        )


@dataclass(frozen=True)
class _StreamFailure:
    error: BaseException


class _StreamCancelled(Exception):
    pass


_STREAM_DONE = object()
# Static KV-cache capacity with compile_lm: prefix (instruction, reference
# speech, text) plus max_decode_length must fit, or that utterance gets its
# own larger cache and a one-off recompilation.
LM_CACHE_CAPACITY = 2048


def _resolve_model_dir(model_dir: str | Path) -> Path:
    """A local model directory as is; otherwise a Hugging Face repo id
    (e.g. "formospeech/cutetts-hakka-community-1"), downloaded once into the
    Hub cache. Gated repos need a logged-in account with access."""
    path = Path(model_dir).expanduser()
    if path.exists() or str(model_dir).count("/") != 1:
        return path
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(str(model_dir)))


class CuteTTS:
    """Load one CuteTTS model directory and synthesize individual utterances."""

    def __init__(self, runtime: RuntimeBundle, compile_lm: bool = False):
        self.runtime = runtime
        self.compile_lm = bool(compile_lm)
        self._cached_embeds_key: str | None = None
        self._model_token = object()  # identifies this model in VoicePrompt._prefix_cache
        self._uncond_embeds: torch.Tensor | None = None
        self._uncond_prefix_cache: dict = {}  # prefilled self._uncond_embeds, see naive_ar_infer

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | Path,
        *,
        device: str | torch.device = "auto",
        compile_lm: bool = False,
        cuda_graph_decode: bool = True,
    ) -> "CuteTTS":
        """compile_lm: CUDA-graph the language model's decode step (static KV
        cache + torch.compile). Several times faster per generated patch, but
        the first generate() call pays a one-time compilation, so it suits
        long-running use (servers, batch synthesis) rather than one-shot CLI
        calls. CUDA only.

        cuda_graph_decode: on CUDA, replay the streaming VAE decoder as a CUDA
        graph, captured once per batch size. It computes the same audio and
        cuts per-chunk decoding from ~8 ms to ~2 ms on an RTX A5000; turn it
        off to save the graphs' GPU memory."""
        runtime = load_runtime(_resolve_model_dir(model_dir), device)
        runtime.processor.acoustic_vae.cuda_graphs = bool(cuda_graph_decode) and (
            runtime.processor.device.type == "cuda"
        )
        set_sampler_compile_mode(
            "eager" if runtime.model.device.type == "mps" else "full-sampler"
        )
        if compile_lm and runtime.model.device.type != "cuda":
            raise ValueError("compile_lm requires a CUDA device.")
        return cls(runtime, compile_lm=compile_lm)

    @property
    def variant(self) -> str:
        return self.runtime.variant

    @property
    def sample_rate(self) -> int:
        return self.runtime.sample_rate

    @staticmethod
    def _seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _prepare_branch(model, prefix, speaker_embedding, speech_embeds=None):
        prefix = prefix.to(model.device)
        embeds, _contains_speech, speech_features = model.prepare_input_embeds(
            prefix,
            lm_speaker_embedding=speaker_embedding,
            speech_embeds=speech_embeds,
        )
        return embeds, speech_features

    @torch.inference_mode()
    def generate(
        self,
        text: str,
        *,
        mode: str = "tts",
        reference_audio: str | Path | VoicePrompt | None = None,
        dialect: str | None = None,
        cfg_strength: float = 2.0,
        diffusion_steps: int | None = None,
        diffusion_sway_coefficient: float | None = None,
        max_decode_length: int = 750,
        seed: int = 42,
        show_progress: bool = True,
        pcm_chunk_callback: Callable[[torch.Tensor], None] | None = None,
    ) -> GenerationResult:
        text = str(text).strip()
        if not text:
            raise ValueError("text must not be empty.")
        steps, sway, cfg_mode, ordinary_cfg, distilled_cfg = self._settings(
            mode,
            reference_audio,
            dialect,
            cfg_strength,
            diffusion_steps,
            diffusion_sway_coefficient,
            max_decode_length,
        )

        self._seed(int(seed))
        processor = self.runtime.processor
        model = self.runtime.model
        plan = build_guidance_plan(mode, cfg_mode, ordinary_cfg)
        cond_embeds, cond_speech, uncond_embeds, initial_uncond, speaker_embedding = (
            self._branches(text, mode, reference_audio, dialect, plan)
        )

        infer_config = NaiveInferConfig(
            diffusion_steps=steps,
            cfg_strength=ordinary_cfg,
            cfg_mode=cfg_mode,
            max_decode_length=int(max_decode_length),
            diffusion_sway_coefficient=sway,
            distilled_cfg_strength=distilled_cfg,
            static_lm_cache=self.compile_lm,
            compile_lm_decode=self.compile_lm,
            lm_cache_capacity=LM_CACHE_CAPACITY if self.compile_lm else None,
        )
        result = naive_ar_infer(
            infer_config,
            processor,
            model,
            cond_embeds,
            uncond_embeds,
            speaker_embedding=dit_speaker_for_plan(plan, speaker_embedding),
            uncond_prefix_cache=self._uncond_prefix_cache,
            initial_previous_cond=initial_previous_from_prefix(
                cond_speech,
                plan.conditional.include_prompt,
            ),
            initial_uncond_previous_cond=initial_uncond,
            separate_cfg_previous_cond=plan.uses_lm_cfg,
            uncond_previous_cond_always_zero=plan.uncond_history_always_zero,
            use_tqdm=bool(show_progress),
            decode_each_patch=mode == "voice_clone" or pcm_chunk_callback is not None,
            pcm_chunk_callback=pcm_chunk_callback,
        )
        return GenerationResult(
            waveform=result.waveforms.detach().cpu(),
            sample_rate=self.runtime.sample_rate,
        )

    @torch.inference_mode()
    def generate_batch(
        self,
        texts: list[str],
        *,
        mode: str = "tts",
        reference_audio: str | Path | VoicePrompt | list[str | Path | VoicePrompt] | None = None,
        dialect: str | list[str | None] | None = None,
        cfg_strength: float = 2.0,
        diffusion_steps: int | None = None,
        diffusion_sway_coefficient: float | None = None,
        max_decode_length: int = 750,
        seed: int = 42,
        batch_size: int | None = None,
        pcm_chunk_callback: Callable[[int, torch.Tensor], None] | None = None,
    ) -> list[GenerationResult]:
        """Synthesize several utterances, `batch_size` (default: all) at a time.

        `reference_audio` and `dialect` take one value for every text or one
        per text. Each batch decodes its rows together (see
        cutetts.inference.batch_generation), so a row is a valid sample from
        the model but not bit-identical to generate() on the same text and
        seed. compile_lm does not apply to batched decoding.

        With `pcm_chunk_callback(index, chunk)`, audio is decoded patch by
        patch with the streaming VAE decoder (one batched call per step) and
        each text's [1, samples] chunks are passed on as they are decoded,
        `index` being its position in `texts`. Without it, each text is
        decoded once at the end, which is faster (per-step decoding cost
        17-25% of throughput on an RTX A5000).
        """
        texts = [str(text).strip() for text in texts]
        if not texts or not all(texts):
            raise ValueError("texts must be a non-empty list of non-empty strings.")
        count = len(texts)
        references = reference_audio if isinstance(reference_audio, list) else [reference_audio] * count
        dialects = dialect if isinstance(dialect, list) else [dialect] * count
        if len(references) != count or len(dialects) != count:
            raise ValueError("reference_audio and dialect lists must match texts in length.")
        settings = [
            self._settings(
                mode, ref, dia, cfg_strength, diffusion_steps,
                diffusion_sway_coefficient, max_decode_length,
            )
            for ref, dia in zip(references, dialects)
        ]
        steps, sway, cfg_mode, ordinary_cfg, distilled_cfg = settings[0]
        plan = build_guidance_plan(mode, cfg_mode, ordinary_cfg)
        infer_config = NaiveInferConfig(
            diffusion_steps=steps,
            cfg_strength=ordinary_cfg,
            cfg_mode=cfg_mode,
            max_decode_length=int(max_decode_length),
            diffusion_sway_coefficient=sway,
            distilled_cfg_strength=distilled_cfg,
        )
        processor = self.runtime.processor
        model = self.runtime.model
        size = count if batch_size is None else max(1, int(batch_size))
        self._seed(int(seed))
        results: list[GenerationResult] = []
        for start in range(0, count, size):
            rows = range(start, min(start + size, count))
            branches = [self._branches(texts[i], mode, references[i], dialects[i], plan) for i in rows]
            speakers = [b[4] for b in branches]
            speaker = None if speakers[0] is None else torch.cat(speakers, dim=0)
            if pcm_chunk_callback is None:
                latents = batched_ar_infer(
                    infer_config,
                    model,
                    [b[0] for b in branches],
                    [b[2] for b in branches] if plan.uses_lm_cfg else None,
                    speaker_embedding=dit_speaker_for_plan(plan, speaker),
                    uncond_prefix_cache=self._uncond_prefix_cache,
                )
                for row_latents in latents:
                    frames = row_latents.reshape(1, -1, row_latents.size(-1))
                    waveform = _offline_decode_generated_latents(processor, model, [frames])
                    results.append(
                        GenerationResult(
                            waveform=waveform.detach().cpu(), sample_rate=self.runtime.sample_rate
                        )
                    )
                continue

            pieces: list[list[torch.Tensor]] = [[] for _ in rows]

            def on_patch(step, latent, active, decoder=None, rows=rows, pieces=pieces):
                with _vae_decode_autocast(processor, processor.device):
                    waveforms = decoder.decode_chunk(latent.to(processor.device))
                waveforms = _normalize_decoded_waveforms(waveforms).detach().float().cpu()
                for row, index in enumerate(rows):
                    if bool(active[row]):
                        chunk = waveforms[row : row + 1].contiguous()
                        pieces[row].append(chunk)
                        pcm_chunk_callback(index, chunk)

            with processor.acoustic_vae.streaming_decode() as decoder:
                batched_ar_infer(
                    infer_config,
                    model,
                    [b[0] for b in branches],
                    [b[2] for b in branches] if plan.uses_lm_cfg else None,
                    speaker_embedding=dit_speaker_for_plan(plan, speaker),
                    uncond_prefix_cache=self._uncond_prefix_cache,
                    on_patch=lambda step, latent, active: on_patch(step, latent, active, decoder=decoder),
                )
            for row_pieces in pieces:
                results.append(
                    GenerationResult(
                        waveform=torch.cat(row_pieces, dim=-1), sample_rate=self.runtime.sample_rate
                    )
                )
        return results

    def generate_batch_stream(self, texts: list[str], **kwargs) -> Iterator[tuple[int, AudioChunk]]:
        """Yield (index, AudioChunk) as each text's audio is decoded, `index`
        being its position in `texts`; chunks of different texts interleave.
        Takes generate_batch's arguments. Generation runs in a worker thread;
        closing the iterator stops it at the next decoded patch. Use only one
        active generation per CuteTTS instance."""
        messages: queue.SimpleQueue = queue.SimpleQueue()
        cancelled = threading.Event()

        def emit(index: int, chunk: torch.Tensor) -> None:
            if cancelled.is_set():
                raise _StreamCancelled()
            messages.put((index, AudioChunk(waveform=chunk, sample_rate=self.runtime.sample_rate)))

        def run() -> None:
            try:
                self.generate_batch(texts, pcm_chunk_callback=emit, **kwargs)
            except _StreamCancelled:
                pass
            except BaseException as error:
                messages.put(_StreamFailure(error))
            finally:
                messages.put(_STREAM_DONE)

        worker = threading.Thread(target=run, name="cutetts-batch-stream", daemon=True)
        worker.start()
        try:
            while True:
                message = messages.get()
                if message is _STREAM_DONE:
                    break
                if isinstance(message, _StreamFailure):
                    raise message.error
                yield message
        finally:
            cancelled.set()
            worker.join()

    @torch.inference_mode()
    def create_voice_prompt(self, reference_audio: str | Path) -> VoicePrompt:
        """Encode a reference recording once; pass the result as
        `reference_audio` to any generate method to skip re-encoding it."""
        processor = self.runtime.processor
        model = self.runtime.model
        reference_wave, speaker_wave = prepare_reference_audio(
            reference_audio,
            self.runtime.sample_rate,
            int(self.runtime.speaker_encoder.sample_rate),
        )
        speaker_device = next(self.runtime.speaker_encoder.parameters()).device
        cuda_devices = {
            device.index if device.index is not None else torch.cuda.current_device()
            for device in (processor.device, speaker_device)
            if device.type == "cuda"
        }
        # The VAE posterior draws an (unused) random std; keep that off the
        # generation RNG so a precomputed prompt and a path give the same audio.
        with torch.random.fork_rng(devices=sorted(cuda_devices)), torch.autocast(
            device_type=self.runtime.model.device.type, enabled=False
        ):
            [[reference_features]] = processor.acoustic_batch_extractor(
                [[reference_wave.to(processor.device)]],
                processor.acoustic_feature_forward,
            )
            speaker_output = self.runtime.speaker_encoder(
                speaker_wave.to(speaker_device),
                int(self.runtime.speaker_encoder.sample_rate),
            )
        manager = processor.segment_manager
        segment = manager.fuse_segments([manager.create_speech_segment(reference_features[None, ...])])[0]
        segment = segment.to(model.device)
        embed_dtype = model.get_input_embeddings().weight.dtype
        _, embeds = model.forward_speech_features(
            segment.speech_tensor.to(embed_dtype), segment.speech_pad_mask
        )
        return VoicePrompt(
            reference_features,
            speaker_output["embedding"].float(),
            embeds[segment.speech_pad_mask].to(embed_dtype),
            self._embeds_key(),
        )

    def _voice_embeds(self, prompt: VoicePrompt, dialect: str | None, plan, speaker_embedding):
        """Embedded voice-clone prompt up to and including the reference,
        cached on the prompt for this model and dialect."""
        key = (self._model_token, dialect)
        embeds = prompt._prefix_cache.get(key)
        if embeds is None:
            processor = self.runtime.processor
            model = self.runtime.model
            manager = processor.segment_manager
            reference = manager.create_speech_segment(
                prompt.reference_features.to(processor.device)[None, ...]
            )
            segment = manager.fuse_segments(
                processor._reference_voice_segments(reference, dialect_clause(dialect))
            )[0]
            reference_embeds = None
            if prompt.reference_embeds is not None and prompt.embeds_key == self._embeds_key():
                reference_embeds = prompt.reference_embeds.to(model.device)
            embeds, _ = self._prepare_branch(
                model,
                segment,
                lm_speaker_for_branch(plan.conditional, speaker_embedding),
                reference_embeds,
            )
            prompt._prefix_cache[key] = embeds
        return embeds

    def _embeds_key(self) -> str:
        """Fingerprint of the weights that turn reference latents into LM
        embeddings, so a VoicePrompt's cached embedding is reused only by a
        model that would compute the same one."""
        if self._cached_embeds_key is None:
            model = self.runtime.model
            digest = hashlib.sha256()
            tensors = {
                f"locenc.{name}": value for name, value in model.locenc.state_dict().items()
            }
            tensors |= {
                f"locenc_to_lm_proj.{name}": value
                for name, value in model.locenc_to_lm_proj.state_dict().items()
            }
            tensors["speech_scaling_factor"] = model.speech_scaling_factor
            tensors["speech_bias_factor"] = model.speech_bias_factor
            for name in sorted(tensors):
                value = tensors[name].detach().cpu().contiguous()
                digest.update(f"{name}:{value.dtype}:{tuple(value.shape)};".encode())
                digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
            self._cached_embeds_key = digest.hexdigest()[:16]
        return self._cached_embeds_key

    def _settings(
        self,
        mode: str,
        reference_audio: str | Path | VoicePrompt | None,
        dialect: str | None,
        cfg_strength: float,
        diffusion_steps: int | None,
        diffusion_sway_coefficient: float | None,
        max_decode_length: int,
    ) -> tuple[int, float, str, float, float | None]:
        """Validate the request; (steps, sway, cfg_mode, ordinary_cfg, distilled_cfg)."""
        if mode not in {"tts", "voice_clone"}:
            raise ValueError("mode must be 'tts' or 'voice_clone'.")
        if mode == "voice_clone" and reference_audio is None:
            raise ValueError("voice_clone requires reference_audio.")
        if dialect is not None and mode != "voice_clone":
            raise ValueError("dialect is only supported with mode='voice_clone'.")
        dialect_clause(dialect)  # validate before any work
        if not math.isfinite(cfg_strength) or cfg_strength < 0.0:
            raise ValueError("cfg_strength must be a finite non-negative number.")
        if max_decode_length <= 0:
            raise ValueError("max_decode_length must be positive.")

        if self.variant == "base":
            defaults = self.runtime.generation_defaults
            steps = int(defaults.get("diffusion_steps", 10)) if diffusion_steps is None else int(diffusion_steps)
            sway = float(defaults.get("diffusion_sway_coefficient", -0.8)) if (
                diffusion_sway_coefficient is None
            ) else float(diffusion_sway_coefficient)
            if steps <= 0:
                raise ValueError("diffusion_steps must be positive for CuteTTS.")
            if not -1.0 <= sway <= 2.0 / (math.pi - 2.0):
                raise ValueError("diffusion_sway_coefficient is outside its valid domain.")
            cfg_mode = "lm"
            ordinary_cfg = cfg_strength
            distilled_cfg = None
        else:
            steps = 4 if diffusion_steps is None else int(diffusion_steps)
            if steps not in {1, 2, 4}:
                raise ValueError("CuteTTS-distill supports diffusion_steps 1, 2, or 4.")
            if diffusion_sway_coefficient not in {None, 0, 0.0}:
                raise ValueError("CuteTTS-distill does not expose sway sampling.")
            if cfg_strength > 5.0:
                raise ValueError("CuteTTS-distill cfg_strength must be in [0, 5].")
            sway = 0.0
            cfg_mode = "nocfg"
            ordinary_cfg = 0.0
            distilled_cfg = cfg_strength

        return steps, sway, cfg_mode, ordinary_cfg, distilled_cfg

    def _branches(
        self,
        text: str,
        mode: str,
        reference_audio: str | Path | VoicePrompt | None,
        dialect: str | None,
        plan,
    ):
        """One request's LM prefix embeddings and speaker embedding:
        (cond_embeds, cond_speech, uncond_embeds, initial_uncond, speaker)."""
        processor = self.runtime.processor
        model = self.runtime.model
        speaker_embedding = None
        if mode == "voice_clone":
            prompt = (
                reference_audio
                if isinstance(reference_audio, VoicePrompt)
                else self.create_voice_prompt(reference_audio)
            )
            speaker_embedding = prompt.speaker_embedding.to(model.device)
            # The voice part of the prompt is embedded once per voice and
            # dialect; per request only the text part is tokenized and
            # embedded. Embeddings are per position, so this matches
            # embedding the whole prompt at once.
            voice_embeds = self._voice_embeds(prompt, dialect, plan, speaker_embedding)
            manager = processor.segment_manager
            text_segment = manager.fuse_segments([processor._reference_text_segment(text)])[0]
            total = voice_embeds.size(1) + text_segment.total_length
            if total > manager.config.max_length:
                raise ValueError(
                    f"Inference prefix length {total} exceeds {manager.config.max_length}."
                )
            text_embeds = model.get_input_embeddings()(text_segment.input_ids.to(model.device))
            cond_embeds = torch.cat([voice_embeds, text_embeds], dim=1)
            cond_speech = None
        else:
            cond_prefix = build_prefix_segment(processor, plan.conditional, target_text=text)
            cond_embeds, cond_speech = self._prepare_branch(model, cond_prefix, None)

        uncond_embeds = None
        initial_uncond = None
        if plan.uses_lm_cfg:
            assert plan.unconditional is not None
            if not plan.unconditional.lm_uncond:
                raise ValueError("Only the text-free unconditional branch is supported.")
            # It holds no text, voice or speaker, so it is the same for every request.
            if self._uncond_embeds is None:
                uncond_prefix = build_prefix_segment(processor, plan.unconditional, target_text=text)
                self._uncond_embeds, _ = self._prepare_branch(model, uncond_prefix, None)
            uncond_embeds = self._uncond_embeds

        return cond_embeds, cond_speech, uncond_embeds, initial_uncond, speaker_embedding

    def generate_stream(
        self,
        text: str,
        *,
        mode: str = "tts",
        reference_audio: str | Path | VoicePrompt | None = None,
        dialect: str | None = None,
        cfg_strength: float = 2.0,
        diffusion_steps: int | None = None,
        diffusion_sway_coefficient: float | None = None,
        max_decode_length: int = 750,
        seed: int = 42,
        show_progress: bool = False,
    ) -> Iterator[AudioChunk]:
        """Yield decoded PCM chunks as autoregressive generation progresses.

        Generation runs in a worker thread so the caller can consume each chunk
        immediately. Closing the iterator stops generation at the next decoded
        patch. Use only one active generation per ``CuteTTS`` instance.
        """

        messages: queue.SimpleQueue[AudioChunk | _StreamFailure | object] = (
            queue.SimpleQueue()
        )
        cancelled = threading.Event()

        def emit(chunk: torch.Tensor) -> None:
            if cancelled.is_set():
                raise _StreamCancelled()
            waveform = chunk.detach().to(device="cpu", dtype=torch.float32)
            if waveform.ndim == 3 and waveform.size(0) == 1 and waveform.size(1) == 1:
                waveform = waveform.squeeze(1)
            elif waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            if waveform.ndim != 2 or waveform.size(0) != 1:
                raise RuntimeError(
                    "Streaming decode must return mono audio with shape [1, samples], "
                    f"got {tuple(waveform.shape)}."
                )
            if waveform.size(1) == 0:
                raise RuntimeError("Streaming decode returned an empty audio chunk.")
            messages.put(
                AudioChunk(
                    waveform=waveform.contiguous(),
                    sample_rate=self.runtime.sample_rate,
                )
            )

        def run() -> None:
            try:
                self.generate(
                    text,
                    mode=mode,
                    reference_audio=reference_audio,
                    dialect=dialect,
                    cfg_strength=cfg_strength,
                    diffusion_steps=diffusion_steps,
                    diffusion_sway_coefficient=diffusion_sway_coefficient,
                    max_decode_length=max_decode_length,
                    seed=seed,
                    show_progress=show_progress,
                    pcm_chunk_callback=emit,
                )
            except _StreamCancelled:
                pass
            except BaseException as error:
                messages.put(_StreamFailure(error))
            finally:
                messages.put(_STREAM_DONE)

        worker = threading.Thread(
            target=run,
            name="cutetts-python-stream",
            daemon=True,
        )
        worker.start()
        try:
            while True:
                message = messages.get()
                if message is _STREAM_DONE:
                    break
                if isinstance(message, _StreamFailure):
                    raise message.error
                assert isinstance(message, AudioChunk)
                yield message
        finally:
            cancelled.set()
            worker.join()
