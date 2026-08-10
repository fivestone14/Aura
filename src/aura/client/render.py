"""Turning a reply into speech with the chosen tone applied.

This is the layer-4 → layer-2 handoff, and it was the project's largest unvalidated
assumption: the tone policy decides a rate, a pitch shift and an energy scale, and
something has to actually *apply* them. A renderer that only accepts text makes the
whole prosody system decorative.

**Resolved 2026-08-09 by reading Kokoro's source.** Its entire forward pass is thirty
lines of plain PyTorch, and all three predictions exist as local variables before being
handed to the vocoder:

    duration = torch.sigmoid(duration).sum(axis=-1) / speed   # per-phoneme timing
    F0_pred, N_pred = self.predictor.F0Ntrain(en, s)          # pitch and energy
    audio = self.decoder(asr, F0_pred, N_pred, ref_s[:, :128])

So a subclass that overrides one method can scale durations, shift the pitch contour and
scale the energy envelope before the vocoder ever runs. No fork, no patching, no
reaching into private state.

⚠️ The "decoder only, no diffusion, no encoder release" note on the model card caused the
original doubt. It refers to the *style diffusion encoder*, which is a different
component — `ProsodyPredictor` is instantiated and used, which is all this needs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from aura.types import ProsodyTarget

if TYPE_CHECKING:  # pragma: no cover
    import torch

SEMITONE_RATIO = 2 ** (1 / 12)
"""Frequency multiplier for one semitone. Pitch is perceived logarithmically, so a shift
is a multiplication in Hz, never an addition."""


@dataclass(frozen=True, slots=True)
class RenderedSpeech:
    """Audio, plus what was actually done to it.

    `applied` records the target the renderer honoured. It is what the controllability
    evaluation compares against measured output — the check that the knobs moved, rather
    than were accepted and ignored (docs/DESIGN.md §6).
    """

    samples: Any
    """Waveform. Typed loosely so this module imports without torch installed."""

    sample_rate: int
    applied: ProsodyTarget


class Renderer(ABC):
    """Speech synthesis with prosody control."""

    @abstractmethod
    def render(self, text: str, target: ProsodyTarget) -> RenderedSpeech:
        """Speak `text` with `target` applied.

        Implementations that cannot honour a dimension must say so rather than silently
        ignoring it — a renderer that accepts a pitch shift and does nothing is the
        failure mode this interface exists to make visible.
        """

    @property
    @abstractmethod
    def controllable_dimensions(self) -> frozenset[str]:
        """Which of rate, pitch, energy and pause this renderer can actually apply.

        Declared rather than assumed, so a caller can check at wiring time instead of
        discovering it from audio that never changes.
        """


class NullRenderer(Renderer):
    """Produces nothing. For tests and for running the pipeline without audio."""

    @property
    def controllable_dimensions(self) -> frozenset[str]:
        return frozenset()

    def render(self, text: str, target: ProsodyTarget) -> RenderedSpeech:
        del text
        return RenderedSpeech(samples=None, sample_rate=24_000, applied=target)


class KokoroRenderer(Renderer):
    """Kokoro with the prosody predictors intercepted.

    Wraps the model rather than editing it: `_render_with_prosody` reimplements the
    thirty-line forward pass with three multiplications inserted. Upstream changes to
    Kokoro are then a visible break here rather than a silent divergence in a fork.

    Torch and kokoro are imported lazily so this module — and the rest of the package —
    stays importable without them.
    """

    SAMPLE_RATE = 24_000

    def __init__(self, *, lang_code: str = "a", voice: str = "af_heart") -> None:
        try:
            import torch  # noqa: F401
            from kokoro import KModel, KPipeline
        except ImportError as exc:  # pragma: no cover — depends on install extras
            raise ImportError(
                "Kokoro is not installed. Install it with:\n"
                "    uv pip install 'aura-core[speech]'"
            ) from exc

        self._torch = __import__("torch")
        self._model = KModel().eval()
        self._pipeline = KPipeline(lang_code=lang_code, model=self._model)
        self._voice = voice

    @property
    def controllable_dimensions(self) -> frozenset[str]:
        """Everything except pause structure.

        Pauses live in the phoneme sequence rather than the acoustic predictions, so
        lengthening them means editing the input, not scaling a tensor. Declared missing
        rather than quietly approximated.
        """
        return frozenset({"rate", "pitch", "energy"})

    def render(self, text: str, target: ProsodyTarget) -> RenderedSpeech:
        phonemes, ref_s = self._prepare(text)
        audio = self._render_with_prosody(phonemes, ref_s, target)
        return RenderedSpeech(
            samples=audio.cpu(), sample_rate=self.SAMPLE_RATE, applied=target
        )

    def _prepare(self, text: str) -> tuple[str, torch.FloatTensor]:
        """Phonemise and fetch the voice embedding."""
        phonemes = next(iter(self._pipeline(text, voice=self._voice))).phonemes
        pack = self._pipeline.load_voice(self._voice)
        return phonemes, pack[len(phonemes) - 1]

    def _render_with_prosody(
        self, phonemes: str, ref_s: torch.FloatTensor, target: ProsodyTarget
    ) -> torch.FloatTensor:
        """Kokoro's forward pass, with the three predictions scaled on the way through.

        Mirrors `KModel.forward_with_tokens`. The inserted lines are marked.
        """
        torch = self._torch
        model = self._model

        ids = [i for i in (model.vocab.get(p) for p in phonemes) if i is not None]
        input_ids = torch.LongTensor([[0, *ids, 0]]).to(model.device)
        ref_s = ref_s.to(model.device).unsqueeze(0)

        lengths = torch.full(
            (input_ids.shape[0],), input_ids.shape[-1], device=model.device, dtype=torch.long
        )
        mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1)
        mask = torch.gt(mask.type_as(lengths) + 1, lengths.unsqueeze(1)).to(model.device)

        with torch.no_grad():
            bert_dur = model.bert(input_ids, attention_mask=(~mask).int())
            d_en = model.bert_encoder(bert_dur).transpose(-1, -2)
            s = ref_s[:, 128:]
            d = model.predictor.text_encoder(d_en, s, lengths, mask)
            x, _ = model.predictor.lstm(d)

            # ── RATE ─────────────────────────────────────────────────────────────
            # Kokoro divides by `speed`; a rate_scale above 1.0 means faster speech,
            # which is shorter durations — hence dividing, not multiplying.
            duration = torch.sigmoid(model.predictor.duration_proj(x)).sum(axis=-1)
            duration = duration / target.rate_scale
            pred_dur = torch.round(duration).clamp(min=1).long().squeeze()

            indices = torch.repeat_interleave(
                torch.arange(input_ids.shape[1], device=model.device), pred_dur
            )
            aln = torch.zeros((input_ids.shape[1], indices.shape[0]), device=model.device)
            aln[indices, torch.arange(indices.shape[0])] = 1
            aln = aln.unsqueeze(0)

            en = d.transpose(-1, -2) @ aln
            f0, energy = model.predictor.F0Ntrain(en, s)

            # ── PITCH ────────────────────────────────────────────────────────────
            # Multiplicative: pitch is perceived logarithmically, so −2 semitones is
            # ×0.89, not −2 Hz.
            if target.pitch_shift_semitones:
                f0 = f0 * (SEMITONE_RATIO**target.pitch_shift_semitones)

            # ── ENERGY ───────────────────────────────────────────────────────────
            if target.energy_scale != 1.0:
                energy = energy * target.energy_scale

            t_en = model.text_encoder(input_ids, lengths, mask)
            return model.decoder(t_en @ aln, f0, energy, ref_s[:, :128]).squeeze()
