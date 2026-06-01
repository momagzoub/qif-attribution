"""Image-generation backends for generation job execution."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from qif_attribution.generation.png import write_solid_rgb_png


@dataclass(frozen=True)
class ImageGenerationRequest:
    job_id: str
    prompt: str
    generator_id: str
    model_id: str
    seed: int
    sampler: str
    steps: int
    cfg_scale: float
    width: int
    height: int
    output_path: Path


class ImageGenerationBackend(Protocol):
    def generate(self, request: ImageGenerationRequest) -> None:
        """Generate one image for a request and write it to `output_path`."""


class PlaceholderImageBackend:
    """Smoke-test backend that writes deterministic solid-color PNGs.

    This backend is only for plumbing tests. Images produced by it must not be
    treated as generated-image research data.
    """

    def generate(self, request: ImageGenerationRequest) -> None:
        digest = hashlib.sha256(
            f"{request.job_id}:{request.prompt}:{request.seed}:{request.generator_id}".encode()
        ).digest()
        rgb = (digest[0], digest[1], digest[2])
        write_solid_rgb_png(
            request.output_path,
            width=request.width,
            height=request.height,
            rgb=rgb,
        )


class DiffusersImageBackend:
    """Local Hugging Face diffusers backend for real text-to-image generation."""

    def __init__(
        self,
        *,
        device: str = "auto",
        torch_dtype: str = "auto",
        local_files_only: bool = False,
    ) -> None:
        self.device = device
        self.torch_dtype = torch_dtype
        self.local_files_only = local_files_only
        self._pipelines: dict[tuple[str, str], object] = {}

    def generate(self, request: ImageGenerationRequest) -> None:
        torch = _import_torch()
        pipe = self._load_pipeline(request.model_id, request.sampler)
        generator_device = self._generator_device(torch)
        generator = torch.Generator(device=generator_device).manual_seed(request.seed)

        result = pipe(
            prompt=request.prompt,
            num_inference_steps=request.steps,
            guidance_scale=request.cfg_scale,
            width=request.width,
            height=request.height,
            generator=generator,
        )
        request.output_path.parent.mkdir(parents=True, exist_ok=True)
        result.images[0].save(request.output_path)

    def _load_pipeline(self, model_id: str, sampler: str):
        cache_key = (model_id, sampler)
        if cache_key in self._pipelines:
            return self._pipelines[cache_key]

        torch = _import_torch()
        diffusers = _import_diffusers()
        dtype = _resolve_torch_dtype(torch, self.torch_dtype)
        device = _resolve_device(torch, self.device)
        pipeline_cls = getattr(diffusers, "AutoPipelineForText2Image", None)
        if pipeline_cls is None:
            pipeline_cls = diffusers.StableDiffusionPipeline

        kwargs = {"local_files_only": self.local_files_only}
        if dtype is not None:
            kwargs["torch_dtype"] = dtype
        pipe = pipeline_cls.from_pretrained(model_id, **kwargs)
        pipe = _configure_scheduler(diffusers, pipe, sampler)
        pipe = _disable_safety_checker(pipe)
        pipe = pipe.to(device)
        if hasattr(pipe, "set_progress_bar_config"):
            pipe.set_progress_bar_config(disable=True)
        self._pipelines[cache_key] = pipe
        return pipe

    def _generator_device(self, torch) -> str:
        device = _resolve_device(torch, self.device)
        if device == "mps":
            return "cpu"
        return device


def make_backend(
    backend: str,
    *,
    device: str = "auto",
    torch_dtype: str = "auto",
    local_files_only: bool = False,
) -> ImageGenerationBackend:
    if backend == "placeholder":
        return PlaceholderImageBackend()
    if backend == "diffusers":
        return DiffusersImageBackend(
            device=device,
            torch_dtype=torch_dtype,
            local_files_only=local_files_only,
        )
    raise ValueError(f"unsupported generation backend: {backend}")


def _import_torch():
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "The diffusers backend requires torch. Install with `pip install -e .[generation]`."
        ) from exc
    return torch


def _import_diffusers():
    try:
        import diffusers  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "The diffusers backend requires diffusers. Install with `pip install -e .[generation]`."
        ) from exc
    return diffusers


def _resolve_device(torch, device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_torch_dtype(torch, torch_dtype: str):
    if torch_dtype == "auto":
        if torch.cuda.is_available():
            return torch.float16
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.float16
        return None
    mapping = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    if torch_dtype not in mapping:
        raise ValueError(f"unsupported torch dtype: {torch_dtype}")
    return mapping[torch_dtype]


def _disable_safety_checker(pipe):
    """Avoid false-positive black placeholder images in controlled benign runs."""

    if hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None
    if hasattr(pipe, "requires_safety_checker"):
        pipe.requires_safety_checker = False
    return pipe


def _configure_scheduler(diffusers, pipe, sampler: str):
    sampler_key = sampler.lower()
    scheduler_map = {
        "euler": "EulerDiscreteScheduler",
        "euler_a": "EulerAncestralDiscreteScheduler",
        "dpmpp_2m": "DPMSolverMultistepScheduler",
        "dpmsolver": "DPMSolverMultistepScheduler",
    }
    scheduler_name = scheduler_map.get(sampler_key)
    if scheduler_name is None:
        return pipe
    scheduler_cls = getattr(diffusers, scheduler_name, None)
    if scheduler_cls is None:
        return pipe
    pipe.scheduler = scheduler_cls.from_config(pipe.scheduler.config)
    return pipe
