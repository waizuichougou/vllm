# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Draft models must honour ``moe_backend`` from --speculative-config.

The draft is loaded with the target's VllmConfig, so without an explicit
override it inherits the target's --moe-backend. An MTP head on a quantized
target is typically unquantized, and quantized-only backends reject it, so the
server fails to start rather than falling back. DeepSeek V4 DSpark also uses
the backend to select between its regular and mega-MoE model paths.
"""

from dataclasses import dataclass, field
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm.config import LoadConfig
from vllm.v1.worker.gpu.spec_decode.dspark import utils as dspark_utils
from vllm.v1.worker.gpu.spec_decode.eagle.utils import load_eagle_model


@dataclass
class _KernelConfig:
    moe_backend: str | None = None


@dataclass
class _CacheConfig:
    cache_dtype: str = "auto"


@dataclass
class _AttentionConfig:
    backend: str | None = None
    use_non_causal: bool = False


@dataclass
class _EPLBConfig:
    num_redundant_experts: int = 0


@dataclass
class _ParallelConfig:
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    enable_eplb: bool = False
    eplb_config: _EPLBConfig = field(default_factory=_EPLBConfig)
    enable_elastic_ep: bool = False


@dataclass
class _SpeculativeConfig:
    attention_backend: str | None = None
    moe_backend: str | None = None
    kv_cache_dtype: str | None = None
    draft_model_config: object = field(
        default_factory=lambda: SimpleNamespace(
            hf_config=SimpleNamespace(model_type="deepseek_v4")
        )
    )
    draft_parallel_config: _ParallelConfig = field(default_factory=_ParallelConfig)


@dataclass
class _VllmConfig:
    kernel_config: _KernelConfig
    cache_config: _CacheConfig
    speculative_config: _SpeculativeConfig
    attention_config: _AttentionConfig = field(default_factory=_AttentionConfig)
    parallel_config: _ParallelConfig = field(default_factory=_ParallelConfig)
    load_config: LoadConfig = field(default_factory=LoadConfig)


def _config(target_moe: str, draft_moe: str | None) -> _VllmConfig:
    return _VllmConfig(
        kernel_config=_KernelConfig(moe_backend=target_moe),
        cache_config=_CacheConfig(),
        speculative_config=_SpeculativeConfig(moe_backend=draft_moe),
    )


class _Captured(Exception):
    """Stops load_eagle_model once we hold the config it would have used."""

    def __init__(self, vllm_config):
        self.vllm_config = vllm_config


def _capture_eagle_draft_config(cfg):
    def _fake_get_model(*, vllm_config, model_config):
        raise _Captured(vllm_config)

    with (
        patch("vllm.v1.worker.gpu.spec_decode.eagle.utils.get_model", _fake_get_model),
        patch(
            "vllm.v1.worker.gpu.spec_decode.utils.get_pp_group",
            return_value=SimpleNamespace(world_size=1),
        ),
        pytest.raises(_Captured) as exc,
    ):
        load_eagle_model(object(), cfg)
    return exc.value.vllm_config


def _capture_dspark_draft_config(cfg):
    def _fake_get_model(*, vllm_config, model_config):
        raise _Captured(vllm_config)

    with (
        patch(
            "vllm.model_executor.model_loader.get_model",
            _fake_get_model,
        ),
        patch(
            "vllm.model_executor.models.qwen3_dflash.dflash_has_any_non_causal",
            return_value=False,
        ),
        patch(
            "vllm.model_executor.models.utils.get_draft_quant_config",
            return_value=None,
        ),
        patch.object(
            dspark_utils,
            "get_pp_safe_draft_load_config",
            side_effect=lambda load_config: load_config,
        ),
        pytest.raises(_Captured) as exc,
    ):
        dspark_utils.load_dspark_model(object(), cfg)
    return exc.value.vllm_config


@pytest.mark.parametrize(
    "capture_draft_config",
    [_capture_eagle_draft_config, _capture_dspark_draft_config],
)
def test_draft_moe_backend_overrides_the_target(capture_draft_config):
    """A draft moe_backend must reach the draft's kernel config."""
    used = capture_draft_config(_config("deep_gemm_mega_moe", "triton"))
    assert used.kernel_config.moe_backend == "triton"


@pytest.mark.parametrize(
    "capture_draft_config",
    [_capture_eagle_draft_config, _capture_dspark_draft_config],
)
def test_draft_inherits_target_when_no_override(capture_draft_config):
    """No override means the draft still inherits the target, as before."""
    used = capture_draft_config(_config("deep_gemm_mega_moe", None))
    assert used.kernel_config.moe_backend == "deep_gemm_mega_moe"


@pytest.mark.parametrize(
    "capture_draft_config",
    [_capture_eagle_draft_config, _capture_dspark_draft_config],
)
def test_override_does_not_mutate_the_target_config(capture_draft_config):
    """The target must keep its own backend after the draft is built."""
    cfg = _config("deep_gemm_mega_moe", "triton")
    capture_draft_config(cfg)
    assert cfg.kernel_config.moe_backend == "deep_gemm_mega_moe"
