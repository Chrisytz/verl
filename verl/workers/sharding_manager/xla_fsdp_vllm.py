# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import inspect
import logging
import os
import time
from collections import OrderedDict

import torch
# from torch.distributed.device_mesh import DeviceMesh
# from torch.distributed.fsdp.api import FullStateDictConfig, ShardedStateDictConfig, StateDictType
# from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP

try:
    # for torch 2.5+
    from torch.distributed.tensor import DTensor
except ImportError:
    from torch.distributed._tensor import DTensor

from dataclasses import asdict

from verl import DataProto
from verl.protocol import all_gather_data_proto
from verl.third_party.vllm import LLM, vllm_version
from verl.third_party.vllm import parallel_state as vllm_ps
from verl.utils.debug import GPUMemoryLogger, log_gpu_memory_usage
from verl.utils.debug.performance import _timer
from verl.utils.device import get_torch_device
# from verl.utils.fsdp_utils import fsdp_version, layered_summon_lora_params, load_fsdp_model_to_gpu, offload_fsdp_model_to_cpu
from verl.utils.model import convert_weight_keys
from verl.utils.torch_functional import check_device_is_available
from verl.utils.vllm_utils import TensorLoRARequest, VLLMHijack, is_version_ge, patch_vllm_moe_model_weight_loader

from .base import BaseShardingManager

from torch_xla import runtime as xr
import numpy as np
import torch_xla.distributed.spmd as xs
from torch_xla.experimental.spmd_fully_sharded_data_parallel import SpmdFullyShardedDataParallel as FSDPv2
from torch_xla.distributed.fsdp.wrap import transformer_auto_wrap_policy
import functools
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class XLAFSDPVLLMShardingManager(BaseShardingManager):
    @check_device_is_available()
    def __init__(self, module: FSDPv2, inference_engine: LLM, model_config, full_params: bool = False, device_mesh: xs.Mesh = None, offload_param: bool = False, load_format: str = "dummy_hf", layered_summon: bool = True):
        self.module = module
        # For AsyncLLM, inference_engine and model_runner are defer initialized in vLLMAsyncRollout.load_model
        self.inference_engine = inference_engine
        # self.model_runner = inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner if inference_engine else None

        if "vllm_v_0_6_3" in str(type(self.inference_engine)) or "vllm_v_0_5_4" in str(type(self.inference_engine)):
            # vLLM <= v0.6.3
            self.model_runner = self.inference_engine.llm_engine.model_executor.worker.model_runner if self.inference_engine else None
        else:
            # vLLM > v0.6.3
            self.model_runner = self.inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner if self.inference_engine else None

        self.model_config = model_config
        self.device_mesh = device_mesh
        self.offload_param = offload_param
        self.load_format = load_format
        self.layered_summon = layered_summon

        # Full params
        self.full_params = full_params
        # if full_params and fsdp_version(self.module) == 1:
        #     FSDP.set_state_dict_type(self.module, state_dict_type=StateDictType.FULL_STATE_DICT, state_dict_config=FullStateDictConfig())
        # elif fsdp_version(self.module) == 1:
        #     FSDP.set_state_dict_type(
        #         self.module,
        #         state_dict_type=StateDictType.SHARDED_STATE_DICT,
        #         state_dict_config=ShardedStateDictConfig(),
        #     )

        self.tp_size = self.device_mesh["infer_tp"].size()
        self.tp_rank = self.device_mesh["infer_tp"].get_local_rank()

        # Note that torch_random_states may be different on each dp rank
        self.torch_random_states = get_torch_device().get_rng_state()
        # get a random rng states
        if self.device_mesh is not None:
            gen_dp_rank = self.device_mesh["dp"].get_local_rank()
            get_torch_device().manual_seed(gen_dp_rank + 1000)  # make sure all tp ranks have the same random states
            self.gen_random_states = get_torch_device().get_rng_state()
            get_torch_device().set_rng_state(self.torch_random_states)
        else:
            self.gen_random_states = None

        self.base_sync_done: bool = "dummy" not in load_format
        if is_version_ge(pkg="vllm", minver="0.7.3"):
            VLLMHijack.hijack()

    @GPUMemoryLogger(role="fsdp vllm sharding_manager", logger=logger)
    def __enter__(self):
        # NOTE: Basically, we only need `get_torch_device().empty_cache()` before vllm wake_up and
        # after vllm sleep, since vllm has its own caching memory allocator CuMemAllocator.
        # Out of vllm scope, we should avoid empty cache to let pytorch using caching memory
        # to speed up memory allocations.
        #
        # pytorch: https://pytorch.org/docs/stable/notes/cuda.html#memory-management
        # vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/device_allocator/cumem.py#L103
        self.timing = {}
        with _timer("reshard", self.timing):
            get_torch_device().empty_cache()

            log_gpu_memory_usage("Before state_dict() in sharding manager memory", logger=logger)
            # if self.offload_param:
                # load_fsdp_model_to_gpu(self.module)

            peft_config = None
            peft_model = getattr(self.module, "_orig_module", self.module)
            if hasattr(peft_model, "peft_config"):
                peft_config = peft_model.peft_config.get("default", None)
            else:
                params = self.module.state_dict()
            params = convert_weight_keys(params, getattr(self.module, "_orig_module", self.module))
            log_gpu_memory_usage("After state_dict() in sharding manager memory", logger=logger)

            # Copy, not share memory
            load_format = "hf" if self.full_params else "dtensor"

            if vllm_version in (
                "0.5.4",
                "0.6.3",
            ):
                self.inference_engine.sync_model_weights(params, load_format=load_format)
                log_gpu_memory_usage("After sync model weights in sharding manager", logger=logger)
                del params
            else:
                if "tags" in inspect.signature(self.inference_engine.wake_up).parameters:
                    self.inference_engine.wake_up(tags=["weights"])
                else:
                    self.inference_engine.wake_up()

                # update model params
                self.update_params(params, peft_config=peft_config)
                log_gpu_memory_usage("After sync model weights in sharding manager", logger=logger)
                del params
                # if self.offload_param:
                    # offload_fsdp_model_to_cpu(self.module)
                get_torch_device().empty_cache()

                if "tags" in inspect.signature(self.inference_engine.wake_up).parameters:
                    self.inference_engine.wake_up(tags=["kv_cache"])

            log_gpu_memory_usage("After del state_dict and empty_cache in sharding manager", logger=logger)

            # important: need to manually set the random states of each tp to be identical.
            if self.device_mesh is not None:
                self.torch_random_states = get_torch_device().get_rng_state()
                get_torch_device().set_rng_state(self.gen_random_states)

    @GPUMemoryLogger(role="fsdp vllm sharding_manager", logger=logger)
    def __exit__(self, exc_type, exc_value, traceback):
        # TODO(ZSL): check this
        if vllm_version in (
            "0.5.4",
            "0.6.3",
        ):
            self.inference_engine.offload_model_weights()
        else:
            self.inference_engine.sleep(level=1)

        self.module.train()

        # add empty cache after each compute
        get_torch_device().empty_cache()

        # restore random states
        if self.device_mesh is not None:
            self.gen_random_states = get_torch_device().get_rng_state()
            get_torch_device().set_rng_state(self.torch_random_states)

    @GPUMemoryLogger(role="fsdp vllm sharding_manager", logger=logger)
    def preprocess_data(self, data: DataProto) -> DataProto:
        """All gather across tp group to make each rank has identical input."""
        if self.tp_size == 1:
            return data

        # TODO: Current impl doesn't consider FSDP with torch micro-dp
        if vllm_version in (
            "0.5.4",
            "0.6.3",
        ):
            group = vllm_ps.get_tensor_model_parallel_group()
        else:
            group = vllm_ps.get_tensor_model_parallel_group().device_group

        all_gather_data_proto(data=data, process_group=group)
        return data

    @GPUMemoryLogger(role="fsdp vllm sharding_manager", logger=logger)
    def postprocess_data(self, data: DataProto) -> DataProto:
        """Get chunk data of this tp rank since we do all gather in preprocess."""
        if self.tp_size == 1:
            return data

        return data.chunk(chunks=self.tp_size)[self.tp_rank]

    def update_params(self, updated_params, peft_config=None):
        model = self.model_runner.model
        if peft_config:
            if self.base_sync_done:
                lora_int_id = int(time.time_ns() % 0x7FFFFFFF)
                lora_reqest = TensorLoRARequest(
                    lora_name=f"{lora_int_id}",
                    lora_int_id=lora_int_id,
                    lora_path="simon_lora_path",
                    peft_config=asdict(peft_config),
                    lora_tensors=updated_params,
                )
                self.inference_engine.llm_engine.add_lora(lora_reqest)
                logger.info(f"vLLM load weights, loaded_params: {len(updated_params)}")
                return
            else:

                def replace_lora_wrapper(k):
                    stacked_params = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
                    if any([k.endswith(f"{s}.weight") for s in stacked_params]):
                        return k.replace(".weight", ".base_layer.weight")
                    if any([k.endswith(f"{s}.bias") for s in stacked_params]):
                        return k.replace(".bias", ".base_layer.bias")
                    return k

                updated_params = {replace_lora_wrapper(k): v for k, v in updated_params.items()}

        patch_vllm_moe_model_weight_loader(model)
        device = get_torch_device().current_device()  # used when fsdp2 set cpu_offload_policy
        loaded_params = model.load_weights(((name, param.to(device, non_blocking=True).full_tensor() if isinstance(param, DTensor) else param) for name, param in updated_params.items()))

        self.base_sync_done = True
        logger.info(f"vLLM load weights, loaded_params: {len(loaded_params) if loaded_params else -1}")
