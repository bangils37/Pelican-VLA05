# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Modifications Copyright 2026 The BaseVLA Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only adapter for separately distributed Cosmos TorchScript models.

This module targets the public NVIDIA Cosmos Tokenizer JIT interface while
intentionally excluding the native-PyTorch network, quantizer, image I/O, and
video utilities from the upstream project.
"""

from pathlib import Path

import torch


class ImageTokenizer(torch.nn.Module):
    """Load a paired Cosmos image encoder and decoder from TorchScript files.

    ``_enc_model`` and ``_dec_model`` remain stable because those names are
    present in existing PelicanVLA05 checkpoint keys.
    """

    def __init__(
        self,
        checkpoint_enc: str | Path,
        checkpoint_dec: str | Path,
        *,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self._runtime_device = torch.device(device)
        self._runtime_dtype = dtype
        self._enc_path = Path(checkpoint_enc).expanduser()
        self._dec_path = Path(checkpoint_dec).expanduser()
        self._enc_model = self._load_script(self._enc_path)
        self._dec_model = self._load_script(self._dec_path)

    def _load_script(self, checkpoint: str | Path) -> torch.jit.ScriptModule:
        path = Path(checkpoint).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Cosmos TorchScript checkpoint not found: {path}")

        module = torch.jit.load(str(path), map_location=self._runtime_device)
        module.eval()
        # Keep device and dtype conversion as separate operations. NVIDIA's
        # published JIT loader follows this sequence, and some CUDA execution
        # plans are sensitive to how the ScriptModule storage is materialized.
        module.to(self._runtime_device)
        module.to(self._runtime_dtype)
        for parameter in module.parameters():
            parameter.requires_grad_(False)
        return module

    def to(self, *args, **kwargs):
        parsed_device, parsed_dtype, _, _ = torch._C._nn._parse_to(*args, **kwargs)
        device_changed = False
        if parsed_device is not None:
            new_device = torch.device(parsed_device)
            if new_device != self._runtime_device:
                self._runtime_device = new_device
                device_changed = True
        if parsed_dtype is not None:
            self._runtime_dtype = parsed_dtype

        if device_changed and hasattr(self, "_enc_path") and hasattr(self, "_dec_path"):
            # TorchScript JIT graph constants (such as CONSTANTS.c0 in Cosmos) do not migrate with .to(device).
            # Reloading on target device ensures all graph constants are allocated on the new device.
            self._enc_model = self._load_script(self._enc_path)
            self._dec_model = self._load_script(self._dec_path)
        else:
            if parsed_device is not None:
                if hasattr(self, "_enc_model"):
                    self._enc_model.to(self._runtime_device)
                if hasattr(self, "_dec_model"):
                    self._dec_model.to(self._runtime_device)
            if parsed_dtype is not None:
                if hasattr(self, "_enc_model"):
                    self._enc_model.to(self._runtime_dtype)
                if hasattr(self, "_dec_model"):
                    self._dec_model.to(self._runtime_dtype)
        return super().to(*args, **kwargs)

    def _module_spec(
        self, module: torch.jit.ScriptModule
    ) -> tuple[torch.device, torch.dtype]:
        for tensor in (*module.parameters(), *module.buffers()):
            if tensor.is_floating_point():
                return tensor.device, tensor.dtype
        return self._runtime_device, self._runtime_dtype

    @staticmethod
    def _tensor_result(result: object, operation: str) -> torch.Tensor:
        if isinstance(result, torch.Tensor):
            return result
        if (
            isinstance(result, (tuple, list))
            and result
            and isinstance(result[0], torch.Tensor)
        ):
            return result[0]
        raise TypeError(
            f"Cosmos {operation} returned an unsupported value: {type(result).__name__}"
        )

    def _execute(
        self,
        module: torch.jit.ScriptModule,
        value: torch.Tensor,
        operation: str,
    ) -> torch.Tensor:
        if value.device != self._runtime_device:
            self.to(value.device)
            module = self._enc_model if operation == "encoder" else self._dec_model

        target_device, target_dtype = self._module_spec(module)
        module_input = value
        if module_input.device != target_device:
            module_input = module_input.to(device=target_device)
        if module_input.is_floating_point() and module_input.dtype != target_dtype:
            module_input = module_input.to(dtype=target_dtype)

        if target_device.type == "cuda":
            with torch.autocast(device_type="cuda", enabled=False):
                result = self._tensor_result(module(module_input), operation)
        else:
            result = self._tensor_result(module(module_input), operation)

        if result.is_floating_point() and value.is_floating_point():
            if result.device != value.device or result.dtype != value.dtype:
                return result.to(device=value.device, dtype=value.dtype)
            return result
        return (
            result.to(device=value.device) if result.device != value.device else result
        )

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self._execute(self._enc_model, images, "encoder").clone()

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self._execute(self._dec_model, latents, "decoder").clone()

    @torch.no_grad()
    def autoencode(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(input_tensor))

    @torch.no_grad()
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.autoencode(input_tensor)
