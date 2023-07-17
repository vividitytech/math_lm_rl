# flake8: noqa

# Copyright 2022 The HuggingFace Team. All rights reserved.
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
from .modeling_base import PreTrainedModelWrapper, create_reference_model
from .modeling_value_head import AutoModelForCausalLMWithValueHead, AutoModelForSeq2SeqLMWithValueHead
from transformers import GPT2LMHeadModel

SUPPORTED_ARCHITECTURES = (
    AutoModelForCausalLMWithValueHead,
    AutoModelForSeq2SeqLMWithValueHead,
    GPT2LMHeadModel,
)
