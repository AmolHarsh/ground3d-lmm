# Copyright (c) 2025 Ye Liu. Licensed under the BSD-3-Clause License.

import random

import torch
import torch.nn as nn
from hydra import compose
from hydra.utils import instantiate
from nncore.nn import constant_init_, xavier_init_
from transformers import (AutoConfig, AutoModelForImageTextToText, AutoProcessor, Qwen3VLConfig,
                          Qwen3VLForConditionalGeneration, Qwen3VLModel, Qwen3VLProcessor)
from transformers.models.auto.modeling_auto import MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRMSNorm

from functools import partial

def save_hidden_state(model, name, module, args):
    setattr(model, name, args[0] if isinstance(args, tuple) else args)


def add_point_embedding(model,  module, args, kwargs):
    inputs_embeds, point_embeds, point_mask, input_ids = (
        kwargs.pop(k, None) for k in ('inputs_embeds', 'point_embeds', 'point_mask', 'input_ids'))

    if point_embeds is not None and point_mask is not None and inputs_embeds.shape[1] >1:
        n_point_features = point_embeds.shape[0]
        if point_mask.sum() != n_point_features:
            raise ValueError(f"point_mask.sum({point_mask.sum()}) != n_point_features({n_point_features})")

        mask_unsqueezed = point_mask.unsqueeze(-1)
        mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
        point_mask_expanded = mask_expanded.to(inputs_embeds.device)

        point_embeds = point_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(point_mask_expanded, point_embeds)
        # print(f"point embedding has been added")
    
    # else:
    #     print(f"point embedding has not been added")

    # ensure gradient tracking (in case that embed_tokens has been frozen)
    
    if model.training and not inputs_embeds.requires_grad:
        inputs_embeds.requires_grad = True

    # ensure the modifications are saved
    kwargs['inputs_embeds'] = inputs_embeds

    return (), kwargs


class PatchedQwen3VLProcessor(Qwen3VLProcessor):

    def _check_special_mm_tokens(self, text, *args, **kwargs):
        self.cache_text = text
        return super()._check_special_mm_tokens(text, *args, **kwargs)


class PointQwen3VLConfig(Qwen3VLConfig):
    model_type = 'point_qwen3_vl'


class PointQwen3VLModel(Qwen3VLModel):
    config_class = PointQwen3VLConfig


class PointQwen3VLForConditionalGeneration(Qwen3VLForConditionalGeneration):
    config_class = PointQwen3VLConfig

    def __init__(self, config):
        super().__init__(config)

        self.model = PointQwen3VLModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)

        self.model.visual.merger.linear_fc1.register_forward_pre_hook(partial(save_hidden_state, self, 'visual_tokens'))
        # self.model.visual = None

        self.model.language_model.norm.register_forward_pre_hook(partial(save_hidden_state, self, 'last_hidden_state'))
        self.model.language_model.register_forward_pre_hook(partial(add_point_embedding, self), with_kwargs=True)

        self.post_init()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=None,
                labels=None,
                use_cache=None,
                return_dict=None,
                pixel_values=None,
                pixel_values_videos=None,
                image_grid_thw=None,
                video_grid_thw=None,
                point_embeds=None,
                point_mask=None,  # pyright: ignore[reportUnusedParameter]
                **kwargs):


        # move input_ids to the correct device (in case of auto device map)
        input_ids = input_ids.to(self.model.language_model.embed_tokens.weight.device)

        # media should either image or video
        media_grid_thw = image_grid_thw if image_grid_thw is not None else video_grid_thw

        
        
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=not self.training,
            return_dict=True,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            media_grid_thw=media_grid_thw,
            point_embeds=point_embeds,
            point_mask= input_ids == self.config.point_token_id,
            **kwargs)


        return outputs

    def prepare_inputs_for_generation(self,
                                      *args,
                                      cache_position=None,
                                      point_embeds=None,
                                      point_mask=None,
                                      **kwargs):
        model_inputs = super().prepare_inputs_for_generation(*args, cache_position=cache_position, **kwargs)
        model_inputs.update({
            'point_embeds': point_embeds if cache_position[0] == 0 else None,
            'point_mask': point_mask if cache_position[0] == 0 else None,
        })

        return model_inputs


# set the patched model to a vision model
MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES[PointQwen3VLConfig.model_type] = 'PointQwen3VLForConditionalGeneration'

AutoConfig.register(PointQwen3VLConfig.model_type, PointQwen3VLConfig)
AutoModelForImageTextToText.register(PointQwen3VLConfig, PointQwen3VLForConditionalGeneration)
AutoProcessor.register(PointQwen3VLConfig, PatchedQwen3VLProcessor)
