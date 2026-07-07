# Copyright (c) 2025 Ye Liu. Licensed under the BSD-3-Clause License.

import random

import torch
import torch.nn as nn
from hydra import compose
from hydra.utils import instantiate
from nncore.nn import constant_init_, xavier_init_
from transformers import (AutoConfig, AutoModel, AutoProcessor, Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration,
                          Qwen2_5_VLModel, Qwen2_5_VLProcessor, Qwen2_5_VLTextModel)
from transformers.models.auto.modeling_auto import MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionTransformerPretrainedModel, Qwen2RMSNorm

def cache_state_hook(module, inputs, ouputs=None):
    module.state = inputs[0] if isinstance(inputs, tuple) else inputs


class PatchedQwen2_5_VLProcessor(Qwen2_5_VLProcessor):

    def _check_special_mm_tokens(self, text, *args, **kwargs):
        self.cache_text = text
        return super()._check_special_mm_tokens(text, *args, **kwargs)


class PointQwen2_5_VLConfig(Qwen2_5_VLConfig):
    model_type = 'point_qwen2_5_vl'


class PointQwen2_5_VisionTransformerPretrainedModel(Qwen2_5_VisionTransformerPretrainedModel):

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.merger.mlp.register_forward_pre_hook(cache_state_hook)


class PointQwen2_5_VLModel(Qwen2_5_VLModel):
    config_class = PointQwen2_5_VLConfig

    def __init__(self, config):
        super(Qwen2_5_VLModel, self).__init__(config)
        self.visual = None # TODO visual encoder is not used for now
        self.language_model = Qwen2_5_VLTextModel._from_config(config.text_config)
        self.rope_deltas = None
        self.post_init()
        self.language_model.norm.register_forward_pre_hook(cache_state_hook)


class PointQwen2_5_VLForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
    config_class = PointQwen2_5_VLConfig

    def __init__(self, config):
        super().__init__(config)

        self.model = PointQwen2_5_VLModel(config)

        self.post_init()


    def forward(self,
                input_ids=None,
                attention_mask=None,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=None,
                labels=None,
                use_cache=None,
                output_attentions=None,
                output_hidden_states=True,
                return_dict=None,
                pixel_values=None,
                pixel_values_videos=None,
                image_grid_thw=None,
                video_grid_thw=None,
                rope_deltas=None,
                cache_position=None,
                second_per_grid_ts=None,
                point_embeds=None,
                **kwargs,):
        
        # if caching := not self.training and (past_key_values is None or len(past_key_values) == 0):
        #     self.seg = []

        mode = 'training' if self.training else ('caching' if (past_key_values is None or len(past_key_values) == 0) else 'generating')

        # move input_ids to the correct device (in case of auto device map)
        input_ids = input_ids.to(self.model.language_model.embed_tokens.weight.device)
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
            device, dtype = inputs_embeds.device, inputs_embeds.dtype

            if pixel_values is not None:
                image_embeds = self.get_image_features(pixel_values, image_grid_thw)
                image_embeds = torch.cat(image_embeds)
                n_image_tokens = (input_ids == self.config.image_token_id).sum()
                n_image_features = image_embeds.shape[0]
                assert n_image_tokens == n_image_features

                mask = input_ids == self.config.image_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                image_mask = mask_expanded.to(device)

                image_embeds = image_embeds.to(device, dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
                video_embeds = torch.cat(video_embeds)
                n_video_tokens = (input_ids == self.config.video_token_id).sum()
                n_video_features = video_embeds.shape[0]
                assert n_video_tokens == n_video_features

                mask = input_ids == self.config.video_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                video_mask = mask_expanded.to(device)

                video_embeds = video_embeds.to(device, dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
            
            if point_embeds is not None:
                if mode in ['training', 'caching']:
                    n_point_tokens = (input_ids == self.config.point_token_id).sum()
                    n_point_features = point_embeds.shape[0]
                    if n_point_tokens != n_point_features:
                        raise ValueError(f"n_point_tokens({n_point_tokens}) != n_point_features({n_point_features})")

                    mask = input_ids == self.config.point_token_id
                    mask_unsqueezed = mask.unsqueeze(-1)
                    mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                    point_mask = mask_expanded.to(device)

                    point_embeds = point_embeds.to(device, dtype)
                    inputs_embeds = inputs_embeds.masked_scatter(point_mask, point_embeds)
                elif mode == 'generating':
                    pass

        # ensure gradient tracking (in case that embed_tokens has been frozen)
        if self.training and not inputs_embeds.requires_grad:
            inputs_embeds.requires_grad = True

        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=not self.training,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            rope_deltas=rope_deltas,
            cache_position=cache_position,
            second_per_grid_ts=second_per_grid_ts,
            **kwargs,)
        if mode in ['training', 'caching']:
            self.hidden_cache = outputs.hidden_states
        elif mode == 'generating':
            pass
            

        # #     # Insert custom cache into outputs for the next generate step via _update_model_kwargs_for_generation
        # if point_cache is not None:
        #     # ModelOutput can accept dynamic attributes
        #     outputs.point_cache = point_cache

        return outputs

    def prepare_inputs_for_generation(self,
                                      *args,
                                      cache_position=None,
                                      point_embeds=None,
                                      **kwargs):
        model_inputs = super().prepare_inputs_for_generation(*args, cache_position=cache_position, **kwargs)

        model_inputs.update({
            'point_embeds': point_embeds,})

        return model_inputs


# set the patched model to a vision model
MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES[PointQwen2_5_VLConfig.model_type] = 'PointQwen2_5_VLForConditionalGeneration'

AutoConfig.register(PointQwen2_5_VLConfig.model_type, PointQwen2_5_VLConfig)
AutoModel.register(PointQwen2_5_VLConfig, PointQwen2_5_VLForConditionalGeneration)
AutoProcessor.register(PointQwen2_5_VLConfig, PatchedQwen2_5_VLProcessor)
