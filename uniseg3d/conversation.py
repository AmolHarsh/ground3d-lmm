# Copyright (c) 2025 Ye Liu. Licensed under the BSD-3-Clause License.

from dataclasses import dataclass
from typing import List


@dataclass
class Conversation:
    style: str
    system: str
    roles: List[str]
    seps: List[str]
    messages: List[str]

    def append_message(self, role, msg):
        self.messages.append([role, msg])

    def clear(self):
        self.messages = []

    def get_prompt(self):
        assert self.style in ('chatml', )

        prompt = self.system + self.seps[0] if self.system is not None else ''

        for i, (role, msg) in enumerate(self.messages):
            prompt += role
            sep = self.seps[i % 2]
            if msg is not None:
                prompt += msg
                if not prompt.endswith(sep):
                    prompt += sep

        prompt = prompt.lstrip('\n')
        return prompt


def get_conv(conv_type, qwen_model='qwen2_5_vl'):
    """Get conversation template based on conversation type and model.
    
    Args:
        conv_type (str): Conversation type, currently only 'chatml' is supported.
        qwen_model (str): Model type, 'qwen2_5_vl' or 'qwen3_vl'.
    
    Returns:
        Conversation: Configured conversation template.
    
    Raises:
        ValueError: If conv_type or qwen_model is not supported.
    """
    if conv_type != 'chatml':
        raise ValueError(f'unknown conversation type: {conv_type}')
    
    # Common chatml configuration
    roles = ('\n<|im_start|>user\n', '\n<|im_start|>assistant\n')
    seps = ('<|im_end|>', '<|im_end|>')
    
    if qwen_model == 'qwen2_5_vl':
        system = '<|im_start|>system\nYou are a helpful assistant.'
    elif qwen_model == 'qwen3_vl':
        system = None
    else:
        raise ValueError(f'unknown qwen model: {qwen_model}')
    
    conv = Conversation(
        style='chatml',
        system=system,
        roles=roles,
        seps=seps,
        messages=[])
    
    return conv
