import torch
import torch.nn as nn
import torch.distributed as dist

from mmengine.model import BaseModule
from mmdet3d.registry import MODELS
from mmengine.dist import get_dist_info
import torch.nn.functional as F


from transformers import (AutoConfig, AutoModel, AutoProcessor, Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration,
                          Qwen2_5_VLModel, Qwen2_5_VLProcessor, Qwen2_5_VLTextModel, AutoTokenizer, Qwen3VLProcessor)
from peft import LoraConfig, PeftModel, get_peft_model
from .prompt_template import PART_GLAMM_INSTRUCTION, SEG_TOKEN, POINT_TOKEN, PART_GLAMM_INSTRUCTION_IMG
import json

from .point_qwen2_5_vl import PointQwen2_5_VLForConditionalGeneration, PointQwen2_5_VLConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2RMSNorm

from .point_qwen3_vl import PointQwen3VLForConditionalGeneration, PointQwen3VLConfig, PatchedQwen3VLProcessor
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRMSNorm

from qwen_vl_utils import process_vision_info

from .conversation import get_conv
from mmengine.logging import MMLogger
from mmengine.dist import is_main_process
import os

IGNORE_INDEX = -100


def preprocess_chatml(input_ids, text, tokenizer, qwen_model='qwen3_vl'):
    """Build training labels from a ChatML conversation.

    Loss is computed **only on the assistant replies**: every other span
    (system prompt, user turns, role headers) is masked with ``IGNORE_INDEX``,
    as is the trailing incomplete assistant turn (the generation prompt).

    Args:
        input_ids (torch.Tensor): ``(1, L)`` token ids of the full conversation.
        text (str): the same conversation as a string (used to split rounds).
        tokenizer: the LMM tokenizer (``self.processor.tokenizer``).
        qwen_model (str): ``'qwen2_5_vl'`` or ``'qwen3_vl'`` — selects the
            ChatML system prompt via :func:`~uniseg3d.conversation.get_conv`.

    Returns:
        torch.Tensor: labels shaped like ``input_ids``, masked with ``IGNORE_INDEX``.
    """
    conv = get_conv('chatml', qwen_model=qwen_model)
    rounds = [m + conv.seps[0] for m in text.split(conv.seps[0])]

    # the final chunk is the incomplete assistant turn -> never supervised
    last_invalide_text = rounds[-1]
    rounds = rounds[:-1]

    if conv.system is None:
        rounds = [''.join(rounds[i:i + 2]) for i in range(0, len(rounds), 2)]
    else:
        rounds = [''.join(rounds[:3])] + [''.join(rounds[i:i + 2]) for i in range(3, len(rounds), 2)]

    labels = input_ids.clone()

    sep = conv.seps[0] + conv.roles[1]
    cur_len = 0

    for rou in rounds:
        if len(rou) == 0:
            break

        # mask everything up to and including the assistant header
        ins = sep.join(rou.split(sep)[:-1]) + sep
        rou_len = tokenizer(rou, return_length=True).length[0]
        ins_len = tokenizer(ins, return_length=True).length[0]
        labels[:, cur_len:cur_len + ins_len] = IGNORE_INDEX
        cur_len += rou_len

    if (labels == IGNORE_INDEX).sum() == labels.size(1):
        raise ValueError('No valid labels found')

    last_invalide = tokenizer(last_invalide_text, return_length=True).length[0]
    labels[:, -last_invalide + 1:] = IGNORE_INDEX

    return labels


class ConvSetAggregator(nn.Module):
    """Aggregate variable-length point features into a fixed number of tokens
    using 1D convolution across the set dimension followed by adaptive pooling.

    Args:
        in_channels (int): Feature dimension per point.
        out_tokens (int): Number of aggregated tokens.
        kernel_size (int): Convolution kernel size for the first layer.
        hidden_channels (int | None): Hidden channels; defaults to in_channels.
    """
    def __init__(self, in_channels, out_tokens, kernel_size=5, hidden_channels=None):
        super().__init__()
        hidden = in_channels if hidden_channels is None else hidden_channels
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=kernel_size, padding=padding),
            nn.GELU(),
            nn.Conv1d(hidden, in_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(out_tokens)

    def forward(self, x):
        # x: (N, D)
        if x.dim() != 2:
            raise ValueError('ConvSetAggregator expects 2D tensor of shape (N, D)')
        n, d = x.shape
        x1 = x.transpose(0, 1).unsqueeze(0)  # (1, D, N)
        y = self.net(x1)
        y = self.pool(y)  # (1, D, K)
        y = y.squeeze(0).transpose(0, 1)  # (K, D)
        return y

class PositionSoftClusterAggregator(nn.Module):
    """Position-aware soft clustering aggregator.

    Aggregates features into K tokens using soft assignment based on
    Euclidean distance between normalized point positions and learnable
    prototype centers. Permutation-invariant and driven by spatial layout.

    Args:
        in_channels (int): Feature dimension per point.
        out_tokens (int): Number of aggregated tokens (cluster centers).
        pos_dim (int): Dimensionality of positions (default 3).
        temperature (float): Softmax temperature over negative distances.
    """
    def __init__(self, in_channels, out_tokens, pos_dim=3, temperature=0.5):
        super().__init__()
        self.out_tokens = out_tokens
        self.temperature = temperature
        self.register_parameter(
            'centers', nn.Parameter(torch.randn(out_tokens, pos_dim)))

    def forward(self, feats, pos):
        # feats: (N, D), pos: (N, P)
        if feats.dim() != 2 or pos.dim() != 2:
            raise ValueError('PositionSoftClusterAggregator expects 2D tensors (N, D) and (N, P)')
        if feats.shape[0] != pos.shape[0]:
            raise ValueError('feats and pos must have same first dimension N')
        # normalize positions per-sample
        p = pos.float()
        p = p - p.mean(dim=0, keepdim=True)
        std = p.std(dim=0, keepdim=True) + 1e-6
        p = p / std
        # distances to centers: (N, K)
        diff = p.unsqueeze(1) - self.centers.unsqueeze(0)
        dist = (diff ** 2).sum(-1)
        assign = torch.softmax(-dist / max(self.temperature, 1e-6), dim=1)  # (N, K)
        # aggregate features: (K, D)
        num = assign.sum(dim=0, keepdim=True).transpose(0, 1) + 1e-6  # (K, 1)
        agg = assign.transpose(0, 1) @ feats  # (K, D)
        agg = agg / num
        return agg


class FPSAggregator(nn.Module):
    """Farthest Point Sampling (FPS) aggregator.
    
    Samples a fixed number of points using FPS algorithm based on feature distances.
    
    Args:
        in_channels (int): Feature dimension per point (not used, kept for compatibility).
        out_tokens (int): Number of sampled points.
    """
    def __init__(self, in_channels, out_tokens):
        super().__init__()
        self.out_tokens = out_tokens
    
    def forward(self, x):
        # x: (N, D)
        if x.dim() != 2:
            raise ValueError('FPSAggregator expects 2D tensor of shape (N, D)')
        n, d = x.shape
        
        if n <= self.out_tokens:
            return x
        
        # FPS sampling based on feature L2 distances
        indices = self.fps_sample(x, self.out_tokens)
        return x[indices]
    
    @staticmethod
    def fps_sample(points, num_samples):
        """Farthest Point Sampling.
        
        Args:
            points: (N, D) tensor
            num_samples: number of samples to select
            
        Returns:
            indices: (num_samples,) tensor of selected indices
        """
        n = points.shape[0]
        device = points.device
        
        # Start with a random point
        indices = torch.zeros(num_samples, dtype=torch.long, device=device)
        distances = torch.ones(n, device=device) * float('inf')
        
        # Compute pairwise distances
        dist_matrix = torch.cdist(points, points)  # (N, N)
        
        # First point: random
        current = torch.randint(0, n, (1,), device=device).item()
        indices[0] = current
        
        for i in range(1, num_samples):
            # Update distances to the current farthest point
            dist_to_current = dist_matrix[current]  # (N,)
            distances = torch.minimum(distances, dist_to_current)
            
            # Select the farthest point
            current = distances.argmax().item()
            indices[i] = current
        
        return indices


class FPSPositionAggregator(nn.Module):
    """FPS aggregator based on position distances.
    
    Samples a fixed number of points using FPS algorithm based on position distances.
    
    Args:
        in_channels (int): Feature dimension per point (not used, kept for compatibility).
        out_tokens (int): Number of sampled points.
        pos_dim (int): Dimensionality of positions (not used, kept for compatibility).
        temperature (float): Not used, kept for compatibility.
    """
    def __init__(self, in_channels, out_tokens, pos_dim=3, temperature=0.5):
        super().__init__()
        self.out_tokens = out_tokens
    
    def forward(self, feats, pos):
        # feats: (N, D), pos: (N, P)
        if feats.dim() != 2 or pos.dim() != 2:
            raise ValueError('FPSPositionAggregator expects 2D tensors (N, D) and (N, P)')
        if feats.shape[0] != pos.shape[0]:
            raise ValueError('feats and pos must have same first dimension N')
        
        n = feats.shape[0]
        if n <= self.out_tokens:
            return feats
        
        # FPS sampling based on position distances
        indices = self.fps_sample(pos, self.out_tokens)
        return feats[indices]
    
    @staticmethod
    def fps_sample(positions, num_samples):
        """Farthest Point Sampling based on positions.
        
        Args:
            positions: (N, P) tensor of positions
            num_samples: number of samples to select
            
        Returns:
            indices: (num_samples,) tensor of selected indices
        """
        n = positions.shape[0]
        device = positions.device
        
        # Start with a random point
        indices = torch.zeros(num_samples, dtype=torch.long, device=device)
        distances = torch.ones(n, device=device) * float('inf')
        
        # Compute pairwise distances
        dist_matrix = torch.cdist(positions, positions)  # (N, N)
        
        # First point: random
        current = torch.randint(0, n, (1,), device=device).item()
        indices[0] = current
        
        for i in range(1, num_samples):
            # Update distances to the current farthest point
            dist_to_current = dist_matrix[current]  # (N,)
            distances = torch.minimum(distances, dist_to_current)
            
            # Select the farthest point
            current = distances.argmax().item()
            indices[i] = current
        
        return indices


def mask_pool(x, mask):
    """
    Args:
        x: [D, M]
        mask: [N, M]
    """
    with torch.no_grad():
        mask = mask.detach()
        mask = (mask > 0).to(mask.dtype)
        denorm = mask.sum(dim=(-1), keepdim=True) + 1e-8

    mask_pooled_x = torch.einsum(
        "dm,nm->nd",
        x,
        mask / denorm,
    )
    return mask_pooled_x


class CrossAttentionLayer(BaseModule):
    """Cross attention layer.

    Args:
        d_model (int): Model dimension.
        num_heads (int): Number of heads.
        dropout (float): Dropout rate.
    """

    def __init__(self, d_model, num_heads, dropout, fix=False):
        super().__init__()
        self.fix = fix
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        # todo: why BaseModule doesn't call it without us?
        self.init_weights()

    def init_weights(self):
        """Init weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, sources, queries, attn_masks=None):
        """Forward pass.

        Args:
            sources (List[Tensor]): of len batch_size,
                each of shape (n_points_i, d_model).
            queries (List[Tensor]): of len batch_size,
                each of shape(n_queries_i, d_model).
            attn_masks (List[Tensor] or None): of len batch_size,
                each of shape (n_queries, n_points).
        
        Return:
            List[Tensor]: Queries of len batch_size,
                each of shape(n_queries_i, d_model).
        """
        outputs = []
        for i in range(len(sources)):
            k = v = sources[i]
            attn_mask = attn_masks[i] if attn_masks is not None else None
            output, _ = self.attn(queries[i], k, v, attn_mask=attn_mask)
            if self.fix:
                output = self.dropout(output)
            output = output + queries[i]
            if self.fix:
                output = self.norm(output)
            outputs.append(output)
        return outputs


class SelfAttentionLayer(BaseModule):
    """Self attention layer.

    Args:
        d_model (int): Model dimension.
        num_heads (int): Number of heads.
        dropout (float): Dropout rate.
    """

    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, interaction_masks=None):
        """Forward pass.

        Args:
            x (List[Tensor]): Queries of len batch_size,
                each of shape(n_queries_i, d_model).
        
        Returns:
            List[Tensor]: Queries of len batch_size,
                each of shape(n_queries_i, d_model).
        """
        out = []


        

        for i, y in enumerate(x):
            nan_mask = torch.isnan(y).any(dim=-1)  # shape: (num_queries,)
        
            # replace NaN with 0
            y_safe = y.clone()
            y_safe[nan_mask] = 0.0
            if not interaction_masks:
                z, _ = self.attn(y_safe, y_safe, y_safe)
            else:
                z, _ = self.attn(y_safe, y_safe, y_safe, attn_mask=interaction_masks[i])
            z = self.dropout(z) + y
            z = self.norm(z)
            out.append(z)
        return out


class FFN(BaseModule):
    """Feed forward network.

    Args:
        d_model (int): Model dimension.
        hidden_dim (int): Hidden dimension.
        dropout (float): Dropout rate.
        activation_fn (str): 'relu' or 'gelu'.
    """

    def __init__(self, d_model, hidden_dim, dropout, activation_fn):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU() if activation_fn == 'relu' else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """Forward pass.

        Args:
            x (List[Tensor]): Queries of len batch_size,
                each of shape(n_queries_i, d_model).
        
        Returns:
            List[Tensor]: Queries of len batch_size,
                each of shape(n_queries_i, d_model).
        """
        out = []
        for y in x:
            z = self.net(y)
            z = z + y
            z = self.norm(z)
            out.append(z)
        return out

@MODELS.register_module()
class QueryDecoder(BaseModule):
    """Query decoder.

    Args:
        num_layers (int): Number of transformer layers.
        num_instance_queries (int): Number of instance queries.
        num_semantic_queries (int): Number of semantic queries.
        num_classes (int): Number of classes.
        in_channels (int): Number of input channels.
        d_model (int): Number of channels for model layers.
        num_heads (int): Number of head in attention layer.
        hidden_dim (int): Dimension of attention layer.
        dropout (float): Dropout rate for transformer layer.
        activation_fn (str): 'relu' of 'gelu'.
        iter_pred (bool): Whether to predict iteratively.
        attn_mask (bool): Whether to use mask attention.
        pos_enc_flag (bool): Whether to use positional enconding.
    """

    def __init__(self, num_layers, num_instance_queries, num_semantic_queries,
                 num_classes, in_channels, d_model, num_heads, hidden_dim,
                 dropout, activation_fn, iter_pred, attn_mask, fix_attention,
                 objectness_flag, sphere_cls, **kwargs):
        super().__init__()
        self.objectness_flag = objectness_flag
        self.sphere_cls = sphere_cls
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, d_model), nn.LayerNorm(d_model), nn.ReLU())
        self.num_queries = num_instance_queries + num_semantic_queries
        if num_instance_queries + num_semantic_queries > 0:
            self.query = nn.Embedding(num_instance_queries + num_semantic_queries, d_model)
        if num_instance_queries == 0:
            self.query_proj = nn.Sequential(
                nn.Linear(in_channels, d_model), nn.ReLU(),
                nn.Linear(d_model, d_model))
        self.cross_attn_layers = nn.ModuleList([])
        self.self_attn_layers = nn.ModuleList([])
        self.ffn_layers = nn.ModuleList([])
        for i in range(num_layers):
            self.cross_attn_layers.append(
                CrossAttentionLayer(
                    d_model, num_heads, dropout, fix_attention))
            self.self_attn_layers.append(
                SelfAttentionLayer(d_model, num_heads, dropout))
            self.ffn_layers.append(
                FFN(d_model, hidden_dim, dropout, activation_fn))
        self.out_norm = nn.LayerNorm(d_model)
        if not self.sphere_cls:
            self.out_cls = nn.Sequential(
                nn.Linear(d_model, d_model), nn.ReLU(),
                nn.Linear(d_model, num_classes + 1))
        if objectness_flag:
            self.out_score = nn.Sequential(
                nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))
        self.x_mask = nn.Sequential(
            nn.Linear(in_channels, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model))
        self.iter_pred = iter_pred
        self.attn_mask = attn_mask
    
    def _get_queries(self, queries=None, batch_size=None):
        """Get query tensor.

        Args:
            queries (List[Tensor], optional): of len batch_size,
                each of shape (n_queries_i, in_channels).
            batch_size (int, optional): batch size.
        
        Returns:
            List[Tensor]: of len batch_size, each of shape
                (n_queries_i, d_model).
        """
        if batch_size is None:
            batch_size = len(queries)
        
        result_queries = []
        for i in range(batch_size):
            result_query = []
            if hasattr(self, 'query'):
                result_query.append(self.query.weight)
            if queries is not None:
                result_query.append(self.query_proj(queries[i]))
            result_queries.append(torch.cat(result_query))
        return result_queries

    def _forward_head(self, queries, mask_feats):
        """Prediction head forward.

        Args:
            queries (List[Tensor] | Tensor): List of len batch_size,
                each of shape (n_queries_i, d_model). Or tensor of
                shape (batch_size, n_queries, d_model).
            mask_feats (List[Tensor]): of len batch_size,
                each of shape (n_points_i, d_model).

        Returns:
            Tuple:
                List[Tensor]: Classification predictions of len batch_size,
                    each of shape (n_queries_i, n_classes + 1).
                List[Tensor]: Confidence scores of len batch_size,
                    each of shape (n_queries_i, 1).
                List[Tensor]: Predicted masks of len batch_size,
                    each of shape (n_queries_i, n_points_i).
                List[Tensor] or None: Attention masks of len batch_size,
                    each of shape (n_queries_i, n_points_i).
        """
        cls_preds, pred_scores, pred_masks, attn_masks = [], [], [], []
        for i in range(len(queries)):
            norm_query = self.out_norm(queries[i])
            cls_preds.append(self.out_cls(norm_query))
            pred_score = self.out_score(norm_query) if self.objectness_flag \
                else None
            pred_scores.append(pred_score)
            pred_mask = torch.einsum('nd,md->nm', norm_query, mask_feats[i])
            if self.attn_mask:
                attn_mask = (pred_mask.sigmoid() < 0.5).bool()
                attn_mask[torch.where(
                    attn_mask.sum(-1) == attn_mask.shape[-1])] = False
                attn_mask = attn_mask.detach()
                attn_masks.append(attn_mask)
            pred_masks.append(pred_mask)
        attn_masks = attn_masks if self.attn_mask else None
        return cls_preds, pred_scores, pred_masks, attn_masks

    def forward_simple(self, x, queries):
        """Simple forward pass.
        
        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, in_channels).
            queries (List[Tensor], optional): of len batch_size, each of shape
                (n_points_i, in_channles).
        
        Returns:
            Dict: with labels, masks, and scores.
        """
        inst_feats = [self.input_proj(y) for y in x]
        mask_feats = [self.x_mask(y) for y in x]
        queries = self._get_queries(queries, len(x))
        for i in range(len(self.cross_attn_layers)):
            queries = self.cross_attn_layers[i](inst_feats, queries)
            queries = self.self_attn_layers[i](queries)
            queries = self.ffn_layers[i](queries)
        cls_preds, pred_scores, pred_masks, _ = self._forward_head(
            queries, mask_feats)
        return dict(
            cls_preds=cls_preds,
            masks=pred_masks,
            scores=pred_scores)

    def forward_iter_pred(self, x, queries):
        """Iterative forward pass.
        
        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, in_channels).
            queries (List[Tensor], optional): of len batch_size, each of shape
                (n_points_i, in_channles).
        
        Returns:
            Dict: with labels, masks, scores, and aux_outputs.
        """
        cls_preds, pred_scores, pred_masks = [], [], []
        inst_feats = [self.input_proj(y) for y in x]
        mask_feats = [self.x_mask(y) for y in x]
        queries = self._get_queries(queries, len(x))
        cls_pred, pred_score, pred_mask, attn_mask = self._forward_head(
            queries, mask_feats)
        cls_preds.append(cls_pred)
        pred_scores.append(pred_score)
        pred_masks.append(pred_mask)
        for i in range(len(self.cross_attn_layers)):
            queries = self.cross_attn_layers[i](inst_feats, queries, attn_mask)
            queries = self.self_attn_layers[i](queries)
            queries = self.ffn_layers[i](queries)
            cls_pred, pred_score, pred_mask, attn_mask = self._forward_head(
                queries, mask_feats)
            cls_preds.append(cls_pred)
            pred_scores.append(pred_score)
            pred_masks.append(pred_mask)

        aux_outputs = [
            {'cls_preds': cls_pred, 'masks': masks, 'scores': scores}
            for cls_pred, scores, masks in zip(
                cls_preds[:-1], pred_scores[:-1], pred_masks[:-1])]
        return dict(
            cls_preds=cls_preds[-1],
            masks=pred_masks[-1],
            scores=pred_scores[-1],
            aux_outputs=aux_outputs)

    def forward(self, x, queries=None, interaction_masks=None):
        """Forward pass.
        
        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, in_channels).
            queries (List[Tensor], optional): of len batch_size, each of shape
                (n_points_i, in_channles).
        
        Returns:
            Dict: with labels, masks, scores, and possibly aux_outputs.
        """
        if self.iter_pred:
            return self.forward_iter_pred(x, queries, interaction_masks=interaction_masks)
        else:
            return self.forward_simple(x, queries)


@MODELS.register_module()
class UnifiedQueryDecoder(QueryDecoder):
    """We simply add semantic prediction for each instance query.
    """
    def __init__(self, num_instance_classes, num_semantic_classes,
                 d_model, num_semantic_linears, sphere_cls, vocabulary_cls_embedding_path, **kwargs):
        super().__init__(
            num_classes=num_instance_classes, d_model=d_model, sphere_cls= sphere_cls, **kwargs)
        if not self.sphere_cls:
            assert num_semantic_linears in [1, 2]
            if num_semantic_linears == 2:
                self.out_sem = nn.Sequential(
                    nn.Linear(d_model, d_model), nn.ReLU(),
                    nn.Linear(d_model, num_semantic_classes + 1))
            else:
                self.out_sem = nn.Linear(d_model, num_semantic_classes + 1)
            
        if self.sphere_cls:
            rank, world_size = get_dist_info()
            cls_embed = torch.load(vocabulary_cls_embedding_path)
            
            _dim = cls_embed.size(2)
            _prototypes = cls_embed.size(1)

            if rank == 0:
                back_token = torch.zeros(1, _dim, dtype=torch.float32, device='cuda')
            else:
                back_token = torch.empty(1, _dim, dtype=torch.float32, device='cuda')
            if world_size > 1:
                dist.broadcast(back_token, src=0)
            back_token = back_token.to(device='cpu')
            cls_embed = torch.cat([
                cls_embed, back_token.repeat(_prototypes, 1)[None]
            ], dim=0)
            self.register_buffer('cls_embed', cls_embed.permute(2, 0, 1).contiguous(), persistent=False)
            
            cls_embed_dim = self.cls_embed.size(0)
            self.cls_proj = nn.Sequential(
                nn.Linear(d_model, d_model), nn.ReLU(inplace=True),
                nn.Linear(d_model, d_model), nn.ReLU(inplace=True),
                nn.Linear(d_model, cls_embed_dim))
            
            self.cls_proj_sem = nn.Sequential(
                nn.Linear(d_model, d_model), nn.ReLU(inplace=True),
                nn.Linear(d_model, d_model), nn.ReLU(inplace=True),
                nn.Linear(d_model, cls_embed_dim))
            
            logit_scale = torch.tensor(4.6052, dtype=torch.float32)
            self.register_buffer('logit_scale', logit_scale, persistent=False)
            
            # Mask Pooling
            self.mask_pooling = mask_pool
            self.mask_pooling_proj = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model)
            )
            
            self.mask_pooling_proj_sem = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model)
            )
    def forward_logit(self, cls_embd, sem=False):
        if sem:
            cls_pred = torch.einsum('nd,dkp->nkp', F.normalize(cls_embd, dim=-1), self.cls_embed)
        else:
            cls_pred = torch.einsum('nd,dkp->nkp', F.normalize(cls_embd, dim=-1), self.cls_embed[:, 2:, :])
        cls_pred = cls_pred.max(-1).values
        cls_pred = self.logit_scale.exp() * cls_pred
        return cls_pred
    
    def _forward_head(self, queries, mask_feats, last_flag):
        """Prediction head forward.

        Args:
            queries (List[Tensor] | Tensor): List of len batch_size,
                each of shape (n_queries_i, d_model). Or tensor of
                shape (batch_size, n_queries, d_model).
            mask_feats (List[Tensor]): of len batch_size,
                each of shape (n_points_i, d_model).

        Returns:
            Tuple:
                List[Tensor]: Classification predictions of len batch_size,
                    each of shape (n_queries_i, n_instance_classes + 1).
                List[Tensor] or None: Semantic predictions of len batch_size,
                    each of shape (n_queries_i, n_semantic_classes + 1).
                List[Tensor]: Confidence scores of len batch_size,
                    each of shape (n_queries_i, 1).
                List[Tensor]: Predicted masks of len batch_size,
                    each of shape (n_queries_i, n_points_i).
                List[Tensor] or None: Attention masks of len batch_size,
                    each of shape (n_queries_i, n_points_i).
        """
        cls_preds, sem_preds, pred_scores, pred_masks, attn_masks = \
            [], [], [], [], []
        for i in range(len(queries)):
            norm_query = self.out_norm(queries[i])
            if not self.sphere_cls:
                cls_preds.append(self.out_cls(norm_query))
                if last_flag:
                    sem_preds.append(self.out_sem(norm_query))
     
            pred_score = self.out_score(norm_query) if self.objectness_flag \
                else None
            pred_scores.append(pred_score)
            pred_mask = torch.einsum('nd,md->nm', norm_query, mask_feats[i])
            
            if self.sphere_cls:
                maskpool_ = self.mask_pooling(x=mask_feats[i].T, mask=pred_mask.detach())
                
                maskpool_embd = self.mask_pooling_proj(maskpool_) 
                cls_embd = self.cls_proj(maskpool_embd + norm_query) 
                cls_pred = self.forward_logit(cls_embd) 
                cls_preds.append(cls_pred)
                if last_flag:
                    norm_query_sem = norm_query.clone()
                    maskpool_embd_sem  = self.mask_pooling_proj_sem(maskpool_)
                    cls_embd_sem = self.cls_proj_sem(maskpool_embd_sem + norm_query_sem)
                    cls_pred_sem = self.forward_logit(cls_embd_sem, sem=True)
                    sem_preds.append(cls_pred_sem)
                
            if self.attn_mask:
                attn_mask = (pred_mask.sigmoid() < 0.5).bool()
                attn_mask[torch.where(
                    attn_mask.sum(-1) == attn_mask.shape[-1])] = False
                attn_mask = attn_mask.detach()
                attn_masks.append(attn_mask)
            pred_masks.append(pred_mask)
        attn_masks = attn_masks if self.attn_mask else None
        sem_preds = sem_preds if last_flag else None
        return cls_preds, sem_preds, pred_scores, pred_masks, attn_masks

    def forward_simple(self, x, queries):
        """Simple forward pass.
        
        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, in_channels).
            queries (List[Tensor], optional): of len batch_size, each of shape
                (n_points_i, in_channles).
        
        Returns:
            Dict: with instance scores, semantic scores, masks, and scores.
        """
        inst_feats = [self.input_proj(y) for y in x]
        mask_feats = [self.x_mask(y) for y in x]
        queries = self._get_queries(queries, len(x))
        for i in range(len(self.cross_attn_layers)):
            queries = self.cross_attn_layers[i](inst_feats, queries)
            queries = self.self_attn_layers[i](queries)
            queries = self.ffn_layers[i](queries)
        cls_preds, sem_preds, pred_scores, pred_masks, _= self._forward_head(
            queries, mask_feats, last_flag=True)
        return dict(
            cls_preds=cls_preds,
            sem_preds=sem_preds,
            masks=pred_masks,
            scores=pred_scores,)

    def forward_iter_pred(self, x, queries, interaction_masks=None):
        """Iterative forward pass.
        
        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, in_channels).
            queries (List[Tensor], optional): of len batch_size, each of shape
                (n_points_i, in_channles).
        
        Returns:
            Dict: with instance scores, semantic scores, masks, scores,
                and aux_outputs.
        """
        cls_preds, sem_preds, pred_scores, pred_masks, contras_embeds = [], [], [], [], []
        inst_feats = [self.input_proj(y) for y in x]
        mask_feats = [self.x_mask(y) for y in x]
        queries = self._get_queries(queries, len(x))
        cls_pred, sem_pred, pred_score, pred_mask, attn_mask= \
            self._forward_head(queries, mask_feats, last_flag=False)
        cls_preds.append(cls_pred)
        sem_preds.append(sem_pred)
        pred_scores.append(pred_score)
        pred_masks.append(pred_mask)
        contras_embeds.append([self.out_norm(queries[i].clone()) for i in range(len(queries))])
        
        for i in range(len(self.cross_attn_layers)):
            queries = self.cross_attn_layers[i](inst_feats, queries, attn_mask)
            queries = self.self_attn_layers[i](queries, interaction_masks=interaction_masks)
            queries = self.ffn_layers[i](queries)
            last_flag = i == len(self.cross_attn_layers) - 1
            cls_pred, sem_pred, pred_score, pred_mask, attn_mask = \
                self._forward_head(queries, mask_feats, last_flag)
            cls_preds.append(cls_pred)
            sem_preds.append(sem_pred)
            pred_scores.append(pred_score)
            pred_masks.append(pred_mask)
            contras_embeds.append([self.out_norm(queries[i].clone()) for i in range(len(queries))])

        aux_outputs = [
            dict(
                cls_preds=cls_pred,
                sem_preds=sem_pred,
                masks=masks,
                scores=scores,
                contras_embeds=contras_embeds)
            for cls_pred, sem_pred, scores, masks, contras_embeds in zip(
                cls_preds[:-1], sem_preds[:-1],
                pred_scores[:-1], pred_masks[:-1], contras_embeds[:-1])]
        return dict(
            cls_preds=cls_preds[-1],
            sem_preds=sem_preds[-1],
            masks=pred_masks[-1],
            scores=pred_scores[-1],
            contras_embeds=contras_embeds[-1],
            aux_outputs=aux_outputs)




@MODELS.register_module()
class Grounded_Decoder_Joint(QueryDecoder):
    """We simply add semantic prediction for each instance query.
    """
    def __init__(self, in_channels, num_instance_classes, num_semantic_classes,
                 d_model, num_semantic_linears, sphere_cls, vocabulary_cls_embedding_path, 
                 target_classes = None,
                reason_file=None,
                qwen_model_path='Qwen/Qwen-2.5-VL',
                lora_type='qkvo_all',
                lora_r=128,
                lora_alpha=256,
                lora_dropout=0.1,
                lora_bias=None,
                base_model='qwen2_5_vl',
                max_point_tokens=2500,
                save_pred_qa_dir=None,
                scannet_image_dir=None,
                scannetpp_image_dir=None,
                 **kwargs):
        super().__init__(
            in_channels=in_channels, num_classes=num_instance_classes, d_model=d_model, sphere_cls= sphere_cls, 
            **kwargs)

        self.save_pred_qa_dir = save_pred_qa_dir
        self.scannet_image_dir = scannet_image_dir
        self.scannetpp_image_dir = scannetpp_image_dir
        if save_pred_qa_dir is not None:
            if not os.path.exists(save_pred_qa_dir):
                os.makedirs(save_pred_qa_dir, exist_ok=True)
        else:
            self.save_pred_qa_dir = None


        self._init_reason_model(
            qwen_model=qwen_model_path,
            lora_type=lora_type,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_bias=lora_bias,
            base_model=base_model)
        
        if base_model == 'qwen2_5_vl':
            # self.token2point = nn.Sequential(
            #     Qwen2RMSNorm(self.reason_mode_config.hidden_size), nn.GELU(),
            #     nn.Linear(self.reason_mode_config.hidden_size, d_model))
                
            self.point2token = nn.Sequential(
                Qwen2RMSNorm(d_model), nn.GELU(),
                nn.Linear(d_model, self.reason_mode_config.hidden_size))
        elif base_model == 'qwen3_vl':
            # self.token2point = nn.Sequential(
            #     Qwen3VLTextRMSNorm(self.reason_mode_config.text_config.hidden_size), 
            #     nn.Linear(self.reason_mode_config.text_config.hidden_size, self.reason_mode_config.text_config.hidden_size),
            #     nn.GELU(), nn.Linear(self.reason_mode_config.text_config.hidden_size, d_model))
                    
            self.point2token = nn.Sequential(
                Qwen3VLTextRMSNorm(d_model), nn.Linear(d_model, d_model),
                nn.GELU(), nn.Linear(d_model, self.reason_mode_config.text_config.hidden_size))
        else:
            raise ValueError(f'unknown base model: {base_model}')


        self.query_proj = nn.Sequential(
                Qwen3VLTextRMSNorm(self.reason_mode_config.text_config.hidden_size),
                nn.Linear(self.reason_mode_config.text_config.hidden_size, d_model), nn.GELU(),
                nn.Linear(d_model, d_model)) 
        self.base_model = base_model

        self.d_model = d_model

        # Learned aggregator to cap point tokens for reasoning
        self.max_point_tokens = max_point_tokens
        self.point_aggregator = FPSAggregator(in_channels=d_model, out_tokens=self.max_point_tokens)
        self.pos_aggregator = FPSPositionAggregator(in_channels=d_model, out_tokens=self.max_point_tokens, pos_dim=3, temperature=0.5)
        self.target_classes = target_classes


    def _init_reason_model(self, qwen_model, lora_type, lora_r, lora_alpha, lora_dropout, lora_bias, base_model='qwen2_5_vl'):

        if base_model == 'qwen2_5_vl':
            self.reason_model = PointQwen2_5_VLForConditionalGeneration.from_pretrained(
                qwen_model, 
                torch_dtype=torch.bfloat16, # Removed
                # attn_implementation=attn_implementation
            )
            self.reason_mode_config = PointQwen2_5_VLConfig.from_pretrained(qwen_model)
            self.processor = Qwen2_5_VLProcessor.from_pretrained(qwen_model, use_fast=True, do_resize=False)
            vocab_size = self.reason_mode_config.vocab_size
        elif base_model == 'qwen3_vl':
            self.reason_model = PointQwen3VLForConditionalGeneration.from_pretrained(qwen_model, torch_dtype=torch.bfloat16, trust_remote_code=True)
            self.reason_mode_config = PointQwen3VLConfig.from_pretrained(qwen_model)
            self.processor = PatchedQwen3VLProcessor.from_pretrained(qwen_model,  trust_remote_code=True)
            vocab_size = self.reason_mode_config.text_config.vocab_size
        else:
            raise ValueError(f'unknown base model: {base_model}')
        
        self.reason_model.requires_grad_(False)

        self.seg_tokens = SEG_TOKEN
        self.point_token = POINT_TOKEN[0]
        special_tokens = SEG_TOKEN + POINT_TOKEN

        new_tokens = self.processor.tokenizer.add_special_tokens(dict(additional_special_tokens=special_tokens))
        if new_tokens > 0 and len(self.processor.tokenizer) > vocab_size:
            MMLogger.get_current_instance().info(f'Expanding vocab size: {vocab_size} -> {len(self.processor.tokenizer)}')
            self.reason_model.resize_token_embeddings(len(self.processor.tokenizer))
            i_emb = self.reason_model.get_input_embeddings().weight.data
            o_emb = self.reason_model.get_output_embeddings().weight.data
            i_emb[-new_tokens:] = i_emb[:-new_tokens].mean(0, keepdim=True)
            o_emb[-new_tokens:] = o_emb[:-new_tokens].mean(0, keepdim=True)

        self.seg_tokens_ids= self.processor.tokenizer.convert_tokens_to_ids(self.seg_tokens)
        self.point_token_id = self.processor.tokenizer.convert_tokens_to_ids(self.point_token)

        self.reason_model.config.point_token_id = self.processor.tokenizer.convert_tokens_to_ids(self.point_token)

        target_modules = self.get_target_modules(self.reason_model, lora_type, base_model)
        MMLogger.get_current_instance().info(f'LoRA target modules: {target_modules}')
        lora_config = LoraConfig(
            task_type='CAUSAL_LM',
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=lora_bias,
            target_modules=target_modules)
        
        self.reason_model = get_peft_model(self.reason_model, lora_config)
        # self.reason_model.eval()
        for name, param in self.reason_model.named_parameters():
            if any(k in name for k in ('embed_tokens', 'lm_head')):
                param.requires_grad = True
            elif 'visual' in name:
                param.requires_grad = False
        

        
        total_params = sum(p.numel() for p in self.reason_model.parameters())
        learnable_params = sum(p.numel() for p in self.reason_model.parameters() if p.requires_grad)
        ratio = round(learnable_params / total_params * 100, 2) if total_params > 0 else 0
        MMLogger.get_current_instance().info(f'Total params: {total_params} Learnable params: {learnable_params} ({ratio}%)')

        return 

    def get_target_modules(self, model, lora_type, base_model):
        layer_type, modules = lora_type.split('_')
        assert layer_type in ('qkvo', 'linear') and modules in ('llm', 'visual', 'all')

        if base_model == 'qwen2_5_vl' or base_model == 'qwen3_vl':
            # all qkvo layers in the visual encoder and the llm
            qkvo_keys = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'attn.qkv', 'attn.proj']

            target_modules = []
            for n, m in model.named_modules():
                if modules == 'llm' and 'visual' in n:
                    continue
                if modules == 'visual' and 'visual' not in n:
                    continue
                if layer_type == 'qkvo' and not any(n.endswith(k) for k in qkvo_keys):
                    continue
                if n in target_modules:
                    continue
                target_modules.append(n)
        else:
            raise ValueError(f'unknown base model: {base_model}')

        return target_modules


    def get_query_tokens(self, hidden_states, input_ids, text, num_qa):
        """
        Extract query tokens from hidden_states, grouped by QA pair.

        Following preprocess_chatml's rounds splitting, determine the token range of each
        assistant reply, then extract embeddings between <p>...</p> within that range.

        Args:
            hidden_states: hidden states of the model's last layer, shape (1, seq_len, hidden_dim)
            input_ids: input token ids, shape (1, seq_len)
            text: full text after apply_chat_template
            num_qa: number of QA pairs

        Returns:
            list of query tokens, length num_qa, each element is a (num_queries, hidden_dim) tensor
        """
        hidden_dim = hidden_states.shape[-1]
        device = hidden_states.device
        tokenizer = self.processor.tokenizer

        # get ids of special tokens
        p_start_id = self.seg_tokens_ids[0]  # <p>
        p_end_id = self.seg_tokens_ids[1]    # </p>

        input_ids_flat = input_ids[0]  # (seq_len,)
        hidden_flat = hidden_states[0]  # (seq_len, hidden_dim)

        # find positions of all <p> and </p>
        p_start_positions = (input_ids_flat == p_start_id).nonzero(as_tuple=True)[0].tolist()
        p_end_positions = (input_ids_flat == p_end_id).nonzero(as_tuple=True)[0].tolist()

        # follow preprocess_chatml splitting
        conv = get_conv('chatml', qwen_model=self.base_model)
        rounds = [m + conv.seps[0] for m in text.split(conv.seps[0])]
        rounds = rounds[:-1]  # drop the final empty round

        # grouping: system + user + assistant as one group, then every two (user + assistant) as a group
        if conv.system is None:
            grouped_rounds = [''.join(rounds[i:i + 2]) for i in range(0, len(rounds), 2)]
        else:
            grouped_rounds = [''.join(rounds[:3])] + [''.join(rounds[i:i + 2]) for i in range(3, len(rounds), 2)]

        # separator: used to split instruction and assistant reply
        sep = conv.seps[0] + conv.roles[1]  # '<|im_end|>\n<|im_start|>assistant\n'

        all_query_tokens = []
        cur_len = 0

        for i, rou in enumerate(grouped_rounds):
            if i >= num_qa:
                break
            if len(rou) == 0:
                all_query_tokens.append(torch.zeros(1, hidden_dim, device=device))
                continue

            # compute the token length of this round
            rou_len = tokenizer(rou, return_length=True).length[0]

            # split instruction and assistant reply
            parts = rou.split(sep)
            if len(parts) < 2:
                # no assistant reply
                all_query_tokens.append(torch.zeros(1, hidden_dim, device=device))
                cur_len += rou_len
                continue

            ins = sep.join(parts[:-1]) + sep
            ins_len = tokenizer(ins, return_length=True).length[0]

            # token range of assistant reply
            assistant_start = cur_len + ins_len
            assistant_end = cur_len + rou_len

            # find <p>...</p> within this range
            qa_queries = []
            used_end_positions = set()

            for p_start in p_start_positions:
                if p_start < assistant_start or p_start >= assistant_end:
                    continue

                # find matching </p>
                for p_end in p_end_positions:
                    if p_end > p_start and p_end <= assistant_end and p_end not in used_end_positions:
                        used_end_positions.add(p_end)

                        # extract token embeddings between <p> and </p>
                        if p_end > p_start + 1:
                            between_embeddings = hidden_flat[p_start + 1 : p_end]
                            avg_embedding = between_embeddings.mean(dim=0, keepdim=True)
                            qa_queries.append(avg_embedding)
                        break

            if len(qa_queries) > 0:
                all_query_tokens.append(torch.cat(qa_queries[:2], dim=0)) # only use the first 2 query tokens
            else:
                # this QA has no <p>...</p>, return NaN
                all_query_tokens.append(torch.zeros(1, hidden_dim, device=device))

            cur_len += rou_len

        # if grouped_rounds count is less than num_qa, pad with NaN
        while len(all_query_tokens) < num_qa:
            all_query_tokens.append(torch.zeros(1, hidden_dim, device=device))
        all_query_tokens = torch.cat(all_query_tokens, dim=0)
        return all_query_tokens



    def get_query_tokens_id(self, hidden_states, input_ids, num_qa):
        """
        Extract query tokens from hidden_states, grouped by QA pair.

        Use the special token positions in input_ids directly to divide each assistant
        reply's range, then extract embeddings between <p>...</p> within that range.

        Difference from get_query_tokens: this function locates assistant ranges based on
        input_ids rather than text, so it is more accurate when extra image tokens, etc.,
        cause text and tokens to be misaligned.

        Args:
            hidden_states: hidden states of the model's last layer, shape (1, seq_len, hidden_dim)
            input_ids: input token ids, shape (1, seq_len)
            num_qa: number of QA pairs

        Returns:
            torch.Tensor: concatenated query tokens, shape (total_queries, hidden_dim)
                          same return format as get_query_tokens
        """
        hidden_dim = hidden_states.shape[-1]
        device = hidden_states.device
        tokenizer = self.processor.tokenizer

        # get ids of special tokens
        p_start_id = self.seg_tokens_ids[0]  # <p>
        p_end_id = self.seg_tokens_ids[1]    # </p>
        im_start_id = tokenizer.convert_tokens_to_ids('<|im_start|>')
        im_end_id = tokenizer.convert_tokens_to_ids('<|im_end|>')

        input_ids_flat = input_ids[0]  # (seq_len,)
        hidden_flat = hidden_states[0]  # (seq_len, hidden_dim)

        # find positions of all <p> and </p>
        p_start_positions = (input_ids_flat == p_start_id).nonzero(as_tuple=True)[0].tolist()
        p_end_positions = (input_ids_flat == p_end_id).nonzero(as_tuple=True)[0].tolist()

        # find positions of all <|im_start|> and <|im_end|>
        im_start_positions = (input_ids_flat == im_start_id).nonzero(as_tuple=True)[0].tolist()
        im_end_positions = (input_ids_flat == im_end_id).nonzero(as_tuple=True)[0].tolist()

        # identify assistant reply ranges
        # strategy: the position of <|im_start|> immediately followed by "assistant" token marks the start of assistant reply
        # the corresponding <|im_end|> marks the end of assistant reply
        assistant_token_id = tokenizer.convert_tokens_to_ids('assistant')

        assistant_ranges = []  # [(start, end), ...]
        for im_start_pos in im_start_positions:
            # check whether <|im_start|> is followed by assistant (there may be a newline etc. in between)
            # the usual format is <|im_start|>assistant\n
            check_end = min(im_start_pos + 3, len(input_ids_flat))
            following_ids = input_ids_flat[im_start_pos + 1 : check_end].tolist()

            if assistant_token_id in following_ids:
                # find the actual start position of the assistant reply (skip "assistant\n")
                assistant_content_start = im_start_pos + 1 + following_ids.index(assistant_token_id) + 1

                # find the matching <|im_end|> (the first one after assistant_content_start)
                assistant_end = None
                for im_end_pos in im_end_positions:
                    if im_end_pos > assistant_content_start:
                        assistant_end = im_end_pos
                        break

                if assistant_end is not None:
                    assistant_ranges.append((assistant_content_start, assistant_end))

        all_query_tokens = []

        for i in range(num_qa):
            if i >= len(assistant_ranges):
                # assistant_ranges is fewer than num_qa, pad with zero vectors (consistent with get_query_tokens)
                all_query_tokens.append(torch.zeros(1, hidden_dim, device=device))
                continue

            assistant_start, assistant_end = assistant_ranges[i]

            # find <p>...</p> within this range
            qa_queries = []
            used_end_positions = set()

            for p_start in p_start_positions:
                if p_start < assistant_start or p_start >= assistant_end:
                    continue

                # find matching </p>
                for p_end in p_end_positions:
                    if p_end > p_start and p_end <= assistant_end and p_end not in used_end_positions:
                        used_end_positions.add(p_end)

                        # extract token embeddings between <p> and </p>
                        if p_end > p_start + 1:
                            between_embeddings = hidden_flat[p_start + 1 : p_end]
                            avg_embedding = between_embeddings.mean(dim=0, keepdim=True)
                            qa_queries.append(avg_embedding)
                        break

            if len(qa_queries) > 0:
                all_query_tokens.append(torch.cat(qa_queries[:2], dim=0))  # only use the first 2 query tokens
            else:
                # this QA has no <p>...</p>, fill with zero vector (consistent with get_query_tokens)
                all_query_tokens.append(torch.zeros(1, hidden_dim, device=device))

        all_query_tokens = torch.cat(all_query_tokens, dim=0)
        return all_query_tokens




    def get_query_tokens_eval(self, hidden_states, input_ids, text):
        """
        Extract query tokens from hidden_states.
        For inference/evaluation, the input contains only the generated answer (without prompt).
        No need to parse multi-turn dialogue; simply search for <p>...</p> in input_ids.

        Args:
            hidden_states: hidden states of the model's last layer, shape (1, seq_len, hidden_dim)
            input_ids: input token ids, shape (1, seq_len)
            text: generated text (unused)

        Returns:
            list of query tokens
        """
        hidden_dim = hidden_states.shape[-1]
        device = hidden_states.device

        # get ids of special tokens
        p_start_id = self.seg_tokens_ids[0]  # <p>
        p_end_id = self.seg_tokens_ids[1]    # </p>

        input_ids_flat = input_ids[0]  # (seq_len,)
        hidden_flat = hidden_states[0]  # (seq_len, hidden_dim)

        # find positions of all <p> and </p>
        p_start_positions = (input_ids_flat == p_start_id).nonzero(as_tuple=True)[0].tolist()
        p_end_positions = (input_ids_flat == p_end_id).nonzero(as_tuple=True)[0].tolist()

        qa_queries = []
        used_end_positions = set()

        for p_start in p_start_positions:
            # find matching </p>
            for p_end in p_end_positions:
                if p_end > p_start and p_end not in used_end_positions:
                    used_end_positions.add(p_end)

                    # extract token embeddings between <p> and </p>
                    if p_end > p_start + 1:
                        between_embeddings = hidden_flat[p_start + 1 : p_end]
                        avg_embedding = between_embeddings.mean(dim=0, keepdim=True)
                        qa_queries.append(avg_embedding)
                    break

        all_query_tokens = []

        if len(qa_queries) > 0:
            # keep original logic, take at most first 2 queries
            # if len(qa_queries) >= 2:
            all_query_tokens.append(torch.cat(qa_queries[:2], dim=0))
        else:
            # no <p>...</p>, return NaN
            all_query_tokens.append(torch.zeros(1, self.reason_mode_config.text_config.hidden_size, device=device))

        return torch.cat(all_query_tokens, dim=0)


    def infer_reason_train(self, point_feats, qa_data, point_positions=None, IGNORE_INDEX = -100, scene_names=None):
        new_point_feat = []
        language_losses = []
        output_logits = []
        query_tokens_concat_list = []
        query_valid_indices_list = []
        query_invalid_indices_list = []
        qa_to_query_count_list = []
        query_list = []


        if qa_data == []:
            queies = [
                torch.full((1, self.reason_mode_config.text_config.hidden_size),float('nan'),device=point_feats[0].device)
                for _ in range(len(point_feats))]
            return new_point_feat, language_losses, output_logits, queies

        # 
        for i in range(len(point_feats)): 
            # print(point_feats[i].shape)
            i_scene_name = scene_names[i]
            i_point_feat = point_feats[i]
            n_points = i_point_feat.shape[0]
            i_point_pos = None if point_positions is None else point_positions[i]

            scene_name, frame_id = i_scene_name.split('_frame_')

            if 'scene' in scene_name:
                image_dir = self.scannet_image_dir
            else:
                image_dir = self.scannetpp_image_dir


            image_path = os.path.join(image_dir, scene_name, frame_id + '.jpg')

            if self.training and n_points > self.max_point_tokens:
                if i_point_pos is not None and i_point_pos.shape[0] == n_points:
                    i_point_feat = self.pos_aggregator(i_point_feat, i_point_pos)
                    reason_str = 'pos-based FPS aggregator'
                else:
                    i_point_feat = self.point_aggregator(i_point_feat)
                    reason_str = 'feature-based FPS aggregator'
                # MMLogger.get_current_instance().info(f'aggregate points from {n_points} to {i_point_feat.shape[0]} via {reason_str}')
                n_points = i_point_feat.shape[0]
            
            point_embeds = self.point2token(i_point_feat).to(self.reason_model.device)  # (N, D)

            # ------------------- Training branch: append label (GT) and build labels/attention_mask -------------------
            def get_text(value):
                    """Ensure a string is returned; handle the list case"""
                    if isinstance(value, list):
                        return value[0] if len(value) > 0 else ""
                    return str(value)


            qa_pairs = []
            for i_qa in qa_data:
                if isinstance(i_qa, dict):
                    try:
                        qa_pairs.append((get_text(i_qa['question']), get_text(i_qa['answer'])))
                    except:
                        print('Wrong format of qa_data: ')
                        print(i_qa)

                elif isinstance(i_qa, list):
                    qa_pairs.extend((get_text(j_qa['question']), get_text(j_qa['answer'])) for j_qa in i_qa)
            
            if self.training:
                # build multi-turn dialogue messages
                messages = []
                for i, (question, gt_answer) in enumerate(qa_pairs):
                    # add instruction and point tokens in the first round
                    if i==0:
                        user_text = PART_GLAMM_INSTRUCTION_IMG.format(
                            question=question, point=self.point_token * n_points
                        ) 
                        messages.append({
                        'role': 'user',
                        'content':  [{'type': 'image', 
                                      'image': image_path,
                                      'max_pixels': 448 * 448,   
                                      'min_pixels': 224 * 224,
                                        },   
                            {'type': 'text', 'text': user_text}]})
                       
                    else:
                        messages.append({
                        'role': 'user',
                        'content':  [
                            {'type': 'text', 'text': question}]})
                    
                    messages.append({
                                    'role': 'assistant',
                                    'content': gt_answer
                                })
                text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
                images, videos, kwargs = process_vision_info(messages, return_video_kwargs=True)
                data = self.processor(text=[text], images=images, videos=videos, return_tensors='pt', **kwargs)
                data['point_embeds'] = point_embeds

                data['point_mask'] = data['input_ids'] == self.point_token_id
                
                # Use ChatML label formatting to compute loss only on assistant replies
                labels = preprocess_chatml(data['input_ids'], text, self.processor.tokenizer, qwen_model=self.base_model)

                output = self.reason_model(**data.to(self.reason_model.device), labels=labels, output_hidden_states=True)
                lang_loss = output.loss

                hidden_states = output.hidden_states[-1]
                query_tokens = self.get_query_tokens_id(hidden_states, data['input_ids'], len(qa_pairs))
                query_list.append(query_tokens)
                output_text = None
                language_losses.append(lang_loss)

        return new_point_feat, language_losses, output_text, query_list

    # TODO: v2 version: get the query and masks question by question
    @staticmethod
    def _get_text(value):
        """Ensure a string is returned; handle the list case"""
        if isinstance(value, list):
            return value[0] if len(value) > 0 else ""
        return str(value)
    
    def _get_target_dtype(self):
        """Get the target dtype required for model inference"""
        target_dtype = self.reason_model.dtype
        if target_dtype == torch.float32:
            try:
                # PeftModel might report float32 even if base model is fp16/bf16
                target_dtype = next(self.reason_model.parameters()).dtype
            except:
                pass
        
        # If still float32, force bfloat16 or float16 for Flash Attention
        if target_dtype == torch.float32:
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                target_dtype = torch.bfloat16
            else:
                target_dtype = torch.float16
        return target_dtype
    
    def _prepare_data_for_device(self, data, target_dtype):
        """Move data to the correct device and dtype"""
        device = self.reason_model.device
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                if torch.is_floating_point(value):
                    data[key] = value.to(device=device, dtype=target_dtype)
                else:
                    data[key] = value.to(device=device)
        return data
    
    def _prepare_messages_data(self, messages, point_embeds):
        """Prepare model input data"""
        # 3D-only checkpoint (model.decoder.image_dir=None) yields image entries with
        # image=None; drop them so the QA message is point + text only (no vision tokens).
        for _m in messages:
            if isinstance(_m.get('content'), list):
                _m['content'] = [_c for _c in _m['content'] if not (
                    isinstance(_c, dict) and _c.get('type') == 'image' and _c.get('image') is None)]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        images, videos, kwargs = process_vision_info(messages, return_video_kwargs=True)
        data = self.processor(text=[text], images=images, videos=videos, return_tensors='pt', **kwargs)
        data['point_embeds'] = point_embeds
        data['point_mask'] = data['input_ids'] == self.point_token_id
        return data
    
    def _generate_response(self, data, target_dtype):
        """Run model generation"""
        try:
            with torch.cuda.amp.autocast(dtype=target_dtype):
                outputs = self.reason_model.generate(
                    **data, 
                    do_sample=False,
                    temperature=None,
                    top_k=None,
                    top_p=None,
                    repetition_penalty=1.2, 
                    max_new_tokens=1024, 
                    output_hidden_states=True,
                    return_dict_in_generate=True
                )
        except:
            print(data)
        return outputs
    
    def _get_hidden_states_and_query_token(self, data, target_dtype, output_ids, output_text):
        """Get hidden states and extract query token"""

        
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=target_dtype):
                with torch.no_grad():
                    outs2 = self.reason_model(
                        **data,
                        output_hidden_states=True,
                        return_dict=True,
                        use_cache=False,
                    )

                hidden_states = outs2.hidden_states[-1]

        output_hidden_states = hidden_states[:, -output_ids.shape[1]:]

        query_token = self.get_query_tokens_eval(output_hidden_states, output_ids, output_text)
        return query_token

    def _process_single_turn_qa(self, i_qa_data, point_embeds, n_points, target_dtype, pred_qa_data, query_token_list, image_path):
        """Handle single-turn QA data"""

        qa_level = i_qa_data['qa_level']
        i_key = i_qa_data['qa_key']
        i_task_type = i_qa_data['qa_task_type']

        # convert tuple-form keys to plain strings
        qa_level = qa_level[0] if isinstance(qa_level, list) else qa_level
        i_key = i_key[0] if isinstance(i_key, list) else i_key
        i_task_type = i_task_type[0] if isinstance(i_task_type, list) else i_task_type
        
        question = self._get_text(i_qa_data['question'])
        gt_answer = self._get_text(i_qa_data['answer'])
        input_txt = PART_GLAMM_INSTRUCTION.format(question=question, point=self.point_token * n_points)

        
        messages = [{'role': 'user', 'content': [{'type': 'image', 
                                      'image': image_path,
                                      'max_pixels': 448 * 448,   
                                      'min_pixels': 224 * 224,
                                        }, {'type': 'text', 'text': input_txt}]}]
    
        # prepare data
        data = self._prepare_messages_data(messages, point_embeds)
        data = self._prepare_data_for_device(data, target_dtype)
        
        # generate response
        outputs = self._generate_response(data, target_dtype)


        seq = outputs.sequences
        
        
        # decode output
        output_ids = seq[:, data['input_ids'].shape[1]:]
        output_text = self.processor.batch_decode(output_ids, skip_special_tokens=False)
        
        # store predictions
        if qa_level not in pred_qa_data:
            pred_qa_data[qa_level] = {}
        if i_key not in pred_qa_data[qa_level]:
            pred_qa_data[qa_level][i_key] = {}
        if i_task_type not in pred_qa_data[qa_level][i_key]:
            pred_qa_data[qa_level][i_key][i_task_type] = []
        
        pred_qa_data[qa_level][i_key][i_task_type].append({
            'question': question, 
            'gt_answer': gt_answer, 
            'pred_answer': output_text[0]
        })
        
        # get query token





        # # gen_outputs.hidden_states has the structure (Step, Layer)
        # # we need to iterate over each step and take the last layer (Index -1)
        # generated_hidden_list = []

        # for step_tuple in outputs.hidden_states[-1]:
        #     # step_tuple is (Layer0, Layer1,..., LayerN)
        #     last_layer_tensor = step_tuple[-1] # Shape: (Batch, 1, Hidden)
        #     generated_hidden_list.append(last_layer_tensor)

        # # concatenate all generation steps along the sequence dim (dim=1)
        # # Shape: (Batch, Gen_Len, Hidden)
        # generated_last_hidden = torch.cat(generated_hidden_list[1:], dim=1)



        prompt_mask = data["attention_mask"].to(seq.device)
        new_len = seq.size(1) - prompt_mask.size(1)
        full_mask = torch.cat(
            [prompt_mask, torch.ones(prompt_mask.size(0), new_len, device=seq.device, dtype=prompt_mask.dtype)],
            dim=1
        )

        forward_inputs = dict(data)
        forward_inputs["input_ids"] = seq
        forward_inputs["attention_mask"] = full_mask



        query_token = self._get_hidden_states_and_query_token(forward_inputs, target_dtype, output_ids, output_text)
        query_token_list.append(query_token)

    def _process_multi_turn_qa(self, i_qa_data, point_embeds, n_points, target_dtype, pred_qa_data, query_token_list, image_path=None):
        """Handle one multi-turn conversation (``i_qa_data`` = list of turn dicts).

        Standard LLaVA-style next-token inference: turn 0 injects the point tokens (+ the
        reference RGB frame for the Joint model); each later turn appends the running Q/A
        history and regenerates. No memory / RAG / architectural change.
        """
        qa_level = 'multi_turn_qa_data'
        if qa_level not in pred_qa_data:
            pred_qa_data[qa_level] = []

        messages = []
        j_mt_qa_results_list = []

        for j_turn_idx, j_turn_qa_data in enumerate(i_qa_data):
            question = self._get_text(j_turn_qa_data['question'])
            gt_answer = self._get_text(j_turn_qa_data['answer'])

            # turn 0 carries the point tokens (+ the reference frame); later turns are question-only
            if j_turn_idx == 0:
                input_txt = PART_GLAMM_INSTRUCTION_IMG.format(question=question, point=self.point_token * n_points)
                content = [{'type': 'image', 'image': image_path,
                            'max_pixels': 448 * 448, 'min_pixels': 224 * 224},
                           {'type': 'text', 'text': input_txt}]
            else:
                content = [{'type': 'text', 'text': question}]
            messages.append({'role': 'user', 'content': content})

            # prepare data
            data = self._prepare_messages_data(messages, point_embeds)
            data = self._prepare_data_for_device(data, target_dtype)

            # generate response
            outputs = self._generate_response(data, target_dtype)

            # decode output
            generated_ids = outputs.sequences[:, data['input_ids'].shape[1]:]
            output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

            # add model response to history for the next round
            messages.append({'role': 'assistant', 'content': [{'type': 'text', 'text': output_text}]})

            j_mt_qa_results_list.append({
                'question': question,
                'gt_answer': gt_answer,
                'pred_answer': output_text,
                'turn_idx': j_turn_idx
            })

            # extract the <SEG> query token (mirror single-turn: rebuild full-sequence inputs)
            seq = outputs.sequences
            prompt_mask = data['attention_mask'].to(seq.device)
            new_len = seq.size(1) - prompt_mask.size(1)
            full_mask = torch.cat(
                [prompt_mask, torch.ones(prompt_mask.size(0), new_len, device=seq.device, dtype=prompt_mask.dtype)],
                dim=1)
            forward_inputs = dict(data)
            forward_inputs['input_ids'] = seq
            forward_inputs['attention_mask'] = full_mask
            query_token = self._get_hidden_states_and_query_token(forward_inputs, target_dtype, generated_ids, output_text)
            query_token_list.append(query_token)

        pred_qa_data[qa_level].append(j_mt_qa_results_list)

    def infer_reason_eval(self, point_feats, qa_data, point_positions=None, IGNORE_INDEX=-100, scene_names=None):
        new_point_feat = []
        language_losses = []
        query_tokens_concat_list = []
        
        # precompute target_dtype to avoid repeated computation
        target_dtype = self._get_target_dtype()
        
        for i in range(len(point_feats)):
            i_scene_name = scene_names[i]
            i_point_feat = point_feats[i]
            n_points = i_point_feat.shape[0]

            scene_name, frame_id = i_scene_name.split('_frame_')

            if 'scene' in scene_name:
                image_dir = self.scannet_image_dir
            else:
                image_dir = self.scannetpp_image_dir

            image_path = os.path.join(image_dir, scene_name, frame_id + '.jpg')
            point_embeds = self.point2token(i_point_feat).to(self.reason_model.device)
            
            pred_qa_data = {}
            query_token_list = []
            
            for i_qa_data in qa_data:

                if isinstance(i_qa_data, dict) and i_qa_data.get('qa_keys') != 'multi_turn':
                    self._process_single_turn_qa(
                        i_qa_data, point_embeds, n_points, target_dtype, 
                        pred_qa_data, query_token_list, image_path
                    )
                else:
                    self._process_multi_turn_qa(
                        i_qa_data, point_embeds, n_points, target_dtype,
                        pred_qa_data, query_token_list, image_path
                    )
            
            query_tokens_concat_list.append(query_token_list)
            
            # save predictions to a JSON file
            with open(os.path.join(self.save_pred_qa_dir, f'{i_scene_name}.json'), 'w') as f:
                json.dump(pred_qa_data, f, indent=4)
        
        return new_point_feat, language_losses, query_tokens_concat_list

    def _forward_head(self, queries, qa_data, mask_feats, last_flag):
        """Prediction head forward.

        Args:
            queries (List[Tensor] | Tensor): List of len batch_size,
                each of shape (n_queries_i, d_model). Or tensor of
                shape (batch_size, n_queries, d_model).
            mask_feats (List[Tensor]): of len batch_size,
                each of shape (n_points_i, d_model).

        Returns:
            Tuple:
                List[Tensor]: Classification predictions of len batch_size,
                    each of shape (n_queries_i, n_instance_classes + 1).
                List[Tensor] or None: Semantic predictions of len batch_size,
                    each of shape (n_queries_i, n_semantic_classes + 1).
                List[Tensor]: Confidence scores of len batch_size,
                    each of shape (n_queries_i, 1).
                List[Tensor]: Predicted masks of len batch_size,
                    each of shape (n_queries_i, n_points_i).
                List[Tensor] or None: Attention masks of len batch_size,
                    each of shape (n_queries_i, n_points_i).
        """
        pred_scores, pred_masks, attn_masks = [], [], []

        for i in range(len(queries)):
            norm_query = self.out_norm(queries[i])
    

            pred_mask = torch.einsum('nd,md->nm', norm_query, mask_feats[i])
                
            if self.attn_mask:
                attn_mask = (pred_mask.sigmoid() < 0.5).bool()
                attn_mask[torch.where(
                    attn_mask.sum(-1) == attn_mask.shape[-1])] = False
                attn_mask = attn_mask.detach()
                attn_masks.append(attn_mask)
            pred_masks.append(pred_mask)
        attn_masks = attn_masks if self.attn_mask else None
        return pred_scores, pred_masks, attn_masks
    

    def _forward_head_eval(self, queries,  mask_feats, last_flag):
        """Prediction head forward.

        Args:
            queries (List[Tensor] | Tensor): List of len batch_size,
                each of shape (n_queries_i, d_model). Or tensor of
                shape (batch_size, n_queries, d_model).
            mask_feats (List[Tensor]): of len batch_size,
                each of shape (n_points_i, d_model).

        Returns:
            Tuple:
                List[Tensor]: Classification predictions of len batch_size,
                    each of shape (n_queries_i, n_instance_classes + 1).
                List[Tensor] or None: Semantic predictions of len batch_size,
                    each of shape (n_queries_i, n_semantic_classes + 1).
                List[Tensor]: Confidence scores of len batch_size,
                    each of shape (n_queries_i, 1).
                List[Tensor]: Predicted masks of len batch_size,
                    each of shape (n_queries_i, n_points_i).
                List[Tensor] or None: Attention masks of len batch_size,
                    each of shape (n_queries_i, n_points_i).
        """
        pred_scores, pred_masks, attn_masks = [], [], []

        for i in range(len(queries)):
            qa_query = queries[i]
            pred_score_, pred_mask_, attn_mask_ = [], [], []


            for i_qa_query in qa_query:
                # here should based on the i_qa_query to get the query tokens
                norm_query = self.out_norm(i_qa_query)
                pred_score = self.out_score(norm_query) if self.objectness_flag else None
                pred_score_.append(pred_score)
                pred_mask = torch.einsum('nd,md->nm', norm_query, mask_feats[i])
                    
                if self.attn_mask:
                    attn_mask = (pred_mask.sigmoid() < 0.5).bool()
                    attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
                    attn_mask = attn_mask.detach()
                    attn_mask_.append(attn_mask)
                pred_mask_.append(pred_mask)
                
            pred_scores.append(pred_score_)
            pred_masks.append(pred_mask_)
            attn_masks.append(attn_mask_)
        
        attn_masks = attn_masks if self.attn_mask else None
        return pred_scores, pred_masks, attn_masks

    def forward_simple(self, x, queries, qa_data):
        """Simple forward pass.
        
        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, in_channels).
            queries (List[Tensor], optional): of len batch_size, each of shape
                (n_points_i, in_channles).
        
        Returns:
            Dict: with instance scores, semantic scores, masks, and scores.
        """
        inst_feats = [self.input_proj(y) for y in x]
        mask_feats = [self.x_mask(y) for y in x]
        queries = self._get_queries(queries, len(x))
        for i in range(len(self.cross_attn_layers)):
            queries = self.cross_attn_layers[i](inst_feats, queries)
            queries = self.self_attn_layers[i](queries)
            queries = self.ffn_layers[i](queries)
        cls_preds, sem_preds, pred_scores, pred_masks, _= self._forward_head(
            queries, mask_feats, last_flag=True)
        return dict(
            cls_preds=cls_preds,
            sem_preds=sem_preds,
            masks=pred_masks,
            scores=pred_scores,)

    def forward_iter_pred(self, x, scene_names, qa_data, point_positions=None):
        """Iterative forward pass.
        
        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, in_channels).
            queries (List[Tensor], optional): of len batch_size, each of shape
                (n_points_i, in_channles).
        
        Returns:
            Dict: with instance scores, semantic scores, masks, scores,
                and aux_outputs.
        """
        pred_scores, pred_masks, contras_embeds = [], [], []
        inst_feats = [self.input_proj(y) for y in x]
        mask_feats = [self.x_mask(y) for y in x]
        

        if self.training:
            _, language_losses, output_text, queries = self.infer_reason_train(mask_feats, qa_data, point_positions=point_positions, scene_names=scene_names)

            interaction_masks = []
            for m, query in enumerate(queries):
                L = query.shape[0]
                cur_interaction_mask = ~torch.eye(L, device=query.device, dtype=torch.bool)
                interaction_masks.append(cur_interaction_mask)
            original_queries = [q.clone() for q in queries]
            queries = self._get_queries(queries, len(x))
            
            pred_score, pred_mask, attn_mask= self._forward_head(queries, qa_data, mask_feats, last_flag=False)

            pred_masks.append(pred_mask)
            contras_embeds.append([self.out_norm(queries[i].clone()) for i in range(len(queries))])
            
            for i in range(len(self.cross_attn_layers)):
                queries = self.cross_attn_layers[i](inst_feats, queries, attn_mask)
                queries = self.self_attn_layers[i](queries, interaction_masks=interaction_masks)
                queries = self.ffn_layers[i](queries)
                last_flag = i == len(self.cross_attn_layers) - 1
                pred_score, pred_mask, attn_mask = self._forward_head(queries, qa_data, mask_feats, last_flag)
                
                pred_scores.append(pred_score)
                pred_masks.append(pred_mask)
                contras_embeds.append([self.out_norm(queries[i].clone()) for i in range(len(queries))])
            

            aux_outputs = [
            dict(
                masks=masks,
                scores=scores)
            for scores, masks in zip(
                pred_scores[:-1], pred_masks[:-1])]
            return dict(
                original_queries=original_queries,
                masks=pred_masks[-1],
                scores=pred_scores[-1],
                aux_outputs=aux_outputs,
                language_losses=language_losses)

        else:
            new_point_feat, language_losses, query_list = self.infer_reason_eval(mask_feats, qa_data, point_positions=point_positions, scene_names=scene_names)

            original_queries = [q_.clone() for q_i in query_list for q_ in q_i]

            pred_scores_, pred_masks_, contras_embeds_ = [], [], []
            for batch_queries in query_list:
                qa_pred_masks, qa_contras_embeds = [], []
                for query in batch_queries:
                    que_masks, que_contras_embeds = [], []
                    queries = [query]
                    interaction_masks = []
                    for m, query in enumerate(queries):
                        L = query.shape[0]
                        cur_interaction_mask = ~torch.eye(L, device=query.device, dtype=torch.bool)
                        interaction_masks.append(cur_interaction_mask)

                    queries = self._get_queries(queries, 1)
                    pred_score, pred_mask, attn_mask = self._forward_head(queries, qa_data, mask_feats, last_flag=False)

                    pred_mask[0][query.abs().sum(-1) == 0] = torch.ones_like(pred_mask[0][query.abs().sum(-1) == 0]) * 1e-4
                    que_masks.append(pred_mask[0])
                    que_contras_embeds.append([self.out_norm(queries[i].clone()) for i in range(len(queries))])

                    for i in range(len(self.cross_attn_layers)):
                        queries = self.cross_attn_layers[i](inst_feats, queries, attn_mask)
                        queries = self.self_attn_layers[i](queries, interaction_masks=interaction_masks)
                        queries = self.ffn_layers[i](queries)
                        last_flag = i == len(self.cross_attn_layers) - 1
                        pred_score, pred_mask, attn_mask = self._forward_head(queries, qa_data, mask_feats, last_flag)
                        pred_mask[0][query.abs().sum(-1) == 0] = torch.ones_like(pred_mask[0][query.abs().sum(-1) == 0]) * 1e-4
                        que_masks.append(pred_mask[0])
                        que_contras_embeds.append([self.out_norm(queries[i].clone()) for i in range(len(queries))])
                    
                    qa_pred_masks.append(que_masks)
                    qa_contras_embeds.append(que_contras_embeds)
        
                pred_masks.append(qa_pred_masks) 
                contras_embeds.append(qa_contras_embeds) # list of list, the first dimension is the batch size, the second dimension is the number of qa pairs
                


            final_pred_masks = [[pred_masks_i[-1] for pred_masks_i in pred_masks_b] for pred_masks_b in pred_masks]

            return dict(
                original_queries=original_queries,
                masks=final_pred_masks,
                language_losses=language_losses)
    

    def forward(self, x, scene_names=None, qa_data=None, point_positions=None):
        """Forward pass.
        
        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, in_channels).
            queries (List[Tensor], optional): of len batch_size, each of shape
                (n_points_i, in_channles).
        
        Returns:
            Dict: with labels, masks, scores, and possibly aux_outputs.
        """
 
        if self.iter_pred:
            return self.forward_iter_pred(x, scene_names=scene_names, qa_data=qa_data, point_positions=point_positions)
        else:
            return self.forward_simple(x, scene_names=scene_names, qa_data=qa_data)


@MODELS.register_module()
class Grounded_Decoder_Eval(QueryDecoder):
    """We simply add semantic prediction for each instance query.
    """
    def __init__(self, in_channels, num_instance_classes, num_semantic_classes,
                 d_model, num_semantic_linears, sphere_cls, vocabulary_cls_embedding_path, 
                 target_classes = None,
                reason_file=None,
                qwen_model_path='Qwen/Qwen-2.5-VL',
                lora_type='qkvo_all',
                lora_r=128,
                lora_alpha=256,
                lora_dropout=0.1,
                lora_bias=None,
                base_model='qwen2_5_vl',
                max_point_tokens=2500,
                save_pred_qa_dir=None,
                image_dir=None,
                save_viz_dir=None,
                 **kwargs):
        super().__init__(
            in_channels=in_channels, num_classes=num_instance_classes, d_model=d_model, sphere_cls= sphere_cls,
            **kwargs)

        self.save_pred_qa_dir = save_pred_qa_dir
        self.image_dir = image_dir
        # When set (e.g. via `--cfg-options model.decoder.save_viz_dir=viz_dumps`), the model dumps
        # predicted per-point masks here for tools/export_grounding_ply.py. Env MASK_DUMP_DIR also works.
        self.save_viz_dir = save_viz_dir
        if save_viz_dir is not None:
            os.makedirs(save_viz_dir, exist_ok=True)
        if save_pred_qa_dir is not None:
            if not os.path.exists(save_pred_qa_dir):
                os.makedirs(save_pred_qa_dir, exist_ok=True)
        else:
            self.save_pred_qa_dir = None


        self._init_reason_model(
            qwen_model=qwen_model_path,
            lora_type=lora_type,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_bias=lora_bias,
            base_model=base_model)
        
        if base_model == 'qwen2_5_vl':
            # self.token2point = nn.Sequential(
            #     Qwen2RMSNorm(self.reason_mode_config.hidden_size), nn.GELU(),
            #     nn.Linear(self.reason_mode_config.hidden_size, d_model))
                
            self.point2token = nn.Sequential(
                Qwen2RMSNorm(d_model), nn.GELU(),
                nn.Linear(d_model, self.reason_mode_config.hidden_size))
        elif base_model == 'qwen3_vl':
            # self.token2point = nn.Sequential(
            #     Qwen3VLTextRMSNorm(self.reason_mode_config.text_config.hidden_size), 
            #     nn.Linear(self.reason_mode_config.text_config.hidden_size, self.reason_mode_config.text_config.hidden_size),
            #     nn.GELU(), nn.Linear(self.reason_mode_config.text_config.hidden_size, d_model))
                    
            self.point2token = nn.Sequential(
                Qwen3VLTextRMSNorm(d_model), nn.Linear(d_model, d_model),
                nn.GELU(), nn.Linear(d_model, self.reason_mode_config.text_config.hidden_size))
        else:
            raise ValueError(f'unknown base model: {base_model}')


        self.query_proj = nn.Sequential(
                Qwen3VLTextRMSNorm(self.reason_mode_config.text_config.hidden_size),
                nn.Linear(self.reason_mode_config.text_config.hidden_size, d_model), nn.GELU(),
                nn.Linear(d_model, d_model)) 
        self.base_model = base_model

        self.d_model = d_model

        # Learned aggregator to cap point tokens for reasoning
        self.max_point_tokens = max_point_tokens
        self.point_aggregator = FPSAggregator(in_channels=d_model, out_tokens=self.max_point_tokens)
        self.pos_aggregator = FPSPositionAggregator(in_channels=d_model, out_tokens=self.max_point_tokens, pos_dim=3, temperature=0.5)
        self.target_classes = target_classes


    def _init_reason_model(self, qwen_model, lora_type, lora_r, lora_alpha, lora_dropout, lora_bias, base_model='qwen2_5_vl'):

        if base_model == 'qwen2_5_vl':
            self.reason_model = PointQwen2_5_VLForConditionalGeneration.from_pretrained(
                qwen_model, 
                torch_dtype=torch.bfloat16, # Removed
                # attn_implementation=attn_implementation
            )
            self.reason_mode_config = PointQwen2_5_VLConfig.from_pretrained(qwen_model)
            self.processor = Qwen2_5_VLProcessor.from_pretrained(qwen_model, use_fast=True, do_resize=False)
            vocab_size = self.reason_mode_config.vocab_size
        elif base_model == 'qwen3_vl':
            self.reason_model = PointQwen3VLForConditionalGeneration.from_pretrained(qwen_model, torch_dtype=torch.bfloat16, trust_remote_code=True)
            self.reason_mode_config = PointQwen3VLConfig.from_pretrained(qwen_model)
            self.processor = PatchedQwen3VLProcessor.from_pretrained(qwen_model,  trust_remote_code=True)
            vocab_size = self.reason_mode_config.text_config.vocab_size
        else:
            raise ValueError(f'unknown base model: {base_model}')
        
        self.reason_model.requires_grad_(False)

        self.seg_tokens = SEG_TOKEN
        self.point_token = POINT_TOKEN[0]
        special_tokens = SEG_TOKEN + POINT_TOKEN

        new_tokens = self.processor.tokenizer.add_special_tokens(dict(additional_special_tokens=special_tokens))
        if new_tokens > 0 and len(self.processor.tokenizer) > vocab_size:
            MMLogger.get_current_instance().info(f'Expanding vocab size: {vocab_size} -> {len(self.processor.tokenizer)}')
            self.reason_model.resize_token_embeddings(len(self.processor.tokenizer))
            i_emb = self.reason_model.get_input_embeddings().weight.data
            o_emb = self.reason_model.get_output_embeddings().weight.data
            i_emb[-new_tokens:] = i_emb[:-new_tokens].mean(0, keepdim=True)
            o_emb[-new_tokens:] = o_emb[:-new_tokens].mean(0, keepdim=True)

        self.seg_tokens_ids= self.processor.tokenizer.convert_tokens_to_ids(self.seg_tokens)
        self.point_token_id = self.processor.tokenizer.convert_tokens_to_ids(self.point_token)

        self.reason_model.config.point_token_id = self.processor.tokenizer.convert_tokens_to_ids(self.point_token)

        target_modules = self.get_target_modules(self.reason_model, lora_type, base_model)
        MMLogger.get_current_instance().info(f'LoRA target modules: {target_modules}')
        lora_config = LoraConfig(
            task_type='CAUSAL_LM',
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=lora_bias,
            target_modules=target_modules)
        
        self.reason_model = get_peft_model(self.reason_model, lora_config)
        # self.reason_model.eval()
        for name, param in self.reason_model.named_parameters():
            if any(k in name for k in ('embed_tokens', 'lm_head')):
                param.requires_grad = True
            elif 'visual' in name:
                param.requires_grad = False
        

        
        total_params = sum(p.numel() for p in self.reason_model.parameters())
        learnable_params = sum(p.numel() for p in self.reason_model.parameters() if p.requires_grad)
        ratio = round(learnable_params / total_params * 100, 2) if total_params > 0 else 0
        MMLogger.get_current_instance().info(f'Total params: {total_params} Learnable params: {learnable_params} ({ratio}%)')

        return 

    def get_target_modules(self, model, lora_type, base_model):
        layer_type, modules = lora_type.split('_')
        assert layer_type in ('qkvo', 'linear') and modules in ('llm', 'visual', 'all')

        if base_model == 'qwen2_5_vl' or base_model == 'qwen3_vl':
            # all qkvo layers in the visual encoder and the llm
            qkvo_keys = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'attn.qkv', 'attn.proj']

            target_modules = []
            for n, m in model.named_modules():
                if modules == 'llm' and 'visual' in n:
                    continue
                if modules == 'visual' and 'visual' not in n:
                    continue
                if layer_type == 'qkvo' and not any(n.endswith(k) for k in qkvo_keys):
                    continue
                if n in target_modules:
                    continue
                target_modules.append(n)
        else:
            raise ValueError(f'unknown base model: {base_model}')

        return target_modules


    def get_query_tokens(self, hidden_states, input_ids, text, num_qa):
        """
        Extract query tokens from hidden_states, grouped by QA pair.

        Following preprocess_chatml's rounds splitting, determine the token range of each
        assistant reply, then extract embeddings between <p>...</p> within that range.

        Args:
            hidden_states: hidden states of the model's last layer, shape (1, seq_len, hidden_dim)
            input_ids: input token ids, shape (1, seq_len)
            text: full text after apply_chat_template
            num_qa: number of QA pairs

        Returns:
            list of query tokens, length num_qa, each element is a (num_queries, hidden_dim) tensor
        """
        hidden_dim = hidden_states.shape[-1]
        device = hidden_states.device
        tokenizer = self.processor.tokenizer

        # get ids of special tokens
        p_start_id = self.seg_tokens_ids[0]  # <p>
        p_end_id = self.seg_tokens_ids[1]    # </p>

        input_ids_flat = input_ids[0]  # (seq_len,)
        hidden_flat = hidden_states[0]  # (seq_len, hidden_dim)

        # find positions of all <p> and </p>
        p_start_positions = (input_ids_flat == p_start_id).nonzero(as_tuple=True)[0].tolist()
        p_end_positions = (input_ids_flat == p_end_id).nonzero(as_tuple=True)[0].tolist()

        # follow preprocess_chatml splitting
        conv = get_conv('chatml', qwen_model=self.base_model)
        rounds = [m + conv.seps[0] for m in text.split(conv.seps[0])]
        rounds = rounds[:-1]  # drop the final empty round

        # grouping: system + user + assistant as one group, then every two (user + assistant) as a group
        if conv.system is None:
            grouped_rounds = [''.join(rounds[i:i + 2]) for i in range(0, len(rounds), 2)]
        else:
            grouped_rounds = [''.join(rounds[:3])] + [''.join(rounds[i:i + 2]) for i in range(3, len(rounds), 2)]

        # separator: used to split instruction and assistant reply
        sep = conv.seps[0] + conv.roles[1]  # '<|im_end|>\n<|im_start|>assistant\n'

        all_query_tokens = []
        cur_len = 0

        for i, rou in enumerate(grouped_rounds):
            if i >= num_qa:
                break
            if len(rou) == 0:
                all_query_tokens.append(torch.zeros(1, hidden_dim, device=device))
                continue

            # compute the token length of this round
            rou_len = tokenizer(rou, return_length=True).length[0]

            # split instruction and assistant reply
            parts = rou.split(sep)
            if len(parts) < 2:
                # no assistant reply
                all_query_tokens.append(torch.zeros(1, hidden_dim, device=device))
                cur_len += rou_len
                continue

            ins = sep.join(parts[:-1]) + sep
            ins_len = tokenizer(ins, return_length=True).length[0]

            # token range of assistant reply
            assistant_start = cur_len + ins_len
            assistant_end = cur_len + rou_len

            # find <p>...</p> within this range
            qa_queries = []
            used_end_positions = set()

            for p_start in p_start_positions:
                if p_start < assistant_start or p_start >= assistant_end:
                    continue

                # find matching </p>
                for p_end in p_end_positions:
                    if p_end > p_start and p_end <= assistant_end and p_end not in used_end_positions:
                        used_end_positions.add(p_end)

                        # extract token embeddings between <p> and </p>
                        if p_end > p_start + 1:
                            between_embeddings = hidden_flat[p_start + 1 : p_end]
                            avg_embedding = between_embeddings.mean(dim=0, keepdim=True)
                            qa_queries.append(avg_embedding)
                        break

            if len(qa_queries) > 0:
                all_query_tokens.append(torch.cat(qa_queries[:2], dim=0)) # only use the first 2 query tokens
            else:
                # this QA has no <p>...</p>, return NaN
                all_query_tokens.append(torch.zeros(1, hidden_dim, device=device))

            cur_len += rou_len

        # if grouped_rounds count is less than num_qa, pad with NaN
        while len(all_query_tokens) < num_qa:
            all_query_tokens.append(torch.zeros(1, hidden_dim, device=device))
        all_query_tokens = torch.cat(all_query_tokens, dim=0)
        return all_query_tokens



    def get_query_tokens_id(self, hidden_states, input_ids, num_qa):
        """
        Extract query tokens from hidden_states, grouped by QA pair.

        Use the special token positions in input_ids directly to divide each assistant
        reply's range, then extract embeddings between <p>...</p> within that range.

        Difference from get_query_tokens: this function locates assistant ranges based on
        input_ids rather than text, so it is more accurate when extra image tokens, etc.,
        cause text and tokens to be misaligned.

        Args:
            hidden_states: hidden states of the model's last layer, shape (1, seq_len, hidden_dim)
            input_ids: input token ids, shape (1, seq_len)
            num_qa: number of QA pairs

        Returns:
            torch.Tensor: concatenated query tokens, shape (total_queries, hidden_dim)
                          same return format as get_query_tokens
        """
        hidden_dim = hidden_states.shape[-1]
        device = hidden_states.device
        tokenizer = self.processor.tokenizer

        # get ids of special tokens
        p_start_id = self.seg_tokens_ids[0]  # <p>
        p_end_id = self.seg_tokens_ids[1]    # </p>
        im_start_id = tokenizer.convert_tokens_to_ids('<|im_start|>')
        im_end_id = tokenizer.convert_tokens_to_ids('<|im_end|>')

        input_ids_flat = input_ids[0]  # (seq_len,)
        hidden_flat = hidden_states[0]  # (seq_len, hidden_dim)

        # find positions of all <p> and </p>
        p_start_positions = (input_ids_flat == p_start_id).nonzero(as_tuple=True)[0].tolist()
        p_end_positions = (input_ids_flat == p_end_id).nonzero(as_tuple=True)[0].tolist()

        # find positions of all <|im_start|> and <|im_end|>
        im_start_positions = (input_ids_flat == im_start_id).nonzero(as_tuple=True)[0].tolist()
        im_end_positions = (input_ids_flat == im_end_id).nonzero(as_tuple=True)[0].tolist()

        # identify assistant reply ranges
        # strategy: the position of <|im_start|> immediately followed by "assistant" token marks the start of assistant reply
        # the corresponding <|im_end|> marks the end of assistant reply
        assistant_token_id = tokenizer.convert_tokens_to_ids('assistant')

        assistant_ranges = []  # [(start, end), ...]
        for im_start_pos in im_start_positions:
            # check whether <|im_start|> is followed by assistant (there may be a newline etc. in between)
            # the usual format is <|im_start|>assistant\n
            check_end = min(im_start_pos + 3, len(input_ids_flat))
            following_ids = input_ids_flat[im_start_pos + 1 : check_end].tolist()

            if assistant_token_id in following_ids:
                # find the actual start position of the assistant reply (skip "assistant\n")
                assistant_content_start = im_start_pos + 1 + following_ids.index(assistant_token_id) + 1

                # find the matching <|im_end|> (the first one after assistant_content_start)
                assistant_end = None
                for im_end_pos in im_end_positions:
                    if im_end_pos > assistant_content_start:
                        assistant_end = im_end_pos
                        break

                if assistant_end is not None:
                    assistant_ranges.append((assistant_content_start, assistant_end))

        all_query_tokens = []

        for i in range(num_qa):
            if i >= len(assistant_ranges):
                # assistant_ranges is fewer than num_qa, pad with zero vectors (consistent with get_query_tokens)
                all_query_tokens.append(torch.zeros(1, hidden_dim, device=device))
                continue

            assistant_start, assistant_end = assistant_ranges[i]

            # find <p>...</p> within this range
            qa_queries = []
            used_end_positions = set()

            for p_start in p_start_positions:
                if p_start < assistant_start or p_start >= assistant_end:
                    continue

                # find matching </p>
                for p_end in p_end_positions:
                    if p_end > p_start and p_end <= assistant_end and p_end not in used_end_positions:
                        used_end_positions.add(p_end)

                        # extract token embeddings between <p> and </p>
                        if p_end > p_start + 1:
                            between_embeddings = hidden_flat[p_start + 1 : p_end]
                            avg_embedding = between_embeddings.mean(dim=0, keepdim=True)
                            qa_queries.append(avg_embedding)
                        break

            if len(qa_queries) > 0:
                all_query_tokens.append(torch.cat(qa_queries[:2], dim=0))  # only use the first 2 query tokens
            else:
                # this QA has no <p>...</p>, fill with zero vector (consistent with get_query_tokens)
                all_query_tokens.append(torch.zeros(1, hidden_dim, device=device))

        all_query_tokens = torch.cat(all_query_tokens, dim=0)
        return all_query_tokens




    def get_query_tokens_eval(self, hidden_states, input_ids, text):
        """
        Extract query tokens from hidden_states.
        For inference/evaluation, the input contains only the generated answer (without prompt).
        No need to parse multi-turn dialogue; simply search for <p>...</p> in input_ids.

        Args:
            hidden_states: hidden states of the model's last layer, shape (1, seq_len, hidden_dim)
            input_ids: input token ids, shape (1, seq_len)
            text: generated text (unused)

        Returns:
            list of query tokens
        """
        hidden_dim = hidden_states.shape[-1]
        device = hidden_states.device

        # get ids of special tokens
        p_start_id = self.seg_tokens_ids[0]  # <p>
        p_end_id = self.seg_tokens_ids[1]    # </p>

        input_ids_flat = input_ids[0]  # (seq_len,)
        hidden_flat = hidden_states[0]  # (seq_len, hidden_dim)

        # find positions of all <p> and </p>
        p_start_positions = (input_ids_flat == p_start_id).nonzero(as_tuple=True)[0].tolist()
        p_end_positions = (input_ids_flat == p_end_id).nonzero(as_tuple=True)[0].tolist()

        qa_queries = []
        used_end_positions = set()

        for p_start in p_start_positions:
            # find matching </p>
            for p_end in p_end_positions:
                if p_end > p_start and p_end not in used_end_positions:
                    used_end_positions.add(p_end)

                    # extract token embeddings between <p> and </p>
                    if p_end > p_start + 1:
                        between_embeddings = hidden_flat[p_start + 1 : p_end]
                        avg_embedding = between_embeddings.mean(dim=0, keepdim=True)
                        qa_queries.append(avg_embedding)
                    break

        all_query_tokens = []

        if len(qa_queries) > 0:
            # keep original logic, take at most first 2 queries
            # if len(qa_queries) >= 2:
            all_query_tokens.append(torch.cat(qa_queries[:2], dim=0))
        else:
            # no <p>...</p>, return NaN
            all_query_tokens.append(torch.zeros(1, self.reason_mode_config.text_config.hidden_size, device=device))

        return torch.cat(all_query_tokens, dim=0)

    def _process_query_tokens_with_nan(self, query_tokens, hidden_states, num_qa_pairs):
        """
        Process query tokens, identify valid and invalid queries (NaN check), and build the index mapping.

        Args:
            query_tokens (List[Tensor]): each element is a QA pair's query tokens, may contain NaN
            hidden_states (Tensor): used to get hidden_dim and device (if there are no valid query_tokens)
            num_qa_pairs (int): total number of QA pairs

        Returns:
            Tuple:
                query_tokens_concat (Tensor): concatenated valid query tokens, shape (num_valid_queries, hidden_dim)
                query_valid_indices (List): index positions of valid query tokens in the final output
                    single query as integer, multiple queries as list (e.g. [6, 7])
                query_invalid_indices (List[int]): index positions of invalid query tokens in the final output
                qa_to_query_count (List[int]): number of queries per QA pair (including invalid ones)
        """
        # identify valid and invalid queries (NaN check)
        valid_indices = []
        invalid_indices = []
        valid_query_tokens = []

        for idx, q_token in enumerate(query_tokens):
            # check whether it is NaN (if the entire tensor is NaN, consider it invalid)
            if torch.isnan(q_token).all():
                invalid_indices.append(idx)
            else:
                valid_indices.append(idx)
                valid_query_tokens.append(q_token)

        # only concatenate valid queries; store the result in a new variable
        if len(valid_query_tokens) > 0:
            query_tokens_concat = torch.cat(valid_query_tokens, dim=0) # (num_valid_queries, hidden_dim)
        else:
            # if there are no valid queries, create an empty tensor
            hidden_dim = query_tokens[0].shape[-1] if len(query_tokens) > 0 else hidden_states.shape[-1]
            device = hidden_states.device
            query_tokens_concat = torch.empty((0, hidden_dim), device=device)

        # save the query count per QA pair (including invalid ones), used to compute total query count and each QA pair's start position in the final output
        qa_to_query_count = [q_token.shape[0] if len(q_token.shape) > 0 else 1 for q_token in query_tokens]

        # compute each QA pair's query token start position in the final output
        qa_to_output_start = {}
        current_output_idx = 0
        for qa_idx in range(num_qa_pairs):
            qa_to_output_start[qa_idx] = current_output_idx
            current_output_idx += qa_to_query_count[qa_idx]

        # build query_valid_indices: stores index positions of valid query tokens in the final output
        # single query as integer, multiple queries as list
        query_valid_indices = []
        for i, qa_idx in enumerate(valid_indices):
            num_queries = valid_query_tokens[i].shape[0]
            output_start = qa_to_output_start[qa_idx]

            if num_queries == 1:
                # single query, store integer index
                query_valid_indices.append(output_start)
            else:
                # multiple queries, store list of indices
                query_valid_indices.append(list(range(output_start, output_start + num_queries)))

        # build query_invalid_indices: stores index positions of invalid query tokens in the final output
        query_invalid_indices = []
        for qa_idx in invalid_indices:
            num_queries = qa_to_query_count[qa_idx]
            output_start = qa_to_output_start[qa_idx]
            # append invalid query token indices to the list
            query_invalid_indices.extend(range(output_start, output_start + num_queries))
        
        return query_tokens_concat, query_valid_indices, query_invalid_indices, qa_to_query_count

    def infer_reason_train(self, point_feats, qa_data, point_positions=None, IGNORE_INDEX = -100, scene_names=None):
        new_point_feat = []
        language_losses = []
        output_logits = []
        query_tokens_concat_list = []
        query_valid_indices_list = []
        query_invalid_indices_list = []
        qa_to_query_count_list = []
        query_list = []


        if qa_data == []:
            queies = [
                torch.full((1, self.reason_mode_config.text_config.hidden_size),float('nan'),device=point_feats[0].device)
                for _ in range(len(point_feats))]
            return new_point_feat, language_losses, output_logits, queies

        # 
        for i in range(len(point_feats)): 
            # print(point_feats[i].shape)
            i_scene_name = scene_names[i]
            i_point_feat = point_feats[i]
            n_points = i_point_feat.shape[0]
            i_point_pos = None if point_positions is None else point_positions[i]

            scene_name, frame_id = i_scene_name.split('_frame_')


            image_path = (None if self.image_dir is None
                          else os.path.join(self.image_dir, scene_name, frame_id + '.jpg'))

            if self.training and n_points > self.max_point_tokens:
                if i_point_pos is not None and i_point_pos.shape[0] == n_points:
                    i_point_feat = self.pos_aggregator(i_point_feat, i_point_pos)
                    reason_str = 'pos-based FPS aggregator'
                else:
                    i_point_feat = self.point_aggregator(i_point_feat)
                    reason_str = 'feature-based FPS aggregator'
                # MMLogger.get_current_instance().info(f'aggregate points from {n_points} to {i_point_feat.shape[0]} via {reason_str}')
                n_points = i_point_feat.shape[0]
            
            point_embeds = self.point2token(i_point_feat).to(self.reason_model.device)  # (N, D)

            # ------------------- Training branch: append label (GT) and build labels/attention_mask -------------------
            def get_text(value):
                    """Ensure a string is returned; handle the list case"""
                    if isinstance(value, list):
                        return value[0] if len(value) > 0 else ""
                    return str(value)


            qa_pairs = []
            for i_qa in qa_data:
                if isinstance(i_qa, dict):
                    try:
                        qa_pairs.append((get_text(i_qa['question']), get_text(i_qa['answer'])))
                    except:
                        print('Wrong format of qa_data: ')
                        print(i_qa)

                elif isinstance(i_qa, list):
                    qa_pairs.extend((get_text(j_qa['question']), get_text(j_qa['answer'])) for j_qa in i_qa)
            
            if self.training:
                # build multi-turn dialogue messages
                messages = []
                for i, (question, gt_answer) in enumerate(qa_pairs):
                    # add instruction and point tokens in the first round
                    if i==0:
                        user_text = PART_GLAMM_INSTRUCTION_IMG.format(
                            question=question, point=self.point_token * n_points
                        ) 
                        messages.append({
                        'role': 'user',
                        'content':  [{'type': 'image', 
                                      'image': image_path,
                                      'max_pixels': 448 * 448,   
                                      'min_pixels': 224 * 224,
                                        },   
                            {'type': 'text', 'text': user_text}]})
                       
                    else:
                        messages.append({
                        'role': 'user',
                        'content':  [
                            {'type': 'text', 'text': question}]})
                    
                    messages.append({
                                    'role': 'assistant',
                                    'content': gt_answer
                                })
                text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
                images, videos, kwargs = process_vision_info(messages, return_video_kwargs=True)
                data = self.processor(text=[text], images=images, videos=videos, return_tensors='pt', **kwargs)
                data['point_embeds'] = point_embeds

                data['point_mask'] = data['input_ids'] == self.point_token_id
                
                # Use ChatML label formatting to compute loss only on assistant replies
                labels = preprocess_chatml(data['input_ids'], text, self.processor.tokenizer, qwen_model=self.base_model)

                output = self.reason_model(**data.to(self.reason_model.device), labels=labels, output_hidden_states=True)
                lang_loss = output.loss

                hidden_states = output.hidden_states[-1]
                query_tokens = self.get_query_tokens_id(hidden_states, data['input_ids'], len(qa_pairs))
                query_list.append(query_tokens)
                # # process query tokens, identify valid and invalid queries, and build the index mapping
                # query_tokens_concat, query_valid_indices, query_invalid_indices, qa_to_query_count = \
                #     self._process_query_tokens_with_nan(query_tokens, hidden_states, len(qa_pairs))

                output_text = None
                language_losses.append(lang_loss)

        return new_point_feat, language_losses, output_text, query_list

    # TODO: v2 version: get the query and masks question by question
    @staticmethod
    def _get_text(value):
        """Ensure a string is returned; handle the list case"""
        if isinstance(value, list):
            return value[0] if len(value) > 0 else ""
        return str(value)
    
    def _get_target_dtype(self):
        """Get the target dtype required for model inference"""
        target_dtype = self.reason_model.dtype
        if target_dtype == torch.float32:
            try:
                # PeftModel might report float32 even if base model is fp16/bf16
                target_dtype = next(self.reason_model.parameters()).dtype
            except:
                pass
        
        # If still float32, force bfloat16 or float16 for Flash Attention
        if target_dtype == torch.float32:
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                target_dtype = torch.bfloat16
            else:
                target_dtype = torch.float16
        return target_dtype
    
    def _prepare_data_for_device(self, data, target_dtype):
        """Move data to the correct device and dtype"""
        device = self.reason_model.device
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                if torch.is_floating_point(value):
                    data[key] = value.to(device=device, dtype=target_dtype)
                else:
                    data[key] = value.to(device=device)
        return data
    
    def _prepare_messages_data(self, messages, point_embeds):
        """Prepare model input data"""
        # 3D-only checkpoint (model.decoder.image_dir=None) yields image entries with
        # image=None; drop them so the QA message is point + text only (no vision tokens).
        for _m in messages:
            if isinstance(_m.get('content'), list):
                _m['content'] = [_c for _c in _m['content'] if not (
                    isinstance(_c, dict) and _c.get('type') == 'image' and _c.get('image') is None)]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        images, videos, kwargs = process_vision_info(messages, return_video_kwargs=True)
        data = self.processor(text=[text], images=images, videos=videos, return_tensors='pt', **kwargs)
        data['point_embeds'] = point_embeds
        data['point_mask'] = data['input_ids'] == self.point_token_id
        return data
    
    def _generate_response(self, data, target_dtype):
        """Run model generation"""
        with torch.cuda.amp.autocast(dtype=target_dtype):
            outputs = self.reason_model.generate(
                **data, 
                do_sample=False,
                temperature=None,
                top_k=None,
                top_p=None,
                repetition_penalty=1.2, 
                max_new_tokens=1024, 
                output_hidden_states=True,
                return_dict_in_generate=True
            )
        return outputs
    
    def _get_hidden_states_and_query_token(self, data, target_dtype, output_ids, output_text):
        """Get hidden states and extract query token"""

        
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=target_dtype):
                with torch.no_grad():
                    outs2 = self.reason_model(
                        **data,
                        output_hidden_states=True,
                        return_dict=True,
                        use_cache=False,
                    )

                hidden_states = outs2.hidden_states[-1]

        output_hidden_states = hidden_states[:, -output_ids.shape[1]:]

        query_token = self.get_query_tokens_eval(output_hidden_states, output_ids, output_text)
        return query_token

    def _process_single_turn_qa(self, i_qa_data, point_embeds, n_points, target_dtype, pred_qa_data, query_token_list, image_path):
        """Handle single-turn QA data"""

        qa_level = i_qa_data['qa_level']
        i_key = i_qa_data['qa_key']
        i_task_type = i_qa_data['qa_task_type']

        # convert tuple-form keys to plain strings
        qa_level = qa_level[0] if isinstance(qa_level, list) else qa_level
        i_key = i_key[0] if isinstance(i_key, list) else i_key
        i_task_type = i_task_type[0] if isinstance(i_task_type, list) else i_task_type
        
        question = self._get_text(i_qa_data['question'])
        gt_answer = self._get_text(i_qa_data['answer'])
        input_txt = PART_GLAMM_INSTRUCTION.format(question=question, point=self.point_token * n_points)

        
        messages = [{'role': 'user', 'content': [{'type': 'image', 
                                      'image': image_path,
                                      'max_pixels': 448 * 448,   
                                      'min_pixels': 224 * 224,
                                        }, {'type': 'text', 'text': input_txt}]}]
    
        # prepare data
        data = self._prepare_messages_data(messages, point_embeds)
        data = self._prepare_data_for_device(data, target_dtype)
        
        # generate response
        outputs = self._generate_response(data, target_dtype)


        seq = outputs.sequences
        
        
        # decode output
        output_ids = seq[:, data['input_ids'].shape[1]:]
        output_text = self.processor.batch_decode(output_ids, skip_special_tokens=False)
        
        # store predictions
        if qa_level not in pred_qa_data:
            pred_qa_data[qa_level] = {}
        if i_key not in pred_qa_data[qa_level]:
            pred_qa_data[qa_level][i_key] = {}
        if i_task_type not in pred_qa_data[qa_level][i_key]:
            pred_qa_data[qa_level][i_key][i_task_type] = []
        
        pred_qa_data[qa_level][i_key][i_task_type].append({
            'question': question, 
            'gt_answer': gt_answer, 
            'pred_answer': output_text[0]
        })
        
        # get query token





        # # gen_outputs.hidden_states has the structure (Step, Layer)
        # # we need to iterate over each step and take the last layer (Index -1)
        # generated_hidden_list = []

        # for step_tuple in outputs.hidden_states[-1]:
        #     # step_tuple is (Layer0, Layer1,..., LayerN)
        #     last_layer_tensor = step_tuple[-1] # Shape: (Batch, 1, Hidden)
        #     generated_hidden_list.append(last_layer_tensor)

        # # concatenate all generation steps along the sequence dim (dim=1)
        # # Shape: (Batch, Gen_Len, Hidden)
        # generated_last_hidden = torch.cat(generated_hidden_list[1:], dim=1)



        prompt_mask = data["attention_mask"].to(seq.device)
        new_len = seq.size(1) - prompt_mask.size(1)
        full_mask = torch.cat(
            [prompt_mask, torch.ones(prompt_mask.size(0), new_len, device=seq.device, dtype=prompt_mask.dtype)],
            dim=1
        )

        forward_inputs = dict(data)
        forward_inputs["input_ids"] = seq
        forward_inputs["attention_mask"] = full_mask



        query_token = self._get_hidden_states_and_query_token(forward_inputs, target_dtype, output_ids, output_text)
        query_token_list.append(query_token)

    def _process_multi_turn_qa(self, i_qa_data, point_embeds, n_points, target_dtype, pred_qa_data, query_token_list, image_path=None):
        """Handle one multi-turn conversation (``i_qa_data`` = list of turn dicts).

        Standard LLaVA-style next-token inference: turn 0 injects the point tokens (+ the
        reference RGB frame for the Joint model); each later turn appends the running Q/A
        history and regenerates. No memory / RAG / architectural change.
        """
        qa_level = 'multi_turn_qa_data'
        if qa_level not in pred_qa_data:
            pred_qa_data[qa_level] = []

        messages = []
        j_mt_qa_results_list = []

        for j_turn_idx, j_turn_qa_data in enumerate(i_qa_data):
            question = self._get_text(j_turn_qa_data['question'])
            gt_answer = self._get_text(j_turn_qa_data['answer'])

            # turn 0 carries the point tokens (+ the reference frame); later turns are question-only
            if j_turn_idx == 0:
                input_txt = PART_GLAMM_INSTRUCTION_IMG.format(question=question, point=self.point_token * n_points)
                content = [{'type': 'image', 'image': image_path,
                            'max_pixels': 448 * 448, 'min_pixels': 224 * 224},
                           {'type': 'text', 'text': input_txt}]
            else:
                content = [{'type': 'text', 'text': question}]
            messages.append({'role': 'user', 'content': content})

            # prepare data
            data = self._prepare_messages_data(messages, point_embeds)
            data = self._prepare_data_for_device(data, target_dtype)

            # generate response
            outputs = self._generate_response(data, target_dtype)

            # decode output
            generated_ids = outputs.sequences[:, data['input_ids'].shape[1]:]
            output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

            # add model response to history for the next round
            messages.append({'role': 'assistant', 'content': [{'type': 'text', 'text': output_text}]})

            j_mt_qa_results_list.append({
                'question': question,
                'gt_answer': gt_answer,
                'pred_answer': output_text,
                'turn_idx': j_turn_idx
            })

            # extract the <SEG> query token (mirror single-turn: rebuild full-sequence inputs)
            seq = outputs.sequences
            prompt_mask = data['attention_mask'].to(seq.device)
            new_len = seq.size(1) - prompt_mask.size(1)
            full_mask = torch.cat(
                [prompt_mask, torch.ones(prompt_mask.size(0), new_len, device=seq.device, dtype=prompt_mask.dtype)],
                dim=1)
            forward_inputs = dict(data)
            forward_inputs['input_ids'] = seq
            forward_inputs['attention_mask'] = full_mask
            query_token = self._get_hidden_states_and_query_token(forward_inputs, target_dtype, generated_ids, output_text)
            query_token_list.append(query_token)

        pred_qa_data[qa_level].append(j_mt_qa_results_list)

    def infer_reason_eval(self, point_feats, qa_data, point_positions=None, IGNORE_INDEX=-100, scene_names=None):
        new_point_feat = []
        language_losses = []
        query_tokens_concat_list = []
        
        # precompute target_dtype to avoid repeated computation
        target_dtype = self._get_target_dtype()
        
        for i in range(len(point_feats)):
            i_scene_name = scene_names[i]
            i_point_feat = point_feats[i]
            n_points = i_point_feat.shape[0]

            scene_name, frame_id = i_scene_name.split('_frame_')
            image_path = (None if self.image_dir is None
                          else os.path.join(self.image_dir, scene_name, frame_id + '.jpg'))
            point_embeds = self.point2token(i_point_feat).to(self.reason_model.device)
            
            pred_qa_data = {}
            query_token_list = []
            
            for i_qa_data in qa_data:

                if isinstance(i_qa_data, dict) and i_qa_data.get('qa_keys') != 'multi_turn':
                    self._process_single_turn_qa(
                        i_qa_data, point_embeds, n_points, target_dtype, 
                        pred_qa_data, query_token_list, image_path
                    )
                else:
                    self._process_multi_turn_qa(
                        i_qa_data, point_embeds, n_points, target_dtype,
                        pred_qa_data, query_token_list, image_path
                    )
            
            query_tokens_concat_list.append(query_token_list)
            
            # save predictions to a JSON file
            with open(os.path.join(self.save_pred_qa_dir, f'{i_scene_name}.json'), 'w') as f:
                json.dump(pred_qa_data, f, indent=4)
        
        return new_point_feat, language_losses, query_tokens_concat_list

    def _forward_head(self, queries, qa_data, mask_feats, last_flag):
        """Prediction head forward.

        Args:
            queries (List[Tensor] | Tensor): List of len batch_size,
                each of shape (n_queries_i, d_model). Or tensor of
                shape (batch_size, n_queries, d_model).
            mask_feats (List[Tensor]): of len batch_size,
                each of shape (n_points_i, d_model).

        Returns:
            Tuple:
                List[Tensor]: Classification predictions of len batch_size,
                    each of shape (n_queries_i, n_instance_classes + 1).
                List[Tensor] or None: Semantic predictions of len batch_size,
                    each of shape (n_queries_i, n_semantic_classes + 1).
                List[Tensor]: Confidence scores of len batch_size,
                    each of shape (n_queries_i, 1).
                List[Tensor]: Predicted masks of len batch_size,
                    each of shape (n_queries_i, n_points_i).
                List[Tensor] or None: Attention masks of len batch_size,
                    each of shape (n_queries_i, n_points_i).
        """
        pred_scores, pred_masks, attn_masks = [], [], []

        for i in range(len(queries)):
            norm_query = self.out_norm(queries[i])
    

            pred_mask = torch.einsum('nd,md->nm', norm_query, mask_feats[i])
                
            if self.attn_mask:
                attn_mask = (pred_mask.sigmoid() < 0.5).bool()
                attn_mask[torch.where(
                    attn_mask.sum(-1) == attn_mask.shape[-1])] = False
                attn_mask = attn_mask.detach()
                attn_masks.append(attn_mask)
            pred_masks.append(pred_mask)
        attn_masks = attn_masks if self.attn_mask else None
        return pred_scores, pred_masks, attn_masks
    

    def _forward_head_eval(self, queries,  mask_feats, last_flag):
        """Prediction head forward.

        Args:
            queries (List[Tensor] | Tensor): List of len batch_size,
                each of shape (n_queries_i, d_model). Or tensor of
                shape (batch_size, n_queries, d_model).
            mask_feats (List[Tensor]): of len batch_size,
                each of shape (n_points_i, d_model).

        Returns:
            Tuple:
                List[Tensor]: Classification predictions of len batch_size,
                    each of shape (n_queries_i, n_instance_classes + 1).
                List[Tensor] or None: Semantic predictions of len batch_size,
                    each of shape (n_queries_i, n_semantic_classes + 1).
                List[Tensor]: Confidence scores of len batch_size,
                    each of shape (n_queries_i, 1).
                List[Tensor]: Predicted masks of len batch_size,
                    each of shape (n_queries_i, n_points_i).
                List[Tensor] or None: Attention masks of len batch_size,
                    each of shape (n_queries_i, n_points_i).
        """
        pred_scores, pred_masks, attn_masks = [], [], []

        for i in range(len(queries)):
            qa_query = queries[i]
            pred_score_, pred_mask_, attn_mask_ = [], [], []


            for i_qa_query in qa_query:
                # here should based on the i_qa_query to get the query tokens
                norm_query = self.out_norm(i_qa_query)
                pred_score = self.out_score(norm_query) if self.objectness_flag else None
                pred_score_.append(pred_score)
                pred_mask = torch.einsum('nd,md->nm', norm_query, mask_feats[i])
                    
                if self.attn_mask:
                    attn_mask = (pred_mask.sigmoid() < 0.5).bool()
                    attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
                    attn_mask = attn_mask.detach()
                    attn_mask_.append(attn_mask)
                pred_mask_.append(pred_mask)
                
            pred_scores.append(pred_score_)
            pred_masks.append(pred_mask_)
            attn_masks.append(attn_mask_)
        
        attn_masks = attn_masks if self.attn_mask else None
        return pred_scores, pred_masks, attn_masks

    def forward_simple(self, x, queries, qa_data):
        """Simple forward pass.
        
        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, in_channels).
            queries (List[Tensor], optional): of len batch_size, each of shape
                (n_points_i, in_channles).
        
        Returns:
            Dict: with instance scores, semantic scores, masks, and scores.
        """
        inst_feats = [self.input_proj(y) for y in x]
        mask_feats = [self.x_mask(y) for y in x]
        queries = self._get_queries(queries, len(x))
        for i in range(len(self.cross_attn_layers)):
            queries = self.cross_attn_layers[i](inst_feats, queries)
            queries = self.self_attn_layers[i](queries)
            queries = self.ffn_layers[i](queries)
        cls_preds, sem_preds, pred_scores, pred_masks, _= self._forward_head(
            queries, mask_feats, last_flag=True)
        return dict(
            cls_preds=cls_preds,
            sem_preds=sem_preds,
            masks=pred_masks,
            scores=pred_scores,)

    def forward_iter_pred(self, x, scene_names, qa_data, point_positions=None):
        """Iterative forward pass.
        
        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, in_channels).
            queries (List[Tensor], optional): of len batch_size, each of shape
                (n_points_i, in_channles).
        
        Returns:
            Dict: with instance scores, semantic scores, masks, scores,
                and aux_outputs.
        """
        pred_scores, pred_masks, contras_embeds = [], [], []
        inst_feats = [self.input_proj(y) for y in x]
        mask_feats = [self.x_mask(y) for y in x]
        

        if self.training:
            _, language_losses, output_text, queries = self.infer_reason_train(mask_feats, qa_data, point_positions=point_positions, scene_names=scene_names)

            interaction_masks = []
            for m, query in enumerate(queries):
                L = query.shape[0]
                cur_interaction_mask = ~torch.eye(L, device=query.device, dtype=torch.bool)
                interaction_masks.append(cur_interaction_mask)
            original_queries = [q.clone() for q in queries]
            queries = self._get_queries(queries, len(x))
            
            pred_score, pred_mask, attn_mask= self._forward_head(queries, qa_data, mask_feats, last_flag=False)

            pred_masks.append(pred_mask)
            contras_embeds.append([self.out_norm(queries[i].clone()) for i in range(len(queries))])
            
            for i in range(len(self.cross_attn_layers)):
                queries = self.cross_attn_layers[i](inst_feats, queries, attn_mask)
                queries = self.self_attn_layers[i](queries, interaction_masks=interaction_masks)
                queries = self.ffn_layers[i](queries)
                last_flag = i == len(self.cross_attn_layers) - 1
                pred_score, pred_mask, attn_mask = self._forward_head(queries, qa_data, mask_feats, last_flag)
                
                pred_scores.append(pred_score)
                pred_masks.append(pred_mask)
                contras_embeds.append([self.out_norm(queries[i].clone()) for i in range(len(queries))])
            

            aux_outputs = [
            dict(
                masks=masks,
                scores=scores)
            for scores, masks in zip(
                pred_scores[:-1], pred_masks[:-1])]
            return dict(
                original_queries=original_queries,
                masks=pred_masks[-1],
                scores=pred_scores[-1],
                aux_outputs=aux_outputs,
                language_losses=language_losses)

        else:
            new_point_feat, language_losses, query_list = self.infer_reason_eval(mask_feats, qa_data, point_positions=point_positions, scene_names=scene_names)

            original_queries = [q_.clone() for q_i in query_list for q_ in q_i]

            pred_scores_, pred_masks_, contras_embeds_ = [], [], []
            for batch_queries in query_list:
                qa_pred_masks, qa_contras_embeds = [], []
                for query in batch_queries:
                    que_masks, que_contras_embeds = [], []
                    queries = [query]
                    interaction_masks = []
                    for m, query in enumerate(queries):
                        L = query.shape[0]
                        cur_interaction_mask = ~torch.eye(L, device=query.device, dtype=torch.bool)
                        interaction_masks.append(cur_interaction_mask)

                    queries = self._get_queries(queries, 1)
                    pred_score, pred_mask, attn_mask = self._forward_head(queries, qa_data, mask_feats, last_flag=False)

                    pred_mask[0][query.abs().sum(-1) == 0] = torch.ones_like(pred_mask[0][query.abs().sum(-1) == 0]) * 1e-4
                    que_masks.append(pred_mask[0])
                    que_contras_embeds.append([self.out_norm(queries[i].clone()) for i in range(len(queries))])

                    for i in range(len(self.cross_attn_layers)):
                        queries = self.cross_attn_layers[i](inst_feats, queries, attn_mask)
                        queries = self.self_attn_layers[i](queries, interaction_masks=interaction_masks)
                        queries = self.ffn_layers[i](queries)
                        last_flag = i == len(self.cross_attn_layers) - 1
                        pred_score, pred_mask, attn_mask = self._forward_head(queries, qa_data, mask_feats, last_flag)
                        pred_mask[0][query.abs().sum(-1) == 0] = torch.ones_like(pred_mask[0][query.abs().sum(-1) == 0]) * 1e-4
                        que_masks.append(pred_mask[0])
                        que_contras_embeds.append([self.out_norm(queries[i].clone()) for i in range(len(queries))])
                    
                    qa_pred_masks.append(que_masks)
                    qa_contras_embeds.append(que_contras_embeds)
        
                pred_masks.append(qa_pred_masks) 
                contras_embeds.append(qa_contras_embeds) # list of list, the first dimension is the batch size, the second dimension is the number of qa pairs
                


            final_pred_masks = [[pred_masks_i[-1] for pred_masks_i in pred_masks_b] for pred_masks_b in pred_masks]

            return dict(
                original_queries=original_queries,
                masks=final_pred_masks,
                language_losses=language_losses)
    

    def forward(self, x, scene_names=None, qa_data=None, point_positions=None):
        """Forward pass.
        
        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, in_channels).
            queries (List[Tensor], optional): of len batch_size, each of shape
                (n_points_i, in_channles).
        
        Returns:
            Dict: with labels, masks, scores, and possibly aux_outputs.
        """
 
        if self.iter_pred:
            return self.forward_iter_pred(x, scene_names=scene_names, qa_data=qa_data, point_positions=point_positions)
        else:
            return self.forward_simple(x, scene_names=scene_names, qa_data=qa_data)



@MODELS.register_module()
class Grounded_Decoder_Eval_MultiImg(Grounded_Decoder_Eval):
    """Multi-image evaluation variant of :class:`Grounded_Decoder_Eval`.

    The single-frame parent looks up exactly one RGB frame per query by
    splitting the scene id on ``_frame_``. ScanRefer / Reason3D scene ids are
    bare (e.g. ``scene0011_00``) and come with a folder of multi-view frames,
    so this subclass instead loads every ``*.jpg`` under
    ``image_dir/<scene_name>/`` and feeds all of them, ordered by their numeric
    basename, alongside the point tokens and question. Generation settings are
    identical to the parent (greedy decoding).
    """

    def _process_single_turn_qa(self, i_qa_data, point_embeds, n_points, target_dtype,
                                pred_qa_data, query_token_list, image_content):
        """Handle single-turn QA data with a list of image content entries."""

        qa_level = i_qa_data['qa_level']
        i_key = i_qa_data['qa_key']
        i_task_type = i_qa_data['qa_task_type']

        # convert tuple-form keys to plain strings
        qa_level = qa_level[0] if isinstance(qa_level, list) else qa_level
        i_key = i_key[0] if isinstance(i_key, list) else i_key
        i_task_type = i_task_type[0] if isinstance(i_task_type, list) else i_task_type

        question = self._get_text(i_qa_data['question'])
        gt_answer = self._get_text(i_qa_data['answer'])
        input_txt = PART_GLAMM_INSTRUCTION_IMG.format(question=question, point=self.point_token * n_points)

        message_content = list(image_content)
        message_content.append({'type': 'text', 'text': input_txt})
        messages = [{'role': 'user', 'content': message_content}]

        # prepare data
        data = self._prepare_messages_data(messages, point_embeds)
        data = self._prepare_data_for_device(data, target_dtype)

        # generate response
        outputs = self._generate_response(data, target_dtype)
        seq = outputs.sequences

        # decode output
        output_ids = seq[:, data['input_ids'].shape[1]:]
        output_text = self.processor.batch_decode(output_ids, skip_special_tokens=False)

        # store predictions
        if qa_level not in pred_qa_data:
            pred_qa_data[qa_level] = {}
        if i_key not in pred_qa_data[qa_level]:
            pred_qa_data[qa_level][i_key] = {}
        if i_task_type not in pred_qa_data[qa_level][i_key]:
            pred_qa_data[qa_level][i_key][i_task_type] = []

        pred_qa_data[qa_level][i_key][i_task_type].append({
            'question': question,
            'gt_answer': gt_answer,
            'pred_answer': output_text[0]
        })

        prompt_mask = data["attention_mask"].to(seq.device)
        new_len = seq.size(1) - prompt_mask.size(1)
        full_mask = torch.cat(
            [prompt_mask, torch.ones(prompt_mask.size(0), new_len, device=seq.device, dtype=prompt_mask.dtype)],
            dim=1
        )

        forward_inputs = dict(data)
        forward_inputs["input_ids"] = seq
        forward_inputs["attention_mask"] = full_mask

        query_token = self._get_hidden_states_and_query_token(forward_inputs, target_dtype, output_ids, output_text)
        query_token_list.append(query_token)

    def infer_reason_eval(self, point_feats, qa_data, point_positions=None, IGNORE_INDEX=-100, scene_names=None):
        new_point_feat = []
        language_losses = []
        query_tokens_concat_list = []

        # precompute target_dtype to avoid repeated computation
        target_dtype = self._get_target_dtype()

        for i in range(len(point_feats)):
            i_scene_name = scene_names[i]
            i_point_feat = point_feats[i]
            n_points = i_point_feat.shape[0]

            # load all frames for this scene, ordered by numeric basename
            image_root = os.path.join(self.image_dir, i_scene_name)
            image_path_list = sorted(
                [os.path.join(image_root, f) for f in os.listdir(image_root)
                 if f.lower().endswith('.jpg') and not f.startswith('._')],
                key=lambda p: int(''.join(filter(str.isdigit, os.path.basename(p))) or '0')
            )

            image_content = [
                {'type': 'image', 'image': image_path,
                 'max_pixels': 448 * 448, 'min_pixels': 224 * 224}
                for image_path in image_path_list
            ]

            point_embeds = self.point2token(i_point_feat).to(self.reason_model.device)

            pred_qa_data = {}
            query_token_list = []

            for i_qa_data in qa_data:
                self._process_single_turn_qa(
                    i_qa_data, point_embeds, n_points, target_dtype,
                    pred_qa_data, query_token_list, image_content
                )

            query_tokens_concat_list.append(query_token_list)

            # save predictions to a JSON file
            with open(os.path.join(self.save_pred_qa_dir, f'{i_scene_name}.json'), 'w') as f:
                json.dump(pred_qa_data, f, indent=4)

        return new_point_feat, language_losses, query_tokens_concat_list


@MODELS.register_module()
class Grounded_Decoder_Eval_Text(Grounded_Decoder_Eval):
    """Text-only (3D) evaluation variant of :class:`Grounded_Decoder_Eval`.

    The single-frame parent looks up exactly one RGB frame per query by
    splitting the scene id on ``_frame_`` and feeds that image alongside the
    point tokens. ScanRefer / Reason3D 3D rows use bare scene ids
    (e.g. ``scene0011_00``) and supply no image, so this subclass instead uses
    the bare ``i_scene_name`` directly and builds a text-only prompt
    (``PART_GLAMM_INSTRUCTION``) from the point tokens and question. Generation
    settings are inherited from the parent (greedy decoding).
    """

    def _process_single_turn_qa(self, i_qa_data, point_embeds, n_points, target_dtype,
                                pred_qa_data, query_token_list):
        """Handle single-turn QA data with no image (text prompt only)."""

        qa_level = i_qa_data['qa_level']
        i_key = i_qa_data['qa_key']
        i_task_type = i_qa_data['qa_task_type']

        # convert tuple-form keys to plain strings
        qa_level = qa_level[0] if isinstance(qa_level, list) else qa_level
        i_key = i_key[0] if isinstance(i_key, list) else i_key
        i_task_type = i_task_type[0] if isinstance(i_task_type, list) else i_task_type

        question = self._get_text(i_qa_data['question'])
        gt_answer = self._get_text(i_qa_data['answer'])
        input_txt = PART_GLAMM_INSTRUCTION.format(question=question, point=self.point_token * n_points)

        # text-only message: point tokens + question, no image
        messages = [{'role': 'user', 'content': input_txt}]

        # prepare data
        data = self._prepare_messages_data(messages, point_embeds)
        data = self._prepare_data_for_device(data, target_dtype)

        # generate response (greedy, inherited settings)
        outputs = self._generate_response(data, target_dtype)
        seq = outputs.sequences

        # decode output
        output_ids = seq[:, data['input_ids'].shape[1]:]
        output_text = self.processor.batch_decode(output_ids, skip_special_tokens=False)

        # store predictions
        if qa_level not in pred_qa_data:
            pred_qa_data[qa_level] = {}
        if i_key not in pred_qa_data[qa_level]:
            pred_qa_data[qa_level][i_key] = {}
        if i_task_type not in pred_qa_data[qa_level][i_key]:
            pred_qa_data[qa_level][i_key][i_task_type] = []

        pred_qa_data[qa_level][i_key][i_task_type].append({
            'question': question,
            'gt_answer': gt_answer,
            'pred_answer': output_text[0]
        })

        prompt_mask = data["attention_mask"].to(seq.device)
        new_len = seq.size(1) - prompt_mask.size(1)
        full_mask = torch.cat(
            [prompt_mask, torch.ones(prompt_mask.size(0), new_len, device=seq.device, dtype=prompt_mask.dtype)],
            dim=1
        )

        forward_inputs = dict(data)
        forward_inputs["input_ids"] = seq
        forward_inputs["attention_mask"] = full_mask

        # if the model emitted no <SEG> token (a stray point token), fall back to a
        # zero query embedding so the segmentation head still receives a valid tensor
        if '<|point|>' not in output_text[0]:
            query_token = self._get_hidden_states_and_query_token(forward_inputs, target_dtype, output_ids, output_text)
        else:
            query_token = torch.zeros(1, self.reason_mode_config.text_config.hidden_size, device=output_ids.device)

        query_token_list.append(query_token)

    def infer_reason_eval(self, point_feats, qa_data, point_positions=None, IGNORE_INDEX=-100, scene_names=None):
        new_point_feat = []
        language_losses = []
        query_tokens_concat_list = []

        # precompute target_dtype to avoid repeated computation
        target_dtype = self._get_target_dtype()

        for i in range(len(point_feats)):
            i_scene_name = scene_names[i]
            i_point_feat = point_feats[i]
            n_points = i_point_feat.shape[0]

            # bare scene id, no frame lookup and no image
            point_embeds = self.point2token(i_point_feat).to(self.reason_model.device)

            pred_qa_data = {}
            query_token_list = []

            for i_qa_data in qa_data:
                self._process_single_turn_qa(
                    i_qa_data, point_embeds, n_points, target_dtype,
                    pred_qa_data, query_token_list
                )

            query_tokens_concat_list.append(query_token_list)

            # save predictions to a JSON file
            with open(os.path.join(self.save_pred_qa_dir, f'{i_scene_name}.json'), 'w') as f:
                json.dump(pred_qa_data, f, indent=4)

        return new_point_feat, language_losses, query_tokens_concat_list
