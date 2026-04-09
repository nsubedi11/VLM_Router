# models/resolution_pooling.py
#
# Hierarchical resolution support for precomputed visual features.
#
# pool_level = 1  →  one  2×2×2 avg-pool  →  up to 8× fewer tokens
# pool_level = 2  →  two  2×2×2 avg-pools →  up to 64× fewer tokens
#
# Public API
# ----------
#   pool_features_and_grid(feature_inputs, video_grid_thw, pool_level, merge_size=2)
#   shrink_video_tokens_in_ids(input_ids, attention_mask, video_token_id,
#                              vision_start_id, vision_end_id,
#                              orig_grid_thw, new_grid_thw, merge_size=2, pad_token_id=0)
#   prepare_pooled_inputs(feature_inputs, video_grid_thw, input_ids, attention_mask,
#                         pool_level, video_token_id, vision_start_id, vision_end_id,
#                         merge_size=2, pad_token_id=0)

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Feature pooling
# ---------------------------------------------------------------------------

def pool_features_and_grid(
    feature_inputs: List[torch.Tensor],
    video_grid_thw: torch.Tensor,
    pool_level: int,
    merge_size: int = 2,
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """
    Apply 2×2×2 average pooling `pool_level` times to each video's visual
    feature tensor and return updated features and grid_thw.

    Args:
        feature_inputs: list of [N_i, D] tensors, one per video.
        video_grid_thw: [num_videos, 3] or list of [1,3] — (T, H, W) patch-space.
        pool_level:     number of 2×2×2 pooling rounds (0 = identity).
        merge_size:     spatial merge size used by the vision encoder (default 2).

    Returns:
        pooled_features: list of [N_i', D] tensors.
        new_grid_thw:    [num_videos, 3] updated patch-space grid.

    Token geometry
    --------------
    Token grid per video: [T, Ht, Wt] where Ht = H // merge_size.
    After p = 2**pool_level rounds:
        new token grid : [T', Ht', Wt']  (each dim = ceil(orig / p))
        new grid_thw   : [T', Ht'*merge_size, Wt'*merge_size]
    Odd dimensions are zero-padded before each pool step.
    """
    if pool_level == 0:
        return feature_inputs, video_grid_thw

    pooled_features: List[torch.Tensor] = []
    new_grids: List[torch.Tensor] = []

    for i, feat in enumerate(feature_inputs):
        thw = video_grid_thw[i]
        if thw.dim() > 1:
            thw = thw.squeeze(0)          # [1, 3] → [3]
        T, H, W = int(thw[0]), int(thw[1]), int(thw[2])
        Ht = H // merge_size
        Wt = W // merge_size
        D   = feat.shape[-1]
        dev, dtype = feat.device, feat.dtype

        # [T*Ht*Wt, D] → [1, D, T, Ht, Wt] for avg_pool3d
        x = feat.reshape(T, Ht, Wt, D).permute(3, 0, 1, 2).unsqueeze(0).float()

        for _ in range(pool_level):
            _, _, cT, cHt, cWt = x.shape
            pad_t = cT  % 2
            pad_h = cHt % 2
            pad_w = cWt % 2
            if pad_t or pad_h or pad_w:
                # F.pad order from last dim: (W_right, W_left, H_right, H_left, T_right, T_left)
                x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_t))
            x = F.avg_pool3d(x, kernel_size=2, stride=2)

        _, _, T2, Ht2, Wt2 = x.shape
        feat_pooled = (
            x.squeeze(0).permute(1, 2, 3, 0)
            .reshape(T2 * Ht2 * Wt2, D)
            .to(dtype=dtype, device=dev)
        )

        pooled_features.append(feat_pooled)
        new_grids.append(
            torch.tensor(
                [T2, Ht2 * merge_size, Wt2 * merge_size],
                dtype=thw.dtype, device=thw.device,
            )
        )

    new_grid_thw = torch.stack(new_grids, dim=0)
    return pooled_features, new_grid_thw


# ---------------------------------------------------------------------------
# Frame-aware input_ids shrinking
# ---------------------------------------------------------------------------

def shrink_video_tokens_in_ids(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    video_token_id: int,
    vision_start_id: int,
    vision_end_id: int,
    orig_grid_thw: torch.Tensor,   # [num_videos, 3] before pooling
    new_grid_thw: torch.Tensor,    # [num_videos, 3] after pooling
    merge_size: int = 2,
    pad_token_id: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Remove video placeholder tokens from input_ids so the frame structure
    matches the pooled grid_thw that get_rope_index expects.

    Strategy (per video in each batch item):
      - Original: T  frame groups, each  vision_start + orig_fsl video_tokens + vision_end
      - Target  : T' frame groups, each  vision_start + new_fsl  video_tokens + vision_end
      Keep the first T' groups, truncate their spatial tokens to new_fsl,
      and drop the remaining T-T' groups (including their vision_start/end wrappers).

    The flat `orig_grid_thw` / `new_grid_thw` tensors index all videos across
    the batch in appearance order.  Batch-item membership is inferred from the
    total number of video tokens observed per sequence row.

    Args:
        input_ids:       [B, L]
        attention_mask:  [B, L]
        video_token_id:  token ID of <|video_pad|>
        vision_start_id: token ID of <|vision_start|>
        vision_end_id:   token ID of <|vision_end|>
        orig_grid_thw:   [num_videos, 3] (T, H, W) before pooling
        new_grid_thw:    [num_videos, 3] (T', H', W') after pooling
        merge_size:      spatial merge size (default 2)
        pad_token_id:    used for right-padding the rebuilt batch

    Returns:
        new_input_ids:      [B, L']
        new_attention_mask: [B, L']
    """
    B = input_ids.shape[0]
    vid_ptr = 0

    new_seqs: List[torch.Tensor]  = []
    new_masks: List[torch.Tensor] = []

    for b in range(B):
        seq  = input_ids[b]
        amsk = attention_mask[b]

        # ---- infer which videos belong to this batch item ----------------
        item_vid_total = int((seq == video_token_id).sum().item())

        item_vids: List[int] = []
        acc = 0
        ptr = vid_ptr
        while acc < item_vid_total and ptr < len(orig_grid_thw):
            thw = orig_grid_thw[ptr]
            if thw.dim() > 1:
                thw = thw.squeeze(0)
            T  = int(thw[0])
            Ht = int(thw[1]) // merge_size
            Wt = int(thw[2]) // merge_size
            acc += T * Ht * Wt
            item_vids.append(ptr)
            ptr += 1
        vid_ptr = ptr

        if not item_vids:
            new_seqs.append(seq)
            new_masks.append(amsk)
            continue

        # ---- build keep mask ---------------------------------------------
        keep = torch.ones(seq.shape[0], dtype=torch.bool, device=seq.device)
        all_vid_positions = (seq == video_token_id).nonzero(as_tuple=True)[0]
        pos_cursor = 0  # index into all_vid_positions

        for vi in item_vids:
            # Original frame geometry
            orig_thw = orig_grid_thw[vi]
            if orig_thw.dim() > 1:
                orig_thw = orig_thw.squeeze(0)
            orig_T  = int(orig_thw[0])
            orig_Ht = int(orig_thw[1]) // merge_size
            orig_Wt = int(orig_thw[2]) // merge_size
            orig_fsl = orig_Ht * orig_Wt   # tokens per original frame

            # Pooled frame geometry
            new_thw = new_grid_thw[vi]
            if new_thw.dim() > 1:
                new_thw = new_thw.squeeze(0)
            new_T  = int(new_thw[0])
            new_Ht = int(new_thw[1]) // merge_size
            new_Wt = int(new_thw[2]) // merge_size
            new_fsl = new_Ht * new_Wt     # tokens per pooled frame

            # All video token positions for this video, shaped [orig_T, orig_fsl]
            vid_pos = all_vid_positions[pos_cursor : pos_cursor + orig_T * orig_fsl]
            vid_pos_2d = vid_pos.reshape(orig_T, orig_fsl)

            # Frames to KEEP (first new_T), frames to DROP (new_T..orig_T-1)
            for j in range(orig_T):
                frame_vid = vid_pos_2d[j]          # [orig_fsl] video token positions
                vs_pos = int(frame_vid[0]) - 1     # vision_start is immediately before
                ve_pos = int(frame_vid[-1]) + 1    # vision_end is immediately after

                if j < new_T:
                    # Keep vision_start/end; drop excess spatial tokens
                    if orig_fsl > new_fsl:
                        keep[frame_vid[new_fsl:]] = False
                else:
                    # Drop the entire frame group
                    # Guard: only remove vision_start/end if they are the right tokens
                    if vs_pos >= 0 and seq[vs_pos] == vision_start_id:
                        keep[vs_pos] = False
                    keep[frame_vid] = False
                    if ve_pos < seq.shape[0] and seq[ve_pos] == vision_end_id:
                        keep[ve_pos] = False

            pos_cursor += orig_T * orig_fsl

        new_seqs.append(seq[keep])
        new_masks.append(amsk[keep])

    max_len = max(s.shape[0] for s in new_seqs)
    new_input_ids = input_ids.new_full((B, max_len), pad_token_id)
    new_attention_mask = attention_mask.new_zeros((B, max_len))

    for b, (s, m) in enumerate(zip(new_seqs, new_masks)):
        n = s.shape[0]
        new_input_ids[b, :n] = s
        new_attention_mask[b, :n] = m

    return new_input_ids, new_attention_mask


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def prepare_pooled_inputs(
    feature_inputs: List[torch.Tensor],
    video_grid_thw: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pool_level: int,
    video_token_id: int,
    vision_start_id: int,
    vision_end_id: int,
    deepstack_feature_inputs: "Optional[List[List[torch.Tensor]]]" = None,
    merge_size: int = 2,
    pad_token_id: int = 0,
) -> "Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, Optional[List[List[torch.Tensor]]]]":
    """
    Pool visual features (and deepstack features) then shrink input_ids frame-
    structure so everything is consistent with the new video_grid_thw.

    Args:
        feature_inputs:          list of [N_i, D] tensors (one per video).
        video_grid_thw:          [num_videos, 3] patch-space grid before pooling.
        input_ids:               [B, L]
        attention_mask:          [B, L]
        pool_level:              0 = no-op, 1 = one 2×2×2 pool, 2 = two 2×2×2 pools.
        video_token_id:          token ID of <|video_pad|>
        vision_start_id:         token ID of <|vision_start|>
        vision_end_id:           token ID of <|vision_end|>
        deepstack_feature_inputs: optional List[List[Tensor]] — [layer][video],
                                  each tensor [N_i, D].  Pooled with the same
                                  geometry as feature_inputs.
        merge_size:              vision encoder spatial merge size (default 2).
        pad_token_id:            used to re-pad the shortened batch.

    Returns:
        pooled_features, new_grid_thw, new_input_ids, new_attention_mask,
        pooled_deepstack  (None if deepstack_feature_inputs was None)
    """
    if pool_level == 0:
        return feature_inputs, video_grid_thw, input_ids, attention_mask, deepstack_feature_inputs

    orig_grid_thw = video_grid_thw   # keep reference before pooling

    pooled_features, new_grid_thw = pool_features_and_grid(
        feature_inputs, orig_grid_thw, pool_level, merge_size
    )

    # Pool deepstack features with the same geometry (one list per layer)
    pooled_deepstack = None
    if deepstack_feature_inputs is not None:
        pooled_deepstack = []
        for layer_tensors in deepstack_feature_inputs:
            pooled_layer, _ = pool_features_and_grid(
                layer_tensors, orig_grid_thw, pool_level, merge_size
            )
            pooled_deepstack.append(pooled_layer)

    new_input_ids, new_attention_mask = shrink_video_tokens_in_ids(
        input_ids, attention_mask,
        video_token_id=video_token_id,
        vision_start_id=vision_start_id,
        vision_end_id=vision_end_id,
        orig_grid_thw=orig_grid_thw,
        new_grid_thw=new_grid_thw,
        merge_size=merge_size,
        pad_token_id=pad_token_id,
    )

    return pooled_features, new_grid_thw, new_input_ids, new_attention_mask, pooled_deepstack
