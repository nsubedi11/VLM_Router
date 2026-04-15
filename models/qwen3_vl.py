# models/qwen3_vl.py
#
# Qwen3-VL subclasses with offline visual feature support.
#
# Qwen3VLWithOfflineFeatures
#   - encode_video(): run the visual encoder in memory-safe temporal chunks
#     and return (visual_tokens, deepstack_features)
#   - forward(): accepts `feature_inputs` / `deepstack_feature_inputs` to
#     bypass the visual encoder entirely
#   - prepare_inputs_for_generation(): propagates offline features through
#     autoregressive decoding (prefill only)
#
# Qwen3VLProcessorWithPrecomputed
#   - call_with_precomputed(): build model inputs from saved video metadata
#     (video_grid_thw + frames_indices + fps) without touching actual frames

from __future__ import annotations

import torch
from typing import List, Optional, Tuple, Union

from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
from .resolution_pooling import prepare_pooled_inputs
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLCausalLMOutputWithPast
from transformers.feature_extraction_utils import BatchFeature
from transformers.utils import logging

logger = logging.get_logger(__name__)


# ---------------------------------------------------------------------------
# Query extraction helpers (used by the resolution router)
# ---------------------------------------------------------------------------

def extract_query_ids(
    input_ids: torch.Tensor,    # [B, L]
    vision_end_id: int,
    pad_token_id: int = 0,
) -> List[torch.Tensor]:
    """
    Extract user-query token IDs from a left-padded input_ids batch.

    The user query is the sub-sequence that follows the last <|vision_end|>
    token in each row, with padding tokens removed.  This covers the text
    question that comes after all video content in the user turn.

    Sequence layout produced by Qwen3VLProcessorWithPrecomputed:
        [pad ...] <|im_start|> system ... <|im_end|>
        <|im_start|> user
          "This is Video N."
          <|vision_start|> [video_pad × frame_seqlen] <|vision_end|>   ← repeated
          [question text tokens]                                         ← query
        <|im_end|>
        <|im_start|> assistant

    If a sequence contains no vision tokens (text-only), all non-padding
    tokens are returned as the query.

    Returns:
        List of 1-D LongTensors, one per batch item (variable length).
    """
    results: List[torch.Tensor] = []

    for b in range(input_ids.shape[0]):
        seq = input_ids[b]

        ve_positions = (seq == vision_end_id).nonzero(as_tuple=False)
        if ve_positions.numel() > 0:
            # Tokens strictly after the last <|vision_end|>
            last_ve = int(ve_positions[-1, 0])
            query = seq[last_ve + 1:]
        else:
            # No video — use entire sequence
            query = seq

        # Drop padding tokens
        query = query[query != pad_token_id]
        results.append(query)

    return results


def get_query_embedding(
    input_ids: torch.Tensor,       # [B, L]
    embed_fn,                      # nn.Embedding — model.get_input_embeddings()
    vision_end_id: int,
    pad_token_id: int = 0,
) -> torch.Tensor:                 # [B, D]
    """
    Return a mean-pooled query embedding [B, D] suitable for a router head.

    Steps:
      1. extract_query_ids  → variable-length token IDs per item
      2. embed each item's tokens via embed_fn
      3. mean-pool across the sequence dimension  → [B, D]

    If a batch item has no query tokens (degenerate input) a zero vector is
    returned for that item so the batch shape is always [B, D].
    """
    query_ids_list = extract_query_ids(input_ids, vision_end_id, pad_token_id)

    embeddings: List[torch.Tensor] = []
    for qids in query_ids_list:
        if qids.numel() == 0:
            D = embed_fn.weight.shape[1]
            embeddings.append(
                torch.zeros(1, D, dtype=embed_fn.weight.dtype,
                            device=embed_fn.weight.device)
            )
        else:
            # embed_fn expects [...] LongTensor; returns [..., D]
            emb = embed_fn(qids)          # [seq, D]
            embeddings.append(emb.mean(dim=0, keepdim=True))  # [1, D]

    return torch.cat(embeddings, dim=0)   # [B, D]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class Qwen3VLWithOfflineFeatures(Qwen3VLForConditionalGeneration):
    """
    Qwen3-VL with support for precomputed visual features.

    Visual tokens (pooler output) and deepstack features produced by the
    vision encoder can be saved offline and injected at inference/training
    time via `feature_inputs` and `deepstack_feature_inputs`, completely
    bypassing the visual encoder forward pass.
    """

    # Number of temporal patches (T units) to encode at once in encode_video.
    # Reduces peak VRAM at the cost of more sequential kernel launches.
    VIDEO_CHUNK_T: int = 8

    # ------------------------------------------------------------------
    # Offline encoding
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_video(
        self,
        pixel_values_videos: torch.Tensor,
        video_grid_thw: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Run the visual encoder in temporal chunks to avoid OOM on long videos.

        Args:
            pixel_values_videos: patchified frames, shape [total_patches, C]
                as returned by the video processor.
            video_grid_thw: [num_videos, 3] — (T, H_grid, W_grid) per video.

        Returns:
            visual_tokens:       [total_visual_tokens, D_llm]
            deepstack_features:  list of [total_visual_tokens, D_llm],
                                 one tensor per deepstack injection point.
                                 Empty list if the model has no deepstack layers.
        """
        visual = self.model.visual
        pixel_values_videos = pixel_values_videos.to(dtype=visual.dtype)

        all_visual_tokens: list[torch.Tensor] = []
        # all_deepstack[layer_idx] accumulates per-chunk tensors for that layer
        all_deepstack: list[list[torch.Tensor]] | None = None

        patch_offset = 0
        for thw in video_grid_thw:
            T, H, W = thw[0].item(), thw[1].item(), thw[2].item()
            patches_per_t = H * W
            tokens_per_t = (H * W) // (visual.spatial_merge_size ** 2)

            video_tokens: list[torch.Tensor] = []
            # video_ds[chunk_idx] = list[Tensor] (one per deepstack layer)
            video_ds: list[list[torch.Tensor]] = []

            for t_start in range(0, T, self.VIDEO_CHUNK_T):
                t_end = min(t_start + self.VIDEO_CHUNK_T, T)
                chunk_T = t_end - t_start
                pad_T = self.VIDEO_CHUNK_T - chunk_T

                chunk_patches = pixel_values_videos[
                    patch_offset + t_start * patches_per_t:
                    patch_offset + t_end * patches_per_t
                ]
                if pad_T > 0:
                    pad_patches = chunk_patches[-patches_per_t:].repeat(pad_T, 1)
                    chunk_patches = torch.cat([chunk_patches, pad_patches], dim=0)

                chunk_grid = torch.tensor(
                    [[self.VIDEO_CHUNK_T, H, W]],
                    device=pixel_values_videos.device,
                )
                hidden_states, ds_feats = visual(chunk_patches, grid_thw=chunk_grid)

                real_tokens = chunk_T * tokens_per_t
                video_tokens.append(hidden_states[:real_tokens].cpu())

                if ds_feats:
                    video_ds.append([df[:real_tokens].cpu() for df in ds_feats])

                del hidden_states, ds_feats

            all_visual_tokens.append(torch.cat(video_tokens, dim=0))

            if video_ds:
                num_ds = len(video_ds[0])
                merged = [torch.cat([ck[li] for ck in video_ds], dim=0) for li in range(num_ds)]
                if all_deepstack is None:
                    all_deepstack = [[] for _ in range(num_ds)]
                for li in range(num_ds):
                    all_deepstack[li].append(merged[li])

            patch_offset += T * patches_per_t

        visual_tokens = torch.cat(all_visual_tokens, dim=0)
        if all_deepstack is not None:
            deepstack_features = [torch.cat(layer_chunks, dim=0) for layer_chunks in all_deepstack]
        else:
            deepstack_features = []

        return visual_tokens, deepstack_features

    # ------------------------------------------------------------------
    # Overridden forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[Union[torch.LongTensor, List[torch.LongTensor]]] = None,
        mm_token_type_ids: Optional[torch.IntTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        # Offline feature arguments
        feature_inputs: Optional[List[torch.FloatTensor]] = None,        # one tensor per video
        deepstack_feature_inputs: Optional[List[List[torch.FloatTensor]]] = None,  # [layer][video]
        use_fixed_pooling: int = -1,
        **kwargs,
    ):
        if feature_inputs is None:
            # Standard path — delegate entirely to parent.
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                mm_token_type_ids=mm_token_type_ids,
                logits_to_keep=logits_to_keep,
                **kwargs,
            )

        # ------------------------------------------------------------------
        # Offline path: build inputs_embeds manually, skip visual encoder.
        # Replicates the logic inside Qwen3VLModel.forward().
        # ------------------------------------------------------------------
        m = self.model  # Qwen3VLModel
        cfg = self.config
        _pad_id = getattr(cfg, "pad_token_id", 0) or 0

        # --- Resolution selection -------------------------------------------
        # Extract text query tokens (content after the last <|vision_end|>).
        _query_ids = extract_query_ids(
            input_ids,
            vision_end_id=cfg.vision_end_token_id,
            pad_token_id=_pad_id,
        )

        # Debug: verify query extraction by decoding captured tokens.
        # Enable by setting:  model._debug_tokenizer = processor.tokenizer
        # if hasattr(self, "_debug_tokenizer"):
        #     for _b, _qids in enumerate(_query_ids):
        #         _decoded = self._debug_tokenizer.decode(
        #             _qids.tolist(), skip_special_tokens=False
        #         )
        #         print(f"[DEBUG query] batch[{_b}] ({_qids.numel()} tokens): "
        #               f"{repr(_decoded[:300])}")

        # Route: ResolutionRouter predicts pool_level per batch item.
        #   query_ids → 2-layer TransformerEncoder → MLP → pool_level ∈ {0, 1, 2}
        # Assign model.resolution_router = ResolutionRouter(input_dim=...) to activate.
        # Falls back to full resolution (pool_level=0) until the router is trained.

        
        # pool_level = 1
        if hasattr(self, "resolution_router") and self.resolution_router is not None:
            _levels = self.resolution_router.predict(_query_ids, m.get_input_embeddings())
            # All items in the batch must share one pool_level (uniform geometry).
            pool_level = int(_levels.max().item())
        else:
            pool_level = 0
        # # print(f"[DEBUG resolution routing] pool_level={pool_level} ")
    
        if use_fixed_pooling >= 0:
            pool_level = use_fixed_pooling
            print(f"[DEBUG resolution routing] using fixed pool_level={pool_level} ")

        self._last_pool_level = pool_level

        if pool_level > 0:
            (feature_inputs, video_grid_thw,
             input_ids, attention_mask
             ) = prepare_pooled_inputs(
                feature_inputs, video_grid_thw, input_ids, attention_mask,
                pool_level=pool_level,
                video_token_id=cfg.video_token_id,
                vision_start_id=cfg.vision_start_token_id,
                vision_end_id=cfg.vision_end_token_id,
                pad_token_id=_pad_id,
                # deepstack_feature_inputs=deepstack_feature_inputs,
            )

        # 1. Embed text tokens
        inputs_embeds = m.get_input_embeddings()(input_ids)

        # 2. Images: processed normally when present (unusual for this project)
        image_mask = None
        deepstack_image_embeds = None
        # if pixel_values is not None:
        #     img_embeds_split, deepstack_image_embeds = m.get_image_features(pixel_values, image_grid_thw)
        #     image_embeds = torch.cat(img_embeds_split, dim=0).to(
        #         inputs_embeds.device, inputs_embeds.dtype
        #     )
        #     image_mask, _ = m.get_placeholder_mask(
        #         input_ids, inputs_embeds, image_features=image_embeds
        #     )
        #     inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        # 3. Videos: inject precomputed tokens (possibly pooled above)
        video_embeds = torch.cat(feature_inputs, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)

        _, video_mask = m.get_placeholder_mask(
            input_ids, inputs_embeds, video_features=video_embeds
        )
        mask_slots = video_mask[..., 0].sum().item()
        expected = video_embeds.shape[0]
        assert mask_slots == expected, (
            f"video_mask has {mask_slots} slots but feature_inputs has {expected} tokens — "
            "visual features cannot be fully scattered"
        )
        # print("DEBUG forward(): scattering video_embeds with shape", video_embeds.shape, "into inputs_embeds with shape", inputs_embeds.shape, "using video_mask with shape", video_mask.shape)
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        # 4. Build visual_pos_masks and deepstack_visual_embeds
        video_mask_1d = video_mask[..., 0]
        if image_mask is not None:
            image_mask_1d = image_mask[..., 0]
            visual_pos_masks = image_mask_1d | video_mask_1d
            if deepstack_image_embeds and deepstack_feature_inputs:
                # Interleave image and video deepstack features spatially
                img_jt = image_mask_1d[visual_pos_masks]
                vid_jt = video_mask_1d[visual_pos_masks]
                joint_ds = []
                for img_ds, vid_ds_list in zip(deepstack_image_embeds, deepstack_feature_inputs):
                    vid_ds = torch.cat(vid_ds_list, dim=0)
                    combined = img_ds.new_zeros(visual_pos_masks.sum(), img_ds.shape[-1])
                    combined[img_jt] = img_ds.to(combined.device, combined.dtype)
                    combined[vid_jt] = vid_ds.to(combined.device, combined.dtype)
                    joint_ds.append(combined)
                deepstack_visual_embeds = joint_ds
            elif deepstack_image_embeds:
                deepstack_visual_embeds = deepstack_image_embeds
            else:
                deepstack_visual_embeds = (
                    [torch.cat(layer_vids, dim=0) for layer_vids in deepstack_feature_inputs]
                    if deepstack_feature_inputs else None
                )
        else:
            visual_pos_masks = video_mask_1d
            # Cat per-video tensors for each layer into a single tensor per layer.
            deepstack_visual_embeds = (
                [torch.cat(layer_vids, dim=0) for layer_vids in deepstack_feature_inputs]
                if deepstack_feature_inputs else None
            )

        # 5. 3D position IDs (M-RoPE) — computed via get_rope_index (prefill only;
        #    decode steps take the super().forward() path with feature_inputs=None)
        if position_ids is None:
            video_grid_thw_cat = torch.cat(video_grid_thw, dim=0) if isinstance(video_grid_thw, list) else video_grid_thw
            _vid_in_ids = int((input_ids == self.config.video_token_id).sum())
            _vid_in_grid = int(video_grid_thw_cat[:, 0].sum() *
                               (video_grid_thw_cat[:, 1] // 2).sum() *
                               (video_grid_thw_cat[:, 2] // 2).sum()) if video_grid_thw_cat is not None else 0
            # print(f"[DEBUG rope] video_tokens_in_ids={_vid_in_ids}  "
            #       f"feature_total={sum(f.shape[0] for f in feature_inputs)}  "
            #       f"video_grid_thw={video_grid_thw_cat.tolist()}")
            position_ids, rope_deltas = m.get_rope_index(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw_cat,
                attention_mask=attention_mask,
            )
            m.rope_deltas = rope_deltas

        # 6. Language model forward (input_ids=None, uses inputs_embeds)
        lm_out = m.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=None,
            **kwargs,
        )

        # 7. LM head + loss
        hidden_states = lm_out[0]
        slice_idx = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_idx, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size
            )

        return Qwen3VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=lm_out.past_key_values,
            hidden_states=lm_out.hidden_states,
            attentions=lm_out.attentions,
            rope_deltas=m.rope_deltas,
        )

    # ------------------------------------------------------------------
    # Generation support
    # ------------------------------------------------------------------

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        cache_position=None,
        feature_inputs=None,
        deepstack_feature_inputs=None,
        use_fixed_pooling: int = -1,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            use_cache=use_cache,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            cache_position=cache_position,
            **kwargs,
        )

        # Offline features are only needed on the first (prefill) step;
        # subsequent decode steps use cached key/values.
        is_prefill = cache_position is not None and cache_position[0] == 0
        if is_prefill:
            model_inputs["feature_inputs"] = feature_inputs
            model_inputs["deepstack_feature_inputs"] = deepstack_feature_inputs
            model_inputs["use_fixed_pooling"] = use_fixed_pooling
        else:
            model_inputs["feature_inputs"] = None
            model_inputs["deepstack_feature_inputs"] = None

        return model_inputs


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------

class Qwen3VLProcessorWithPrecomputed(Qwen3VLProcessor):
    """
    Qwen3-VL processor that can build tokenized inputs from precomputed video
    metadata (video_grid_thw + frames_indices + fps) without loading frames.

    Use call_with_precomputed() instead of __call__() when visual features
    have been precomputed offline.
    """

    def _build_mm_token_type_ids(self, ids_list: list) -> torch.Tensor:
        """Mark image/video pad token positions as 1, all other positions as 0."""
        import numpy as np
        arr = np.array(ids_list, dtype=np.int64)
        mm_type = np.zeros_like(arr, dtype=np.int64)
        mm_type[arr == self.image_token_id] = 1
        mm_type[arr == self.video_token_id] = 1
        return torch.from_numpy(mm_type)

    def call_with_precomputed(
        self,
        text: Union[str, list[str]],
        precomputed_info: list[dict],
        images=None,
        padding: Union[bool, str] = True,
        return_tensors: str = "pt",
        **tokenizer_kwargs,
    ) -> BatchFeature:
        """
        Build model inputs using precomputed video metadata.

        Args:
            text: string(s) from processor.apply_chat_template (with
                  <|vision_start|><|video_pad|><|vision_end|> placeholders).
            precomputed_info: one dict per video (across the whole batch),
                in the order they appear in text.  Each dict must have:
                  "video_grid_thw"  – Tensor[3] or Tensor[1,3] (T, H, W)
                  "frames_indices"  – Tensor or list of int frame indices
                  "fps"             – float, original video FPS
            images: optional PIL images (processed normally if provided).
            padding: passed to the tokenizer.
            return_tensors: "pt" (default).
            **tokenizer_kwargs: forwarded to self.tokenizer().

        Returns:
            BatchFeature with keys:
              input_ids, attention_mask, mm_token_type_ids, video_grid_thw
              (and image keys if images are provided).
        """
        if not isinstance(text, list):
            text = [text]
        text = list(text)  # copy — modified in-place below

        merge_length = self.video_processor.merge_size ** 2
        temporal_patch_size = self.video_processor.temporal_patch_size

        all_grid_thw: list[torch.Tensor] = []
        video_idx = 0

        for i in range(len(text)):
            video_placeholder_full = (
                f"{self.vision_start_token}{self.video_token}{self.vision_end_token}"
            )
            while self.video_token in text[i]:
                info = precomputed_info[video_idx]
                thw = info["video_grid_thw"]
                if thw.dim() > 1:
                    thw = thw.squeeze(0)
                T, H, W = thw[0].item(), thw[1].item(), thw[2].item()

                frames_indices = info["frames_indices"]
                fps = float(info["fps"])

                timestamps = self._calculate_timestamps(
                    frames_indices, fps, merge_size=temporal_patch_size
                )

                frame_seqlen = (H * W) // merge_length
                video_placeholder = ""
                for frame_idx in range(T):
                    curr_time = timestamps[frame_idx]
                    video_placeholder += f"<{curr_time:.1f} seconds>"
                    video_placeholder += (
                        self.vision_start_token
                        + "<|placeholder|>" * frame_seqlen
                        + self.vision_end_token
                    )

                if video_placeholder_full in text[i]:
                    text[i] = text[i].replace(video_placeholder_full, video_placeholder, 1)
                else:
                    text[i] = text[i].replace(self.video_token, video_placeholder, 1)

                all_grid_thw.append(thw.view(1, 3).long())
                video_idx += 1

            text[i] = text[i].replace("<|placeholder|>", self.video_token)

        # Handle images normally if present
        image_inputs: dict = {}
        if images is not None:
            image_inputs = self.image_processor(images=images)

        text_inputs = self.tokenizer(
            text,
            padding=padding,
            return_tensors=return_tensors,
            **tokenizer_kwargs,
        )

        self._check_special_mm_tokens(text, text_inputs, modalities=["image", "video"])

        ids_list = (
            text_inputs["input_ids"].tolist()
            if isinstance(text_inputs["input_ids"], torch.Tensor)
            else text_inputs["input_ids"]
        )
        mm_type_ids = self._build_mm_token_type_ids(ids_list)

        data: dict = {**text_inputs, **image_inputs}
        data["mm_token_type_ids"] = mm_type_ids
        if all_grid_thw:
            data["video_grid_thw"] = torch.cat(all_grid_thw, dim=0)

        return BatchFeature(data=data, tensor_type=return_tensors)
