import torch
import models
import utils
import torch.nn.functional as F
from models import pipelines, sam, model_dict, torch_device
from utils import parse, guidance, attn, latents, vis
from prompt import (
    DEFAULT_SO_NEGATIVE_PROMPT,
    DEFAULT_OVERALL_NEGATIVE_PROMPT,
)
from easydict import EasyDict
import numpy as np
from func import deal_word_embeddings
from PIL import Image, ImageFilter
from torchvision import transforms

vae, tokenizer, text_encoder, unet, scheduler, dtype = (
    model_dict.vae,
    model_dict.tokenizer,
    model_dict.text_encoder,
    model_dict.unet,
    model_dict.scheduler,
    model_dict.dtype,
)

version = "lmd"

# Hyperparams
height = 512  # default height of Stable Diffusion
width = 512  # default width of Stable Diffusion
H, W = height // 8, width // 8  # size of the latent
guidance_scale = 7.5  # Scale for classifier-free guidance

# batch size: set to 1
overall_batch_size = 1

# attn keys for semantic guidance
overall_guidance_attn_keys = pipelines.DEFAULT_GUIDANCE_ATTN_KEYS

# Start attention aggregation from t steps (take the mean over 50-t steps), used for latent masking
attn_aggregation_step_start = 10

# sigma for gaussian filtering the attn, different if we select point input or box input
gaussian_sigma_point_input = 1.5
gaussian_sigma_box_input = 0.1

# discourage masks with confidence below
discourage_mask_below_confidence = 0.85

# discourage masks with iou (with coarse binarized attention mask) below
discourage_mask_below_coarse_iou = 0.25

mask_th_for_box = 0.05
n_erode_dilate_mask_for_box = 1

offload_guidance_cross_attn_to_cpu = False


def generate_single_object_with_box(
    idx,
    descriptions,
    prompt,
    box,
    phrase,
    word,
    input_latents,
    input_embeddings,
    semantic_guidance_kwargs,
    obj_attn_key,
    saved_cross_attn_keys,
    sam_refine_kwargs,
    num_inference_steps,
    verbose=False,
    visualize=False,
    **kwargs,
):
    bboxes, phrases, words = [box], [phrase], [word]
    prompts = phrase.split(' ')
    prompts.insert(len(prompts), descriptions[idx][1])
    prompts = [' '.join(prompts)]
    if verbose:
        print(f"Getting token map (prompt: {prompt})")

    object_positions, word_token_indices = guidance.get_phrase_indices(
        tokenizer=tokenizer,
        prompt=prompt,
        phrases=phrases,
        words=words,
        return_word_token_indices=True,
        # Since the prompt for single object is from background prompt + object name, we will not have the case of not found
        add_suffix_if_not_found=False,
        verbose=verbose,
    )
    
    # phrases only has one item, so we select the first item in word_token_indices
    word_token_index = word_token_indices[0]

    if verbose:
        print("object positions:", object_positions)
        print("word_token_index:", word_token_index)

    # `offload_guidance_cross_attn_to_cpu` will greatly slow down generation
    (
        latents,
        single_object_images,
        saved_attns,
        single_object_pil_images_box_ann,
        latents_all,
    ) = pipelines.generate_semantic_guidance(
        model_dict,
        input_latents,
        input_embeddings,
        num_inference_steps,
        bboxes,
        prompts,
        object_positions,
        guidance_scale=guidance_scale,
        return_cross_attn=False,
        return_saved_cross_attn=True,
        semantic_guidance_kwargs=semantic_guidance_kwargs,
        saved_cross_attn_keys=[obj_attn_key, *saved_cross_attn_keys],
        return_cond_ca_only=True,
        return_token_ca_only=word_token_index,
        offload_guidance_cross_attn_to_cpu=offload_guidance_cross_attn_to_cpu,
        offload_cross_attn_to_cpu=False,
        return_box_vis=True,
        save_all_latents=True,
        dynamic_num_inference_steps=True,
        **kwargs,
    )
    # `saved_cross_attn_keys` kwargs may have duplicates

    # Since we only return token CA (only one token), token id is 0.
    token_attn_np = attn.get_token_attnv2(
        token_id=0,
        saved_attns=saved_attns,
        attn_key=obj_attn_key,
        attn_aggregation_step_start=attn_aggregation_step_start,
        return_np=True,
        input_ca_has_condition_only=True,
    )

    utils.free_memory()

    single_object_pil_image_box_ann = single_object_pil_images_box_ann[0]

    if visualize:
        print("Single object image")
        vis.display(single_object_pil_image_box_ann)

    mask_selected, conf_score_selected = sam.sam_refine_attn(
        sam_input_image=single_object_images[0],
        token_attn_np=token_attn_np,
        model_dict=model_dict,
        verbose=verbose,
        **sam_refine_kwargs,
    )

    mask_selected_tensor = torch.tensor(mask_selected)

    # if visualize:
    #     vis.visualize(mask_selected, "Mask (selected) after resize")
    #     # This is only for visualizations
    #     masked_latents = latents_all * mask_selected_tensor[None, None, None, ...]
    #     vis.visualize_masked_latents(
    #         latents_all, masked_latents, timestep_T=False, timestep_0=True
    #     )

    return (
        latents_all,
        mask_selected_tensor,
        saved_attns,
        single_object_pil_image_box_ann,
    )


def get_masked_latents_all_list(
    descriptions,
    so_prompt_phrase_word_box_list,
    input_latents_list,
    so_input_embeddings,
    verbose=False,
    **kwargs,
):
    latents_all_list, mask_tensor_list, saved_attns_list, so_img_list = [], [], [], []

    if not so_prompt_phrase_word_box_list:
        return latents_all_list, mask_tensor_list, saved_attns_list, so_img_list

    so_uncond_embeddings, so_cond_embeddings = so_input_embeddings

    for idx, ((prompt, phrase, word, box), input_latents) in enumerate(
        zip(so_prompt_phrase_word_box_list, input_latents_list)
    ):
        so_current_cond_embeddings = so_cond_embeddings[idx*2 + 1 : idx*2 + 2]
        so_current_text_embeddings = torch.cat(
            [so_uncond_embeddings, so_current_cond_embeddings], dim=0
        )
        so_current_input_embeddings = (
            so_current_text_embeddings,
            so_uncond_embeddings,
            so_current_cond_embeddings,
        )

        latents_all, mask_tensor, saved_attns, so_img = generate_single_object_with_box(
            idx,
            descriptions,
            prompt,
            box,
            phrase,
            word,
            input_latents,
            input_embeddings=so_current_input_embeddings,
            verbose=verbose,
            **kwargs,
        )
        latents_all_list.append(latents_all)
        mask_tensor_list.append(mask_tensor)
        saved_attns_list.append(saved_attns)
        so_img_list.append(so_img)

    return latents_all_list, mask_tensor_list, saved_attns_list, so_img_list


# Note: need to keep the supervision, especially the box corrdinates, corresponding to each other in single object and overall.


def run(
    spec,
    bg_seed=1,
    overall_prompt_override="",
    fg_seed_start=20,
    frozen_step_ratio=0.8,
    num_inference_steps=50,
    loss_scale=5,
    loss_threshold=5.0,
    max_iter=[4] * 5 + [3] * 5 + [2] * 5 + [2] * 5 + [1] * 10,
    max_index_step=30,
    overall_loss_scale=5,
    overall_loss_threshold=5.0,
    overall_max_iter=[4] * 5 + [3] * 5 + [2] * 5 + [2] * 5 + [1] * 10,
    overall_max_index_step=30,
    fg_top_p=0.2,
    bg_top_p=0.2,
    overall_fg_top_p=0.2,
    overall_bg_top_p=0.2,
    fg_weight=1.0,
    bg_weight=4.0,
    overall_fg_weight=1.0,
    overall_bg_weight=4.0,
    ref_ca_loss_weight=2.0,
    so_center_box=True,
    fg_blending_ratio=0.01,
    so_negative_prompt=DEFAULT_SO_NEGATIVE_PROMPT,
    overall_negative_prompt=DEFAULT_OVERALL_NEGATIVE_PROMPT,
    mask_th_for_point=0.25,
    so_horizontal_center_only=False,
    align_with_overall_bboxes=True,
    horizontal_shift_only=False,
    use_fast_schedule=False,
    so_vertical_placement="floor_padding",
    so_floor_padding=0.2,
    use_box_input=False,
    # Transfer the cross-attention from single object generation (with ref_ca_saved_attns)
    # Use reference cross attention to guide the cross attention in the overall generation
    use_ref_ca=True,
    use_autocast=False,
    verbose=False,
    # NEW: optional input image (PIL.Image or tensor) — when provided, background latents will be taken
    # from this image so background remains unchanged; only box areas are modified.
    input_image=None,
):
    """
    spec: the spec for generation (see generate.py for how to construct a spec)
    bg_seed: background seed
    overall_prompt_override: use custom overall prompt (rather than the object prompt)
    fg_seed_start: each foreground has a seed (fg_seed_start + i), where i is the index of the foreground
    frozen_step_ratio: how many steps should be frozen (as a ratio to inference steps)
    num_inference_steps: number of inference steps
    (overall_)loss_scale: loss scale for per box or overall generation
    (overall_)loss_threshold: loss threshold for per box or overall generation, below which the loss will not be optimized to prevent artifacts
    (overall_)max_iter: max iterations of loss optimization for each step. If scaler, this is applied to all steps.
    (overall_)max_index_step: max index to apply loss optimization to.
    (overall_)fg_top_p and (overall_)bg_top_p: the top P fraction to optimize
    (overall_)fg_weight and (overall_)bg_weight: the weight for foreground and background optimization.
    ref_ca_loss_weight: weight for attention transfer (i.e., attention reference loss) to ensure the per-box generation is similar to overall generation in the masked region
    so_center_box: using centered box in single object generation to ensure better spatial control in the generation
    fg_blending_ratio: how much should each foreground initial noise deviate from the background initial noise (and each other)
    so_negative_prompt and overall_negative_prompt: negative prompt for single object (per-box) or overall generation
    mask_th_for_point: the threshold for SAM
    so_horizontal_center_only: move to the center horizontally only
    align_with_overall_bboxes: Align the center of the mask, latents, and cross-attention with the center of the box in overall bboxes
    horizontal_shift_only: only shift horizontally for the alignment of mask, latents, and cross-attention
    use_fast_schedule: since the per-box generation, after the steps for latent and attention transfer, is only used by SAM (which does not need to be precise), we skip steps after the steps needed fo[...]
    so_vertical_placement and so_floor_padding: not used if so_horizontal_center_only is set to True (default)
    use_box_input: True for box input and False for point input for SAM
    use_ref_ca: Use reference cross attention to guide the cross attention in the overall generation
    use_autocast: enable automatic mixed precision (saves memory and makes generation faster)
    input_image: optional PIL.Image (or path or tensor) — when provided, the background latents will be encoded from this image so the background remains unchanged.
    """

    frozen_step_ratio = min(max(frozen_step_ratio, 0.0), 1.0)
    frozen_steps = int(num_inference_steps * frozen_step_ratio)

    (
        so_prompt_phrase_word_box_list,
        overall_prompt,
        overall_phrases_words_bboxes,
        descriptions,
        more_des,
    ) = parse.convert_spec(spec, height, width, verbose=verbose)

    if overall_prompt_override and overall_prompt_override.strip():
        overall_prompt = overall_prompt_override.strip()

    overall_phrases, overall_words, overall_bboxes = (
        [item[0] for item in overall_phrases_words_bboxes],
        [item[1] for item in overall_phrases_words_bboxes],
        [item[2] for item in overall_phrases_words_bboxes],
    )

    # The so box is centered but the overall boxes are not (since we need to place to the right place).
    if so_center_box:
        centered_box_kwargs = dict(
            horizontal_center_only=so_horizontal_center_only,
            vertical_placement=so_vertical_placement,
            floor_padding=so_floor_padding,
        )
        so_prompt_phrase_word_box_list = [
            (prompt, phrase, word, utils.get_centered_box(bbox, **centered_box_kwargs))
            for prompt, phrase, word, bbox in so_prompt_phrase_word_box_list
        ]
        if verbose:
            print(
                f"centered so_prompt_phrase_word_box_list: {so_prompt_phrase_word_box_list}"
            )
    so_boxes = [item[-1] for item in so_prompt_phrase_word_box_list]

    if "extra_neg_prompt" in spec and spec["extra_neg_prompt"]:
        so_negative_prompt = spec["extra_neg_prompt"] + ", " + so_negative_prompt
        overall_negative_prompt = (
            spec["extra_neg_prompt"] + ", " + overall_negative_prompt
        )

    gaussian_sigma = (
        gaussian_sigma_box_input if use_box_input else gaussian_sigma_point_input
    )

    semantic_guidance_kwargs = dict(
        loss_scale=loss_scale,
        loss_threshold=loss_threshold,
        max_iter=max_iter,
        max_index_step=max_index_step,
        fg_top_p=fg_top_p,
        bg_top_p=bg_top_p,
        fg_weight=fg_weight,
        bg_weight=bg_weight,
        use_ratio_based_loss=False,
        guidance_attn_keys=overall_guidance_attn_keys,
        verbose=True,
    )

    sam_refine_kwargs = dict(
        use_box_input=use_box_input,
        gaussian_sigma=gaussian_sigma,
        mask_th_for_box=mask_th_for_box,
        n_erode_dilate_mask_for_box=n_erode_dilate_mask_for_box,
        mask_th_for_point=mask_th_for_point,
        discourage_mask_below_confidence=discourage_mask_below_confidence,
        discourage_mask_below_coarse_iou=discourage_mask_below_coarse_iou,
        height=height,
        width=width,
        H=H,
        W=W,
    )

    # Helper: convert PIL.Image (or path) to latents using current VAE
    def pil_to_latents(img):
        # Accept path or PIL.Image
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")
        elif isinstance(img, torch.Tensor):
            # assume already preprocessed pixel tensor in [0,1]
            img = img
            if img.dim() == 3:
                img = img.unsqueeze(0)
            img = img.to(torch_device, dtype=dtype)
            # map to [-1,1]
            img = img * 2.0 - 1.0
            with torch.no_grad():
                try:
                    lat = vae.encode(img).latent_dist.mean
                except Exception:
                    lat = vae.encode(img).sample()
            return lat
        else:
            img = img.convert("RGB")

        img = img.resize((width, height), resample=Image.LANCZOS)
        to_tensor = transforms.ToTensor()
        img_t = to_tensor(img).unsqueeze(0).to(torch_device, dtype=dtype)
        # map to [-1,1], typical for SD vae
        img_t = img_t * 2.0 - 1.0
        with torch.no_grad():
            try:
                lat = vae.encode(img_t).latent_dist.mean
            except Exception:
                lat = vae.encode(img_t).sample()
        # some VAE implementations scale latents; attempt to adjust if available
        try:
            scale = vae.config.scaling_factor
            lat = lat * scale
        except Exception:
            pass
        return lat

    # if verbose:
    #     vis.visualize_bboxes(
    #         bboxes=[item[-1] for item in so_prompt_phrase_word_box_list], H=H, W=W
    #     )

    # Note that so and overall use different negative prompts

    with torch.autocast("cuda", enabled=use_autocast):
        so_prompts = [item[0] for item in so_prompt_phrase_word_box_list]

        text = []
        text.append(descriptions)
        text.extend(more_des)
        final_descriptions = deal_word_embeddings(unet, tokenizer, text_encoder, text, torch_device)
        for i in range(len(so_prompts)):
            item = so_prompts[i*2].split(' ')
            item.insert(len(item), final_descriptions[i])

            so_prompts.insert(i*2+1, ' '.join(item))
        # print(so_prompts)

        if so_prompts:
            so_input_embeddings = models.encode_prompts(
                prompts=so_prompts,
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                negative_prompt=so_negative_prompt,
                one_uncond_input_only=True,
            )
        else:
            so_input_embeddings = []

        # Generate initial latents (may be replaced for background if input_image provided)
        input_latents_list, latents_bg = latents.get_input_latents_list(
            model_dict,
            bg_seed=bg_seed,
            fg_seed_start=fg_seed_start,
            so_boxes=so_boxes,
            fg_blending_ratio=fg_blending_ratio,
            height=height,
            width=width,
            verbose=False,
        )
        def decode_latents_to_pil(latents_tensor):
            """
            Robust decode for latents of various shapes:
            - supports shapes: (B,C,H,W), (C,H,W), (T,B,C,H,W), (B,T,C,H,W), (1,1,C,H,W), ...
            - picks the last timestep if a timestep dimension exists.
            Returns a PIL image for the first item in batch.
            """
            import torch
            from PIL import Image
            try:
                lat = latents_tensor.clone()

                # move to correct device/dtype
                lat = lat.to(torch_device).to(dtype)

                # Handle 5D tensors which may be (T,B,C,H,W) or (B,T,C,H,W)
                if lat.dim() == 5:
                    # Common patterns:
                    # (T, B, C, H, W) -> choose last timestep lat[-1] => (B,C,H,W)
                    # (B, T, C, H, W) -> choose last timestep lat[:, -1] => (B,C,H,W)
                    # use heuristic based on where channel dim 4 appears (C usually == 4)
                    if lat.shape[2] == 4:
                        # treat as (T, B, C, H, W)
                        lat = lat[-1]
                    elif lat.shape[3] == 4:
                        # treat as (B, T, C, H, W)
                        lat = lat[:, -1]
                    else:
                        # fallback: squeeze singleton dims if present
                        lat = lat.squeeze(0)
                        lat = lat.squeeze(0)

                # If after processing we have 3D (C,H,W), unsqueeze batch
                if lat.dim() == 3:
                    lat = lat.unsqueeze(0)

                # Now lat should be 4D: (B,C,H,W)
                if lat.dim() != 4:
                    raise ValueError(f"Unexpected latent shape after normalization: {lat.shape}")

                # Some VAEs expect a scaling before decode; try to undo common scaling if present
                try:
                    scale = getattr(vae.config, "scaling_factor", None)
                    if scale is not None and scale != 0:
                        lat_for_decode = lat / scale
                    else:
                        lat_for_decode = lat
                except Exception:
                    lat_for_decode = lat

                with torch.no_grad():
                    decoded = vae.decode(lat_for_decode)
                    # diffusers sometimes returns an object with .sample
                    try:
                        image_tensor = decoded.sample
                    except Exception:
                        image_tensor = decoded

                # image_tensor expected in [-1,1]; map to [0,1]
                image_tensor = (image_tensor / 2 + 0.5).clamp(0, 1)

                # Convert to HWC numpy uint8 and PIL
                img_np = image_tensor.cpu().permute(0, 2, 3, 1).numpy()
                img_np = (img_np * 255).round().astype("uint8")
                pil = Image.fromarray(img_np[0])
                return pil
            except Exception as e:
                print("decode_latents_to_pil failed:", e)
                raise

        # If user provided an input image, encode it to latents and use it as the background latents.
        # This ensures the background (outside boxes) remains exactly as the input image.
        if input_image is not None:
            if verbose:
                print("Encoding input_image to latents and using it as latents_bg to preserve background.")
            try:
                latents_bg_encoded = pil_to_latents(input_image)
                # Ensure same device and dtype
                latents_bg = latents_bg_encoded.to(latents_bg.device if isinstance(latents_bg, torch.Tensor) else torch_device, dtype=latents_bg_encoded.dtype)
            except Exception as e:
                # Fallback: keep original latents_bg and warn
                print("Warning: failed to encode input_image to latents. Falling back to random background latents.", e)

        if use_fast_schedule:
            fast_after_steps = (
                max(frozen_steps, overall_max_index_step)
                if use_ref_ca
                else frozen_steps
            )
        else:
            fast_after_steps = None

        if use_ref_ca or frozen_steps > 0:
            (
                latents_all_list,
                mask_tensor_list,
                saved_attns_list,
                so_img_list,
            ) = get_masked_latents_all_list(
                descriptions,
                so_prompt_phrase_word_box_list,
                input_latents_list,
                semantic_guidance_kwargs=semantic_guidance_kwargs,
                obj_attn_key=("down", 2, 1, 0),
                saved_cross_attn_keys=overall_guidance_attn_keys if use_ref_ca else [],
                sam_refine_kwargs=sam_refine_kwargs,
                so_input_embeddings=so_input_embeddings,
                num_inference_steps=num_inference_steps,
                fast_after_steps=fast_after_steps,
                fast_rate=2,
                verbose=verbose,
            )
        else:
            # No per-box guidance
            (latents_all_list, mask_tensor_list, saved_attns_list, so_img_list) = (
                [],
                [],
                [],
                [],
            )
        # [新增] 產生背景的雜訊歷史 (Noisy Background History)
        # 這是為了確保混合時，背景不是全黑的 0，而是帶有正確雜訊的 input_image
        if input_image is not None:
            print("Generating noisy background history for mixing...")
            # 1. 確保 scheduler 時間步已設定
            scheduler.set_timesteps(num_inference_steps)
            
            # 2. 準備一個列表來存每一步的背景
            bg_latents_all = []
            
            # 3. 對於每一個時間步 t，都將雜訊加到 latents_bg (原圖) 上
            # 注意：latents_bg 目前是乾淨的原圖 (z0)
            for t in scheduler.timesteps:
                # 產生隨機雜訊
                noise = torch.randn_like(latents_bg)
                # 使用 scheduler 加噪: z_t = alpha * z_0 + sigma * noise
                noisy_bg = scheduler.add_noise(latents_bg, noise, t)
                bg_latents_all.append(noisy_bg)
            
            # 4. 最後補上原本乾淨的 latents_bg (做為最後一步)
            bg_latents_all.append(latents_bg)
            
            # 5. 堆疊成 5D 張量 [T+1, B, C, H, W]
            latents_bg = torch.stack(bg_latents_all)
            
            # [重要] 確保 pipeline 開始的起點是全雜訊 (T)，而不是乾淨圖
            # 因為 latents_bg[0] 對應到 timesteps[0] (
        (
            composed_latents,
            foreground_indices,
            offset_list,
        ) = latents.compose_latents_with_alignment(
            model_dict,
            latents_all_list,
            mask_tensor_list,
            num_inference_steps,
            overall_batch_size,
            height,
            width,
            latents_bg=latents_bg,
            align_with_overall_bboxes=align_with_overall_bboxes,
            overall_bboxes=overall_bboxes,
            horizontal_shift_only=horizontal_shift_only,
            use_fast_schedule=use_fast_schedule,
            fast_after_steps=fast_after_steps,
        )

        b = []
        for i, des in enumerate(final_descriptions):
            a = overall_prompt.split(',')
            a = a[i].strip().split(' ')
            a.insert(len(a), des)
            
            b.append(' '.join(a))
            
        overall_prompt = ', '.join(b)
        # print("overall_prompt:", overall_prompt)

        # NOTE: need to ensure overall embeddings are generated after the update of overall prompt
        (
            overall_object_positions,
            overall_word_token_indices,
            overall_prompt,
        ) = guidance.get_phrase_indices(
            tokenizer=tokenizer,
            prompt=overall_prompt,
            phrases=overall_phrases,
            words=overall_words,
            verbose=verbose,
            return_word_token_indices=True,
            add_suffix_if_not_found=True,
        )

        overall_input_embeddings = models.encode_prompts(
            prompts=[overall_prompt],
            tokenizer=tokenizer,
            negative_prompt=overall_negative_prompt,
            text_encoder=text_encoder,
        )

        if use_ref_ca:
            # ref_ca_saved_attns has the same hierarchy as bboxes
            ref_ca_saved_attns = []

            flattened_box_idx = 0
            for bboxes in overall_bboxes:
                # bboxes: correspond to a phrase
                ref_ca_current_phrase_saved_attns = []
                for bbox in bboxes:
                    # each individual bbox
                    saved_attns = saved_attns_list[flattened_box_idx]
                    if align_with_overall_bboxes:
                        offset = offset_list[flattened_box_idx]
                        saved_attns = attn.shift_saved_attns(
                            saved_attns,
                            offset,
                            guidance_attn_keys=overall_guidance_attn_keys,
                            horizontal_shift_only=horizontal_shift_only,
                        )
                    ref_ca_current_phrase_saved_attns.append(saved_attns)
                    flattened_box_idx += 1
                ref_ca_saved_attns.append(ref_ca_current_phrase_saved_attns)

        # Reference attn: Transfer the cross-attention from single object generation (with ref_ca_saved_attns)

        # This is currently not-shared with the single object one.
        overall_semantic_guidance_kwargs = dict(
            loss_scale=overall_loss_scale,
            loss_threshold=overall_loss_threshold,
            max_iter=overall_max_iter,
            max_index_step=overall_max_index_step,
            fg_top_p=overall_fg_top_p,
            bg_top_p=overall_bg_top_p,
            fg_weight=overall_fg_weight,
            bg_weight=overall_bg_weight,
            # ref_ca comes from the attention map of the word token of the phrase in single object generation, so we apply it only to the word token of the phrase in overall generation.
            ref_ca_word_token_only=True,
            # If a word is not provided, we use the last token.
            ref_ca_last_token_only=True,
            ref_ca_saved_attns=ref_ca_saved_attns if use_ref_ca else None,
            word_token_indices=overall_word_token_indices,
            guidance_attn_keys=overall_guidance_attn_keys,
            ref_ca_loss_weight=ref_ca_loss_weight,
            use_ratio_based_loss=False,
            verbose=True,
        )

        # Generate with composed latents

        # NEW: freeze background (so only bounding-box regions / foreground can be modified)
        # foreground_indices == 0 means background; we want to freeze background so the background remains unchanged.

        # frozen_mask = foreground_indices != 0

        # 2. 建立一個代表「所有 Bounding Box 區域」的遮罩
        #    (True 代表在 Bounding Box 內)
        all_boxes_mask = torch.zeros((H, W), dtype=torch.bool, device=torch_device)
        
        # 遍歷 overall_bboxes (這是一個 box 的列表的列表)
        for box_list in overall_bboxes:
            for box in box_list:
                # utils.proportion_to_mask 會將 BBox 轉為 2D 遮罩
                # 我們使用 OR (|=) 運算將所有 BBox 的遮罩合併起來
                all_boxes_mask |= (utils.proportion_to_mask(box, H, W) > 0)
        
        # 3. 建立一個代表「精確物件區域」的遮罩
        #    (True 代表在精確的物件上)
        precise_object_mask = (foreground_indices != 0)

        # 4. 建立一個代表「Bounding Box 以外背景」的遮罩
        #    (True 代表在所有 Bounding Box 之外)
        background_outside_boxes_mask = ~all_boxes_mask

        # 5. 產生最終的 frozen_mask
        #    我們要凍結的是「精確的物件」OR「BBox 以外的背景」
        frozen_mask = precise_object_mask | background_outside_boxes_mask
        # print(overall_bboxes)
        # print(np.array(overall_bboxes) * 512)
        # print(foreground_indices.size())
        # print(latents_bg.size())
        # print(torch.zeros(latents_bg.shape[-2:], dtype=torch.long).size())
        # num_inference_steps = 50
        # frozen_steps = 40
        regen_latents, images = pipelines.generate_partial_frozen(
            model_dict,
            composed_latents.cuda(),
            frozen_mask.cuda(),
            precise_object_mask,
            background_outside_boxes_mask,
            overall_input_embeddings,
            num_inference_steps,#50
            frozen_steps,
            guidance_scale,
            bboxes=overall_bboxes,
            phrases=overall_phrases,
            object_positions=overall_object_positions,
            semantic_guidance_kwargs=overall_semantic_guidance_kwargs,
        )

        print(
            f"Generation with spatial guidance from input latents and first {frozen_steps} steps frozen (background frozen so only boxes may change)"
        )
        print("Generation from composed latents (with semantic guidance)")

    utils.free_memory() 
    decode_latents_to_pil(composed_latents[-1]).save("test.png")
    decode_latents_to_pil(composed_latents[0]).save("test_bg.png")
    decode_latents_to_pil(regen_latents).save("result.png")

    return EasyDict(image=images[0], so_img_list=so_img_list)