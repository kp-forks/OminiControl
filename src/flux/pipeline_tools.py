from diffusers.pipelines import FluxPipeline
from diffusers.utils import logging
from diffusers.pipelines.flux.pipeline_flux import logger
from torch import Tensor


def encode_images(pipeline: FluxPipeline, images: Tensor):
    images = pipeline.image_processor.preprocess(images)
    images = images.to(pipeline.device).to(pipeline.dtype)
    images = pipeline.vae.encode(images).latent_dist.sample()
    images = (
        images - pipeline.vae.config.shift_factor
    ) * pipeline.vae.config.scaling_factor
    images_tokens = pipeline._pack_latents(images, *images.shape)
    try:
        images_ids = pipeline._prepare_latent_image_ids(
            batch_size=images.shape[0],
            height=images.shape[2],
            width=images.shape[3],
            device=pipeline.device,
            dtype=pipeline.dtype,
        )
    except:
        images_ids = pipeline._prepare_latent_image_ids(
            height=images.shape[2],
            width=images.shape[3],
            device=pipeline.device,
            dtype=pipeline.dtype,
        )
    return images_tokens, images_ids


def prepare_text_input(pipeline: FluxPipeline, prompts, max_sequence_length=512):
    # Turn off warnings (CLIP overflow)
    logger.setLevel(logging.ERROR)
    (
        prompt_embeds,
        pooled_prompt_embeds,
        text_ids,
    ) = pipeline.encode_prompt(
        prompt=prompts,
        prompt_2=None,
        prompt_embeds=None,
        pooled_prompt_embeds=None,
        device=pipeline.device,
        num_images_per_prompt=1,
        max_sequence_length=max_sequence_length,
        lora_scale=None,
    )
    # Turn on warnings
    logger.setLevel(logging.WARNING)
    return prompt_embeds, pooled_prompt_embeds, text_ids
