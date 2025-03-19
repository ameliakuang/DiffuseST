import random
from diffusers.pipelines import BlipDiffusionPipeline
from diffusers import DDIMScheduler, PNDMScheduler
from diffusers.pipelines.blip_diffusion.pipeline_blip_diffusion import EXAMPLE_DOC_STRING
from diffusers.pipelines.pipeline_utils import ImagePipelineOutput
from diffusers.utils import load_image
from diffusers.utils.doc_utils import replace_example_docstring
import numpy as np
import torch
import glob
from typing import List, Optional, Union
import PIL.Image
import os
from pathlib import Path
import torch.nn as nn
import torchvision.transforms as T
import argparse
from PIL import Image
import yaml
from tqdm import tqdm
from transformers import logging
from pnp_utils_style import *
import time

def load_img1(self, image_path):
        image_pil = T.Resize(512)(Image.open(image_path).convert("RGB"))
        return image_pil


class PNP(nn.Module):
    def __init__(self, pipe, config):
        super().__init__()
        self.config = config

        self.device = config.device

        self.pipe = pipe
        self.pipe.scheduler.set_timesteps(config.n_timesteps, device=self.device)

    def init_pnp(self, conv_injection_t, qk_injection_t):
        self.qk_injection_timesteps = self.pipe.scheduler.timesteps[:qk_injection_t] if qk_injection_t >= 0 else []
        self.conv_injection_timesteps = self.pipe.scheduler.timesteps[:conv_injection_t] if conv_injection_t >= 0 else []
        register_attention_control_efficient(self.pipe, self.qk_injection_timesteps)
        register_conv_control_efficient(self.pipe, self.conv_injection_timesteps)
        return self.qk_injection_timesteps
    

    def run_pnp(self):
        content_paths = Path(self.config["image_path"])
        content_paths = [f for f in content_paths.glob('*')]
        style_paths = Path(self.config["image_path1"])
        style_paths = [f for f in style_paths.glob('*')]

        all_times = []
        for content_path in content_paths:
            for style_path in style_paths:
                start_time = time.time()
                
                pnp_f_t = int(self.config["n_timesteps"] * self.config["alpha"])
                pnp_attn_t = int(self.config["n_timesteps"] * self.config["alpha"])
                content_step = self.init_pnp(conv_injection_t=pnp_f_t, qk_injection_t=pnp_attn_t)
                cond_subject = ""
                tgt_subject = ""
                text_prompt_input = ""
                # print(1)
                # print(style_path)
                cond_image = load_img1(self,style_path)
                guidance_scale = 7.5
                num_inference_steps = 50
                negative_prompt = "over-exposure, under-exposure, saturated, duplicate, out of frame, lowres, cropped, worst quality, low quality, jpeg artifacts, morbid, mutilated, out of frame, ugly, bad anatomy, bad proportions, deformed, blurry, duplicate"
                latents_path = os.path.join(self.config["latents_path"], os.path.splitext(os.path.basename(content_path))[0], f'noisy_latents_{self.pipe.scheduler.timesteps[0]}.pt')
                latent = torch.load(latents_path).to(self.device)

                output = self.pipe(
                    content_path,
                    style_path,
                    text_prompt_input,
                    cond_image,
                    cond_subject,
                    tgt_subject,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    neg_prompt=negative_prompt,
                    latents=latent,
                    height=512,
                    width=512,
                    content_step=content_step,
                ).images
                end_time = time.time()

                all_times.append(end_time-start_time)
                content_path1 = os.path.basename(content_path)
                style_path1 = os.path.basename(style_path)
                print(content_path1)
                print(style_path1)
                output[0].save(f'{config["output_path"]}/{content_path1}_+_{style_path1}.png')
        print(all_times)
        print(np.array(all_times).mean())

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
class BLIP(BlipDiffusionPipeline):    
    @torch.no_grad()
    def __call__(
        self,
        content_path,
        style_path,
        prompt: List[str],
        reference_image: PIL.Image.Image,
        source_subject_category: List[str],
        target_subject_category: List[str],
        latents: Optional[torch.FloatTensor] = None,
        guidance_scale: float = 7.5,
        content_step = None,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        neg_prompt: Optional[str] = "",
        prompt_strength: float = 1.0,
        prompt_reps: int = 20,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ):
        
        device = self._execution_device

        reference_image = self.image_processor.preprocess(
            reference_image, image_mean=self.config.mean, image_std=self.config.std, return_tensors="pt"
        )["pixel_values"]
        reference_image = reference_image.to(device)

        if isinstance(prompt, str):
            prompt = [prompt]
        if isinstance(source_subject_category, str):
            source_subject_category = [source_subject_category]
        if isinstance(target_subject_category, str):
            target_subject_category = [target_subject_category]

        batch_size = len(prompt)

        prompt = self._build_prompt(
            prompts=prompt,
            tgt_subjects=target_subject_category,
            prompt_strength=prompt_strength,
            prompt_reps=prompt_reps,
        )
        query_embeds = self.get_query_embeddings(reference_image, source_subject_category)
        text_embeddings = self.encode_prompt(query_embeds, prompt, device)
        do_classifier_free_guidance = guidance_scale > 1.0
        if do_classifier_free_guidance:
            max_length = self.text_encoder.text_model.config.max_position_embeddings

            uncond_input = self.tokenizer(
                [neg_prompt] * batch_size,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )
            uncond_embeddings = self.text_encoder(
                input_ids=uncond_input.input_ids.to(device),
                ctx_embeddings=None,
            )[0]
            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            print(uncond_embeddings.shape)
            print(text_embeddings.shape)
            text_embeddings = torch.cat([uncond_embeddings, uncond_embeddings, text_embeddings])

        scale_down_factor = 2 ** (len(self.unet.config.block_out_channels) - 1)

        extra_set_kwargs = {}
        self.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)
        #print(self.unet)
        for i, t in enumerate(self.progress_bar(self.scheduler.timesteps)):
            # expand the latents if we are doing classifier free guidance
            register_time(self, t.item())
            do_classifier_free_guidance = guidance_scale > 1.0
            source_latents = load_source_latents_t(t, os.path.join(config["latents_path"], os.path.splitext(os.path.basename(content_path))[0]))
            style_latents = load_source_latents_t(t, os.path.join(config["latents_path"], os.path.splitext(os.path.basename(style_path))[0]))

            if t in content_step:
                latent_model_input = torch.cat([source_latents] + [latents] * 2 ) if do_classifier_free_guidance else latents
            else:
                latent_model_input = torch.cat([style_latents] + [latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = torch.tensor(latent_model_input, dtype=torch.float16)
            #print(latent_model_input.shape,text_embeddings.shape)
            noise_pred = self.unet(
                latent_model_input,
                timestep=t,
                encoder_hidden_states=text_embeddings,
                
                down_block_additional_residuals=None,
                mid_block_additional_residual=None,
            )["sample"]

            # perform guidance
            if do_classifier_free_guidance:
                _, noise_pred_uncond, noise_pred_text = noise_pred.chunk(3)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = self.scheduler.step(
                noise_pred,
                t,
                latents,
            )["prev_sample"]
            
        latents = (latents).half()
        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        image = self.image_processor.postprocess(image, output_type=output_type)
        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='config_pnp.yaml')
    opt = parser.parse_args()
    with open(opt.config_path, "r") as f:
        config = yaml.safe_load(f)
    os.makedirs(config["output_path"], exist_ok=True)
    with open(os.path.join(config["output_path"], "config.yaml"), "w") as f:
        yaml.dump(config, f)
    
    seed_everything(config["seed"])
    print(config)
    model_key = "/mnt/hdd/ZhuangChenyi/pretrained_models/application/blipdiffusion"
    blip_diffusion_pipe = BLIP.from_pretrained(model_key, torch_dtype=torch.float16).to("cuda")
    scheduler = PNDMScheduler.from_pretrained(model_key, subfolder="scheduler")
    scheduler.set_timesteps(config["n_timesteps"])

    pnp = PNP(blip_diffusion_pipe, config)

    pnp.run_pnp()