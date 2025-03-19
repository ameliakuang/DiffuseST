from transformers import CLIPTextModel, CLIPTokenizer, logging
#from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from diffusers import PNDMScheduler
from diffusers.pipelines import BlipDiffusionPipeline

# suppress partial model loading warning
logging.set_verbosity_error()

import os
from PIL import Image
from tqdm import tqdm, trange
import torch
import torch.nn as nn
import argparse
from pathlib import Path
from pnp_utils_style import *
import torchvision.transforms as T
from preprocess_style import get_timesteps, Preprocess
from pnp_style import PNP, BLIP


def run(opt):
    model_key = "blipdiffusion"

    blip_diffusion_pipe = BLIP.from_pretrained(model_key, torch_dtype=torch.float16).to("cuda")
    
    scheduler = PNDMScheduler.from_pretrained(model_key, subfolder="scheduler")
    scheduler.set_timesteps(opt.ddpm_steps)
    content_path = Path(opt.content_path)
    content_path = [f for f in content_path.glob('*')]
    style_path = Path(opt.style_path)
    style_path = [f for f in style_path.glob('*')]

    extraction_path = "latents_reverse" if opt.extract_reverse else "latents_forward"
    base_save_path = os.path.join(opt.output_dir, extraction_path)
    os.makedirs(base_save_path, exist_ok=True)

    pnp = PNP(blip_diffusion_pipe, opt)

    all_content_latents = []
    for content_file in content_path:
        timesteps_to_save, num_inference_steps = get_timesteps(
            scheduler, num_inference_steps=opt.ddpm_steps,
            strength=1.0,
            device=opt.device
        )

        seed_everything(opt.seed)
        if opt.steps_to_save < opt.ddpm_steps:
            timesteps_to_save = timesteps_to_save[-opt.steps_to_save:]

        model = Preprocess(blip_diffusion_pipe, opt.device, scheduler=scheduler, sd_version=opt.sd_version, hf_key=None)
        
        save_path = os.path.join(base_save_path, os.path.splitext(os.path.basename(content_file))[0])
        os.makedirs(save_path, exist_ok=True)
        check_path = os.path.join(save_path, 'noisy_latents_0.pt')

        if not os.path.exists(check_path):
            print(f"No available latents, start extraction for {content_file}")
            _, content_latents = model.extract_latents(
                data_path=content_file,
                num_steps=opt.ddpm_steps,
                save_path=save_path,
                timesteps_to_save=timesteps_to_save,
                inversion_prompt=opt.inversion_prompt,
                extract_reverse=opt.extract_reverse
            )
        else:
            content_latents = []
            for t in range(opt.ddpm_steps):
                latents_path = os.path.join(save_path, f'noisy_latents_{t}.pt')
                if os.path.exists(latents_path):
                    content_latents.append(torch.load(latents_path))
            content_latents = torch.cat(content_latents, dim=0).to("cuda")
        all_content_latents.append(content_latents)

        all_style_latents = []
        for style_file in style_path:
            
            save_path = os.path.join(base_save_path, os.path.splitext(os.path.basename(style_file))[0])
            os.makedirs(save_path, exist_ok=True)
            check_path = os.path.join(save_path, f'noisy_latents_0.pt')
            if not os.path.exists(check_path):
                print(f"No available latents, start extraction for {style_file}")
                timesteps_to_save = timesteps_to_save[-int(opt.ddpm_steps*opt.alpha):]
                model.scheduler.set_timesteps(opt.ddpm_steps)

                _, style_latents = model.extract_latents(
                    data_path=style_file,
                    num_steps=opt.ddpm_steps,
                    save_path=save_path,
                    timesteps_to_save=timesteps_to_save,
                    inversion_prompt=opt.inversion_prompt,
                    extract_reverse=opt.extract_reverse
                )

            else:
                style_latents = []
                for t in range(int(opt.ddpm_steps * opt.alpha)):
                    latents_path = os.path.join(save_path, f'noisy_latents_{t}.pt')
                    if os.path.exists(latents_path):
                        style_latents.append(torch.load(latents_path))
                style_latents = torch.cat(style_latents, dim=0).to("cuda")
            all_style_latents.append(style_latents)
            
        
    for content_latents, content_file in zip(all_content_latents, content_path):
        for style_latents, style_file in zip(all_style_latents, style_path):
            pnp.run_pnp(content_latents, style_latents, style_file, content_fn=content_file, style_fn=style_file)
            torch.cuda.empty_cache()

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--content_path', type=str,
                        default='images/content')
    parser.add_argument('--style_path', type=str,
                        default='images/style')
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--sd_version', type=str, default='2.1', choices=['1.5', '2.0', '2.1'],
                        help="stable diffusion version")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--alpha', type=int, default=0.1)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--ddpm_steps', type=int, default=999)
    parser.add_argument('--steps_to_save', type=int, default=1000)
    parser.add_argument('--ddim_steps', type=int, default=50)
    
    parser.add_argument('--inversion_prompt', type=str, default='')
    parser.add_argument('--extract-reverse', default=False, action='store_true', help="extract features during the denoising process")
    opt = parser.parse_args()

    run(opt)