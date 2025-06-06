# ──────────────────────────────────────────────────────────────────────────────
# File: HunyuanVideo-Avatar/hymm_sp/sample_batch_simple.py
# ──────────────────────────────────────────────────────────────────────────────

import os
import time
import uuid
import torch
import numpy as np
from pathlib import Path
from loguru import logger
from einops import rearrange
import imageio

from hymm_sp.config import parse_args
from hymm_sp.sample_inference_audio import HunyuanVideoSampler

def main():
    args = parse_args()
    # We assume parse_args() has been modified so that:
    #   --prompt <string>   is required
    #   --ckpt   <path>     is required
    #   --save-path <dir>   is required
    #
    # (In your existing `config.py` you will need to add parser.add_argument("--prompt", ...) 
    #  and drop any reliance on a CSV for “input”.)

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the video sampler (text-to-video model)
    logger.info("Loading HunyuanVideoSampler from checkpoint: {}".format(args.ckpt))
    model = HunyuanVideoSampler.from_pretrained(args.ckpt, args=args, device=device)
    logger.info("Loaded HunyuanVideoSampler successfully.")

    # Prepare a single‐item “batch” dictionary
    prompt = args.prompt
    batch = {
        "text": [prompt],
        "fps": torch.tensor([25]),           # dummy fps
        "audio_len": torch.tensor([1]),      # dummy audio length
        "audio_path": [""],                  # no actual audio
        "videoid": [str(uuid.uuid4().hex)],  # random ID for naming output
        "image_path": [""]                   # if you occasionally pass a ref image, fill this
    }

    # Run the forward pass
    out = model.predict(args, batch, wav2vec=None, feature_extractor=None, align_instance=None)

    # Extract the first (and only) sample latent tensor
    sample_latent = out["samples"][0].unsqueeze(0)  # shape: (1, C, F, H, W)

    # Convert to CPU numpy frames
    video = rearrange(sample_latent[0], "c f h w -> f h w c")
    video = (video * 255.0).clamp(0,255).cpu().numpy().astype(np.uint8)

    # Write frames to a single MP4
    save_dir = Path(args.save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    vid_id = batch["videoid"][0]
    out_path = save_dir / f"{vid_id}.mp4"
    imageio.mimsave(out_path.as_posix(), video, fps=int(batch["fps"].item()))

    print(f"[SAMPLE_SIMPLE] Saved video to {out_path}")

if __name__ == "__main__":
    main()
