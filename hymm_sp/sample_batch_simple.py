# ──────────────────────────────────────────────────────────────────────────────
# File: HunyuanVideo-Avatar/hymm_sp/sample_batch_simple.py
# ──────────────────────────────────────────────────────────────────────────────

import argparse
import os
import torch
import numpy as np
from pathlib import Path
from loguru import logger
from einops import rearrange
import imageio
import uuid

# We assume that HunyuanVideoSampler is in your PYTHONPATH under hymm_sp/sample_inference_audio.py
from hymm_sp.sample_inference_audio import HunyuanVideoSampler

def main():
    # ──────────────────────────────────────────────────────────────────────────
    # 1) Parse only the arguments we actually need for inference.
    # ──────────────────────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Minimal text→video sampler for HunyuanVideo-Avatar"
    )
    parser.add_argument(
        "--prompt", 
        type=str, 
        required=True,
        help="Text prompt (e.g. 'A cute cat riding a skateboard')"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to the HunyuanVideo-T2V checkpoint (e.g. weights/ckpts/.../mp_rank_00_model_states.pt)"
    )
    parser.add_argument(
        "--save-path",
        type=str,
        required=True,
        help="Directory where the resulting .mp4 should be saved"
    )
    parser.add_argument(
        "--sample-n-frames",
        type=int,
        default=129,
        help="(Optional) number of frames to sample. Default=129."
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=704,
        help="(Optional) spatial resolution (height=width). Default=704."
    )
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=7.5,
        help="(Optional) classifier-free guidance scale. Default=7.5."
    )
    parser.add_argument(
        "--infer-steps",
        type=int,
        default=50,
        help="(Optional) number of denoising steps. Default=50."
    )
    parser.add_argument(
        "--use-deepcache",
        type=int,
        default=1,
        help="(Optional) use deepcache (1) or not (0). Default=1."
    )
    parser.add_argument(
        "--flow-shift-eval-video",
        type=float,
        default=5.0,
        help="(Optional) flow-shift factor for video schedule. Default=5.0."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="(Optional) random seed. Default=42."
    )
    args = parser.parse_args()

    # ──────────────────────────────────────────────────────────────────────────
    # 2) Set up device and load the HunyuanVideoSampler model
    # ──────────────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading HunyuanVideoSampler checkpoint from: {args.ckpt}")
    model = HunyuanVideoSampler.from_pretrained(
        args.ckpt,
        args=args,       # passes all the extra flags (sample‐n‐frames, image‐size, etc.)
        device=device
    )
    logger.info("Successfully loaded HunyuanVideoSampler.")

    # ──────────────────────────────────────────────────────────────────────────
    # 3) Build a fake “batch” dictionary containing only the prompt
    # ──────────────────────────────────────────────────────────────────────────
    # We only need a single‐element batch. We give it:
    #   • text: [args.prompt]
    #   • fps:   we can hardcode 25 (or any positive integer)
    #   • audio_len: set to 1 (dummy; there’s no actual audio)
    #   • videoid: random UUID so that the final MP4 has a unique name
    #   • image_path: an empty string (we’re not using a reference image here)
    batch = {
        "text": [args.prompt],
        "fps": torch.tensor([25]),             # dummy frames-per-second
        "audio_len": torch.tensor([1]),        # dummy “audio length”
        "audio_path": [""],                    # no real audio file
        "videoid": [str(uuid.uuid4().hex)],    # random ID → output filename
        "image_path": [""]                     # only used if you ever pass a reference image
    }

    # ──────────────────────────────────────────────────────────────────────────
    # 4) Run the “predict” call
    # ──────────────────────────────────────────────────────────────────────────
    out = model.predict(
        args,
        batch,
        wav2vec=None,
        feature_extractor=None,
        align_instance=None
    )

    # “out['samples']” is a list containing one tensor of shape (C, F, H, W).
    # We grab that tensor, add a batch dimension, then convert → numpyframes.
    sample_latent = out["samples"][0].unsqueeze(0)  # (1, C, F, H, W)

    # Rearrange to (F, H, W, C) and scale to [0,255]
    video_np = rearrange(sample_latent[0], "c f h w -> f h w c")
    video_np = (video_np * 255.0).clamp(0, 255).cpu().numpy().astype(np.uint8)

    # ──────────────────────────────────────────────────────────────────────────
    # 5) Write a single .mp4 in --save-path
    # ──────────────────────────────────────────────────────────────────────────
    save_dir = Path(args.save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    vid_id = batch["videoid"][0]
    out_vid = save_dir / f"{vid_id}.mp4"

    # imageio.mimsave will write all frames into a single mp4
    imageio.mimsave(out_vid.as_posix(), video_np, fps=int(batch["fps"].item()))
    print(f"[SAMPLE_SIMPLE] Saved video to {out_vid}")

if __name__ == "__main__":
    main()
