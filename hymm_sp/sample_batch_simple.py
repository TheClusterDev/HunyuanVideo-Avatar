import argparse
import os
import torch
import numpy as np
from pathlib import Path
from loguru import logger
from einops import rearrange
import imageio
import uuid

from hymm_sp.sample_inference_audio import HunyuanVideoSampler  # our main inference entry

def main():
    # ──────────────────────────────────────────────────────────────────────────
    # (1) Minimal argument parser
    # ──────────────────────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Minimal text→video sampler for HunyuanVideo-Avatar"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="HYVideo-T/2",
        help="Which key in HUNYUAN_VIDEO_CONFIG to use (e.g. 'HYVideo-T/2')."
    )
    parser.add_argument(
        "--text_projection",
        type=str,
        default="single_refiner",
        help="(Required by HYVideoDiffusionTransformer) text‐projection type"
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
    # (2) Inject the missing “training‐script” fields that HunyuanVideoSampler.from_pretrained expects
    # ──────────────────────────────────────────────────────────────────────────
    # In hymm_sp/inference.py, the code does something like:
    #     factor_kwargs = {'device': 'cpu' if args.cpu_offload else device, 'dtype': PRECISION_TO_TYPE[args.precision]}
    #
    # Therefore we must at least define:
    #     args.cpu_offload   (bool)
    #     args.precision     (one of “fp32”/“fp16”/“bf16”), so that PRECISION_TO_TYPE[args.precision] works.
    #
    # We’ll choose cpu_offload=False and precision="fp16" by default:
    args.cpu_offload = False
    args.precision = "fp16"

    # The sampler also needs to know which VAE / text encoders it should load.
    # The default HYVideo-T code expects at least these attributes on “args”:
    args.vae = "884-16c-hy0801"
    args.vae_precision = "fp16"
    args.latent_channels = None  # inference code will infer from vae name
    args.rope_theta = 256

    args.text_encoder = "llava-llama-3-8b"
    args.text_encoder_precision = "fp16"
    args.tokenizer = "llava-llama-3-8b"
    args.text_encoder_infer_mode = "encoder"
    args.hidden_state_skip_layer = 2
    args.apply_final_norm = False

    args.text_encoder_2 = "clipL"
    args.text_encoder_precision_2 = "fp16"
    args.tokenizer_2 = "clipL"
    args.text_states_dim = 4096
    args.text_states_dim_2 = 768

    # Certain flags that exist in the original training‐script but do not harm inference:
    args.reproduce = False
    args.prompt_template_video = None
    args.use_attention_mask = True
    args.pad_face_size = 0.7  # for face alignment; not used if no face present
    args.image_path = ""
    args.pos_prompt = ""
    args.neg_prompt = ""
    args.ip_cfg_scale = 0
    args.use_fp8 = False
    args.use_linear_quadratic_schedule = False
    args.flow_reverse = True

    # ──────────────────────────────────────────────────────────────────────────
    # (3) Set up device and load HunyuanVideoSampler
    # ──────────────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading HunyuanVideoSampler checkpoint from: {args.ckpt}")
    model = HunyuanVideoSampler.from_pretrained(
        args.ckpt,
        args=args,    # pass our “augmented” args namespace
        device=device
    )
    logger.info("Successfully loaded HunyuanVideoSampler.")

    # ──────────────────────────────────────────────────────────────────────────
    # (4) Build a minimal “batch” dict containing only our single prompt
    # ──────────────────────────────────────────────────────────────────────────
    # HunyuanVideoSampler.predict(...) expects a batch with:
    #   • "text":        a list of prompt strings
    #   • "fps":         a 1D tensor of length=batch_size
    #   • "audio_len":   a 1D tensor of length=batch_size
    #   • "audio_path":  a list of length=batch_size
    #   • "videoid":     a list of length=batch_size (used for final filename)
    #   • "image_path":  a list of reference‐image paths (empty if no reference)
    #
    # Since we only want one video, batch_size=1:
    batch = {
        "text": [args.prompt],
        "fps": torch.tensor([25]),            # dummy FPS
        "audio_len": torch.tensor([1]),       # dummy “audio length”
        "audio_path": [""],
        "videoid": [str(uuid.uuid4().hex)],   # random ID → output filename
        "image_path": [""],                   # no reference image
    }

    # ──────────────────────────────────────────────────────────────────────────
    # (5) Run the sampler’s predict(...) call
    # ──────────────────────────────────────────────────────────────────────────
    out = model.predict(
        args,
        batch,
        wav2vec=None,           # we are not using speech‐driven generation
        feature_extractor=None,
        align_instance=None
    )

    # “out['samples']” is a list with one tensor of shape (C, F, H, W).
    # We take that tensor, add a batch dimension, then convert → numpy frames:
    sample_latent = out["samples"][0].unsqueeze(0)   # shape = (1, C, F, H, W)
    video_np = rearrange(sample_latent[0], "c f h w -> f h w c")
    video_np = (video_np * 255.0).clamp(0, 255).cpu().numpy().astype(np.uint8)

    # ──────────────────────────────────────────────────────────────────────────
    # (6) Write a single .mp4 in --save-path
    # ──────────────────────────────────────────────────────────────────────────
    save_dir = Path(args.save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    vid_id = batch["videoid"][0]
    out_vid = save_dir / f"{vid_id}.mp4"

    imageio.mimsave(out_vid.as_posix(), video_np, fps=int(batch["fps"].item()))
    print(f"[SAMPLE_SIMPLE] Saved video to {out_vid}")

if __name__ == "__main__":
    main()
