import os
import sys
import subprocess

def main():
    recenter_root = sys.argv[1] if len(sys.argv) > 1 else "data/processed_body"
    exo_root = sys.argv[2] if len(sys.argv) > 2 else "data/processed_exo_real"
    extra_args = sys.argv[3:] if len(sys.argv) > 2 else sys.argv[1:] if len(sys.argv) > 1 else []
    
    raw_cache = os.environ.get("RAW_CACHE", "data/raw/babel_renders")
    body_cache = os.environ.get("BODY_CACHE", "data/raw/babel_recentered")
    reframe_device = os.environ.get("REFRAME_DEVICE", "cuda")
    reframe_size = os.environ.get("REFRAME_SIZE", "512")
    reframe_workers = os.environ.get("REFRAME_WORKERS", "4")
    reframe_detect_every = os.environ.get("REFRAME_DETECT_EVERY", "6")
    reframe_crop_scale = os.environ.get("REFRAME_CROP_SCALE", "1.8")
    reframe_min_score = os.environ.get("REFRAME_MIN_SCORE", "0.65")
    workers = os.environ.get("WORKERS", "16")
    skip_quality_filter = os.environ.get("SKIP_QUALITY_FILTER", "0")
    
    print("Building BABEL view pair")
    print(f"  recenter_root: {recenter_root}")
    print(f"  exo_root: {exo_root}")
    print(f"  raw_cache: {raw_cache}")
    print(f"  body_cache: {body_cache}")
    print(f"  reframe_device: {reframe_device}")
    print(f"  reframe_size: {reframe_size}")
    print(f"  reframe_workers: {reframe_workers}")
    print(f"  detect_every: {reframe_detect_every}")
    print(f"  crop_scale: {reframe_crop_scale}")
    print(f"  min_score: {reframe_min_score}")
    print(f"  workers: {workers}")
    print(f"  skip_quality_filter: {skip_quality_filter}")
    print("  note: babel_recenter is a detector-based body-centered crop, not the AMASS peripersonal renderer used for E3.")
    
    cmd = [
        "python", "-m", "egomuscle.data.build_min_t_dataset",
        "--video-source", "babel_recenter",
        "--output-root", recenter_root,
        "--paired-output-root", exo_root,
        "--video-cache", body_cache,
        "--paired-video-cache", raw_cache,
        "--render-device", reframe_device,
        "--render-size", reframe_size,
        "--render-workers", reframe_workers,
        "--recenter-detect-every", reframe_detect_every,
        "--recenter-crop-scale", reframe_crop_scale,
        "--recenter-min-score", reframe_min_score,
        "--workers", workers
    ]
    
    if skip_quality_filter != "0":
        cmd.append("--skip-quality-filter")
        
    cmd.extend(extra_args)
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
