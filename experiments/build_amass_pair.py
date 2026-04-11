import os
import sys
import subprocess

def main():
    ego_root = sys.argv[1] if len(sys.argv) > 1 else "data/processed_amass"
    exo_root = sys.argv[2] if len(sys.argv) > 2 else "data/processed_exo"
    extra_args = sys.argv[3:] if len(sys.argv) > 2 else sys.argv[1:] if len(sys.argv) > 1 else []
    
    amass_root = os.environ.get("AMASS_ROOT", "data/raw/amass")
    smpl_root = os.environ.get("SMPL_ROOT", "data/raw/models")
    ego_cache = os.environ.get("EGO_CACHE", "data/raw/amass_renders/ego")
    exo_cache = os.environ.get("EXO_CACHE", "data/raw/amass_renders/exo")
    render_device = os.environ.get("RENDER_DEVICE", "cuda")
    render_size = os.environ.get("RENDER_SIZE", "512")
    render_chunk_size = os.environ.get("RENDER_CHUNK_SIZE", "128")
    render_workers = os.environ.get("RENDER_WORKERS", "1")
    skip_quality_filter = os.environ.get("SKIP_QUALITY_FILTER", "0")
    
    print("Building AMASS pair")
    print(f"  ego_root: {ego_root}")
    print(f"  exo_root: {exo_root}")
    print(f"  amass_root: {amass_root}")
    print(f"  smpl_root: {smpl_root}")
    print(f"  render_device: {render_device}")
    print(f"  render_size: {render_size}")
    print(f"  render_chunk_size: {render_chunk_size}")
    print(f"  render_workers: {render_workers}")
    print(f"  skip_quality_filter: {skip_quality_filter}")
    
    cmd = [
        "python", "-m", "egomuscle.data.build_min_t_dataset",
        "--video-source", "amass_ego",
        "--output-root", ego_root,
        "--paired-output-root", exo_root,
        "--video-cache", ego_cache,
        "--paired-video-cache", exo_cache,
        "--amass-root", amass_root,
        "--smpl-model-root", smpl_root,
        "--render-device", render_device,
        "--render-size", render_size,
        "--render-chunk-size", render_chunk_size,
        "--render-workers", render_workers,
        "--workers", "4"
    ]
    
    if skip_quality_filter != "0":
        cmd.append("--skip-quality-filter")
        
    cmd.extend(extra_args)
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
