import os
import sys
import subprocess
from datetime import datetime

def main():
    config = sys.argv[1] if len(sys.argv) > 1 else "egomuscle/training/config.yaml"
    extra_args = sys.argv[2:] if len(sys.argv) > 1 else []
    
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = os.environ.get("SWEEP_ROOT", f"experiments/results/loss_sweep/{stamp}")
    os.makedirs(root, exist_ok=True)
    
    runs = [
        ("baseline", "1.0", "0.10", "0.04", "0.04"),
        ("low_var", "1.0", "0.10", "0.01", "0.04"),
        ("low_cov", "1.0", "0.10", "0.04", "0.01"),
        ("low_reg", "1.0", "0.05", "0.01", "0.01"),
        ("pred_only", "1.0", "0.00", "0.00", "0.00"),
    ]
    
    print(f"Loss sweep root: {root}")
    print(f"Config: {config}")
    
    for name, pred, temp, var, cov in runs:
        run_root = os.path.join(root, name)
        ckpt_dir = os.path.join(run_root, "checkpoints")
        log_dir = os.path.join(run_root, "logs")
        train_log = os.path.join(run_root, "train.log")
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        print(f"\n=== {name} ===")
        print(f"pred={pred} temp={temp} var={var} cov={cov}")
        
        cmd = [
            "python", "-m", "egomuscle.training.train",
            "--config", config,
            "--override", f"output_dir={ckpt_dir}",
            "--override", f"logging.save_dir={log_dir}",
            "--override", f"logging.run_name={name}",
            "--override", f"training.loss_weights.pred={pred}",
            "--override", f"training.loss_weights.temp={temp}",
            "--override", f"training.loss_weights.var={var}",
            "--override", f"training.loss_weights.cov={cov}"
        ] + extra_args
        
        with open(train_log, "w") as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
            
    print(f"\nFinished loss sweep: {root}")

if __name__ == "__main__":
    main()
