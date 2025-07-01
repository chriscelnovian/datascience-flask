import os
from datetime import datetime

def log_model_results(results, best_model_name, model_path, log_dir="artifacts/logs"):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"model_log_{timestamp}.txt"
    log_path = os.path.join(log_dir, log_filename)

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("=== Model Evaluation Log ===\n")
        f.write(f"Timestamp   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 35 + "\n\n")

        for name, metrics in results.items():
            f.write(f"Model       : {name}\n")
            f.write(f"  Accuracy  : {metrics['accuracy']:.4f}\n")
            f.write(f"  Precision : {metrics['precision']:.4f}\n")
            f.write(f"  Recall    : {metrics['recall']:.4f}\n")
            f.write(f"  F1 Score  : {metrics['f1_score']:.4f}\n")
            f.write("\n")

        f.write("=" * 35 + "\n")
        f.write(f"Best Model  : {best_model_name}\n")
        f.write(f"F1 Score    : {results[best_model_name]['f1_score']:.4f}\n")
        f.write(f"Model Saved : {model_path}\n")
        f.write("=" * 35 + "\n")

    return log_path
