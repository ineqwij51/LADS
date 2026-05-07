import subprocess
import json
import pandas as pd
import os
import glob
import sys
import argparse

# --- Configuration ---
# IMPORTANT: Please confirm the dataset path and feature type
DATA_PATH = "autism_multimodal_dataset_20250726.pkl" 
FEATURE = "skeleton"
BASE_OUTPUT_DIR = "./comparison_Regularization"
SEED = 42
W_VALUE = 12 # Use the optimal W value determined previously
EPOCHS = 50
# ---------------------

# Define the regularization configurations to test
# Format: ConfigName: (use_diff, dropout, weight_decay, DisplayLabel)
CONFIGS = {
    "LADS_Full": (1, 0.1, 1e-4, "Full LADS (with Diffusion)"),
    "Baseline_NoDiff": (0, 0.1, 1e-4, "Baseline (No Diffusion)"),
    "Dropout_0.5": (0, 0.5, 1e-4, "Increased Dropout (0.5)"),
    "WD_1e-2": (0, 0.1, 1e-2, "Stronger L2 Decay (1e-2)"),
}

def check_prerequisites():
    """Checks if necessary files exist."""
    if not os.path.exists(DATA_PATH):
        print(f"Error: Dataset file not found at {DATA_PATH}")
        print("Please update DATA_PATH in the script.")
        return False
        
    if not os.path.exists("Diffusion_new_patched.py"):
        print("Error: Diffusion_new_patched.py not found in the current directory.")
        return False
    return True

def run_experiment(config_name, config_params):
    """Executes the main script for a specific configuration."""
    use_diff, dropout, weight_decay, _ = config_params
    
    # Set a unique output directory for this specific run
    output_dir = os.path.join(BASE_OUTPUT_DIR, config_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*40}\nStarting experiment: {config_name}\n{'='*40}")
    print(f"Params: use_diff={use_diff}, dropout={dropout}, weight_decay={weight_decay}")

    # Base arguments (matching the paper's best configuration for LADS)
    command = [
        "python", "Diffusion_new_patched.py",
        "--data_path", DATA_PATH,
        "--features", FEATURE,
        "--epochs", str(EPOCHS),
        "--batch_size", "8",
        "--lr", "3e-4",
        "--root_out", output_dir
    ]

    # Add the specific regularization parameters
    command.extend([
        "--use_diff", str(use_diff),
        "--dropout", str(dropout),
        "--weight_decay", str(weight_decay),
    ])

    # Execute the script
    try:
        # We use Popen to capture and print the output in real-time during long experiments
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        print(f"Executing command: {' '.join(command)}\n")
        
        # Print output line by line
        if process.stdout:
            for line in process.stdout:
                print(line, end='')
        
        process.wait()
        
        if process.returncode != 0:
            print(f"Error running experiment {config_name}. Return code: {process.returncode}")
            return False
            
        print(f"\nFinished experiment: {config_name}")
        return True
        
    except FileNotFoundError:
        print("Error: Python interpreter not found.")
        return False

def aggregate_results():
    """Aggregates the results from the generated JSON files."""
    print("\nAggregating results...")
    results = []

    # Iterate in the defined order
    for config_name, config_params in CONFIGS.items():
        _, _, _, display_label = config_params
        
        # Search pattern to find the summary file
        # Results path structure: <root_out>/<feature>/<tag>/seed<SEED>/summary_overall.json
        # We use the unique config_name directory and a wildcard for the <tag>
        search_pattern = os.path.join(
            BASE_OUTPUT_DIR, config_name, FEATURE, "*", f"seed{SEED}", "summary_overall.json"
        )
        
        files = glob.glob(search_pattern)
        
        if files:
            # If multiple match (unlikely), take the first one
            summary_file = files[0] 
            try:
                with open(summary_file, 'r') as f:
                    data = json.load(f)
                
                results.append({
                    "Configuration": display_label,
                    "Accuracy": f"{data['mean_test_acc']:.4f} ± {data['std_test_acc']:.4f}",
                    "F1-Score": f"{data['mean_test_f1']:.4f} ± {data['std_test_f1']:.4f}",
                    "Acc_Mean_Sort": data['mean_test_acc'] # Hidden column for sorting
                })
            except Exception as e:
                print(f"Could not process summary file {summary_file}: {e}")
        else:
            print(f"Warning: Results not found for {config_name}. Searched: {search_pattern}")

    # Display the results
    if results:
        df = pd.DataFrame(results)
        # Sort by accuracy mean for the final presentation
        df = df.sort_values(by="Acc_Mean_Sort", ascending=False)
        
        print("\nAggregated Regularization Comparison Results (5-fold CV):")
        # Display the table suitable for the paper (Markdown format)
        print(df[['Configuration', 'Accuracy', 'F1-Score']].to_markdown(index=False))
    else:
        print("No results were aggregated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Regularization Comparison Study")
    parser.add_argument('--aggregate_only', action='store_true', help='Skip running experiments and only aggregate existing results.')
    args = parser.parse_args()

    if not check_prerequisites():
        sys.exit(1)

    if not args.aggregate_only:
        print(f"Starting regularization comparison study...")
        success_count = 0
        for config_name, config_params in CONFIGS.items():
            if run_experiment(config_name, config_params):
                success_count += 1
        
        if success_count == 0:
            print("\nNo experiments completed successfully.")
            # Do not exit, attempt aggregation anyway in case previous runs exist
        else:
            print("\nAll experiments finished.")

    aggregate_results()