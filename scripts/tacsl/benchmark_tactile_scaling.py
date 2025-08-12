#!/usr/bin/env python3
"""
Benchmark analysis script to visualize tactile simulation scaling results.

This script loads pre-computed benchmark results from res_distance.txt
and generates scaling performance visualizations and analysis.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import glob





def load_results_from_file(filename):
    """
    Load benchmark results from file.

    Args:
        filename (str): Path to the results file

    Returns:
        list: List of FPS metrics dictionaries
    """
    results = []
    
    try:
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if line.startswith('#') or not line:
                    continue
                
                try:
                    # Parse the dictionary from the line
                    result = eval(line)
                    
                    # Map the keys to match what the plotting function expects
                    mapped_result = {
                        'num_envs': result['num_envs'],
                        'total_fps': result['total_fps'],
                        'physics_fps': result['physics_fps'],
                        'tactile_img_fps': result['camera_fps'],  # Map camera_fps to tactile_img_fps
                        'tactile_ff_fps': result['force_field_fps']  # Map force_field_fps to tactile_ff_fps
                    }
                    
                    results.append(mapped_result)
                    
                except (SyntaxError, KeyError, TypeError) as e:
                    print(f"Error parsing line: {line}")
                    print(f"Error: {e}")
                    continue
    
    except FileNotFoundError:
        print(f"Error: Results file '{filename}' not found")
        return []
    
    # Group by num_envs and average the results for each environment count
    grouped_results = {}
    for result in results:
        num_envs = result['num_envs']
        if num_envs not in grouped_results:
            grouped_results[num_envs] = []
        grouped_results[num_envs].append(result)
    
    # Calculate averages for each environment count
    averaged_results = []
    for num_envs, env_results in grouped_results.items():
        if len(env_results) == 1:
            averaged_results.append(env_results[0])
        else:
            # Average multiple runs for the same environment count
            avg_result = {'num_envs': num_envs}
            for key in ['total_fps', 'physics_fps', 'tactile_img_fps', 'tactile_ff_fps']:
                avg_result[key] = sum(r[key] for r in env_results) / len(env_results)
            averaged_results.append(avg_result)
    
    # Sort by number of environments
    averaged_results.sort(key=lambda x: x['num_envs'])
    
    return averaged_results


def plot_scaling_results(results, output_dir=".", suffix=""):
    """
    Create visualizations showing scaling performance.

    Args:
        results (list): List of FPS metrics dictionaries
        output_dir (str): Directory to save the plot files
    """
    if not results:
        print("No results to plot")
        return

    # Extract data for plotting
    num_envs = [r['num_envs'] for r in results]
    total_fps = [r['total_fps'] for r in results]
    physics_fps = [r['physics_fps'] for r in results]
    tactile_img_fps = [r['tactile_img_fps'] for r in results]
    tactile_ff_fps = [r['tactile_ff_fps'] for r in results]

    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: Overall FPS scaling
    ax1.loglog(num_envs, total_fps, 'o-', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Environments')
    ax1.set_ylabel('Total FPS')
    ax1.set_title('Total FPS vs Number of Environments')
    ax1.set_xticks(num_envs)
    ax1.set_xticklabels([str(n) for n in num_envs])
    ax1.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=10))
    ax1.yaxis.set_minor_locator(ticker.LogLocator(base=10, subs=[2, 3, 4, 5, 6, 7, 8, 9], numticks=50))
    ax1.grid(True, alpha=0.3)

    # Plot 2: Physics FPS scaling
    ax2.loglog(num_envs, physics_fps, 'o-', color='orange', linewidth=2,
                 markersize=8)
    ax2.set_xlabel('Number of Environments')
    ax2.set_ylabel('Physics FPS')
    ax2.set_title('Physics FPS vs Number of Environments')
    ax2.set_xticks(num_envs)
    ax2.set_xticklabels([str(n) for n in num_envs])
    ax2.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=10))
    ax2.yaxis.set_minor_locator(ticker.LogLocator(base=10, subs=[2, 3, 4, 5, 6, 7, 8, 9], numticks=50))
    ax2.grid(True, alpha=0.3)

    # Plot 3: Tactile RGB FPS scaling
    ax3.loglog(num_envs, tactile_img_fps, 'o-', color='green',
                 linewidth=2, markersize=8)
    ax3.set_xlabel('Number of Environments')
    ax3.set_ylabel('Tactile RGB FPS')
    ax3.set_title('Tactile RGB FPS vs Number of Environments')
    ax3.set_xticks(num_envs)
    ax3.set_xticklabels([str(n) for n in num_envs])
    ax3.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=10))
    ax3.yaxis.set_minor_locator(ticker.LogLocator(base=10, subs=[2, 3, 4, 5, 6, 7, 8, 9], numticks=50))
    ax3.grid(True, alpha=0.3)

    # Plot 4: Tactile Force Field FPS scaling
    ax4.loglog(num_envs, tactile_ff_fps, 'o-', color='red', linewidth=2,
                 markersize=8)
    ax4.set_xlabel('Number of Environments')
    ax4.set_ylabel('Tactile Force Field FPS')
    ax4.set_title('Tactile Force Field FPS vs Number of Environments')
    ax4.set_xticks(num_envs)
    ax4.set_xticklabels([str(n) for n in num_envs])
    ax4.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=10))
    ax4.yaxis.set_minor_locator(ticker.LogLocator(base=10, subs=[2, 3, 4, 5, 6, 7, 8, 9], numticks=50))
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    filename = f'tactile_scaling_results{suffix}.png' if suffix else 'tactile_scaling_results.png'
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    # plt.show()

    # Create a comparison plot showing all metrics together
    fig2, ax = plt.subplots(1, 1, figsize=(12, 8))

    ax.loglog(num_envs, total_fps, 'o-', label='Total FPS', linewidth=2,
                markersize=8)
    ax.loglog(num_envs, physics_fps, 'o-', label='Physics FPS',
                linewidth=2, markersize=8)
    ax.loglog(num_envs, tactile_img_fps, 'o-', label='Tactile RGB FPS',
                linewidth=2, markersize=8)
    ax.loglog(num_envs, tactile_ff_fps, 'o-',
                label='Tactile Force Field FPS', linewidth=2, markersize=8)

    ax.set_xlabel('Number of Environments')
    ax.set_ylabel('FPS')
    ax.set_title('Tactile Simulation FPS Scaling Comparison')
    ax.set_xticks(num_envs)
    ax.set_xticklabels([str(n) for n in num_envs])
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=10))
    ax.yaxis.set_minor_locator(ticker.LogLocator(base=10, subs=[2, 3, 4, 5, 6, 7, 8, 9], numticks=50))
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    filename = f'tactile_scaling_comparison{suffix}.png' if suffix else 'tactile_scaling_comparison.png'
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    # plt.show()

    # Calculate and display scaling efficiency
    print("\n" + "=" * 60)
    print("SCALING ANALYSIS")
    print("=" * 60)

    base_envs = num_envs[0]
    base_total_fps = total_fps[0]

    print(f"{'Num Envs':<10} {'Total FPS':<12} {'Scaling Factor':<15} "
          f"{'Efficiency':<12}")
    print("-" * 60)

    for i, (envs, fps) in enumerate(zip(num_envs, total_fps)):
        scaling_factor = envs / base_envs
        efficiency = (fps / base_total_fps) / scaling_factor * 100
        print(f"{envs:<10} {fps:<12.2f} {scaling_factor:<15.1f} "
              f"{efficiency:<12.1f}%")


def main():
    """Main function to run the scaling benchmark analysis."""
    print("Tactile Simulation Scaling Benchmark Analysis")
    print("=" * 50)

    # Find the newest result file based on timestamped filename
    import glob
    
    # Get the current script directory and construct relative path to benchmark dir
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(current_script_dir, ".." )
    benchmark_dir = os.path.join(project_root, "tactile_record", "benchmark")
    pattern = os.path.join(benchmark_dir, "res_txt", "res-*.txt")
    result_files = glob.glob(pattern)
    
    if not result_files:
        # Fallback to old filename if no timestamped files found
        results_file = os.path.join(benchmark_dir, "res.txt")
        if not os.path.exists(results_file):
            print(f"Error: No result files found in {benchmark_dir}")
            print("Looking for files matching pattern: res-YYYYMMDD_HHMMSS.txt")
            return
    else:
        # Sort by filename (which includes timestamp) and get the newest
        result_files.sort()
        results_file = result_files[-1]  # Last in sorted order is newest
        
    print(f"Using result file: {os.path.basename(results_file)}")
    
    # Extract suffix from filename (e.g., "20241223_142030" from "res-20241223_142030.txt")
    filename = os.path.basename(results_file)
    if filename.startswith("res-") and filename.endswith(".txt"):
        suffix = "-" + filename[4:-4]  # Extract the part between "res-" and ".txt"
    else:
        suffix = ""  # For fallback files like "res.txt"
    
    # Check if results file exists
    if not os.path.exists(results_file):
        print(f"Error: Results file '{results_file}' not found")
        print("Please make sure the results file is in the current directory.")
        return

    print(f"Loading results from {results_file}...")
    results = load_results_from_file(results_file)

    if results:
        print(f"\nSuccessfully loaded results for {len(results)} configurations")
        
        # Display summary of loaded results
        print("\nLoaded data summary:")
        print("-" * 40)
        for result in results:
            print(f"  {result['num_envs']} envs: Total FPS = {result['total_fps']:.2f}")
        
        print("\nGenerating visualizations...")
        plot_scaling_results(results, benchmark_dir, suffix)

        # Save processed results to JSON file
        import json
        json_filename = f'tactile_scaling_results{suffix}.json' if suffix else 'tactile_scaling_results.json'
        json_path = os.path.join(benchmark_dir, json_filename)
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nProcessed results saved to {json_path}")

    else:
        print("No valid results found in the file")


if __name__ == "__main__":
    main()
