import os
import glob
import torch
import numpy as np
import pandas as pd
import tqdm
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# --- UTILITY FUNCTIONS ---

def calculate_entropy(labels):
    """Calculates the Shannon entropy of a label distribution."""
    if len(labels) == 0:
        return 0
    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

# --- CORE LOGIC: PROCESSES A SINGLE REGION ---

def process_region_zeroshot(region_filename, processed_data_dir, num_total_classes):
    """
    Processes a single Matterport region .pth file.
    """
    pth_file = os.path.join(processed_data_dir, region_filename)

    try:
        # Your script saves a tuple: (coords, colors, normal, vertex_labels)
        # We only need the labels, which are the 4th element (index 3).
        # _, _, _, labels_in = torch.load(pth_file)
        _, _, labels_in = torch.load(pth_file)
        # Check if labels_in is a numpy array, if so, convert it to int64 using numpy.
        if isinstance(labels_in, np.ndarray):
            # Check if the dtype is not int64, and convert it if necessary.
            if labels_in.dtype != np.int64:
                # print(f"Converting labels_in from {labels_in.dtype} to np.int64")
                labels_in = labels_in.astype(np.int64)
    except Exception as e:
        print(f"Warning: Could not load or parse {pth_file}: {e}")
        return None

    # --- Label Preprocessing ---
    # Valid labels are 0-20, and 255 is the ignored class.
    valid_labels = labels_in[labels_in != 255]

    if len(valid_labels) == 0:
        return None

    # --- Metric Calculation ---
    nc = len(np.unique(valid_labels))
    hc = calculate_entropy(valid_labels)
    full_histogram = np.bincount(valid_labels, minlength=num_total_classes).astype(np.float32)

    # The "scene" key now stores the unique region name without the extension
    region_name = os.path.splitext(region_filename)[0]

    return {
        "scene": region_name,
        "Nc": nc,
        "Hc": hc,
        "full_histogram": full_histogram,
    }

def load_or_compute_metrics(input_file, processed_data_dir, num_total_classes, output_file, num_workers):
    """Loads region metrics from a file, or computes them by scanning for all .pth files."""
    if os.path.exists(output_file):
        print(f"Loading pre-computed region metrics from {output_file}")
        df = pd.read_csv(output_file, sep='\t')
        df['full_histogram'] = df['full_histogram'].apply(lambda x: np.fromstring(x.strip('[]'), sep=' '))
        results = df.to_dict('records')
    else:
        print(f"Computing region metrics and saving to {output_file}")

        if os.path.exists(input_file):
            print(f"Loading region list from {input_file}")
            with open(input_file, 'r') as f:
                # The input file should list region names without extension (e.g., scene1_region0)
                region_names = [line.strip() for line in f.readlines() if line.strip()]
        else:
            print(f"Input file '{input_file}' not found.")
            print(f"Scanning for all .pth regions in: {processed_data_dir}")
            try:
                # List all .pth files directly and remove extension for the name list
                region_names = sorted([os.path.splitext(f)[0] for f in os.listdir(processed_data_dir) if f.endswith('.pth')])
            except FileNotFoundError:
                print(f"Error: Processed data directory '{processed_data_dir}' not found.")
                return []

        if not region_names:
            print("Error: No regions found to process.")
            return []

        print(f"Found {len(region_names)} regions to process.")

        results = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # The filename passed to the function includes the extension
            futures = {executor.submit(process_region_zeroshot, name + ".pth", processed_data_dir, num_total_classes): name for name in region_names}
            for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Processing Regions"):
                result = future.result()
                if result is not None:
                    results.append(result)

        if not results:
            raise RuntimeError("No regions were processed successfully. Check file paths and data.")

        df = pd.DataFrame(results)
        df.to_csv(output_file, sep='\t', index=False)

    return results


# --- FILTERING, CLUSTERING, SCORING, AND SELECTION (No changes needed) ---

def filter_scenarios(results, min_nc, min_hc):
    return [r for r in results if r['Nc'] >= min_nc and r['Hc'] >= min_hc]

def create_clustering_features(filtered_results):
    return np.array([r['full_histogram'] for r in filtered_results])

def cluster_scenarios(features, k):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    return kmeans.fit_predict(features)

def find_optimal_k(features, max_k=20):
    wcss = []
    n_samples = len(features)
    max_k = min(max_k, n_samples - 1)
    if max_k <= 1: return [0] * n_samples
    for i in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=i, random_state=42, n_init='auto')
        kmeans.fit(features)
        wcss.append(kmeans.inertia_)
    return wcss

def score_scenarios(filtered_results, gamma):
    hc_values = np.array([r['Hc'] for r in filtered_results]).reshape(-1, 1)
    nc_values = np.array([r['Nc'] for r in filtered_results]).reshape(-1, 1)
    scaler = MinMaxScaler()
    hc_normalized = scaler.fit_transform(hc_values).flatten()
    nc_normalized = scaler.fit_transform(nc_values).flatten()
    for i, r in enumerate(filtered_results):
        r['score'] = hc_normalized[i] + gamma * nc_normalized[i]
    return filtered_results

def select_initial_subset(scored_results, initial_subset_size):
    df = pd.DataFrame(scored_results)
    if 'cluster' not in df.columns or df['cluster'].nunique() == 0:
        return df.sort_values('score', ascending=False).head(initial_subset_size).to_dict('records')
    selected_scenes, num_clusters = [], df['cluster'].nunique()
    scenarios_per_cluster, remainder = initial_subset_size // num_clusters, initial_subset_size % num_clusters
    for cluster_id in range(num_clusters):
        cluster_df = df[df['cluster'] == cluster_id].sort_values('score', ascending=False)
        num_to_select = scenarios_per_cluster + (1 if cluster_id < remainder else 0)
        selected_scenes.extend(cluster_df.head(num_to_select).to_dict('records'))
    return selected_scenes


# --- MAIN EXECUTION BLOCK ---

def main():
    parser = argparse.ArgumentParser(description="Select diverse regions for Zero-Shot training from processed Matterport3D data.")
    parser.add_argument("--processed_data_dir", type=str, default='data/scannet_3d/val', help="Path to the directory containing your processed .pth files.")
    parser.add_argument("--input_file", type=str, default='dataset/scannet_val.txt', help="Path to region names list. If not found, will scan the processed data directory.")
    parser.add_argument("--num_total_classes", type=int, default=21, help="Total number of valid semantic classes for Matterport3D (0-20).")
    parser.add_argument("--output_file", type=str, default="dataset/scannet_val_metrics.tsv", help="Path to save/load the computed region metrics.")
    parser.add_argument("--selected_output_file", type=str, default="dataset/scannet_val.tsv", help="Path to save the final selected regions and their info.")
    parser.add_argument("--num_workers", type=int, default=16, help="Number of parallel worker threads.")
    parser.add_argument("--gamma", type=float, default=1.0, help="Weight for Nc (class richness) in the scoring function.")
    parser.add_argument("--num_clusters", type=int, default=21, help="Number of clusters for k-means. If 0, use elbow method.")
    parser.add_argument("--initial_subset_size", type=int, default=20, help="The total number of regions to select.")
    args = parser.parse_args()

    # Phase 1: Load or Compute Metrics for all regions
    results = load_or_compute_metrics(args.input_file, args.processed_data_dir, args.num_total_classes, args.output_file, args.num_workers)
    if not results:
        print("Exiting: No results to process.")
        return

    # Phase 2: Filter regions
    ncs, hcs = [r['Nc'] for r in results], [r['Hc'] for r in results]
    median_nc, median_hc = np.median(ncs), np.median(hcs)
    print(f"\n--- Statistics ---\nMedian Nc: {median_nc:.2f}\nMedian Hc: {median_hc:.2f}")
    filtered_results = filter_scenarios(results, median_nc, median_hc)
    print(f"\nFiltered down to {len(filtered_results)} regions from {len(results)}.")
    if not filtered_results:
        print("No regions passed the filter.")
        return

    # Phase 3: Cluster regions
    features = create_clustering_features(filtered_results)
    optimal_k = args.num_clusters
    if optimal_k == 0:
        wcss_values = find_optimal_k(features)
        print("\nWCSS values for different k:", [f'{v:.2f}' for v in wcss_values])
        try:
            optimal_k = int(input("Enter the optimal k based on the elbow plot: "))
        except (ValueError, EOFError):
            optimal_k = 5
            print(f"Invalid input. Defaulting to k={optimal_k}.")
    optimal_k = max(1, min(optimal_k, len(filtered_results)))
    print(f"Using k={optimal_k} for clustering.")
    clusters = cluster_scenarios(features, optimal_k)
    for i, r in enumerate(filtered_results): r['cluster'] = clusters[i]

    # Phase 4: Score regions
    scored_results = score_scenarios(filtered_results, args.gamma)

    # Phase 5: Select final subset of regions
    selected_regions_info = select_initial_subset(scored_results, args.initial_subset_size)
    print(f"\n--- Results ---\nSelected {len(selected_regions_info)} regions for the final subset.")

    print("\nTop 10 selected regions (sample):")
    selected_regions_info_sorted = sorted(selected_regions_info, key=lambda x: x['score'], reverse=True)
    for r in selected_regions_info_sorted[:10]:
        print(f"Region: {r['scene']}, Nc: {r['Nc']}, Hc: {r['Hc']:.4f}, Score: {r['score']:.4f}, Cluster: {r['cluster']}")

    # Save results
    selected_df = pd.DataFrame(selected_regions_info)
    selected_df.to_csv(args.selected_output_file, sep='\t', index=False)
    print(f"\nDetailed info saved to {args.selected_output_file}")

    # output_txt_filename = "scannet_val.txt"
    # # The 'scene' column now contains region names
    # selected_df['scene'].to_csv(output_txt_filename, index=False, header=False)
    # print(f"Selected region names saved to {output_txt_filename}")

if __name__ == "__main__":
    main()