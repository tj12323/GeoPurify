import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

# ---- Visualization Utilities ----

def get_color_palette(num_classes=20):
    """Generate a color palette for semantic classes (e.g., ScanNet 20 classes)."""
    # Using matplotlib's tab20 colormap as an example
    cmap = plt.get_cmap('tab20')
    colors = [cmap(i % 20)[:3] for i in range(num_classes)]  # RGB tuples (0-1 range)
    return np.array(colors) * 255  # Scale to 0-255

def visualize_2d_semantic(img, pred_2d, gt_2d, mapping, x_label, y_label, palette, save_path, scene_id, idx):
    """Visualize 2D predictions, ground truth, and projected predictions."""
    # x_label, y_label = y_label, x_label  # This is the fix!
    img_np = img[0].cpu().numpy().astype(np.uint8)  # (3, 648, 484) - Corrected shape
    img_np = img_np.transpose(1, 2, 0)  # (648, 484, 3) - Transpose to HWC for OpenCV
    pred_2d_np = pred_2d.cpu().numpy()  # (15408,) - Use .cpu() for safety.
    gt_2d_np = gt_2d[0].cpu().numpy()  # (648, 484)
    h, w = gt_2d_np.shape # h=648, w=484 Use the ground truth shape, more robust.

    # Initialize semantic maps
    pred_map = np.zeros((h, w, 3), dtype=np.uint8)  # (648, 484, 3)
    gt_map = np.zeros((h, w, 3), dtype=np.uint8)    # (648, 484, 3)
    proj_pred_map = np.zeros((h, w, 3), dtype=np.uint8)  # (648, 484, 3)

    # Fill ground truth map
    valid_gt = (gt_2d_np >= 0) & (gt_2d_np < len(palette))  # (648, 484)
    gt_colors = palette[gt_2d_np[valid_gt].astype(int)].astype(np.uint8)
    gt_map[valid_gt] = gt_colors

    # Project predictions to 2D (for both pred_map and proj_pred_map)
    x_coords = y_label.cpu().numpy() - 1  # Move to CPU
    y_coords = x_label.cpu().numpy() - 1
    valid_mask = (x_coords >= 0) & (x_coords < w) & (y_coords >= 0) & (y_coords < h)
    x_coords = x_coords[valid_mask].astype(int) # Ensure integer coordinates after masking.
    y_coords = y_coords[valid_mask].astype(int)

    # Correctly handle the case where valid_mask might be empty
    if valid_mask.any():
        proj_colors = palette[pred_2d_np[valid_mask].astype(int)].astype(np.uint8)
        proj_pred_map[y_coords, x_coords] = proj_colors
        pred_map[y_coords, x_coords] = proj_colors   # Same projection for simplicity
        
        proj_pred_map[y_coords, x_coords] = proj_colors
        pred_map[y_coords, x_coords] = proj_colors   # Same projection for simplicity
    # No else needed:  proj_pred_map and pred_map are initialized as zeros.

    # Overlay with transparency
    alpha = 0.6
    pred_overlay = cv2.addWeighted(img_np, 1 - alpha, pred_map, alpha, 0)
    gt_overlay = cv2.addWeighted(img_np, 1 - alpha, gt_map, alpha, 0)
    proj_overlay = cv2.addWeighted(img_np, 1 - alpha, proj_pred_map, alpha, 0)

    # Concatenate images horizontally
    vis_img = np.hstack((img_np, pred_overlay, gt_overlay, proj_overlay))
    cv2.imwrite(str(save_path / f'{scene_id}_{idx:04d}_2d.png'), vis_img[:, :, ::-1])  # RGB to BGR


import open3d as o3d  # Import Open3D at the top of the file
import numpy as np  # Ensure numpy is imported
from pathlib import Path # Ensure pathlib is imported

def save_3d_point_cloud(coords, pred_labels, gt_labels, palette, vis_dir, scene_id, idx, use_gt=False):
    """
    Saves the 3D point cloud with predicted or ground truth labels as colors.

    Args:
        coords (torch.Tensor):  A tensor of shape (N, 3) representing the 3D coordinates.
        pred_labels (torch.Tensor): A tensor of shape (N,) containing the predicted class labels.
        gt_labels (torch.Tensor): A tensor of shape (N,) containing the ground truth labels.
        palette (list): A list of RGB colors (tuples) for each class.
        vis_dir (Path): The directory to save the point cloud to.
        scene_id (str): The ID of the scene.
        idx (int):  An index for the current batch (used for unique filenames).
        use_gt (bool): If True, use ground truth labels for coloring; otherwise, use predicted labels.
    """

    coords = coords.cpu().numpy()
    if use_gt:
        labels = gt_labels.cpu().numpy()
        filename = f"{scene_id}_{idx}_gt.ply"
    else:
        labels = pred_labels.cpu().numpy()
        filename = f"{scene_id}_{idx}_pred.ply"
    print(f"label shape {labels.shape}, range {labels.min()}, {labels.max()}")
    # Map labels to colors
    colors = np.array([[palette[label*3],palette[label*3+1],palette[label*3+2]] for label in labels]) / 255.0

    # --- CRUCIAL DATA TYPE CONVERSION AND CONTIGUITY ---
    # import pdb;pdb.set_trace()
    # coords = coords[:, 1:]
    coords = np.ascontiguousarray(coords, dtype=np.float64)  # Ensure float64 and contiguity
    colors = np.ascontiguousarray(colors, dtype=np.float64)  # Ensure float64 and contiguity
    # No need to convert labels if you're not directly using them with Open3D
    # print(f"coords shape {coords.shape}, colors shape {colors.shape}")

    # Create an Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Save the point cloud (PLY is a good, common format)
    filepath = vis_dir / filename
    o3d.io.write_point_cloud(str(filepath), pcd)
    print(f"Saved point cloud to: {filepath}")


import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import open3d as o3d
import time

# Define a mapping for your classes (Example for ScanNet20 + potential novel)
# Adjust this based on YOUR actual label IDs and base/novel split
# Format: label_id: (label_name, type) - type can be 'base', 'novel', 'ignore'
SCANNET_CATEGORY_INFO = {
    0: ("wall", 'base'), 1: ("floor", 'base'), 2: ("cabinet", 'base'),
    3: ("bed", 'base'), 4: ("chair", 'base'), 5: ("sofa", 'base'),
    6: ("table", 'base'), 7: ("door", 'base'), 8: ("window", 'base'),
    9: ("bookshelf", 'novel'), 10: ("picture", 'novel'), 11: ("counter", 'novel'),
    12: ("desk", 'novel'), 13: ("curtain", 'base'), 14: ("refridgerator", 'novel'),
    15: ("shower curtain", 'novel'), 16: ("toilet", 'novel'), 17: ("sink", 'novel'),
    18: ("bathtub", 'novel'),
    19: ("ignore", 'ignore') # Use your actual ignore label value
    # Or often 255 is used for ignore:
    # 255: ("ignore", 'ignore')
}


def visualize_pred_3d(
    pred_3d: torch.Tensor,
    point_coords: torch.Tensor,
    point_gt_labels: torch.Tensor,
    category_info: dict = SCANNET_CATEGORY_INFO,
    method: str = 'tsne',
    n_components: int = 3,
    use_open3d_vis: bool = True,
    subsample_tsne: int = 10000,
    output_dir: str = "debug_visualizations",
    scene_name: str = "scene_xyz",
    ignore_label_val: int = 255,
    save_ply: bool = True
):
    """
    Visualizes high-dimensional point features while preserving spatial relationships.
    Creates visualizations that can be meaningfully compared with ground truth.
    """
    print(f"--- Visualizing pred_3d for {scene_name} ---")
    assert pred_3d.shape[0] == point_coords.shape[0] == point_gt_labels.shape[0], \
        "Mismatched shapes between features, coordinates, and labels!"
    assert n_components in [2, 3], "n_components must be 2 or 3"

    import os
    os.makedirs(output_dir, exist_ok=True)

    # Move to CPU and convert to NumPy
    pred_3d_np = pred_3d.detach().cpu().numpy()
    coords_np = point_coords.detach().cpu().numpy()
    labels_np = point_gt_labels.detach().cpu().numpy().astype(int)

    # Filter out ignore labels
    valid_mask = (labels_np != ignore_label_val)
    if not np.any(valid_mask):
        print("Warning: All points have ignore labels. Skipping visualization.")
        return

    pred_3d_valid = pred_3d_np[valid_mask]
    coords_valid = coords_np[valid_mask]
    labels_valid = labels_np[valid_mask]
    print(f"Total points: {pred_3d_np.shape[0]}, Valid (non-ignore) points: {pred_3d_valid.shape[0]}")

    # --- 1. Save Original GT Point Cloud ---
    if use_open3d_vis:
        print("Saving original point cloud with GT colors...")
        pcd_gt = o3d.geometry.PointCloud()
        pcd_gt.points = o3d.utility.Vector3dVector(coords_np)

        # Generate colors based on GT labels
        max_label_id = max(category_info.keys()) if category_info else labels_np.max()
        colors_gt = plt.cm.get_cmap("turbo", max_label_id + 1)(labels_np / (max_label_id + 1))[:, :3]
        colors_gt[labels_np == ignore_label_val] = [0.5, 0.5, 0.5]  # Gray for ignore

        pcd_gt.colors = o3d.utility.Vector3dVector(colors_gt)
        gt_path = os.path.join(output_dir, f"{scene_name}_gt_colored.ply")
        o3d.io.write_point_cloud(gt_path, pcd_gt)
        print(f"Saved GT colored point cloud to {gt_path}")

    # --- 2. Dimensionality Reduction for Feature Analysis ---
    print(f"Performing {method.upper()} reduction to {n_components}D...")
    start_time = time.time()

    data_to_reduce = pred_3d_valid
    coords_for_embedding = coords_valid
    labels_for_embedding = labels_valid
    
    # Keep track of which points we're using
    subsample_indices = np.arange(len(labels_valid))

    if method == 'tsne':
        n_points = data_to_reduce.shape[0]
        if n_points > subsample_tsne:
            print(f"Subsampling {subsample_tsne}/{n_points} points for t-SNE...")
            subsample_indices = np.random.choice(n_points, subsample_tsne, replace=False)
            data_to_reduce = data_to_reduce[subsample_indices]
            coords_for_embedding = coords_valid[subsample_indices]
            labels_for_embedding = labels_valid[subsample_indices]

        tsne = TSNE(n_components=n_components, perplexity=30, n_iter=300, random_state=42, verbose=1)
        embedding = tsne.fit_transform(data_to_reduce)

    elif method == 'pca':
        pca = PCA(n_components=n_components)
        embedding = pca.fit_transform(data_to_reduce)

    print(f"Reduction took {time.time() - start_time:.2f} seconds.")

    # --- 3. Create Feature-based Coloring ---
    # Map embedding values to colors for visualization on original coordinates
    if n_components == 3:
        # Use RGB channels directly from 3D embedding
        # Normalize embedding to [0, 1] range for RGB
        embedding_normalized = embedding.copy()
        for i in range(3):
            emb_min, emb_max = embedding[:, i].min(), embedding[:, i].max()
            if emb_max > emb_min:
                embedding_normalized[:, i] = (embedding[:, i] - emb_min) / (emb_max - emb_min)
            else:
                embedding_normalized[:, i] = 0.5
        
        feature_colors = embedding_normalized  # Direct RGB mapping
        
    elif n_components == 2:
        # For 2D embedding, create colors using a colormap
        # Combine the two components into a single value for coloring
        combined_embedding = np.arctan2(embedding[:, 1], embedding[:, 0])  # Angle-based coloring
        combined_embedding = (combined_embedding + np.pi) / (2 * np.pi)  # Normalize to [0, 1]
        feature_colors = plt.cm.get_cmap("hsv")(combined_embedding)[:, :3]

    # --- 4. Save Point Clouds with Feature-based Colors ---
    if save_ply:
        print("Saving feature-colored point clouds...")
        
        # 4a. Original coordinates with feature-based colors (MAIN COMPARISON FILE)
        pcd_feature = o3d.geometry.PointCloud()
        pcd_feature.points = o3d.utility.Vector3dVector(coords_for_embedding)  # Original spatial coordinates
        pcd_feature.colors = o3d.utility.Vector3dVector(feature_colors)  # Colors from embedding
        
        feature_path = os.path.join(output_dir, f"{scene_name}_feature_colored_{method}_{n_components}d.ply")
        o3d.io.write_point_cloud(feature_path, pcd_feature)
        print(f"Saved feature-colored point cloud to {feature_path}")
        
        # 4b. Pure embedding space visualization (for understanding feature clusters)
        if n_components == 3:
            pcd_embedding_space = o3d.geometry.PointCloud()
            pcd_embedding_space.points = o3d.utility.Vector3dVector(embedding)  # Embedding coordinates
            
            # Color by GT labels in embedding space
            max_label_id = max(category_info.keys()) if category_info else labels_for_embedding.max()
            embedding_gt_colors = plt.cm.get_cmap("turbo", max_label_id + 1)(labels_for_embedding / (max_label_id + 1))[:, :3]
            pcd_embedding_space.colors = o3d.utility.Vector3dVector(embedding_gt_colors)
            
            embedding_space_path = os.path.join(output_dir, f"{scene_name}_embedding_space_{method}_3d.ply")
            o3d.io.write_point_cloud(embedding_space_path, pcd_embedding_space)
            print(f"Saved embedding space visualization to {embedding_space_path}")

        # 4c. Create side-by-side comparison dataset
        create_side_by_side_comparison(
            coords_np, labels_np, coords_for_embedding, feature_colors, 
            labels_for_embedding, output_dir, scene_name, method, n_components
        )

    # --- 5. Generate Plots ---
    print("Generating comparison plots...")
    create_comparison_plots(
        coords_for_embedding, labels_for_embedding, embedding, feature_colors,
        category_info, method, n_components, output_dir, scene_name
    )

    # --- 6. Print Summary ---
    print(f"\n--- Visualization Summary for {scene_name} ---")
    print(f"Method: {method.upper()}, Components: {n_components}D")
    print(f"Points processed: {len(labels_for_embedding)}/{len(labels_np)}")
    
    if save_ply:
        print("\n--- Files for comparison (use same spatial coordinates) ---")
        print(f"1. GT labels: {scene_name}_gt_colored.ply")
        print(f"2. Feature clusters: {scene_name}_feature_colored_{method}_{n_components}d.ply")
        print(f"3. Side-by-side: {scene_name}_comparison_{method}_{n_components}d.ply")
        
        print(f"\n--- Load both for comparison ---")
        print("import open3d as o3d")
        print(f"gt_pcd = o3d.io.read_point_cloud('{gt_path}')")
        print(f"feature_pcd = o3d.io.read_point_cloud('{feature_path}')")
        print("o3d.visualization.draw_geometries([gt_pcd, feature_pcd])")


def create_side_by_side_comparison(coords_all, labels_all, coords_subset, feature_colors, 
                                  labels_subset, output_dir, scene_name, method, n_components):
    """Create a side-by-side comparison with GT and feature coloring."""
    
    # Create GT colored version (left side)
    coords_gt_side = coords_all.copy()
    coords_gt_side[:, 0] -= (coords_all[:, 0].max() - coords_all[:, 0].min()) * 1.2  # Shift left
    
    max_label_id = max(labels_all) if len(labels_all) > 0 else 1
    colors_gt_side = plt.cm.get_cmap("turbo", max_label_id + 1)(labels_all / (max_label_id + 1))[:, :3]
    colors_gt_side[labels_all == 255] = [0.5, 0.5, 0.5]  # Gray for ignore
    
    # Create feature colored version (right side)  
    coords_feature_side = coords_subset.copy()
    coords_feature_side[:, 0] += (coords_subset[:, 0].max() - coords_subset[:, 0].min()) * 1.2  # Shift right
    
    # Combine both point clouds
    combined_coords = np.vstack([coords_gt_side, coords_feature_side])
    combined_colors = np.vstack([colors_gt_side, feature_colors])
    
    # Save combined point cloud
    pcd_comparison = o3d.geometry.PointCloud()
    pcd_comparison.points = o3d.utility.Vector3dVector(combined_coords)
    pcd_comparison.colors = o3d.utility.Vector3dVector(combined_colors)
    
    comparison_path = os.path.join(output_dir, f"{scene_name}_comparison_{method}_{n_components}d.ply")
    o3d.io.write_point_cloud(comparison_path, pcd_comparison)
    print(f"Saved side-by-side comparison to {comparison_path}")


def create_comparison_plots(coords, labels, embedding, feature_colors, category_info, 
                           method, n_components, output_dir, scene_name):
    """Create matplotlib plots showing the relationships."""
    
    fig = plt.figure(figsize=(20, 8))
    
    # Plot 1: Original spatial distribution colored by GT
    ax1 = fig.add_subplot(131, projection='3d')
    unique_labels = np.unique(labels)
    colors_map = plt.cm.get_cmap("turbo", max(category_info.keys()) + 1)
    
    for label_id in unique_labels:
        mask = (labels == label_id)
        label_name, label_type = category_info.get(label_id, (f"Unknown_{label_id}", "unknown"))
        color = colors_map(label_id / (max(category_info.keys()) + 1))
        
        ax1.scatter(coords[mask, 0], coords[mask, 1], coords[mask, 2],
                   c=[color], label=f"{label_name}", alpha=0.6, s=1)
    
    ax1.set_title("Original Spatial Distribution\n(Colored by GT Labels)")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y") 
    ax1.set_zlabel("Z")
    
    # Plot 2: Original spatial distribution colored by features
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(coords[:, 0], coords[:, 1], coords[:, 2], 
               c=feature_colors, alpha=0.6, s=1)
    ax2.set_title(f"Original Spatial Distribution\n(Colored by {method.upper()} Features)")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    
    # Plot 3: Embedding space (if 3D)
    if n_components == 3:
        ax3 = fig.add_subplot(133, projection='3d')
        for label_id in unique_labels:
            mask = (labels == label_id)
            label_name, label_type = category_info.get(label_id, (f"Unknown_{label_id}", "unknown"))
            color = colors_map(label_id / (max(category_info.keys()) + 1))
            
            ax3.scatter(embedding[mask, 0], embedding[mask, 1], embedding[mask, 2],
                       c=[color], label=f"{label_name}", alpha=0.6, s=1)
        
        ax3.set_title(f"{method.upper()} Feature Space\n(Colored by GT Labels)")
        ax3.set_xlabel(f"{method.upper()} 1")
        ax3.set_ylabel(f"{method.upper()} 2")
        ax3.set_zlabel(f"{method.upper()} 3")
    else:
        ax3 = fig.add_subplot(133)
        for label_id in unique_labels:
            mask = (labels == label_id)
            label_name, label_type = category_info.get(label_id, (f"Unknown_{label_id}", "unknown"))
            color = colors_map(label_id / (max(category_info.keys()) + 1))
            
            ax3.scatter(embedding[mask, 0], embedding[mask, 1],
                       c=[color], label=f"{label_name}", alpha=0.6, s=1)
        
        ax3.set_title(f"{method.upper()} Feature Space\n(Colored by GT Labels)")
        ax3.set_xlabel(f"{method.upper()} 1")
        ax3.set_ylabel(f"{method.upper()} 2")
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"{scene_name}_comparison_plot_{method}_{n_components}d.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot to {plot_path}")
    plt.close()

def get_pca_color(feat, brightness=1.25, center=True):
    u, s, v = torch.pca_lowrank(feat, center=center, niter=15)
    projection = feat @ v
    projection = projection[:, :3] * 0.6 + projection[:, 3:6] * 0.4
    min_val = projection.min(dim=-2, keepdim=True)[0]
    max_val = projection.max(dim=-2, keepdim=True)[0]
    div = torch.clamp(max_val - min_val, min=1e-6)
    color = (projection - min_val) / div * brightness
    color = color.clamp(0.0, 1.0)
    return color
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2 # For resizing feature maps
import open3d as o3d
import os
import time
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE # If you want t-SNE for projected features

# Assume SCANNET_CATEGORY_INFO and visualize_pred_3d (or similar for point clouds) exist

def visualize_2d_feature_map_overlay(
    features: dict,
    original_image: torch.Tensor,
    scale_to_vis: str = 's3', # Which feature scale to visualize (e.g., 's3', 's4')
    output_dir: str = "debug_visualizations/2d_features",
    scene_name: str = "scene_xyz",
    agg_method: str = 'mean', # 'mean' or 'max' channel aggregation
):
    """
    Visualizes a 2D feature map by overlaying its aggregated version onto the original image.
    Assumes batch size is 1.
    """
    print(f"--- Visualizing 2D Feature Map ({scale_to_vis}) for {scene_name} ---")
    if scale_to_vis not in features:
        print(f"Warning: Scale '{scale_to_vis}' not found in features dictionary. Skipping overlay visualization.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # --- Prepare Feature Map ---
    feature_map = features[scale_to_vis].squeeze(0).detach() # Remove batch dim -> [C, H', W']

    if agg_method == 'mean':
        feature_map_agg = torch.mean(feature_map, dim=0) # Aggregate channels -> [H', W']
    elif agg_method == 'max':
        feature_map_agg = torch.max(feature_map, dim=0)[0] # Aggregate channels -> [H', W']
    else:
         print(f"Warning: Unknown aggregation method '{agg_method}'. Using 'mean'.")
         feature_map_agg = torch.mean(feature_map, dim=0)

    feature_map_np = feature_map_agg.cpu().numpy()

    # Normalize feature map for visualization
    if feature_map_np.max() > feature_map_np.min():
         feature_map_norm = (feature_map_np - feature_map_np.min()) / (feature_map_np.max() - feature_map_np.min())
    else:
         feature_map_norm = np.zeros_like(feature_map_np) # Avoid division by zero if flat

    # --- Prepare Original Image ---
    img_tensor = original_image.squeeze(0).detach().cpu() # Remove batch dim -> [3, H, W]
    img_np = img_tensor.numpy().transpose(1, 2, 0) # -> [H, W, 3]

    # Denormalize image if necessary (assuming input was 0-255 originally)
    # Check if your dataloader normalizes; if using self.pixel_mean/std, reverse that.
    # Simple assumption: input was 0-1 or 0-255. Let's assume 0-255 for now.
    if img_np.max() <= 1.0 and img_np.min() >= 0.0:
         img_np = (img_np * 255).astype(np.uint8)
    else: # Maybe it's already denormalized or normalized differently
         # Attempt to scale to 0-255 safely
         img_min, img_max = img_np.min(), img_np.max()
         if img_max > img_min:
             img_np = ((img_np - img_min) / (img_max - img_min) * 255).astype(np.uint8)
         else:
             img_np = np.zeros_like(img_np, dtype=np.uint8)


    # --- Resize Feature Map to Image Size ---
    target_height, target_width = img_np.shape[:2]
    feature_map_resized = cv2.resize(feature_map_norm, (target_width, target_height),
                                     interpolation=cv2.INTER_LINEAR) # Linear or Nearest

    # Convert feature map to colormap (e.g., JET or VIRIDIS)
    heatmap = plt.get_cmap('jet')(feature_map_resized)[:, :, :3] # Get RGB, discard alpha
    heatmap = (heatmap * 255).astype(np.uint8)

    # --- Create Overlay ---
    alpha = 0.5 # Transparency of the heatmap
    overlay_img = cv2.addWeighted(img_np, 1 - alpha, heatmap, alpha, 0)

    # --- Plotting ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(img_np)
    axes[0].set_title(f"Original Image ({scene_name})")
    axes[0].axis('off')

    axes[1].imshow(feature_map_resized, cmap='jet')
    axes[1].set_title(f"Feature Map ({scale_to_vis}, {agg_method} agg.)")
    axes[1].axis('off')

    axes[2].imshow(overlay_img)
    axes[2].set_title("Overlay")
    axes[2].axis('off')

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"{scene_name}_feature_overlay_{scale_to_vis}.png")
    plt.savefig(plot_path)
    print(f"Saved feature map overlay to {plot_path}")
    plt.close(fig)

def visualize_projected_2d_features(
    features: dict,
    batch_input: dict,
    scene_config: dict, # Contains fusion.img_dim
    category_info: dict = SCANNET_CATEGORY_INFO,
    scale_to_vis: str = 's3', # Feature scale matching Gaussian update logic
    output_dir: str = "debug_visualizations/2d_features_projected",
    scene_name: str = "scene_xyz",
    color_by: str = 'feature_mean', # 'feature_mean', 'gt_label', 'feature_pca'
    ignore_label_val: int = 255,
):
    """
    Corrected visualization of 2D features projected onto the 3D point cloud.
    Assumes batch_input['unique_map'] is [N_total, 4] -> [?, u, v, visibility]
    and batch_input['ori_coords'], ['labels_3d'] are [N_total, ...]
    Assumes batch size is 1.
    """
    print(f"--- Visualizing Corrected Projected 2D Features ({scale_to_vis}) for {scene_name} ---")
    required_keys = ["unique_map", "ori_coords", "labels_3d"]
    if not all(k in batch_input for k in required_keys):
        print(f"Warning: Missing required keys {required_keys} in batch_input. Skipping projection visualization.")
        return
    if scale_to_vis not in features:
        print(f"Warning: Scale '{scale_to_vis}' not found in features dictionary. Skipping projection visualization.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # --- Get Data ---
    features_2d = features[scale_to_vis] # Shape [1, C, H', W']
    mapping = batch_input["unique_map"]      # Shape [N_total, 4] (?, u, v, visibility)
    point_coords_original = batch_input["ori_coords"][:, 1:] # Shape [N_total, 3]
    point_labels_original = batch_input["labels_3d"]      # Shape [N_total]

    # --- Filter by Visibility ---
    # Visibility is in column 3
    mapping_tensor = torch.as_tensor(mapping, device=features_2d.device) # Ensure tensor on correct device
    visibility_mask = mapping_tensor[:, 3] == 1
    num_visible = torch.sum(visibility_mask).item()

    if num_visible == 0:
        print("Warning: No points marked as visible in 'unique_map' (column 3). Skipping projection.")
        return
    print(f"Found {num_visible} / {mapping.shape[0]} visible points.")

    # Get the original indices of ONLY the visible points
    original_indices_visible = torch.where(visibility_mask)[0] # Indices from 0 to N_total-1

    # Get the mapping entries ONLY for visible points
    mapping_visible = mapping_tensor[visibility_mask] # Shape [N_visible, 4]

    # --- Extract Features for Visible Points ---
    features_2d_b = features_2d[0] # Batch size 1 -> [C, H', W']
    mapping_visible_clone = mapping_visible.clone() # Use the filtered map

    # Use scene_config to get original image dimensions used for projection mapping
    img_h, img_w = scene_config.fusion.img_dim[0], scene_config.fusion.img_dim[1]
    _, C, h_prime, w_prime = features_2d.shape

    # Extract u, v coordinates (columns 1 and 2) for visible points
    u_coords = mapping_visible_clone[:, 1]
    v_coords = mapping_visible_clone[:, 2]

    # Scale mapping coords from original image space to feature map space
    mapped_x = (u_coords / (img_w / w_prime)).long() # u -> x -> width
    mapped_y = (v_coords / (img_h / h_prime)).long() # v -> y -> height

    # Clamp coordinates to be within feature map bounds
    mapped_x = torch.clamp(mapped_x, 0, w_prime - 1)
    mapped_y = torch.clamp(mapped_y, 0, h_prime - 1)

    # Lookup features using the mapped coordinates
    try:
        # Indexing: features_2d_b[channel, height_idx, width_idx]
        features_mapping_b = features_2d_b[:, mapped_y, mapped_x] # Shape [C, N_visible]
        features_mapping_b = features_mapping_b.permute(1, 0).contiguous() # Shape [N_visible, C]
    except IndexError as e:
         print(f"Error during feature lookup: {e}")
         print(f"Max mapped x: {mapped_x.max()}, Max mapped y: {mapped_y.max()}")
         print(f"Feature map dims: C={C}, H'={h_prime}, W'={w_prime}")
         return

    features_mapping_np = features_mapping_b.detach().cpu().numpy()
    print(f"Extracted features shape: {features_mapping_np.shape}")
    print(f"Feature range: min={features_mapping_np.min()}, max={features_mapping_np.max()}, mean={features_mapping_np.mean()}")

    visible_coords = point_coords_original.detach().cpu().numpy() # Shape [N_visible, 3]
    visible_labels = point_labels_original.detach().cpu().numpy().astype(int) # Shape [N_visible]

    # --- Create Point Cloud for Visualization ---
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(visible_coords)
    print(f"Point cloud created with {len(pcd.points)} points.")

    # --- Color the Point Cloud ---
    point_colors = np.zeros_like(visible_coords) + 0.5 # Default to gray

    valid_colors_generated = False
    if color_by == 'feature_mean':
        print("Coloring by mean feature activation...")
        feature_means = features_mapping_np.mean(axis=1)
        feat_min, feat_max = feature_means.min(), feature_means.max()
        print(f"  Feature means range: min={feat_min}, max={feat_max}")
        if feat_max > feat_min:
            feature_means_norm = (feature_means - feat_min) / (feat_max - feat_min)
            point_colors = plt.get_cmap('viridis')(feature_means_norm)[:, :3]
            valid_colors_generated = True
        else:
            print("  Warning: Feature means are uniform. Coloring gray.")
            point_colors[:] = [0.5, 0.5, 0.5] # Gray if flat

    elif color_by == 'gt_label':
        print("Coloring by ground truth label...")
        unique_lbls = np.unique(visible_labels)
        print(f"  Visible labels: {unique_lbls[:20]}... ({len(unique_lbls)} unique)") # Print some unique labels
        max_label_id = max(category_info.keys()) if category_info else visible_labels.max()
        # Ensure max_label_id >= 0 for colormap range
        norm_factor = max(max_label_id, 0) + 1
        try:
            # Use turbo map, ensure labels are normalized correctly
            point_colors = plt.get_cmap("turbo", norm_factor)((visible_labels % norm_factor) / norm_factor)[:, :3]
            # Optional: Color ignore points gray explicitly
            point_colors[visible_labels == ignore_label_val] = [0.5, 0.5, 0.5]
            valid_colors_generated = True
        except Exception as e:
            print(f"  Error applying colormap for labels: {e}")
            point_colors[:] = [0.5, 0.5, 0.5] # Fallback to gray


    elif color_by == 'feature_pca':
         print("Coloring by PCA of features (first 3 components mapped to RGB)...")
         if features_mapping_np.shape[1] < 3:
             print("  Warning: Cannot perform 3-component PCA with fewer than 3 features. Coloring gray.")
             point_colors[:] = [0.5, 0.5, 0.5]
         else:
             try:
                 pca = PCA(n_components=3)
                 features_pca = pca.fit_transform(features_mapping_np)
                 # Normalize PCA components to 0-1 range for RGB mapping
                 feat_min = features_pca.min(axis=0)
                 feat_max = features_pca.max(axis=0)
                 f_range = feat_max - feat_min
                 # Prevent division by zero for components with no variance
                 zero_range_mask = (f_range == 0)
                 f_range[zero_range_mask] = 1.0
                 point_colors = (features_pca - feat_min) / f_range
                 # Set components with no variance to mid-gray (0.5)
                 point_colors[:, zero_range_mask] = 0.5

                 point_colors = np.clip(point_colors, 0, 1) # Ensure values are in [0, 1]
                 print(f"  PCA range (min): {feat_min}, (max): {feat_max}")
                 valid_colors_generated = True
             except Exception as e:
                 print(f"  Error during PCA calculation: {e}")
                 point_colors[:] = [0.5, 0.5, 0.5] # Fallback to gray

    else:
        print(f"Warning: Unknown color_by option '{color_by}'. Using default (gray).")
        point_colors[:] = [0.5, 0.5, 0.5]

    # --- Assign Colors and Save ---
    if valid_colors_generated:
        print(f"Assigning colors. Shape: {point_colors.shape}, Range: min={point_colors.min()}, max={point_colors.max()}")
        pcd.colors = o3d.utility.Vector3dVector(point_colors)
    else:
        print("Warning: Valid colors were not generated. Point cloud might appear gray or black.")
        # Assign default gray colors if not already set
        if not hasattr(pcd, 'colors') or len(pcd.colors) == 0:
             pcd.colors = o3d.utility.Vector3dVector(np.zeros_like(visible_coords) + 0.5)


    # --- Save Point Cloud ---
    vis_path = os.path.join(output_dir, f"{scene_name}_projected_{scale_to_vis}_color_{color_by}.ply")
    try:
        o3d.io.write_point_cloud(vis_path, pcd, write_ascii=True) # Use ASCII for easier debugging if needed
        print(f"Saved projected 2D feature visualization to {vis_path}")
        # Check if the file exists and has size
        if os.path.exists(vis_path) and os.path.getsize(vis_path) > 0:
             print("File saved successfully.")
             # Try reading back to confirm colors (optional advanced check)
             # pcd_read = o3d.io.read_point_cloud(vis_path)
             # if pcd_read.has_colors():
             #    print("Verified: Saved file has colors.")
             # else:
             #    print("Warning: Saved file does NOT have colors according to Open3D reader.")
        else:
             print("Warning: File saving seemed to fail (missing or zero size).")

    except Exception as e:
        print(f"Error saving point cloud to {vis_path}: {e}")

    # Optional: Display interactively
    # try:
    #    print("Attempting to display point cloud...")
    #    o3d.visualization.draw_geometries([pcd], window_name=f"{scene_name} - Proj Feat ({scale_to_vis}, {color_by})")
    # except Exception as e:
    #    print(f"Error displaying point cloud: {e}")

def visualize_query_embeddings(
    mask_embed: torch.Tensor,    # Shape [N_queries, D_embed]
    pred_masks: torch.Tensor,    # Shape [N_queries, H_mask, W_mask]
    pred_logits: torch.Tensor,   # Shape [N_queries, N_classes_total + 1] (N_classes_total usually num_classes_eval)
    original_image: torch.Tensor,# Shape [3, H_orig, W_orig] (single image)
    num_classes_eval: int, # Total number of actual semantic classes (e.g., len(cfg.all_label))
    category_info: dict = SCANNET_CATEGORY_INFO,
    output_dir: str = "debug_visualizations/query_embeddings",
    scene_name: str = "scene_xyz",
    method: str = 'pca', # 'pca' or 'tsne'
    n_components: int = 2,
    confidence_threshold: float = 0.3, # Min top-1 score to include query in plot/mask viz
    subsample_tsne: int = 2000, # Max queries for t-SNE
    top_n_masks: int = 10, # Number of top masks (based on top-1 score) to visualize overlay
    ignore_label_val: int = 255, # Value representing ignore/background
):
    """
    Visualizes query embeddings using dimensionality reduction and overlays
    corresponding masks on the image. Assumes batch size is 1.
    Now includes top-2 prediction info in mask_viz_list.

    Args:
        mask_embed: Query embeddings tensor.
        pred_masks: Predicted masks for each query.
        pred_logits: Classification logits for each query.
        original_image: Original input image tensor (before model normalization).
        category_info: Dictionary mapping label IDs to (name, type).
        num_classes_eval: Total number of semantic classes model predicts for (excluding null).
        output_dir: Directory to save visualizations.
        scene_name: Name for saving files.
        method: 'pca' or 'tsne'.
        n_components: 2 or 3.
        confidence_threshold: Minimum top-1 softmax score to consider a query for plotting.
        subsample_tsne: Max points for t-SNE.
        top_n_masks: How many top-scoring (top-1) masks to overlay.
        ignore_label_val: The integer value used for ignore/background class.
    """
    print(f"--- Visualizing Query Embeddings for {scene_name} ---")
    assert mask_embed.shape[0] == pred_masks.shape[0] == pred_logits.shape[0], \
        "Mismatched number of queries between embeddings, masks, and logits!"
    assert n_components in [2, 3], "n_components must be 2 or 3"
    num_queries = mask_embed.shape[0]

    os.makedirs(output_dir, exist_ok=True)

    # --- Prepare Data (CPU, NumPy) ---
    mask_embed_np = mask_embed.detach().cpu().numpy()
    pred_masks_np = pred_masks.detach().cpu().numpy() # Keep masks for later viz
    pred_logits_detached = pred_logits.detach() # Keep on device for softmax

    # --- Get Top-2 Predicted Classes and Scores ---
    scores_softmax = F.softmax(pred_logits_detached, dim=-1) # [N_queries, N_classes_total + 1]
    
    # Get top-2 predictions
    # k=2 might be an issue if N_classes_total + 1 < 2, ensure there are enough classes
    k_top = min(2, scores_softmax.shape[-1])
    if k_top < 2:
        print(f"Warning: Not enough classes ({scores_softmax.shape[-1]}) to get top-2. Will get top-{k_top}.")

    topk_scores, topk_labels_raw = torch.topk(scores_softmax, k=k_top, dim=-1) 

    # Top-1 predictions (for filtering and primary plot color)
    pred_scores_top1 = topk_scores[:, 0]
    pred_labels_top1_raw = topk_labels_raw[:, 0]

    # Assume the last class in logits is the 'null' or 'background' class
    # This index should correspond to num_classes_eval if logits are [0...num_classes_eval-1, null_class]
    null_class_index = pred_logits_detached.shape[-1] - 1
    
    # Adjust top-1 labels: map the "null" class index to ignore_label_val
    pred_labels_top1_adjusted = pred_labels_top1_raw.clone()
    pred_labels_top1_adjusted[pred_labels_top1_raw == null_class_index] = ignore_label_val
    
    pred_scores_top1_np = pred_scores_top1.cpu().numpy()
    pred_labels_top1_adjusted_np = pred_labels_top1_adjusted.cpu().numpy().astype(int)

    # Prepare top-2 (if k_top was indeed 2) for mask_viz_list
    if k_top == 2:
        pred_scores_top2 = topk_scores[:, 1]
        pred_labels_top2_raw = topk_labels_raw[:, 1]
        pred_labels_top2_adjusted = pred_labels_top2_raw.clone()
        pred_labels_top2_adjusted[pred_labels_top2_raw == null_class_index] = ignore_label_val
        
        pred_scores_top2_np_all_queries = pred_scores_top2.cpu().numpy()
        pred_labels_top2_adjusted_np_all_queries = pred_labels_top2_adjusted.cpu().numpy().astype(int)
    else: # Handle cases where only top-1 is available
        pred_scores_top2_np_all_queries = np.full_like(pred_scores_top1_np, float('nan'))
        pred_labels_top2_adjusted_np_all_queries = np.full_like(pred_labels_top1_adjusted_np, ignore_label_val)


    # --- Filter Queries by Top-1 Confidence ---
    keep_query_mask = (pred_scores_top1_np >= confidence_threshold)
    num_kept_queries = np.sum(keep_query_mask)

    if num_kept_queries == 0:
        print(f"Warning: No queries passed confidence threshold {confidence_threshold}. Skipping visualization.")
        return
    print(f"Keeping {num_kept_queries} / {num_queries} queries with top-1 score >= {confidence_threshold:.2f}")

    mask_embed_filtered = mask_embed_np[keep_query_mask]
    # For plotting, use the top-1 adjusted labels of the filtered queries
    pred_labels_for_plot = pred_labels_top1_adjusted_np[keep_query_mask] 
    # pred_scores_top1_filtered = pred_scores_top1_np[keep_query_mask] # Not directly used in plot, but for sorting masks
    
    # Get original indices of queries that are kept *and* will be among top_n_masks
    # First, get scores of all queries that passed the confidence threshold
    confident_query_indices = np.where(keep_query_mask)[0]
    confident_query_scores = pred_scores_top1_np[keep_query_mask]
    
    # Then, sort these confident queries by score to pick the top_n_masks
    num_masks_to_show = min(top_n_masks, len(confident_query_scores))
    sorted_confident_indices_for_masks = np.argsort(confident_query_scores)[::-1][:num_masks_to_show]
    
    # Finally, map these back to the *original* query indices (0 to N_queries-1)
    top_original_query_indices_for_masks = confident_query_indices[sorted_confident_indices_for_masks]


    # --- Dimensionality Reduction (using embeddings of confident queries) ---
    print(f"Performing {method.upper()} reduction to {n_components}D...")
    start_time = time.time()

    data_to_reduce = mask_embed_filtered # Embeddings of queries that passed confidence_threshold
    labels_for_dim_reduction_plot = pred_labels_for_plot # Top-1 labels for these queries

    if method == 'tsne':
        n_points_for_tsne = data_to_reduce.shape[0]
        if n_points_for_tsne > subsample_tsne:
            print(f"Subsampling {subsample_tsne}/{n_points_for_tsne} points for t-SNE...")
            indices_tsne = np.random.choice(n_points_for_tsne, subsample_tsne, replace=False)
            data_to_reduce = data_to_reduce[indices_tsne]
            labels_for_dim_reduction_plot = labels_for_dim_reduction_plot[indices_tsne]
        tsne = TSNE(n_components=n_components, perplexity=min(30, data_to_reduce.shape[0]-1), n_iter=300, random_state=42, verbose=0)
        embedding = tsne.fit_transform(data_to_reduce)
    elif method == 'pca':
        pca = PCA(n_components=n_components)
        embedding = pca.fit_transform(data_to_reduce)
        # labels_for_dim_reduction_plot is already set
    else:
        raise ValueError("Method must be 'pca' or 'tsne'")
    print(f"Reduction took {time.time() - start_time:.2f} seconds.")

    # --- Plotting the Embedding ---
    print("Generating embedding plot...")
    plt.figure(figsize=(12, 10))
    ax = plt.gca()
    unique_labels_in_plot = np.unique(labels_for_dim_reduction_plot)

    max_cat_id_from_info = 0
    if category_info:
        valid_keys = [k for k in category_info.keys() if isinstance(k, int) and k != ignore_label_val]
        if valid_keys:
            max_cat_id_from_info = max(valid_keys)
    
    max_possible_label = max(np.max(pred_labels_top1_adjusted_np), num_classes_eval -1, max_cat_id_from_info )
    colors = plt.cm.get_cmap("turbo", max_possible_label + 2) # +2 for safety with ignore
    handles = {} 

    for label_id in unique_labels_in_plot:
        mask_plot = (labels_for_dim_reduction_plot == label_id)
        label_name, label_type = category_info.get(label_id, (f"ID_{label_id}", "unknown"))

        color = colors(label_id / (max_possible_label + 1)) if label_id != ignore_label_val else [0.5, 0.5, 0.5] 
        marker = 'o' if label_type == 'base' else '^' if label_type == 'novel' else 'x' 
        size = 15 if label_type == 'base' else 25 if label_type == 'novel' else 10
        plot_label_text = f"{label_name} ({label_type})" if label_id != ignore_label_val else f"Ignore/BG ({label_id})"


        if n_components == 2:
            scatter_handle = ax.scatter(embedding[mask_plot, 0], embedding[mask_plot, 1],
                                       color=color, label=plot_label_text,
                                       alpha=0.8, marker=marker, s=size)
        elif n_components == 3: # Simplified: plotting first 2 components for 3D request
            print("Warning: 3D plotting requested but showing 2D projection (Component 1 vs 2).")
            scatter_handle = ax.scatter(embedding[mask_plot, 0], embedding[mask_plot, 1],
                                       color=color, label=plot_label_text,
                                       alpha=0.8, marker=marker, s=size)
        
        if label_type not in handles or (label_id == ignore_label_val and "ignore" not in handles) :
             type_key = "ignore" if label_id == ignore_label_val else label_type
             handles[type_key] = plt.Line2D([0], [0], marker=marker, color='w', 
                                           label=type_key.capitalize(),
                                           markerfacecolor=color, markersize=np.sqrt(size)*2)


    legend_elements_sorted = [handles[lt] for lt in sorted(handles.keys())]
    ax.legend(handles=legend_elements_sorted, title="Predicted Type", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.title(f'Query Embed ({method.upper()}, {n_components}D, Top-1 Score>{confidence_threshold:.2f}) for {scene_name}')
    plt.xlabel(f"{method.upper()} Component 1")
    plt.ylabel(f"{method.upper()} Component 2")
    plt.grid(True)
    plt.tight_layout(rect=[0, 0, 0.85, 1]) 
    plot_path = os.path.join(output_dir, f"{scene_name}_query_embedding_{method}_{n_components}D.png")
    plt.savefig(plot_path)
    print(f"Saved embedding plot to {plot_path}")
    plt.close()


    # --- Visualize Top N Masks Overlay ---
    print(f"Generating overlay image for top {num_masks_to_show} masks (by top-1 score)...")

    img_tensor_cpu = original_image.cpu() 
    img_vis_np = img_tensor_cpu.numpy().transpose(1, 2, 0)
    if img_vis_np.max() <= 1.0 and img_vis_np.min() >= 0.0:
        img_vis_np = (img_vis_np * 255).astype(np.uint8)
    else: 
        img_min, img_max = img_vis_np.min(), img_vis_np.max()
        if img_max > img_min:
            img_vis_np = ((img_vis_np - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        else:
            img_vis_np = np.zeros_like(img_vis_np, dtype=np.uint8)
    
    # Ensure 3 channels for cvtColor
    if img_vis_np.shape[2] == 1: # Grayscale
        img_vis_np = cv2.cvtColor(img_vis_np, cv2.COLOR_GRAY2BGR)
    elif img_vis_np.shape[2] == 4: # RGBA
        img_vis_np = cv2.cvtColor(img_vis_np, cv2.COLOR_RGBA2BGR)
    else: # Assuming RGB
        img_vis_np = cv2.cvtColor(img_vis_np, cv2.COLOR_RGB2BGR)

    H_orig, W_orig = img_vis_np.shape[:2]
    overlay_img = img_vis_np.copy()
    mask_viz_list = []

    if num_masks_to_show > 0:
        for i, query_idx in enumerate(top_original_query_indices_for_masks):
            mask_np = pred_masks_np[query_idx] 

            # Top-1 info for this query_idx
            score_top1_current = pred_scores_top1_np[query_idx]
            label_top1_adjusted_current = pred_labels_top1_adjusted_np[query_idx]
            label_name_top1, label_type_top1 = category_info.get(label_top1_adjusted_current, 
                                                                  (f"ID_{label_top1_adjusted_current}", "unknown"))
            
            # Top-2 info for this query_idx
            score_top2_current = pred_scores_top2_np_all_queries[query_idx]
            label_top2_adjusted_current = pred_labels_top2_adjusted_np_all_queries[query_idx]
            label_name_top2, label_type_top2 = ("N/A", "N/A") # Default if no top-2
            if not np.isnan(score_top2_current) and label_top2_adjusted_current != ignore_label_val : # Check if top-2 is valid
                 label_name_top2, label_type_top2 = category_info.get(label_top2_adjusted_current, 
                                                                    (f"ID_{label_top2_adjusted_current}", "unknown"))
            elif np.isnan(score_top2_current): # if k_top was 1
                 label_name_top2, label_type_top2 = ("OnlyTop1", "OnlyTop1")


            # Use top-1 label for coloring the mask contour
            color_tuple_viz = colors(label_top1_adjusted_current / (max_possible_label + 1))[:3] \
                if label_top1_adjusted_current != ignore_label_val else (0.5, 0.5, 0.5)
            color_bgr_viz = tuple(int(c * 255) for c in reversed(color_tuple_viz))

            mask_resized_viz = cv2.resize(mask_np, (W_orig, H_orig), interpolation=cv2.INTER_LINEAR)
            mask_binary_viz = (mask_resized_viz > 0.5).astype(np.uint8)
            contours_viz, _ = cv2.findContours(mask_binary_viz, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay_img, contours_viz, -1, color_bgr_viz, thickness=2)

            if contours_viz:
                M = cv2.moments(contours_viz[0])
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    text_on_img = f"{i}:{label_name_top1}({score_top1_current:.2f})"
                    cv2.putText(overlay_img, text_on_img, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 2)
                    cv2.putText(overlay_img, text_on_img, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_bgr_viz, 1)
            
            mask_viz_list.append({
                "query_original_idx": query_idx,
                "rank_in_visualization": i,
                "top1_label_name": label_name_top1,
                "top1_label_type": label_type_top1,
                "top1_label_id": int(label_top1_adjusted_current),
                "top1_score": float(f"{score_top1_current:.4f}"),
                "top2_label_name": label_name_top2,
                "top2_label_type": label_type_top2,
                "top2_label_id": int(label_top2_adjusted_current) if not np.isnan(score_top2_current) and label_top2_adjusted_current != ignore_label_val else -1, # use -1 or similar for invalid/NA
                "top2_score": float(f"{score_top2_current:.4f}") if not np.isnan(score_top2_current) else -1.0
            })

        overlay_path = os.path.join(output_dir, f"{scene_name}_top_{num_masks_to_show}_masks_overlay.png")
        cv2.imwrite(overlay_path, overlay_img)
        print(f"Saved mask overlay image to {overlay_path}")
        print(f"Top mask info (with top-2 predictions):")
        for item in mask_viz_list:
            print(f"  Query Original Idx: {item['query_original_idx']}, Vis Rank: {item['rank_in_visualization']}")
            print(f"    Top-1: {item['top1_label_name']} ({item['top1_label_type']}, ID:{item['top1_label_id']}) - Score: {item['top1_score']}")
            if item['top2_score'] != -1.0:
                 print(f"    Top-2: {item['top2_label_name']} ({item['top2_label_type']}, ID:{item['top2_label_id']}) - Score: {item['top2_score']}")
            else:
                 print(f"    Top-2: N/A")

    else:
        print("Skipping mask overlay generation as no queries met criteria for visualization.")

    print(f"--- Finished Visualizing Query Embeddings for {scene_name} ---\n")