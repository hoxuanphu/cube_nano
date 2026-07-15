import os
import glob
import numpy as np
import tifffile as tiff
from pathlib import Path
from tqdm import tqdm
import argparse

def process_image(img_name, base_path, output_dir, patch_size=384, cloud_threshold=0.05, channels=3):
    """
    Reads channels (RGB or RGB+NIR) and GT for a given image name prefix,
    extracts 384x384 source patches. CloudDataset randomly crops these to
    256x256 during training.
    """
    # Define paths for each channel and ground truth
    # 38-Cloud format: e.g. red channel file is red_xxx.tif
    # We will search for the files matching the img_name (scene id)
    red_file = os.path.join(base_path, 'train_red', f'red_{img_name}.TIF')
    if not os.path.exists(red_file):
        # Fallback to lowercase extension or different prefix
        red_file = os.path.join(base_path, 'train_red', f'red_{img_name}.tif')
    
    green_file = os.path.join(base_path, 'train_green', f'green_{img_name}.TIF')
    if not os.path.exists(green_file):
        green_file = os.path.join(base_path, 'train_green', f'green_{img_name}.tif')
        
    blue_file = os.path.join(base_path, 'train_blue', f'blue_{img_name}.TIF')
    if not os.path.exists(blue_file):
        blue_file = os.path.join(base_path, 'train_blue', f'blue_{img_name}.tif')
        
    nir_file = os.path.join(base_path, 'train_nir', f'nir_{img_name}.TIF')
    if not os.path.exists(nir_file):
        nir_file = os.path.join(base_path, 'train_nir', f'nir_{img_name}.tif')
        
    gt_file = os.path.join(base_path, 'train_gt', f'gt_{img_name}.TIF')
    if not os.path.exists(gt_file):
        gt_file = os.path.join(base_path, 'train_gt', f'gt_{img_name}.tif')

    required_files = [red_file, green_file, blue_file, gt_file]
    if channels == 4:
        required_files.append(nir_file)

    if not all([os.path.exists(f) for f in required_files]):
        print(f"Missing files for {img_name}, skipping...")
        return

    # Read images
    try:
        r = tiff.imread(red_file)
        g = tiff.imread(green_file)
        b = tiff.imread(blue_file)
        gt = tiff.imread(gt_file)
        if channels == 4:
            nir = tiff.imread(nir_file)
    except Exception as e:
        print(f"Error reading {img_name}: {e}")
        return

    # Stack into a single tensor (H, W, C)
    if channels == 4:
        img = np.stack([r, g, b, nir], axis=-1)
    else:
        img = np.stack([r, g, b], axis=-1)
    
    h, w, c = img.shape
    
    # Extract patches
    patch_id = 0
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            # Keep only full-size patches for a fixed TensorRT input shape.
            if i + patch_size > h or j + patch_size > w:
                continue
                
            patch_img = img[i:i+patch_size, j:j+patch_size, :]
            patch_gt = gt[i:i+patch_size, j:j+patch_size]
            
            # Calculate cloud percentage
            # GT usually contains 0 for clear and 255 (or 1) for cloud
            cloud_pixels = np.sum(patch_gt > 0)
            total_pixels = patch_size * patch_size
            cloud_ratio = cloud_pixels / total_pixels
            
            # Determine label
            is_cloud = cloud_ratio >= cloud_threshold
            label_str = 'cloud' if is_cloud else 'clear'
            
            # Save patch
            out_file = os.path.join(output_dir, label_str, f'{img_name}_p{patch_id}.npy')
            np.save(out_file, patch_img)
            patch_id += 1


def main():
    parser = argparse.ArgumentParser(description="Preprocess 38-Cloud dataset")
    parser.add_argument('--data_dir', type=str, default='data/38-Cloud/38-Cloud_training', help='Path to 38-Cloud training folder')
    parser.add_argument('--out_dir', type=str, default='data/processed/train', help='Output directory for patches')
    parser.add_argument(
        '--patch_size', type=int, default=384,
        help='Source patch size saved for training crops (default: 384)',
    )
    parser.add_argument('--threshold', type=float, default=0.05, help='Cloud threshold (default: 0.05)')
    parser.add_argument('--channels', type=int, default=4, choices=[3, 4], help='Number of channels (3 for RGB, 4 for RGB+NIR)')
    parser.add_argument('--force', action='store_true', help='Delete existing processed .npy files before preprocessing')
    args = parser.parse_args()

    # Setup directories
    os.makedirs(os.path.join(args.out_dir, 'cloud'), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, 'clear'), exist_ok=True)
    existing = glob.glob(os.path.join(args.out_dir, 'cloud', '*.npy')) + glob.glob(os.path.join(args.out_dir, 'clear', '*.npy'))
    if existing and not args.force:
        raise FileExistsError(
            f"Found {len(existing)} existing patches in {args.out_dir}. "
            "Use --force to rebuild them with the new source size."
        )
    if args.force:
        for path in existing:
            os.remove(path)

    # Find all image names from the red folder
    red_folder = os.path.join(args.data_dir, 'train_red')
    if not os.path.exists(red_folder):
        print(f"Error: Could not find directory {red_folder}")
        print("Please download and extract the Kaggle 38-Cloud dataset into data/38-Cloud/")
        print("Command: kaggle datasets download -d sorour/38cloud-cloud-segmentation-in-satellite-images")
        return

    red_files = glob.glob(os.path.join(red_folder, '*.TIF')) + glob.glob(os.path.join(red_folder, '*.tif'))
    # Extract unique image prefixes (e.g., red_patch_10_11.TIF -> patch_10_11)
    img_names = [os.path.basename(f).replace('red_', '').replace('.TIF', '').replace('.tif', '') for f in red_files]

    print(f"Found {len(img_names)} images. Starting processing...")
    
    for img_name in tqdm(img_names):
        process_image(img_name, args.data_dir, args.out_dir, args.patch_size, args.threshold, args.channels)
        
    print("Preprocessing completed!")

if __name__ == '__main__':
    main()
