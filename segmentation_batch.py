import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage import color, morphology, measure, segmentation
import os
import glob

# ==========================================
# 1. CONFIGURATION
# ==========================================
INPUT_FOLDER = 'input_images'   # Put your images here
OUTPUT_FOLDER = 'output_results' # Results will be saved here
EXTENSIONS = ['*.jpg', '*.jpeg', '*.png', '*.bmp']

# ==========================================
# 2. PREPROCESSING & UTILITIES
# ==========================================

def load_and_preprocess(image_path):
    """
    Loads image and converts to RGB.
    """
    try:
        img_bgr = cv2.imread(image_path)
        if img_bgr is None: raise FileNotFoundError
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return img_rgb
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return None

def post_process_mask(binary_mask):
    """
    Refines the mask using Morphological Operations.
    1. Opening: Removes small noise.
    2. Closing: Fills internal holes/gaps.
    3. Largest Component: Keeps only the main object.
    """
    # 1. Morphological Opening (Remove small speckles)
    selem = morphology.disk(3)
    mask_open = morphology.binary_opening(binary_mask, selem)
    
    # 2. Morphological Closing (Fill holes inside the object)
    mask_closed = morphology.binary_closing(mask_open, selem)
    
    # 3. Keep only the largest connected component
    labeled_blobs = measure.label(mask_closed)
    regions = measure.regionprops(labeled_blobs)
    
    if not regions:
        return mask_closed 
        
    largest_region = max(regions, key=lambda x: x.area)
    final_mask = np.zeros_like(binary_mask)
    for coord in largest_region.coords:
        final_mask[coord[0], coord[1]] = 1
        
    return final_mask

# ==========================================
# 3. METHODOLOGIES (For your Report)
# ==========================================

# --- Method A: K-Means Clustering ---
def apply_kmeans_lab(image, k=3):
    """
    Segments based on Color Similarity (Clustering).
    """
    img_lab = color.rgb2lab(image)
    h, w, c = img_lab.shape
    ab_channels = img_lab[:, :, 1:].reshape((h * w, 2))
    
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(ab_channels)
    return labels.reshape((h, w))

def select_cluster_by_roi(label_image, roi):
    """
    Selects the cluster that appears most frequently in the center ROI.
    """
    x, y, w, h = roi
    # Ensure ROI is within bounds
    if x < 0: x = 0
    if y < 0: y = 0
    
    roi_labels = label_image[y:y+h, x:x+w]
    if roi_labels.size == 0: return 0 # Fallback
    
    vals, counts = np.unique(roi_labels, return_counts=True)
    return vals[np.argmax(counts)]

# --- Method B: Region Growing ---
def apply_region_growing(image, seed_point, tolerance=25):
    """
    Segments based on Spatial Connectivity and Similarity.
    """
    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Ensure seed point is within bounds
    y, x = seed_point
    if y >= img_gray.shape[0]: y = img_gray.shape[0] - 1
    if x >= img_gray.shape[1]: x = img_gray.shape[1] - 1
    
    mask = segmentation.flood(img_gray, (y, x), tolerance=tolerance)
    return mask.astype(int)

# --- Method C: Iterative Global Thresholding ---
def apply_iterative_threshold(image):
    """
    Segments based on Intensity Thresholding.
    Iterative algorithm to find optimal T.
    """
    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    T = np.mean(img_gray)
    T_prev = 0
    delta = 0.5 
    
    # Iterate until convergence
    while abs(T - T_prev) > delta:
        T_prev = T
        region_1 = img_gray[img_gray > T]
        region_2 = img_gray[img_gray <= T]
        
        if len(region_1) == 0 or len(region_2) == 0: break
        
        m1 = np.mean(region_1)
        m2 = np.mean(region_2)
        T = (m1 + m2) / 2
        
    # Assume object is the 'different' part (usually brighter or distinct)
    mask = (img_gray < T).astype(int) 
    
    # Heuristic: If mask covers > 75% of image, assume we inverted it and flip back
    if np.sum(mask) > (mask.size * 0.75):
        mask = 1 - mask
        
    return mask

# --- Extra: Canny Edges ---
def apply_canny(image):
    """
    Detects Discontinuities (Edges).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return edges

# ==========================================
# 4. BATCH PROCESSING EXECUTION
# ==========================================

if __name__ == "__main__":
    # 1. Setup Folders
    if not os.path.exists(INPUT_FOLDER):
        os.makedirs(INPUT_FOLDER)
        print(f"Created folder '{INPUT_FOLDER}'. Please put your images there and run again.")
        exit()
        
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    # 2. Get all image files
    image_files = []
    for ext in EXTENSIONS:
        image_files.extend(glob.glob(os.path.join(INPUT_FOLDER, ext)))
    
    if not image_files:
        print(f"No images found in '{INPUT_FOLDER}'. Add .jpg or .png files.")
        exit()
        
    print(f"Found {len(image_files)} images. Starting batch processing...")

    # 3. Process Each Image
    for i, img_path in enumerate(image_files):
        filename = os.path.basename(img_path)
        print(f"[{i+1}/{len(image_files)}] Processing {filename}...")
        
        img_rgb = load_and_preprocess(img_path)
        if img_rgb is None: continue

        h, w, _ = img_rgb.shape

        # --- DYNAMIC ROI CALCULATION ---
        # We assume the object is roughly in the center.
        # We create a box that is 20% of the image size in the center.
        box_w, box_h = int(w * 0.2), int(h * 0.2)
        center_x, center_y = w // 2, h // 2
        
        roi_rect = (center_x - box_w//2, center_y - box_h//2, box_w, box_h)
        seed_point = (center_y, center_x)

        # --- RUN METHODS ---
        
        # Method 1: K-Means
        label_img = apply_kmeans_lab(img_rgb, k=3)
        target_id = select_cluster_by_roi(label_img, roi_rect)
        mask_kmeans = post_process_mask(label_img == target_id)
        
        # Method 2: Region Growing
        mask_rg = apply_region_growing(img_rgb, seed_point, tolerance=30)
        mask_rg = post_process_mask(mask_rg)

        # Method 3: Iterative Thresholding
        mask_thresh = apply_iterative_threshold(img_rgb)
        mask_thresh = post_process_mask(mask_thresh)

        # Canny Edges
        edges = apply_canny(img_rgb)

        # --- SAVE RESULTS ---
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f"Segmentation Results: {filename}", fontsize=16)

        # 1. Original
        img_copy = img_rgb.copy()
        cv2.rectangle(img_copy, (roi_rect[0], roi_rect[1]), 
                      (roi_rect[0]+roi_rect[2], roi_rect[1]+roi_rect[3]), (255, 255, 0), 3)
        axes[0, 0].imshow(img_copy)
        axes[0, 0].set_title("Original + ROI Box (Center)")
        axes[0, 0].axis('off')

        # 2. Canny
        axes[0, 1].imshow(edges, cmap='gray')
        axes[0, 1].set_title("Canny Edges")
        axes[0, 1].axis('off')

        # 3. Comparison
        axes[0, 2].axis('off') # Placeholder

        # 4. K-Means Result
        res1 = img_rgb.copy()
        res1[~mask_kmeans.astype(bool)] = 0
        axes[1, 0].imshow(res1)
        axes[1, 0].set_title("Method 1: K-Means (Color)")
        axes[1, 0].axis('off')

        # 5. Region Growing Result
        res2 = img_rgb.copy()
        res2[~mask_rg.astype(bool)] = 0
        axes[1, 1].imshow(res2)
        axes[1, 1].set_title("Method 2: Region Growing (Spatial)")
        axes[1, 1].axis('off')

        # 6. Thresholding Result
        res3 = img_rgb.copy()
        res3[~mask_thresh.astype(bool)] = 0
        axes[1, 2].imshow(res3)
        axes[1, 2].set_title("Method 3: Iterative Threshold")
        axes[1, 2].axis('off')

        # Save to disk
        save_path = os.path.join(OUTPUT_FOLDER, f"result_{filename}")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig) # Important to free memory
        
        print(f"Saved result to: {save_path}")

    print("Batch processing complete.")