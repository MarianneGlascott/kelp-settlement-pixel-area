import os
import cv2
import numpy as np
import csv
import argparse

# ---------------------------------------------------------
# Gametophyte settlement quantification (pixel-area metric)
# Methods: black-hat morphology (default) or grayscale adaptive thresholding
#
# Usage examples:
#   Batch QC (interactive overlay):
#     python gametophyte_count.py --method blackhat --visualize --input_dir D:\Tile_photos_cropped_batch_2
#
#   Final CSV export:
#     python gametophyte_count.py --method blackhat --input_dir D:\Tile_photos_cropped_batch_2 --output_csv C:\path\to\gametophyte_counts.csv
# ---------------------------------------------------------

# Default paths (edit to suit your machine)
INPUT_DIR_DEFAULT  = r"D:\Tile_photos_cropped_batch_2"
OUTPUT_CSV_DEFAULT = r"C:\Users\maria\OneDrive - University of Sussex\PHD 2024\Analysis\Python\Tile_calibration_2025\gametophyte_counts.csv"

# Default segmentation parameters (fixed globally across images)
MIN_AREA_PX = 1        # minimum connected-component area to retain
MAX_AREA_PX = 200      # maximum connected-component area to retain (filters glare/shadows)

BLACKHAT_KERNEL = (15, 15)   # black-hat structuring element size
BLACKHAT_THRESH = 30         # fixed threshold applied to black-hat response
MORPH_KERNEL = (3, 3)        # open/close kernel for cleanup

ADAPTIVE_BLOCKSIZE = 25
ADAPTIVE_C = 10


# ---------------------------------------------------------
# Segmentation Methods
# ---------------------------------------------------------
def segment_blackhat(image_bgr: np.ndarray) -> np.ndarray:
    """Return binary mask (uint8 0/255) of segmented gametophyte material using black-hat morphology."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, BLACKHAT_KERNEL)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    _, mask = cv2.threshold(blackhat, BLACKHAT_THRESH, 255, cv2.THRESH_BINARY)

    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k2, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k2, iterations=1)

    # Filter connected components by area
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    filtered = np.zeros_like(mask)

    for i in range(1, num_labels):  # skip background label 0
        area = stats[i, cv2.CC_STAT_AREA]
        if MIN_AREA_PX <= area <= MAX_AREA_PX:
            filtered[labels == i] = 255

    return filtered


def segment_gray(image_bgr: np.ndarray) -> np.ndarray:
    """Fallback grayscale adaptive threshold segmentation."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    mask = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=ADAPTIVE_BLOCKSIZE,
        C=ADAPTIVE_C
    )

    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k2, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k2, iterations=1)

    return mask


# ---------------------------------------------------------
# Utilities
# ---------------------------------------------------------
def pixel_area(mask: np.ndarray) -> int:
    """Count non-zero pixels in a binary mask."""
    return int(cv2.countNonZero(mask))


def process_image(path: str, method: str, visualize: bool) -> int:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image (cv2.imread returned None): {path}")

    if method == 'blackhat':
        mask = segment_blackhat(img)
    elif method == 'gray':
        mask = segment_gray(img)
    else:
        raise ValueError(f"Unknown method: {method}")

    area = pixel_area(mask)

    if visualize:
        print(f"{os.path.basename(path)} => pixel_area: {area}")
        overlay = cv2.addWeighted(img, 0.7, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), 0.3, 0)
        cv2.putText(
            overlay, f"pixel_area: {area}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )
        cv2.imshow('QC overlay', overlay)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return area


# ---------------------------------------------------------
# Batch Processing
# ---------------------------------------------------------
VALID_EXT = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')

def batch_process(input_dir: str, output_csv: str, method: str, visualize: bool) -> None:
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'pixel_area'])

        for fn in sorted(os.listdir(input_dir)):
            if fn.lower().endswith(VALID_EXT):
                full_path = os.path.join(input_dir, fn)
                area = process_image(full_path, method, visualize)
                writer.writerow([fn, area])
                print(f"{fn}: {area}")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Quantify settlement as pixel-area from tile photographs.")
    parser.add_argument('--input_dir', default=INPUT_DIR_DEFAULT, help="Directory containing tile images.")
    parser.add_argument('--method', choices=['blackhat', 'gray'], default='blackhat', help="Segmentation method.")
    parser.add_argument('--visualize', action='store_true', help="Show QC overlay for each image (press any key to advance).")
    parser.add_argument('--output_csv', default=OUTPUT_CSV_DEFAULT, help="Path to output CSV.")
    args = parser.parse_args()

    batch_process(args.input_dir, args.output_csv, args.method, args.visualize)
