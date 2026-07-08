import cv2
import os
from tqdm import tqdm


def align_optic_disc(input_dir, output_dir):
    """
    Scans a directory of fundus images, locates the optic disc,
    and horizontally flips the image if the disc is on the left.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    valid_exts = (".png", ".jpg", ".jpeg")
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_exts)]

    print(f"[*] Processing {len(image_files)} images from {input_dir}...")

    flipped_count = 0

    for filename in tqdm(image_files, desc="Aligning Geometry"):
        img_path = os.path.join(input_dir, filename)
        out_path = os.path.join(output_dir, filename)

        # 1. Read the image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read {filename}")
            continue

        # 2. Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 3. Apply extreme Gaussian Blur (kernel size 101x101)
        # This destroys all high-frequency details (veins, small diseases)
        # and isolates the massive, bright optic disc.
        blurred = cv2.GaussianBlur(gray, (101, 101), 0)

        # 4. Find the coordinates of the absolute brightest pixel
        _, _, _, max_loc = cv2.minMaxLoc(blurred)
        brightest_x = max_loc[0]

        width = img.shape[1]

        # 5. Flip if the optic disc is on the left half of the image
        if brightest_x < (width // 2):
            aligned_img = cv2.flip(img, 1)  # 1 = horizontal flip
            flipped_count += 1
        else:
            aligned_img = img

        # 6. Save to the new directory
        cv2.imwrite(out_path, aligned_img)

    print(f"\n[*] Alignment Complete!")
    print(f"[*] Flipped {flipped_count} images to ensure right-side optic discs.")
    print(f"[*] Saved aligned dataset to: {output_dir}")


if __name__ == "__main__":
    # Point this to your raw training images
    INPUT_DATASET = "../Raw_datasets/A. RFMiD_All_Classes_Dataset/1. Original Images/val"

    # The new directory for your mathematically aligned data
    OUTPUT_DATASET = "./Raw_datasets/A. RFMiD_All_Classes_Dataset/2. Aligned_Images/val"

    align_optic_disc(INPUT_DATASET, OUTPUT_DATASET)
