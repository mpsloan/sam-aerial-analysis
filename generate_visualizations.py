"""
generate_visualizations.py

Generates SAM automatic mask generation visualizations for a single
aerial image. This is the script used to produce the qualitative
overlay figures included in the project's paper and presentation
(e.g., the best- and worst-performing image visualizations, such as
P2085_best.png and P1557_worst.png).

Given an image filename from the DOTA/iSAID validation set, this
script runs SAM's automatic mask generator on the resized image and
saves a color-coded overlay showing every predicted mask.

NOTE: Mask colors are randomly assigned per run for visual distinction
only and carry no semantic meaning; colors in the paper/presentation figures
will not exactly match a fresh run of this script.

Usage:
    python generate_visualizations.py <filename> [output_path]

    filename:    Name of the image file (e.g., "P1557.png") located
                 in the validation image directory.
    output_path: Optional path to save the resulting visualization.
                 Defaults to "figures/<filename>".
"""

import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry


def show_anns(anns):
    """Overlay SAM's predicted masks onto the current matplotlib axes.

    Each mask is drawn as a random, semi-transparent color, with
    larger objects drawn first so smaller objects remain visible on
    top. Used to qualitatively inspect SAM's automatic mask
    generation output on aerial imagery.

    Args:
        anns (list[dict]): List of mask dictionaries as returned by
            SamAutomaticMaskGenerator.generate(), each containing a
            'segmentation' boolean mask and an 'area' value.
    """
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=lambda x: x['area'], reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0],
                   sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


def generate_visualization(filename, image_directory, sam_checkpoint,
                           output_path, model_type="vit_h", device="mps"):
    """Run SAM automatic mask generation on one image and save a visualization.

    Args:
        filename (str): Name of the image file to process (e.g., "P1557.png").
        image_directory (str): Directory containing the source image.
        sam_checkpoint (str): Path to the SAM model checkpoint file.
        output_path (str): File path to save the resulting overlay figure.
        model_type (str): SAM model variant to load. Defaults to "vit_h".
        device (str): Torch device to run inference on. Defaults to "mps".
    """
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)

    image_path = os.path.join(image_directory, filename)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (1024, 1024))  # matches sam_miou.py's inference resolution

    masks = mask_generator.generate(image)
    print(f"Number of masks generated for {filename}: {len(masks)}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_anns(masks)
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved visualization to {output_path}")


def main():
    """Parse command-line arguments and generate a single visualization."""
    if len(sys.argv) < 2:
        print("Usage: python generate_visualizations.py <filename> [output_path]")
        sys.exit(1)

    filename = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else f"figures/{filename}"

    image_directory = '../validation_dataset/images_v1/'
    sam_checkpoint = "../segment-anything/sam_vit_h_4b8939.pth"

    generate_visualization(filename, image_directory, sam_checkpoint, output_path)


if __name__ == "__main__":
    main()
