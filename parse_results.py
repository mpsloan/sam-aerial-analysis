"""
parse_results.py

Parses the raw output of sam_miou.py (miou_results.json) and extracts
the specific summary statistics reported in the project's final paper
and presentation:

  - Overall mIoU across the full evaluation set
  - Per-category mIoU, sorted from best to worst performing
  - The single best-performing image (highest mIoU)
  - The single worst-performing image (lowest mIoU)

Usage:
    python parse_results.py [path_to_miou_results.json]

    If no path is given, defaults to "miou_results.json" in the
    current directory.
"""

import json
import sys


def load_results(path):
    """Load the raw mIoU results JSON produced by sam_miou.py.

    Args:
        path (str): File path to the miou_results.json file to load.

    Returns:
        dict: Parsed JSON contents, containing the keys
            'overall_miou', 'total_gt_objects', 'total_images_evaluated',
            'per_category_miou', and 'per_image_results'.
    """
    with open(path, "r") as f:
        return json.load(f)


def print_overall_miou(data):
    """Print the overall mIoU across all evaluated images/objects.

    Args:
        data (dict): Parsed results dictionary, as returned by
            load_results(), containing 'overall_miou',
            'total_gt_objects', and 'total_images_evaluated'.
    """
    print("=" * 50)
    print("OVERALL mIoU")
    print("=" * 50)
    print(f"Overall mIoU:            {data['overall_miou']:.4f}")
    print(f"Total GT objects:        {data['total_gt_objects']}")
    print(f"Total images evaluated:  {data['total_images_evaluated']}")
    print()


def print_per_category_miou(data):
    """Print per-category mIoU, sorted from best to worst performing.

    Args:
        data (dict): Parsed results dictionary, as returned by
            load_results(), containing a 'per_category_miou' key
            mapping category names (str) to mIoU scores (float).
    """
    print("=" * 50)
    print("PER-CATEGORY mIoU (best to worst)")
    print("=" * 50)
    categories = sorted(
        data["per_category_miou"].items(),
        key=lambda item: item[1],
        reverse=True,
    )
    for category, miou in categories:
        print(f"  {category:<20s} {miou:.4f}")
    print()


def print_best_and_worst_images(data):
    """Print the single best and worst performing images by mIoU.

    Args:
        data (dict): Parsed results dictionary, as returned by
            load_results(), containing a 'per_image_results' key: a
            list of per-image dicts, each with 'filename', 'miou',
            'num_gt_objects', 'num_sam_masks', and 'categories'.
    """
    images = data["per_image_results"]
    best = max(images, key=lambda img: img["miou"])
    worst = min(images, key=lambda img: img["miou"])

    print("=" * 50)
    print("BEST-PERFORMING IMAGE")
    print("=" * 50)
    print(f"  Filename:        {best['filename']}")
    print(f"  mIoU:            {best['miou']:.4f}")
    print(f"  GT objects:      {best['num_gt_objects']}")
    print(f"  SAM masks:       {best['num_sam_masks']}")
    print(f"  Categories:      {', '.join(best['categories'])}")
    print()

    print("=" * 50)
    print("WORST-PERFORMING IMAGE")
    print("=" * 50)
    print(f"  Filename:        {worst['filename']}")
    print(f"  mIoU:            {worst['miou']:.4f}")
    print(f"  GT objects:      {worst['num_gt_objects']}")
    print(f"  SAM masks:       {worst['num_sam_masks']}")
    print(f"  Categories:      {', '.join(worst['categories'])}")
    print()


def main():
    """Load the results file and print all summary statistics.

    Reads the path to miou_results.json from the command line if
    provided (sys.argv[1]); otherwise defaults to
    "miou_results.json" in the current working directory.
    """
    results_path = sys.argv[1] if len(sys.argv) > 1 else "miou_results.json"

    try:
        data = load_results(results_path)
    except FileNotFoundError:
        print(f"Error: could not find '{results_path}'.")
        print("Run sam_miou.py first, or pass the path to an existing "
              "miou_results.json as an argument.")
        sys.exit(1)

    print_overall_miou(data)
    print_per_category_miou(data)
    print_best_and_worst_images(data)


if __name__ == "__main__":
    main()
