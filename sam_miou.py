import numpy as np
import cv2
import json
import os
import random
import time
from segment_anything import SamAutomaticMaskGenerator
from segment_anything import sam_model_registry


def load_annotations(annotation_path):
    """Load iSAID annotations and organize by image filename."""
    with open(annotation_path, 'r') as f:
        data = json.load(f)

    image_id_to_filename = {img['id']: img['file_name'] for img in data['images']}
    image_id_to_size = {img['id']: (img['width'], img['height']) for img in data['images']}

    annotations_by_file = {}
    for ann in data['annotations']:
        filename = image_id_to_filename[ann['image_id']]
        if filename not in annotations_by_file:
            annotations_by_file[filename] = []
        annotations_by_file[filename].append({
            'segmentation': ann['segmentation'],
            'category_name': ann['category_name'],
            'area': ann['area']
        })

    return annotations_by_file, image_id_to_size, image_id_to_filename


def polygon_to_mask(polygon, height, width):
    """Convert a COCO polygon to a binary mask."""
    mask = np.zeros((height, width), dtype=np.uint8)
    pts = np.array(polygon, dtype=np.int32).reshape(-1, 2)
    cv2.fillPoly(mask, [pts], 1)
    return mask


def compute_iou(mask1, mask2):
    """Compute Intersection over Union between two binary masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return intersection / union


def main():
    random.seed(42)

    image_directory = '../validation_dataset/images_v1/'
    annotation_path = '../validation_dataset/Annotations/iSAID_val.json'

    all_images = [f for f in os.listdir(image_directory) if f.endswith('.png')]
    selected_images = random.sample(all_images, 100)

    annotations_by_file, image_id_to_size, _ = load_annotations(annotation_path)

    sam_checkpoint = "../segment-anything/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "mps"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(sam)

    all_ious = []
    image_miou_results = []
    global_category_ious = {}

    for i, filename in enumerate(selected_images):
        print(f"Processing {filename} ({i + 1}/{len(selected_images)})")

        image_path = os.path.join(image_directory, filename)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        orig_h, orig_w = image.shape[:2]
        image_resized = cv2.resize(image, (1024, 1024))

        start = time.time()
        masks = mask_generator.generate(image_resized)
        elapsed = time.time() - start

        print(f"  SAM masks generated: {len(masks)} | Inference time: {elapsed:.1f}s")

        if filename not in annotations_by_file:
            print(f"  No annotations found for {filename}, skipping\n")
            continue

        gt_annotations = annotations_by_file[filename]

        scale_x = 1024 / orig_w
        scale_y = 1024 / orig_h

        image_ious = []
        category_ious = {}

        for ann in gt_annotations:
            polygon = ann['segmentation'][0]
            scaled_polygon = []
            for j in range(0, len(polygon), 2):
                scaled_polygon.append(polygon[j] * scale_x)
                scaled_polygon.append(polygon[j + 1] * scale_y)

            gt_mask = polygon_to_mask(scaled_polygon, 1024, 1024)

            if gt_mask.sum() == 0:
                continue

            best_iou = 0.0
            for sam_mask in masks:
                iou = compute_iou(gt_mask, sam_mask['segmentation'])
                if iou > best_iou:
                    best_iou = iou

            image_ious.append(best_iou)

            # Track per-category IoU
            cat = ann['category_name']
            if cat not in category_ious:
                category_ious[cat] = []
            category_ious[cat].append(best_iou)

            # Track global per-category IoU
            if cat not in global_category_ious:
                global_category_ious[cat] = []
            global_category_ious[cat].append(best_iou)

        if image_ious:
            image_miou = np.mean(image_ious)
            image_miou_results.append({
                'filename': filename,
                'miou': image_miou,
                'num_gt_objects': len(image_ious),
                'num_sam_masks': len(masks),
                'inference_time': round(elapsed, 1),
                'categories': list(category_ious.keys()),
                'per_category_miou': {cat: round(float(np.mean(ious)), 4)
                                      for cat, ious in category_ious.items()}
            })
            all_ious.extend(image_ious)
            print(f"  GT objects: {len(image_ious)} | Image mIoU: {image_miou:.4f}\n")

    # Overall results
    overall_miou = np.mean(all_ious)
    global_category_miou = {cat: round(float(np.mean(ious)), 4)
                            for cat, ious in global_category_ious.items()}

    print(f"\n{'='*40}")
    print(f"Overall mIoU across {len(image_miou_results)} images: {overall_miou:.4f}")
    print(f"Total GT objects evaluated: {len(all_ious)}")
    print(f"\nPer-category mIoU:")
    for cat, miou in sorted(global_category_miou.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cat}: {miou:.4f}")

    with open('miou_results.json', 'w') as f:
        json.dump({
            'overall_miou': round(float(overall_miou), 4),
            'total_gt_objects': len(all_ious),
            'total_images_evaluated': len(image_miou_results),
            'per_category_miou': global_category_miou,
            'per_image_results': image_miou_results
        }, f, indent=2)

    print("\nResults saved to miou_results.json")


if __name__ == "__main__":
    main()
