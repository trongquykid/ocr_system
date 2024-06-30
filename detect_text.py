import argparse
import copy
import json
from collections import defaultdict

from surya.input.load import load_from_folder, load_from_file
from surya.model.detection.segformer import load_model, load_processor
from surya.detection import batch_text_detection
from surya.postprocessing.affinity import draw_lines_on_image
from surya.postprocessing.heatmap import draw_polys_on_image
from surya.settings import settings
import os
from tqdm import tqdm

INPUT_PATH = 'image_test/Image_pdf/08-2022-TT-BCA_page-0001.jpg'

def process_input(input_path):

    # Defaults
    results_dir = os.path.join(settings.RESULT_DIR, "surya")
    max_pages = None
    save_images = True
    debug_mode = False
    use_math_model = False

    # Determine the model checkpoint
    checkpoint = settings.DETECTOR_MATH_MODEL_CHECKPOINT if use_math_model else settings.DETECTOR_MODEL_CHECKPOINT
    model = load_model(checkpoint=checkpoint)
    processor = load_processor(checkpoint=checkpoint)

    # Load images from input_path
    if os.path.isdir(input_path):
        images, names = load_from_folder(input_path, max_pages)
        folder_name = os.path.basename(input_path)
    else:
        images, names = load_from_file(input_path, max_pages)
        folder_name = os.path.basename(input_path).split(".")[0]

    # Perform text detection
    predictions = batch_text_detection(images, model, processor)
    result_path = os.path.join(results_dir, folder_name)
    os.makedirs(result_path, exist_ok=True)

    # Save images if required
    if save_images:
        for idx, (image, pred, name) in enumerate(zip(images, predictions, names)):
            polygons = [p.polygon for p in pred.bboxes]
            bbox_image = draw_polys_on_image(polygons, copy.deepcopy(image))
            bbox_image.save(os.path.join(result_path, f"{name}_{idx}_bbox.png"))

            column_image = draw_lines_on_image(pred.vertical_lines, copy.deepcopy(image))
            column_image.save(os.path.join(result_path, f"{name}_{idx}_column.png"))

            if debug_mode:
                heatmap = pred.heatmap
                heatmap.save(os.path.join(result_path, f"{name}_{idx}_heat.png"))

                affinity_map = pred.affinity_map
                affinity_map.save(os.path.join(result_path, f"{name}_{idx}_affinity.png"))

    # Collect predictions by page
    predictions_by_page = defaultdict(list)
    for idx, (pred, name, image) in enumerate(zip(predictions, names, images)):
        out_pred = pred.model_dump(exclude=["heatmap", "affinity_map"])
        out_pred["page"] = len(predictions_by_page[name]) + 1
        predictions_by_page[name].append(out_pred)

    # Save results to JSON file
    with open(os.path.join(result_path, "results.json"), "w+", encoding="utf-8") as f:
        json.dump(predictions_by_page, f, ensure_ascii=False)

    print(f"Wrote results to {result_path}")

process_input(INPUT_PATH)