import argparse
import json
import copy
from collections import defaultdict

from surya.input.langs import replace_lang_with_code, get_unique_langs
from surya.input.load import load_from_folder, load_from_file, load_lang_file
from surya.model.detection.segformer import load_model as load_detection_model, load_processor as load_detection_processor
from surya.model.recognition.model import load_model as load_recognition_model
from surya.model.recognition.processor import load_processor as load_recognition_processor
from surya.model.recognition.tokenizer import _tokenize
from surya.ocr import run_ocr_v2
from surya.detection import batch_text_detection
from surya.postprocessing.text import draw_text_on_image, draw_text_on_image_v2
from surya.postprocessing.affinity import draw_lines_on_image
from surya.postprocessing.heatmap import draw_polys_on_image
from surya.settings import settings

from VietnameseOcrCorrection.inferenceModel import check_correct_ocr, count_words, check_correct_paragraph
from get_information import get_content, save_json
import os

# INPUT_PATH: the input path of the image
INPUT_PATH = 'image_test/Passport/Specimen_Personal_Information_Page_South_Korean_Passport.jpg'

# TYPE: types of images to be processed. Ex: passport, cccd, pdf, ...
TYPE = "passport"

# LANG: Specified language used. Ex: English: en, Vietnamese: vi
LANG = 'en'

# Increase tọa độ
def adjust_y_coordinates(polygon, decrease_percentage, increase_percentage):
    min_y = min(polygon, key=lambda point: point[1])[1]
    max_y = max(polygon, key=lambda point: point[1])[1]

    adjusted_polygon = []
    for x, y in polygon:
        if y == min_y:
            adjusted_y = int(y * (1 - decrease_percentage / 100))
        elif y == max_y:
            adjusted_y = int(y * (1 + increase_percentage / 100))
        else:
            adjusted_y = y
        adjusted_polygon.append([x, adjusted_y])
    
    return adjusted_polygon

def save_results(result_path, file_json, predictions, names, images):

    predictions_note = defaultdict(list)
    
    for pred, name, image in zip(predictions, names, images):
        out_pred = pred.model_dump(exclude=["heatmap", "affinity_map"])
        out_pred["page"] = len(predictions_note[name]) + 1
        predictions_note[name].append(out_pred)

    # Save results to JSON file
    save_json(os.path.join(result_path, file_json), predictions_note)
    return predictions_note

# Text Detection
def detect_bboxes(input_path, results_dir=os.path.join(settings.RESULT_DIR, "surya"), 
                  save_images=True, max_pages=None, start_page=0, type = 'pdf'):

    if os.path.isdir(input_path):
        images, names = load_from_folder(input_path, max_pages, start_page)
        folder_name = os.path.basename(input_path)
    else:
        images, names = load_from_file(input_path, max_pages, start_page, type)
        folder_name = os.path.basename(input_path).split(".")[0]

    result_path = os.path.join(results_dir, folder_name)

    os.makedirs(result_path, exist_ok=True)

    det_processor = load_detection_processor()
    det_model = load_detection_model()
    det_predictions = batch_text_detection(images, det_model, det_processor)

    if save_images:
        for idx, (image, pred, name) in enumerate(zip(images, det_predictions, names)):
            polygons = [p.polygon for p in pred.bboxes]

            polygons = [adjust_y_coordinates(polygon, 0.1, 0.2) for polygon in polygons] 
            
            confidence = [round(p.confidence,2) for p in pred.bboxes]
            bbox_image = draw_polys_on_image(polygons, copy.deepcopy(image))
            bbox_image.save(os.path.join(result_path, f"{name}_{idx}_bbox.png"))

            column_image = draw_lines_on_image(pred.vertical_lines, copy.deepcopy(image))
            column_image.save(os.path.join(result_path, f"{name}_{idx}_column.png"))

    save_results(result_path, "results.json", det_predictions, names, images)

    return images, names, result_path, det_predictions

# Text Recognition
def recognize_text(images, names, result_path, det_predictions, 
                   save_images=True,  lang_file=None, langs=None, type = 'pdf'):
    
    assert langs or lang_file, "Must provide either --langs or --lang_file"
    type = type.lower()
    
    if lang_file:
        langs = load_lang_file(lang_file, names)
        for lang in langs:
            replace_lang_with_code(lang)
        image_langs = langs
    else:
        langs = langs.split(",")
        replace_lang_with_code(langs)
        image_langs = [langs] * len(images)

    _, lang_tokens = _tokenize("", get_unique_langs(image_langs))
    rec_model = load_recognition_model(langs=lang_tokens)
    rec_processor = load_recognition_processor()
    
    predictions_by_image = run_ocr_v2(images, image_langs, det_predictions, rec_model, rec_processor)

    if save_images:
        for idx, (name, image, pred, langs) in enumerate(zip(names, images, predictions_by_image, image_langs)):

            bboxes = [l.bbox for l in pred.text_lines]
            pred_text = [l.text for l in pred.text_lines]
            pred_confidence = [l.confidence for l in pred.text_lines]
            pred_text = check_correct_ocr(pred_text) # Check correct Vietnameses

            page_image = draw_text_on_image_v2(bboxes, pred_text, pred_confidence, image.size, langs, has_math="_math" in langs)
            page_image.save(os.path.join(result_path, f"{name}_{idx}_text.png"))
    predictions_note = save_results(result_path, "results.json", predictions_by_image, names, images)

    if type == 'passport':
        info_passport = defaultdict(list)
        info_passport = get_content(predictions_note, names[0], "Passport No", info_passport)
        info_passport = get_content(predictions_note, names[0], "Date of issue", info_passport)
        save_json(os.path.join(result_path, "info_passport.json"), info_passport)

if __name__ == "__main__":
    images, names, result_path, det_predictions= detect_bboxes(INPUT_PATH, type=TYPE)
    recognize_text(images, names, result_path, det_predictions, langs=LANG)