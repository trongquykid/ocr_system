import argparse
import json
from collections import defaultdict

import torch

from surya.input.langs import replace_lang_with_code, get_unique_langs
from surya.input.load import load_from_folder, load_from_file, load_lang_file
from surya.model.detection.segformer import load_model as load_detection_model, load_processor as load_detection_processor
from surya.model.recognition.model import load_model as load_recognition_model
from surya.model.recognition.processor import load_processor as load_recognition_processor
from surya.model.recognition.tokenizer import _tokenize
from surya.ocr import run_ocr
from surya.postprocessing.text import draw_text_on_image
from surya.settings import settings

from VietnameseOcrCorrection.inferenceModel import check_correct_ocr, count_words
import os

INPUT_PATH = 'image_test/Image_pdf/Screenshot 2024-06-28 210551.png'

def detect_bboxes(input_path, max_pages=None, start_page=0):

    if os.path.isdir(input_path):
        images, names = load_from_folder(input_path, max_pages, start_page)
        folder_name = os.path.basename(input_path)
    else:
        images, names = load_from_file(input_path, max_pages, start_page)
        folder_name = os.path.basename(input_path).split(".")[0]

    det_processor = load_detection_processor()
    det_model = load_detection_model()

    return images, names, folder_name, det_model, det_processor

def recognize_text(images, names, folder_name, det_model, 
                   det_processor, results_dir=os.path.join(settings.RESULT_DIR, "surya"), 
                   save_images=True,  lang_file=None, langs=None):
    
    assert langs or lang_file, "Must provide either --langs or --lang_file"

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

    result_path = os.path.join(results_dir, folder_name)
    os.makedirs(result_path, exist_ok=True)

    predictions_by_image = run_ocr(images, image_langs, det_model, det_processor, rec_model, rec_processor)

    if save_images:
        for idx, (name, image, pred, langs) in enumerate(zip(names, images, predictions_by_image, image_langs)):
            bboxes = [l.bbox for l in pred.text_lines]
            pred_text = [l.text for l in pred.text_lines]

            # text = ' '.join(pred_text)
            pred_text = check_correct_ocr(pred_text) # Check correct Vietnameses

            page_image = draw_text_on_image(bboxes, pred_text, image.size, langs, has_math="_math" in langs)
            page_image.save(os.path.join(result_path, f"{name}_{idx}_text.png"))

    out_preds = defaultdict(list)
    for name, pred, image in zip(names, predictions_by_image, images):
        out_pred = pred.model_dump()
        out_pred["page"] = len(out_preds[name]) + 1
        out_preds[name].append(out_pred)

    with open(os.path.join(result_path, "results.json"), "w+", encoding="utf-8") as f:
        json.dump(out_preds, f, ensure_ascii=False)

    print(f"Wrote results to {result_path}")

images, names, folder_name, det_model, det_processor = detect_bboxes(INPUT_PATH)
recognize_text(images, names, folder_name, det_model, det_processor, langs='vi')