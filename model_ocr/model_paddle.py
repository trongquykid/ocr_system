from paddleocr import PaddleOCR
import cv2
ocr = PaddleOCR(lang='en',rec_algorithm='CRNN')

def perform_ocr(image):
    ocr_res = ocr.ocr(image, cls=False, det=False)
    return ocr_res[0][0][0], ocr_res[0][0][1]