from pipeline_functions import object_detection, crop_image, enhance_image, morphological_transform, hoffman_transform, pytesseract_rotate, ocr
import os

def main(path):
    img_name = os.path.basename(path)
    seg_result, img_file = object_detection(path)
    croped_img = crop_image(seg_result, img_file, img_name)
    image = enhance_image(croped_img, img_name)
    processed_img = morphological_transform(image)
    rotated_image, image = hoffman_transform(processed_img, image)
    img_name = pytesseract_rotate(rotated_image, image, img_name)
    ocr(img_name)
    
main('/media/prit/New Volume/AI/shipping_label_project/ObjectDetection/images/parcel_img13.png')