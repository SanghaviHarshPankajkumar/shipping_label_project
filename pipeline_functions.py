import cv2
from ultralytics import YOLO 
import numpy as np
from paddleocr import PaddleOCR


from ObjectDetection.functions import generateMask, cropBlackBackground, enhanceImage
from OCR.rotation_functions import hoffman_transformation, rotate, pytesseractRotate

import os
from dotenv import load_dotenv
from pathlib import Path
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

path = {
        'SEG_MODEL_PATH': str(os.getenv('SEG_MODEL_PATH')),
        'MAIN_FLOW_GRAY_IMG_DIR_PATH': str(os.getenv('MAIN_FLOW_GRAY_IMG_DIR_PATH')),
        'MAIN_FLOW_INFERENCE_FOLDER': str(os.getenv('MAIN_FLOW_INFERENCE_FOLDER')),
        }
seg_model = YOLO(path['SEG_MODEL_PATH'])

CONF = 0.7




def object_detection(file):
    print("**************************** PERFORMING_OBJECT_DETECTION **************************** ")
    img_file = cv2.imread(file,0)
    img_name = os.path.basename(file)

    Path(os.path.join(path['MAIN_FLOW_GRAY_IMG_DIR_PATH'])).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(os.path.join(path['MAIN_FLOW_GRAY_IMG_DIR_PATH'],img_name),img_file)
    result = seg_model(os.path.join(path['MAIN_FLOW_GRAY_IMG_DIR_PATH'],img_name),conf = CONF,save = True,name = path['MAIN_FLOW_INFERENCE_FOLDER'],exist_ok = True)
    
    return result, img_file
    
    
def crop_image(seg_result, img_file, img_name):
    print("**************************** CROPPING_IMAGE **************************** ")
    for res in seg_result:
        
        croped_img, mask = generateMask(res, img_file)

        if croped_img is not None:
            croped_img = cropBlackBackground(croped_img)

            # save to file
            try: 
                Path('runs').mkdir(parents=True, exist_ok=True)
                Path(os.path.join('runs', 'segment')).mkdir(parents=True, exist_ok=True)
                Path(os.path.join('runs', 'segment', path['MAIN_FLOW_INFERENCE_FOLDER'])).mkdir(parents=True, exist_ok=True)
                Path(os.path.join('runs', 'segment', path['MAIN_FLOW_INFERENCE_FOLDER'], 'crops_seg')).mkdir(parents=True, exist_ok=True)
                Path(os.path.join('runs', 'segment', path['MAIN_FLOW_INFERENCE_FOLDER'], 'masks')).mkdir(parents=True, exist_ok=True)
            except OSError as error:  
                print(error)
                pass
                
            cv2.imwrite(os.path.join('runs', 'segment', path['MAIN_FLOW_INFERENCE_FOLDER'], 'masks', img_name), mask)
            cv2.imwrite(os.path.join('runs', 'segment', path['MAIN_FLOW_INFERENCE_FOLDER'], 'crops_seg', img_name), croped_img )
            return croped_img
    return img_file 

def enhance_image(croped_img, img_name):
    print("**************************** ENHANCE_IMAGE **************************** ")
    image = None
    if croped_img is not None:
        image = enhanceImage(croped_img)
        
        if image is not None:
            try: 
                Path('runs').mkdir(parents=True, exist_ok=True)
                Path(os.path.join('runs', 'segment')).mkdir(parents=True, exist_ok=True)
                Path(os.path.join('runs', 'segment', path['MAIN_FLOW_INFERENCE_FOLDER'])).mkdir(parents=True, exist_ok=True)
                Path(os.path.join('runs', 'segment', path['MAIN_FLOW_INFERENCE_FOLDER'], 'enhanced')).mkdir(parents=True, exist_ok=True)
            except OSError as error:  
                print(error)
                pass
                
        cv2.imwrite(os.path.join('runs', 'segment', path['MAIN_FLOW_INFERENCE_FOLDER'], 'enhanced', img_name), image )
    
    return image

def morphological_transform(image):

    print("**************************** APPLY_MORPHOLOGICAL_TRANSFORM **************************** ")
    processed_img = cv2.resize(image,None,fx=2.7, fy=3)
    kernel = np.ones((2,2),np.uint8)
    processed_img = cv2.dilate(processed_img,kernel)
    sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    processed_img = cv2.filter2D(processed_img, -1, sharpen_kernel)
    
    return processed_img

def hoffman_transform(processed_img, original_img):
    print("**************************** APPLY_HOFFMAN_TRANSFORM **************************** ")
    rotated_image,angle = hoffman_transformation(processed_img, True)
    original_img  = rotate(original_img,angle)
    
    return rotated_image, original_img

def pytesseract_rotate(rotated_image, original_img, img_name):
    print("**************************** APPLY_PYTESSERACT_ROTATION **************************** ")
    rotated_image = pytesseractRotate(rotated_image,original_img,1)

    if rotated_image is not None:
        try: 
            Path('runs').mkdir(parents=True, exist_ok=True)
            Path(os.path.join('runs', 'segment')).mkdir(parents=True, exist_ok=True)
            Path(os.path.join('runs', 'segment', path['MAIN_FLOW_INFERENCE_FOLDER'])).mkdir(parents=True, exist_ok=True)
            Path(os.path.join('runs', 'segment', path['MAIN_FLOW_INFERENCE_FOLDER'], 'rotated_image')).mkdir(parents=True, exist_ok=True)
        except OSError as error:  
            print(error)
            pass
        
        cv2.imwrite(os.path.join('runs', 'segment', path['MAIN_FLOW_INFERENCE_FOLDER'], 'rotated_image', img_name), rotated_image)
        
    return img_name

def ocr(img_name):
    print("**************************** APPLY_OCR **************************** ")
    ocr = PaddleOCR(use_angle_cls=True, lang='en') # need to run only once to download and load model into memory
    result = ocr.ocr(os.path.join('runs', 'segment', path['MAIN_FLOW_INFERENCE_FOLDER'], 'rotated_image', img_name), cls=True)
    
    ocr_output_paddle = []
    if result is not None:
        try:
            for i in result:
                ocr_output_paddle.append(" ".join([line[1][0] for line in i]))
        except:
            pass
        try: 
            Path('runs').mkdir(parents=True, exist_ok=True)
            Path(os.path.join('runs', 'segment')).mkdir(parents=True, exist_ok=True)
            Path(os.path.join('runs', 'segment', path['MAIN_FLOW_INFERENCE_FOLDER'])).mkdir(parents=True, exist_ok=True)
            Path(os.path.join('runs', 'segment', path['MAIN_FLOW_INFERENCE_FOLDER'], 'ocr_label_data')).mkdir(parents=True, exist_ok=True)
        except OSError as error:  
            print(error)

    with open(os.path.join('runs', 'segment', path['MAIN_FLOW_INFERENCE_FOLDER'], 'ocr_label_data', img_name.split('.')[0]) +'.txt',"w+") as f:
        f.write("\n".join(ocr_output_paddle))