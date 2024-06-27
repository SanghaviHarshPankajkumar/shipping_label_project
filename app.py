import streamlit as st
import cv2
from pipeline import main
from pathlib import Path
import pandas as pd
import os
from dotenv import load_dotenv
from pathlib import Path
from pipeline_functions import object_detection, crop_image, enhance_image, morphological_transform, hoffman_transform, pytesseract_rotate, ocr,ner

env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

path = {
        'SEG_MODEL_PATH': str(os.getenv('SEG_MODEL_PATH')),
        'MAIN_FLOW_GRAY_IMG_DIR_PATH': str(os.getenv('MAIN_FLOW_GRAY_IMG_DIR_PATH')),
        'MAIN_FLOW_INFERENCE_FOLDER': str(os.getenv('MAIN_FLOW_INFERENCE_FOLDER')),
        }

with st.sidebar:
    st.title("Shipping Label Extraction")
    data = st.file_uploader(label='Upload Image of Parcel',type=['png','jpg','jpeg'])


if data:
    Path('grey_images').mkdir(parents=True, exist_ok=True)

    with open(os.path.join('grey_images',data.name),'wb') as f:
        f.write(data.getvalue())
        
    img = cv2.imread(os.path.join('grey_images',data.name),0)
    
    if img.shape[0] > 1500:
        height, width = img.shape 
        img = img[height//4:-height//4, width//4:-width//4]
    
    cv2.imwrite(os.path.join('grey_images',data.name), img)

    #call main function
    # main(os.path.join('grey_images',data.name))
    file_path = os.path.join('grey_images',data.name)
    img_name = os.path.basename(file_path)


    col1,col2 = st.columns(2)

    with col1:
        st.markdown("<h3 style='text-align: center;'>Grey Image</h1>", unsafe_allow_html=True) 
        st.image(os.path.join('grey_images',data.name))

        # Object detection and enhance image
        seg_result, img_file = object_detection(file_path)
        croped_img = crop_image(seg_result, img_file, img_name)
        image = enhance_image(croped_img, img_name)

        st.markdown("<h3 style='text-align: center;'>Enhanced Image</h1>", unsafe_allow_html=True) 
        st.image(os.path.join('runs', 'segment', path['MAIN_FLOW_INFERENCE_FOLDER'], 'enhanced', data.name))

    
    with col2:
        st.markdown("<h3 style='text-align: center;'>Detected Image</h1>", unsafe_allow_html=True) 
        st.image(os.path.join('runs', 'segment',path['MAIN_FLOW_INFERENCE_FOLDER'],data.name))

        # Rotation
        processed_img = morphological_transform(image)
        rotated_image, image = hoffman_transform(processed_img, image)
        img_name = pytesseract_rotate(rotated_image, image, img_name)

        st.markdown("<h3 style='text-align: center;'>Rotated Image</h1>", unsafe_allow_html=True) 
        st.image(os.path.join('runs', 'segment', path['MAIN_FLOW_INFERENCE_FOLDER'], 'rotated_image', data.name))

    # Apply OCR and NER
    file_name = ocr(img_name)
    Output_dict = ner(file_name)
    # df = pd.DataFrame(Output_dict)

    ocr_data = ""
    with open(os.path.join('runs', 'segment', path['MAIN_FLOW_INFERENCE_FOLDER'], 'ocr_label_data', data.name.split('.')[0]+'.txt'),'r+') as f :
        ocr_data = f.read()
    st.header("OCR Text Output")
    st.text(ocr_data)

    st.header("NER Output")
    
    new_df = pd.DataFrame()
    new_df['Entity'] = list(Output_dict.keys())
    
    # print(df)
    new_df['Value'] = list(Output_dict.values())
    new_df['Value'] = new_df['Value'].astype('str')
    st.table(new_df)

else:
    img_name = '3.jpg'
    img = cv2.imread(img_name,0)
    
    if img.shape[0] > 1500:
        height, width = img.shape 
        img = img[height//4:-height//4, width//4:-width//4]
    
    cv2.imwrite(os.path.join('grey_images',img_name), img)

    #call main function
    # main(os.path.join('grey_images',img_name))
    file_path = os.path.join('grey_images',img_name)
    img_name = os.path.basename(file_path)


    col1,col2 = st.columns(2)

    with col1:
        st.markdown("<h3 style='text-align: center;'>Grey Image</h1>", unsafe_allow_html=True) 
        st.image(os.path.join('grey_images',img_name))

        # Object detection and enhance image
        seg_result, img_file = object_detection(file_path)
        croped_img = crop_image(seg_result, img_file, img_name)
        image = enhance_image(croped_img, img_name)

        st.markdown("<h3 style='text-align: center;'>Enhanced Image</h1>", unsafe_allow_html=True) 
        st.image(os.path.join('runs', 'segment', path['MAIN_FLOW_INFERENCE_FOLDER'], 'enhanced', img_name))

    
    with col2:
        st.markdown("<h3 style='text-align: center;'>Detected Image</h1>", unsafe_allow_html=True) 
        st.image(os.path.join('runs', 'segment',path['MAIN_FLOW_INFERENCE_FOLDER'],img_name))

        # Rotation
        processed_img = morphological_transform(image)
        rotated_image, image = hoffman_transform(processed_img, image)
        img_name = pytesseract_rotate(rotated_image, image, img_name)

        st.markdown("<h3 style='text-align: center;'>Rotated Image</h1>", unsafe_allow_html=True) 
        st.image(os.path.join('runs', 'segment', path['MAIN_FLOW_INFERENCE_FOLDER'], 'rotated_image', img_name))

    # Apply OCR and NER
    file_name = ocr(img_name)
    Output_dict = ner(file_name)
    # df = pd.DataFrame(Output_dict)

    ocr_data = ""
    with open(os.path.join('runs', 'segment', path['MAIN_FLOW_INFERENCE_FOLDER'], 'ocr_label_data', img_name.split('.')[0]+'.txt'),'r+') as f :
        ocr_data = f.read()
    st.header("OCR Text Output")
    st.text(ocr_data)

    st.header("NER Output")
    
    new_df = pd.DataFrame()
    new_df['Entity'] = list(Output_dict.keys())
    
    # print(df)
    new_df['Value'] = list(Output_dict.values())
    new_df['Value'] = new_df['Value'].astype('str')
    st.table(new_df)