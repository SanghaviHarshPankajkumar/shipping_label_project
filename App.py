import streamlit as st
import cv2
from pipeline import main
from pathlib import Path
import pandas as pd
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
    Output_dict= main(os.path.join('grey_images',data.name))
    df = pd.DataFrame(Output_dict)

    col1,col2 = st.columns(2)

    with col1:
        st.markdown("<h3 style='text-align: center;'>Grey Image</h1>", unsafe_allow_html=True) 
        st.image(os.path.join('grey_images',data.name))


        st.markdown("<h3 style='text-align: center;'>Enhanced Image</h1>", unsafe_allow_html=True) 
        st.image(os.path.join('runs', 'segment', path['MAIN_FLOW_INFERENCE_FOLDER'], 'enhanced', data.name))

    with col2:
        st.markdown("<h3 style='text-align: center;'>Detected Image</h1>", unsafe_allow_html=True) 
        st.image(os.path.join('runs', 'segment',path['MAIN_FLOW_INFERENCE_FOLDER'],data.name))


        st.markdown("<h3 style='text-align: center;'>Rotated Image</h1>", unsafe_allow_html=True) 
        st.image(os.path.join('runs', 'segment', path['MAIN_FLOW_INFERENCE_FOLDER'], 'rotated_image', data.name))


    ocr_data = ""
    with open(os.path.join('runs', 'segment', path['MAIN_FLOW_INFERENCE_FOLDER'], 'ocr_label_data', data.name.split('.')[0]+'.txt'),'r+') as f :
        ocr_data = f.read()
    st.header("OCR Text Output")
    st.text(ocr_data)

    st.header("NER Output")
    st.table(df)