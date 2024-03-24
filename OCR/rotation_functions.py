# method 4 mix two methods 
import cv2
import numpy as np
import os 
import re
import pytesseract 
pytesseract.pytesseract.tesseract_cmd = (r"C:\Program Files\Tesseract-OCR\tesseract.exe")



def hoffman_transformation(image, verbose=False):
    """
    this function performs hoffman transformation method which fixes the rotation of image in 4 angles 0,90,270,360.
    Args:
        image (ndarray): gets image and perform hoffman tarnsformation
        verbose (bool, optional): for seeing image transformation using matplotlib plots. Defaults to False.

    Returns:
        rotated_image: returns rotated image which can be only 4 angles rotated label 
    """
# Define our parameters for Canny
    low_threshold = 50
    high_threshold = 100
    kernel = np.ones((8,8),dtype=np.uint8)
    eroded_image = cv2.erode(image,kernel=kernel)
    eroded_image = cv2.dilate(eroded_image,kernel)

    # perform canny edge detection 
    edges = cv2.Canny(eroded_image, low_threshold, high_threshold)
  
    edges = cv2.erode(edges,(50,50))
 
# Define the Hough transform parameters
# Make a blank the same size as our image to draw on
    rho = 1
    theta = np.pi/180
    threshold = 60
    min_line_length = 10
    max_line_gap = 5
    line_image = np.copy(image) #creating an image copy to draw lines on

    # Run Hough on the edge-detected image
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)
    
    # Iterate over the output "lines" and draw lines on the image copy
    angles_count = {}
    final_angle = 0
    if lines is not None:
        for line in lines:
          if line is not None:
                for x1,y1,x2,y2 in line:
                    cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)
                    
                    angle = 0
                    if abs(x1-x2) < 0.000001:
                        angle = np.pi/2
                    else:
                        angle = (y1-y2)/(x1-x2)
                        angle = np.arctan(angle)
                    angle = angle*180/np.pi
                    angle = np.round(angle)
                    if angle%10 < 5:
                        angle = angle- angle%10
                    else:
                        angle = angle + 10 - angle%10
                    if angle in angles_count:
                        angles_count[angle] += 1
                    else:
                        angles_count[angle] = 1
                    
                    final_angle = max(angles_count, key=angles_count.get)
                    

    line_image = cv2.putText(line_image, str(final_angle), (20,30), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 3, cv2.LINE_8, False)

    angle= 360
    angle-= final_angle
    angle = -(90 + angle) if angle < -45 else -angle

    # rotate image at final_angle using rotation matrix and warpAffine transformation
    h, w = image.shape[:2]
    (c_x, c_y) = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D((c_x, c_y), angle, 1.0)
    
    cos = np.abs(matrix[0, 0])
    sin = np.abs(matrix[0, 1])
    
    n_w = int((h * sin) + (w * cos))
    n_h = int((h * cos) + (w * sin))
    
    matrix[0, 2] += (n_w / 2) - c_x
    matrix[1, 2] += (n_h / 2) - c_y
    
    rotated_image =  cv2.warpAffine(image, matrix, (n_w, n_h), borderValue=(255, 255, 255))
    return rotated_image,angle



def rotate(
        image: np.ndarray, angle: float
) -> np.ndarray:
    """ this function rotates the image at given angle and returns the rotated image

    Args:
        image (np.ndarray): _description_
        angle (float): _description_

    Returns:
        np.ndarray: _description_
    """
    h, w = image.shape[:2]
    (c_x, c_y) = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D((c_x, c_y), angle, 1.0)
    
    cos = np.abs(matrix[0, 0])
    sin = np.abs(matrix[0, 1])
    
    n_w = int((h * sin) + (w * cos))
    n_h = int((h * cos) + (w * sin))
    
    matrix[0, 2] += (n_w / 2) - c_x
    matrix[1, 2] += (n_h / 2) - c_y
    
    return cv2.warpAffine(image, matrix, (n_w, n_h), borderValue=(255, 255, 255))


def pytesseractRotate(image,original_image, grid=3):
    """ this function takes one image and apply pytesseract osd method and gives orientation and script details and returns 0 degree oriented parcel image.

    Args:
        image (ndarray): takes image and perform osd
        original_image (ndarray): _description_
        grid (int, optional): _description_. Defaults to 3.

    Returns:
       rotated_image (ndarray): 
    """
    h, w = image.shape[:2]
    
    images_list = []
    angles_list = {}
    for i in range(1, grid+1):
        for j in range(1, grid+1):
            tx, ty = (w//grid)*(j-1), (h//grid)*(i-1)
            bx, by = (w//grid)*j, (h//grid)*i
            
            
            img = image[ty:by, tx: bx]
            images_list.append(img)
        
    for i in range(len(images_list)):
        try:
            result = pytesseract.image_to_osd(images_list[i], config="osd --psm 0  -c min_characters_to_try=200", output_type='dict')
            pytesseract_angle = result['rotate']
            orientation_conf = result['orientation_conf']
            script = result['script']
            script_conf = result['script_conf']
            
            script_list = ['Latin','Cyrillic']
            
            if script in script_list and script_conf > 0:
                if pytesseract_angle in angles_list:
                    angles_list[pytesseract_angle].append(orientation_conf)
                else:
                    angles_list[pytesseract_angle] = [orientation_conf]
                
        except Exception as error:
            print(error)
    
    confidence_list = []
    for key in angles_list.keys():
        mean = sum(angles_list[key])/len(angles_list[key])
        confidence_list.append((len(angles_list[key]), mean, key))
    
    confidence_list = sorted(confidence_list)

    final_angle = 360
    if len(confidence_list) > 0:        
        final_angle -= confidence_list[-1][-1]
    else:
        final_angle -= 0
        
    rotated_image = rotate(original_image, final_angle)

    return rotated_image
