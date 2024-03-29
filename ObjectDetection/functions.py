import torch
import cv2
import numpy as np

def cropBlackBackground(img):
    """Removes black background of labbel image and returns portion which contains label only.
    It will need gray image.

    Args:
        img (ndarray): Numpy array representation of image

    Returns:
        img (ndarray): Numpy array representation of cropped image
    """
    try:
        _, binary = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x, y, w, h = cv2.boundingRect(contours[0])
        img = img[y:y+h, x:x+w]
        
        return img
    except Exception as e:
        print(e)
        return None

def     enhanceImage(img, block_size: int = 19, constant: int = 5, adaptive_thresold_type = "GAUSSIAN", need_to_sharp: bool = True):
    """Enhance image by appling adaptive thresolding and filter2D

    Args:
        img (ndarray): Numpy array representation of image
        block_size (int, optional): Block size for adaptive thresolding. Defaults to 25.
        constant (int, optional): Constant for adaptive thresolding. Defaults to 10.
        adaptive_thresold_type (str, optional): "GAUSSIAN" or "MEAN. Defaults to "GAUSSIAN".
        need_to_sharp (bool, optional): Defaults to True.

    Returns:
        img (ndarray): Numpy array representation of enhanced image
    """
    try:
        if block_size < 2:
            block_size = 2
        block_size = block_size + 1 if block_size%2 == 0 else block_size
        
        final_img = img
        if adaptive_thresold_type == "MEAN":
            final_img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,block_size,constant)
        else:
            final_img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,block_size,constant)

        if need_to_sharp:
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            final_img = cv2.filter2D(final_img, -1, kernel)
            
        return final_img
    except Exception as e:
        print(e)
        return None

def generateMask(res, original_img):
    """_summary_

    Args:
        res: Resulf of yolo image segmentation for single mask
        original_img (ndarray): Numpy array representation of original image

    Returns:
        tupple (ndarray, ndarray): (crop_img, mask)
    """
    
    try:
        height,width  = original_img.shape
        masks = res.masks.data
        boxes = res.boxes.data
        
        # extract classes
        clss = boxes[:, 5]
        # get indices of ress where class is 0 
        label_indices = torch.where(clss == 0)
        # use these indices to extract the relevant masks
        print("INSIDE GENEARED")
        label_masks = masks[label_indices]
        # scale for visualizing ress
        label_mask = torch.any(label_masks, dim=0).int() * 255

        #final mask 
        final_mask = label_mask.cpu().numpy()
        height_mask,width_mask =final_mask.shape
        fy = height/height_mask
        fx = width/width_mask
        final_mask = cv2.resize(final_mask,(0,0),fx =fx ,fy = fy,interpolation=cv2.INTER_NEAREST)

        original_img =  original_img.astype(np.uint8)
        final_mask = final_mask.astype(np.uint8)
        
        # Expand boundries
        kernel = np.ones((40,40), np.uint8)
        expanded_mask =  cv2.dilate(final_mask, kernel)

        #crop_img
        crop_img = cv2.bitwise_and(original_img,original_img,mask=expanded_mask)


        return crop_img, expanded_mask

    except Exception as e:
        print(e)
        return None, None