import numpy as np
import cv2


'''
Contract the larger dimension of the rectangle make the ROI dimensions square
'''
def roi_square(roi):
    x, y, w, h = roi
    if w > h:
        x += int((w-h)/2.0)
        w = h
    elif h > w:
        y += int((h-w)/2.0)
        h = w
    return x, y, w, h


'''
Scale the ROI dimensions by a factor.
'''
def roi_scale(roi, factor):
    x, y, w, h = roi
    x -= int(w*(factor-1.0)/2.0)
    y -= int(h*(factor-1.0)/2.0)
    w = int(w*factor)
    h = int(h*factor)
    return x, y, w, h


'''
Resize ROI. Square the ROI and scale. Return the resulting ROI image. Padded
with zeros if the new ROI spills out of the image.
'''
def roi_image_resize(image, x, y, w, h, factor, width, height):
    # Get the new ROI dimensions
    x, y, w, h = roi_square(roi_scale((x, y, w, h), factor))

    # Get the valid portion of the ROI
    valid = image[0 if y < 0 else y:image.shape[0] if y+h > image.shape[0] else y+h,
                  0 if x < 0 else x:image.shape[1] if x+w > image.shape[1] else x+w]

    # Zero pad to the expected dimensions
    padded = cv2.copyMakeBorder(valid,
                                -y if y < 0 else 0,
                                (y+h)-image.shape[0] if y+h > image.shape[0] else 0,
                                -x if x < 0 else 0,
                                (x+w)-image.shape[1] if x+w > image.shape[1] else 0,
                                cv2.BORDER_CONSTANT,
                                (0,0,0))

    return (x, y, w, h), cv2.resize(padded, (width, height))
    

'''
Preproccessing: These functions must work on multi-channel images
'''

'''
Resize made easy
'''
def resize_factor(image, factor):
    assert factor >= 0                
    if factor == 0:
        return image
    elif factor > 1:
        method = cv2.INTER_AREA
    else:
        method = cv2.INTER_CUBIC

    return cv2.resize(image, 
                      (int(factor*image.shape[1]), int(factor*image.shape[0])), 
                      interpolation=method)


'''
Rotates an image 90 degrees counter-clockwise
'''
def rotate_90(image):
    dest = np.zeros([image.shape[1], image.shape[0], len(image.shape)], image.dtype)
    for channel in range(len(image.shape)):
        dest[:,:,channel] =  np.flip(np.transpose(image[:,:,channel]), axis=1)
    return dest


'''
Rotates an image 180 degrees counter-clockwise
'''
def rotate_180(image):
    for channel in range(len(image.shape)):
        image[:,:,channel] = np.flip(np.flip(image[:,:,channel], axis=0), axis=1)
    return image


'''
Rotates an image 270 degrees counter-clockwise
'''
def rotate_270(image):
    dest = np.zeros([image.shape[1], image.shape[0], len(image.shape)], image.dtype)
    for channel in range(len(image.shape)):
        dest[:,:,channel] = np.flip(np.transpose(image[:,:,channel]), axis=0)
    return dest
