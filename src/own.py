import numpy as np
import cv2

def translate_image(image, dx, dy):
    """Translate the image by dx and dy pixels horizontally and vertically, respectively.
    
    Parameters:
    - image: The input image to be translated.
    - dx: The number of pixels to shift the image horizontally (positive for right, negative for left).
    - dy: The number of pixels to shift the image vertically (positive for down, negative for up).
    
    Returns:
    - The translated image.
    """
    translation_matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted_image = cv2.warpAffine(image, translation_matrix, (image.shape[1], image.shape[0]))
    return shifted_image

def rotate_image(image, angle_degrees, scale_factor):
    """Rotate the image by a specified angle in degrees with a given scale factor.
    
    Parameters:
    - image: The input image to be rotated.
    - angle_degrees: The rotation angle in degrees (positive for counterclockwise, negative for clockwise).
    - scale_factor: The scaling factor for the rotation.
    
    Returns:
    - The rotated image.
    """
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle_degrees, scale_factor)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    return rotated_image

def resize_image(image, new_width=None, new_height=None):
    """Resize the image to the specified width and/or height while maintaining the aspect ratio if only one is provided.
    
    Parameters:
    - image: The input image to be resized.
    - new_width: The desired width of the image (optional).
    - new_height: The desired height of the image (optional).
    
    Returns:
    - The resized image.
    """
    height, width = image.shape[:2]
    
    if new_width is None and new_height is None:
        return image
    elif new_width is None:
        aspect_ratio = new_height / height
        new_dimensions = (int(width * aspect_ratio), new_height)
    elif new_height is None:
        aspect_ratio = new_width / width
        new_dimensions = (new_width, int(height * aspect_ratio))
    else:
        new_dimensions = (new_width, new_height)
    
    resized_image = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_AREA)
    return resized_image

def extract_contours(find_contours_output):
    """Extract the list of contours from the output of cv2.findContours.
    
    Parameters:
    - find_contours_output: The output tuple from cv2.findContours.
    
    Returns:
    - The list of contours.
    """
    return find_contours_output[0]
