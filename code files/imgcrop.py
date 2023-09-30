import cv2
import numpy as np
def crop(word_paths):
    threshold = 200
    cropped_image_paths=[]
    cropped_images=[]
    for word_path in word_paths:
        img = cv2.imread(word_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cropped_row = delete_rows(gray, threshold)
        cropped_image = delete_columns(cropped_row, threshold)
        cropped_image_path = 'res_' + word_path
        cv2.imwrite(cropped_image_path, np.asarray(cropped_image))
        cropped_images.append(cropped_image)
        cropped_image_paths.append(cropped_image_path)
    return cropped_images
def delete_rows(array, threshold):
    return [row for row in array if not all(element > threshold for element in row)]
def delete_columns(array, threshold):
    array = np.array(array)
    delete_cols = np.all(array > threshold, axis=0)
    mask = np.where(delete_cols == False)[0]
    new_array = array[:, mask]
    return new_array.tolist()
