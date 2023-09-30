import cv2
import numpy as np
import imgcrop
import os
from PIL import Image as im
from heapq import *
from collections import defaultdict
import typing
from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder

class ImageToParagraphModel(OnnxInferenceModel):

    def _init_(self, char_list: typing.Union[str, list], *args, **kwargs):
        super()._init_(*args, **kwargs)
        self.char_list = char_list

    def predict(self, image: np.ndarray):
        image = np.array(image)
        image = im.fromarray(image)
        image = image.resize((128, 32))
        image = np.array(image)
        image = cv2.resize(image, self.input_shape[:2][::-1])
        image_pred = np.expand_dims(image, axis=0).astype(np.float32)
        image_pred = np.expand_dims(image_pred, axis=3)
        image_pred = np.concatenate([image_pred] * 3, axis=3)
        image_pred = np.transpose(image_pred, (0, 1, 2, 3))
        preds = self.model.run(None, {self.input_name: image_pred})[0]
        text = ctc_decoder(preds, configs.vocab)[0]
        return text

    def predict_paragraph(self, word_paths: list):
        cropped_images = imgcrop.crop(word_paths)
        paragraph_text = ""
        for image in cropped_images:
            word_text = self.predict(image)
            paragraph_text += word_text + " "
        paragraph_text = paragraph_text.strip()
        return paragraph_text
def heuristic(a, b):
    return (b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2

def a_star(arr, start, goal):
    closed_set = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: heuristic(start, goal)}
    oheap = []
    heappush(oheap, (fscore[start], start))
    while oheap:
        curr = heappop(oheap)[1]
        if curr == goal:
            data = []
            while curr in came_from:
                data.append(curr)
                curr = came_from[curr]
            return data
        closed_set.add(curr)
        for i, j in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
            neighbor = curr[0] + i, curr[1] + j
            temp_g_score = gscore[curr] + heuristic(curr, neighbor)
            if 0 <= neighbor[0] < arr.shape[0]:
                if 0 <= neighbor[1] < arr.shape[1]:
                    if arr[neighbor[0]][neighbor[1]] == 1:
                        continue
                else:
                    continue
            else:
                continue
            if neighbor in closed_set and temp_g_score >= gscore.get(neighbor, 0):
                continue
            if temp_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1] for i in oheap]:
                came_from[neighbor] = curr
                gscore[neighbor] = temp_g_score
                fscore[neighbor] = temp_g_score + heuristic(neighbor, goal)
                heappush(oheap, (fscore[neighbor], neighbor))

    return []

def sobel_edge_detector(img):
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    grad = np.sqrt(grad_x ** 2 + grad_y ** 2)
    grad_norm = (grad * 255 / grad.max()).astype(np.uint8)
    return grad_norm

def h_projection(sobel_image):
    row_sums = []
    for row in range(sobel_image.shape[0] - 1):
        row_sums.append(np.sum(sobel_image[row, :]))
    return row_sums

def v_projection(img):
    return np.sum(img, axis=0)

def find_peaks(hp, threshold):
    peaks = []
    peak_indices = []
    for i, hpv in enumerate(hp):
        if hpv < threshold:
            peaks.append([i, hpv])
    peak_indices = np.array(peaks)[:, 0].astype(int)
    return peaks, peak_indices

def process_lines(img):
    sobel_image = sobel_edge_detector(img)

    hp = h_projection(sobel_image)

    divider = 2
    threshold = (np.max(hp) - np.min(hp)) / divider
    _, peak_indices = find_peaks(hp, threshold)

    diffs = np.diff(peak_indices)

    large_diffs = np.where(diffs > 1)[0].flatten()
    peak_grps = np.split(peak_indices, large_diffs)
    peak_grps = [item for item in peak_grps if len(item) > 10]

    (_, binary_img) = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    sep_lines = []
    for i, sub_image_i in enumerate(peak_grps):
        nmap = binary_img[sub_image_i[0]:sub_image_i[-1]]
        path = np.array(a_star(nmap, (int(nmap.shape[0] / 2), 0), (int(nmap.shape[0] / 2), nmap.shape[1] - 1)))
        offset_top = sub_image_i[0]
        path[:, 0] += offset_top
        sep_lines.append(path)

    line_imgs = []
    for i, line_segments in enumerate(sep_lines):
        if i < len(sep_lines) - 1:
            lower_line = np.min(sep_lines[i][:, 0])
            upper_line = np.max(sep_lines[i + 1][:, 0])
            line_imgs.append(img[lower_line:upper_line])

    return line_imgs

def find_whitespaces(vertical_projection, height):
    whitespace_lengths = []
    whitespace = 0
    for vp in vertical_projection:
        if vp == height:
            whitespace += 1
        elif vp != height:
            if whitespace != 0:
                whitespace_lengths.append(whitespace)
            whitespace = 0
    return whitespace_lengths

def avg_whitespace_length(whitespace_lengths):
    return np.mean(whitespace_lengths)

def get_dividers(vertical_projection, height, avg_white_space_length):
    whitespace_length = 0
    divider_indices = []
    for index, vp in enumerate(vertical_projection):
        if vp == height:
            whitespace_length = whitespace_length + 1
        elif vp != height:
            if whitespace_length != 0 and whitespace_length > avg_white_space_length:
                divider_indices.append(index - int(whitespace_length / 2))
            whitespace_length = 0
    return divider_indices
def avg_whitespace_length(whitespace_lengths):
    return np.mean(whitespace_lengths)

def get_dividers(vertical_projection, height, avg_white_space_length):
    whitespace_length = 0
    divider_indices = []
    for index, vp in enumerate(vertical_projection):
        if vp == height:
            whitespace_length = whitespace_length + 1
        elif vp != height:
            if whitespace_length != 0 and whitespace_length > avg_white_space_length:
                divider_indices.append(index - int(whitespace_length / 2))
            whitespace_length = 0
    return divider_indices

def process_words(line_imgs):
    word_dict = defaultdict(list)
    for i, line in enumerate(line_imgs):
        (_, binary_img) = cv2.threshold(line, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        binary_img = cv2.bitwise_not(binary_img)
        vertical_projection = v_projection(binary_img)
        height = max(vertical_projection)
        whitespace_lengths = find_whitespaces(vertical_projection, height)
        avg_white_space_length = avg_whitespace_length(whitespace_lengths)
        divider_indices = get_dividers(vertical_projection, height, avg_white_space_length)
        divider_indices.append(line.shape[1])
        divider_indices = np.array(divider_indices)
        dividers = np.column_stack((divider_indices[:-1], divider_indices[1:]))
        for index, window in enumerate(dividers):
            word_dict[i].append(line[:, window[0]:window[1]])
    return word_dict

def segment_words(image_path):
    img = np.asarray(im.open(image_path).convert('L'))
    line_imgs = process_lines(img)
    words = process_words(line_imgs)
    return words

image_path = 'hwtest2.jpeg'
output_path = 'C:\\Users\\gudug\\PycharmProjects\\hw'
words = segment_words(image_path)

word_paths = []
for line_idx, word_list in words.items():
    # print(f"Line {line_idx + 1}:")
    for word_idx, word in enumerate(word_list):

        word_path = f"word_{line_idx}_{word_idx}.jpg"
        filepath = os.path.join(output_path, word_path)
        cv2.imwrite(filepath, word)
        word_paths.append(word_path)

        cv2.imshow(f"Word", word)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

print(word_paths)
words=imgcrop.crop(word_paths)

from mltu.configs import BaseModelConfigs

configs = BaseModelConfigs.load("Models\\03_handwriting_recognition\\202306251138\\configs.yaml")

model = ImageToParagraphModel(model_path=configs.model_path)
    # print(model.input_shape)

image_path = "C:\\Users\\gudug\\PycharmProjects\\hw\\hwtest2.jpeg"

prediction_text = model.predict_paragraph(word_paths)
print("predicted text:"+prediction_text)