import csv
import re
import matplotlib.path as mplPath
import torch
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

model = torch.hub.load('ultralytics/yolov5', 'yolov5m', verbose = False)

class ParkingPlace:
    def __init__(self, x_coords, y_coords):
        self.x_coords = x_coords
        self.y_coords = y_coords

    def __repr__(self):
        return str(self.x_coords) + str(self.y_coords)

    def contains_point(self, x, y):
        bbPath = mplPath.Path([point for point in zip(self.x_coords, self.y_coords)])
        return bbPath.contains_point((x, y))

def frame_processing(parking_places, image):
    results = model(image).pandas().xyxy[0].values

    for place in parking_places:
        flag = False
        for bb in results:
            if (place.contains_point(bb[0] + (bb[2] - bb[0]) / 2, bb[1] + (bb[3] - bb[1]) * 3/4)):
                flag = True
                break
        if flag:
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)
        cv2.polylines(image, [np.column_stack((place.x_coords, place.y_coords))], True, color, 1)

    for bb in results:
        if bb[5] == 2 or bb[5] == 7:
            cv2.rectangle(image, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), (255, 0, 255), 3)
    
    return image

def video_processing(parking_places, video_path):
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    frames_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    otput_path = result_path(video_path)
    output = cv2.VideoWriter(otput_path, cv2.VideoWriter_fourcc(*"XVID"), fps, frame_size)
    with tqdm(total = frames_count) as pbar:
        while(True):
            pbar.update(1)
            ret, frame = video.read()
            if(ret):
                frame_processing(parking_places, frame)
                output.write(frame)
            else:
                break
    output.release()
    video.release()
    return otput_path

def result_path(path):
    path = Path(path)
    return "{0}/{1}_processed{2}".format(path.parent, path.stem, path.suffix)

def analyze_parking_spaces(labels, source):
    parking_places = []
    with open(labels) as file:
        labels = csv.reader(file, delimiter = ",")
        for row in labels:
            coords = re.findall(r'(?<=\[)(?:\d+,?)+(?=\])',row[5])
            if len(coords) == 2:
                parking_places.append(ParkingPlace([int(x) for x in coords[0].split(',')], [int(y) for y in coords[1].split(',')]))
    
    images_paths = []
    if isinstance(source, list):
        images_paths = source
    else:
        if (source.endswith(('.mp4'))):
            return video_processing(parking_places, source)
        source = Path(source)
        if source.is_dir():
            images_paths = [str(path) for path in source.glob('**/*.*')]
            images_paths = [impath for impath in images_paths if impath.endswith(('.jpg', '.png', '.jpeg'))]
        else:
            images_paths = [str(source)]

    results = []
    for image_path in images_paths:
        result = frame_processing(parking_places, cv2.imread(image_path))
        results.append(result)
        cv2.imwrite(result_path(image_path), result)
        
    if len(results) == 1:
        return results[0]
    return results