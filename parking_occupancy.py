import csv
import re
import matplotlib.path as mplPath
import torch
import cv2
import numpy as np
from pathlib import Path

class ParkingPlace:
    def __init__(self, x_coords, y_coords):
        self.x_coords = x_coords
        self.y_coords = y_coords

    def __repr__(self):
        return str(self.x_coords) + str(self.y_coords)

    def contains_point(self, x, y):
        bbPath = mplPath.Path([point for point in zip(self.x_coords, self.y_coords)])
        return bbPath.contains_point((x, y))

def image_processing(parking_places, image_path):
    image = cv2.imread(image_path)

    model = torch.hub.load('ultralytics/yolov5', 'yolov5m')
    results = model(image_path).pandas().xyxy[0].values

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
        image = cv2.polylines(image, [np.column_stack((place.x_coords, place.y_coords))], True, color, 2)

    for bb in results:
        if bb[5] == 2 or bb[5] == 7:
            image = cv2.rectangle(image, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), (255, 0, 255), 5)
    
    path = Path(image_path)
    cv2.imwrite("{0}/{1}_processed{2}".format(path.parent, path.stem, path.suffix), image)
    return image

def analyze_parking_spaces(labels, source):
    parking_places = []
    with open(labels) as file:
        labels = csv.reader(file, delimiter = ",")
        for row in labels:
            coords = re.findall(r'(?<=\[)(?:\d+,?)+(?=\])',row[5])
            if len(coords) == 2:
                parking_places.append(ParkingPlace([int(x) for x in coords[0].split(',')], [int(y) for y in coords[1].split(',')]))
    
    images = []
    if isinstance(source, list):
        images = source
    else:
        source = Path(source)
        if source.is_dir():
            images = [str(image) for image in source.glob('**/*.*')]
            images = [image for image in images if image.endswith(('.jpg', '.png', '.jpeg'))]
        else:
            images = [str(source)]

    results = []
    for image in images:
        results.append(image_processing(parking_places, image))
        
    if len(results) == 1:
        return results[0]
    return results