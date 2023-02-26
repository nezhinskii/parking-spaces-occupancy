import csv
import re
import matplotlib.path as mplPath
import torch
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

class ParkingPlace:

    def __init__(self, x_coords, y_coords):
        self.x_coords = x_coords
        self.y_coords = y_coords

    def __repr__(self):
        return str(self.x_coords) + str(self.y_coords)

    def contains_point(self, x, y):
        bbPath = mplPath.Path([point for point in zip(self.x_coords, self.y_coords)])
        return bbPath.contains_point((x, y))

class ParkingOccupancy:

    def __init__(self):
        model = torch.hub.load('ultralytics/yolov5', 'custom', path='../weights/coco-voc.pt', verbose = False, force_reload=True)

    def load_spaces_annotation(self, labels):
        self.parking_places = []
        for _, row in labels.iterrows():
            coords = re.findall(r'(?<=\[)(?:\d+,?)+(?=\])',row[5])
            if len(coords) == 2:
                self.parking_places.append(ParkingPlace([int(x) for x in coords[0].split(',')], [int(y) for y in coords[1].split(',')]))

    def load_image_source(self, image):
        # if image.shape[2] == 3:
        #     image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
        self.image_source = image

    def has_annotation(self):
        return self.parking_places is not None and len(self.parking_places) > 0
    
    def create_prewiew(self):
        self.spaces_mask = np.zeros((self.image_source.shape[0], self.image_source.shape[1], 3), dtype=np.uint8)
        for place in self.parking_places:
            color = (255,255,255)
            cv2.polylines(self.spaces_mask, [np.column_stack((place.x_coords, place.y_coords))], True, color, 2)
        self.preview = cv2.addWeighted(self.image_source, 1, self.spaces_mask, 1, 0.6)

