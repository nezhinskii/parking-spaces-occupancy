import re
import matplotlib.path as mplPath
import cv2
import numpy as np

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
    LINE_THIKNESS = 3
    FREE_COLOR = (0, 255, 0)
    BUSY_COLOR = (0, 0, 255)
    DEFAULT_COLOR = (255,255,255)
    BB_COLOR = (255, 0, 255)
    CONFIDENCE_TRESHOLD = 0.5

    def __init__(self, model):
        self.model = model

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
            cv2.polylines(self.spaces_mask, [np.column_stack((place.x_coords, place.y_coords))], True, self.DEFAULT_COLOR, self.LINE_THIKNESS)
        self.preview = cv2.addWeighted(self.image_source, 1, self.spaces_mask, 1, 0)

    def __place_busy(sself, place, bb):
        return place.contains_point(bb['xmin'] + (bb['xmax'] - bb['xmin']) / 2, bb['ymin'] + (bb['ymax'] - bb['ymin']) * 3/4)
    
    def process_frame(self):
        results = self.model(self.image_source).pandas().xyxy[0]
        result_mask = np.zeros((self.image_source.shape[0], self.image_source.shape[1], 3), dtype=np.uint8)
        for place in self.parking_places:
            flag = False
            for _, bb in results.iterrows():
                if bb['confidence'] > self.CONFIDENCE_TRESHOLD and self.__place_busy(place, bb):
                    flag = True
                    break
            if flag:
                color = self.BUSY_COLOR
            else:
                color = self.FREE_COLOR
            cv2.polylines(result_mask, [np.column_stack((place.x_coords, place.y_coords))], True, color, self.LINE_THIKNESS)

        for _, bb in results.iterrows():
            if bb['confidence'] > self.CONFIDENCE_TRESHOLD:
                cv2.rectangle(result_mask, (int(bb['xmin']), int(bb['ymin'])), (int(bb['xmax']), int(bb['ymax'])), self.BB_COLOR, self.LINE_THIKNESS)

        return cv2.addWeighted(self.image_source, 1, result_mask, 1, 0.0)