import csv
import re
import matplotlib.path as mplPath
import torch
from PIL import Image, ImageDraw
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
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image, 'RGBA') 

    model = torch.hub.load('ultralytics/yolov5', 'yolov5m')
    results = model(image_path).pandas().xyxy[0].values
    
    for place in parking_places:
        flag = False
        for bb in results:
            if (place.contains_point(bb[0] + (bb[2] - bb[0]) / 2, bb[1] + (bb[3] - bb[1]) * 3/4)):
                flag = True
                break
        if flag:
            draw.polygon(xy = [point for point in zip(place.x_coords, place.y_coords)], fill = (255, 0, 0, 40), outline = (255, 0, 0), width = 2)
        else:
            draw.polygon(xy = [point for point in zip(place.x_coords, place.y_coords)], fill = (0, 255, 0, 40), outline = (0, 255, 0), width = 2)

    for bb in results:
        if bb[5] == 2 or bb[5] == 7:
            draw.rectangle(xy = [(bb[0], bb[1]), (bb[2], bb[3])], outline = (255, 255, 0), width = 3)
    
    path = Path(image_path)   
    image.save("{0}/{1}_processed{2}".format(path.parent, path.stem, path.suffix))
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
            images = [str(image) for image in source.glob('**/*.jpg')]
        else:
            images = [str(source)]

    for image in images:
        image_processing(parking_places, image)


analyze_parking_spaces(labels = 'examples/several_images/parking_places.csv', 
    source = ['examples/several_images/2015-12-18_0748.jpg', 'examples/several_images/2015-12-18_0948.jpg'])
