import re
import threading
import matplotlib.path as mplPath
import cv2
import numpy as np
import requests
import json
import websockets
import websocket
import asyncio
import streamlit as st
import widgets.shared as shared

class ParkingPlace:

    def __init__(self, x_coords, y_coords):
        self.x_coords = x_coords
        self.y_coords = y_coords

    def __repr__(self):
        return str(self.x_coords) + str(self.y_coords)

    def contains_point(self, x, y):
        bbPath = mplPath.Path([point for point in zip(self.x_coords, self.y_coords)])
        return bbPath.contains_point((x, y))
    
    def to_json(self):
        return {"points_x":self.x_coords, "points_y":self.y_coords}

class ParkingOccupancy:
    LINE_THIKNESS = 2
    FREE_COLOR = (0, 255, 0)
    BUSY_COLOR = (0, 0, 255)
    DEFAULT_COLOR = (255,255,255)
    BB_COLOR = (255, 0, 255)
    CONFIDENCE_TRESHOLD = 0.5

    def __init__(self, backend_url):
        self.parking_places = None
        self.backend_url = backend_url

    def load_spaces_annotation_csv(self, labels):
        self.parking_places = {}
        key = 0
        for _, row in labels.iterrows():
            coords = re.findall(r'(?<=\[)(?:\d+,? ?)+(?=\])',row[5])
            if len(coords) == 2:
                self.parking_places[str(key)] = ParkingPlace([int(x) for x in coords[0].split(',')], [int(y) for y in coords[1].split(',')])
                key += 1

    def load_spaces_annotation_json(self, labels, scale_factor):
        self.parking_places = {}
        key = 0
        for place in [l['path'] for l in labels if l['type'] == 'path']:
            if len(place) >= 3:
                self.parking_places[str(key)] = ParkingPlace([int(point[1] / scale_factor) for point in place if len(point) == 3], [int(point[2] / scale_factor)  for point in place if len(point) == 3])
                key += 1

    def load_image_source(self, image_bytes):
        # if image.shape[2] == 3:
        #     image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
        self.image_bytes = image_bytes
        self.opencv_image = cv2.imdecode(image_bytes, 1)

    def has_annotation(self):
        return self.parking_places is not None and len(self.parking_places) > 0
    
    def create_prewiew(self):
        self.spaces_mask = np.zeros((self.opencv_image.shape[0], self.opencv_image.shape[1], 3), dtype=np.uint8)
        for key in self.parking_places:
            place = self.parking_places[key]
            cv2.polylines(self.spaces_mask, [np.column_stack((place.x_coords, place.y_coords))], True, self.DEFAULT_COLOR, self.LINE_THIKNESS)
        self.preview = cv2.addWeighted(self.opencv_image, 1, self.spaces_mask, 1, 0)
    
    def build_parking_spaces_json(self):
        results = {}
        print(self.parking_places)
        for key in self.parking_places:
            place = self.parking_places[key]
            results[key] = place.to_json()
        return results

    def get_image_prediction(self):
        parking_spaces_json = self.build_parking_spaces_json()
        response = requests.post(
            f'{self.backend_url}/predict_image', 
            files={
                "image": ("image.jpg", self.image_bytes, "image/jpeg"),
            },
            data={
                'parking_spaces': json.dumps(parking_spaces_json),
                'confidance': self.CONFIDENCE_TRESHOLD
            }
        )
        return response.json()

    def process_frame(self):
        results = self.get_image_prediction()
        result_mask = np.zeros((self.opencv_image.shape[0], self.opencv_image.shape[1], 3), dtype=np.uint8)
        for key in self.parking_places:
            place = self.parking_places[key]
            if results[key]:
                color = self.BUSY_COLOR
            else:
                color = self.FREE_COLOR
            cv2.polylines(result_mask, [np.column_stack((place.x_coords, place.y_coords))], True, color, self.LINE_THIKNESS)
        return cv2.addWeighted(self.opencv_image, 1, result_mask, 1, 0.0)

    def process_stream(self, stream_url):
        def on_message(ws, message):
            print(message)
            results = json.loads(message)
            if shared.stream_mask is None:
                return
            new_mask = np.zeros((shared.stream_mask.shape[0], shared.stream_mask.shape[1], 3), dtype=np.uint8)
            print(self.parking_places)
            for key in self.parking_places:
                place = self.parking_places[key]
                if results[key]:
                    color = self.BUSY_COLOR
                else:
                    color = self.FREE_COLOR
                cv2.polylines(new_mask, [np.column_stack((place.x_coords, place.y_coords))], True, color, self.LINE_THIKNESS)
            shared.stream_mask = new_mask

        def on_error(ws, error):
            print("Error:", error)

        def on_close(ws, _, __):
            print("WebSocket closed")

        def on_open(ws):
            print('Connected')
            ws.send(json.dumps({
                "stream_url": stream_url,
                "confidence": self.CONFIDENCE_TRESHOLD,
                "parking_spaces": self.build_parking_spaces_json()
            }))
            print("WebSocket opened")

        ws = websocket.WebSocketApp(f'{self.backend_url}/predict_stream'.replace('http','ws'), 
                                    on_message=on_message, 
                                    on_error=on_error, 
                                    on_close=on_close)
        shared.ws = ws
        ws.close()
        ws.on_open = on_open
        ws.run_forever()
