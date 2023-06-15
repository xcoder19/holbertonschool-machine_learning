#!/usr/bin/env python3
"""yolo class"""
import tensorflow.keras as K


class Yolo:
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        self.model = K.models.load_model(model_path)
        self.class_names = self.load_classes(classes_path)
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    @staticmethod
    def load_classes(classes_path):
        """load classes"""
        with open(classes_path, 'r') as file:
            class_names = file.readlines()
        class_names = [name.strip() for name in class_names]
        return class_names
