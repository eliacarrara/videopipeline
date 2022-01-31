from .. import core
import cv2
import numpy as np


def read_video_file(filepath):
    cap = cv2.VideoCapture(filepath)
    assert cap.isOpened(), filepath

    while cap.grab():
        yield cap.retrieve()[1]

    cap.release()


def read_movement_data_file(filepath):
    with open(filepath) as f:
        movement_data = np.array([[int(e) for e in line.strip().split(';')] for line in f.readlines()])
        length = movement_data[:, 0].max() + 1

    for i in range(length):
        mask = movement_data[:, 0] == i
        yield movement_data[mask, 1:]
        movement_data = movement_data[np.logical_not(mask), :]


def flatten(iterable, zipped=False):
    if zipped:
        for item in zip(iterable):
            yield item
    else:
        for item in iterable:
            yield item


class ReadMovementData(core.Generator):

    def __init__(self, **kwargs):
        super().__init__(read_movement_data_file, **kwargs)


class ReadVideoFile(core.Generator):

    def __init__(self, filepath, **kwargs):
        super().__init__(lambda *args: read_video_file(filepath), **kwargs)


class Flatten(core.Generator):

    def __init__(self, zipped=False, **kwargs):
        super().__init__(lambda itr: flatten(itr, zipped), **kwargs)
