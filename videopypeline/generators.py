"""

"""

import cv2

from videopypeline import core


def iterate(iterable):
    for item in iterable:
        yield item


def endless_value(c):
    while True:
        yield c


def read_video_file(filepath):
    cap = cv2.VideoCapture(filepath)
    assert cap.isOpened(), filepath

    while cap.grab():
        yield cap.retrieve()[1]  # BGR-format

    cap.release()


class Value(core.Generator):
    def __init__(self, value, **kwargs):
        super().__init__(lambda: iterate([value]), **kwargs)


class Iteration(core.Generator):
    def __init__(self, iterable, **kwargs):
        super().__init__(lambda: iterate(iterable), **kwargs)


class EndlessValue(core.Generator):
    def __init__(self, c, **kwargs):
        super().__init__(lambda: endless_value(c), **kwargs)


class ReadVideoFile(core.Generator):
    def __init__(self, filepath, **kwargs):
        super().__init__(lambda: read_video_file(filepath), **kwargs)
