"""

"""

import cv2

from videopypeline import core


def flatten(iterable):
    for item in iterable:
        yield item


def read_video_file(filepath):
    cap = cv2.VideoCapture(filepath)
    assert cap.isOpened(), filepath

    while cap.grab():
        yield cap.retrieve()[1]  # BGR-format

    cap.release()


def constant_generator(c):
    while True:
        yield c


class Flatten(core.Generator):
    def __init__(self, iterable, **kwargs):
        super().__init__(lambda: flatten(iterable), **kwargs)


class ReadVideoFile(core.Generator):
    def __init__(self, filepath, **kwargs):
        super().__init__(lambda: read_video_file(filepath), **kwargs)


class Constant(core.Generator):
    def __init__(self, c, **kwargs):
        super().__init__(lambda: constant_generator(c), **kwargs)
