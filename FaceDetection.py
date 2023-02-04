from insightface.app import FaceAnalysis
import numpy as np
from PIL import ImageDraw, Image


class FaceDetection:
    def __init__(self):
        super(FaceDetection, self).__init__()
        # load model Face Detection (SCRFD - Lib: Insightface)
        app = FaceAnalysis(allowed_modules=['detection'])
        app.prepare(ctx_id=0, det_size=(640, 640))
        self.app = app

    def prepareFaces(self, image):
        faces = self.app.get(np.array(image))
        return faces

    def detection(self, image):
        faces = self.prepareFaces(image)
        faces_crop = []
        for i in range(len(faces)):
            face = image.crop(faces[i].bbox).resize((112, 112))
            faces_crop.append(face)
        return faces_crop

    def detection2(self, image):
        draw = ImageDraw.Draw(image)
        faces = self.prepareFaces(image)
        for i in range(len(faces)):
            draw.rectangle([(faces[i].bbox[0], faces[i].bbox[1]), (faces[i].bbox[2], faces[i].bbox[3])],
                           outline='green', width=3)
        del draw
        return image
