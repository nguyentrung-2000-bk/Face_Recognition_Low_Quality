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
        # self.arcface_dst = np.array(
        #     [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]],
        #     dtype=np.float32)

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

    # def estimate_norm(self, lmk, image_size=112):
    #     assert lmk.shape == (5, 2)
    #
    #     ratio = float(image_size) / 112.0
    #     diff_x = 0
    #
    #     dst = self.arcface_dst * ratio
    #     dst[:, 0] += diff_x
    #     tform = trans.SimilarityTransform()
    #     tform.estimate(lmk, dst)
    #     M = tform.params[0:2, :]
    #     return M
    #
    # def norm_crop(self, img, landmark, image_size=112):
    #     M = self.estimate_norm(landmark, image_size)
    #     warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    #     return warped
    #
    # def alignment(self, image):
    #     faces = self.prepareFaces(image)
    #     faces_align = []
    #     for i in range(len(faces)):
    #         align = self.norm_crop(cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB), faces[i].kps)
    #         faces_align.append(align)
    #     return faces_align
