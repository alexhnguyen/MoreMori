import cv2
from PIL.Image import fromarray

from evaluate_model.utils.base_dataset import get_transform
from evaluate_model.models import create_model
from evaluate_model.utils.eval_options import EvalOptions
from evaluate_model.utils.util import tensor2im

from get_face.haar.face_haar import GetFaceHaar
from get_face.deeplearning.face_dl import GetFaceDL
from get_face.utils.utils import roi_image_resize


class Evaluator:

    model = None
    transformer = None

    def __init__(self, opt):
        opt.no_flip = True
        self.model = create_model(opt)
        self.model.setup(opt)
        self.transformer = get_transform(opt)

    def evaluate(self, img_cv):
        img_t = self.__convert_cv2_to_tensor(img_cv)
        img_t = self.model.netG_A(img_t)
        return self.__convert_tensor_to_cv2(img_t)

    @staticmethod
    def replace(eval_img, frame, roi_dimensions):
        # Find the edges and slice them off if required
        x, y, w, h = roi_dimensions
        xc = 0 if x < 0 else x
        yc = 0 if y < 0 else y
        wn = x + w - xc
        hn = y + h - yc
        wc = frame.shape[1] - xc if xc + wn > frame.shape[1] else wn
        hc = frame.shape[0] - yc if yc + hn > frame.shape[0] else hn

        frame[yc:yc + hc, xc:xc + wc] = cv2.resize(eval_img, (h, w))[yc - y:yc - y + hc, xc - x:xc - x + wc]
        return frame

    def __convert_cv2_to_tensor(self, img_cv2_):
        img_cv = cv2.cvtColor(img_cv2_, cv2.COLOR_BGR2RGB)
        img_t = self.transformer(fromarray(img_cv))
        return img_t.unsqueeze(0)

    @staticmethod
    def __convert_tensor_to_cv2(img_t_):
        return cv2.cvtColor(tensor2im(img_t_), cv2.COLOR_RGB2BGR)


def main():

    opt = EvalOptions().parse()
    if opt.detector == 'haar':
        detector = GetFaceHaar()
    elif opt.detector == 'deeplearning':
        detector = GetFaceDL()
    else:
        raise ValueError("Invalid detector %s" % opt.detector)

    model = Evaluator(opt)
    camera = cv2.VideoCapture(opt.camera)
    while True:
        (grabbed, frame) = camera.read()
        if not grabbed:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rois = detector(gray)

        scale_factor = 1.8
        for (x, y, w, h) in rois:
            roi_dimensions, roi_image = roi_image_resize(frame, x, y, w, h,
                                                         scale_factor, opt.fineSize, opt.fineSize)
            eval_img = model.evaluate(roi_image)
            frame = model.replace(eval_img, frame, roi_dimensions)

        # Show the image
        width, height = frame.shape[:2]
        show_scale = 2
        frame = cv2.resize(frame, (show_scale*height, show_scale*width))
        cv2.imshow("images", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


if __name__ == '__main__':
    main()
