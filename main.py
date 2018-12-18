import cv2
import argparse

import torch
import torchvision.transforms as transforms

from network.networks import define_G
from network.utils import tensor2im

from get_face.haar.face_haar import GetFaceHaar
from get_face.utils.utils import roi_image_resize


class FaceSwapper:
    def __init__(self, net_path, device, eval_size, gpu_ids=None):
        if gpu_ids is None:
            gpu_ids = []
        # Define image size
        self.image_size = eval_size
        # Define the network (we used channel images, 64 filters, and resnet_9blocks)
        self.net = define_G(3, 3, 64, 'resnet_9blocks', norm='instance', use_dropout=False, init_type='normal',
                            init_gain=0.02, gpu_ids=gpu_ids)
        # Load weights
        state_dict = torch.load(net_path, map_location=device)
        self.net.load_state_dict(state_dict)
        # Configure to evaluate mode
        self.net.eval()

    def evaluate(self, tensor_image):
        tensor_image = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(tensor_image)
        assert tensor_image.size() == torch.Size([3, self.image_size, self.image_size])
        return self.net(tensor_image.unsqueeze(0))


class Transformer:
    """
    Transformer class: Takes a cv input image and transforms it. This wraps up
    the selector, preprocessor, evaluator, and postprocessor.
    """
    def __init__(self, net_path, device, eval_size):
        # Define the detector
        self.detector = GetFaceHaar()
        # Define the model
        self.model = FaceSwapper(net_path, device, eval_size)

    def __call__(self, cv_image):
        # Detect ROIS
        rois = self.detector(cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY))

        # Apply transformation to each ROI
        for (x, y, w, h) in rois:
            img_dim = self.model.image_size
            roi_dimensions, roi_image = roi_image_resize(cv_image, x, y, w, h,
                                                         1.8, img_dim, img_dim)
            # Convert to RGB tensor
            rgb = cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB)
            tensor_input = transforms.ToTensor()(rgb)

            # Generate
            tensor_output = self.model.evaluate(tensor_input)

            # Convert to image
            bgr = cv2.cvtColor(tensor2im(tensor_output), cv2.COLOR_RGB2BGR)

            # Find the edges and slice them off if required
            x_, y_, w_, h_ = roi_dimensions
            xc = 0 if x_ < 0 else x_
            yc = 0 if y_ < 0 else y_
            wn = x_ + w_ - xc
            hn = y_ + h_ - yc
            wc = cv_image.shape[1] - xc if xc + wn > cv_image.shape[1] else wn
            hc = cv_image.shape[0] - yc if yc + hn > cv_image.shape[0] else hn

            # Paste the generated result in the original image
            cv_image[yc:yc + hc, xc:xc + wc] = cv2.resize(bgr, (h_, w_))[yc - y_:yc - y_ + hc, xc - x_:xc - x_ + wc]

        return cv_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--net_path", required=True, type=str, help="Path to network")
    parser.add_argument("-e", "--eval_size", type=int, default=128, help="Image size for evaluating model")
    parser.add_argument("-d", "--device", default='cpu', type=str, help="Device to evaluate network",
                        choices=["cpu", "gpu"])
    parser.add_argument("-g", "--gpu_ids", type=str, default='-1', help="gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU")
    parser.add_argument("-s", "--scale", type=int, default=1, help="How much to scale shown frame")
    parser.add_argument("-c", "--camera", type=int, default=0, help="Which camera ID")
    args = parser.parse_args()

    transformer = Transformer(args.net_path, args.device, args.eval_size)
    camera = cv2.VideoCapture(args.camera)
    while True:
        (grabbed, frame) = camera.read()
        if not grabbed:
            break

        # Apply transformations
        frame = transformer(frame)

        # Show the image
        width, height = frame.shape[:2]
        frame = cv2.resize(frame, (height*args.scale, width*args.scale))
        cv2.imshow("images", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


if __name__ == '__main__':
    main()
