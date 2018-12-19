import cv2
import argparse

import torch
import torchvision.transforms as transforms

from network.networks import get_norm_layer
from network.networks import ResnetGenerator
from network.utils import tensor2im

from get_face.haar.face_haar import GetFaceHaar
from get_face.utils.utils import roi_image_resize


class FaceSwapper:
    def __init__(self, net_path, eval_size, gpu_id=-1,
                 input_nc=3, output_nc=3, ngf=64, norm='instance', no_dropout=True):
        # Define image size
        self.image_size = eval_size
        # Define the network
        self.net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=get_norm_layer(norm_type=norm),
                                   use_dropout=not no_dropout, n_blocks=9, padding_type='reflect')
        # Load weights
        state_dict = torch.load(net_path)
        self.net.load_state_dict(state_dict)
        # Configure for device
        if gpu_id != -1:
            assert(torch.cuda.is_available())
            self.net.to(gpu_id)
            self.net = torch.nn.DataParallel(self.net, [gpu_id])
        # Configure to evaluate mode
        self.net.eval()

    def evaluate(self, cv_image):
        tensor_input = transforms.ToTensor()(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        assert tensor_input.size() == torch.Size([3, self.image_size, self.image_size])

        # Profiling
        tensor_output = self.net(tensor_input.unsqueeze(0))

        return cv2.cvtColor(tensor2im(tensor_output), cv2.COLOR_RGB2BGR)


class Transformer:
    """
    Transformer class: Takes a cv input image and transforms it. This wraps up
    the selector, preprocessor, evaluator, and postprocessor.
    """
    def __init__(self, net_path, eval_size, gpu_id=-1,
                 input_nc=3, output_nc=3, ngf=64, norm='instance', no_dropout=True):
        # Define the detector
        self.detector = GetFaceHaar()
        # Define the model
        self.model = FaceSwapper(net_path, eval_size, gpu_id=gpu_id,
                                 input_nc=input_nc, output_nc=output_nc, ngf=ngf, norm=norm, no_dropout=no_dropout)

    def __call__(self, cv_image):
        # Detect ROIS
        rois = self.detector(cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY))

        # Apply transformation to each ROI
        for (x, y, w, h) in rois:
            img_dim = self.model.image_size
            roi_dimensions, roi_image = roi_image_resize(cv_image, x, y, w, h,
                                                         1.8, img_dim, img_dim)
            # Generate
            out = self.model.evaluate(roi_image)

            # Find the edges and slice them off if required
            x_, y_, w_, h_ = roi_dimensions
            xc = 0 if x_ < 0 else x_
            yc = 0 if y_ < 0 else y_
            wn = x_ + w_ - xc
            hn = y_ + h_ - yc
            wc = cv_image.shape[1] - xc if xc + wn > cv_image.shape[1] else wn
            hc = cv_image.shape[0] - yc if yc + hn > cv_image.shape[0] else hn

            # Paste the generated result in the original image
            cv_image[yc:yc + hc, xc:xc + wc] = cv2.resize(out, (h_, w_))[yc - y_:yc - y_ + hc, xc - x_:xc - x_ + wc]

        return cv_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--net_path", required=True, type=str, help="Path to network")
    parser.add_argument("-e", "--eval_size", type=int, default=128, help="Image size for evaluating model")
    parser.add_argument("-g", "--gpu_id", type=int, default=-1, help="gpu id used when device is 'gpu'")
    parser.add_argument("-s", "--scale", type=int, default=1, help="How much to scale shown frame")
    parser.add_argument("-c", "--camera", type=int, default=0, help="Which camera ID")

    parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
    parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
    parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
    parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')
    # parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
    # in the original CycleGAN there is a bug where `no_dropout` is always True

    args = parser.parse_args()

    transformer = Transformer(args.net_path, args.eval_size, args.gpu_id,
                              args.input_nc, args.output_nc, args.ngf, args.norm)
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
