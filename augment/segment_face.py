import os
import argparse

import numpy as np
import cv2
from PIL import Image

import caffe


def get_files_in_dir(path):
    folder = os.fsencode(path)
    filenames = []
    for file in os.listdir(folder):
        filename = os.fsdecode(file)
        if filename.endswith(('jpg', '.jpeg', '.png', '.gif')):
            filenames.append('{bg}/{f}'.format(bg=path, f=filename))
    return filenames


def get_segment_model():
    caffe.set_device(0)
    caffe.set_mode_gpu()
    path1 = 'face_seg_fcn8s/face_seg_fcn8s_deploy.prototxt'
    path2 = 'face_seg_fcn8s/face_seg_fcn8s.caffemodel'
    return caffe.Net(path1, path2, caffe.TEST)


def get_face(segment_model, file, convex_fill=False):
    image, bool_mask = _get_mask(segment_model, file, convex_fill=convex_fill)
    return _apply_mask(image, bool_mask)


def _apply_mask(image, bool_mask):
    return np.array(np.multiply(image, bool_mask), dtype=np.uint8)


def _get_mask(segment_model, file_path, convex_fill=False):
    im = Image.open(file_path)
    width, height = im.size
    im = im.resize((500, 500))
    in_ = np.array(im, dtype=np.float32)
    in_ = in_[:, :, ::-1]

    # subtract mean
    in_ -= np.array((104.00698793, 116.66876762, 122.67891434))
    in_ = in_.transpose((2, 0, 1))

    # shape for input (data blob is N x C x H x W), set data
    segment_model.blobs['data'].reshape(1, *in_.shape)
    segment_model.blobs['data'].data[...] = in_

    # run net and take argmax for prediction
    segment_model.forward()
    mask = segment_model.blobs['score'].data[0].argmax(axis=0)
    mask = (mask - np.min(mask))/np.ptp(mask)
    mask = (mask*255).astype(np.uint8)

    if convex_fill:
        mask = __apply_convex_fill(mask)

    mask = cv2.cvtColor(cv2.resize(mask, (width, height)), cv2.COLOR_GRAY2BGR)

    bool_mask = (mask != 0)
    return cv2.imread(file_path), bool_mask


def __apply_convex_fill(image, kernel_size=8):
    border_size = kernel_size * 2
    image_ = cv2.copyMakeBorder(image, top=border_size, bottom=border_size, left=border_size, right=border_size,
                                borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
    kernel = np.ones((kernel_size, kernel_size))
    image_ = cv2.dilate(image_, kernel, iterations=1)
    image_ = __column_fill(image_)
    image_ = __row_fill(image_)
    image_ = cv2.erode(image_, kernel, iterations=1)
    width_n, height_n = image_.shape[:2]
    return image_[border_size:height_n - border_size, border_size:width_n - border_size]


def __column_fill(mask):
    width, height = mask.shape[:2]
    for i in range(width):
        non_zero = np.nonzero(mask[:, i])[0]
        if len(non_zero) == 0:
            continue
        ind_min = min(non_zero)
        ind_max = max(non_zero)
        mask[ind_min:ind_max, i] = 255
    return mask


def __row_fill(mask):
    width, height = mask.shape[:2]
    for i in range(height):
        non_zero = np.nonzero(mask[i, :])[0]
        if len(non_zero) == 0:
            continue
        ind_min = min(non_zero)
        ind_max = max(non_zero)
        mask[i, ind_min:ind_max] = 255
    return mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", default=None, type=str, help="Single file to segment")
    parser.add_argument("-d",  "--dir", default=None, type=str, help="Directory of files to segment")
    parser.add_argument("-o", "--output", default=None, type=str, required=True,
                        help="Where to output/save the segments to")
    parser.add_argument("-c", "--convex_fill", default=False, action='store_true', help="Whether to fill masks")
    args = parser.parse_args()
    if args.file is None and args.dir is None:
        raise ValueError("At least one of `file` or `dir` must be given.")
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    segment_model = get_segment_model()

    i = 0
    if args.file:
        img = get_face(segment_model, args.file, convex_fill=args.convex_fill)
        cv2.imwrite('{output}/{ind}.png'.format(output=args.output, ind=i), img)
        i += 1

    if args.dir:
        img_paths = get_files_in_dir(args.dir)
        for img_path in img_paths:
            img = get_face(segment_model, img_path, convex_fill=args.convex_fill)
            cv2.imwrite('{output}/{ind}.jpg'.format(output=args.output, ind=i), img)
            i += 1


if __name__ == "__main__":
    main()
