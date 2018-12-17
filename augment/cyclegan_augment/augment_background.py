import os

import numpy as np
import cv2
from PIL.Image import fromarray


class AugmentBackground(object):

    net = None
    backgrounds = None
    rand = None

    def __init__(self, backgrounds_path=os.path.join('data', 'augment_data', 'backgrounds')):
        self.backgrounds_path = self._get_files_in_dir(backgrounds_path)

    def __call__(self, img):
        bool_mask = self._get_mask(img)
        face_mask = self._apply_mask(img, bool_mask)
        face_background = self._apply_background(bool_mask, face_mask)
        return fromarray(face_background)

    def __repr__(self):
        return self.__class__.__name__ + '()'

    @staticmethod
    def _get_mask(img):
        img_np = np.array(img)
        bool_mask = (img_np != 0)
        return bool_mask

    @staticmethod
    def _apply_mask(image, bool_mask):
        return np.array(np.multiply(image, bool_mask), dtype=np.uint8)

    def _apply_background(self, mask, face_mask):
        background = self.__get_random_background()
        top, bot, left, right = self.__get_random_position(mask.shape, background.shape)
        mask_int = mask * 1
        mask_background = cv2.copyMakeBorder(mask_int, top, bot, left, right, cv2.BORDER_CONSTANT, value=(1, 1, 1))
        apply_mask_background = np.array(np.multiply(background, mask_background), dtype=np.uint8)
        apply_mask_face_padding = cv2.copyMakeBorder(face_mask, top, bot, left, right, cv2.BORDER_CONSTANT,
                                                     value=(0, 0, 0))
        return apply_mask_face_padding + apply_mask_background

    @staticmethod
    def _get_files_in_dir(path):
        folder = os.fsencode(path)
        filenames = []
        for file in os.listdir(folder):
            filename = os.fsdecode(file)
            if filename.endswith(('jpg', '.jpeg', '.png', '.gif')):
                filenames.append('{bg}/{f}'.format(bg=path, f=filename))
        return filenames

    def __get_random_background(self):
        return cv2.cvtColor(
            cv2.imread(self.backgrounds_path[self.rand.randint(0, len(self.backgrounds_path)-1)]),
            cv2.COLOR_BGR2RGB)

    def __get_random_position(self, mask_shape, background_shape):
        if mask_shape[0] > background_shape[0]:
            raise ValueError("Your background needs to be longer")
        if mask_shape[1] > background_shape[1]:
            raise ValueError("Your background needs to be wider")
        top_max = background_shape[0] - mask_shape[0]
        left_max = background_shape[1] - mask_shape[1]
        top = self.rand.randint(0, top_max)
        bot = top_max - top
        left = self.rand.randint(0, left_max)
        right = left_max - left
        return top, bot, left, right
