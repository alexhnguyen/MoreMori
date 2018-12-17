import numpy as np
import cv2

from utils.utils import roi_image_resize
from utils.utils import resize_factor, rotate_90, rotate_180, rotate_270

from haar.face_haar import GetFaceHaar
from deeplearning.face_dl import GetFaceDL


def main():
    '''
    Argument parsing
    '''
    import argparse
    parser = argparse.ArgumentParser(description='Cut faces from input video.')
    parser.add_argument('--skipframe', type=int, action='store', default=0,
            help='number of frames to skip between grabs')
    parser.add_argument('--size', type=int, action='store', default=256,
            help='output size (ROI)')
    parser.add_argument('--factor', type=float, action='store', default=1.5,
            help='ROI area factor')
    parser.add_argument('--detector', type=str.lower, action='store', default='haar', 
            choices=['haar', 'deeplearning'], help='face detection method.')
    parser.add_argument('--resize', type=float, action='store', default=0,
            help='Input resize factor')
    parser.add_argument('--rotation', type=str.lower, action='store', default='0', 
            choices=['0', '90', '180', '270'], help='Counter clockwise rotation in degrees.')
    parser.add_argument('--output', type=str, action='store', default='',
            help='output path prefix')
    parser.add_argument('inputs', type=str, nargs='+', help='input video files')
    parser.add_argument('--prefix', type=str, action='store', default='',
            help='prefix for file name')
    args = parser.parse_args()

    '''
    Select a detection method
    '''
    if args.detector == 'haar':
        detector = GetFaceHaar()
    elif args.detector == 'deeplearning':

        detector = GetFaceDL()
    else:
        raise NotImplementedError('Detection method ' + args.detector)

    '''
    Compose transforms
    '''
    # Resize
    if args.resize >= 0:
        resize = lambda image: resize_factor(image, args.resize)
    else:
        raise ValueError('Resize factor ' + str(args.resize) + ' less than 0')

    # Rotation
    if args.rotation == '0':
        rotate = lambda image: image
    elif args.rotation == '90':
        rotate = lambda image: rotate_90(image)
    elif args.rotation == '180':
        rotate = lambda image: rotate_180(image)
    elif args.rotation == '270':
        rotate = lambda image: rotate_270(image)
    else:
        raise NotImplementedError('Rotation ' + args.rotation)
    
    transforms = lambda image: resize(rotate(image))
    
    '''
    Perform detection on input(s) and save ROI(s) in ouput directory
    Note: possible to have more than one ROI per input frame
    '''
    for count, inp in enumerate(args.inputs):
        cap = cv2.VideoCapture(inp)
        ind = 0
        while cap.isOpened():
            # Get a frame from the video
            for i in range(0, args.skipframe):
                ret, frame = cap.read()

            ret, frame = cap.read()
            if not ret:
                break

            # Apply the requested transforms
            frame = transforms(frame)

            # Convert to grey
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Use the detector to get ROI(s)
            rois = detector(gray)

            # Save each roi image
            saved_rois = []
            for (x, y, w, h) in rois:
                roi_dimensions, roi_image = roi_image_resize(frame, x, y, w, h, 
                                             args.factor, args.size, args.size)
                cv2.imwrite('{path}{prefix}_{ind}.jpg'.format(path=args.output,
                                                              prefix=args.prefix,
                                                              ind=ind), roi_image)
                saved_rois.append(roi_dimensions)
                ind += 1

            # Draw in bounding boxes
            for (x, y, w, h) in saved_rois:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Show the image with bounding boxes
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Cleanup
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
