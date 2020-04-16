import argparse
import cv2
from functools import partial

from scripts.cropper import align_crop


parser = argparse.ArgumentParser()
parser.add_argument('--img_dir', dest='img_dir', default='./workspace/cpfs-data/Data/img_align_celeba',
                    help='path of your celeb dataset')
parser.add_argument('--save_dir', dest='save_dir', default='D:/Data/img_celeba/aligned',
                    help='saving path for the aligned data')
parser.add_argument('--landmark_file', dest='landmark_file', default='D:/Data/img_celeba/landmark.txt',
                    help='landmark coordinates saved for each image')
parser.add_argument('--standard_landmark_file', dest='standard_landmark_file',
                    default='D:/Data/img_celeba/standard_landmark_68pts.txt',
                    help='imgs will be alinged by calculating perspective matrix between landmark and standard landmark')
parser.add_argument('--crop_size_h', dest='crop_size_h', type=int, default=572, help='height of the cropped img')
parser.add_argument('--crop_size_w', dest='crop_size_w', type=int, default=572, help='width of the cropped img')
parser.add_argument('--move_h', dest='move_h', type=float, default=0.25,
                    help='relative height compared to the coordinates of the standard landmark')
parser.add_argument('--move_w', dest='move_w', type=float, default=0.,
                    help='relative width compared to the coordinates of the standard landmark')
parser.add_argument('--save_format', dest='save_format', choices=['.jpg', '.png'], default='.jpg',
                    help='extension of the saved img')
parser.add_argument('--n_worker', dest='n_worker', type=int, default=8, help='number threads for using data alignment')
parser.add_argument('--face_factor', dest='face_factor', type=float, default=0.45,
                    help='the factor of face area relative to the output image')
args = parser.parse_args()


_DEFAULT_JPG_QUALITY = 95
imread = cv2.imread
imwrite = partial(cv2.imwrite, params=[int(cv2.IMWRITE_JPEG_QUALITY), _DEFAULT_JPG_QUALITY])


# count landmarks
with open(args.landmark_file)
