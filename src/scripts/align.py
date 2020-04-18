import argparse
import os
import cv2
import numpy as np
from functools import partial
from multiprocessing import Pool

from cropper import align_crop


parser = argparse.ArgumentParser()
parser.add_argument('--img_dir', dest='img_dir', default='/workspace/cpfs-data/Data/img_celeba',
                    help='path of your celeb dataset')
parser.add_argument('--save_dir', dest='save_dir', default='/workspace/cpfs-data/Data/img_celeba/aligned',
                    help='saving path for the aligned data')
parser.add_argument('--n_landmark', dest='n_landmark', type=int, default=68,
                    help='number of landmarks for used in celebA dataset')
parser.add_argument('--landmark_file', dest='landmark_file', default='/workspace/cpfs-data/Data/img_celeba/landmark.txt',
                    help='landmark coordinates saved for each image')
parser.add_argument('--standard_landmark_file', dest='standard_landmark_file',
                    default='/workspace/cpfs-data/Data/img_celeba/standard_landmark_68pts.txt',
                    help='imgs will be alinged by calculating perspective matrix between landmark and standard landmark')
parser.add_argument('--crop_size_h', dest='crop_size_h', type=int, default=572, help='height of the cropped img')
parser.add_argument('--crop_size_w', dest='crop_size_w', type=int, default=572, help='width of the cropped img')
parser.add_argument('--move_h', dest='move_h', type=float, default=0.25,
                    help='relative height compared to the coordinates of the standard landmark')
parser.add_argument('--move_w', dest='move_w', type=float, default=0.,
                    help='relative width compared to the coordinates of the standard landmark')
parser.add_argument('--save_format', dest='save_format', choices=['jpg', 'png'], default='jpg',
                    help='extension of the saved img')
parser.add_argument('--n_worker', dest='n_worker', type=int, default=8, help='number threads for using data alignment')
parser.add_argument('--face_factor', dest='face_factor', type=float, default=0.45,
                    help='the factor of face area relative to the output image')
args = parser.parse_args()


_DEFAULT_JPG_QUALITY = 95
imread = cv2.imread
imwrite = partial(cv2.imwrite, params=[int(cv2.IMWRITE_JPEG_QUALITY), _DEFAULT_JPG_QUALITY])

def read_landmark_files():
    # read data
    img_names_ = np.genfromtxt(args.landmark_file, dtype=np.str, usecols=[0])
    landmarks_ = np.genfromtxt(
        args.landmark_file, dtype=np.float, usecols=range(1, args.n_landmark * 2 + 1)).reshape(-1, args.n_landmark, 2)
    standard_landmark_ = np.genfromtxt(args.standard_landmark_file, dtype=np.float).reshape(args.n_landmark, 2)
    standard_landmark_[:, 0] += args.move_w
    standard_landmark_[:, 1] += args.move_h
    return img_names_, landmarks_, standard_landmark_


if __name__ == '__main__':
    print("Reading img paths, and it will take about 1 minute...")

    img_names, landmarks, standard_landmark = read_landmark_files()
    print('num of img_names: {}'.format(len(img_names)))
    print('landmarks shape: {}'.format(landmarks.shape))
    print('standard_landmark shape: {}'.format(standard_landmark.shape))
    assert len(img_names) == landmarks.shape[0], " [*] The number of imgs is different!"
    assert landmarks.shape[1] == standard_landmark.shape[0], \
        " [*] The dimension of landmark betwen the file and standard is different!"

    # data dir
    save_dir = os.path.join(args.save_dir, 'align_size(%d, %d)_move(%.3f, %.3f)_face_factor(%.3f)_%s' % (
        args.crop_size_h, args.crop_size_w, args.move_h, args.move_w, args.face_factor, args.save_format))
    data_dir = os.path.join(save_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)

    pool = Pool(args.n_worker)
