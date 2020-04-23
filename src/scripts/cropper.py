import cv2
import numpy as np


def align_crop(img, landmarks, standard_landmarks, crop_size=572, face_factor=0.45, align_type='similarity', order=3,
               mode='edge'):
    """ Align and crop a face image by landmarks

    :param img:                 Face image to be aligned and cropped.
    :param landmarks:           [[x_1, y_1], ..., [x_n, y_n]].
    :param standard_landmarks:  Standard landmarks that landmarks are aligned according them.
    :param crop_size:           Output image size, should be int for (crop_size, crop_size).
    :param face_factor:         The factor of face area relative to the output image.
    :param align_type:          'similarity' or 'affine'.
    :param order:               The order of interpolation, The order has to be in the range 0-5:
                                    0: INTER_NEAREST;
                                    1: INTER_LINEAR;
                                    2: INTER_AREA;
                                    3: INTER_CUBIC;
                                    4: INTER_LANCZOS4;
                                    5: INTER_LANCZOS4.
    :param mode:                One of ['constant', 'edge', 'symmetric', 'reflect', 'wrap']. Points outside the boundaries of the
                                input are filled according to the given mode.
    :return:                    cropped img and transformed landmarks
    """
    interpolation = {0: cv2.INTER_NEAREST,
                     1: cv2.INTER_LINEAR,
                     2: cv2.INTER_AREA,
                     3: cv2.INTER_CUBIC,
                     4: cv2.INTER_LANCZOS4,
                     5: cv2.INTER_LANCZOS4}
    border = {'constant': cv2.BORDER_CONSTANT,
              'edge': cv2.BORDER_REPLICATE,
              'symmetric': cv2.BORDER_REFLECT,
              'reflect': cv2.BORDER_REFLECT101,
              'wrap': cv2.BORDER_WRAP}

    # check
    assert align_type in ['affine', 'similarity'], \
        " [!] Invalid 'align_type'! The {} is not included in ['affine' and 'similarity']!".format(align_type)
    assert order in [0, 1, 2, 3, 4, 5], \
        " [!] Invalid 'order'! The {} is not included in [0, 1, 2, 3, 4, 5]!".format(order)
    assert mode in ['constant', 'edge', 'symmetric', 'reflect', 'wrap'], \
        " [!] Invalid 'mode'! the {} is not included in ['constant', 'edge', 'symmetric', 'reflect', and 'wrap']".format(mode)

    # crop size
    if isinstance(crop_size, (list, tuple)) and len(crop_size) == 2:
        crop_size_h = crop_size[0]
        crop_size_w = crop_size[1]
    elif isinstance(crop_size, int):
        crop_size_h = crop_size_w = crop_size
    else:
        raise Exception(" [!] Invalid 'crop_size'! The 'crop_size' should be (1) one integer for (crop_size, crop_size) ar (2) (int, int) for (crop_size_h, crop_size_w)!")

    # estimate transform matrix
    target_landmarks = standard_landmarks * max(crop_size_h, crop_size_w) * face_factor + np.array([crop_size_w // 2, crop_size_h // 2])
    if align_type == 'affine':  # 6 degree of freedom
        transform_matrix, _ = cv2.estimateAffine2D(target_landmarks, landmarks, ransacReprojThreshold=np.Inf)
    else:  # 4 degree of freedom: using the combinations of translation, rotation, and uniform scaling
        transform_matrix, _ = cv2.estimateAffinePartial2D(target_landmarks, landmarks, ransacReprojThreshold=np.Inf)

    # warp image by given transform
    img_crop = cv2.warpAffine(img, transform_matrix, dsize=(crop_size_w, crop_size_h),
                              flags=cv2.WARP_INVERSE_MAP + interpolation[order], borderMode=border[mode])

    # get transformed landmarks
    transformed_landmarks = cv2.transform(np.expand_dims(landmarks, axis=0), m=cv2.invertAffineTransform(transform_matrix))

    return img_crop, transformed_landmarks


