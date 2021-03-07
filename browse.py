import argparse

import cv2 as cv

import utils.browse_nyu as nyu
import utils.browse_cvpr2013 as cvpr

from utils.image_tools import draw_matlab_segments


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--path',
                        required=True,
                        help='Path to chosen database')
    parser.add_argument('--database',
                        default='NYU',
                        choices=['NYU', 'CVPR2013'],
                        help='Database name')
    parser.add_argument('--multiple-axes',
                        action='store_true',
                        help='Browse multiple axes dataset')

    args = parser.parse_args()
    return args


def read_eval_dict(path, database, multiple_axes=True):
    if database == 'NYU':
        return nyu.paths_and_gt_dict(path,
                                     'multiple' if multiple_axes else 'single')
    elif database == 'CVPR2013':
        train_dict = cvpr.paths_and_gt_dict(
            path,
            'train_multiple' if multiple_axes else 'train_single')
        test_dict = cvpr.paths_and_gt_dict(
            path,
            'test_multiple' if multiple_axes else 'test_single')

        train_dict['gt'].update(test_dict['gt'])
        train_dict['image_paths'].extend(test_dict['image_paths'])
        return train_dict
    else:
        print('Unknown database: ', database)
        return {}


if __name__ == '__main__':
    args = parse_args()
    paths_and_gt = read_eval_dict(args.path, args.database,
                                  args.multiple_axes)

    images_list = paths_and_gt['image_paths']
    gt_dict = paths_and_gt['gt']

    n_images = len(images_list)

    window_name = 'View {} database'.format(args.database)
    window_width = 300
    window_height = 300

    def on_trackbar_change(pos):
        path_to_img = images_list[pos]
        img_name = path_to_img.parts[-1]
        img = cv.imread(str(path_to_img))
        draw_matlab_segments(img, gt_dict[img_name])
        cv.imshow(window_name, img)

    cv.namedWindow(window_name, cv.WINDOW_GUI_NORMAL)
    cv.resizeWindow(window_name, window_width, window_height)
    cv.createTrackbar('N', window_name, 0, n_images - 1, on_trackbar_change)
    cv.setTrackbarMin('N', window_name, 0)
    on_trackbar_change(0)
    cv.waitKey(0)
