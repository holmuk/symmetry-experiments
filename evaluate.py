import argparse
import os
import textwrap
import json

import cv2 as cv
import numpy as np

from tqdm import tqdm

from utils.utils import normal_of_segment, angles_equal
from utils.utils import estimate_affine_transform, estimate_reflection_naive, estimate_reflection_from_transform
from utils.image_tools import reflect_image_x, draw_matlab_segments
from utils.correlation import ransac_matching

from matching import matching as matching_fn

import utils.browse_nyu as nyu
import utils.browse_cvpr2013 as cvpr


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--path',
                        required=True,
                        help='Path to chosen image or database')
    parser.add_argument('--database',
                        default='NYU',
                        choices=['NYU', 'CVPR2013'],
                        help='Database name')
    parser.add_argument('--multiple-axes',
                        action='store_true',
                        help='Evaluate on multiple axes dataset')
    parser.add_argument('--print-json',
                        action='store_true',
                        help='Print JSON output')
    parser.add_argument('--save-path',
                        default=None)

    parser.add_argument('--method',
                        default='ransac',
                        choices=['sift', 'ransac'])

    parser.add_argument('--line-threshold',
                        type=float,
                        default=0.2,
                        help='Line threshold')
    parser.add_argument('--angle-threshold',
                        type=float,
                        default=0.97,
                        help='Angle threshold')
    parser.add_argument('--eigenvalue-threshold',
                        type=float,
                        default=1e-1,
                        help='Eigenvalue threshold')
    parser.add_argument('--fitting-method',
                        default='line',
                        choices=['line', 'eigen'])
    parser.add_argument('--min-point-distance-coeff',
                        type=float,
                        default=0.1)
    parser.add_argument('--ransac-box',
                        type=int,
                        default=40)
    parser.add_argument('--ransac-threshold',
                        type=float,
                        default=0.01)
    parser.add_argument('--ransac-samples',
                        type=int,
                        default=200)
    parser.add_argument('--ransac-matching-function',
                        default='ccoeff',
                        choices=['ccoeff', 'ccorr', 'sqdiff'])
    parser.add_argument('--ransac-matching-threshold',
                        type=float,
                        default=0.75)
    parser.add_argument('--ransac-variance-threshold',
                        type=float,
                        default=1e-4)

    parser.add_argument('--affine-method',
                        default='RANSAC',
                        choices=['LMEDS', 'RANSAC'])
    parser.add_argument('--affine-ransac-reproj-threshold',
                        type=float,
                        default=5)
    parser.add_argument('--affine-max-iters',
                        type=int,
                        default=2500)
    parser.add_argument('--affine-confidence',
                        type=float,
                        default=0.9)
    parser.add_argument('--affine-refine-iters',
                        type=int,
                        default=10)

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


class Matcher():
    def __init__(self,
                 path,
                 save_path=None,
                 database='NYU',
                 method='ransac',
                 multiple_axes=False,
                 min_point_distance_coeff=0.1,
                 lowe_ratio=0.75,
                 line_threshold=0.2,
                 angle_threshold=0.97,
                 eigenvalue_threshold=1e-1,
                 fitting_method='line',
                 ransac_box=40,
                 ransac_samples=200,
                 ransac_matching_function='sqdiff',
                 ransac_matching_threshold=0.5,
                 ransac_variance_threshold=1e-2,
                 ransac_threshold=0.02,
                 print_json=False,
                 affine_method='RANSAC',
                 affine_ransac_reproj_threshold=3,
                 affine_max_iters=2000,
                 affine_confidence=0.9,
                 affine_refine_iters=10):
        self.save_path = save_path
        if os.path.isdir(path):
            self.eval_dict = read_eval_dict(path, database, multiple_axes)

        self.line_threshold = line_threshold
        self.angle_threshold = angle_threshold
        self.eigenvalue_threshold = eigenvalue_threshold
        self.fitting_method = fitting_method
        self.print_json = print_json
        self.min_point_distance_coeff = min_point_distance_coeff
        self.lowe_ratio = lowe_ratio
        self.method = method

        self.ransac_box = ransac_box
        self.ransac_samples = ransac_samples
        self.ransac_threshold = ransac_threshold
        self.ransac_matching_threshold = ransac_matching_threshold
        self.ransac_variance_threshold = ransac_variance_threshold

        self.ransac_matching_function_text = ransac_matching_function
        self.ransac_matching_function = matching_fn.__getattribute__(
            'tm_{}_matching'.format(ransac_matching_function)
        )

        if method == 'sift':
            self.sift = cv.xfeatures2d.SIFT_create()
            self.matcher = self.create_matcher()

            self.affine_params = {
                'method': cv.RANSAC if affine_method == 'RANSAC' else cv.LMEDS,
                'ransacReprojThreshold': affine_ransac_reproj_threshold,
                'maxIters': affine_max_iters,
                'confidence': affine_confidence,
                'refineIters': affine_refine_iters
            }

    def create_matcher(self):
        index_params = {
            'algorithm': 1,
            'trees': 5
        }

        search_params = {}

        matcher = cv.FlannBasedMatcher(index_params, search_params)
        return matcher

    def match_raw_keypoints(self, img1, img2):
        img_w = img1.shape[1]

        keypoints1, descr1 = self.sift.detectAndCompute(img1, None)
        keypoints2, descr2 = self.sift.detectAndCompute(img2, None)

        matches_knn = self.matcher.knnMatch(descr2, descr1, k=2)

        k1points = []
        k2points = []
        matches = []
        for m, n in matches_knn:
            if m.distance < self.lowe_ratio * n.distance:
                point1 = list(map(int, keypoints1[m.trainIdx].pt))
                point2 = list(map(int, keypoints2[m.queryIdx].pt))

                r = np.linalg.norm([point1[0] - point2[0], point1[1] - point2[1]])
                if r < self.min_point_distance_coeff * img_w:
                    continue

                matches.append(m)
                k1points.append(keypoints1[m.trainIdx].pt)
                k2points.append(keypoints2[m.queryIdx].pt)

        k1points = np.array(k1points)
        k2points = np.array(k2points)
        matches = np.array(matches)

        return matches, k1points, k2points

    def find_symmetry_sift(self, img, draw_points=False):
        jimg, S, N, d = reflect_image_x(img)

        matches, k1p, k2p = self.match_raw_keypoints(img, jimg)
        p, v, points_img = None, None, None

        if self.fitting_method == 'line':
            p, v, _ = estimate_reflection_naive(k1p, k2p)
        else:
            M = estimate_affine_transform(k2p, k1p, self.affine_params)
            if M is not None:
                p, v = estimate_reflection_from_transform(
                    S, M,
                    w_offset=d,
                    normal_vector=N,
                    eigenvalue_threshold=self.eigenvalue_threshold
                )

        if draw_points:
            points_img = cv.cvtColor(img.copy(), cv.COLOR_GRAY2BGR)
            for p1_temp, p2_temp in zip(k1p, k2p):
                points_img = cv.circle(points_img, tuple(map(int, p1_temp)), 2, (0, 255, 0))
                points_img = cv.circle(points_img, tuple(map(int, p2_temp)), 2, (255, 0, 0))
                cv.line(points_img, tuple(map(int, p1_temp)), tuple(map(int, p2_temp)), (255, 255, 0))
            # for pm in m:
              #   points_img = cv.circle(points_img, tuple(map(int, pm)), 2, (0, 0, 255))

        return p, v, points_img

    def find_symmetry_ransac(self, img):
        reflected_img, S, N, d = reflect_image_x(img)

        tmatrix = ransac_matching(img, reflected_img,
                                  matching_fn=self.ransac_matching_function,
                                  n_samples=self.ransac_samples,
                                  box_size=self.ransac_box,
                                  ransac_threshold=self.ransac_threshold,
                                  variance_threshold=self.ransac_variance_threshold)

        if tmatrix is None:
            return None, None

        p, v = estimate_reflection_from_transform(S, tmatrix,
                                                  w_offset=d,
                                                  normal_vector=N,
                                                  eigenvalue_threshold=self.eigenvalue_threshold)
        return p, v

    def interaction(self,
                    path_to_img,
                    gt_segments,
                    window_name=None,
                    window_name_matches=None):
        img = cv.imread(path_to_img, cv.IMREAD_GRAYSCALE)

        save_image = self.save_path
        if save_image is not None:
            if os.path.isdir(save_image):
                save_image = os.path.join(save_image, os.path.basename(path_to_img))

        if self.method == 'ransac':
            p, v = self.find_symmetry_ransac(img)
            img_matches = None
        else:
            p, v, img_matches = self.find_symmetry_sift(img)

        if p is not None:
            if not (window_name is None and
                    window_name_matches is None and
                    save_image is None):
                img_c = cv.imread(path_to_img, cv.IMREAD_COLOR)

                if gt_segments is not None:
                    draw_matlab_segments(img_c, gt_segments)

                cv.line(img_c, tuple(map(int, p + 250*v)), tuple(map(int, p - 250*v)), (255, 0, 0), 1)
                cv.circle(img_c, tuple(map(int, p)), 1, (255, 0, 255), -1)

                if window_name is not None:
                    cv.namedWindow(window_name, cv.WINDOW_GUI_NORMAL)
                    cv.imshow(window_name, img_c)
                    cv.waitKey(0)

                if window_name_matches is not None and img_matches is not None:
                    cv.namedWindow(window_name_matches, cv.WINDOW_GUI_NORMAL)
                    cv.imshow(window_name_matches, img_matches)
                    cv.waitKey(0)

                if save_image is not None:
                    cv.imwrite(save_image, img_c)

            return p, v

        return None, None

    def evaluate(self):
        fp = 0
        fn = 0
        tp = 0

        tp_angle = 0
        fp_angle = 0
        fn_angle = 0

        img_list = self.eval_dict['image_paths']
        img_gt = self.eval_dict['gt']

        total = len(img_list)

        for path in tqdm(img_list):
            gt_segments = img_gt[path.name]

            p, v = self.interaction(str(path), gt_segments,
                                    window_name=None,
                                    window_name_matches=None)
            if p is None:
                fn += 1
                fn_angle += 1
            else:
                v_orig, len_orig, c = normal_of_segment(gt_segments[0])

                a_eq = angles_equal(v_orig, v,
                                    eps=self.angle_threshold)

                d = c - np.ravel(p)
                b_eq = np.abs(-v[1] * d[0] + v[0] * d[1]) < self.line_threshold * len_orig

                if a_eq:
                    tp_angle += 1
                else:
                    fp_angle += 1

                if a_eq and b_eq:
                    tp += 1
                else:
                    fp += 1

        summary = {
            'method': self.method,
            'ransac_params': {
                'ransac_threshold': self.ransac_threshold,
                'ransac_matching_function': self.ransac_matching_function_text,
                'ransac_box': self.ransac_box,
                'ransac_samples': self.ransac_samples
            },
            'registration_params': {
                'fitting_method': self.fitting_method,
                'min_point_distance_coeff': self.min_point_distance_coeff,
                'estimator_params': self.affine_params
            },
            'eigenvalue_threshold': self.eigenvalue_threshold,
            'angle_threshold': self.angle_threshold,
            'line_threshold': self.line_threshold,
            'results': {
                'fp': fp,
                'tp': tp,
                'fn': fn,
                'accuracy': tp / total,
                'precision': tp / (tp + fp),
                'recall': tp / (tp + fn),
                'precision_angle': tp_angle / (tp_angle + fp_angle),
                'recall_angle': tp_angle / (tp_angle + fn_angle)
            }
        }

        self.print_summary(summary)
        return summary

    def print_summary(self, summary):
        """"Print the summary."""
        if self.print_json:
            print(json.dumps(summary, indent=4))
        else:
            print(textwrap.dedent("""
                FP: {fp},
                FN: {fn},
                TP: {tp},
                Accuracy: {accuracy},
                Precision: {precision},
                Recall: {recall},
                Precision (Angle): {precision_angle},
                Recall (Angle): {recall_angle}
            """).format(
                **summary['results']
            ))


if __name__ == '__main__':
    args = parse_args()

    matcher = Matcher(**vars(args))
    if os.path.isfile(args.path):
        matcher.interaction(args.path, None,
                            window_name='Result')
    else:
        matcher.evaluate()
