import os

from pathlib import Path

import scipy.io

__all__ = ['paths_and_gt_dict']


class CVPR2013Config:
    reflection_training_single_images = 'reflection_training/single_training/'
    reflection_training_single_gt_mat = 'reflection_training/singleGT_training/singleGT_training.mat'

    reflection_training_multiple_images = 'reflection_training/multiple_training/'
    reflection_training_multiple_gt_mat = 'reflection_training/multipleGT_training/multipleGT_training.mat'

    reflection_testing_single_images = 'reflection_testing/single_testing/'
    reflection_testing_multiple_images = 'reflection_testing/multiple_testing/'
    reflection_testing_gt_mat = 'reflectionTestGt2013.mat'

    config_dict = {
        'test_multiple': {
            'gt_mat': reflection_testing_gt_mat,
            'images_path': reflection_testing_multiple_images
        },
        'test_single': {
            'gt_mat': reflection_testing_gt_mat,
            'images_path': reflection_testing_single_images
        },
        'train_multiple': {
            'gt_mat': reflection_training_multiple_gt_mat,
            'images_path': reflection_training_multiple_images
        },
        'train_single': {
            'gt_mat': reflection_training_single_gt_mat,
            'images_path': reflection_training_single_images
        }
    }

    def __class_getitem__(cls, key: tuple):
        db_path, db_name = key
        if not db_name in cls.config_dict:
            raise KeyError

        item = cls.config_dict[db_name]
        mat_path = os.path.join(db_path, item['gt_mat'])
        images_path = os.path.join(db_path, item['images_path'])

        return mat_path, images_path


def parse_mat_data_testing(names, vectors):
    output_dict = {}

    for i, v_name in enumerate(names):
        name = str.strip(v_name[0][0])
        v = list(map(int, vectors[i]))
        output_dict.setdefault(name, [])
        output_dict[name].append(v)

    return output_dict


def parse_mat_data_training(gt_data):
    output_dict = {}

    for item in gt_data:
        name = str.strip(item[0][0][0])
        vectors = [list(map(int, s)) for s in item[0][1]]
        output_dict[name] = vectors

    return output_dict


def paths_and_gt_dict(path, db):
    """
    Return a list of image paths and their (images) ground-truth values.

    Parameters
    ----------
    path
        Path to database.
    db
        'test_multiple', 'test_single',
        'train_multiple' or 'train_single'

    Returns
    -------
    dict
        image_paths
            List of image paths.
        gt
            Dict with gt values.
    """
    gt_mat_path, images_path = CVPR2013Config[path, db]

    gt_mat = scipy.io.loadmat(gt_mat_path)

    db_name = db
    if db_name == 'test_multiple':
        gt_dict = parse_mat_data_testing(gt_mat['gtM'][0][0][0], gt_mat['gtM'][0][0][1])
    elif db_name == 'test_single':
        gt_dict = parse_mat_data_testing(gt_mat['gtS'][0][0][0], gt_mat['gtS'][0][0][1])
    else:
        gt_dict = parse_mat_data_training(gt_mat['gt'])

    images_list = list(Path(images_path).glob('*.jpg'))

    return {
        'image_paths': images_list,
        'gt': gt_dict
    }
