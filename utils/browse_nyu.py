import os

from pathlib import Path

import scipy.io

__all__ = ['paths_and_gt_dict']


class NYUConfig:
    reflection_single_images_data = 'S/'
    reflection_multiple_images_data = 'M/'

    config_dict = {
        'multiple': {
            'data': reflection_multiple_images_data
        },
        'single': {
            'data': reflection_single_images_data
        }
    }

    def __class_getitem__(cls, key: tuple):
        db_path, db_name = key
        if not db_name in cls.config_dict:
            raise KeyError

        item = cls.config_dict[db_name]
        data_path = os.path.join(db_path, item['data'])

        return data_path


def paths_and_gt_dict(path, db):
    """
    Return a list of image paths and their (images) ground-truth values.

    Parameters
    ----------
    path
        Path to database.
    db
        'single' or 'multiple'

    Returns
    -------
    dict
        image_paths
            List of image paths.
        gt
            Dict with gt values.
    """
    images_path = NYUConfig[path, db]
    images_list = list(Path(images_path).glob('*.png'))

    gt_dict = {}
    for path in images_list:
        name = path.name
        path_to_mat = path.parent / (path.stem + '.mat')
        mat_data = scipy.io.loadmat(str(path_to_mat))
        gt_dict.setdefault(name, [])
        segments = [
            [int(p[0][0]), int(p[0][1]), int(p[1][0]), int(p[1][1])]
            for p in mat_data['segments'][0]
        ]
        gt_dict[name].extend(segments)

    return {
        'image_paths': images_list,
        'gt': gt_dict
    }
