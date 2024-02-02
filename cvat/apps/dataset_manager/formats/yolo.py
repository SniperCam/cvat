# Copyright (C) 2019-2022 Intel Corporation
# Copyright (C) 2023-2024 CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

import os.path as osp
from glob import glob

from pyunpack import Archive
import random

from cvat.apps.dataset_manager.bindings import (GetCVATDataExtractor,
    import_dm_annotations, match_dm_item, find_dataset_root)
from cvat.apps.dataset_manager.util import make_zip_archive
from datumaro.components.extractor import DatasetItem
from datumaro.components.project import Dataset
from datumaro.plugins.yolo_format.extractor import YoloExtractor

from .registry import dm_env, exporter, importer


@exporter(name='YOLO', ext='ZIP', version='1.1')
def _export(dst_file, temp_dir, instance_data, save_images=False):
    with GetCVATDataExtractor(instance_data, include_images=save_images) as extractor:
        dataset = Dataset.from_extractors(extractor, env=dm_env)
        dataset.export(temp_dir, 'yolo', save_images=save_images)

    make_zip_archive(temp_dir, dst_file)

@exporter(name='YOLO_split', ext='ZIP', version='1.1')
def _export_split(dst_file, temp_dir, instance_data, save_images=False):
    with GetCVATDataExtractor(instance_data, include_images=save_images) as extractor:
        dataset = Dataset.from_extractors(extractor, env=dm_env)
        # Convert the dataset to a list
        items = list(dataset)

        # Shuffle the list
        random.shuffle(items)

        # Calculate the indices for the training, validation, and testing subsets
        total_items = len(items)
        train_end = int(total_items * 0.8)
        val_end = train_end + int(total_items * 0.1)

        # Use slicing to get the subsets
        train_items = items[:train_end]
        val_items = items[train_end:val_end]
        test_items = items[val_end:]

        train_dataset = Dataset.from_iterable(train_items, env=dm_env)
        val_dataset = Dataset.from_iterable(val_items, env=dm_env)
        test_dataset = Dataset.from_iterable(test_items, env=dm_env)

        train_dataset.export(temp_dir+ '/train', 'coco_instances', save_images=save_images,
            merge_images=True)
        val_dataset.export(temp_dir+ '/validate', 'coco_instances', save_images=save_images,
            merge_images=True)
        test_dataset.export(temp_dir+ '/test', 'coco_instances', save_images=save_images,
            merge_images=True)

    make_zip_archive(temp_dir, dst_file)

@importer(name='YOLO', ext='ZIP', version='1.1')
def _import(src_file, temp_dir, instance_data, load_data_callback=None, **kwargs):
    Archive(src_file.name).extractall(temp_dir)

    image_info = {}
    frames = [YoloExtractor.name_from_path(osp.relpath(p, temp_dir))
        for p in glob(osp.join(temp_dir, '**', '*.txt'), recursive=True)]
    root_hint = find_dataset_root(
        [DatasetItem(id=frame) for frame in frames], instance_data)
    for frame in frames:
        frame_info = None
        try:
            frame_id = match_dm_item(DatasetItem(id=frame), instance_data,
                root_hint=root_hint)
            frame_info = instance_data.frame_info[frame_id]
        except Exception: # nosec
            pass
        if frame_info is not None:
            image_info[frame] = (frame_info['height'], frame_info['width'])

    dataset = Dataset.import_from(temp_dir, 'yolo',
        env=dm_env, image_info=image_info)
    if load_data_callback is not None:
        load_data_callback(dataset, instance_data)
    import_dm_annotations(dataset, instance_data)
