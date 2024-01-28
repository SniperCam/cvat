# Copyright (C) 2018-2022 Intel Corporation
# Copyright (C) 2023-2024 CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT
import logging
import zipfile
import random

from datumaro.components.dataset import Dataset
from datumaro.components.annotation import AnnotationType

from cvat.apps.dataset_manager.bindings import GetCVATDataExtractor, \
    import_dm_annotations
from cvat.apps.dataset_manager.util import make_zip_archive

from .registry import dm_env, exporter, importer

@exporter(name='COCO_split', ext='ZIP', version='1.0')
def _export_split(dst_file, temp_dir, instance_data, save_images=False):
    dataset = Dataset.from_extractors(GetCVATDataExtractor(
        instance_data, include_images=save_images), env=dm_env)
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

    train_dataset.export(temp_dir+ '/train_coco', 'coco_instances', save_images=save_images,
        merge_images=True)
    val_dataset.export(temp_dir+ '/val_coco', 'coco_instances', save_images=save_images,
        merge_images=True)
    test_dataset.export(temp_dir+ '/test_coco', 'coco_instances', save_images=save_images,
        merge_images=True)

    make_zip_archive(temp_dir, dst_file)

@exporter(name='COCO', ext='ZIP', version='1.0')
def _export(dst_file, temp_dir, instance_data, save_images=False):
    with GetCVATDataExtractor(instance_data, include_images=save_images) as extractor:
        dataset = Dataset.from_extractors(extractor, env=dm_env)
        dataset.export(temp_dir, 'coco_instances', save_images=save_images,
            merge_images=True)

    make_zip_archive(temp_dir, dst_file)

@importer(name='COCO', ext='JSON, ZIP', version='1.0')
def _import(src_file, temp_dir, instance_data, load_data_callback=None, **kwargs):
    if zipfile.is_zipfile(src_file):
        zipfile.ZipFile(src_file).extractall(temp_dir)
        dataset = Dataset.import_from(temp_dir, 'coco_instances', env=dm_env)
        if load_data_callback is not None:
            load_data_callback(dataset, instance_data)
        import_dm_annotations(dataset, instance_data)
    else:
        dataset = Dataset.import_from(src_file.name,
            'coco_instances', env=dm_env)
        import_dm_annotations(dataset, instance_data)

@exporter(name='COCO Keypoints', ext='ZIP', version='1.0')
def _export(dst_file, temp_dir, instance_data, save_images=False):
    with GetCVATDataExtractor(instance_data, include_images=save_images) as extractor:
        dataset = Dataset.from_extractors(extractor, env=dm_env)
        dataset.export(temp_dir, 'coco_person_keypoints', save_images=save_images,
            merge_images=True)

    make_zip_archive(temp_dir, dst_file)

@importer(name='COCO Keypoints', ext='JSON, ZIP', version='1.0')
def _import(src_file, temp_dir, instance_data, load_data_callback=None, **kwargs):
    def remove_extra_annotations(dataset):
        for item in dataset:
            annotations = [ann for ann in item.annotations
                if ann.type != AnnotationType.bbox]
            item.annotations = annotations

    if zipfile.is_zipfile(src_file):
        zipfile.ZipFile(src_file).extractall(temp_dir)
        dataset = Dataset.import_from(temp_dir, 'coco_person_keypoints', env=dm_env)
        remove_extra_annotations(dataset)
        if load_data_callback is not None:
            load_data_callback(dataset, instance_data)
        import_dm_annotations(dataset, instance_data)
    else:
        dataset = Dataset.import_from(src_file.name,
            'coco_person_keypoints', env=dm_env)
        remove_extra_annotations(dataset)
        import_dm_annotations(dataset, instance_data)
