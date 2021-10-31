import io
import json
import os
import random
from itertools import chain as chain

import h5py
import numpy as np
import slowfast.utils.logging as logging
import torch
import torch.utils.data
from fvcore.common.file_io import PathManager
from PIL import Image

from . import image_decoder as image_decoder
from . import utils as utils
from . import video_container as container
from .build import DATASET_REGISTRY
from .data_augment import (
    augment_and_mix_transform,
    auto_augment_transform,
    rand_augment_transform,
)
from .image_decoder import get_start_end_idx
from .rand_erasing import RandomErasing

logger = logging.get_logger(__name__)


@DATASET_REGISTRY.register()
class Ssv2(torch.utils.data.Dataset):
    """
    Something-Something v2 (SSV2) video loader. Construct the SSV2 video loader,
    then sample clips from the videos. For training and validation, a single
    clip is randomly sampled from every video with random cropping, scaling, and
    flipping. For testing, multiple clips are uniformaly sampled from every
    video with uniform cropping. For uniform cropping, we take the left, center,
    and right crop if the width is larger than height, or take top, center, and
    bottom crop if the height is larger than the width.
    """

    def __init__(self, cfg, mode, num_retries=10):
        """
        Load Something-Something V2 data (frame paths, labels, etc. ) to a given
        Dataset object. The dataset could be downloaded from Something-Something
        official website (https://20bn.com/datasets/something-something).
        Please see datasets/DATASET.md for more information about the data format.
        Args:
            cfg (CfgNode): configs.
            mode (string): Options includes `train`, `val`, or `test` mode.
                For the train and val mode, the data loader will take data
                from the train or val set, and sample one clip per video.
                For the test mode, the data loader will take data from test set,
                and sample multiple clips per video.
            num_retries (int): number of retries for reading frames from disk.
        """
        # Only support train, val, and test mode.
        assert mode in [
            "train",
            "val",
            "test",
        ], "Split '{}' not supported for Something-Something V2".format(mode)
        self.mode = mode
        self.cfg = cfg

        self._video_meta = {}
        self._num_retries = num_retries
        # For training or validation mode, one single clip is sampled from every
        # video. For testing, NUM_ENSEMBLE_VIEWS clips are sampled from every
        # video. For every clip, NUM_SPATIAL_CROPS is cropped spatially from
        # the frames.
        if self.mode in ["train", "val"]:
            self._num_clips = 1
        elif self.mode in ["test"]:
            self._num_clips = (
                cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS
            )

        logger.info("Constructing Something-Something V2 {}...".format(mode))
        self._construct_loader()
        self._construct_augmentations()

    def _construct_augmentations(self):
        self.random_erasing = (
            RandomErasing(probability=self.cfg.DATA.RAND_CROP, syncronized=True)
            if self.cfg.DATA.RAND_CROP > 0.0
            else None
        )
        if self.cfg.DATA.AUTOAUGMENT:
            self.auto_augment = auto_augment_transform(
                "v0", dict(translate_const=int(224 * 0.45))
            )
        else:
            self.auto_augment = None

        if self.cfg.DATA.RANDAUGMENT:
            self.rand_augment = rand_augment_transform(
                "rand-m20-n2", dict(translate_const=int(224 * 0.45))
            )
        else:
            self.rand_augment = None

    def _construct_loader(self):
        """
        Construct the video loader.
        """

        self._video_names = []
        self._labels = []
        self._path_to_videos = []
        self.h5_file = None

        # Loading label names.
        with PathManager.open(
            os.path.join(
                self.cfg.DATA.PATH_TO_DATA_DIR,
                f"{self.mode}_videofolder.txt"
                if self.mode in ["train", "val"]
                else "val_videofolder.txt",
            ),
            "r",
        ) as f:
            # Loading labels and names
            for line in f:
                splitline = line.split(" ")
                self._video_names.append(splitline[0])
                self._labels.append(int(splitline[2]))
                vpath = os.path.join(
                    self.cfg.DATA.PATH_TO_DATA_DIR,
                    "seq_h5",
                    splitline[0] + ".h5",
                )
                self._path_to_videos.append(vpath)

        # Extend self when self._num_clips > 1 (during testing).
        self._path_to_videos = list(
            chain.from_iterable(
                [[x] * self._num_clips for x in self._path_to_videos]
            )
        )
        self._video_names = list(
            chain.from_iterable(
                [[x] * self._num_clips for x in self._video_names]
            )
        )
        self._labels = list(
            chain.from_iterable([[x] * self._num_clips for x in self._labels])
        )
        self._spatial_temporal_idx = list(
            chain.from_iterable(
                [
                    range(self._num_clips)
                    for _ in range(len(self._path_to_videos))
                ]
            )
        )
        logger.info(
            "Something-Something V2 dataloader constructed "
            " (size: {}) ".format(len(self._path_to_videos))
        )

    def get_seq_frames(self, video_binary, temporal_sample_index):
        """
        Given the video index, return the list of sampled frame indexes.
        Args:
            index (int): the video index.
        Returns:
            seq (list): the indexes of frames of sampled from the video.
        """
        num_frames = self.cfg.DATA.NUM_FRAMES
        video_length = len(video_binary)

        seg_size = float(video_length - 1) / num_frames
        seq = []
        for i in range(num_frames):
            start = int(np.round(seg_size * i))
            end = int(np.round(seg_size * (i + 1)))
            if self.mode == "train":
                seq.append(random.randint(start, end))
            elif self.mode == "val":
                seq.append((start + end) // 2)
            elif self.mode == "test":
                if temporal_sample_index == 0:
                    seq.append((start + end) // 2)
                elif temporal_sample_index == 1:
                    seq.append(start)
                elif temporal_sample_index == 2:
                    seq.append(end)

        return seq

    def get_seq_frames_dense(self, video_binary, temporal_sample_index):
        """
        Given the video index, return the list of sampled frame indexes.
        Args:
            index (int): the video index.
        Returns:
            seq (list): the indexes of frames of sampled from the video.
        """
        num_frames = self.cfg.DATA.NUM_FRAMES
        video_length = len(video_binary)

        sampling_fps = num_frames * self.cfg.DATA.SAMPLING_RATE

        start_idx, end_idx = get_start_end_idx(
            video_length,
            sampling_fps,
            temporal_sample_index,
            self.cfg.TEST.NUM_ENSEMBLE_VIEWS,
        )
        index = torch.linspace(start_idx, end_idx, num_frames)
        index = torch.clamp(index, 0, video_length - 1).long().tolist()

        return index

    def __getitem__(self, index):
        """
        Given the video index, return the list of frames, label, and video
        index if the video frames can be fetched.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (int): the label of the current video.
            index (int): the index of the video.
        """

        temporal_sample_index = None
        if self.mode in ["train", "val"]:
            # -1 indicates random sampling.
            spatial_sample_index = -1
            min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
            max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
            crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
        elif self.mode in ["test"]:
            temporal_sample_index = (
                self._spatial_temporal_idx[index]
                // self.cfg.TEST.NUM_SPATIAL_CROPS
            )
            # spatial_sample_index is in [0, 1, 2]. Corresponding to left,
            # center, or right if width is larger than height, and top, middle,
            # or bottom if height is larger than width.
            spatial_sample_index = (
                self._spatial_temporal_idx[index]
                % self.cfg.TEST.NUM_SPATIAL_CROPS
            )
            min_scale, max_scale, crop_size = [self.cfg.DATA.TEST_CROP_SIZE] * 3
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale, crop_size}) == 1
        else:
            raise NotImplementedError(
                "Does not support {} mode".format(self.mode)
            )

        hdf5_video_key = self._video_names[index]
        single_h5 = os.path.join(
            self.cfg.DATA.PATH_TO_DATA_DIR, "Something-Something-v2-frames.h5"
        )
        if os.path.isfile(single_h5):
            if self.h5_file is None:
                self.h5_file = h5py.File(single_h5, "r")
            video_binary = self.h5_file[hdf5_video_key]
        else:
            video_binary = h5py.File(self._path_to_videos[index], "r")[
                hdf5_video_key
            ]

        label = self._labels[index]

        seq = self.get_seq_frames(video_binary, temporal_sample_index)

        data = [video_binary[i] for i in seq]
        frames = []
        for raw_frame in data:
            frames.append(
                np.asarray(Image.open(io.BytesIO(raw_frame)).convert("RGB"))
            )

        if self.mode == "train":
            if self.auto_augment is not None:
                frames = self.auto_augment(frames)
            elif self.rand_augment is not None:
                frames = self.rand_augment(frames)

        frames = torch.as_tensor(np.stack(frames))

        # Perform color normalization.
        frames = utils.tensor_normalize(
            frames, self.cfg.DATA.MEAN, self.cfg.DATA.STD
        )

        # T H W C -> C T H W.
        frames = frames.permute(3, 0, 1, 2)
        # Perform data augmentation.
        frames, was_flipped = utils.spatial_sampling(
            frames,
            spatial_idx=spatial_sample_index,
            min_scale=min_scale,
            max_scale=max_scale,
            crop_size=crop_size,
            random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
            inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
        )

        # Apply rand erasing
        if self.mode == "train":
            if self.random_erasing is not None:
                frames = self.random_erasing(frames)

        seq = torch.tensor(seq)
        frames = utils.pack_pathway_output(self.cfg, frames, seq)
        return frames, label, index, {}

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._path_to_videos)
