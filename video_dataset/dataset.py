#!/usr/bin/env python

import os, sys
from typing import Optional
import av
import io
import numpy as np
import glob
from pathlib import Path
from PIL import Image

import torch
from torchvision import transforms

from .transform import create_random_augment, random_resized_crop
from .drive_to_local_cache import resolve_cached_path

import random
from collections import defaultdict

class VideoDataset(torch.utils.data.Dataset):

    def __init__(
        self, frames_available: int, list_path: str, data_root: str,
        num_spatial_views: int, num_temporal_views: int, random_sample: bool,
        num_frames: int, sampling_rate: int, spatial_size: int,
        mean: torch.Tensor, std: torch.Tensor,
        auto_augment: Optional[str] = None, interpolation: str = 'bicubic',
        mirror: bool = False, n_shots: int = -1,
        local_cache_dir: Optional[str] = None,
    ):
        self.frames_available = frames_available
        self.data_root = data_root
        self.local_cache_dir = (local_cache_dir or "").strip() or None
        self.interpolation = interpolation
        self.spatial_size = spatial_size
        self.n_shots = n_shots
        self.list_path = list_path
        self.mean, self.std = mean, std
        self.num_frames, self.sampling_rate = num_frames, sampling_rate

        if random_sample:
            assert num_spatial_views == 1 and num_temporal_views == 1
            self.random_sample = True
            self.mirror = mirror
            self.auto_augment = auto_augment
        else:
            assert auto_augment is None and not mirror
            self.random_sample = False
            self.num_temporal_views = num_temporal_views
            self.num_spatial_views = num_spatial_views

        if self.n_shots != -1:
            print('N-shots: ', self.n_shots)
            self.data_list = self.sampleNshots()
        else:
            with open(self.list_path, encoding='utf-8') as f:
                self.data_list = f.read().splitlines()
        #print(self.data_list)
        print(len(self.data_list))


    def sampleNshots(self):
        # return n samples from each class
        n = self.n_shots
        class_samples = defaultdict(list)

        # Read the input file and populate the dictionary
        with open(self.list_path, 'r', encoding='utf-8') as f:
            for line in f:
                filepath, class_label = line.strip().split('\t')
                class_samples[class_label].append(filepath)

        # List to store randomly selected samples in (filepath, class) format
        selected_samples = []

        # Randomly select n samples from each class
        for class_label, files in class_samples.items():
            if len(files) <= n:
                selected_samples.extend([(filepath, class_label) for filepath in files])
            else:
                selected_files = random.sample(files, n)
                selected_samples.extend([(filepath, class_label) for filepath in selected_files])


        # format selected samples in (filepath, class) format
        selected_list =[]
        for filepath, class_label in selected_samples:
            selected_list.append(f"{filepath}\t{class_label}")

        return selected_list


    def __len__(self):
        return len(self.data_list)
    

    def __getitem__(self, idx):
        try:
            line = self.data_list[idx]
            #print(line)
            parts = line.strip().split('\t')
            relpath, label = parts[0], int(parts[1])  #line.split(' ') Hamzah
            if self.local_cache_dir:
                path = resolve_cached_path(
                    self.frames_available, self.data_root, relpath, self.local_cache_dir
                )
            else:
                path = os.path.join(self.data_root, relpath)
            #print('============== ', path , '**** ', self.frames_available)
            #print('*********** ', len(self.data_list),' **** ',label)
        except:
            print('Error with: ', line)
        if self.frames_available:
            # Use pathlib.glob — handles Unicode (Arabic/Urdu) folder names on Windows
            frame_dir = Path(path).parent / Path(path).stem
            framesNames = sorted(frame_dir.glob("*.png"))
            if not framesNames:
                framesNames = sorted(frame_dir.glob("*.jpg"))

            # Sample indices FIRST, then load only the needed frames (not all frames)
            if self.random_sample:
                frame_idx = self._random_sample_frame_idx(len(framesNames))
            else:
                # For val/test: evenly spaced indices across all frames
                total = len(framesNames)
                seg_len = (self.num_frames - 1) * self.sampling_rate + 1
                if total < seg_len:
                    frame_idx = self.frames_downUpSamples(total, self.num_frames)
                else:
                    mid_start = (total - seg_len) // 2
                    frame_idx = list(range(mid_start, mid_start + self.num_frames * self.sampling_rate, self.sampling_rate))

            frames = []
            for i in frame_idx:
                if i < len(framesNames):
                    frames.append(np.array(Image.open(str(framesNames[i])).convert('RGB')))

        else:
            # Full-file PyAV decode every __getitem__ — costly on slow storage (e.g. Drive). Prefer
            # local staging, frames_available=1, or a future refactor: seek/sample so only the needed
            # temporal span is decoded.
            #print('[Hamzah] Path :', path)
            container = av.open(path)
            frames = {}
            for frame in container.decode(video=0):
                frames[frame.pts] = frame
            container.close()
            frames = [frames[k] for k in sorted(frames.keys())]
            #print(len(frames))
        #print('[Hamzah] Path :', path, ' : ', len(frames))
        if self.random_sample:
            try:
                if not self.frames_available:
                    # Video-decode path: frames are av or PIL objects, need sampling + conversion
                    frame_idx = self._random_sample_frame_idx(len(frames))
                    frames = [np.array(frames[x]) if isinstance(frames[x], Image.Image) else frames[x].to_rgb().to_ndarray() for x in frame_idx]
                # frames_available path: already sampled numpy arrays from above
                frames = torch.as_tensor(np.stack(frames)).float() / 255.

                if self.auto_augment is not None:
                    aug_transform = create_random_augment(
                        input_size=(frames.size(1), frames.size(2)),
                        auto_augment=self.auto_augment,
                        interpolation=self.interpolation,
                    )
                    frames = frames.permute(0, 3, 1, 2) # T, C, H, W
                    frames = [transforms.ToPILImage()(frames[i]) for i in range(frames.size(0))]
                    frames = aug_transform(frames)
                    frames = torch.stack([transforms.ToTensor()(img) for img in frames])
                    frames = frames.permute(0, 2, 3, 1)

                frames = (frames - self.mean) / self.std
                frames = frames.permute(3, 0, 1, 2) # C, T, H, W
                frames = random_resized_crop(
                    frames, self.spatial_size, self.spatial_size,
                )
            except Exception as e:
                print('[Hamzah-1] Path :', path, ' frames:', len(frames), 'error:', e)
                # Return a zero tensor so the collator doesn't crash
                frames = torch.zeros(3, self.num_frames, self.spatial_size, self.spatial_size)
        else:
            try:
                if self.frames_available:
                    # frames are already numpy arrays (H, W, C) from lazy loading above
                    frames = torch.as_tensor(np.stack(frames)).float() / 255.
                else:
                    frames = [x.to_rgb().to_ndarray() for x in frames]
                    frames = torch.as_tensor(np.stack(frames)).float() / 255.
                frames = (frames - self.mean) / self.std
                frames = frames.permute(3, 0, 1, 2) # C, T, H, W
            except Exception as e:
                print('[Hamzah-2] Path :', path, 'error:', e)
                frames = torch.zeros(3, self.num_frames, self.spatial_size, self.spatial_size)
                      
            if isinstance(frames, torch.Tensor) and frames.shape[-2] == self.spatial_size and frames.shape[-1] == self.spatial_size:
                # already 224×224 (zero-tensor fallback) — skip resize/crop
                return frames, label
            if frames.size(-2) < frames.size(-1):
                new_width = frames.size(-1) * self.spatial_size // frames.size(-2)
                new_height = self.spatial_size
            else:
                new_height = frames.size(-2) * self.spatial_size // frames.size(-1)
                new_width = self.spatial_size
            frames = torch.nn.functional.interpolate(
                frames, size=(new_height, new_width),
                mode='bilinear', align_corners=False,
            )

            frames = self._generate_spatial_crops(frames)
            frames = sum([self._generate_temporal_crops(x) for x in frames], [])
            if len(frames) > 1:
                frames = torch.stack(frames)

        #print('[Hamzah] Path :', path, " ", frames.shape) 
        return frames, label


    def _generate_temporal_crops(self, frames):
        seg_len = (self.num_frames - 1) * self.sampling_rate + 1
        if frames.size(1) < seg_len:
            frames = torch.cat([frames, frames[:, -1:].repeat(1, seg_len - frames.size(1), 1, 1)], dim=1)
        slide_len = frames.size(1) - seg_len

        crops = []
        for i in range(self.num_temporal_views):
            if self.num_temporal_views == 1:
                st = slide_len // 2
            else:
                st = round(slide_len / (self.num_temporal_views - 1) * i)

            crops.append(frames[:, st: st + self.num_frames * self.sampling_rate: self.sampling_rate])
        
        return crops


    def _generate_spatial_crops(self, frames):
        if self.num_spatial_views == 1:
            assert min(frames.size(-2), frames.size(-1)) >= self.spatial_size
            h_st = (frames.size(-2) - self.spatial_size) // 2
            w_st = (frames.size(-1) - self.spatial_size) // 2
            h_ed, w_ed = h_st + self.spatial_size, w_st + self.spatial_size
            return [frames[:, :, h_st: h_ed, w_st: w_ed]]

        elif self.num_spatial_views == 3:
            assert min(frames.size(-2), frames.size(-1)) == self.spatial_size
            crops = []
            margin = max(frames.size(-2), frames.size(-1)) - self.spatial_size
            for st in (0, margin // 2, margin):
                ed = st + self.spatial_size
                if frames.size(-2) > frames.size(-1):
                    crops.append(frames[:, :, st: ed, :])
                else:
                    crops.append(frames[:, :, :, st: ed])
            return crops
        
        else:
            raise NotImplementedError()


    def _random_sample_frame_idx(self, len):
        frame_indices = []

        if self.sampling_rate < 0: # tsn sample
            seg_size = (len - 1) / self.num_frames
            for i in range(self.num_frames):
                start, end = round(seg_size * i), round(seg_size * (i + 1))
                frame_indices.append(np.random.randint(start, end + 1))
        elif self.sampling_rate * (self.num_frames - 1) + 1 >= len:
            # Hamzah: modified
            frame_indices = self.frames_downUpSamples(len, self.num_frames)
            #print(frame_indices)
            # for i in range(self.num_frames):
            #     print('i=', i, ' --- ', i * self.sampling_rate, ' Len ', len)
            #     #print(frame_indices)
            #     frame_indices.append(i * self.sampling_rate if i * self.sampling_rate < len else frame_indices[-1])
        else:
            start = np.random.randint(len - self.sampling_rate * (self.num_frames - 1))
            frame_indices = list(range(start, start + self.sampling_rate * self.num_frames, self.sampling_rate))

        return frame_indices

    def frames_downUpSamples(self, vidoeFrames, nFramesTarget):
        """ Adjust number of frames (eg 123) to nFramesTarget (eg 79)
        works also if originally less frames then nFramesTarget
        """
        
        if vidoeFrames == nFramesTarget: return range(nFramesTarget)

        # down/upsample the list of frames
        fraction = vidoeFrames / nFramesTarget
        index = [int(fraction * i) for i in range(nFramesTarget)]
         

        return index

class DummyDataset(torch.utils.data.Dataset):

    def __init__(self, frames_available: int, list_path: str, num_frames: int, num_views: int, spatial_size: int, n_shots: int):
        with open(list_path, encoding='utf-8') as f:
            self.len = len(f.read().splitlines())
        self.frames_available = frames_available
        self.num_frames = num_frames
        self.num_views = num_views
        self.spatial_size = spatial_size
        self.n_shots= n_shots

    def __len__(self):
        return self.len

    def __getitem__(self, _):
        shape = [3, self.num_frames, self.spatial_size, self.spatial_size]
        if self.num_views != 1:
            shape = [self.num_views] + shape
        return torch.zeros(shape), 0
