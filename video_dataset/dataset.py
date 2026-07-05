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
        presampled = False  # set True by paths that already select final frames (frames_available / partial decode)
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
            presampled = True  # frames_available already selected final frames above

        else:
            if self.random_sample:
                # Training path: decode ONLY the frames the sampler selects, without ever holding
                # the whole video in memory. Big videos previously forced a full-video decode every
                # step (the epoch bottleneck). On ANY problem this falls back to the original full
                # decode below, so a decode quirk degrades to "slow but correct", never wrong.
                try:
                    frames = self._decode_sampled_frames_train(path)  # list of RGB ndarrays, already sampled
                    presampled = True
                except Exception as _partial_e:
                    print('[partial-decode fallback] Path :', path, 'error:', _partial_e)
                    try:
                        # Full-decode fallback must itself be crash-proof: a corrupt/missing file
                        # must degrade to the zero-tensor path below, never kill the worker.
                        _container = av.open(path)
                        try:
                            _fr = {}
                            for frame in _container.decode(video=0):
                                _fr[frame.pts] = frame
                        finally:
                            _container.close()
                        frames = [_fr[k] for k in sorted(_fr.keys())]  # av.VideoFrame list, sampled below
                        presampled = False
                    except Exception as _fallback_e:
                        print('[full-decode also failed] Path :', path, 'error:', _fallback_e)
                        frames = []
                        presampled = True  # signal "already final" so the block below skips re-sampling an empty list
            else:
                # Val/test path: unchanged — multi-view temporal crops need the full frame sequence.
                container = av.open(path)
                frames = {}
                for frame in container.decode(video=0):
                    frames[frame.pts] = frame
                container.close()
                frames = [frames[k] for k in sorted(frames.keys())]
                presampled = False
        #print('[Hamzah] Path :', path, ' : ', len(frames))
        if self.random_sample:
            try:
                if not presampled:
                    # Full-decode fallback path: frames are av or PIL objects, need sampling + conversion
                    frame_idx = self._random_sample_frame_idx(len(frames))
                    frames = [np.array(frames[x]) if isinstance(frames[x], Image.Image) else frames[x].to_rgb().to_ndarray() for x in frame_idx]
                # presampled path (frames_available / partial decode): already sampled numpy arrays from above
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


    def _decode_sampled_frames_train(self, path):
        """Training-path partial decode: decode ONLY the frames the sampler selects, without ever
        holding the whole video in memory. Returns a list of RGB uint8 ndarrays for the sampled
        indices.

        Takes the fast path ONLY when the container reports a frame count (vstream.frames > 0) AND
        decoded frames arrive with strictly-increasing pts AND the decoded count matches; otherwise
        it raises and __getitem__ falls back to the original full decode. When the reported count is
        accurate (standard constant-frame-rate MP4) the selected frames are IDENTICAL to the
        full-decode path.

        Caveat: if a container reports a non-zero but UNDER-counted frame total (rare; some
        VFR/remuxed files), the fast path still returns a valid random temporal sample of the
        correct video with the correct label — training stays correct — but the sampling window may
        differ from the full-decode path (not bit-identical). This is an accepted trade: verifying
        the true count would require a full extra pass, negating the speedup."""
        container = av.open(path)
        try:
            vstream = container.streams.video[0]
            n = int(getattr(vstream, "frames", 0) or 0)
            if n <= 0:
                # No reliable frame-count metadata (common for some remuxed/VFR containers).
                # Don't pay for an extra full demux pass just to guess a count that may still be
                # wrong for these containers -- fall back straight to the proven full-decode path.
                raise ValueError("no reliable frame count metadata; skipping partial decode")

            # SAME sampler the full-decode path uses -> same indices -> same frames selected,
            # PROVIDED decoded-frame count actually equals n (checked via the loop below).
            frame_idx = [int(i) for i in self._random_sample_frame_idx(n)]
            if not frame_idx:
                raise ValueError("empty frame index")
            needed = set(frame_idx)
            max_idx = max(needed)

            # Positional index (cur) equals display index ONLY while decoded frames arrive in
            # strictly increasing presentation (pts) order with no duplicates -- the same
            # assumption the original path's "sorted(dict-by-pts)" relies on. Any duplicate,
            # out-of-order, or missing pts means our positional count could diverge from the
            # original's post-sort indexing, so we bail to the full-decode fallback rather than
            # risk returning different frames.
            collected = {}
            cur = 0
            prev_pts = None
            for frame in container.decode(video=0):
                pts = frame.pts
                if pts is None or (prev_pts is not None and pts <= prev_pts):
                    raise ValueError("missing/duplicate/non-monotonic pts during partial decode")
                prev_pts = pts
                if cur in needed:
                    collected[cur] = frame.to_rgb().to_ndarray()
                    if len(collected) == len(needed):
                        break
                cur += 1
                if cur > max_idx:
                    break

            if len(collected) != len(needed):
                # Metadata said n frames but decode produced fewer -> our sample indices don't
                # match what the full-decode path would have computed. Fall back for consistency
                # rather than silently return a different temporal sample.
                raise ValueError(
                    f"frame count mismatch: metadata said {n}, only decoded {cur} "
                    f"({len(collected)}/{len(needed)} needed frames found)"
                )
            return [collected[i] for i in frame_idx]
        finally:
            container.close()


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
