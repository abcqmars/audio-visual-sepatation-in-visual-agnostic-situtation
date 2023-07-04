import os
import torch
import random
import numpy as np
from .base import BaseDataset
from PIL import Image
import soundfile as sf
from mmaction.datasets.pipelines import Compose
from utils import save_video, combine_video_audio


def get_id_converter(format=1):
    """
    >>> cvt = get_id_converter()
    >>> cvt(15)
    '000015'
    """
    # fn: frame name.
    if format == 1:
        def fmt(id, N_0 = 6):
            # int: 123 -> str: "000123"
            return str(id).zfill(N_0)
    return fmt

def torch2numpy(datas):
    if isinstance(datas, list):
        for i, data in enumerate(datas):
            datas[i] = torch2numpy(data)
    elif isinstance(datas, torch.Tensor):
        return np.array(datas)
    elif isinstance(datas, np.ndarray):
        return datas
    else:
        assert False, f"{type(datas)} can not be converted into torch.tensor"
    return datas

class Music11Formater:
    @staticmethod
    def format(*args):
        cvter = get_id_converter()
        if len(args) == 1:
            return cvter(args[0]) + ".jpg"
        elif len(args) == 2:
            # e.g. first: 'x', second: 9
            return cvter(args[1]) + f"_{args[0]}"+ ".jpg"
        else:
            assert False, f"Length {len(args)} of input arguments is not supported."


class MUSICMixDataset(BaseDataset):
    def __init__(self, csv_path, params, debug=False, random_sample = False, vis_data = False, **kwargs):
        super(MUSICMixDataset, self).__init__(
            csv_path, params, debug = debug, **kwargs)
        self.random_sample = random_sample
        self.vis_data = vis_data
        
    def get_samples(self, index, option = "dc"):
        """
        info: info list of samples[index]
        """
        info_list = []
        info_list.append(self.list_samples[index])
        _, _, _, _, _, sound_cls = self.list_samples[index]
        # sample from different classes.
        if option == "dc":
            left_classes = self.classes.copy()
            for i in range(self.num_mix - 1):
                left_classes.remove(sound_cls)
                slct_cls = random.choice(left_classes)
                info_list.append(random.choice(self.dict_samples[slct_cls]))
        
        # sample from the same class.
        elif option == "sc":
            for i in range(self.num_mix - 1):
                info_list.append(random.choice(self.dict_samples[sound_cls]))
        
        # sample from the same video.
        elif option == "sv":
            for i in range(self.num_mix - 1):
                info_list.append(self.list_samples[index])
        
        elif option == "random":
            for i in range(self.num_mix - 1):
                indexN = random.randint(0, len(self.list_samples)-1)
                info_list.append(self.list_samples[indexN])

        elif option == "vis1":
            info_list = [random.choice(self.dict_samples["cello"])]
            for i in range(self.num_mix - 1):
                info_list.append(random.choice(self.dict_samples["flute"]))
        
        assert self.num_mix == len(info_list)
        return info_list

    
    def get_audios(self, infos):
        audios = []

        center_times = []
        for n in range(self.num_mix):
            apath, _, num_f, fps, a_len, _ = infos[n]
            a_len = float(a_len)
            act_len = min(int(num_f)/float(fps), a_len)
            for j in range(10):
                # assert (act_len - self.margin - self.audSec/2) > (self.margin + self.audSec/2), f"acti_len: {act_len} for {apath}"
                end = act_len - self.margin - self.audSec/2
                start = self.margin + self.audSec/2
                if start>end:
                    end = act_len - self.audSec/2
                    start = self.audSec/2
                t = random.uniform(0 + start, end)
                aud = self._load_audio(apath, t)
                # is_silent = np.all(aud==0) if self.split == "train" else ((np.abs(aud) < 0.001).sum()/self.audLen) < self.max_silent
                if self.split == 'train':
                    is_silent = np.all(aud==0)
                else:
                    is_silent = ((np.abs(aud) < 0.001).sum()/self.audLen) > self.max_silent
                if not is_silent: # ((np.abs(aud) < 0.001).sum()/self.audLen) < self.max_silent:
                    center_times.append(t)
                    audios.append(aud/self.num_mix)
                    break
                if j == 9:
                    center_times.append(t)
                    audios.append(aud/self.num_mix)
                    print(f"Load {apath} failed.")
                    #E assert False, "Load audio 10 times, silent still fail."
        mixtures = np.asarray(audios).sum(axis=0)

        assert len(audios) == self.num_mix and len(center_times) == self.num_mix
        return audios, mixtures, center_times

    def get_frames(self, infos, center_times):
        frames_ls = []
        time_shifts = []
        for n in range(self.num_mix):
            frames_pth = []
            center_t = center_times[n]
            _, fpath, num_f, fps, _, _ = infos[n]
            fps = float(fps)
            num_f = float(num_f)
            center_idx = round(center_t * fps)
            time_shifts.append(center_t - center_idx/fps)
            if self.one_frame:
                idx_shift = random.randint(int(-1*self.stride_frames), int(1*self.stride_frames))
                frames_pth.append(os.path.join(
                        fpath,
                        '{:06d}.jpg'.format(center_idx + idx_shift)))
            else:
                for i in range(self.num_frames):
                    idx_offset = (i - self.num_frames // 2) * self.stride_frames
                    frames_pth.append(os.path.join(
                            fpath,
                            '{:06d}.jpg'.format(center_idx + idx_offset)))
            frames_ls.append(self._load_frames(frames_pth))
        assert len(frames_ls)==self.num_mix and len(time_shifts)==self.num_mix
        return frames_ls, time_shifts

    def make_mmcv_dict(self, fps, total_f, center_time, frame_dir, filename_tmpl, modality):
        return  {
                'clip_sec': self.audSec,
                'fps':  fps,
                'total_f': total_f,
                'center_time': center_time,
                'frame_dir': frame_dir,
                'filename_tmpl': filename_tmpl,
                'modality': modality,
                'offset': 0,
                }
    
    def get_pipeline(self):
        img_norm_cfg = dict(
            mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
        ret_keys = ['imgs']
        if self.debug:
            ret_keys = ['imgs', 'imgs_org']
        if self.split == 'train':
            pipeline = [
                dict(type='CustomSampleInds', clip_len=self.clip_len),
                dict(type='RawFrameDecode_Plus', num_frames = self.num_frames, stride = self.stride_frames),
                dict(type='Resize', scale=(-1, 256)),
                dict(type='RandomResizedCrop'),
                dict(type='Resize', scale=(224, 224), keep_ratio=False),
                dict(type='Flip', flip_ratio=0.5),
                dict(type='Normalize', **img_norm_cfg),
                dict(type='FormatShape', input_format='NCHW'),
                dict(type='Collect', keys=ret_keys, meta_keys=[]),
                dict(type='ToTensor', keys=ret_keys)
            ]
        elif self.split == 'val':
            pipeline = [
                dict(type='CustomSampleInds', clip_len=self.clip_len),
                dict(type='RawFrameDecode_Plus', num_frames = self.num_frames, stride = self.stride_frames),
                dict(type='Resize', scale=(-1, 256)),
                dict(type='CenterCrop', crop_size=224),
                dict(type='Normalize', **img_norm_cfg),
                dict(type='FormatShape', input_format='NCHW'),
                dict(type='Collect', keys=ret_keys, meta_keys=[]),
                dict(type='ToTensor', keys=ret_keys)
            ]
        return Compose(pipeline)

    def get_frames_mmcv(self, infos, center_times):
        """Load clip frames, optical flows, center frames by mmcv backend.

        Args:
            infos (list): 
            center_times (float): 

        Returns:
            center_frames: (list(np.narray)): 
        """
        center_frames = []
        clips = []
        pipeline = self.get_pipeline()
        if self.debug:
            org_imgs = []
        for n in range(self.num_mix):
            center_t = center_times[n]
            _, fpath, num_f, fps, _, _ = infos[n]
            data_dict = self.make_mmcv_dict(float(fps), num_f, center_t, fpath, Music11Formater, 'RGB')
            data_dict = pipeline(data_dict)
            # print(data_dict['imgs'].shape) # T, C, H, W
            center_frames.append(data_dict['imgs'][self.clip_len:].permute(1, 0, 2, 3))
            clips.append(data_dict['imgs'][:self.clip_len].permute(1, 0, 2, 3))
            org_imgs.append(data_dict['imgs_org']) if self.debug else None
        if self.debug:
            return center_frames, clips, org_imgs
        
        return center_frames, clips

    def get_ids_labels(self, infos, index, center_times):
        class_ls = []
        class_id_ls = []
        id_ls = []
        center_times = [str(round(t)) for t in center_times ]
        for n in range(self.num_mix):
            apath, _, _, _, _, sound_cls = infos[n]
            class_idx = self.class_int_map[sound_cls]
            class_ls.append(str(class_idx))
            class_id_ls.append(class_idx)
            id_ls.append(os.path.basename(apath).split('.')[0][:4])
        smp_name = str(index) + "_" + "cls" + "_".join(class_ls)+ '_'+ "ids" + '_'.join(id_ls) + "_" + "ct" + "_".join(center_times)
        return smp_name, torch.tensor(class_id_ls)

    def make_tensor(self, datas):
        if isinstance(datas, list):
            for i, data in enumerate(datas):
                datas[i] = self.make_tensor(data)
        elif isinstance(datas, np.ndarray):
            return torch.tensor(datas)
        elif isinstance(datas, torch.Tensor):
            return datas
        else:
            assert False, f"{type(datas)} can not be converted into torch.tensor"
        
        return datas

    def save_sample(self, inds, save_dir = "/mnt/data2/he/src/Conv_TasNet/debug"):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for ind in inds:
            data = self[ind]
            id = data['id']
            # Save mixtures.
            sf.write(os.path.join(save_dir, id + "_mix.wav"), torch2numpy(data['audio_mix']), self.audRate)
            # Save audios.
            for i, audio in enumerate(torch2numpy(data['audios'])):
                audio_path = os.path.join(save_dir, id + f"_aud_{i}.wav")
                sf.write(audio_path, audio, self.audRate)
                clip_frames = np.array(data['frames_org'][i][:self.clip_len])
                video_path = os.path.join(save_dir, id + f"_vid_{i}.mp4")
                save_video(video_path, clip_frames, fps=self.clip_len/self.audSec)
                combine_video_audio(video_path, audio_path, os.path.join(save_dir, id + f"_AV_{i}.mp4"))
        
    def __getitem__(self, index):
        # Return as dict.
        ret_dict = {}
        
        # Set random seed.
        random.seed(index)
        
        if self.random_sample:
            infos = self.get_samples(index, option = "random")
        elif self.vis_data:
            infos = self.get_samples(index, option = self.vis_data)
        else:
            # Get samples from given strategy:
            if random.random() < self.rate_dc:
                infos = self.get_samples(index, option = "dc")
            elif random.random() < self.rate_dc + self.rate_sc:
                infos = self.get_samples(index, option = "sc")
            elif random.random() < self.rate_dc + self.rate_sc + self.rate_sv:
                infos = self.get_samples(index, option = "sv")

        # Get truncated audio.
        # * the silent area < 0.2 of total.
        audios, mixture, center_times = self.get_audios(infos)

        # Get frames of videos.
        if self.load_clips:
            if self.debug:
                frames, clips, org_frames = self.get_frames_mmcv(infos, center_times)
                ret_dict['clips'] = clips
                ret_dict['frames_org'] = org_frames
                ret_dict['center_times'] = center_times
            else:
                frames, clips = self.get_frames_mmcv(infos, center_times)
                ret_dict['clips'] = clips
        else: 
            frames, time_shifts =  self.get_frames(infos, center_times)

        # Get name of each mixture.
        smp_name, class_id_ls = self.get_ids_labels(infos, index, center_times)

        if self.use_spec:
            mag_mix, mags, phase_mix = self._mix_n_and_stft(audios, mixture)
            mag_mix, mags, phase_mix = self.make_tensor([mag_mix, mags, phase_mix])
            ret_dict['mag_mix'] =  mag_mix
            ret_dict['mags'] = mags
            ret_dict['phase_mix'] = phase_mix

        audios, mixture, frames = self.make_tensor([audios, mixture, frames])

        ret_dict['infos'] = infos
        ret_dict['audios'] = audios
        ret_dict['audio_mix'] = mixture
        ret_dict['frames'] = frames
        ret_dict['id'] = smp_name
        ret_dict['class'] = class_id_ls
        
        return ret_dict
