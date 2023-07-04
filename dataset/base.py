import random
import csv
import numpy as np
import torch
import torch.utils.data as torchdata
from torchvision import transforms
import torchaudio
import librosa
from PIL import Image
import soundfile as sf
import torchvision.transforms  as T
from . import video_transforms as vtransforms

MUSIC11_CLASSES = ["accordion", "acoustic_guitar", "cello", "clarinet",
            "erhu", "flute", "saxophone", "trumpet", "tuba", "violin", "xylophone" ]

class BaseDataset(torchdata.Dataset):
    def __init__(self, csv_path, params={}, split='val', debug= False, seed=None):
        self.debug = debug
        
        # params
        self.num_frames = params["num_frames"]#opt.num_frames
        self.imgSize = params["imgSize"]
        self.stride_frames = params["stride_frames"]
        self.max_sample = -1
        self.one_frame = params["one_frame"]
        # self.frameRate = opt.frameRate
        self.load_clips = params["load_clips"]
        self.clip_len = params["clip_len"]
        
        self.audRate = params["audRate"]
        self.audLen = params["audLen"]
        self.audSec = 1. * self.audLen / self.audRate
        self.binary_mask = params["binary_mask"]
        # STFT params
        self.use_spec = params["use_spec"]
        self.log_freq = params["log_freq"]
        self.stft_frame = params["stft_frame"]
        self.stft_hop = params["stft_hop"]
        self.HS = self.stft_frame // 2 + 1
        self.WS = (self.audLen + 1) // self.stft_hop

        self.num_mix = params["num_mix"]
        self.classes = MUSIC11_CLASSES
        self.rate_dc = params['rate_dc']
        self.rate_sc = params['rate_sc']
        self.rate_sv = params['rate_sv']
        self.margin  = params['margin']
        self.max_silent = params['max_silent']
        self.class_int_map = {k:v for v, k in enumerate(self.classes)}

        self.split = split
        self.seed = seed if seed is not None else params["seed"]
        random.seed(self.seed)

        # initialize video transform
        self._init_vtransform()

        # list_samples can be a python list or a csv file of list
        if isinstance(csv_path, str):
            # self.list_samples = [x.rstrip() for x in open(list_samples, 'r')]
            self.list_samples = []
            for row in csv.reader(open(csv_path, 'r'), delimiter=','):
                if len(row) < 2:
                    continue
                self.list_samples.append(row)
        elif isinstance(csv_path, list):
            self.list_samples = csv_path
        else:
            raise('Error list_samples!')
        print(f"Length of dataset: {len(self.list_samples)}")
        # Make dict samples:
        self.dict_samples = self.make_dict_samples(self.list_samples)
        
        # Repeat list samples.
        if self.split == 'train':
            self.list_samples *= params["train_repeat"]
            if not self.debug:
                random.shuffle(self.list_samples)
        else:
            self.list_samples *= params["val_repeat"]

        if self.max_sample > 0:
            self.list_samples = self.list_samples[0:self.max_sample]

        num_sample = len(self.list_samples)
        assert num_sample > 0
        # print('# samples: {}'.format(num_sample))

    def __len__(self):
        return len(self.list_samples)

    # video transform funcs
    def _init_vtransform(self):
        transform_list = []
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        if self.split == 'train':
            transform_list.append(vtransforms.Resize(int(self.imgSize * 1.1), T.InterpolationMode.BICUBIC))
            transform_list.append(vtransforms.RandomCrop(self.imgSize))
            transform_list.append(vtransforms.RandomHorizontalFlip())
        else:
            transform_list.append(vtransforms.Resize(self.imgSize, T.InterpolationMode.BICUBIC))
            transform_list.append(vtransforms.CenterCrop(self.imgSize))

        transform_list.append(vtransforms.ToTensor())
        transform_list.append(vtransforms.Normalize(mean, std))
        transform_list.append(vtransforms.Stack())
        self.vid_transform = transforms.Compose(transform_list)

    # image transform funcs, deprecated
    def _init_transform(self):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        if self.split == 'train':
            self.img_transform = transforms.Compose([
                transforms.Scale(int(self.imgSize * 1.2)),
                transforms.RandomCrop(self.imgSize),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])
        else:
            self.img_transform = transforms.Compose([
                transforms.Scale(self.imgSize),
                transforms.CenterCrop(self.imgSize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])

    def _load_frames(self, paths):
        frames = []
        for path in paths:
            frames.append(self._load_frame(path))
        frames = self.vid_transform(frames)
        return frames

    def _load_frame(self, path):
        img = Image.open(path).convert('RGB')
        return img

    def _stft(self, audio):
        spec = librosa.stft(
            audio, n_fft=self.stft_frame, hop_length=self.stft_hop)
        amp = np.abs(spec)
        phase = np.angle(spec)
        return torch.from_numpy(amp), torch.from_numpy(phase)

    def _load_audio_file(self, path, center_t):
        assert path.endswith(".wav")
        offset = center_t - self.margin - self.audSec/2
        duration = self.margin * 2 + self.audSec
        audio_raw, rate = librosa.load(path, sr=self.audRate, mono=True, offset=offset, duration = duration)
        return audio_raw, rate

    def _load_audio(self, path, center_t, nearest_resample=False):
        audio = np.zeros(self.audLen, dtype=np.float32)
        audio_raw, rate = self._load_audio_file(path, center_t = center_t)
        center_idx = int((self.margin + self.audSec/2) * self.audRate)
        start = max(0, center_idx - self.audLen // 2)
        end = min(len(audio_raw), center_idx + self.audLen // 2 + self.audLen % 2)
        audio[:end-start] = audio_raw[start: end]

        # Augmentation:
        if self.split == 'train':
            scale = random.random() + 0.5     # 0.5-1.5
            audio *= scale
        audio[audio > 1.] = 1.
        audio[audio < -1.] = -1.

        # assert not np.all(audio==0)
        return audio

    def _mix_n_and_stft(self, audios, audio_mix):
        N = len(audios)
        mags = [None for n in range(N)]

        # STFT
        amp_mix, phase_mix = self._stft(audio_mix)
        for n in range(N):
            ampN, _ = self._stft(audios[n])
            mags[n] = ampN.unsqueeze(0)

        # to tensor
        # audio_mix = torch.from_numpy(audio_mix)
        for n in range(N):
            audios[n] = torch.from_numpy(audios[n])

        return amp_mix.unsqueeze(0), mags, phase_mix.unsqueeze(0)

    def dummy_mix_data(self, N):
        frames = [None for n in range(N)]
        audios = [None for n in range(N)]
        mags = [None for n in range(N)]

        amp_mix = torch.zeros(1, self.HS, self.WS)
        phase_mix = torch.zeros(1, self.HS, self.WS)

        for n in range(N):
            frames[n] = torch.zeros(
                3, self.num_frames, self.imgSize, self.imgSize)
            audios[n] = torch.zeros(self.audLen)
            mags[n] = torch.zeros(1, self.HS, self.WS)

        return amp_mix, mags, frames, audios, phase_mix

    def make_dict_samples(self, list_samples):
        dict_samples = {}
        for sample in list_samples:
            cls_name = sample[-1]
            if cls_name not in dict_samples:
                dict_samples[cls_name] = [sample]
            else:
                dict_samples[cls_name].append(sample)
        return dict_samples