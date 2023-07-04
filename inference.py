from multiprocessing.context import ForkServerContext
import os

from cv2 import phase, transpose
import cv2
import csv
from sklearn import mixture
import torch
import random
import librosa
import numpy as np
from PIL import Image
import soundfile as sf
from arguments import ArgParser
import torch.nn.functional as F
import torchvision.transforms  as T
from imageio import imwrite as imsave
from models import ModelBuilder, activate
from torchvision import transforms
from utils import AverageMeter, \
    recover_rgb, magnitude2heatmap,\
    istft_reconstruction, warpgrid, \
    combine_video_audio, save_video, makedirs
import dataset.video_transforms as vtransforms
# Model Forward pass:
# * Takes single sample as input.
# * 

class NetWrapper(torch.nn.Module):
    def __init__(self, nets):
        super(NetWrapper, self).__init__()
        if len(nets) == 2:
            self.net_sound, self.net_frame = nets
            self.load_clips = False
        elif len(nets) == 3:
            self.net_sound, self.net_frame, self.net_motion = nets
            self.load_clips = True

    def prepare_inferdata(self, audios, frames, args):
        N = args.num_mix
        mag_mix, phase_mix = audios
        mag_mix = mag_mix + 1e-10
        # print(mag_mix.shape)
        B = mag_mix.size(0)
        T = mag_mix.size(3)
        
        grid_warp = torch.from_numpy(
            warpgrid(B, 256, T, warp=True)).to(args.device)
        mag_mix = F.grid_sample(mag_mix, grid_warp, align_corners=False)
        log_mag_mix = torch.log(mag_mix).detach()

        return frames, mag_mix, log_mag_mix, phase_mix
    
    def forward_ao(self, data, args):
        _, mag_mix, log_mag_mix, phase_mix = data
        feat_sound, meta = self.net_sound(log_mag_mix, None)
        pred_masks = activate(feat_sound, args.output_activation) # B x C x H x W
        pred_masks = torch.permute(pred_masks, (0, 2, 3, 1))  # B x H x W x C
        return {'pred_masks': [pred_masks[:, :, :, i].unsqueeze(1) for i in range(2)], 'mag_mix': mag_mix, 'phase_mix':phase_mix,  'maps':meta[1]} 

    def forward_av(self, data, args):
        # Info-guided pipeline.
        N = args.num_mix
        frames, mag_mix, log_mag_mix, phase_mix = data
        isDuet = len(frames) == 1
        if frames[0].dim() ==5:
            for n in range(len(frames)):
                frames[n] = frames[n][:, :, 0, :]
        # 1. Get the guidence information. => Bx1xC
        feat_frames = []
        if isDuet:
            feat_frames.append(self.net_frame.forward(frames[0], pool = args.not_pool_vis))
            feat_frames *= 2
        else:
            for n in range(N):
                tmp_feat = self.net_frame.forward(frames[n], pool = args.not_pool_vis)
                feat_frames.append(activate(tmp_feat, args.img_activation))

        # 2. Get mask with guidence -> BxCxHxW
        pred_masks = [None for n in range(N)]
        # vis_in = feat_frames if not mix_batch else [torch.cat(feat_frames, dim=3)]*2
        feat_sound, meta = self.net_sound(log_mag_mix, feat_frames)
        for n in range(N):
            pred_masks[n] = activate(feat_sound[:, n, :].unsqueeze(1), args.output_activation)
        match_loss = meta[0]
        return  {'pred_masks': pred_masks, 'mag_mix': mag_mix, 'phase_mix':phase_mix, 'match_loss': match_loss.reshape(1), 'maps':meta[1]}

    def share_forward(self, batch_data, args, use_vis):
        N = args.num_mix
        frames, mag_mix, log_mag_mix, phase_mix  = batch_data
        if frames[0].dim() ==5:
            for n in range(len(frames)):
                frames[n] = frames[n][:, :, 0, :]
        # frames:  [B, C, T, H, W] * S
        # gt_masks: [B, 1, F, T] * 2
        
        # Visual forward passing:
        if use_vis:
            # mix_frame = torch.cat(frames, dim=-1) # B, C, T, H, W * S
            feat_frame = self.net_frame.forward(frames[0], pool = args.not_pool_vis)
            feat_frame = activate(feat_frame, args.img_activation) # B, C, H, W * S
        else:
            # Generate All black signal.
            # mix_frame = torch.cat(frames, dim=-1)
            mix_frame = torch.zeros_like(frames[0])
            feat_frame = self.net_frame.forward(mix_frame, pool = args.not_pool_vis)
            feat_frame = activate(feat_frame, args.img_activation) # B, C, H, W * S
        
        # Audio forward passing:
        feat_sound, meta = self.net_sound(log_mag_mix, [feat_frame]) #B, 
        pred_masks = activate(feat_sound, args.output_activation) # B, S, F, T
        pred_masks = torch.permute(pred_masks, (0, 2, 3, 1))  # B, F, T, S
        
        if use_vis:
            match_loss = meta[0]
            att_maps = meta[1]
        else:
            match_loss = None
            att_maps = None
            match_loss = None
        
        return  {'pred_masks': [pred_masks[:, :, :, i].unsqueeze(1) for i in range(2)], 
             'mag_mix': mag_mix, 'match_loss':match_loss, 'phase_mix':phase_mix, 'maps': att_maps}

    def forward(self, mag_mix, frames, args, use_vis=True):
        data = self.prepare_inferdata(mag_mix, frames, args)
        if args.fusion_type == "share":
            return self.share_forward(data, args, use_vis)
        if use_vis:
            if args.fusion_type== "MixVis":
                out = forward_avmiximg(self, data, args)
            else:
                out = self.forward_av(data, args)
        else:
            out = self.forward_ao(data, args)
        return out

def forward_avmiximg(self, batch_data, args):
    N = args.num_mix

    # frames, _, mags, mag_mix, log_mag_mix, gt_masks, weight = batch_data
    frames, mag_mix, log_mag_mix, phase_mix = batch_data
    # frames:  [B, C, T, H, W] * S
    # gt_masks: [B, 1, F, T] * 2
    
    # Visual forward passing:
    mix_frame = torch.cat(frames, dim=-1) # B, C, T, H, W * S
    # print(mix_frame.shape)
    feat_frame = self.net_frame.forward_multiframe(mix_frame, pool = args.not_pool_vis)
    feat_frame = activate(feat_frame, args.img_activation) # B, C, H, W * S
    if frames[0].dim() ==5:
        for n in range(len(frames)):
            frames[n] = frames[n][:, :, 0, :]
    
    # Audio forward passing:
    feat_sound, meta = self.net_sound(log_mag_mix, [feat_frame]) # B
    pred_masks = activate(feat_sound, args.output_activation) # B, S, F, T
    pred_masks = torch.permute(pred_masks, (0, 2, 3, 1))  # B, F, T, S
    # gt_masks = torch.stack(gt_masks, dim=-1)[:, 0, :] # B, F, T, S

    return {'pred_masks': [pred_masks[:, :, :, i].unsqueeze(1) for i in range(2)], 'phase_mix':phase_mix, 'mag_mix': mag_mix,  'maps':meta[1]}


def forward_Exp1_plus(self, mag_mix, frames, args, use_vis=True):
    def forward_av(self, data, args):
        # Info-guided pipeline.
        N = args.num_mix
        frames, mag_mix, log_mag_mix, phase_mix = data
        assert len(frames) == 2
        isDuet = len(frames) == 1
        if frames[0].dim() ==5:
            for n in range(len(frames)):
                frames[n] = frames[n][:, :, 0, :]
        
        # 1. Get the guidence information. => Bx1xC
        feat_frames = []
        for n in range(N):
            tmp_feat = self.net_frame.forward(frames[n], pool = args.not_pool_vis)
            feat_frames.append(activate(tmp_feat, args.img_activation)) # Each B, C, H, W
        feat_frames = [torch.cat(feat_frames, dim=-1)] * 2 # B, C, H, W*2
        
        # 2. Get mask with guidence -> BxCxHxW
        pred_masks = [None for n in range(N)]
        # vis_in = feat_frames if not mix_batch else [torch.cat(feat_frames, dim=3)]*2
        feat_sound, meta = self.net_sound(log_mag_mix, feat_frames)
        for n in range(N):
            pred_masks[n] = activate(feat_sound[:, n, :].unsqueeze(1), args.output_activation)
        match_loss = meta[0]
        return  {'pred_masks': pred_masks, 'mag_mix': mag_mix, 'phase_mix':phase_mix, 'match_loss': match_loss.reshape(1), 'maps':meta[1]}
    data = self.prepare_inferdata(mag_mix, frames, args)
    if use_vis:
        out = forward_av(self, data, args)
    else:
        assert False
    return out

def get_audfuncs():
    audSec = 6
    audLen = 65535
    margin = 0
    audRate = 11025
    random.seed(8)# 4

    def get_wav(a_path, num_f, fps, a_len):
        act_len = min(int(num_f)/float(fps), float(a_len))
        end = act_len - margin - audSec/2
        start = margin + audSec/2
        c_t = random.uniform(start, end)
        wav = np.zeros(audLen, dtype=np.float32)
        offset = c_t - margin - audSec/2
        duration = margin * 2 + audSec
        audio_raw, rate = librosa.load(a_path, sr=audRate, mono=True, offset=offset, duration = duration)
        center_idx = int((margin + audSec/2) * audRate)
        start = max(0, center_idx - audLen // 2)
        end = min(len(audio_raw), center_idx + audLen // 2 + audLen % 2)
        wav[:end-start] = audio_raw[start: end]
        wav[wav > 1.] = 1.
        wav[wav < -1.] = -1.
        return wav, c_t

    def get_spec(wav):
        spec = librosa.stft(
            wav, n_fft=1022, hop_length=256)
        amp = np.abs(spec)[np.newaxis,np.newaxis,:,:] # 1, 1, F, T
        phase = np.angle(spec)[np.newaxis,np.newaxis,:,:] # 1, 1, F, T
        return (torch.from_numpy(amp).cuda(), torch.from_numpy(phase).cuda())
    
    return get_wav, get_spec

def get_audio(info):
    get_wav, get_spec = get_audfuncs()
    a_path, _, num_f, fps, a_len = info
    wav, c_t = get_wav(a_path, num_f, fps, a_len)
    amp, phase = get_spec(wav)
    return amp, phase, c_t

def get_sythesis_audio(infos):
    N = len(infos)
    get_wav, get_spec = get_audfuncs()
    c_ts = []
    mix_wav = None
    for info in infos:
        a_path, _, num_f, fps, a_len = info
        wav, c_t = get_wav(a_path, num_f, fps, a_len)
        c_ts.append(c_t)
        if mix_wav is None:
            mix_wav = wav
        else:
            mix_wav += wav
    mix_wav /= N
    amp, phase = get_spec(mix_wav)
    return amp, phase, c_ts

def vis_aug(img_ls):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    imgSize = 224
    transform_list = []
    transform_list.append(vtransforms.Resize(imgSize, T.InterpolationMode.BICUBIC))
    # transform_list.append(vtransforms.PadImg(128, imgSize, "constant"))
    transform_list.append(vtransforms.CenterCrop((224, 224))) # H, W
    transform_list.append(vtransforms.ToTensor())
    transform_list.append(vtransforms.Normalize(mean, std))
    transform_list.append(vtransforms.Stack())
    return transforms.Compose(transform_list)(img_ls)


def get_visfuncs():
    num_frames = 1
    stride_frames = 8
    def load_frame(v_path, num_f, fps, t):
        fps = float(fps)
        num_f = float(num_f)
        center_idx = round(t * fps) + 1
        img_ls = []
        for i in range(num_frames):
            idx_offset = (i - num_frames // 2) * stride_frames
            frame_pth = os.path.join(
                    v_path,
                    '{:06d}.jpg'.format(center_idx + idx_offset))
            img_ls.append(Image.open(frame_pth).convert('RGB'))
        frames = vis_aug(img_ls)
        return [frames.unsqueeze(0).cuda()] # [1, H, W, 3]
    
    def concate_frames(frames_ls):
        return [torch.cat(frames_ls, dim=-1)]
    return load_frame, concate_frames

def get_single_vis(info, c_t):
    load_frame, concate_frames = get_visfuncs()
    _, v_path, num_f, fps, a_len = info
    return load_frame(v_path, num_f, fps, c_t)

def get_sythesis_vis(infos, c_ts):
    N = len(infos)
    frames = []
    load_frame, concate_frames = get_visfuncs()
    for i, info in enumerate(infos):
        _, v_path, num_f, fps, a_len = info
        frames += load_frame(v_path, num_f, fps, c_ts[i])
    frame = concate_frames(frames)
    return frame

def get_multi_vis(infos, c_ts):
    N = len(infos)
    frames = []
    load_frame, concate_frames = get_visfuncs()
    for i, info in enumerate(infos):
        _, v_path, num_f, fps, a_len = info
        frames.append(load_frame(v_path, num_f, fps, c_ts[i])[0])
    # frame = concate_frames(frames)
    return frames

def datapipeline_systhesis(infos):
    amp, phase, c_ts = get_sythesis_audio(infos)
    frame = get_sythesis_vis(infos, c_ts)
    return (amp, phase), frame

def datapipeline_multivis(infos):
    amp, phase, c_ts = get_sythesis_audio(infos)
    frames = get_multi_vis(infos, c_ts)
    return (amp, phase), frames

def main(args):
    # Build model.
    builder = ModelBuilder()
    
    net_sound = builder.build_sound(
        arch=args.arch_sound,
        fc_dim=args.num_channels,
        weights=args.weights_sound,
        fusion_type=args.fusion_type,
        att_type=args.att_type
        )
    
    net_frame = builder.build_frame(
        arch=args.arch_frame,
        fc_dim=args.vis_channels,
        pool_type=args.img_pool,
        weights=args.weights_frame
        )
    nets = (net_sound, net_frame)
    
    # Wrap networks
    netWrapper = NetWrapper(nets)
    # netWrapper = torch.nn.DataParallel(netWrapper, device_ids=range(args.num_gpus))
    netWrapper.to(args.device)
    netWrapper.eval()
    save_folder = "/mnt/data2/he/src/JointSoP/duet_inference"
    if os.path.exists(save_folder):
        cmd = "rm -r {}".format(save_folder)
        os.system(cmd)
    os.mkdir(save_folder)
    print("run the model")
    # meta = ("./data/audio_duet/xylophone acoustic_guitar/tHvLBLCBHyU.wav","./data/frames_duet//xylophone acoustic_guitar/tHvLBLCBHyU.mp4",3155,30.0,105.33)
    # meta = ("./data/audio/accordion/c7Jyu_OiEB0.wav", "./data/frames/accordion/c7Jyu_OiEB0.mp4",2554,30.0,85.519)
    # meta = ("./data/audio_duet/acoustic_guitar violin/i53jcmLQESI.wav","./data/frames_duet//acoustic_guitar violin/i53jcmLQESI.mp4",10165,30.0,339.29)
    # meta = ("./data/audio_duet/flute violin/RWP6BHh_c7Y.wav","./data/frames_duet//flute violin/RWP6BHh_c7Y.mp4",3314,30.0,110.55)
    # meta = ("./data/audio_duet/cello acoustic_guitar/HnUOaSfTA6c.wav","./data/frames_duet//cello acoustic_guitar/HnUOaSfTA6c.mp4",7854,30.0,261.9)
    # meta = ("./data/audio_duet/trumpet tuba/Zqg4DwXmYBI.wav","./data/frames_duet/trumpet tuba/Zqg4DwXmYBI.mp4",1332,30.0,44.47)
    # meta = ("./data/audio_duet/clarinet acoustic_guitar/nTMJTm6tqoY.wav","./data/frames_duet//clarinet acoustic_guitar/nTMJTm6tqoY.mp4",9923,30.0,330.74)
    # meta = ("./data/audio_duet/clarinet acoustic_guitar/5GWxV8I21AQ.wav","./data/frames_duet//clarinet acoustic_guitar/5GWxV8I21AQ.mp4",5405,30.0,180.19)
    # meta = ("./data/audio_duet/saxophone acoustic_guitar/-HLTNgdajqw.wav","./data/frames_duet//saxophone acoustic_guitar/-HLTNgdajqw.mp4",4850,24.0,202.34)
    
    ############# Vis single video #########
    # meta = ("./data/audio_duet/saxophone acoustic_guitar/1vZ-IKkcPL4.wav","./data/frames_duet//saxophone acoustic_guitar/1vZ-IKkcPL4.mp4",2318,24.0,96.69)
    # vis_video(netWrapper, meta, save_folder)
    # assert False
    # assert False 
    ########
    
    ############# Image concatenation forward ##########
    # meta = ("./data/audio/acoustic_guitar/MVZGmcXU2D8.wav","./data/frames/acoustic_guitar/MVZGmcXU2D8.mp4",6162,30.0,205.659)
    # meta2 = ("./data/audio/trumpet/8jxZMGYbQIw.wav", "./data/frames/trumpet/8jxZMGYbQIw.mp4",2316,30.0,77.392)
    
    # aud, frame = datapipeline_systhesis([meta, meta2]) # Concatenation of Imgs.
    # meta = netWrapper.forward(aud, frame, args, True)
    # recover_aud(meta, aud, save_folder)
    # recover_visual(meta, frame, save_folder)
    # assert False

    ############# Mix & Separate Evaluation ############
    # save_folder = "/mnt/data2/he/src/JointSoP/duet_inference2"
    # if os.path.exists(save_folder):
    #     cmd = "rm -r {}".format(save_folder)
    #     os.system(cmd)
    # os.mkdir(save_folder)
    # print("run the model")
    # aud, frame = datapipeline_multivis([meta, meta2])
    # meta = netWrapper.forward(aud, frame, args, True)
    # # meta = forward_Exp1_plus(netWrapper, aud, frame, args, True)
    # recover_aud(meta, aud, save_folder)
    # recover_visual(meta, frame, save_folder)

    ############# Audio Only test  ##############
    # save_folder = "/mnt/data2/he/src/JointSoP/duet_inference3"
    # if os.path.exists(save_folder):
    #     cmd = "rm -r {}".format(save_folder)
    #     os.system(cmd)
    # os.mkdir(save_folder)
    # print("run the model")
    # aud, frame = datapipeline_multivis([meta, meta2])
    # meta = netWrapper.forward(aud, frame, args, False)
    # recover_aud(meta, aud, save_folder)
    
    ############# Concatenation of features ##############
    # save_folder = "/mnt/data2/he/src/JointSoP/duet_inference4"
    # if os.path.exists(save_folder):
    #     cmd = "rm -r {}".format(save_folder)
    #     os.system(cmd)
    # os.mkdir(save_folder)
    # print("run the model")
    # aud, frame = datapipeline_multivis([meta, meta2])
    # # meta = netWrapper.forward(aud, frame, args, True)
    # meta = forward_Exp1_plus(netWrapper, aud, frame, args, True)
    # recover_aud(meta, aud, save_folder)
    # frame = [torch.cat(frame, dim=-1)]*2
    # recover_visual(meta, frame, save_folder)    
    
    ########### Visualization on all real mixtures. ##############
    # csv_path = "/mnt/data2/he/src/JointSoP/data/duet.csv"
    # for i, row in enumerate(csv.reader(open(csv_path, 'r'), delimiter=',')):
    #     if len(row) < 2:
    #         continue
    #     print(f"Processing video {i}.")
    #     vis_video(netWrapper, row[:5], os.path.join(save_folder, "duet_" + str(i)), True)
        
    ########### Visualization on single mixture.   ###############
    meta = ("./data/audio_duet/saxophone acoustic_guitar/1vZ-IKkcPL4.wav","./data/frames_duet//saxophone acoustic_guitar/1vZ-IKkcPL4.mp4",2318,24.0,96.69)
    vis_video(netWrapper, meta, save_folder)
    

def recover_aud(meta, audios, save_folder):
    """ Given the model output, recover audio spectrum back to audio waveform.

    Args:
        meta (dict): output dictionary from netwarpper.
        audios (tuple): _description_
        save_folder (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    pred_masks = meta['pred_masks']
    mag_mix, phase_mix = audios
    mag_mix = mag_mix.detach().cpu().numpy()
    phase_mix = phase_mix.detach().cpu().numpy()

    # Recoder mixed audio.
    mix_wav = istft_reconstruction(mag_mix[0, 0], phase_mix[0, 0], hop_length=args.stft_hop)
    mix_amp = magnitude2heatmap(mag_mix[0, 0])
    filename_mixwav = 'mix.wav'
    filename_mixmag = 'mix.jpg'
    imsave(os.path.join(save_folder, filename_mixmag), mix_amp[::-1, :, :])
    sf.write(os.path.join(save_folder, filename_mixwav), mix_wav, args.audRate, subtype='PCM_24')
    
    grid_unwarp = torch.from_numpy(
    warpgrid(1, 1022//2+1, pred_masks[0].size(3), warp=False)).cuda()
    pred_masks_linear = [None for n in range(2)]
    for n in range(2):
        pred_masks_linear[n] = F.grid_sample(pred_masks[n], grid_unwarp, align_corners=False)
        pred_masks[n] = pred_masks[n].detach().cpu().numpy()
        pred_masks_linear[n] = pred_masks_linear[n].detach().cpu().numpy()
        if True:
            pred_masks[n] = (pred_masks[n] > 0.5).astype(np.float32)
            pred_masks_linear[n] = (pred_masks_linear[n] > 0.5).astype(np.float32)
    
    # Recover predicted audio.
    preds_wav = [None for n in range(2)]
    aud_pths = [None for n in range(2)]
    for n in range(2):
        # GT and predicted audio recovery
        pred_mag = mag_mix[0, 0] * pred_masks_linear[n][0, 0]
        preds_wav[n] = istft_reconstruction(pred_mag, phase_mix[0, 0], hop_length=256)

        # output masks
        filename_predmask = 'predmask{}.jpg'.format(n+1)
        pred_mask = (np.clip(pred_masks[n][0, 0], 0, 1) * 255).astype(np.uint8)
        imsave(os.path.join(save_folder, filename_predmask), pred_mask[::-1, :])

        # ouput spectrogram (log of magnitude, show colormap)
        filename_predmag = 'predamp{}.jpg'.format(n+1)
        pred_mag = magnitude2heatmap(pred_mag)
        imsave(os.path.join(save_folder, filename_predmag), pred_mag[::-1, :, :])

        # output audio
        filename_predwav = 'pred{}.wav'.format(n+1)
        sf.write(os.path.join(save_folder, filename_predwav), preds_wav[n], 11025, subtype='PCM_24')
        aud_pths[n] = os.path.join(save_folder, filename_predwav)
    return aud_pths
    
def recover_visual(meta, frames, save_folder):
    """A wrapper to bridge single visual input and multiple visual input.

    Args:
        meta (dict): output dictionary from netwarpper.
        frames (list(torch.Tensor)): [1, H, W, 3] * 1 or 2 ## Augmented visual frames.
        save_folder (str): path of the save directory.

    Returns:
        att_imgs (list(np.ndarray)): [H, W, 3] * 2  ## np.ndarray-type attended image.
    """
    if len(frames) == 1:
        frames *= 2
    att_imgs = plot_save_att(frames, meta['maps'], save_folder)
    return att_imgs

def plot_save_att(imgs, maps, save_folder):
    """Plot and save the attented images.
    Args:
        imgs (list(torch.Tensor)): [1, H, W, 3] * 2 ## for mixture of two.
        maps (torch.Tensor): B, C, H, W             ## B: batch size, C: num of sources, H: height, W: width.
        save_folder (str): directory to save the attended images.
    Returns:
        att_imgs (list(np.ndarray)): [H, W, 3] * 2  ## np.ndarray-type attended image.
    """
    num_mix = 2
    att_imgs = []
    for b in range(1):
        for n in range(num_mix):
            img = recover_rgb(imgs[n].cpu()[b, :])[:, :, ::-1] # H, W, 3
            # Normalization
            one_map = maps[b][n].squeeze().cpu().detach().numpy()# H, W
            one_map = 255*(one_map-one_map.min())/(one_map.max()-one_map.min())
            one_map = np.stack([one_map.astype(np.uint8)]*3, axis=-1) # H, W, 3
            # Color.
            one_map = cv2.resize(one_map, img.shape[:2][::-1]) # img.shape[:2][::-1]
            heat_map = cv2.applyColorMap(one_map, cv2.COLORMAP_JET)
            att_img = heat_map * 0.4 + np.uint(img) * 0.6
            att_imgs.append(att_img.astype(np.uint8))
            cv2.imwrite(os.path.join(save_folder, "img_att_"+f"_{n}.jpg"), att_img)
            cv2.imwrite(os.path.join(save_folder, "img_"+f"_{n}.jpg"), np.uint(img))
    return att_imgs


def vis_video(netWrapper, meta, save_folder, use_vis=True):
    """Visualize an single video
    Args:
        netWrapper (torch.nn.Module): The netwarpper model.
        meta (list): [audio_path, video_frame_path, num_frames, frame_rate, audio_length(second)]
        save_folder (str): Path of the directory to save the results. 
        use_vis (bool, optional): Audio-visual or Audio-only Separation. Defaults to True.
    """
    if os.path.exists(save_folder):
        cmd = "rm -r {}".format(save_folder)
        os.system(cmd)
    os.mkdir(save_folder)
    frames_ls = []
    _, _, _, fps, _ = meta
    fps = float(fps)
    audSec = 6
    # audLen = 65535
    # audRate = 11025
    amp, phase,  t = get_audio(meta)
    audios = (amp, phase)
    for i in range(int(fps * audSec)):
        frames_ls.append(get_single_vis(meta, t - 6/2 + i / fps))
    video_1 = []
    video_2 = []
    for i, frames in enumerate(frames_ls):
        netWrapper.eval()
        meta = netWrapper.forward(audios, frames, args, use_vis)
        if i == int(len(frames_ls)/2):
            aud_pths = recover_aud(meta, audios, save_folder)
        if use_vis:
            att_imgs = recover_visual(meta, frames, save_folder)
            video_1.append(att_imgs[0])
            video_2.append(att_imgs[1])
    if use_vis:
        video_1 = np.array(video_1)
        video_2 = np.array(video_2) # 180, H, W, C
        video_path_1 = os.path.join(save_folder, "video_1.mp4")
        video_path_2 = os.path.join(save_folder, "video_2.mp4")
        save_video(video_path_1, video_1, fps=fps)
        save_video(video_path_2, video_2, fps=fps)
        combine_video_audio(video_path_1, aud_pths[0], os.path.join(save_folder,  f"AV_{0}.mp4"))
        combine_video_audio(video_path_2, aud_pths[1], os.path.join(save_folder,  f"AV_{1}.mp4"))
    
if __name__=="__main__":
    parser = ArgParser()
    args = parser.parse_train_arguments()
    args.batch_size = args.num_gpus * args.batch_size_per_gpu
    args.device = torch.device("cuda")
    args.ckpt = os.path.join(args.ckpt, args.id)
    if args.mode == 'eval':
        args.weights_sound = os.path.join(args.ckpt, 'sound_best.pth')
        args.weights_frame = os.path.join(args.ckpt, 'frame_best.pth')
    main(args)