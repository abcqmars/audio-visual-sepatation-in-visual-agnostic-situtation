# System libssf
from multiprocessing import pool
import os
import csv
import shutil
import random
import time
from tkinter.messagebox import NO

import warnings

from matplotlib import use
warnings.filterwarnings('ignore')

# Numerical libs
import torch
import torch.nn.functional as F
import numpy as np
# import scipy.io.wavfile as wavfile
# from scipy.misc import imsave # Deprecated!
from imageio import imwrite as imsave
import soundfile as sf

#  from mir_eval.separation import bss_eval_sources
from asteroid.metrics import get_metrics
from torch.utils.data import ConcatDataset

# Our libs
from arguments import ArgParser
from dataset import MUSICMixDataset
from models import ModelBuilder, activate
from utils import AverageMeter, \
    recover_rgb, magnitude2heatmap,\
    istft_reconstruction, warpgrid, \
    combine_video_audio, save_video, makedirs
from viz import plot_loss_metrics, HTMLVisualizer

# Network wrapper, defines forward pass
class NetWrapper(torch.nn.Module):
    def __init__(self, nets, crit_ao, crit_av):
        super(NetWrapper, self).__init__()
        if len(nets) == 2:
            self.net_sound, self.net_frame = nets
            self.load_clips = False
        elif len(nets) == 3:
            self.net_sound, self.net_frame, self.net_motion = nets
            self.load_clips = True
        self.crit_ao = crit_ao
        self.crit_av = crit_av

    def prepare(self, batch_data, args, use_vis, is_share):
        mag_mix = batch_data['mag_mix']
        mags = batch_data['mags']
        mag_mix = mag_mix + 1e-10

        N = args.num_mix
        B = mag_mix.size(0)
        T = mag_mix.size(3)

        # 0.0 warp the spectrogram
        if args.log_freq:
            grid_warp = torch.from_numpy(
                warpgrid(B, 256, T, warp=True)).to(args.device)
            mag_mix = F.grid_sample(mag_mix, grid_warp, align_corners=False)
            for n in range(N):
                mags[n] = F.grid_sample(mags[n], grid_warp, align_corners=False)

        # 0.1 calculate loss weighting coefficient: magnitude of input mixture
        if args.weighted_loss:
            weights = torch.log1p(mag_mix)
            weights = torch.clamp(weights, 1e-3, 10)
        else:
            weights = torch.ones_like(mag_mix)

        # 0.2 ground truth masks are computed after warpping!
        gt_masks = [None for n in range(N)]
        for n in range(N):
            if args.binary_mask:
                gt_masks[n] = (mags[n] > 0.5 * mag_mix).float()
            else:
                gt_masks[n] = mags[n] / mag_mix
                gt_masks[n].clamp_(0., 5.)

        # LOG magnitude
        log_mag_mix = torch.log(mag_mix).detach()
        
        if use_vis or args.fusion_type == "share" or is_share:
            frames = batch_data['frames']
            if self.load_clips:
                clips = batch_data['clips']
            else:
                clips = None
            return frames, clips, mags, mag_mix, log_mag_mix, gt_masks, weights
        else:
            return mags, mag_mix, log_mag_mix, gt_masks, weights

    def forward_ao(self, data, args):
        mags, mag_mix, log_mag_mix, gt_masks, weight = data
        feat_sound, *meta = self.net_sound(log_mag_mix, None)
        pred_masks = activate(feat_sound, args.output_activation) # B x C x H x W
        pred_masks = torch.permute(pred_masks, (0, 2, 3, 1))  # B x H x W x C
        gt_masks = torch.stack(gt_masks, dim=-1)[:, 0, :]
        weight = torch.stack([weight[:, 0, :]]*2, dim=-1)
        err, permutations = self.crit_ao(pred_masks, gt_masks, weight)
        err = torch.mean(err)
        ordered_pred = self.crit_ao.reorder_tensor(pred_masks, permutations)
        
        return err, \
            {'pred_masks': [ordered_pred[:, :, :, i].unsqueeze(1) for i in range(2)], 'gt_masks': [gt_masks[:, :, :, i].unsqueeze(1) for i in range(2)],
             'mag_mix': mag_mix, 'mags': mags, 'weight': weight,
             }

    def forward_av(self, data, args):
        # Info-guided pipeline.
        N = args.num_mix
        frames, _, mags, mag_mix, log_mag_mix, gt_masks, weight = data
        # 1. Get the guidence information. => Bx1xC
        feat_frames = [None for n in range(N)]
        for n in range(N):
            feat_frames[n] = self.net_frame.forward_multiframe(frames[n], pool = args.not_pool_vis)

            feat_frames[n] = activate(feat_frames[n], args.img_activation)

        # total_feat = torch.cat(feat_frames[::-1], dim=1)
        match_loss = 0
        err1 = 0
        pred_masks = [None for n in range(N)]
        vis_in = feat_frames[::-1]
        feat_sound, meta = self.net_sound(log_mag_mix, vis_in) # B x C x H x W

        for n in range(N):
            pred_masks[n] = activate(feat_sound[:, n, :].unsqueeze(1), args.output_activation)
        err1 += self.crit_av(pred_masks, gt_masks[::-1], weight).reshape(1)
        match_loss += meta[0]
        # 2. Get mask with guidence -> BxCxHxW
        # total_feat = torch.cat(feat_frames, dim=1)
        err0 = 0
        pred_masks = [None for n in range(N)]

        vis_in = feat_frames
        feat_sound, meta = self.net_sound(log_mag_mix, vis_in)
        for n in range(N):
            pred_masks[n] = activate(feat_sound[:, n, :].unsqueeze(1), args.output_activation)
        err0 += self.crit_av(pred_masks, gt_masks, weight).reshape(1)
        match_loss += meta[0]
        return ((err1+err0)/2 + args.match_weight * match_loss).reshape(1), \
            {'pred_masks': pred_masks, 'gt_masks': gt_masks,
             'mag_mix': mag_mix, 'mags': mags, 'weight': weight, 'match_loss': match_loss.reshape(1)}

    def forward(self, batch_data, args, use_vis, is_share = False):
        data = self.prepare(batch_data, args, use_vis, is_share)

        if use_vis:
            if args.fusion_type== "MixVis":
                out = forward_avmiximg(self, data, args)
            else:
                out = self.forward_av(data, args)
        else:
            out = self.forward_ao(data, args)
        return out

def forward_avmiximg(self, batch_data, args):
    # Concatenated frames as the input of model.
    # Results: The model can learn how to separate but learn a trival solution in terms of localization.
    N = args.num_mix

    frames, _, mags, mag_mix, log_mag_mix, gt_masks, weight = batch_data

    # frames:  [B, C, T, H, W] * S
    # gt_masks: [B, 1, F, T] * 2
    
    # Visual forward passing:
    mix_frame = torch.cat(frames, dim=-1) # B, C, T, H, W * S
    # print(mix_frame.shape)
    feat_frame = self.net_frame.forward_multiframe(mix_frame, pool = args.not_pool_vis)
    feat_frame = activate(feat_frame, args.img_activation) # B, C, H, W * S

    # Audio forward passing:
    feat_sound, meta = self.net_sound(log_mag_mix, [feat_frame]) # B
    pred_masks = activate(feat_sound, args.output_activation) # B, S, F, T
    pred_masks = torch.permute(pred_masks, (0, 2, 3, 1))  # B, F, T, S
    gt_masks = torch.stack(gt_masks, dim=-1)[:, 0, :] # B, F, T, S
    err, permutations = self.crit_ao(pred_masks, gt_masks, weight)
    # print(err.shape)
    err = torch.mean(err).reshape(1)
    pred_masks = self.crit_ao.reorder_tensor(pred_masks, permutations)
    match_loss = meta[0]
    att_maps = meta[1]
    err += match_loss * args.match_weight
    return err, \
        {'pred_masks': [pred_masks[:, :, :, i].unsqueeze(1) for i in range(2)], 'gt_masks': [gt_masks[:, :, :, i].unsqueeze(1) for i in range(2)],
            'mag_mix': mag_mix, 'mags': mags, 'weight': weight, 'match_loss':match_loss, 'maps': att_maps}



# Calculate metrics
def calc_metrics(batch_data, outputs, args):
    # meters
    sdr_mix_meter = AverageMeter()
    sdr_meter = AverageMeter()
    sir_meter = AverageMeter()
    sar_meter = AverageMeter()
    si_sdr_meter = AverageMeter()

    # fetch data and predictions
    mag_mix = batch_data['mag_mix']
    phase_mix = batch_data['phase_mix']
    audios = batch_data['audios']

    pred_masks_ = outputs['pred_masks']

    # unwarp log scale
    N = args.num_mix
    B = mag_mix.size(0)
    pred_masks_linear = [None for n in range(N)]
    for n in range(N):
        if args.log_freq:
            grid_unwarp = torch.from_numpy(
                warpgrid(B, args.stft_frame//2+1, pred_masks_[0].size(3), warp=False)).to(args.device)
            pred_masks_linear[n] = F.grid_sample(pred_masks_[n], grid_unwarp, align_corners=False)
        else:
            pred_masks_linear[n] = pred_masks_[n]

    # convert into numpy
    mag_mix = mag_mix.numpy()
    phase_mix = phase_mix.numpy()
    for n in range(N):
        pred_masks_linear[n] = pred_masks_linear[n].detach().cpu().numpy()

        # threshold if binary mask
        if args.binary_mask:
            pred_masks_linear[n] = (pred_masks_linear[n] > args.mask_thres).astype(np.float32)

    # loop over each sample
    sdr_ls = []
    sir_ls = []
    si_sdr_ls = []
    for j in range(B):
        # save mixture
        mix_wav = istft_reconstruction(mag_mix[j, 0], phase_mix[j, 0], hop_length=args.stft_hop)

        # save each component
        preds_wav = [None for n in range(N)]
        for n in range(N):
            # Predicted audio recovery
            pred_mag = mag_mix[j, 0] * pred_masks_linear[n][j, 0]
            preds_wav[n] = istft_reconstruction(pred_mag, phase_mix[j, 0], hop_length=args.stft_hop)
            if np.all(preds_wav[n]==0):
                preds_wav[n] = 0.01 * np.random.rand(*preds_wav[n].shape)
                
        # separation performance computes
        L = preds_wav[0].shape[0]
        gts_wav = [None for n in range(N)]
        valid = True
        for n in range(N):
            gts_wav[n] = audios[n][j, 0:L].numpy()
            valid *= np.sum(np.abs(gts_wav[n])) > 1e-5
            valid *= np.sum(np.abs(preds_wav[n])) > 1e-5
        
        metrics_dict = get_metrics(mix_wav[0:L], np.asarray(gts_wav), np.asarray(preds_wav), 
            sample_rate=11025, metrics_list=['sdr', 'sir', 'sar', 'si_sdr'], )
        
        sdr = metrics_dict['sdr']
        sir = metrics_dict['sir']
        sar = metrics_dict['sar']
        si_sdr = metrics_dict['si_sdr']
        
        # sdr_mix_meter.update(sdr_mix.mean())
        sdr_meter.update(sdr)
        sir_meter.update(sir)
        sar_meter.update(sar)
        sdr_ls.append(sdr)
        sir_ls.append(sir)
        si_sdr_ls.append(si_sdr)
        si_sdr_meter.update(si_sdr)

    return [# sdr_mix_meter.average(),
            0, 
            sdr_meter.average(),
            sir_meter.average(),
            sar_meter.average(),
            sdr_ls,
            sir_ls,
            si_sdr_ls,
            si_sdr_meter.average()
            ]


# Visualize predictions
def output_visuals(vis_rows, batch_data, outputs, args, sdr_ls, sir_ls, use_vis_eval):
    
    # create saving folder:
    save_pth = os.path.join(args.vis, 'av' if use_vis_eval else 'ao')
    makedirs(save_pth, remove=False)
    
    # fetch data and predictions
    mag_mix = batch_data['mag_mix']
    phase_mix = batch_data['phase_mix']
    frames = batch_data['frames']
    infos = batch_data['infos']
    ids = batch_data['id']

    pred_masks_ = outputs['pred_masks']
    gt_masks_ = outputs['gt_masks']
    mag_mix_ = outputs['mag_mix']
    weight_ = outputs['weight']

    # unwarp log scale
    N = args.num_mix
    B = mag_mix.size(0)
    pred_masks_linear = [None for n in range(N)]
    gt_masks_linear = [None for n in range(N)]
    for n in range(N):
        if args.log_freq:
            grid_unwarp = torch.from_numpy(
                warpgrid(B, args.stft_frame//2+1, gt_masks_[0].size(3), warp=False)).to(args.device)
            pred_masks_linear[n] = F.grid_sample(pred_masks_[n], grid_unwarp, align_corners=False)
            gt_masks_linear[n] = F.grid_sample(gt_masks_[n], grid_unwarp, align_corners=False)
        else:
            pred_masks_linear[n] = pred_masks_[n]
            gt_masks_linear[n] = gt_masks_[n]

    # convert into numpy
    mag_mix = mag_mix.numpy()
    mag_mix_ = mag_mix_.detach().cpu().numpy()
    phase_mix = phase_mix.numpy()
    weight_ = weight_.detach().cpu().numpy()
    for n in range(N):
        pred_masks_[n] = pred_masks_[n].detach().cpu().numpy()
        pred_masks_linear[n] = pred_masks_linear[n].detach().cpu().numpy()
        gt_masks_[n] = gt_masks_[n].detach().cpu().numpy()
        gt_masks_linear[n] = gt_masks_linear[n].detach().cpu().numpy()

        # threshold if binary mask
        if args.binary_mask:
            pred_masks_[n] = (pred_masks_[n] > args.mask_thres).astype(np.float32)
            pred_masks_linear[n] = (pred_masks_linear[n] > args.mask_thres).astype(np.float32)

    # loop over each sample
    for j in range(B):
        row_elements = []

        # video names
        prefix = []
        for n in range(N):
            prefix.append('-'.join(infos[n][0][j].split('/')[-2:]).split('.')[0])
        prefix = '+'.join(prefix)
        prefix = ids[j] # + "_" + score
        makedirs(os.path.join(save_pth, prefix))

        # save mixture
        mix_wav = istft_reconstruction(mag_mix[j, 0], phase_mix[j, 0], hop_length=args.stft_hop)
        mix_amp = magnitude2heatmap(mag_mix_[j, 0])
        weight = magnitude2heatmap(weight_[j, 0], log=False, scale=100.)
        filename_mixwav = os.path.join(prefix, 'mix.wav')
        filename_mixmag = os.path.join(prefix, 'mix.jpg')
        filename_weight = os.path.join(prefix, 'weight.jpg')
        imsave(os.path.join(save_pth, filename_mixmag), mix_amp[::-1, :, :])
        imsave(os.path.join(save_pth, filename_weight), weight[::-1, :])
        sf.write(os.path.join(save_pth, filename_mixwav), mix_wav, args.audRate, subtype='PCM_24')
        row_elements += [{'text': prefix}, {'image': filename_mixmag, 'audio': filename_mixwav}]

        # save each component
        preds_wav = [None for n in range(N)]
        for n in range(N):
            # GT and predicted audio recovery
            gt_mag = mag_mix[j, 0] * gt_masks_linear[n][j, 0]
            gt_wav = istft_reconstruction(gt_mag, phase_mix[j, 0], hop_length=args.stft_hop)
            pred_mag = mag_mix[j, 0] * pred_masks_linear[n][j, 0]
            preds_wav[n] = istft_reconstruction(pred_mag, phase_mix[j, 0], hop_length=args.stft_hop)

            # output masks
            filename_gtmask = os.path.join(prefix, 'gtmask{}.jpg'.format(n+1))
            filename_predmask = os.path.join(prefix, 'predmask{}.jpg'.format(n+1))
            gt_mask = (np.clip(gt_masks_[n][j, 0], 0, 1) * 255).astype(np.uint8)
            pred_mask = (np.clip(pred_masks_[n][j, 0], 0, 1) * 255).astype(np.uint8)
            imsave(os.path.join(save_pth, filename_gtmask), gt_mask[::-1, :])
            imsave(os.path.join(save_pth, filename_predmask), pred_mask[::-1, :])

            # ouput spectrogram (log of magnitude, show colormap)
            filename_gtmag = os.path.join(prefix, 'gtamp{}.jpg'.format(n+1))
            filename_predmag = os.path.join(prefix, 'predamp{}.jpg'.format(n+1))
            gt_mag = magnitude2heatmap(gt_mag)
            pred_mag = magnitude2heatmap(pred_mag)
            imsave(os.path.join(save_pth, filename_gtmag), gt_mag[::-1, :, :])
            imsave(os.path.join(save_pth, filename_predmag), pred_mag[::-1, :, :])

            # output audio
            filename_gtwav = os.path.join(prefix, 'gt{}.wav'.format(n+1))
            filename_predwav = os.path.join(prefix, 'pred{}.wav'.format(n+1))
            sf.write(os.path.join(save_pth, filename_gtwav), gt_wav, args.audRate, subtype='PCM_24')
            sf.write(os.path.join(save_pth, filename_predwav), preds_wav[n], args.audRate, subtype='PCM_24')

            # output video
            if args.one_frame:
                frames_tensor = [recover_rgb(frames[n][j, :, t]) for t in range(1)]
            else:
                frames_tensor = [recover_rgb(frames[n][j, :, t]) for t in range(args.num_frames)]
            frames_tensor = np.asarray(frames_tensor)
            path_video = os.path.join(save_pth, prefix, 'video{}.mp4'.format(n+1))
            save_video(path_video, frames_tensor, fps=args.frameRate/args.stride_frames)

            # combine gt video and audio
            filename_av = os.path.join(prefix, 'av{}.mp4'.format(n+1))
            combine_video_audio(
                path_video,
                os.path.join(save_pth, filename_gtwav),
                os.path.join(save_pth, filename_av))

            row_elements += [
                {'video': filename_av},
                {'image': filename_predmag, 'audio': filename_predwav},
                {'image': filename_gtmag, 'audio': filename_gtwav},
                {'image': filename_predmask},
                {'image': filename_gtmask}]

        row_elements += [{'image': filename_weight}]
        vis_rows.append(row_elements)


def evaluate(netWrapper, loader, history, itera, args, use_vis_eval = True):
    print('Evaluating at {} iterations...'.format(itera))
    torch.set_grad_enabled(False)

    # remove previous viz results
    makedirs(args.vis, remove=False)
    
    # switch to eval mode
    netWrapper.eval()

    # initialize meters
    loss_meter = AverageMeter()
    match_loss_meter = AverageMeter()
    sdr_mix_meter = AverageMeter()
    sdr_meter = AverageMeter()
    sir_meter = AverageMeter()
    sar_meter = AverageMeter()
    si_sdr_meter = AverageMeter()

    vis_rows = []
    csv_ls = []
    print("Start evaluation for", 'audio visual' if use_vis_eval else 'audio only')
    for i, batch_data in enumerate(loader):
        err, outputs = netWrapper.forward(batch_data, args, use_vis_eval)
        err = err.mean()

        loss_meter.update(err.item())
        match_loss_meter.update(outputs['match_loss'].mean().item() if use_vis_eval else 0)

        # calculate metrics
        sdr_mix, sdr, sir, sar, sdr_ls, sir_ls, si_sdr_ls, si_sdr = calc_metrics(batch_data, outputs, args)
        sdr_mix_meter.update(sdr_mix)
        sdr_meter.update(sdr)
        sir_meter.update(sir)
        sar_meter.update(sar)
        si_sdr_meter.update(si_sdr)
        
        for n in range(len(sdr_ls)):
            csv_ls.append({
                "id": batch_data['id'][n],
                "sdr": sdr_ls[n],
                "sir": sir_ls[n],
                "si-snr": si_sdr_ls[n],
            })

        # output visualization
        if len(vis_rows) < args.num_vis:
            output_visuals(vis_rows, batch_data, outputs, args, sdr_ls, sir_ls, use_vis_eval)

    print('[Eval Summary] iterations: {}, Loss: {:.4f}, '
          'Loss_match: {:.4f}'
          'SDR_mixture: {:.4f}, SI-SDR: {:.4f}, SDR: {:.4f}, SIR: {:.4f}, SAR: {:.4f}'
          .format(itera, loss_meter.average(),
                  match_loss_meter.average(),
                  sdr_mix_meter.average(),
                  si_sdr_meter.average(),
                  sdr_meter.average(),
                  sir_meter.average(),
                  sar_meter.average()))
    if use_vis_eval:
        keyname = 'val_av'
    else:
        keyname = 'val_ao'
    history[keyname]['iter'].append(itera)
    history[keyname]['err'].append(loss_meter.average())
    history[keyname]['sdr'].append(sdr_meter.average())
    history[keyname]['sir'].append(sir_meter.average())
    history[keyname]['sar'].append(sar_meter.average())
    history[keyname]['si_sdr'].append(si_sdr_meter.average())


    # Save csv.
    save_pth = os.path.join(args.vis, 'av' if use_vis_eval else 'ao', 'results.csv')
    # makedirs(save_pth, remove=False)
    with open(save_pth, "w") as results_csv:
        writer = csv.DictWriter(results_csv, fieldnames=['id', 'sdr', 'sir', 'si-snr'])
        writer.writeheader()
        writer.writerows(csv_ls)
        
    # Plot figure
    if itera > 0:
        print('Plotting figures...')
        plot_loss_metrics(args.ckpt, history)


def checkpoint(nets, history, itera, args):
    print('Saving checkpoints at {} iterations.'.format(itera))
    suffix_latest = 'latest.pth'
    suffix_best = 'best.pth'
    if args.load_clips:
        (net_sound, net_frame, net_motion) = nets
        torch.save(net_motion.state_dict(),
                '{}/motion_{}'.format(args.ckpt, suffix_latest))
    else: 
        (net_sound, net_frame) = nets
    
    torch.save(history,
               '{}/history_{}'.format(args.ckpt, suffix_latest))
    torch.save(net_sound.state_dict(),
               '{}/sound_{}'.format(args.ckpt, suffix_latest))
    torch.save(net_frame.state_dict(),
               '{}/frame_{}'.format(args.ckpt, suffix_latest))

    cur_err =- history['val_ao']['si_sdr'][-1]
    if cur_err < args.best_err:
        args.best_err = cur_err
        torch.save(net_sound.state_dict(),
                   '{}/sound_{}'.format(args.ckpt, suffix_best))
        torch.save(net_frame.state_dict(),
                   '{}/frame_{}'.format(args.ckpt, suffix_best))
        if args.load_clips:
            torch.save(net_motion.state_dict(),
                    '{}/motion_{}'.format(args.ckpt, suffix_best))


def create_optimizer(nets, args):
    param_groups = []
    if args.load_clips:
        (net_sound, net_frame, net_motion) = nets
        param_groups += [{'params': net_motion.parameters(), 'lr': args.lr_motion}]
    else: 
        (net_sound, net_frame) = nets
    param_groups += [{'params': net_sound.parameters(), 'lr': args.lr_sound}]
    param_groups += [{'params': net_frame.fc.parameters(), 'lr': args.lr_sound}]
    if not args.fix_vis:
        param_groups += [{'params': net_frame.features.parameters(), 'lr': args.lr_frame}]
    return torch.optim.SGD(param_groups, momentum=args.beta1, weight_decay=args.weight_decay)


def adjust_learning_rate(optimizer, args):
    args.lr_sound *= 0.1
    args.lr_frame *= 0.1
    args.lr_motion *= 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] *= 0.1

def train_step(model, batch, optimizer, use_vis):
    torch.set_grad_enabled(True)
    model.train()
    model.zero_grad()
    err, outputs = model.forward(batch, args, use_vis)
    err = err.mean()
    err.backward()
    optimizer.step()
    if use_vis:
        match_loss = outputs['match_loss'].mean().item()
    else:
        match_loss = None
    return err.item(), match_loss


def get_av_ao_batch(av_loader, ao_loader, args):
    av_loader_iter = iter(av_loader)
    ao_loader_iter = iter(ao_loader)
    def return_batch(i):
        nonlocal av_loader_iter
        nonlocal ao_loader_iter
        if args.start_av_first:
            cond = i % args.iter_per_av == 0 or i < args.num_fsteps
        else:
            cond = i % args.iter_per_av == 0 and i > args.num_fsteps
        if cond:
            use_vis = True
            # Load audio-visual data.
            try:
                batch = next(av_loader_iter)
            except:
                av_loader_iter = iter(av_loader)
                batch = next(av_loader_iter)

        else: 
            # Load audio-only data.
            use_vis = False
            try:
                batch = next(ao_loader_iter)
            except:
                ao_loader_iter = iter(ao_loader)
                batch = next(ao_loader_iter)
        return batch, use_vis
    
    return return_batch


def main(args):
    # Network Builders
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
    if args.load_clips:
        net_motion = builder.build_motion()
        nets  = (net_sound, net_frame, net_motion)
    
    # build av_criterion
    crit_av = builder.build_criterion(arch=args.loss)
    crit_ao = builder.build_criterion(arch=args.loss, use_pit=True)

    # Dataset and Loader

    av_dataset_train = ConcatDataset( [MUSICMixDataset( csv_path, vars(args), split='train', random_sample = False) for csv_path in args.av_list_train])
    av_loader_train = torch.utils.data.DataLoader(
        av_dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=int(args.workers),
        drop_last=False)

    # Audio only dataset.
    ao_dataset_train = ConcatDataset([MUSICMixDataset( csv_path, vars(args), split='train', seed=10, random_sample = False) for csv_path in args.ao_list_train])
    ao_loader_train = torch.utils.data.DataLoader(
        ao_dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=int(args.workers),
        drop_last=False)

    dataset_val = ConcatDataset( [MUSICMixDataset( csv_path, vars(args), split='val') for csv_path in args.list_val])

    loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False)

    # Wrap networks
    netWrapper = NetWrapper(nets, crit_ao, crit_av)
    netWrapper = torch.nn.DataParallel(netWrapper, device_ids=range(args.num_gpus))
    netWrapper.to(args.device)

    # Set up optloader_trainlimizer
    optimizer = create_optimizer(nets, args)

    # History of peroformance
    history = {
        'train': {'iter': [], 'err': []},
        'train_ao': {'iter': [], 'err': []},
        'train_av': {'iter': [], 'err': []},
        'val_av': {'iter': [], 'err': [], 'sdr': [], 'sir': [], 'sar': [], 'si_sdr': []},
        'val_ao': {'iter': [], 'err': [], 'sdr': [], 'sir': [], 'sar': [], 'si_sdr': []}
        }
    start_i = 0
    if args.load_ckpt:
        print("Recovered from history.")
        history = torch.load(os.path.join(args.ckpt, "history_latest.pth"))
        start_i = history['train']['iter'][-1]

    # Eval mode
    # evaluate(netWrapper, loader_val, history, 0, args, True)
    # evaluate(netWrapper, loader_val, history, 0, args, False)
    if args.mode == 'eval':
        evaluate(netWrapper, loader_val, history, 0, args, True)
        evaluate(netWrapper, loader_val, history, 0, args, False)
        print('Evaluation Done!')
        return

    # 
    iter_time = AverageMeter()
    data_time = AverageMeter()
    match_loss_meter = AverageMeter()
    batch_machine = get_av_ao_batch(av_loader_train, ao_loader_train, args)
    
    err_total = 0
    err_av = 0
    av_count = 0
    err_ao = 0
    ao_count = 0
    for i in range(start_i+1, args.num_iters):

        torch.cuda.synchronize()
        tic = time.perf_counter()
        
        batch, use_vis = batch_machine(i)
        torch.cuda.synchronize()
        data_time.update(time.perf_counter() - tic)

        err, match_loss = train_step(netWrapper, batch, optimizer, use_vis)
        if use_vis:
            match_loss_meter.update(match_loss)
        torch.cuda.synchronize()
        iter_time.update(time.perf_counter() - tic)
        
        # Add err
        err_total += err
        if use_vis:
            err_av += (err - match_loss*args.match_weight)
            av_count += 1
        else:
            err_ao += err
            ao_count += 1
        
        if i % args.disp_iter == 0 and i != 0:
            print('iter: [{}/{}], Time: {:.2f}, Data: {:.2f}, '
                  'lr_sound: {}, lr_frame: {}, '
                  'loss: {:.3f}, loss_ao: {:.3f}, loss_av: {:.3f} '
                  'loss_match {:.3f}'
                  .format(i, args.num_iters,
                          iter_time.average(), data_time.average(),
                          args.lr_sound, args.lr_frame,
                          err_total/args.disp_iter,
                          err_ao/ao_count if ao_count!=0  else 0.66,
                          err_av/av_count if av_count!=0  else 0.25,
                          match_loss_meter.average() if use_vis else 0
                          ))
            match_loss_meter.initialize(0, 0)
            history['train']['iter'].append(i)
            history['train']['err'].append(err_total/args.disp_iter)
            if ao_count:
                history['train_ao']['iter'].append(i)
                history['train_ao']['err'].append(err_ao/ao_count)
            if av_count:
                history['train_av']['iter'].append(i)
                history['train_av']['err'].append(err_av/av_count)
            # Reset.
            err_total = 0
            err_av = 0
            av_count = 0
            err_ao = 0
            ao_count = 0

        # Evaluation and visualization
        if i % args.eval_iter == 0 and i > 1:
            evaluate(netWrapper, loader_val, history, i, args, True)
            evaluate(netWrapper, loader_val, history, i, args, False)
            # checkpointing
            checkpoint(nets, history, i, args)

        # drop learning rate
        if i in args.lr_steps:
            adjust_learning_rate(optimizer, args)
    
    print('Training Done!')


if __name__ == '__main__':
    # arguments
    parser = ArgParser()
    args = parser.parse_train_arguments()
    args.batch_size = args.num_gpus * args.batch_size_per_gpu
    args.device = torch.device("cuda")
    print('Model ID: {}'.format(args.id))

    # paths to save/load output
    ckpt_path = args.ckpt
    args.ckpt = os.path.join(args.ckpt, args.id)
    args.vis = os.path.join(args.ckpt, 'visualization/')
    if args.mode == 'train':
        makedirs(args.ckpt, remove=False)
        shutil.copyfile("./scripts/train_MUSIC.sh", os.path.join(args.ckpt, "train_MUSIC.sh"), follow_symlinks=True)
        if args.load_ckpt:
            args.weights_sound = os.path.join(args.ckpt, 'sound_latest.pth')
            args.weights_frame = os.path.join(args.ckpt, 'frame_latest.pth')
            if args.load_clips:
                args.weights_motion = os.path.join(args.ckpt, 'motion_latest.pth')
            # args.weights_synthesizer = os.path.join(args.ckpt, 'synthesizer_best.pth')
    elif args.mode == 'eval':
        args.weights_sound = os.path.join(args.ckpt, 'sound_best.pth')
        args.weights_frame = os.path.join(args.ckpt, 'frame_best.pth')
        if args.load_clips:
            args.weights_motion = os.path.join(args.ckpt, 'motion_best.pth')
        # args.weights_synthesizer = os.path.join(args.ckpt, 'synthesizer_best.pth')

    # initialize best error with a big number
    args.best_err = float("inf")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    main(args)
