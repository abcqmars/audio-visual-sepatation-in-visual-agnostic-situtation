from utils import combine_video_audio, save_video
import os
import cv2
import numpy as np
import librosa
import soundfile as sf

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

def get_odir_converter(format=1):
    """
    >>> cvt = get_odir_converter()
    >>> cvt("./date/videos/cello/XXXX.mp4")
    './date/frames/cello/XXXX.mp4'
    >>> cvt = get_odir_converter(format=2)
    >>> cvt("./date/videos/solo/cello/XXXX.mp4")
    './date/audio/cello/XXXX.wav'
    """
    # odir: out dir.
    if format == "frames":
        def fmt(in_path):
            # ./date/videos/solo/cello/XXXX.mp4 -> ./data/frames/cello
            return in_path.replace("/videos/solo/", "/frames/")
    elif format == "audios":
        def fmt(in_path, endwith = '.wav'):
            # ./date/videos/solo/cello/XXXX.mp4 -> ./date/audio/cello/XXXX.wav
            path = in_path.replace(".mp4", endwith)
            return path.replace("/videos/solo/", "/audio/")
    elif format == "aud_duet":
        def fmt(in_path, endwith = '.wav'):
            # ./date/videos/duet/cello/XXXX.mp4 -> ./date/audio_duet/cello/XXXX.wav
            path = in_path.replace(".mp4", endwith)
            return path.replace("/videos/duet/", "/audio_duet/")
    elif format == "vis_duet":
        def fmt(in_path):
            # ./date/videos/duet/cello/XXXX.mp4 -> ./data/frames_duet/cello
            return in_path.replace("/videos/duet/", "/frames_duet/")
    elif format == "optical":
        def fmt(in_path):
            # ./date/videos/duet/cello/XXXX.mp4 -> ./date/optical/duet/cello/XXXX.mp4
            return in_path.replace("/videos/", "/optical/")
    elif format == "opti_x":
        def fmt(in_path):
            # ./date/videos/solo/cello/XXXX.mp4/00000x.jpg -> ./date/videos/solo/cello/XXXX.mp4/00000x_x.jpg
            in_path = in_path.replace("/videos/", "/optical/")
            in_path = in_path.split(".")[0] + "_x" + "." + in_path.split(".")[1]
            return # in_path.replace("/videos/", "/optical/")
    return fmt

def frame_maker(frames_path):
    sub_images = []
    for path in frames_path:
        sub_images.append(cv2.imread(path)) # H, W, C
    # np.concatenate(sub_images, axis = 1)
    return np.concatenate(sub_images, axis = 1)


def make_video(frames_pth_ls, fps, save_path = "/mnt/data2/he/data/mit_music/demo"):
    frames = []
    for frames_path in frames_pth_ls:
        # print(frames_path)
        image = frame_maker(frames_path)
        frames.append(image)
    frames = np.asarray(frames)
    video_path = os.path.join(save_path, "video_only.mp4")
    save_video(video_path, frames, fps=30)
    return video_path

def make_audio(audio_path, offset, duration, save_path = "/mnt/data2/he/data/mit_music/demo"):
    audio_raw, rate = librosa.load(audio_path, mono=True, offset = offset, duration = duration)
    aud_save_pth = os.path.join(save_path, "trct_aud.wav")
    sf.write(aud_save_pth, audio_raw, rate, subtype='PCM_24')
    return aud_save_pth

def frames_path_loader(start, num, path):
    id_cvt = get_id_converter()
    frames_path_ls = []
    for j in range(start, num):
        fid = id_cvt(j) + ".jpg"
        oid = id_cvt(j) + "_x" + ".jpg"
        f_pth = os.path.join(path, fid)
        o_id = os.path.join(path, oid)
        frames_path_ls.append([f_pth, o_id])
    return frames_path_ls
    
    
if __name__ == "__main__":
    video_path = "./videos/solo/acoustic_guitar/26HLgXWF-Co.mp4"
    audio_path = get_odir_converter("audios")(video_path)
    print("audio_path", audio_path)
    optical_path = get_odir_converter("optical")(video_path)
    print("optical_path", optical_path)
    # frame_path = get_odir_converter("frames")(video_path)
    num_frames = 4735
    fps = 30
    trunc_fnum = 1800
    start_frame = 0
    aud_save_pth = make_audio(audio_path, start_frame/fps, trunc_fnum/fps)
    print("audio saved succeed.")
    paths = frames_path_loader(start_frame, trunc_fnum, optical_path)
    
    # assert False, f"{paths}"
    vid_save_pth = make_video(paths, 30)
    print("video saved succeed.")
    combine_video_audio(vid_save_pth, aud_save_pth, "/mnt/data2/he/data/mit_music/demo/VA.mp4")
    
    
    
    
    