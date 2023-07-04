import os,sys
import numpy as np
import cv2
from PIL import Image
from multiprocessing import Pool
import argparse
from IPython import embed #to debug
import skvideo.io
import scipy.misc
import glob
import imageio
from skimage import img_as_ubyte

def ToImg(raw_flow,bound):
    '''
    this function scale the input pixels to 0-255 with bi-bound

    :param raw_flow: input raw pixel value (not in 0-255)
    :param bound: upper and lower bound (-bound, bound)
    :return: pixel value scale from 0 to 255
    '''
    flow=raw_flow
    flow[flow>bound]=bound
    flow[flow<-bound]=-bound
    flow-=-bound
    flow*=(255/float(2*bound))
    return flow

def save_flows(flows,image,save_dir,num,bound):
    '''
    To save the optical flow images and raw images
    :param flows: contains flow_x and flow_y
    :param image: raw image
    :param save_dir: save_dir name (always equal to the video id)
    :param num: the save id, which belongs one of the extracted frames
    :param bound: set the bi-bound to flow images
    :return: return 0
    '''
    
    id_cvt = get_id_converter()
    fid = id_cvt(num)
    #rescale to 0~255 with the bound setting
    flow_x=ToImg(flows[...,0],bound)
    flow_y=ToImg(flows[...,1],bound)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    #save the image
    # save_img=os.path.join(save_dir,f'{fid}.jpg')
    # if not os.path.exists(save_x):
    #     imageio.imwrite(save_img,image)

    #save the flows
    save_x=os.path.join(save_dir,f'{fid}_x.jpg')
    save_y=os.path.join(save_dir,f'{fid}_y.jpg')
    # import pdb; pdb.set_trace()
    flow_x_img=Image.fromarray(flow_x/255)
    flow_y_img=Image.fromarray(flow_y/255)
    if not os.path.exists(save_x):
        imageio.imwrite(save_x,img_as_ubyte(flow_x_img))
    if not os.path.exists(save_y):
        imageio.imwrite(save_y,img_as_ubyte(flow_y_img))
    return 0

def dense_flow(augs):
    '''
    To extract dense_flow images
    :param augs:the detailed augments:
        video_name: the video name which is like: 'v_xxxxxxx',if different ,please have a modify.
        save_dir: the destination path's final direction name.
        step: num of frames between each two extracted frames
        bound: bi-bound parameter
    :return: no returns
    '''
    video_name,save_dir,step,bound=augs
    print(video_name)
    video_path=video_name

    # provide two video-read methods: cv2.VideoCapture() and skvideo.io.vread(), both of which need ffmpeg support

    # videocapture=cv2.VideoCapture(video_path)
    # if not videocapture.isOpened():
    #     print 'Could not initialize capturing! ', video_name
    #     exit()
    # try:
    videocapture=skvideo.io.vread(video_path)
    # except :
    #     print( '{} read error! '.format(video_name))
    #     return 0
    # if extract nothing, exit!
    if videocapture.sum()==0:
        print ('Could not initialize capturing',video_name)
        exit()
    
    len_frame=len(videocapture)
    dtvl1=cv2.optflow.createOptFlow_DualTVL1()
    frames = [None for j in range(1 + step)]
    frame_id = 0
    while True:
        if frame_id>=len_frame:
            break
        
        frame = videocapture[frame_id] # 0, 1, 2
        if frame_id < step:
            ls_id = frame_id % (step + 1)
            frames[ls_id] = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            frame_id += 1
            continue
        
        if frame_id != step:
            frames.append(frames.pop(0))
        frames[-1] = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        flowDTVL1=dtvl1.calc(frames[0],frames[-1],None)
        save_flows(flowDTVL1, frames[0], save_dir, frame_id-step, bound)
        frame_id += 1

    ####
    
    # while True:
    #     #frame=videocapture.read()
    #     if num0>=len_frame:
    #         break
    #     frame=videocapture[num0]
    #     num0+=1
    #     if frame_num==0:
    #         image=np.zeros_like(frame)
    #         gray=np.zeros_like(frame)
    #         prev_gray=np.zeros_like(frame)
    #         prev_image=frame
    #         prev_gray=cv2.cvtColor(prev_image,cv2.COLOR_RGB2GRAY)
    #         frame_num+=1
    #         # to pass the out of stepped frames
    #         step_t=step
    #         while step_t>1:
    #             #frame=videocapture.read()
    #             num0+=1
    #             step_t-=1
    #         continue

    #     image=frame
    #     gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    #     frame_0=prev_gray
    #     frame_1=gray
    #     ##default choose the tvl1 algorithm
    #     dtvl1=cv2.optflow.createOptFlow_DualTVL1()
    #     flowDTVL1=dtvl1.calc(frame_0,frame_1,None)
    #     save_flows(flowDTVL1,image,save_dir,frame_num,bound) #this is to save flows and img.
    #     prev_gray=gray
    #     prev_image=image
    #     frame_num+=1
    #     # to pass the out of stepped frames
    #     step_t=step
    #     while step_t>1:
    #         #frame=videocapture.read()
    #         num0+=1
    #         step_t-=1


# def get_video_list():
#     video_list=[]
#     for cls_names in os.listdir(videos_root):
#         cls_path=os.path.join(videos_root,cls_names)
#         for video_ in os.listdir(cls_path):
#             video_list.append(video_)
#     video_list.sort()
#     return video_list,len(video_list)


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
    if format == 1:
        def fmt(in_path):
            # ./date/videos/solo/cello/XXXX.mp4 -> ./data/frames/cello
            return in_path.replace("/videos/solo/", "/frames/")
    elif format == 2:
        def fmt(in_path, endwith = '.wav'):
            # ./date/videos/solo/cello/XXXX.mp4 -> ./date/audio/cello/XXXX.wav
            path = in_path.replace(".mp4", endwith)
            return path.replace("/videos/solo/", "/audio/")
    elif format == 3:
        def fmt(in_path, endwith = '.wav'):
            # ./date/videos/duet/cello/XXXX.mp4 -> ./date/audio_duet/cello/XXXX.wav
            path = in_path.replace(".mp4", endwith)
            return path.replace("/videos/duet/", "/audio_duet/")
    elif format == 4:
        def fmt(in_path):
            # ./date/videos/duet/cello/XXXX.mp4 -> ./data/frames_duet/cello
            return in_path.replace("/videos/duet/", "/frames_duet/")
    elif format == "optical":
        def fmt(in_path):
            # ./date/videos/duet/cello/XXXX.mp4 -> ./date/optical/duet/cello/XXXX.mp4
            return in_path.replace("/videos/", "/optical/")
    return fmt

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

def parse_args():
    parser = argparse.ArgumentParser(description="densely extract the video frames and optical flows")
    parser.add_argument('--dataset',default='ucf101',type=str,help='set the dataset name, to find the data path')
    parser.add_argument('--data_root',default='/n/zqj/video_classification/data',type=str)
    # parser.add_argument('--new_dir',default='flows',type=str)
    parser.add_argument('--num_workers',default=4,type=int,help='num of workers to act multi-process')
    parser.add_argument('--step',default=1,type=int,help='gap frames')
    parser.add_argument('--bound',default=15,type=int,help='set the maximum of optical flow')
    parser.add_argument('--mode',default='run',type=str,help='set \'run\' if debug done, otherwise, set debug')
    args = parser.parse_args()
    return args

if __name__ =='__main__':
    # Get videos' paths
    video_list = glob.glob("./videos/solo/*/*.mp4")
    num_vid = len(video_list)
    video_list = video_list[:100]
    path_cvt = get_odir_converter(format="optical")
    flows_dirs = [path_cvt(p) for p in video_list]
    pool=Pool(4)
    pool.map(dense_flow,zip(video_list,flows_dirs,[2]*len(video_list),[15]*len(video_list)))