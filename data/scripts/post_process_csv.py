import os
import csv
import cv2
import soundfile as sf



def get_fps(v_path):
    videocap = cv2.VideoCapture(v_path)
    return videocap.get(cv2.CAP_PROP_FPS)

def get_audio_length(a_path):
    info = sf.SoundFile(a_path)
    sr = info.samplerate
    num = info.frames
    return num/sr

def get_class_name(a_path):
    return os.path.basename(os.path.dirname(a_path))

if __name__=="__main__":
    csv_path = "/mnt/data2/he/src/SoP2/data/duet.csv"
    with open(csv_path, 'r+') as f:
        writer = csv.writer(f)
        reader = csv.reader(f)
        info = []
        for line in reader:
            # fps = round(get_fps(line[1].replace("/frames_duet/", "/videos/duet/")), 0)
            # al = round(get_audio_length(line[0]), 3)
            class_name = get_class_name(line[0])
            line += [class_name]
            info.append(line)
        for line in info:
            writer.writerow(line)


