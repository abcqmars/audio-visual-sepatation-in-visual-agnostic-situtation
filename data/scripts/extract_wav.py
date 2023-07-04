import os
import glob
import ffmpeg


if __name__ == "__main__":
    base_dir = "./MIT-Music/solo"
    out_dir = "./audio"
    os.makedirs(out_dir, exist_ok=True)

    class_list = sorted(glob.glob(os.path.join(base_dir, "*")))
    mov_list = [sorted(glob.glob(os.path.join(c, "*.mp4"))) for c in class_list]
    #print(mov_list)

    for c, m_list in zip(class_list, mov_list):
        #os.makedirs(os.path.join(out_dir, os.path.basename(c)), exist_ok=True)
        os.makedirs(os.path.join(out_dir, os.path.basename(os.path.dirname(m_list[0]))), exist_ok=True)
        print(os.path.basename(c))
        print(os.path.basename(os.path.dirname(m_list[0])))
        for m in m_list:
            #stream = ffmpeg.input(m)
            #print(os.path.basename(m))
            w_o = os.path.join(out_dir, os.path.basename(c), os.path.basename(m)[:-4] + ".wav")
            #print(w_o)
            #stream = ffmpeg.output(stream, w_o, **{"ar": "11025"})
            #ffmpeg.run(stream)

