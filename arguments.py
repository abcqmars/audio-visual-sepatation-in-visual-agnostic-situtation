import argparse
from cmath import cos


class ArgParser(object):
    def __init__(self):
        parser = argparse.ArgumentParser()
        # Model related arguments
        parser.add_argument('--id', default='',
                            help="a name for identifying the model")
        parser.add_argument('--num_mix', default=2, type=int,
                            help="number of sounds to mix")
        parser.add_argument('--arch_sound', default='unet7',
                            help="architecture of net_sound")
        parser.add_argument('--arch_frame', default='resnet18dilated',
                            help="architecture of net_frame")
        parser.add_argument('--arch_synthesizer', default='linear',
                            help="architecture of net_synthesizer")
        parser.add_argument('--fusion_type', default='con',
                            help="fusion module type.")
        parser.add_argument('--weights_sound', default='',
                            help="weights to finetune net_sound")
        parser.add_argument('--weights_frame', default='',
                            help="weights to finetune net_frame")
        parser.add_argument('--weights_synthesizer', default='',
                            help="weights to finetune net_synthesizer")
        parser.add_argument('--num_channels', default=32, type=int,
                            help='number of channels')
        parser.add_argument('--num_frames', default=1, type=int,
                            help='number of frames')
        parser.add_argument('--stride_frames', default=1, type=int,
                            help='sampling stride of frames')
        parser.add_argument('--img_pool', default='maxpool',
                            help="avg or max pool image features")
        parser.add_argument('--img_activation', default='sigmoid',
                            help="activation on the image features")
        parser.add_argument('--sound_activation', default='no',
                            help="activation on the sound features")
        parser.add_argument('--output_activation', default='sigmoid',
                            help="activation on the output")
        parser.add_argument('--binary_mask', default=1, type=int,
                            help="whether to use bianry masks")
        parser.add_argument('--mask_thres', default=0.5, type=float,
                            help="threshold in the case of binary masks")
        parser.add_argument('--loss', default='l1',
                            help="loss function to use")
        parser.add_argument('--weighted_loss', default=0, type=int,
                            help="weighted loss")
        parser.add_argument('--log_freq', default=1, type=int,
                            help="log frequency scale")
        parser.add_argument('--vis_channels', default=512, type=int)
        parser.add_argument('--not_pool_vis', action="store_false", default=True)

        # Data related arguments
        parser.add_argument('--num_gpus', default=1, type=int,
                            help='number of gpus to use')
        parser.add_argument('--batch_size_per_gpu', default=32, type=int,
                            help='input batch size')
        parser.add_argument('--workers', default=32, type=int,
                            help='number of data loading workers')
        parser.add_argument('--num_val', default=-1, type=int,
                            help='number of images to evalutate')
        parser.add_argument('--num_vis', default=40, type=int,
                            help='number of images to evalutate')

        parser.add_argument('--audLen', default=65535, type=int,
                            help='sound length')
        parser.add_argument('--audRate', default=11025, type=int,
                            help='sound sampling rate')
        parser.add_argument('--stft_frame', default=1022, type=int,
                            help="stft frame length")
        parser.add_argument('--stft_hop', default=256, type=int,
                            help="stft hop length")

        # Video related arguments:
        parser.add_argument('--imgSize', default=224, type=int,
                            help='size of input frame')
        parser.add_argument('--frameRate', default=8, type=float,
                            help='video frame sampling rate')
        parser.add_argument('--load_clips', action='store_true', default=False,
                            help='whether or not to use 3D viual info.')
        parser.add_argument('--clip_len', default=32, type=int,
                            help='Num of frames used for a clip.')
        
        # Misc arguments
        parser.add_argument('--seed', default=1234, type=int,
                            help='manual seed')
        parser.add_argument('--ckpt', default='./ckpt',
                            help='folder to output checkpoints')
        parser.add_argument('--disp_iter', type=int, default=20,
                            help='frequency to display')
        parser.add_argument('--eval_epoch', type=int, default=1,
                            help='frequency to evaluate')

        self.parser = parser

    def add_train_arguments(self):
        parser = self.parser

        parser.add_argument('--mode', default='train',
                            help="train/eval")
        parser.add_argument('--list_train', nargs = '+',
                            default=['data/train.csv'])
        parser.add_argument('--list_val', nargs = '+',
                            default=['data/val.csv'])
        parser.add_argument('--av_list_train', nargs = '+',
                            default=['data/train_av.csv'])
        parser.add_argument('--ao_list_train', nargs = '+',
                            default=['data/train_ao.csv'])

        # optimization related arguments
        parser.add_argument('--num_epoch', default=100, type=int,
                            help='epochs to train for')
        parser.add_argument('--num_iters', default=120000, type=int,
                            help='iters to train')
        parser.add_argument('--eval_iter', default=7500, type = int,
                            help='iters to evaluate')
        parser.add_argument('--iter_per_av', default=2, type = int,)
        parser.add_argument('--lr_frame', default=1e-4, type=float, help='LR')
        parser.add_argument('--lr_sound', default=1e-3, type=float, help='LR')
        parser.add_argument('--lr_motion', default=1e-4, type=float, help='LR')
        parser.add_argument('--lr_synthesizer',
                            default=1e-3, type=float, help='LR')
        parser.add_argument('--lr_steps',
                            nargs='+', type=int, default=[20000, 40000],
                            help='steps to drop LR in epochs')
        parser.add_argument('--start_av_first', action='store_true', default=False,
                            help = "To train av or ao case only for first_steps.")
        parser.add_argument('--num_fsteps', default=40000, type=int)
        parser.add_argument('--beta1', default=0.9, type=float,
                            help='momentum for sgd, beta1 for adam')
        parser.add_argument('--weight_decay', default=1e-4, type=float,
                            help='weights regularizer')
        parser.add_argument('--train_repeat', default=100, type=int,
                            help='duplicate so that one epoch has more iters')
        self.parser = parser

    def print_arguments(self, args):
        print("Input arguments:")
        for key, val in vars(args).items():
            print("{:16} {}".format(key, val))

    def add_other_arguments(self):
        parser = self.parser
        parser.add_argument('--load_ckpt', type=str, default = False,
                            help = "Whether or not to restart from latest checkpoint.")
        parser.add_argument('--use_spec', action = "store_false",  default = True,
                            help = "Whether or not use spectrogram. Always used for Spec-based method. ")
        parser.add_argument('--rate_dc', type=float, default = 1.0,
                            help = "Rate of samples from differnt classes.")
        parser.add_argument('--rate_sc', type=float, default = 0.05,
                            help = "Rate of samples from the same class. (deprecated when there is no motion information)")
        parser.add_argument('--rate_sv', type=float, default = 0.0,
                            help = "Rate of sample from the same video.  (deprecated when there is no motion information)")
        parser.add_argument('--margin', type=float, default = 3.0,
                            help = "Margin to truncate the audio.")
        parser.add_argument('--max_silent', type=float, default = 0.67,
                            help = "Maximal rate of silent area.")
        parser.add_argument('--val_repeat', type=int, default = 12,
                            help = "Repeat the val-set for evaluation.")
        parser.add_argument('--match_weight', default=0.6, type=float, 
                            help="weight of the PiT attention loss")
        parser.add_argument('--one_frame', action="store_true", default=False,
                            help="Regardless of num_frames parameter, only use one frame to perform separation, this would increase the robustness of localization")
        parser.add_argument('--fix_vis', action="store_true", default=False,
                            help="Whether or not fix the backbone of visual net.")
        parser.add_argument('--att_type', type=str, default='cos', 
                            help="The similarity kernel of attention. cos: cosine similarity. sig: dot-product + sigmoid")
        self.parser = parser
        

    def parse_train_arguments(self):
        self.add_train_arguments()
        self.add_other_arguments()
        args = self.parser.parse_args()
        self.print_arguments(args)
        return args
