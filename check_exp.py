import torch
import numpy as np

if __name__=="__main__":
    dict_path = "/mnt/data2/he/src/JointSoP/ckpt/Exp5_Base_Sig/history_latest.pth"
    his_dict = torch.load(dict_path)
    ao_max_idx = np.argmax(his_dict['val_ao']['si_sdr'])
    print("Best AO model: ")
    print(f"AO: sir: {(round(his_dict['val_ao']['sir'][ao_max_idx], 2))}, SDR: {round(his_dict['val_ao']['sdr'][ao_max_idx], 2)}, SI-SNR: {round(his_dict['val_ao']['si_sdr'][ao_max_idx], 2)}" )
    print(f"AV: sir: {round(his_dict['val_av']['sir'][ao_max_idx], 2)}, SDR: {round(his_dict['val_av']['sdr'][ao_max_idx], 2)}, SI-SNR: {round(his_dict['val_av']['si_sdr'][ao_max_idx], 2)}" )
    print("+++++++++++++++++++++++++++++++++++++++++++")
    print("Best AV model: ")
    ao_max_idx = np.argmax(his_dict['val_av']['si_sdr'])
    print(f"AO sir: {round(his_dict['val_ao']['sir'][ao_max_idx], 2)}, SDR: {round(his_dict['val_ao']['sdr'][ao_max_idx], 2)}, SI-SNR: {round(his_dict['val_ao']['si_sdr'][ao_max_idx], 2)}" )
    print(f"AV sir: {round(his_dict['val_av']['sir'][ao_max_idx], 2)}, SDR: {round(his_dict['val_av']['sdr'][ao_max_idx], 2)}, SI-SNR: {round(his_dict['val_av']['si_sdr'][ao_max_idx], 2)}" )