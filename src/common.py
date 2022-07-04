import torch
import torchaudio
import matplotlib.pyplot as plt



def convert_to_log_mel(path_to_file, num_mel_bins=128, target_length=1024):
    waveform, sample_rate = torchaudio.load(path_to_file)
    fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sample_rate, use_energy=False,
                                              window_type='hanning', num_mel_bins=num_mel_bins, dither=0.0,
                                              frame_shift=10)  # using parameters from AST
    n_frames = fbank.size(dim=0)
    p = target_length - n_frames
    # cut and pad
    if p > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, p))
        fbank = m(fbank)
    elif p < 0:
        fbank = fbank[0:target_length, :]
    return fbank

def generate_ROC_curve(FPR,TRP,output_location):
    x_axis=[0]
    y_axis=[0]
    for i in range(len(FPR)):
        x_axis.append(FPR[i])
        y_axis.append(TRP[i])
    x_axis.append(1)
    y_axis.append(1)
    plt.plot(x_axis,y_axis)
    plt.plot([0,1],[0,1])
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.savefig(output_location)
    plt.close()