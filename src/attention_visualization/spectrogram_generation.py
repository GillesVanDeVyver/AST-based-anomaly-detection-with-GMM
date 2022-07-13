import yaml
import os
import matplotlib.pyplot as plt
import numpy as np
import librosa
import sys
import librosa.display
from tqdm import tqdm

wd=os.path.dirname(__file__)
with open(os.path.join(wd,"spectrogram_generation.yaml")) as stream:
    param = yaml.safe_load(stream)

def file_to_vectors(file_name,
                    n_mels=64,
                    n_frames=5,
                    n_fft=1024,
                    hop_length=512,
                    power=2.0,
                    flatten=True,
                    log_mel=True):
    """
    convert file_name to a vector array.

    file_name : str
        target .wav file

    return : numpy.array( numpy.array( float ) )
        vector array
        * dataset.shape = (dataset_size, feature_vector_length)
    """
    # calculate the number of dimensions
    dims = n_mels * n_frames

    # generate melspectrogram using librosa
    y, sr = librosa.load(file_name, sr=None, mono=True)
    if log_mel:
        spectrogram = librosa.feature.melspectrogram(y=y,
                                                     sr=sr,
                                                     n_fft=n_fft,
                                                     hop_length=hop_length,
                                                     n_mels=n_mels,
                                                     power=power)
    else:
        spectrogram = librosa.stft(y=y,
                                   n_fft=n_fft,
                                   hop_length=hop_length)
    # convert melspectrogram to log mel energies
    log_mel_spectrogram = 20.0 / power * np.log10(np.maximum(spectrogram, sys.float_info.epsilon))
    if flatten:
        # calculate total vector size
        n_vectors = len(log_mel_spectrogram[0, :]) - n_frames + 1

        # skip too short clips
        if n_vectors < 1:
            return np.empty((0, dims))

        # generate feature vectors by concatenating multiframes
        vectors = np.zeros((dims, n_vectors))
        for t in range(n_frames):
            vectors[n_mels * t: n_mels * (t + 1), :] = log_mel_spectrogram[:, t: t + n_vectors]
        return vectors
    else:
        return log_mel_spectrogram


def convert_to_spectrogram_and_save(file_location,output_location, log_mel=True):
    log_mel_spectrogram = file_to_vectors(file_location,
                                          n_mels=param["n_mels"],
                                          n_frames=param["n_frames"],
                                          n_fft=param["n_fft"],
                                          hop_length=param["hop_length"],
                                          power=param["power"],
                                          flatten=False,
                                          log_mel=log_mel)
    plt.figure()
    plt.axis('off')  # no axis
    plt.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])  # Remove the white edge
    librosa.display.specshow(log_mel_spectrogram)
    plt.savefig(os.path.join(output_location))
    plt.close()


# generate spectrograms of some samples of the data
# the start and end index determine what sample is processed
def generate_spectrograms_as_png(start_index,end_index):
    dev_data_directory=os.path.join(wd,param['dev_data_location'])
    output_base_directory=os.path.join(wd, param['spectrogram_as_png_location'])
    for machine in tqdm(param['machine_types']):
        for domain in os.listdir(dev_data_directory + "/" + machine):
            input_directory = dev_data_directory + machine + "/" + domain
            output_directory = output_base_directory + machine+'/'+domain
            count = 0
            for filename in os.listdir(input_directory)[start_index:end_index]:
                if filename.endswith(".wav"):
                    file_location = os.path.join(input_directory, filename)
                    sample_name = os.path.splitext(file_location[len(input_directory):])[0]
                    output_location = output_directory + sample_name + ".png"
                    if not os.path.exists(output_directory):
                        os.makedirs(output_directory)
                    convert_to_spectrogram_and_save(file_location, output_location)
                count+=1
