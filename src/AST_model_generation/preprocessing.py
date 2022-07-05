import os
from src import common
import torch
import pandas as pd
import re
import yaml as yaml
from tqdm import tqdm
import numpy as np

with open("preprocessing.yaml") as stream:
    param = yaml.safe_load(stream)


# compute and save spectrograms of raw wav data
def generate_spectrograms():
    input_base_directory = param['dev_data_location']
    output_base_directory= param['spectrograms_location']
    if param['verbose']:
        print('Generating spectrograms')
    for machine in tqdm(os.listdir(input_base_directory)):
        for domain in os.listdir(input_base_directory+"/"+machine):
            input_directory = input_base_directory + machine + "/" + domain
            output_directory = output_base_directory + machine+'/'+domain
            if not os.path.exists(output_directory):
                os.makedirs(output_directory)
            for filename in os.listdir(input_directory):
                if filename.endswith(".wav"):
                    file_location = os.path.join(input_directory, filename)
                    sample_name = os.path.splitext(file_location[len(input_directory):])[0]
                    output_location = output_directory + sample_name + ".pt"
                    log_mel = common.convert_to_log_mel(file_location)
                    torch.save(log_mel, output_location)

def convert_to_one_hot(X,nb_sections):
    one_hot_labels_soft = np.full((len(X), nb_sections), param["epsilon"], dtype=float)
    one_hot_labels = np.zeros((len(X), nb_sections), dtype=float)
    i = 0
    for label in X:
        one_hot_labels_soft[i][label]=1-(nb_sections - 1) * param["epsilon"]
        one_hot_labels[i][label] = 1
        i += 1
    return one_hot_labels,one_hot_labels_soft


# generate dataframes with same spectrogram data and the index labels for outlier exposure
# (TODO make more efficient)
def generate_dataframes():
    input_base_directory= param['spectrograms_location']
    output_base_directory = param['spectrogram_dataframes_location']
    for machine in tqdm(param['machine_types']):
        for domain in os.listdir(input_base_directory+"/"+machine):
            tensors_in_domain = None
            anomaly_lables = []
            index_lables=[]
            if param['verbose']:
                print("starting " + machine+" "+domain)
            input_directory = input_base_directory + machine + "/" + domain
            output_directory = output_base_directory + machine+'/'+domain+"/"
            if not os.path.exists(output_directory):
                os.makedirs(output_directory)
            for filename in tqdm(os.listdir(input_directory)):
                if filename.endswith(".pt"):
                    file_location = os.path.join(input_directory, filename)
                    if "anomaly" in filename:
                        anomaly_lables.append(1)
                    else:
                        anomaly_lables.append(0)
                    section_as_str = filename[9]
                    section = int(section_as_str)
                    index_lables.append(section)
                    loaded_tensor = torch.unsqueeze(torch.load(file_location),0)
                    if tensors_in_domain == None:
                        tensors_in_domain = loaded_tensor
                    else:
                        tensors_in_domain = torch.cat((tensors_in_domain, loaded_tensor))
            output_location_dataframe = output_directory + "dataframe.pt"
            output_location_one_hot_labels = output_directory + "one_hot_labels.pt"
            output_location_one_hot_labels_soft = output_directory + "one_hot_labels_soft.pt"
            output_location_anomaly_labels = output_directory + "anomaly_labels.pt"
            output_location_indices = output_directory + "index_labels.pt"
            nb_classes = max(index_lables) + 1
            one_hot_labels,one_hot_labels_soft=convert_to_one_hot(index_lables,nb_classes)
            torch.save(one_hot_labels, output_location_one_hot_labels)
            torch.save(one_hot_labels_soft, output_location_one_hot_labels_soft)
            torch.save(index_lables, output_location_indices)
            torch.save(tensors_in_domain.detach(),output_location_dataframe)
            torch.save(anomaly_lables,output_location_anomaly_labels)

