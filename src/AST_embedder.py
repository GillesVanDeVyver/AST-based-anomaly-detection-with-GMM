import os, sys
import torch
import common
import pandas as pd
from models import AST_based_embedding_extractor
import yaml as yaml
from tqdm import tqdm
import numpy as np

with open("AST_embedder.yaml") as stream:
    param = yaml.safe_load(stream)

class ast_embedder(): # anomaly detection ast_model
    def __init__(self,verbose=True):

        # adapted version of AST were we skip the MLP head and adjust the number of layers
        audio_model_ast = AST_based_embedding_extractor.ASTModel(input_tdim=param['ast_model']['input_tdim'],
                                imagenet_pretrain=param['ast_model']['imagenet_pretrain'],
                                audioset_pretrain=param['ast_model']['audioset_pretrain'],
                                verbose=True, number_of_layers = param['ast_model']['nb_layers'])

        self.input_tdim = param['ast_model']['input_tdim']
        self.audio_model = torch.nn.DataParallel(audio_model_ast)
        self.num_mel_bins=param['ast_model']['num_mel_bins']
        self.embedding_dimension=param['ast_model']['embedding_dimension']
        self.nb_layers=param['ast_model']['nb_layers']
        self.embedding_base_directory=param['embeddings_base_location']+"_v"+param['version']+"/"
        self.dataframes_base_directory=param['dataframes_base_location']+"_v"+param['version']+"/"
        self.verbose=verbose

        if not os.path.exists(self.embedding_base_directory):
            os.makedirs(self.embedding_base_directory)
        if not os.path.exists(self.dataframes_base_directory):
            os.makedirs(self.dataframes_base_directory)

        f_info = open(self.embedding_base_directory + 'version_info.txt','w')
        f_info.write("hyperparameters:\n" +
                        "nb_layers: " + str(param['ast_model']['nb_layers']) +
                        ", input_tdim: " + str(param['ast_model']['input_tdim']) +
                        ", num_mel_bins: " + str(param['ast_model']['num_mel_bins']) +
                        ", imagenet_pretrain: " + str(param['ast_model']['imagenet_pretrain']) +
                        ", audioset_pretrain: " + str(param['ast_model']['audioset_pretrain']) +
                        ", embedding_dimension: " + str(param['ast_model']['embedding_dimension']))

    def get_ast_embedding_single_file(self,file_location,device):
        log_mel = common.convert_to_log_mel(file_location, num_mel_bins=self.num_mel_bins, target_length=self.input_tdim)
        input = torch.unsqueeze(log_mel, dim=0)
        input=input.to(device)
        self.audio_model=self.audio_model.to(device)
        output = self.audio_model(input)
        return output

    def generate_and_save_embeddings(self,input_location,output_directory,sample_name,device):
        output = self.get_ast_embedding_single_file(input_location,device)
        output_location = output_directory + sample_name + ".pt"
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        torch.save(output,output_location)

    def generate_intermediate_tensors(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dev_data_directory=param['dev_data_location']
        for machine in tqdm(param['machine_types']):
            for domain in tqdm(os.listdir(dev_data_directory+"/"+machine)):
                if self.verbose:
                    print("generating intermediary files for "+machine+" " + domain)
                input_directory = dev_data_directory + machine + "/" + domain
                output_directory = self.embedding_base_directory + machine+'/'+domain
                for filename in os.listdir(input_directory):
                    if filename.endswith(".wav"):
                        file_location = os.path.join(input_directory, filename)
                        sample_name = os.path.splitext(file_location[len(input_directory):])[0]
                        self.generate_and_save_embeddings(file_location,output_directory,sample_name,device)
                print(machine+" "+domain+" done")

    def generate_dataframes(self,format="GMM",debug=False):
        tensors_in_domain = None
        lables = []
        for machine in tqdm(param['machine_types']):
            machine_dir=self.embedding_base_directory+"/"+machine
            for domain in tqdm(os.listdir(machine_dir)):
                domain_dir = machine_dir+"/"+domain
                if debug:
                    files = os.listdir(domain_dir)[:5]
                    print("WARNING: debug mode is on")
                else:
                    files=os.listdir(domain_dir)
                for filename in files:
                    if filename.endswith(".pt"):
                        if format=="one_class_svm":
                            if "anomaly" in filename:
                                lables.append(-1)
                            else:
                                lables.append(1)
                        elif format=="GMM":
                            if "anomaly" in filename:
                                lables.append(1)
                            else:
                                lables.append(0)
                        file_location = domain_dir + "/" + filename
                        loaded_tensor = torch.load(file_location)
                        if tensors_in_domain == None:
                            tensors_in_domain = loaded_tensor
                        else:
                            loaded_tensor.to("cpu")
                            tensors_in_domain.to('cpu')
                            tensors_in_domain = torch.cat((tensors_in_domain, loaded_tensor))
                px = pd.DataFrame(tensors_in_domain.detach().cpu().numpy())
                output_dir = self.dataframes_base_directory + machine + '/'
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                output_location = output_dir + domain+".pkl"
                px.to_pickle(output_location)
                output_location_labels=output_dir + domain+"_labels.npy"
                np.save(output_location_labels,np.array(lables))


def prepare_intemediate_data(debug=False):
    embedder=ast_embedder()
    embedder.generate_intermediate_tensors()
    print("Done generating intermediate embeddings")
    print("Converting intermediary data files into dataframes")
    embedder.generate_dataframes(debug=debug)





