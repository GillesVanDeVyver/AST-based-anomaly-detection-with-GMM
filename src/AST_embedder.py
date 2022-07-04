import os, sys
import torch
import common
import pandas as pd
from models import ast_based_model
import yaml as yaml

with open("AST_embedder.yaml") as stream:
    param = yaml.safe_load(stream)

class ast_embedder(): # anomaly detection ast_model
    def __init__(self,verbose=True):

        # adapted version of AST were we skip the MLP head and adjust the number of layers
        audio_model_ast = ast_based_model.ASTModel(input_tdim=param['ast_model']['input_tdim'],
                                imagenet_pretrain=param['ast_model']['imagenet_pretrain'],
                                audioset_pretrain=param['ast_model']['audioset_pretrain'],
                                verbose=True, number_of_layers = param['ast_model']['nb_layers'])

        self.input_tdim = param['ast_model']['input_tdim']
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #device = torch.device("cuda")
        self.audio_model = torch.nn.DataParallel(audio_model_ast)
        self.num_mel_bins=param['ast_model']['num_mel_bins']
        self.embedding_dimension=param['ast_model']['embedding_dimension']
        self.nb_layers=param['ast_model']['nb_layers']
        self.embedding_base_directory=param['embeddings_base_location']+"_v0.1/"
        self.verbose=verbose

        if not os.path.exists(self.embedding_base_directory):
            os.makedirs(self.embedding_base_directory)

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
        output_base_directory=self.embedding_base_directory
        for machine in param['machine_types']:
            for domain in os.listdir(dev_data_directory+"/"+machine):
                if self.verbose:
                    print("generating intermediary files for "+machine+" " + domain)
                input_directory = dev_data_directory + machine + "/" + domain
                output_directory = output_base_directory + machine+'/'+domain
                for filename in os.listdir(input_directory):
                    if filename.endswith(".wav"):
                        file_location = os.path.join(input_directory, filename)
                        sample_name = os.path.splitext(file_location[len(input_directory):])[0]
                        self.generate_and_save_embeddings(file_location,output_directory,sample_name,device)
                print(machine+" "+domain+" done")



def generate_lables_and_pandas_dataframe(input_directory,format="GMM"):
    tensors_in_domain = None
    lables = []
    for filename in os.listdir(input_directory):
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
            file_location = input_directory + "/" + filename
            loaded_tensor = torch.load(file_location)
            if tensors_in_domain == None:
                tensors_in_domain = loaded_tensor
            else:
                loaded_tensor.to("cpu")
                tensors_in_domain.to('cpu')
                tensors_in_domain = torch.cat((tensors_in_domain, loaded_tensor))
    px = pd.DataFrame(tensors_in_domain.detach().cpu().numpy())
    return lables,px


def save_as_dataframe(embedding_base_directory,verbose=True):
    for machine in param['machine_types']:
        if os.path.isdir(embedding_base_directory+"/"+machine):
            if verbose:
                print(machine)
            for domain in os.listdir(embedding_base_directory + "/" + machine):
                input_directory = embedding_base_directory + machine + "/" + domain
                if os.path.isdir(input_directory):
                    X=common.load_embeddings(input_directory)
                    pickle_location=input_directory+"/"+"dataframe.pkl"
                    X.to_pickle(pickle_location)
                if verbose:
                    print(machine+" "+domain+" done")
    #common.combine_embeddings(embedding_base_directory)



def prepare_intemediate_data():
    embedder=ast_embedder()
    embedder.generate_intermediate_tensors()
    print("done generating intermediate embeddings")

    #save_as_dataframe(embedding_base_directory)

prepare_intemediate_data()




