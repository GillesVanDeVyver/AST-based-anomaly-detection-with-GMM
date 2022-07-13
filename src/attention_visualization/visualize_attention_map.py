# large parts copied from https://gist.github.com/zlapp/40126608b01a5732412da38277db9ff5
import sys

import yaml
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
from PIL import Image
import gc
from AST_based_anomaly_detection_with_GMM.src.models import AST_based_model


wd=os.path.dirname(__file__)
with open(os.path.join(wd,"visualize_attention_map.yaml")) as stream:
    param = yaml.safe_load(stream)

class AST_attention_map_generator():
    def __init__(self):

        self.attention_maps_loaction=os.path.join(wd,param['attention_maps_loaction'],"v_"+param['version']+"/")
        if not os.path.exists(self.attention_maps_loaction):
            os.makedirs(self.attention_maps_loaction)
        f_info = open(self.attention_maps_loaction + 'version_info.txt', 'w')
        f_info.write("hyperparameters:\n" +
                     "nb_layers: " + str(param['ast_model']['nb_layers']) +
                     ", input_tdim: " + str(param['ast_model']['input_tdim']) +
                     ", num_mel_bins: " + str(param['ast_model']['num_mel_bins']) +
                     ", imagenet_pretrain: " + str(param['ast_model']['imagenet_pretrain']) +
                     ", audioset_pretrain: " + str(param['ast_model']['audioset_pretrain']) +
                     ", embedding_dimension: " + str(param['ast_model']['embedding_dimension']) +
                     ", finetuned_version: " + str(param['ast_model']['finetuned_version']))
        f_info.close()

    def get_attention_map(self,input_location,img, model,get_mask=False):
        att_mat = model.get_ast_embedding_single_file(input_location)
        att_mat = torch.stack(att_mat).squeeze(1)
        att_mat=att_mat.to('cpu')
        att_mat = torch.mean(att_mat, dim=1)

        # To account for residual connections, we add an identity matrix to the
        # attention matrix and re-normalize the weights.
        residual_att = torch.eye(att_mat.size(1))
        aug_att_mat = att_mat + residual_att

        aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

        # Recursively multiply the weight matrices
        joint_attentions = torch.zeros(aug_att_mat.size())
        joint_attentions[0] = aug_att_mat[0]

        for n in range(1, aug_att_mat.size(0)):
            joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n - 1])
        v = joint_attentions[-1]
        mask = (v[0,2:].detach().numpy()+v[1,2:].detach().numpy())/2
        mask = mask.reshape(12, 101)
        mask[:2,:]=mask[:2,:]/10 # adjust for the fact that high up in the frequency spectrum the values are lower
        #mask = mask[:,:-2] # adjust for padding
        mask = mask[2:,:-2]
        if get_mask:
            result = cv2.resize(mask / mask.max(), img.size)
        else:
            mask = cv2.resize(mask / mask.max(), img.size)[..., np.newaxis]
            mask_avg = np.average(mask)
            mask=mask/mask_avg
            result = (mask * img).astype("uint8")
            for row in result:
                for culumn in row:
                    for i in range(3):
                        if culumn[i]>=255:
                            culumn[i]=255
                    culumn[3]=255
        return result

    def plot_attention_map(self,original_img, att_map):
        plt.figure()
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 16))
        ax1.set_title('Original')
        ax2.set_title('Attention Map Last Layer')
        _ = ax1.imshow(original_img)
        _ = ax2.imshow(att_map)

    def plot_attention_map_and_safe(self,original_img, att_map,output_location):
        plt.figure()
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 16))
        ax1.set_title('Original')
        ax2.set_title('Attention Map Last Layer')
        _ = ax1.imshow(original_img)
        _ = ax2.imshow(att_map)
        plt.savefig(os.path.join(output_location))
        fig.clf()
        plt.clf()
        plt.close("all")
        del att_map
        del original_img
        del fig
        gc.collect()

    # generate attation maps of the samples
    def generate_attention_maps(self):
        adast_mdl = AST_based_model()
        base_directory_spectrograms=os.path.join(wd, param['spectrogram_as_png_location'])
        input_directory= os.path.join(wd,param['dev_data_location'])
        output_base_directory=os.path.join(wd,param['./attention_maps'])
        for machine in param['machine_types']:
            for domain in os.listdir(input_directory + "/" + machine):
                input_directory_specrograms = base_directory_spectrograms + machine + "/" + domain
                input_directory = input_directory + machine + "/" + domain
                output_directory = output_base_directory + machine+'/'+domain
                for filename in os.listdir(input_directory_specrograms):
                    if filename.endswith(".png"):
                        wav_file_location = os.path.join(input_directory, filename[:-3] + "wav")
                        png_file_location = os.path.join(input_directory_specrograms, filename)
                        sample_name = os.path.splitext(wav_file_location[len(input_directory):])[0]
                        output_location = output_directory + sample_name + "_attention_map.png"
                        img=Image.open(png_file_location)
                        result = self.get_attention_map(wav_file_location,img,adast_mdl)
                        self.plot_attention_map_and_safe(img, result,output_location)


def generate_attention_maps():
    generator = AST_attention_map_generator()
    #TODO