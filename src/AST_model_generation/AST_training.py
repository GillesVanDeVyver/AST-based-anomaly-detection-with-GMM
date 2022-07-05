# -*- coding: utf-8 -*-
# @Time    : 6/10/21 11:00 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : traintest.py

import sys
import os
import datetime
sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))
import torch
from torch.cuda.amp import autocast,GradScaler
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from sklearn import metrics
import warmup_scheduler
from src import common
import yaml as yaml
from src.models import AST_classifier
from torch import nn
from tqdm import tqdm
import numpy as np

with open("AST_training.yaml") as stream:
    param = yaml.safe_load(stream)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def custom_plot(epochs, loss,figname,debug):
    plt.plot(epochs, loss)
    if debug:
        plt.savefig("debug_results/"+figname+".png")
    else:
        plt.savefig("results/"+figname+".png")

def calc_AUC(audio_model, X,anomaly_labels,one_hot_labels,loss_fn,source,log=True,tb=None,epoch=None,debug=False,max_fpr=0.1,device=device):
    if log and tb==None:
        raise Exception("no tensorboard to log to given")
    losses = []
    i=0
    for sample in X:
        one_hot_labels_sample=torch.tensor([one_hot_labels[i]]).to(device)
        with autocast():
            sample = torch.unsqueeze(sample, dim=0)
            sample=sample.to(device)
            sample_output = audio_model(sample.detach())
            sample_output = sample_output.detach()
            sample_loss = loss_fn(sample_output, one_hot_labels_sample)
        losses.append(1/sample_loss.item())
        i+=1
    auc = metrics.roc_auc_score(anomaly_labels, losses)
    if param['verbose']:
        print("auc: "+str(auc))
    pauc= metrics.roc_auc_score(anomaly_labels, losses,max_fpr=max_fpr)
    if log:
        label = 'target'
        if source:
            label = 'source'
        tb.add_scalar('AUC_scores/'+label, auc, epoch)
        tb.add_scalar('pAUC_scores/'+label, pauc,epoch)
    return losses,auc,pauc

def generate_roc_curve(y,labels,title):
    fpr_source, tpr_source, thresholds_source = metrics.roc_curve(labels, y)
    ROC_location_source="results/"+title+".png"
    common.generate_ROC_curve(fpr_source, tpr_source, ROC_location_source)



def train(machine,debug=False):

    soft= param['soft_labels']

    dataframe_dir = param['spectrogram_dataframes_location'] + machine+"/"

    train_location = dataframe_dir+"train/dataframe.pt"
    if soft:
        train_index_labels_location = dataframe_dir+"train/one_hot_labels_soft.pt"
    else:
        train_index_labels_location = dataframe_dir+"train/one_hot_labels.pt"
    train_anomaly_labels_location = dataframe_dir+"train/anomaly_labels.pt"

    validation_source_location = dataframe_dir+"source_test/dataframe.pt"
    if soft:
        validation_source_index_labels_location = dataframe_dir+"source_test/one_hot_labels_soft.pt"
    else:
        validation_source_index_labels_location = dataframe_dir + "source_test/one_hot_labels.pt"
    validation_source_anomaly_labels_location = dataframe_dir + "source_test/anomaly_labels.pt"

    validation_target_location = dataframe_dir+"target_test/dataframe.pt"
    if soft:
        validation_target_index_labels_location = dataframe_dir + "target_test/one_hot_labels_soft.pt"
    else:
        validation_target_index_labels_location = dataframe_dir + "target_test/one_hot_labels.pt"
    validation_target_anomaly_labels_location = dataframe_dir + "target_test/anomaly_labels.pt"

    X_train = torch.load(train_location)
    X_train_index_labels = torch.load(train_index_labels_location)

    X_validation_source = torch.load(validation_source_location)
    X_validation_source_index_labels = torch.load(validation_source_index_labels_location)
    validation_source_anomaly_labels = torch.load(validation_source_anomaly_labels_location)

    X_validation_target = torch.load(validation_target_location)
    X_validation_target_index_labels = torch.load(validation_target_index_labels_location)
    validation_target_anomaly_labels = torch.load(validation_target_anomaly_labels_location)

    nb_classes = len(X_train_index_labels[0])
    audio_model = AST_classifier.ASTModel(label_dim=nb_classes,input_tdim=param['AST_model']['input_tdim'],
                                          imagenet_pretrain=param['AST_model']['imagenet_pretrain'],
                                          audioset_pretrain=param['AST_model']['audioset_pretrain'],
                                          model_size=param['AST_model']['model_size'],
                                          verbose=param['verbose'],
                                          number_of_layers=param['AST_model']['nb_layers'])

    audio_model = audio_model.to(device)
    audio_model.train()

    model_title = "AST_classifier_"+param['version']
    if param['verbose']:
        print('running on ' + str(device))
    torch.set_grad_enabled(True)



    # Set up the optimizer
    trainables = [p for p in audio_model.parameters() if p.requires_grad]
    if param['verbose']:
        print('Number of trainable parameters is : {:.3f} million'.format(sum(p.numel() for p in trainables) / 1e6))

    optimizer = torch.optim.Adam(trainables,param['AST_model']['lr'])
    if param['AST_model']['warmup']:
        scheduler = warmup_scheduler.WarmupLR(optimizer,warmup_steps=param['AST_model']['warmup_steps'])
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [2,3,4,5], gamma=0.5, last_epoch=-1)
    loss_fn = nn.BCEWithLogitsLoss() #copied from AST
    scaler = GradScaler()
    global_step, epoch = 0, 1

    if debug:
        X_validation_source = X_validation_source[:6]
        X_validation_source_labels = X_validation_source_index_labels[:6]
        X_validation_target = X_validation_target[:6]
        X_validation_target_labels = X_validation_target_index_labels[:6]
    X_train.to(device, non_blocking=True)
    batch_size = param['AST_model']['batch_size']
    nb_batches = round(len(X_train)/batch_size)
    title = model_title+"_"+machine
    if debug:
        log_folder = "runs/debug/"+machine+"/"
    else:
        log_folder = "runs/"+machine+"/"

    tb = SummaryWriter(log_folder+title)
    train_loss_vals=  []
    epoch = 1
    n_epochs=param['AST_model']['n_epochs']
    if debug:
        n_epochs = 5
    while epoch < n_epochs:
        if param['AST_model']['shuffle']:
            X_train=X_train[torch.randperm(X_train.size()[0])]
        if param['verbose']:
            print('---------------')
            print(datetime.datetime.now())
            print("current #epochs=%s, #steps=%s" % (epoch, global_step))
        pos = 0
        epoch_loss= []
        if debug:
            nb_batches = 2
        audio_model.train()
        for i in range(nb_batches):
            if i%10 == 0 & param['verbose']:
                print("current batch: " + str(i))
            if (pos + batch_size>len(X_train)):
                X_batch = X_train[pos:]
                ground_truth_labels_batch=X_train_index_labels[pos:]
            else:
                X_batch = X_train[pos:pos + batch_size]
                ground_truth_labels_batch = X_train_index_labels[pos:pos + batch_size]
            X_batch=X_batch.to(device, non_blocking=True)
            with autocast():
                estimated_labels = audio_model(X_batch)
                ground_truth_labels_batch_tensor=torch.tensor(ground_truth_labels_batch)
                ground_truth_labels_batch_tensor=ground_truth_labels_batch_tensor.to(device)
                loss = loss_fn(ground_truth_labels_batch_tensor, estimated_labels)

            optimizer.zero_grad()
            if device=="cuda":
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            epoch_loss.append(loss.item())
            global_step += 1

        avg_epoch_loss=sum(epoch_loss)/len(epoch_loss)
        train_loss_vals.append(avg_epoch_loss)
        tb.add_scalar('Loss/train', avg_epoch_loss, epoch)
        scheduler.step(avg_epoch_loss)

        audio_model.eval() # log validation accuracy during training (and without interfering with training)


        calc_AUC(audio_model, X_validation_source,validation_source_anomaly_labels
                                                    ,X_validation_source_index_labels,loss_fn,True,
                                                    log=True,tb=tb,epoch=epoch,debug=debug)
        calc_AUC(audio_model, X_validation_target,validation_target_anomaly_labels,
                                                    X_validation_target_index_labels,loss_fn,False,
                                                    log=True,tb=tb,epoch=epoch,debug=debug)
        epoch += 1

    if not debug:
        save_location = param['fine_tuned_models_location']
        if not os.path.exists(save_location):
            os.makedirs(save_location)
        torch.save(audio_model.state_dict(), save_location+title+".pt")

    tb.close()


def train_all_machines(debug=False):

    for machine in tqdm(param['machine_types']):

        train(debug=debug,machine=machine)
        if param['verbose']:
            print(machine + " done")

