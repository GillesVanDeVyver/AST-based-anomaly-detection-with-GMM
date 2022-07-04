import os
import numpy as np
import yaml as yaml
import pandas as pd
import common
from sklearn import metrics
from pathlib import Path
from sklearn import mixture
import AST_embedder
from tqdm import tqdm
import numpy as np

def evaluate(labels,X,domain,generate_ROC_curve=True):
    prediction = -model.score_samples(X) # higher value means more 'normal' => - for anomaly score
    auc = metrics.roc_auc_score(labels, prediction)
    pauc= metrics.roc_auc_score(labels, prediction,max_fpr=param["max_fpr"])
    fpr, tpr, thresholds = metrics.roc_curve(labels, prediction)
    if generate_ROC_curve:
        ROC_location=result_dir+"ROC_curve_"+domain+"_"+str(machine)+".png"
        common.generate_ROC_curve(fpr, tpr, ROC_location)
    return auc,pauc


with open("GMM_model.yaml") as stream:
    param = yaml.safe_load(stream)

with open("AST_embedder.yaml") as stream:
    param_ast_embeddings = yaml.safe_load(stream)

verbose= param['verbose']

result_dir="../results/AST_v"+str(param_ast_embeddings["version"])+"-GMM_v"+str(param["version"])+"/"
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
f_results = open(result_dir+'accuracies.txt',"w")
f_parameters = open(result_dir+'hyperparameters.txt',"w")
f_parameters.write("hyperparameters:\n" +
                "nb_comp: " + str(param["fit"]["nb_comp"])+", n_init: "+ str(param["fit"]["n_init"])+\
                ", covariance_type: "+ param["fit"]["cov_type"])
f_parameters.close()
embedding_base_directory=param_ast_embeddings['embeddings_base_location']+"_v"+param_ast_embeddings["version"]+"/"
dataframes_base_directory = param_ast_embeddings['dataframes_base_location'] + "_v" + param_ast_embeddings['version'] + "/"

for machine in tqdm(param['machine_types']):
    if verbose:
        print("Starting training GMM model for "+machine)
    #machine_dir = embedding_base_directory + machine
    X_train = pd.read_pickle(dataframes_base_directory+machine+"/train.pkl")
    lables_train=np.load(dataframes_base_directory+machine+"/train_labels.npy")
    # training
    model = mixture.GaussianMixture(n_components= param["fit"]["nb_comp"],n_init=param["fit"]["n_init"],
                                    covariance_type=param["fit"]["cov_type"])
    model.fit(X_train)

    #evaluation
    X_source_test = pd.read_pickle(dataframes_base_directory+machine+"/source_test.pkl")
    labels_source_test=np.load(dataframes_base_directory+machine+"/source_test_labels.npy")
    auc_source,pauc_source = evaluate(labels_source_test, X_source_test, 'source')

    X_target_test = pd.read_pickle(dataframes_base_directory+machine+"/target_test.pkl")
    labels_target_test=np.load(dataframes_base_directory+machine+"/target_test_labels.npy")
    auc_target,pauc_target = evaluate(labels_target_test, X_target_test, 'target')

    f_results.write(str(machine) + ":\n"+
            "AUC source test="  + str(auc_source) + ", pAUC source test=" + str(pauc_source) + "\n"+
            "AUC target test=" + str(auc_target) + ", pAUC target test=" + str(pauc_target) +  "\n")

f_results.close()
