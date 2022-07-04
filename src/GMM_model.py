import os
import numpy as np
import yaml as yaml
import pandas as pd
import common
from sklearn import metrics
from pathlib import Path
from sklearn import mixture


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
result_dir="../results/AST_GMM/"+str(param["version"])+"/"
Path(result_dir).mkdir(parents=True, exist_ok=True)
f_results = open(result_dir+'/accuracies.txt')

f_results.write("hyperparameters:\n" +
                "nb_comp: " + str(param["nb_comp"])+", n_init: "+ str(param["fit"]["n_init"])+\
                ", covariance_type: "+ param["fit"]["cov_type"])
embedding_base_directory=param['embeddings_location']
for machine in param['machine_types']:
    machine_dir = embedding_base_directory + machine
    train_pickle_location = machine_dir + "/train/" + "dataframe.pkl"

    # training
    lables_train, X_train= 'temp' #TODO
    X_train = pd.read_pickle(train_pickle_location)
    model = mixture.GaussianMixture(n_components= param["nb_comp"],n_init=param["fit"]["n_init"],
                                    covariance_type=param["fit"]["cov_type"])
    model.fit(X_train)

    #evaluation
    labels_source_test, X_source_test = 'temp' #TODO
    auc_source,pauc_source = evaluate(labels_source_test, X_source_test, 'source')

    labels_target_test, X_target_test = 'temp' #TODO
    auc_target,pauc_target = evaluate(labels_target_test, X_target_test, 'target')

    f_results.write(str(machine) + ":\n"+
            "AUC source test="  + str(auc_source) + ", pAUC source test=" + str(pauc_source) + "\n"+
            "AUC target test=" + str(auc_target) + ", pAUC target test=" + str(pauc_target) +  "\n")


