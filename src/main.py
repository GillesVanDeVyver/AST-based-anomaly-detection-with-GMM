import GMM_model
import AST_embedder
from AST_model_generation import preprocessing,AST_training


#Part 1

#preprocessing.generate_spectrograms() #only needs to be done once
#preprocessing.generate_dataframes()   #only needs to be done once
#AST_training.train_all_machines()

#Part 2

AST_embedder.prepare_intemediate_data()
#GMM_model.fit_and_eval_all_machines()

