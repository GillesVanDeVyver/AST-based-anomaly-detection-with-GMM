import GMM_model
import AST_embedder
from AST_model_generation import preprocessing,AST_training
from attention_visualization import spectrogram_generation,visualize_attention_map

#Part 1

#preprocessing.generate_spectrograms() #only needs to be done once
#preprocessing.generate_dataframes()   #only needs to be done once
#AST_training.train_all_machines()

#Part 2

AST_embedder.prepare_intemediate_data()
#GMM_model.fit_and_eval_all_machines()


# attention visualization
#spectrogram_generation.generate_spectrograms_as_png(0,10)
#visualize_attention_map.generate_attention_maps()