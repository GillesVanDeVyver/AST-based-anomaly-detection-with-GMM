# AST-based-anomaly-detection-with-GMM

## Directory structure

    .
    ├── ...                                # parent directory containing the project
    |    ├── intermediate_data             # Here the intermediary AST embeddings are stored that serve as input the the GMM model
    |    ├── intermediate_dataframes       # This contains the same information as in intermediate_data, but now stored as pandas dataframes
    |    ├── pretrianed_models             # Here the pretrained AST models are cached
    |    ├── src                    
    |    └── results   
    └── data
         ├── dev_data              # Development dataset of DCASE dataset should be added here
         |      ├── fan
         |      |    ├── source_test
         |      |    ├── target_test
         |      |    ├── train
         |      ├── gearbox
         |      
         
Note that the directories data, intermediate_data,intermediate_dataframes and pretrianed_models are not present on the git repo.
The directory dev_data (with corresponding data files) needs to be added manually at the correct location shown above.
The directories intermediate_data,intermediate_dataframes and pretrianed_models will be created if not present when generating the intermediate data.
     
## General workflow

main.py invokes the methods of the two parts described below.

### Part 1: AST embedding

The first part generates embeddings originating from withing the AST and stores them as intermediary data files.
It also converts the indivual files into pandas dataframes usable by the GMM model
AST_embedder.py is the central file for this step.
AST_embedder.yaml contains settings and parameters. Please increment/change the version number for different parameter settings.

The AST source code is adapted in order to extract embeddings out of the AST structure (see novelties below for more info).

#### Finetuned AST model generation
Instead of using the vanilla, pretrained model of AST, the model generation part trains the AST on the DCASE data.
To do this the training procedure uses outlier detection, i.e. the model trains to classify the machine indices within each machine type available in the dataset.
This way, the AST layers is finetuned to the dataset of the task at hand and provide more usefule embeddings.

### Part 2: GMM model
This parts defines a GMM model, fits it on the train pandas dataframes and evaluates it on the source and target test ones.
GMM_model.py is the central file for this step.
GMM_model.yaml contains settings and parameters. Please increment/change the version number for different parameter settings.
Note that the AST_embedder version and GMM_model version do not necesarrily have to be the same.


## Novelties

### 1. AST for anomaly detection

The AST (acoustic spectrogram transfoermer) is a pure attention-based model inspired by ViT and is used for classification. 
Here we use the AST for anomaly detection by using it to generate intermediate embeddings.
By doing this we take advantage of the powerful feature extracting properties of the AST model.

### 2. Adaptions to AST model

The feature extracting models in this work are a variantion of the vanilla AST model. 
The models have a customizable amount of transformer encoder layers and trainable layers.
To finetune the AST to the task at hand, the final layer(s) of the MLP are adjusted to take into account the number of classes (here machine indices).

Finally, to do the actual feature extraction, the MLP head is removed and the output tensors of the transformer encoder are fed to the anomaly detector, in this case a GMM model





