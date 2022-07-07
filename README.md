# AST-based-anomaly-detection-with-GMM

## Directory structure

    .
    ├── ...                                # parent directory containing the project
    |    ├── finetuned_models              # Here the AST models are stored that are finetuned on the DCASE data during step 1   
    |    ├── intermediate_data             # Here the intermediary AST embeddings are stored that serve as input the the GMM model
    |    ├── intermediate_dataframes       # This contains the same information as the directory intermediate_data, but now stored as pandas dataframes
    |    ├── pretrianed_models             # Here the pretrained (on ImageNet/AuioSet) AST models are cached
    |    ├── results   
    |    ├── spectrogram_dataframes        # This contains the same information as the directory spectrograms, but now stored as pandas dataframes
    |    ├── spectrograms                  # The DCASE data transformed to spectrograms (see preprocessing.py)
    |    └── src                      
    └── data
         ├── dev_data              # Development dataset of DCASE dataset should be added here
         |      ├── fan
         |      |    ├── source_test
         |      |    ├── target_test
         |      |    ├── train
         |      ├── gearbox
         |     ...  
         
Note that the directories data, intermediate_data,intermediate_dataframes and pretrianed_models are not present on the git repo.
The directory dev_data (with corresponding data files) needs to be added manually at the correct location shown above.
The directories intermediate_data,intermediate_dataframes and pretrianed_models will be created if not present when generating the intermediate data.
     
## General workflow

main.py invokes the methods of the two parts described below.

### Part 1: AST embedding

The first part generates embeddings originating from withing the AST. 
These embeddings serve as input for the second step where the actual anomaly detection happens.
This can be seen as moving from the raw data feature space (the spectrogram) to an AST embedding feature space.

#### Finetuned AST model generation

Instead of using the vanilla, pretrained model of AST for embedding extraction, the model generation part trains the AST on the DCASE data prior to extracting features.
To do this, the training procedure trains to classify the machine indices within each machine type available in the dataset (see DCASE dataset section).
This way, the AST will be finetuned to classify differences between sections and so will be more data specific as compared to the vanilla AST model that is trained for classification on AudioSet.
In other words, the AST layers are finetuned to the dataset of the task at hand and provide more useful embeddings.
The AST_model_generation directory contains the source code for this finetuning step.
Figure Part1_finetune_block_diagram.PNG shows the finetuning schematically.

#### Data conversion

The first part generates embeddings originating from withing the AST and stores them as intermediary data files.
It also converts the indivual files into pandas dataframes usable by the GMM model
AST_embedder.py is the central file for this step.
AST_embedder.yaml contains settings and parameters. Please increment/change the version number for different parameter settings.

The AST source code is adapted in order to extract embeddings out of the AST structure (see novelties below for more info).

### Part 2: GMM model
This parts defines a GMM model, fits it on the pandas dataframes of the train data and evaluates it on the dataframes of the source and target test sets.
GMM_model.py is the central file for this step.
GMM_model.yaml contains settings and parameters. Please increment/change the version number for different parameter settings.
Note that the AST_embedder version and GMM_model version do not necesarrily have to be the same.
Figure Part2_block_diagram.PNG shows step 2 schematically.
Note that the AST based model from step 1 is kept constant in this step, i.e. not trained.


## Novelties

### 1. AST combination with GMM for anomaly detection

The AST (acoustic spectrogram transfoermer) is a pure attention-based model inspired by ViT and is used for classification. 
Here we use the AST for anomaly detection by using it to generate intermediate embeddings.
These embeddings are then used by the anomaly detector, in this case a GMM.
By doing this we take advantage of the powerful feature extracting properties of the AST model and give an answer to the problem of limited available data.
Figures AST-GMM_structure.png and AST-GMM_structure_block_diagram.png shows exactly where the intermediate embeddings are extracted.

### 2. Adaptions to AST model

The feature extracting models in this work are a variantion of the vanilla AST model. 

- The models have a customizable amount of transformer encoder layers and trainable layers.
- To finetune the AST to the task at hand, the final layer(s) of the MLP are adjusted to take into account the number of classes (here machine indices).
- Finally, to do the actual feature extraction, the MLP head is removed and the output tensors of the transformer encoder are fed to the anomaly detector, in this case a GMM model





