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

### AST embedding

The first part generates embeddings originating from withing the AST and stores them as intermediary data files.
It also converts the indivual files into pandas dataframes usable by the GMM model
AST_embedder.py is the central file for this step.
AST_embedder.yaml contains settings and parameters. Please increment/change the version number for different parameter settings.

### GMM model
This parts defines a GMM model, fits it on the train pandas dataframes and evaluates it on the source and target test ones.
GMM_model.py is the central file for this step.
GMM_model.yaml contains settings and parameters. Please increment/change the version number for different parameter settings.
Note that the AST_embedder version and GMM_model version do not necesarrily have to be the same

