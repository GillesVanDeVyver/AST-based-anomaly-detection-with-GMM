import GMM_model
import AST_embedder


AST_embedder.prepare_intemediate_data()

GMM_model.fit_and_eval_all_machines()

