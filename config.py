from os.path import join, realpath, dirname

# path param
BASE_PATH = dirname(realpath(__file__))
DATA_PATH = join(BASE_PATH, 'data')
GENE_PATH = join(DATA_PATH, 'genes')
PATHWAY_PATH = join(DATA_PATH, 'pathways')
REACTOM_PATHWAY_PATH = join(PATHWAY_PATH, 'Reactome')
PROSTATE_DATA_PATH = join(DATA_PATH, 'prostate')
#RUN_PATH = join(BASE_PATH, 'train')
LOG_PATH = join(BASE_PATH, 'logs')
PROSTATE_LOG_PATH = join(LOG_PATH, 'p1000')
PARAMS_PATH = join(DATA_PATH, 'trainparams')
#POSTATE_PARAMS_PATH = join(PARAMS_PATH, 'P1000')
RESULT_PATH = join(BASE_PATH, 'results')

# state param
debug = True  # Is it debug mode
local = False  # Is it remote running
save_train = False  # Whether to save the model or not in result path
save_res = True  # Whether to save evaluation of model or not in result path


# data_access param
selected_genes = 'tcga_prostate_expressed_genes_and_cancer_genes.csv'
data_params = {'id': 'ALL', 'type': 'prostate_paper',
             'params': {
                 'data_type': ['mut_important', 'cnv_del', 'cnv_amp'],
                 'drop_AR': False,
                 'cnv_levels': 3,
                 'mut_binary': True,
                 'balanced_data': False,
                 'combine_type': 'union',  # intersection
                 'use_coding_genes_only': True,
                 'selected_genes': selected_genes,
                 'training_split': 0,
             }
             }

#nn param
n_hidden_layers = 5
base_dropout = 0.5
wregs = [0.001] * 7
loss_weights = [2, 7, 20, 54, 148, 400]
wreg_outcomes = [0.01] * 6
pre = {'type': None}

models_params = {
    'type': 'nn',
    'id': 'P-net',
    'params':
        {
            #'build_fn': build_pnet2,
            'model_params': {
                'trainable_mask': True,
                'full_train': False,
                'use_bias': True,
                'w_reg': wregs,
                'w_reg_outcomes': wreg_outcomes,
                'dropout': [base_dropout] + [0.1] * (n_hidden_layers + 1),
                'loss_weights': loss_weights,
                'optimizer': 'Adam',
                'activation': 'tanh',
                'data_params': data_params,
                'add_unk_genes': False,
                'shuffle_genes': False,
                'kernel_initializer': 'lecun_uniform',
                'n_hidden_layers': n_hidden_layers,
                'attention': False,
                'dropout_testing': False  # keep dropout in testing phase, useful for bayesian inference

            }, 'fitting_params': dict(samples_per_epoch=10,
                                      select_best_model=False,
                                      monitor='val_o6_f1',
                                      verbose=2,
                                      epoch=300,
                                      shuffle=True,
                                      batch_size=50,
                                      save_name='pnet',
                                      debug=False,
                                      save_gradient=False,
                                      class_weight='auto',
                                      n_outputs=n_hidden_layers + 1,
                                      prediction_output='average',
                                      early_stop=False,
                                      reduce_lr=False,
                                      reduce_lr_after_nepochs=dict(drop=0.25, epochs_drop=50),
                                      lr=0.001,
                                      max_f1=True
                                      ),
            'feature_importance': 'deepexplain_deeplift'
        },
}