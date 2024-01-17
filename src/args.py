import argparse
import yaml
from omegaconf import OmegaConf
def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def load_args_from_yaml(yaml_file, parser):
    # Read YAML file
    with open(yaml_file, 'r') as stream:
        try:
            args_from_yaml = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            return

    # Flatten the hierarchical structure of the YAML args
    flat_args = flatten_dict(args_from_yaml)

    # Update the parser with the flattened arguments
    parser.set_defaults(**flat_args)
def get_args(cwd):
    
    parser = argparse.ArgumentParser()
    
    # Dataset and Task Configuration
    parser.add_argument('--dataset', default='AWA1', type=str, help="Name of the dataset")
    parser.add_argument('--seen_classes', default=8, type=int, help="Number of seen classes per task")
    parser.add_argument('--novel_classes', default=2, type=int, help="Number of novel classes per task")
    parser.add_argument('--num_tasks', default=5, type=int, help="Number of continual learning tasks")
    parser.add_argument('--all_classes', default=50, type=int, help="Total classes in the dataset")
    parser.add_argument('--image_embedding', default='res101', help="Type of features used")
    parser.add_argument('--feature_size', default=2048, type=int, help="Size of the feature vector")
    parser.add_argument('--class_embedding', default='att', help="Type of embeddings for semantic information")
    parser.add_argument('--attribute_size', default=85, type=int, help="Size of the attribute vector")
    parser.add_argument('--preprocessing', action='store_true', default=False, help='Enable MinMaxScaler on features')
    parser.add_argument('--custom', default=False, action='store_true', help="Customize class number and task number")

    # Directory and Data Configuration
    parser.add_argument('--data_dir', default=f'{cwd}/data', type=str, help="Path to the data directory")
    parser.add_argument('--matdataset', default=True, help='Data in MATLAB format')
    parser.add_argument('--dataroot', default=cwd + '/data', help='Path to dataset')
    
    

    # Learning Parameters
    parser.add_argument('--d_lr', type=float, default=0.005, help="Discriminator learning rate")
    parser.add_argument('--g_lr', type=float, default=0.005, help="Generator learning rate")
    parser.add_argument('--dic_lr', type=float, default=0.05, help="Dictionary learning rate")
    parser.add_argument('--t', type=float, default=10.0, help="Cosine similarity loss coefficient")
    parser.add_argument('--alpha', type=float, default=1.0, help="Classification loss coefficient")
    parser.add_argument('--Neighbors', type=int, default=3, help="Neighbors in semantic similarity measure")

    # Experimental Configuration
    parser.add_argument("--seed", type=int, default=2222, help= "Random seed for reproducibility")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for seen classes")
    parser.add_argument('--unsn_batch_size', type=int, default=16, help="Batch size for unseen classes")
    parser.add_argument('--epochs', type=int, default=50, help="Training epochs")
    parser.add_argument('--validation', action='store_true', default=False, help='Enable cross-validation mode')

    # Method
    parser.add_argument('--attribute_generation_method', default='interpolation', choices=["interpolation", "learnable", "none"], type=str, help="Attribute generation method")
    parser.add_argument('--creative_weight', type=float, default=0.1, help='Weight for creative loss')
    parser.add_argument("--grw_creative_weight", type=float, default=0.1, help="Weight for creative loss in GRW")
    parser.add_argument("--corr_weight", type=float, default=1, help="Weight of correlation loss for LSR")
    parser.add_argument("--rw_steps", type=int, default=3, help="Random walk steps")
    parser.add_argument("--buffer_size", type=int, default=5000, help="Memory replay buffer size")
    parser.add_argument("--decay_coef", type=float, default=0.7, help="Decay coefficient for random walk")
    
    # Log
    parser.add_argument('--run_name', default='testjob', type=str, help="Name of the experiment")
    parser.add_argument('--wandb_log', action='store_true', default=False, help="Name of the experiment")
    
    # reproduce
    parser.add_argument("--load_best_hp", action='store_true', default=True, help="Load the best hyperparameters")

    # load_args_from_yaml("rw_config.yaml",parser)

    opt = parser.parse_args()
    
    
    
    # Set dataset-specific parameters
    if not opt.custom:
        if opt.dataset in ['AWA1', 'AWA2']:
            opt.seen_classes = 10
            opt.novel_classes = 10
            opt.num_tasks = 5
            opt.all_classes = 50
            opt.attribute_size = 85

        elif opt.dataset == 'SUN':
            opt.seen_classes = 47
            opt.novel_classes = 47
            opt.num_tasks = 15
            opt.all_classes = 717
            opt.attribute_size = 102

        elif opt.dataset == 'CUB':
            opt.seen_classes = 10
            opt.novel_classes = 10
            opt.num_tasks = 20
            opt.all_classes = 200
            opt.attribute_size = 312
            
    # Update args 
    
    # Check and load best hyperparameters if required
    if opt.load_best_hp:
        # Load hyperparameters from a YAML file
        hp_file = f'best_param/{opt.dataset}_{opt.attribute_generation_method}.yaml'
        with open(hp_file, 'r') as file:
            best_hp = yaml.safe_load(file)
    else:
        best_hp = {}
            
        # Update the opt with loaded hyperparameters
        
        # opt.creative_weight = best_hp.get('creative_weight', opt.creative_weight)
        # opt.grw_creative_weight = best_hp.get('grw_creative_weight', opt.grw_creative_weight)
        # opt.rw_steps = best_hp.get('rw_steps', opt.rw_steps)
        # opt.decay_coef = best_hp.get('decay_coef', opt.decay_coef)
    opt = vars(opt)
    opt.update(best_hp)
    # # load random walk config 
    with open("rw_config.yaml", "r") as f:
        rw_config = yaml.load(f, Loader=yaml.Loader)

    # Update opt with values from rw_config
    opt.update(rw_config)


    opt = OmegaConf.create(opt)


    

    # overwrite specific values from opt to rw_config
    opt.rw_params.num_steps = opt.rw_steps
    opt.loss_weights.gen.creative = opt.grw_creative_weight
    opt.rw_params.decay_coef = opt.decay_coef


    
    return opt