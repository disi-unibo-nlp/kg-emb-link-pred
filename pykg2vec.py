import sys

from pykg2vec.data.kgcontroller import KnowledgeGraph
from pykg2vec.common import Importer, KGEArgParser
from pykg2vec.utils.trainer import Trainer
from pykg2vec.utils.logger import Logger

model_name = sys.argv[1]
dataset_name = sys.argv[2]

_logger = Logger().get_logger(__name__)

args = KGEArgParser().get_args([])

args.model_name = model_name

args.hp_abs_file= "./hyperparams/"+model_name+".yaml"
args.exp = True
args.plot_embedding = True
args.device = 'cuda'

_logger.info("Preparing data..")
# Preparing data and cache the data for later usage
knowledge_graph = KnowledgeGraph(dataset=args.dataset_name, custom_dataset_path=args.dataset_path)
knowledge_graph.prepare_data()
knowledge_graph.dump()

_logger.info("Start extracting config..")
# Extracting the corresponding model config and definition from Importer().
config_def, model_def = Importer().import_model_config(model_name.lower())
config = config_def(args)
model = model_def(**config.__dict__)
_logger.info("End extracting config")

config_param = []
config_wn = {}
if model_name=="TransE":
    config_param = ["epochs", "learning_rate", "l1_flag", "hidden_size", "batch_size", "margin"]
    for p in config_param:
        config_wn[p] = config.__dict__[p]
elif model_name=="QuatE":
    config_param = ["learning_rate","l1_flag", "hidden_size", "batch_size", "epochs", "margin", "optimizer", "sampling", "lmbda", "alpha"]
    for p in config_param:
        config_wn[p] = config.__dict__[p]
elif model_name=="DualE":
    config_param = ["learning_rate","l1_flag", "hidden_size", "batch_size", "epochs", "margin", "optimizer", "sampling", "lmbda", "alpha"]
    for p in config_param:
        config_wn[p] = config.__dict__[p]
elif model_name=="InteractE":
    config_param = ["learning_rate", "l1_flag", "feature_permutation", "num_filters", "kernel_size", "reshape_height", "reshape_width", 
                    "input_dropout", "feature_map_dropout", "hidden_dropout", "label_smoothing", "batch_size", "optimizer"]
    for p in config_param:
        config_wn[p] = config.__dict__[p]
elif model_name=="ConvE":
    config_param = ["learning_rate", "l1_flag", "hidden_size", "hidden_size_1", "input_dropout", "feature_map_dropout", 
                      "hidden_dropout", "label_smoothing", "batch_size", "optimizer"]
    for p in config_param:
        config_wn[p] = config.__dict__[p]

wandb.init(project=dataset_name, config=dict(config_wn) )

wandb.run.name = model_name

# Create, Compile and Train the model. While training, several evaluation will be performed.
print("Start training...")
trainer = Trainer(model, config)
trainer.build_model()
trainer.train_model()
