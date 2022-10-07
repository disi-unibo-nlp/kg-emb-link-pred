import sys

from pykg2vec.data.kgcontroller import KnowledgeGraph
from pykg2vec.common import Importer, KGEArgParser
from pykg2vec.utils.trainer import Trainer
from pykg2vec.utils.logger import Logger

model_name = "TransE"
dataset_name = "fb15k"

_logger = Logger().get_logger(__name__)

args = KGEArgParser().get_args([])

args.model_name = model_name

args.hp_abs_file= "/home/ferrari/kge/hyperparams/"+model_name+".yaml"
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

# Create, Compile and Train the model. While training, several evaluation will be performed.
print("Start training...")
trainer = Trainer(model, config)
trainer.build_model()
trainer.train_model()
