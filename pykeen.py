import sys 
import os

from pykeen.pipeline import pipeline_from_path

model_name = sys.argv[1]
dataset_name = sys.argv[2]

config_path = "/home/ferrari/kge_pykeen/config/"+model_name.lower()+"_"+dataset_name.lower()

if os.path.isfile(config_path+".yaml"):
  config_path = config_path+".yaml"
else:
  config_path = config_path+".json"

result = pipeline_from_path(
  path = config_path,
  # other kargs
  device="cuda",
  use_tqdm=True,
  result_tracker='wandb',
  result_tracker_kwargs=dict(
    project=dataset_name,
    notes=model_name,
  )  
)

save_location = "./results_"+model_name+"_"+dataset_name  # this directory
result.save_to_directory(save_location)
os.listdir(save_location) 