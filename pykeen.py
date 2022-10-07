import sys 
import os

from pykeen.pipeline import pipeline_from_path

model_name = "TransE"
dataset_name = "FB15k"

result = pipeline_from_path(
  path = "./config/"+model_name.lower()+"_"+dataset_name.lower()+".json",
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