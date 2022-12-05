#import numpy as np
#timestamp generate
from datetime import datetime
TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")

#GCP
from google.cloud import storage as gcs
import google.cloud.aiplatform as aip
from typing import NamedTuple


# kubeflow SDK
import kfp
from kfp.v2 import dsl
from kfp.v2.dsl import (Artifact, Dataset, Input, InputPath, Model, Output,
                        OutputPath, component)


### 学習に必要なパラメータ。いずれはコマンドラインから取れるように。 ##############
params = {
    "project_id": "test-hyron",
    "bucket_name": "aruha-mnist",
    "pipeline_root": "gs://aruha-mnist/pipeline_log", 
    
    "model_url": "gs://aruha-mnist/model",
    "code_url": "gs://aruha-mnist/code/components_script.py",
    #"result_url":"gs://aruha-mnist/result/result.csv",
    
    "train_url":"gs://aruha-mnist/data/mnist_train.csv",
    "test_url" :"gs://aruha-mnist/data/mnist_test.csv",
    
    
    "valid_proportion": 0.2,
    "TIMESTAMP": TIMESTAMP,
    
    "epochs" : 5,
    "batch_size" : 32
}
############################################################################

### GCSから学習用のコードをDL ########################
### もっといい方法あれば... ##########################

project_id = params["project_id"]
client = gcs.Client(params["project_id"])
bucket = client.get_bucket(params["bucket_name"])

def data_load_from_url(path_):
    path_ = path_.split("gs://" + params["bucket_name"] + "/")[1]
    blob = bucket.blob(path_)
    content = blob.download_to_filename("components_script.py")

data_load_from_url(params["code_url"])
####################################################

import sys
from components_script import preprocess, train_model, test_model
from kfp.v2 import compiler  


# パイプラインを定義
@dsl.pipeline(
    pipeline_root=params["pipeline_root"],
    name="example-pipeline",
) 

def pipeline(
    project_id: str,
    bucket_name: str,
    pipeline_root: str, 
    
    model_url: str,
    code_url: str,
    
    train_url:str,
    test_url :str,
    
    valid_proportion: float,
    TIMESTAMP: str,
    
    batch_size: int,
    epochs : int
):
    # データの前処理など行う
    preprocess_task = preprocess(
        project_id, 
        bucket_name, 
        train_url,
        test_url,
        valid_proportion
    )
    
    
    # モデルのtrainを行う
    train_model_task = train_model(
        train_dataset = preprocess_task.outputs["train_dataset"],
        valid_dataset = preprocess_task.outputs["valid_dataset"],

        batch_size    = batch_size,
        epochs        = epochs
    )

    # モデルのtestを行う
    test_model_task = test_model(
        test_dataset = preprocess_task.outputs["test_dataset"], 
        model        = train_model_task.outputs["model"],
        timestamp = params["TIMESTAMP"],
        model_name = "mnist_cnn",
        param_1    =  1.0,
    )
    



    
if __name__ == '__main__':
    compiler.Compiler().compile(
        pipeline_func=pipeline, package_path="test_pipeline.json"
    )


    DISPLAY_NAME = "test" + TIMESTAMP

    job = aip.PipelineJob(
        display_name=DISPLAY_NAME,
        template_path="test_pipeline.json",
        pipeline_root=params["pipeline_root"],
        parameter_values= params,
    )

    job.run()
