#import numpy as np
#timestamp generate
from datetime import datetime, timezone, timedelta

t_delta = timedelta(hours=9)  # 9時間
JST = timezone(t_delta, 'JST') 
TIMESTAMP = datetime.now(JST).strftime("%Y%m%d%H%M%S")

#GCP
from google.cloud import storage as gcs
import google.cloud.aiplatform as aip
from typing import NamedTuple


# kubeflow SDK
import kfp
from kfp.v2 import dsl
from kfp.v2.dsl import (Artifact, Dataset, Input, InputPath, Model, Output,
                        OutputPath, component)

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--project_id")
parser.add_argument("--bucket_name")
parser.add_argument("--target_table")
parser.add_argument("--pipeline_root")
parser.add_argument("--model_url")
parser.add_argument("--code_url")
parser.add_argument("--train_url")
parser.add_argument("--test_url")
parser.add_argument("--model_name")
parser.add_argument("--valid_proportion")
parser.add_argument("--epochs")
parser.add_argument("--batch_size")

args = parser.parse_args()


### 最終的にはパイプライン形式でわたすので辞書で管理 ##############
params = {
    "project_id": args.project_id,
    "bucket_name": args.bucket_name,
    "target_table": args.target_table,

    "pipeline_root": args.pipeline_root, 
    
    "model_url": args.model_url,
    "code_url":  args.code_url,
    #"result_url":"gs://aruha-mnist/result/result.csv",
    
    "train_url":args.train_url,
    "test_url" :args.test_url,
    "model_name" : args.model_name,
    "valid_proportion": float(args.valid_proportion),
    "timestamp": TIMESTAMP,
    
    "epochs" : int(args.epochs),
    "batch_size" : int(args.batch_size)
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
from components_script import preprocess, train_model, test_model, export_to_bq
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
    model_name:str,
    
    model_url: str,
    code_url: str,
    train_url:str,
    test_url :str,
    
    valid_proportion: float,
    timestamp:str,
    
    batch_size: int,
    epochs : int,

    DISPLAY_NAME:str,
    target_table:str
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
        model_name = model_name,
        param_1    =  1.0,
    )

    bq_export_task = export_to_bq(
        project_id = project_id,
        experiment_id = DISPLAY_NAME, 
        timestamp = timestamp,  
        target_table = target_table,
        result = test_model_task.outputs["result"]
    )
    



    
if __name__ == '__main__':
    DISPLAY_NAME = "test" + TIMESTAMP
    params["DISPLAY_NAME"] = DISPLAY_NAME    
    
    compiler.Compiler().compile(
        pipeline_func=pipeline, package_path="test_pipeline.json"
    )


    job = aip.PipelineJob(
        display_name=DISPLAY_NAME,
        template_path="test_pipeline.json",
        pipeline_root=params["pipeline_root"],
        parameter_values= params,
    )

    job.run()
