
import kfp
from kfp.v2 import dsl
from kfp.v2.dsl import (Artifact, Dataset, Input, InputPath, Model, Output,
                        OutputPath, component)



@component(base_image='tensorflow/tensorflow:latest')
def preprocess(
    project_id: str,
    bucket_name: str,
    train_url:str,
    test_url :str,
    valid_proportion: float,
    
    train_dataset: Output[Dataset],
    test_dataset: Output[Dataset],
    valid_dataset: Output[Dataset]
):
    import numpy as np
    import tensorflow as tf
    from google.cloud import storage as gcs
    from io import BytesIO
    
    
    project_id = project_id
    client = gcs.Client(project_id)
    bucket = client.get_bucket(bucket_name)
    
    def data_load_from_url(bucket, bucket_name, url):
        path_ = url.split("gs://" + bucket_name + "/")[1]
        blob = bucket.blob(path_)
        content = blob.download_as_bytes()
        output = np.loadtxt(BytesIO(content), delimiter=',')
        
        return output
        
    train = data_load_from_url(bucket, bucket_name, train_url)
    test  = data_load_from_url(bucket, bucket_name, test_url)
    
    np.random.shuffle(train)
    np.random.shuffle(test)
    
    #時間短縮のため参照するデータを少なく
    train = train[:10000]
    test = test[:1000]

    valid_data  = train[:int(len(train) * (valid_proportion))]
    train_data  = train[int(len(train) * (valid_proportion)) + 1:]
    test_data   = test
    
    #ファイルの受け渡しはsave / loadによってなされる。kubeflowは保存場所を提供しているにすぎない。
    #kubeflowの提供した場所にmodelを保存
    _train_dataset = tf.data.Dataset.from_tensor_slices(( train_data[:, 1:]/255.0, train_data[:, 0].reshape(-1, 1)))
    _test_dataset = tf.data.Dataset.from_tensor_slices(( test_data[:, 1:]/255.0, test_data[:, 0].reshape(-1, 1)))
    _valid_dataset = tf.data.Dataset.from_tensor_slices(( valid_data[:, 1:]/255.0, valid_data[:, 0].reshape(-1, 1)))
    
    
    _train_dataset.save(train_dataset.path, compression=None, shard_func=None, checkpoint_args=None)
    _test_dataset.save(test_dataset.path, compression=None, shard_func=None, checkpoint_args=None)
    _valid_dataset.save(valid_dataset.path, compression=None, shard_func=None, checkpoint_args=None)
    

@component(base_image='tensorflow/tensorflow:latest')
def train_model(
    train_dataset :Input[Dataset],
    valid_dataset :Input[Dataset],
    
    batch_size    :int,
    epochs        :int,
    
    model         :Output[Artifact]

):
    import tensorflow as tf
    import os
    import shutil
    import csv
    
    class Simple_CNN(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.conv_2d_0 = tf.keras.layers.Conv2D(32, (4,4), activation = "relu", input_shape = (28,28,1))
            self.pooling_0 = tf.keras.layers.MaxPooling2D((2, 2))

            self.conv_2d_1 = tf.keras.layers.Conv2D(32, (4,4), activation = "relu")
            self.pooling_1 = tf.keras.layers.MaxPooling2D((2, 2))

            self.conv_2d_2 = tf.keras.layers.Conv2D(32, (4,4), activation = "relu")

            self.flatten = tf.keras.layers.Flatten()

            self.dense_0 = tf.keras.layers.Dense(64, activation = "relu")
            self.dense_1 = tf.keras.layers.Dense(10, activation = "softmax")


        def call(self, inputs):
            x = tf.reshape(inputs, [-1,28,28,1])
            x = self.pooling_0(self.conv_2d_0(x))
            x = self.pooling_1(self.conv_2d_1(x))
            x = self.conv_2d_2(x)

            x = self.flatten(x)
            x = self.dense_0(x)
            x = self.dense_1(x)

            return x
        
    def train_model(batch_size,epochs, train_dataset, valid_dataset):
        model = Simple_CNN()
        model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy']
        )
        
        history = model.fit(
            train_dataset,
            epochs = epochs, 
            validation_data = valid_dataset
        )

        return model, history
    
    _train_dataset = tf.data.Dataset.load(train_dataset.path)
    _valid_dataset = tf.data.Dataset.load(valid_dataset.path)
    
    _model, _history = train_model(batch_size, epochs,_train_dataset, _valid_dataset)
    
    _model.save(model.path)
        
        

@component(
    base_image='tensorflow/tensorflow:latest',
    packages_to_install=['pandas']
    )
def test_model(
    test_dataset: Input[Dataset], 
    model       : Input[Artifact],
    #TIMESTAMP   : str,
    model_name  : str,
    param_1     : float,
    
    result     : Output[Dataset]
):
    import tensorflow as tf
    import numpy as np
    import pandas as pd
    
    _test_dataset = tf.data.Dataset.load(test_dataset.path)
    
    _model = tf.keras.models.load_model(model.path)
        
    _loss, _accuracy = _model.evaluate(_test_dataset)
    
    
    #base_ = np.array([["param_1", "accuracy", "loss"]])
    result_df = np.array([[param_1,  _accuracy, _loss]])
    result_df = pd.DataFrame(result_df, columns = ["param_1", "accuracy", "loss"])
    
    result_df.to_csv(result.path)
#    np.savetxt(summary.path + ".csv", result, fmt="%s", delimiter=',')
#     with open(loss.path, "w") as f:
#         f.write(str(_loss))
        
#     with open(accuracy.path, "w") as f:
#         f.write(str(_accuracy))


#### 以下共通部分
@component(
    base_image='tensorflow/tensorflow:latest',
    packages_to_install=['pandas', 'google-cloud-bigquery']
    )
def export_to_bq(
    project_id: str,
    experiment_id: str, 
    timestamp   : str, 
    target_table: str,

    result :Input[Dataset]
):
    from google.cloud import bigquery
    import numpy as np
    import pandas as pd

    client = bigquery.Client(project = project_id)
    
    # experiment_idとtsを付与
    dataframe = pd.read_csv(result.path, index_col = 0)

    dataframe.insert(0, "experiment_id", experiment_id)
    dataframe.insert(0, "timestamp", timestamp)
    
    
    bq_coltypes = []
    
    # カラム情報をまとめる。
    for col_name, dtype in zip(dataframe.columns, dataframe.dtypes):

        if dtype == np.object_:
            bq_coltypes.append(bigquery.SchemaField(col_name, bigquery.enums.SqlTypeNames.STRING))
            #sample_result[col_name] = sample_result[col_name].astype(str)

        elif dtype == np.int64:
            bq_coltypes.append(bigquery.SchemaField(col_name, bigquery.enums.SqlTypeNames.INTEGER))

        elif dtype == np.float64:
            bq_coltypes.append(bigquery.SchemaField(col_name, bigquery.enums.SqlTypeNames.FLOAT))

        else:
            #日付系はいっぱいあってしんどいので例外処理
            bq_coltypes.append(bigquery.SchemaField(col_name, bigquery.enums.SqlTypeNames.TIMESTAMP))
    
    
    job_config = bigquery.LoadJobConfig(
        schema = bq_coltypes
    )
    
    job = client.load_table_from_dataframe(
        dataframe, target_table, job_config = job_config
    )    