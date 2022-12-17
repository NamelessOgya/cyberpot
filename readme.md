# Cyberpot  
## テスト実行  
### GCEインスタンス  
- アクセススコープ
  - すべての Cloud API に完全アクセス権を許可　に設定
    - 実際にはいろいろとIAMの設定が必要そう

- e2 midium
- ストレージと同じディレクトリに

### GCS
- 以下をバケット配下に
  - code/ data/ pipeline_log/ result
    - 学習データはzip圧縮してるので解凍お願いします。

### 権限  
  - BigQuery ジョブユーザー
  - BigQuery データオーナー
  - BigQuery データ編集者
  - BigQuery 管理者
  - Vertex AI Custom Code サービス エージェント
  - Vertex AI ユーザー
  - サービス アカウント ユーザー

### 実行
下記はインストール必要  
- sudo apt install python3-pip
- pip3 install --upgrade google-api-python-client
- pip install --upgrade google-cloud-storage
- pip install --upgrade google-cloud-aiplatform
- pip install --upgrade google-cloud-bigquery
- pip install kfp  
  
pythonファイルをGCEインスタンス内にアップロード  
python ファイルを実行すると、パイプラインが周り前処理→学習→テストが回る。  

```
python3 cyberpot.py \
--project_id [project_id] \
--bucket_name [bucket_name] \
--target_table [project_id].mnist.result \
--pipeline_root gs://[bucket_name]/pipeline_log \
--model_url gs://[bucket_name]/model \
--code_url gs://[bucket_name]/code/components_script.py \
--train_url gs://[bucket_name]/data/mnist_train.csv \
--test_url gs://[bucket_name]/data/mnist_test.csv \
--model_name  mnist_cnn \
--valid_proportion 0.2 \
--epochs  5 \
--batch_size  32
--gpu_limit 1 \
--gpu_type NVIDIA_TESLA_T4
```
  
## 今後のtodo
- 学習時のインスタンスタイプ、指定できるように(GPU指定対応完了)
- 学習モードてとテストモードの分割
- パラメータをコマンドラインから指定できるように(対応完了！)
- BQとの連携(対応完了!)

