# Cyberpot  

## GCEインスタンス  
- アクセススコープ
  - すべての Cloud API に完全アクセス権を許可　に設定
    - 実際にはいろいろとIAMの設定が必要そう

- e2 midium
- ストレージと同じディレクトリに

## GCS
- 以下をバケット配下に
  - code/ data/ pipeline_log/ result

## 実行
下記はインストール必要  
- sudo apt install python3-pip
- pip3 install --upgrade google-api-python-client
- pip install --upgrade google-cloud-storage
- pip install --upgrade google-cloud-aiplatform

python ファイルを実行すると、パイプラインが周り前処理→学習→テストが回る。  
- pip install kfp

