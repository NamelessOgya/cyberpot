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

### 実行
下記はインストール必要  
- sudo apt install python3-pip
- pip3 install --upgrade google-api-python-client
- pip install --upgrade google-cloud-storage
- pip install --upgrade google-cloud-aiplatform
- pip install kfp
python ファイルを実行すると、パイプラインが周り前処理→学習→テストが回る。  
- 
  
## 今後のtodo
- 学習時のインスタンスタイプ、指定できるように
- 学習モードてとテストモードの分割
- パラメータをコマンドラインから指定できるように
- BQとの連携
