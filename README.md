# SSAP再現実装公開用リポジトリ
[SSAP [Proposal-freeなInstance Segmentation手法] の紹介と実験](https://blog.albert2005.co.jp/2020/08/18/ssap/)で使用したプログラムです.

## ファイル構成について
```
├─ README.md                         # プロジェクトの説明
├─ requirements.txt                  # すべてのPythonプログラムのベースとなるPythonパッケージ
├─ data/                             # データフォルダ
│   ├─ train2014/                    # COCO2014の学習データ
│   ├─ val2014/                      # COCO2014の評価データ
│   ├─ annotations/                  # COCO2014のannotationデータ
│   │   ├─ instances_train2014.json  # 学習データのannotationデータ
│   │   └─ instances_val2014.json    # 評価データのannotationデータ
│   ├─ t_class_name.txt              # 各クラスの名前
│   ├─ t_color.txt                   # 各クラスのSemantic Segmentationの色
│   └─ resize/                       # resizeしたデータ
│       ├─ train2014/                # resizeした学習データ
│       ├─ val2014/                  # resizeした評価データ
│       ├─ semantic_train/           # Semantic Segmentationの正解データ（学習データ）        
│       ├─ semantic_val/             # Semantic Segmentationの正解データ（評価データ）
│       ├─ instance_train/           # Instance Segmentationの正解データ（学習データ） 
│       └─ instance_val/             # Instance Segmentationの正解データ（評価データ）
├─ notebooks/                        # Jupyter Notebook
│   ├─ SSAP.ipynb                    # SSAPの実行
│   └─ make_coco_dataset.ipynb       # 実験に使用するデータの作成
├─ src/                              # pythonモジュール
└─ exp/                              # 学習・推論の実行情報と実行結果の保存先
    ├─ exp_1/                        # 実験ごとにフォルダを分ける
    │   ├─ graph/                    # lossの推移を示すグラフ
    │   ├─ metrics/                  # 評価結果
    │   ├─ trained_model/            # 学習モデル
    │   └─ log                       # log
    ├─ exp_2/
```


## 実行方法
- `requirements.txt` に書かれているライブラリをインストールしてください．
- [MSCOCOのホームページ](https://cocodataset.org/#download)から学習/評価画像とそのアノテーションデータをダウンロードし，`data/`以下に格納してください(ブログでは2014年のデータを使用しました)．
- `make_coco_dataset.ipynb`を上から実行することで，アノテーションデータからSegmentation maskを作成し，データのresizeとcropを行います．3つ目のセルでtrain/valが変更できるので，これを変更し学習データと評価データを作成します．
- `SSAP.ipynb`を上から実行していくことで、学習/評価が行えます.

## 元論文
SSAP: Single-Shot Instance Segmentation With Affinity Pyramid
- arXiv: https://arxiv.org/abs/1909.01616
