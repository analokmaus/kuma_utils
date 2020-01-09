# クマさんの道具箱

```
　 　 　┼╂┼
　 　 ∩＿┃＿∩
    |ノ      ヽ
   /   ●    ● |         機械学習業務やコンペでちょっと役に立つかもよ
  |     (_●_) ミ        基本的に自分用なので説明は少なめです。。 
 彡､     |∪|  ､｀＼
/ ＿＿   ヽノ /´>  )
(＿＿＿）    / (_／
```


## 中身
```
┣ common.py             - 様々な細かい便利ツール

┣ visualization.py
    ┣ KS_test           - Kolmogorov-Smirnov検定
    ┣ explore_dataframe - 自動でいい感じにEDAする奴

┣ preprocessing.py
    ┣ CatEncoder        - カテゴリカル変数をいろいろな方法でエンコードする奴
    ┣ DistTransformer   - データの分布を変換する奴
    ┣ MICE              - NAフラグ入りのMICEを行う奴

┣ training.py
    ┣ Trainer           - いろいろなモデルを学習しやすくするためのラッパー
    ┣ CrossValidator    - CVをシュッと行い、結果のセーブやロードも可能
    ┣ AdvFeatureSelection   - Adversarial validationを使った特徴量選択
    ┣ StratifiedGroupKFold  - 層別化したGroupKFold

┣ metrics.py
    ┣ SeUnderSp         - 特異度固定時の感度を最大する目的関数

┣ nn                    - PyTorchのための道具たち
    ┣ datasets.py
        ┣ category2embedding    - カテゴリカル変数をembedding層に入れるための前処理

    ┣ logger.py
        ┣ Logger                - TensorBoard形式のログを記録する奴

    ┣ metrics.py                - PyTorch用のmetrics
        ┣ auc
        ┣ accuracy

    ┣ models.py
        ┣ TabularNet            - テーブルデータをスマートに学習してくれるDNN

    ┣ snapshot.py
        ┣ ...                   - スナップショットの読み書き関連
        
    ┣ training.py
        ┣ NeuralTrainer         - シュッとNNを訓練する奴
        ┣ EarlyStopping         - 文字通り
```