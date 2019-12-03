# クマさんの道具箱

```
　 　 　┼╂┼
　 　 ∩＿┃＿∩
    |ノ      ヽ
   /   ●    ● |     機械学習業務やコンペでちょっと役に立つかもよ
  |     (_●_) ミ
 彡､     |∪|  ､｀＼
/ ＿＿   ヽノ /´>  )
(＿＿＿）    / (_／
```


## 中身
```
┣ preprocessing.py
    ┣ KS_test           - Kolmogorov-Smirnov検定
    ┣ CatEncoder        - カテゴリカル変数をいろいろな方法でエンコードする奴
    ┣ DistTransformer   - データの分布を変換する奴

┣ training.py
    ┣ Trainer           - いろいろなモデルを学習しやすくするためのラッパー
    ┣ CrossValidator    - CVをシュッと行い、結果のセーブやロードも可能
    ┣ AdvFeatureSelection   - Adversarial validationを使った特徴量選択

┣ metrics.py
    ┣ SeUnderSp         - 特異度固定時の感度を最大する目的関数

┣ nn                    - Pytorchのための道具たち
    ┣ datasets.py
        ┣ category2embedding    - カテゴリカル変数をembedding層に入れるための前処理

    ┣ logger.py
        ┣ Logger                - TensorBoard形式のログを記録する奴

    ┣ models.py
        ┣ TabularNet            - テーブルデータをスマートに学習してくれるDNN

    ┣ snapshot.py
        ┣ ...                   - スナップショットの読み書き関連
    ┣ training.py
        ┣ NeuralTrainer         - シュッとNNを訓練する奴
        ┣ EarlyStopping         - 文字通り
```