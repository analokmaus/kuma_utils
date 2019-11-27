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
    ┣ CatEncoder        - カテゴリカルデータをいろいろな方法でエンコードする奴
    ┣ DistTransformer   - データの分布を変換する奴

┣ training.py
    ┣ Trainer           - いろいろなモデルを学習しやすくするためのラッパー
    ┣ CrossValidator    - CVをシュッと描けるようにし、結果のセーブやロードも可能

```