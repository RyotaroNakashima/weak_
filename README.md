- **（余談）今回は無しだが、LoRAも結構理にかなっている**
  - 学習時間は短くなる  
  - 元のパラメータを保持してパラメータ追加するので、破滅的忘却は比較的起きづらいらしい  
  - 最新の研究では「InfLoRA」のように、新タスクの学習による既存タスクへの**干渉 (interference)** を理論的に最小化する枠組みも登場している  
    - [InfLoRA (arXiv:2404.00228)](https://arxiv.org/abs/2404.00228#:~:text=on%20PEFT%2C%20most%20of%20them,Experimental)

- **Replay（リプレイ、リハーサル (Rehearsal) 手法とも呼ぶ。一番有力）**
  - 過去タスクの一部データ（エクゼンプロ）を保存しておき、新データと混ぜて再学習する方法  
  - 少し工夫して、ベースデータは17万枚の中から毎 epoch で10万枚分選ばれるようにして、弱点データは10万枚そのままぶちこんでしまえば良いのでは？  
  - ベースデータは既にベースモデルで30 epoch ほど学習されているわけで、弱点データと同様の epoch を回す必要はないのでは？  
  - こうすることでベースデータの忘却を抑えつつ、弱点データの補強のバランスが取れるかも。類似研究を見つけたい  
  - 1 epoch 毎に出てきたモデルをベースモデルに設定し、データを 17 万枚からランダム抽出し、MLFlow の Run ID は同一にして、毎回データをロードし直せば実現可能。ただし、データロードの時間が毎回 5 時間位かかる点で非現実的。そういうことができるライブラリを探すか、工夫して実装するしかない  
  - 元データの分布を保持しながら学習を進められる  
  - 参考になりそうな論文  
    - **Learn the Time to Learn: Replay Scheduling (TMLR 2023)**  
      - [https://arxiv.org/abs/2209.08660?utm_source=chatgpt.com](https://arxiv.org/abs/2209.08660?utm_source=chatgpt.com)  
      - どのタスク（≒データ分布）を「いつ」「何割」リプレイするかをスケジューラが学習して自動決定。リソースと性能のトレードオフを最適化  
      - 旧データ全部を毎 epoch 混ぜない設計。「旧＝100k 抽出，新＝100k 固定」のような比率をRLで最適化している点がそっくり。  
    - **Adaptive Memory Replay (CVPR W 2024)**  
      - リプレイ・サンプリングを多腕バンディットで動的に制御。計算バジェット内で最も有用な旧サンプルを都度選ぶ  
      - 「旧データはメモリに全部保持、でも GPU に流すのは選抜」という発想。データロードの無駄撃ちを抑えたい時に有効。  
      - [https://arxiv.org/html/2404.12526v1?utm_source=chatgpt.com](https://arxiv.org/html/2404.12526v1?utm_source=chatgpt.com)  
    - **そもそも旧データは忘却対策とはいえ何 epoch も再学習が必要なのか？**  
      - ICCV 2021 の解析では「旧サンプルの再訪回数は 1～2 回で十分」と結論付けられている。Donut との学習条件の乖離は要確認。

- **EWC（Elastic Weight Consolidation）**  
  - 過去のタスクで「重要だった重み」は、新しいタスクでできるだけ変えないようにする。  
  - 正則項を追加して実現  
  - 新タスク学習中も旧タスクの知識を保持しやすくする  

- **クラスタリング系**
  - **3A フレームワーク**  
    1. 分布近似 (Approximate)  
    2. 性能保持型蒸留 (Adapt)  
    3. 微分プライバシ (Anonymize)  
  - 上記の3つで構成されているが、1＋2 のみ関心がある（3はプライバシー保護の話）  
    - 1 で分布近似（クラスタリング）し、2 で元データの勾配情報を保つように「合成データ点（inducing points）」を最適化する  
    - [3A: A Framework for Privacy-Preserving Training Data Release for Machine Learning](https://assets.amazon.science/15/94/4069bb034d86b4f436520498d7c1/approximate-adapt-anonymize-3a-a-framework-for-privacy-preserving-training-data-release-for-machine-learning.pdf#:~:text=,order%20to%20preserve%20data%20distribution)

  - **特徴空間での Herding アルゴリズム**  
    - iCaRL (Incremental Classifier and Representation Learning) という論文内で言及されている  
      - 「ストレージ制限付きでクラスを順番に学んでも破滅的忘却を起こさない」を目標にした、2017年の代表的クラス増分学習フレームワーク。  
      - 以下の3つで構成（多分 1,2 までで弱点取り込みには十分なはず）  
        1. エグゼンプラー（少数サンプル）だけを保存し、  
        2. その平均ベクトルだけで分類し、  
        3. 知識蒸留を加えながら表現も同時に更新する  
      - 各クラス（タスク）の平均特徴ベクトルに近い順にサンプルを反復的に選んでいく Herding 法を用いる  
      - 少数サンプルで元分布を近似する戦略  
      - 各クラスの「代表ベクトル」に対し最も代表的な実例を貪欲に追加していくことで、少ないデータでも元のデータ分布に近い性能を発揮できることが示されている  
      - [iCaRL: Incremental Classifier and Representation Learning (CVPR 2017)](https://openaccess.thecvf.com/content_cvpr_2017/papers/Rebuffi_iCaRL_Incremental_Classifier_CVPR_2017_paper.pdf#:~:text=mean%20vector,fewer%20samples%20to%20achieve%20a)

  - **PCA や t-SNE を使う場合**  
    - データ分布を可視化し、外れ値を除去したり均等にサンプルを選んだりする方法  
    - t-SNE マップ上でデータが密集している領域・希薄な領域を把握し、密集領域では代表点のみ選択、希薄領域のサンプルはできるだけ残す、といったヒューリスティクスも考えられる  
    - 重要なのは、データ縮小後の分布が元の分布と可能な限り近いことを定量的に確認すること  
    - 評価指標としては、元データとサブセットデータ間の分布類似度（例えば KL ダイバージェンスやフレシェ距離など）を測定し、一定以下の差に収まるようサンプル数を調整する、といった手法が考えられる

- まとめ
 - 「旧データを毎エポックでサブサンプリ，新データはフル学習」 という設計は，Replay Scheduling 系論文群で既に効果が実証されており，十分研究的裏付けがあります。
 - ロード時間問題 は ストリーミング Dataset（WebDataset, Reverb）＋オンザフライシャッフルでほぼ解決可能。
 - 旧データ縮小には Herding / Multi-criteria 抽出 が定番。
