# keras_training_code

## レポジトリの目的．
kerasでcifar10 or cifar100でのモデルの精度を確かめるレポジトリ．  

## 実装済み．  
wide resnet model: 精度95%程度達成．caifer100の学習方法を検討中，精度は80%超を目指す．  
```shell
実行コード．
python simple_train/main.py --config_path conf/base_config_c10.py --output_dir output_dir_name
```
Pruning code: wide resnetのモデルをDense層以外を90％圧縮しても約95%達成.  
```shell
実行コード．
python prune_train/main.py --config_path conf/prune_config.py --output_dir output_dir_name
```

## 追加予定．  
- Distillation code.  
: soft targetの学習を実装してみる．温度Tはcallbackでコントロールする．またモデルを作成する際に温度Tを計算するlayerを加える．  
- Cyclic learning rate code.  
: fast.aiのコードを参考に実装してみる．  

## コード整理TODO．
- modelが複数できたのでをstringでコントロールできるクラスを作成する．
- 各main.pyのコードが重複しているので統合する．
