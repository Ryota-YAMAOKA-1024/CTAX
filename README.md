# CTAX Spin-Wave Plotter

磁気散乱の Q–E 線図をプロットするためのシンプルな Python ツールです。`code/qe_plot.py` は `code/list.txt`（または同形式の設定ファイル）を読み取り、指定した 3 つのパス [(H,H,0), (1/3+K, 1/3−K, 0), (1/3, 1/3, L)] に沿った分散と、測定可能領域の運動学的制約を重ね合わせた PNG を生成します。

## 必要環境
- Python 3.10 以上（3.13 で動作確認）
- pip でインストール可能な `numpy`, `matplotlib`

## セットアップ手順
```bash
cd /Users/yamaokaryota/Desktop/CTAX
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip numpy matplotlib
```

## 使い方
1. `code/list.txt` を編集し、交換定数や走査範囲、`outdir` などのパラメータを設定します。
2. ルートディレクトリで以下を実行します。
   ```bash
   .venv/bin/python code/qe_plot.py code/list.txt
   ```
3. 生成された図は `outdir` で指定したパス（例: `figures/`）に保存されます。

### 設定ファイルのポイント
- `Q_110`, `Q_1m10`, `Q_001` に直交基底の絶対値 (Å⁻¹) を記入します。
- `outdir` は相対パス/絶対パスいずれも指定可能です。相対パスはリポジトリ直下を基準に解決されます。
- `show = 1` にすると matplotlib ウィンドウでもグラフを表示します。

## よくある操作
- パラメータを変えて複数条件を計算する場合は、設定ファイルをコピーして別名を渡します（例: `code/list_custom.txt`）。
- コマンドを簡略化したい場合は `source .venv/bin/activate` 後に `python code/qe_plot.py code/list.txt` と実行します。

不明点や追加機能の希望があれば `code/qe_plot.py` を参照のうえ調整してください。
