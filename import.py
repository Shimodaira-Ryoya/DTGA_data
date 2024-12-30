
#%%
import pandas as pd
from sklearn.datasets import load_digits

#%%
# 1. データセットをロード
digits = load_digits()

# 2. データセットのデータ部分（特徴量）とターゲット部分（ラベル）を取り出す
data = digits.data  # 特徴量 (64次元の数値データ)
target = digits.target  # ラベル (0~9の数字)
#%%
# 3. データを DataFrame に変換
df = pd.DataFrame(data, columns=[digits.feature_names])  # 各ピクセルの列名をつける
df['class'] = target  # ラベルを新しい列として追加

# 4. DataFrame を CSV ファイルに書き出し
df.to_csv('digits_dataset.csv')  # ファイル名を指定
print("CSVファイル 'digits_dataset.csv' にデータを書き込みました。")
# %%
from ucimlrepo import fetch_ucirepo 
dry_bean_dataset = fetch_ucirepo(id=602) 
X=dry_bean_dataset.data.features
y = dry_bean_dataset.data.targets
df = pd.concat([X, y], axis=1)
print(df)
df.to_csv('drybeans_dataset.csv')
# %%
