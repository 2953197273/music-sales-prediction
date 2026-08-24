import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import warnings
import os

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100
plt.rcParams['figure.facecolor'] = 'white'
save_path = r"D:\数据挖掘\大作业2\plots"
if not os.path.exists(save_path):
    os.makedirs(save_path)


def load_and_rename_datasets():
    songs_path = r"D:\数据挖掘\大作业2\data\mars_tianchi_songs.csv"
    actions_path = r"D:\数据挖掘\大作业2\data\mars_tianchi_user_actions.csv"

    try:
        df_songs = pd.read_csv(songs_path, encoding='utf-8')
        df_songs.columns = ['song_id', 'artist_id', 'release_date', 'language', 'genre', 'song_length']
        print("歌曲表读取+重命名成功！")
        print(f"歌曲表维度：{df_songs.shape}")
    except Exception as e:
        print(f"歌曲表处理失败：{e}")
        return None, None

    try:
        df_actions = pd.read_csv(actions_path, encoding='utf-8').sample(frac=0.1, random_state=42)
        df_actions.columns = ['user_id', 'song_id', 'gmt_create', 'action_type', 'dt']
        print("\n用户行为表读取+重命名成功！")
        print(f"用户行为表维度（采样后）：{df_actions.shape}")
    except Exception as e:
        print(f"用户行为表处理失败：{e}")
        return None, None

    return df_songs, df_actions


df_songs, df_actions = load_and_rename_datasets()
if df_songs is None or df_actions is None:
    exit()

action_features = df_actions.groupby('song_id').agg({
    'user_id': 'count',
    'action_type': lambda x: (x == 1).sum()
}).reset_index()
action_features.columns = ['song_id', '总行为次数', '播放次数']

df_merged = pd.merge(df_songs, action_features, on='song_id', how='left')

# ===================== 数据质量检查模块 =====================
print("\n" + "=" * 60 + " 数据质量检查 " + "=" * 60)

# 1. 缺失值检查
print("\n1. 缺失值检查")
missing_songs = pd.DataFrame({
    '缺失数量': df_songs.isnull().sum(),
    '缺失率(%)': round((df_songs.isnull().sum() / len(df_songs) * 100), 2)
})
missing_actions = pd.DataFrame({
    '缺失数量': df_actions.isnull().sum(),
    '缺失率(%)': round((df_actions.isnull().sum() / len(df_actions) * 100), 2)
})
missing_merged = pd.DataFrame({
    '缺失数量': df_merged.isnull().sum(),
    '缺失率(%)': round((df_merged.isnull().sum() / len(df_merged) * 100), 2)
})
print("歌曲表缺失值：")
print(missing_songs)
print("\n用户行为表缺失值：")
print(missing_actions)
print("\n合并数据集缺失值：")
print(missing_merged)

plt.figure(figsize=(10, 4))
sns.heatmap(df_merged.isnull(), cbar=False, cmap='viridis', yticklabels=False)
plt.title("合并数据集缺失值分布")
plt.savefig(f"{save_path}/缺失值分布.png", bbox_inches='tight')
plt.show()

# 2. 重复值检查
print("\n2. 重复值检查")
dup_songs = df_songs.duplicated().sum()
dup_actions = df_actions.duplicated().sum()
dup_merged = df_merged.duplicated().sum()
print(f"歌曲表重复记录数：{dup_songs}")
print(f"用户行为表重复记录数：{dup_actions}")
print(f"合并数据集重复记录数：{dup_merged}")

# 3. 异常值检查（IQR法）
print("\n3. 异常值检查（数值型字段）")
numeric_cols = ['song_length', '总行为次数', '播放次数']
df_numeric = df_merged[numeric_cols].dropna()
for col in numeric_cols:
    Q1 = df_numeric[col].quantile(0.25)
    Q3 = df_numeric[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outlier_count = len(df_numeric[(df_numeric[col] < lower) | (df_numeric[col] > upper)])
    outlier_rate = round((outlier_count / len(df_numeric) * 100), 2)
    print(f"{col}：")
    print(f"  四分位数：Q1={Q1:.2f}, Q3={Q3:.2f}, IQR={IQR:.2f}")
    print(f"  异常值阈值：[{lower:.2f}, {upper:.2f}]")
    print(f"  异常值数量：{outlier_count} | 占比：{outlier_rate}%")

plt.figure(figsize=(8, 4))
sns.boxplot(x=df_merged['播放次数'].dropna(), color='lightblue')
plt.title("播放次数异常值分布")
plt.savefig(f"{save_path}/播放次数异常值.png", bbox_inches='tight')
plt.show()

# 4. 分类字段唯一性检查
print("\n4. 分类字段唯一性检查")
cat_cols = ['language', 'genre', 'action_type']
for col in cat_cols:
    if col in df_merged.columns:
        unique_count = df_merged[col].nunique()
        unique_sample = df_merged[col].unique()[:5]
        print(f"{col}：唯一值数量={unique_count} | 前5个值={unique_sample}")

plt.figure(figsize=(10, 4))
df_merged['language'].value_counts().head(10).plot(kind='bar', color='lightgreen')
plt.title("语言类型分布（TOP10）")
plt.xlabel("语言编码")
plt.ylabel("歌曲数量")
plt.savefig(f"{save_path}/语言类型分布.png", bbox_inches='tight')
plt.show()

# ===================== 数据预处理 =====================
df_merged['播放次数'].fillna(0, inplace=True)
df_merged['总行为次数'].fillna(0, inplace=True)

df_merged['language'] = df_merged['language'].astype(str).str.strip()
df_merged = df_merged[df_merged['language'] != '0']
df_merged = df_merged[df_merged['language'].str.isdigit()]
df_merged['release_year'] = df_merged['release_date'].astype(str).str[:4]
df_merged = df_merged[df_merged['release_year'].str.isdigit() & (df_merged['release_year'] != '0000')]
df_merged = df_merged[df_merged['播放次数'] <= df_merged['播放次数'].quantile(0.95)]

# ===================== 可视化 =====================
print("\n" + "=" * 60 + " 优化版业务可视化 " + "=" * 60)

plt.figure(figsize=(10, 5))
sns.histplot(df_merged['播放次数'], bins=50, kde=True, color='#1f77b4', alpha=0.7)
plt.title("歌曲播放次数分布")
plt.xlabel("播放次数")
plt.ylabel("歌曲数量")
plt.grid(alpha=0.3)
plt.savefig(f"{save_path}/播放次数分布.png", bbox_inches='tight')
plt.show()

top10_genre = df_merged.groupby('genre')['播放次数'].mean().sort_values(ascending=False).head(10).index
df_genre = df_merged[df_merged['genre'].isin(top10_genre)]

plt.figure(figsize=(12, 6))
sns.boxplot(x='genre', y='播放次数', data=df_genre, palette='Set2')
plt.title("TOP10流派播放次数对比")
plt.xlabel("流派ID")
plt.ylabel("播放次数")
plt.xticks(rotation=45)
plt.grid(alpha=0.3)
plt.savefig(f"{save_path}/流派播放次数对比.png", bbox_inches='tight')
plt.show()

lang_play = df_merged.groupby('language')['播放次数'].sum().sort_values(ascending=False).head(8)

plt.figure(figsize=(10, 6))
bars = sns.barplot(x=lang_play.index, y=lang_play.values, palette='Set3')
plt.title("TOP8语言歌曲总播放次数（优化版）")
plt.xlabel("语言编码（0=无效已过滤）")
plt.ylabel("总播放次数")
plt.grid(alpha=0.3, axis='y')

for bar in bars.patches:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2., height + 50,
             f'{int(height)}', ha='center', va='bottom', fontsize=9)

plt.savefig(f"{save_path}/语言播放次数对比_优化版.png", bbox_inches='tight')
plt.show()

year_play = df_merged.groupby('release_year')['播放次数'].mean().sort_index()
plt.figure(figsize=(12, 5))
sns.lineplot(x=year_play.index, y=year_play.values, marker='o', color='#ff7f0e')
plt.title("歌曲发布年份 vs 平均播放次数")
plt.xlabel("发布年份")
plt.ylabel("平均播放次数")
plt.xticks(rotation=45)
plt.grid(alpha=0.3)
plt.savefig(f"{save_path}/年份-播放次数趋势.png", bbox_inches='tight')
plt.show()

plt.figure(figsize=(10, 6))
sns.regplot(x='总行为次数', y='播放次数', data=df_merged, scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'})
plt.title("总行为次数 vs 播放次数（带拟合线）")
plt.xlabel("总行为次数（曝光度）")
plt.ylabel("播放次数（流行度）")
plt.grid(alpha=0.3)
plt.savefig(f"{save_path}/行为次数-播放次数相关性.png", bbox_inches='tight')
plt.show()

# ===================== 建模与评估 =====================
target = '播放次数'
encode_cols = ['artist_id', 'language', 'genre', 'release_year']
for col in encode_cols:
    le = LabelEncoder()
    df_merged[col] = le.fit_transform(df_merged[col].astype(str))

features = [col for col in df_merged.columns if col not in ['song_id', target, 'release_date', 'dt', 'gmt_create']]
features = [col for col in features if df_merged[col].dtype in ['int64', 'float64']]
X = df_merged[features]
y = df_merged[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

plt.figure(figsize=(10, 6))
sorted_idx = np.argsort(y_test.values)
y_test_sorted = y_test.values[sorted_idx]
y_pred_sorted = y_pred[sorted_idx]

plt.plot(y_test_sorted, label='真实值', color='blue', linewidth=2)
plt.plot(y_pred_sorted, label='预测值', color='red', linewidth=2, alpha=0.8)
plt.fill_between(range(len(y_test_sorted)), y_test_sorted, y_pred_sorted, color='gray', alpha=0.2, label='误差带')
plt.title("播放次数预测：真实值 vs 预测值（排序后）")
plt.xlabel("样本序号（按真实值排序）")
plt.ylabel("播放次数")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(f"{save_path}/预测值-真实值拟合线.png", bbox_inches='tight')
plt.show()

residuals = y_test - y_pred
plt.figure(figsize=(10, 5))
sns.histplot(residuals, bins=50, kde=True, color='#2ca02c')
plt.axvline(x=0, color='red', linestyle='--', label='残差=0')
plt.title("模型残差分布（预测误差）")
plt.xlabel("残差（真实值-预测值）")
plt.ylabel("频次")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(f"{save_path}/模型残差分布.png", bbox_inches='tight')
plt.show()

feature_importance = pd.DataFrame({
    "特征": features,
    "重要性": model.feature_importances_
}).sort_values(by="重要性", ascending=False).head(8)
feature_importance['重要性(%)'] = (feature_importance['重要性'] * 100).round(2)

plt.figure(figsize=(10, 6))
bars = sns.barplot(x='重要性(%)', y='特征', data=feature_importance, palette='viridis')
plt.title("影响播放次数的核心特征（前8）")
plt.xlabel("重要性（%）")
plt.ylabel("特征")
for i, v in enumerate(feature_importance['重要性(%)']):
    plt.text(v + 0.5, i, f"{v}%", va='center')
plt.grid(alpha=0.3)
plt.savefig(f"{save_path}/核心特征重要性.png", bbox_inches='tight')
plt.show()

print("\n" + "=" * 60 + " 模型评估 " + "=" * 60)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"平均绝对误差（MAE）：{mae:.2f}")
print(f"均方根误差（RMSE）：{rmse:.2f}")
print(f"决定系数（R²）：{r2:.2f}")

param_grid = {'n_estimators': [80, 100, 120], 'max_depth': [8, 10, 12]}
grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5, scoring='r2')
grid_search.fit(X_train, y_train)
print(f"\n最优参数：{grid_search.best_params_}")
print(f"调优后R²：{r2_score(y_test, grid_search.best_estimator_.predict(X_test)):.2f}")