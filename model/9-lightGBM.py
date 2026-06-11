import math
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt

# ===================== 全局字体设置 =====================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 28

# ===================== 加载数据 =====================
print("加载数据中...")
train_df = pd.read_csv(
    '../model/model_based_on_iACP/new_feature_select/feature/'
    'fused_bert_train_features.csv'
)

print(f"原始数据形状: {train_df.shape}")
print(f"列名: {train_df.columns.tolist()[:10]}... (显示前10列)")

# ===================== 提取特征和标签 =====================
sequence_id_col = train_df.columns[0]   # 第一列是 sequence_id
label_col = train_df.columns[-1]        # 最后一列是 LABEL

print(f"\nSequence ID列: {sequence_id_col}")
print(f"标签列: {label_col}")

feature_cols = [
    col for col in train_df.columns
    if col not in [sequence_id_col, label_col]
]

print(f"特征列数: {len(feature_cols)}")
print(f"特征列示例: {feature_cols[:5]}")

X_train = train_df[feature_cols].values
y_train = train_df[label_col].values

print(f"\n特征矩阵形状: {X_train.shape}")
print(f"标签形状: {y_train.shape}")
print(f"标签分布: 类别1={np.sum(y_train)}, 类别0={np.sum(y_train == 0)}")

# ===================== 使用 LightGBM 进行特征选择 =====================
print("\n" + "=" * 70)
print("使用LightGBM进行特征选择...")
print("=" * 70)

model = lgb.LGBMClassifier(
    boosting_type='gbdt',
    objective='binary',
    num_leaves=31,
    learning_rate=0.05,
    n_estimators=100,
    verbose=-1,
    random_state=42
)

model.fit(X_train, y_train)

feature_importances = model.feature_importances_
feature_importance_dict = dict(zip(feature_cols, feature_importances))
sorted_features = sorted(
    feature_importance_dict.items(),
    key=lambda x: x[1],
    reverse=True
)

# ===================== 筛选重要性大于 2 的特征 =====================
print(f"\n所有特征数: {len(feature_cols)}")
print(f"重要性大于2的特征数: {len([f for f in sorted_features if f[1] > 2])}")

selected_features = [f[0] for f in sorted_features if f[1] > 2]
selected_importances = [f[1] for f in sorted_features if f[1] > 2]

print(f"\n筛选后的特征数: {len(selected_features)}")
print(f"\n重要性大于2的特征列表:")
print("-" * 70)

for i, (feature, importance) in enumerate(
    sorted_features[:20] if len(sorted_features) > 20 else sorted_features
):
    status = "✓ 保留" if importance > 2 else "✗ 删除"
    print(f"{i + 1:3d}. {feature:30s} | 重要性: {importance:8.3f} | {status}")

if len(sorted_features) > 20:
    print(f"... (还有 {len(sorted_features) - 20} 个特征,省略显示)")

# ===================== 创建新的 CSV 文件 =====================
print("\n" + "=" * 70)
print("保存筛选后的数据...")
print("=" * 70)

selected_cols = [sequence_id_col] + selected_features + [label_col]
train_selected = train_df[selected_cols].copy()

output_file = '../data/csv/case_analysis/case_analysis_bert_lightGBM.csv'
train_selected.to_csv(output_file, index=False)

print(f"新文件形状: {train_selected.shape}")
print(f"新文件列数: {train_selected.shape[1]}")
print(f"特征列数: {len(selected_features)}")

# ===================== 保存特征选择信息 =====================
feature_info_df = pd.DataFrame({
    '排序': range(1, len(sorted_features) + 1),
    '特征名': [f[0] for f in sorted_features],
    '重要性': [f[1] for f in sorted_features],
    '是否保留': ['是' if f[1] > 2 else '否' for f in sorted_features]
})

info_file = 'feature_importance_info_lm.csv'
# feature_info_df.to_csv(info_file, index=False)
print(f"✓ 特征信息已保存: {info_file}")

# ===================== 绘制特征重要性图（纵坐标刻度间距固定为15） =====================
print("\n绘制特征重要性分布图...")

fig, ax = plt.subplots(figsize=(12, 6))

top_n = 20
top_features = sorted_features[:top_n]
feature_names = [f[0] for f in top_features]
feature_vals = [f[1] for f in top_features]

x_pos = np.arange(len(feature_names))

ax.bar(
    x_pos,
    feature_vals,
    color='#A1BC98',
    edgecolor='white',
    linewidth=0.5
)

ax.set_xticks(x_pos)
ax.set_xticklabels(
    feature_names,
    rotation=45,
    ha='right',
    fontsize=28,
    fontfamily='Times New Roman'
)

ax.set_ylabel(
    'Feature importance',
    fontsize=28,
    fontfamily='Times New Roman',
    fontweight='normal'
)

# ---------- 设置 y 轴刻度间距为 15 ----------
tick_step = 15
max_val = max(feature_vals) if len(feature_vals) > 0 else 0
# 向上取整到 tick_step 的整数倍，确保顶部留有一点空白
y_max_ceiled = int(math.ceil((max_val + 1e-8) / tick_step) * tick_step)
if y_max_ceiled == 0:
    y_max_ceiled = tick_step
ax.set_ylim(0, y_max_ceiled)
yticks = np.arange(0, y_max_ceiled + tick_step, tick_step)
ax.set_yticks(yticks)
# ---------------------------------------------

ax.tick_params(axis='y', labelsize=28)

for label in ax.get_yticklabels():
    label.set_fontfamily('Times New Roman')
    label.set_fontsize(28)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)

ax.grid(False)

plt.tight_layout()
save_name = 'feature_importance_analysis_2.png'
plt.savefig(save_name, dpi=300, bbox_inches='tight')
plt.show()

print(f"✓ 特征重要性图已保存: {save_name}")

# ===================== 打印总结 =====================
print("\n" + "=" * 70)
print("特征选择总结")
print("=" * 70)
print(f"原始特征数: {len(feature_cols)}")
print(f"保留特征数: {len(selected_features)}")
print(f"删除特征数: {len(feature_cols) - len(selected_features)}")
print(f"特征保留率: {len(selected_features) / len(feature_cols) * 100:.2f}%")
print(f"\n原始数据大小: {train_df.shape}")
print(f"筛选后数据大小: {train_selected.shape}")
print(f"\n最高重要性: {sorted_features[0][1]:.3f}")
print(f"最低重要性: {sorted_features[-1][1]:.3f}")
print(f"平均重要性: {np.mean(feature_importances):.3f}")
print(f"保留特征平均重要性: {np.mean(selected_importances):.3f}")
print("=" * 70)
