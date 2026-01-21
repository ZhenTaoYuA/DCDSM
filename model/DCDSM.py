import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_curve, roc_auc_score, confusion_matrix,
                             accuracy_score, f1_score, matthews_corrcoef,
                             precision_recall_curve, auc)

# ===================== 设置随机种子 =====================
RANDOM_SEED = 42

# 设置 NumPy 随机种子
np.random.seed(RANDOM_SEED)

# 设置 Python 随机种子
import random

random.seed(RANDOM_SEED)

# 设置 TensorFlow 随机种子
try:
    # TensorFlow 2.x
    tf.random.set_seed(RANDOM_SEED)
except AttributeError:
    # TensorFlow 1.x
    tf.set_random_seed(RANDOM_SEED)

# 设置 Python 哈希种子
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)

# 为了完全复现，建议禁用 GPU 上的非确定性操作（可选，会影响性能）
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # 使用 CPU
# tf.config.run_functions_eagerly(True)

print(f"随机种子已设置: {RANDOM_SEED}")

# ===================== 加载数据 =====================
print("加载数据中...")

# 加载手工特征（CSV格式）
train_manual_df = pd.read_csv('../new_feature_select/feature/train_RandomForest_top337.csv')
test_manual_df = pd.read_csv('../new_feature_select/feature/test_RandomForest_top337.csv')

# 提取标签
y_train = train_manual_df['LABEL'].values
y_test = test_manual_df['LABEL'].values

# 提取手工特征（排除sequence_id和LABEL列）
manual_feature_cols = [col for col in train_manual_df.columns if col not in ['sequence_id', 'LABEL']]
x_train_manual = train_manual_df[manual_feature_cols].values
x_test_manual = test_manual_df[manual_feature_cols].values

print(f"手工特征形状 - Train: {x_train_manual.shape}, Test: {x_test_manual.shape}")

# 加载语言模型特征（CSV格式，已经特征选择过）
train_bert_df = pd.read_csv('../new_feature_select/feature/train_bert_RandomForest_top474.csv')
test_bert_df = pd.read_csv('../new_feature_select/feature/test_bert_RandomForest_top474.csv')

# 提取语言模型特征（排除sequence_id和LABEL列）
bert_feature_cols = [col for col in train_bert_df.columns if col not in ['sequence_id', 'LABEL']]
x_train_bert = train_bert_df[bert_feature_cols].values
x_test_bert = test_bert_df[bert_feature_cols].values

print(f"语言模型特征形状 - Train: {x_train_bert.shape}, Test: {x_test_bert.shape}")

# ===================== 特征预处理 =====================
print("\n特征预处理中...")

# 标准化手工特征
scaler_manual = StandardScaler()
x_train_manual = scaler_manual.fit_transform(x_train_manual)
x_test_manual = scaler_manual.transform(x_test_manual)

# 标准化语言模型特征
scaler_bert = StandardScaler()
x_train_bert = scaler_bert.fit_transform(x_train_bert)
x_test_bert = scaler_bert.transform(x_test_bert)

# 获取特征的真实维度
manual_feature_dim = x_train_manual.shape[1]
bert_feature_dim = x_train_bert.shape[1]

# 为两种特征都加一个维度方便卷积：(N, feature_dim) -> (N, feature_dim, 1)
x_train_bert = np.expand_dims(x_train_bert, axis=2)
x_test_bert = np.expand_dims(x_test_bert, axis=2)
x_train_manual = np.expand_dims(x_train_manual, axis=2)
x_test_manual = np.expand_dims(x_test_manual, axis=2)

print(f"标准化和维度调整后:")
print(f"  手工特征 - Train: {x_train_manual.shape}, Test: {x_test_manual.shape}")
print(f"  语言模型特征 - Train: {x_train_bert.shape}, Test: {x_test_bert.shape}")


# ===================== 构建模型 =====================
def create_cnn_model(bert_feature_dim, manual_feature_dim):
    """
    构建多模态深度学习模型（双分支CNN-BiLSTM版本）
    - 输入1: 语言模型特征 (N, bert_feature_dim, 1)
    - 输入2: 手工特征 (N, manual_feature_dim, 1)
    """

    # ========== 语言模型分支 (CNN-BiLSTM) ==========
    input_bert = tf.keras.Input(shape=(bert_feature_dim, 1), name='bert_features')

    # CNN层
    y = tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(input_bert)
    y = tf.keras.layers.MaxPooling1D(pool_size=2)(y)

    # BiLSTM层
    y = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=128, return_sequences=True))(y)
    y = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=64, return_sequences=False))(y)
    y = tf.keras.layers.Dropout(0.1)(y)

    bert_output = tf.keras.layers.Dense(128, activation='relu', name='bert_encoding')(y)

    # ========== 手工特征分支 (CNN-BiLSTM) ==========
    input_manual = tf.keras.Input(shape=(manual_feature_dim, 1), name='manual_features')

    # CNN层
    x = tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(input_manual)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)

    # BiLSTM层
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=128, return_sequences=True))(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=64, return_sequences=False))(x)
    x = tf.keras.layers.Dropout(0.1)(x)

    manual_output = tf.keras.layers.Dense(128, activation='relu', name='manual_encoding')(x)

    # ========== 融合层 ==========
    merged = tf.keras.layers.concatenate([bert_output, manual_output])
    attention_layer = tf.keras.layers.Attention()([merged, merged, merged])

    # 输出层
    d = tf.keras.layers.Dense(256, activation='relu')(attention_layer)
    d = tf.keras.layers.Dropout(0.1)(d)
    d = tf.keras.layers.Dense(128, activation='relu')(d)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(d)

    model = tf.keras.Model([input_bert, input_manual], output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


# ===================== 10折交叉验证训练 =====================
print("\n" + "=" * 60)
print("开始10折交叉验证训练")
print("=" * 60)

num_folds = 10
skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=RANDOM_SEED)

best_accuracy = 0.0
best_model = None
best_params = None
fold_accuracies = []
fold_aucs = []
fold_auprcs = []  # 添加AUPRC列表
fold_sensitivities = []
fold_specificities = []
fold_mccs = []

for fold, (train_idx, val_idx) in enumerate(skf.split(x_train_bert, y_train)):
    print(f"\n{'=' * 60}")
    print(f"Fold {fold + 1}/{num_folds}")
    print(f"{'=' * 60}")

    # 准备折叠数据
    x_train_fold = {
        'bert_features': x_train_bert[train_idx],
        'manual_features': x_train_manual[train_idx]
    }
    x_val_fold = {
        'bert_features': x_train_bert[val_idx],
        'manual_features': x_train_manual[val_idx]
    }
    y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

    # 创建和训练模型
    model = create_cnn_model(bert_feature_dim, manual_feature_dim)
    early_stopping = EarlyStopping(monitor='val_loss', patience=20,
                                   restore_best_weights=True)

    history = model.fit(
        x_train_fold, y_train_fold,
        epochs=300,
        batch_size=32,
        validation_data=(x_val_fold, y_val_fold),
        callbacks=[early_stopping],
        verbose=1
    )

    # 评估验证集
    val_proba = model.predict(x_val_fold, verbose=0)
    val_predictions = (val_proba > 0.5).astype(int)

    val_accuracy = accuracy_score(y_val_fold, val_predictions)
    val_auc = roc_auc_score(y_val_fold, val_proba)

    # 计算AUPRC
    precision, recall, _ = precision_recall_curve(y_val_fold, val_proba)
    val_auprc = auc(recall, precision)

    fold_accuracies.append(val_accuracy)
    fold_aucs.append(val_auc)
    fold_auprcs.append(val_auprc)  # 保存AUPRC

    # 计算混淆矩阵指标
    conf_matrix = confusion_matrix(y_val_fold, val_predictions)
    tn, fp, fn, tp = conf_matrix.ravel()
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0
    sn = tp / (tp + fn) if (tp + fn) > 0 else 0

    fold_specificities.append(sp)
    fold_sensitivities.append(sn)

    mcc = matthews_corrcoef(y_val_fold, val_predictions)
    fold_mccs.append(mcc)

    print(f"Fold {fold + 1} Validation - Acc: {val_accuracy:.4f}, AUC: {val_auc:.4f}, "
          f"AUPRC: {val_auprc:.4f}, Sn: {sn:.4f}, Sp: {sp:.4f}, MCC: {mcc:.4f}")

    # 绘制损失曲线
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title(f'Fold {fold + 1} - Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'loss_fold_{fold + 1}.png', dpi=100, bbox_inches='tight')
    plt.close()

    # 保存最佳模型
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        best_model = model
        best_params = model.get_weights()
        print(f"✓ 新的最佳模型 (Accuracy: {best_accuracy:.4f})")

# ===================== 输出交叉验证结果 =====================
print(f"\n{'=' * 60}")
print("10折交叉验证结果")
print(f"{'=' * 60}")

for i in range(num_folds):
    print(f"Fold {i + 1:2d}: Acc={fold_accuracies[i]:.4f}, AUC={fold_aucs[i]:.4f}, "
          f"AUPRC={fold_auprcs[i]:.4f}, Sn={fold_sensitivities[i]:.4f}, Sp={fold_specificities[i]:.4f}")

mean_accuracy = np.mean(fold_accuracies)
mean_auc = np.mean(fold_aucs)
mean_auprc = np.mean(fold_auprcs)  # 计算平均AUPRC
mean_sensitivity = np.mean(fold_sensitivities)
mean_specificity = np.mean(fold_specificities)
mean_mcc = np.mean(fold_mccs)

print(f"\n平均验证结果:")
print(f"  Accuracy:    {mean_accuracy:.4f} ± {np.std(fold_accuracies):.4f}")
print(f"  AUC:         {mean_auc:.4f} ± {np.std(fold_aucs):.4f}")
print(f"  AUPRC:       {mean_auprc:.4f} ± {np.std(fold_auprcs):.4f}")
print(f"  Sensitivity: {mean_sensitivity:.4f} ± {np.std(fold_sensitivities):.4f}")
print(f"  Specificity: {mean_specificity:.4f} ± {np.std(fold_specificities):.4f}")
print(f"  MCC:         {mean_mcc:.4f} ± {np.std(fold_mccs):.4f}")

# ===================== 测试集评估 =====================
print(f"\n{'=' * 60}")
print("最佳模型测试集结果")
print(f"{'=' * 60}")

best_model.set_weights(best_params)

x_test_dict = {
    'bert_features': x_test_bert,
    'manual_features': x_test_manual
}

test_proba = best_model.predict(x_test_dict, verbose=0)
test_predictions = (test_proba > 0.5).astype(int)

test_accuracy = accuracy_score(y_test, test_predictions)
test_auc = roc_auc_score(y_test, test_proba)
test_f1 = f1_score(y_test, test_predictions)

# 计算测试集AUPRC
precision_test, recall_test, _ = precision_recall_curve(y_test, test_proba)
test_auprc = auc(recall_test, precision_test)

print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test AUC:      {test_auc:.4f}")
print(f"Test AUPRC:    {test_auprc:.4f}")
print(f"Test F1-Score: {test_f1:.4f}")

# 混淆矩阵详细指标
cm = confusion_matrix(y_test, test_predictions)
tn, fp, fn, tp = cm.ravel()
sp = tn / (tn + fp)
sn = tp / (tp + fn)
precision = tp / (tp + fp)
mcc = matthews_corrcoef(y_test, test_predictions)

print(f"\n混淆矩阵:")
print(cm)
print(f"\n详细指标:")
print(f"  Sensitivity (Sn/Recall): {sn:.4f}")
print(f"  Specificity (Sp):        {sp:.4f}")
print(f"  Precision:               {precision:.4f}")
print(f"  MCC:                     {mcc:.4f}")

# ===================== 绘制ROC曲线 =====================
fpr, tpr, thresholds = roc_curve(y_test, test_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {test_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve - Test Set')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.savefig('roc_curve_test.png', dpi=100, bbox_inches='tight')
plt.show()

# ===================== 绘制Precision-Recall曲线 =====================
plt.figure(figsize=(8, 6))
plt.plot(recall_test, precision_test, color='blue', lw=2,
         label=f'PR curve (AUPRC = {test_auprc:.4f})')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve - Test Set')
plt.legend(loc="lower left")
plt.grid(True, alpha=0.3)
plt.savefig('pr_curve_test.png', dpi=100, bbox_inches='tight')
plt.show()

print("\n✓ 模型训练完成！")