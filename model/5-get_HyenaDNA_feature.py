# !/usr/bin/env python
# coding=utf-8
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import os
import subprocess
import numpy as np
import pandas as pd
from transformers import PreTrainedModel, AutoModelForCausalLM, PretrainedConfig
from standalone_hyenadna import HyenaDNAModel
from standalone_hyenadna import CharacterTokenizer
from load import inject_substring, load_weights, HyenaDNAPreTrainedModel

import torch

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f"使用设备: {device}")


def load_sequence_data(file_path):
    """加载序列数据（只读取偶数行的序列信息，跳过序列ID行）"""
    try:
        # 读取所有行
        with open(file_path, 'r') as f:
            lines = f.readlines()

        # 提取偶数行的序列（索引1, 3, 5...对应文件中的第2, 4, 6行）
        sequences = []
        for i in range(1, len(lines), 2):  # 从索引1开始，步长为2
            sequence = lines[i].strip()  # 去除换行符
            if sequence:  # 确保序列不为空
                sequences.append(sequence)

        # 转换为DataFrame
        data = pd.DataFrame(sequences, columns=[0])
        print(f"成功加载文件: {file_path}, 序列数量: {len(data)}")
        return data
    except Exception as e:
        print(f"加载文件失败 {file_path}: {e}")
        return None


def extract_features(model, tokenizer, sequences, device, max_length):
    """
    提取序列特征

    Args:
        model: HyenaDNA模型
        tokenizer: 分词器
        sequences: 序列数据
        device: 计算设备
        max_length: 最大长度

    Returns:
        features_np: 特征矩阵 (N, 401, 128)
    """
    # 初始化特征矩阵 - 注意这里应该是401而不是原文件中的序列长度
    features_np = np.zeros((len(sequences), 401, 128), dtype=np.float32)

    print(f"开始提取特征，序列数量: {len(sequences)}")

    with torch.inference_mode():
        for i in range(len(sequences)):
            if i % 100 == 0:  # 每100个序列显示进度
                print(f"处理进度: {i}/{len(sequences)}")

            sequence = sequences.iloc[i, 0]

            # 分词化序列
            tok_seq = tokenizer(sequence)
            tok_seq = tok_seq["input_ids"]

            # 转换为tensor并添加batch维度
            tok_seq = torch.LongTensor(tok_seq).unsqueeze(0)
            tok_seq = tok_seq.to(device=device)

            # 获取嵌入特征
            embeddings = model(tok_seq)

            # 将特征保存到numpy数组
            features_np[i] = embeddings.cpu().numpy()

    print("特征提取完成")
    return features_np


def main():
    """主函数"""
    print("=" * 60)
    print("HyenaDNA特征提取与差值计算")
    print("=" * 60)

    # 配置模型参数
    pretrained_model_name = 'hyenadna-tiny-1k-seqlen'
    pretrained_model_path = '../model_pretrained/HyenaDNA_pretrained/HuggingFace'

    max_lengths = {
        'hyenadna-tiny-1k-seqlen': 1024,
        'hyenadna-small-32k-seqlen': 32768,
        'hyenadna-medium-160k-seqlen': 160000,
        'hyenadna-medium-450k-seqlen': 450000,
        'hyenadna-large-1m-seqlen': 1_000_000,
    }
    max_length = max_lengths[pretrained_model_name]

    batch_size = 4
    use_padding = True
    rc_aug = False
    add_eos = False
    use_head = False
    n_classes = 2
    backbone_cfg = None

    print(f"模型配置: {pretrained_model_name}")
    print(f"最大序列长度: {max_length}")

    # 加载序列数据
    print("\n1. 加载序列数据...")
    alt_data = load_sequence_data("../data/sequence/language_model_sequence/case_analysis/HyenaDNA_sequence_399bp/case_analysis_38_399bp_num_alt.txt")  # 突变后序列
    ref_data = load_sequence_data("../data/sequence/language_model_sequence/case_analysis/HyenaDNA_sequence_399bp/case_analysis_399bp_num_ref.txt")  # 参考序列（突变前）

    if alt_data is None or ref_data is None:
        print("数据加载失败，退出程序")
        return

    if len(alt_data) != len(ref_data):
        print(f"警告: 两个文件的序列数量不一致 - ALT: {len(alt_data)}, REF: {len(ref_data)}")
        min_len = min(len(alt_data), len(ref_data))
        alt_data = alt_data.iloc[:min_len]
        ref_data = ref_data.iloc[:min_len]
        print(f"已截取到相同长度: {min_len}")

    # 加载预训练模型
    print("\n2. 加载HyenaDNA模型...")
    try:
        model = HyenaDNAPreTrainedModel.from_pretrained(
            pretrained_model_path,
            pretrained_model_name,
            download=False,
            config=backbone_cfg,
            device=device,
            use_head=use_head,
            n_classes=n_classes,
        )
        model.to(device=device)
        model.eval()
        print("模型加载成功")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    # 创建分词器
    print("\n3. 创建分词器...")
    tokenizer = CharacterTokenizer(
        characters=['A', 'C', 'G', 'T', 'N'],
        model_max_length=max_length + 2,
        add_special_tokens=False,
        padding_side='left',
    )
    print("分词器创建成功")

    # 提取ALT序列特征（突变后）
    print("\n4. 提取ALT序列特征（突变后）...")
    alt_features = extract_features(model, tokenizer, alt_data, device, max_length)
    print(f"ALT特征矩阵形状: {alt_features.shape}")

    # 提取REF序列特征（参考序列/突变前）
    print("\n5. 提取REF序列特征（突变前）...")
    ref_features = extract_features(model, tokenizer, ref_data, device, max_length)
    print(f"REF特征矩阵形状: {ref_features.shape}")

    # 计算差值矩阵 (ALT - REF)
    print("\n6. 计算差值矩阵...")
    diff_features = alt_features - ref_features
    print(f"差值矩阵形状: {diff_features.shape}")
    print(f"第一个样本的差值矩阵:")
    print(diff_features[0])

    # 保存特征矩阵
    print("\n7. 保存特征矩阵...")

    # 保存为numpy格式
    np.save('../data/feature/language_model_feature/case_analysis/HyenaDNA_case_analysis_alt_features.npy', alt_features)
    np.save('../data/feature/language_model_feature/case_analysis/HyenaDNA_case_analysis_ref_features.npy', ref_features)
    np.save('../data/feature/language_model_feature/case_analysis/HyenaDNA_case_analysis_diff_features.npy', diff_features)



    # 验证保存的文件
    print("\n8. 验证保存的文件...")
    try:
        loaded_diff = np.load('../data/feature/language_model_feature/case_analysis/HyenaDNA_case_analysis_diff_features.npy')
        print(f"验证成功 - 差值矩阵形状: {loaded_diff.shape}")
        print(f"验证成功 - 数据一致性: {np.allclose(diff_features, loaded_diff)}")
    except Exception as e:
        print(f"验证失败: {e}")

    print("\n" + "=" * 60)
    print("处理完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()