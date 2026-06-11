import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import torch
from transformers import AutoModel, FeatureExtractionPipeline, BertTokenizer
import os
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ===== 模型配置 =====
dimm = 16  # 保留前256维向量
smiles_dict = {
    "A": 'Nc1ncnc2[nH]cnc12',
    "T": 'CC1=CNC(=O)NC1=O',
    "C": 'C1=C(NC(=O)N=C1)N',
    "G": 'O=C1c2ncnc2nc(N)N1'
}

model_configs = {
    "4mC-BERT": {"path": "../model_pretrained/chemcialbert_pretrained/lib/bert_model/4mC-bert-base-cased",
                 "tokenizer": "../model_pretrained/chemcialbert_pretrained/lib/smile_token/vocab.txt"},
    "4mC-XLM-R": {"path": "../model_pretrained/chemcialbert_pretrained/lib/bert_model/4mC-xlm-roberta-base",
                  "tokenizer": "../model_pretrained/chemcialbert_pretrained/lib/smile_token/vocab.txt"},
    "PubChem10M": {"path": "../model_pretrained/chemcialbert_pretrained/lib/bert_model/4mC-PubChem10M-SMILES-BPE-450k",
                   "tokenizer": "../model_pretrained/chemcialbert_pretrained/lib/smile_token/vocab.txt"}
}


# 辅助函数
def wins(vec, size):
    return vec[:size]


def load_model_tokenizer(model_path, tokenizer_path):
    model = AutoModel.from_pretrained(model_path)
    tokenizer = BertTokenizer(vocab_file=tokenizer_path)
    return FeatureExtractionPipeline(model=model, tokenizer=tokenizer)


def get_base_vectors(pipeline):
    base_vecs = {}
    for base, smiles in smiles_dict.items():
        tokens = " ".join(smiles)
        vec = np.array(pipeline(tokens))[:, 0, :].flatten()
        base_vecs[base] = torch.tensor(wins(vec, dimm), dtype=torch.float)
    return base_vecs


# 初始化并缓存碱基向量
model_feature_dict = {}
for name, cfg in model_configs.items():
    logging.info(f"[加载模型] {name}")
    pipeline = load_model_tokenizer(cfg['path'], cfg['tokenizer'])
    model_feature_dict[name] = get_base_vectors(pipeline)


# 提取序列的化学特征
def extract_chemical_features(seq):
    features = []
    for model_name in model_configs:
        base_vecs = model_feature_dict[model_name]
        encoded = [base_vecs.get(base, torch.zeros(dimm)) for base in seq]
        features.append(torch.stack(encoded))  # (L, dimm)
    return torch.cat(features, dim=1)  # (L, dimm*num_models)


# 解析自定义txt序列格式，不使用Bio.SeqIO
def parse_txt(path):
    records = []
    with open(path) as f:
        seq = ''
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if seq:
                    records.append(seq)
                seq = ''
            else:
                seq += line
        if seq:
            records.append(seq)
    return records


# 主流程：读取自定义txt并保存特征张量，输出形状
# 保存为三个numpy文件，形状: (N, L, dimm*3)
def txt_to_tensors(ref_path, alt_path, output_dir):
    ref_seqs = parse_txt(ref_path)
    alt_seqs = parse_txt(alt_path)
    assert len(ref_seqs) == len(alt_seqs), "序列数量不一致"

    os.makedirs(output_dir, exist_ok=True)

    ref_list, alt_list, diff_list = [], [], []
    for ref_seq, alt_seq in zip(ref_seqs, alt_seqs):
        ref_feat = extract_chemical_features(ref_seq)
        alt_feat = extract_chemical_features(alt_seq)
        diff_feat = alt_feat - ref_feat
        ref_list.append(ref_feat)
        alt_list.append(alt_feat)
        diff_list.append(diff_feat)

    ref_tensor = torch.stack(ref_list)
    alt_tensor = torch.stack(alt_list)
    diff_tensor = torch.stack(diff_list)

    # 转换为numpy数组并保存为.npy格式
    ref_array = ref_tensor.numpy()
    alt_array = alt_tensor.numpy()
    diff_array = diff_tensor.numpy()

    out_ref = os.path.join(output_dir, 'chemcialbert_test_ref_features.npy')
    out_alt = os.path.join(output_dir, 'chemcialbert_test_alt_features.npy')
    out_diff = os.path.join(output_dir, 'chemcialbert_test_diff_features.npy')

    np.save(out_ref, ref_array)
    np.save(out_alt, alt_array)
    np.save(out_diff, diff_array)

    logging.info(f"[✔] Ref array shape: {tuple(ref_array.shape)} saved to {out_ref}")
    logging.info(f"[✔] Alt array shape: {tuple(alt_array.shape)} saved to {out_alt}")
    logging.info(f"[✔] Diff array shape: {tuple(diff_array.shape)} saved to {out_diff}")
    print("\n=== First sample feature matrices ===")
    print("Ref first sample (shape {}):\n".format(ref_array[0].shape), ref_array[0])
    print("Alt first sample (shape {}):\n".format(alt_array[0].shape), alt_array[0])
    print("Diff first sample (shape {}):\n".format(diff_array[0].shape), diff_array[0])


# 调用示例
txt_to_tensors('../data/sequence/balanced_test_Ann_401bp_num.txt', '../data/sequence/ALT/balanced_test_ALT.txt',
               '../data/feature')