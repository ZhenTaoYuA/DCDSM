import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

# 自动检测 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 预训练模型根目录（可通过环境变量 PRETRAIN_ROOT 指定）
PRETRAIN_ROOT = os.environ.get("PRETRAIN_ROOT", "../model_pretrained/dnabert_pretrained")
# 特征输出根目录
FEATURE_ROOT = "../data/feature/language_model_feature/case_analysis"
os.makedirs(FEATURE_ROOT, exist_ok=True)

# 读取FASTA文件，返回序列列表
def read_fasta(file_path):
    sequences = []
    with open(file_path, 'r') as f:
        seq = ''
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if seq:
                    sequences.append(seq)
                    seq = ''
            else:
                seq += line
        if seq:
            sequences.append(seq)
    return sequences

# 将序列转换为 k-mer 串（用空格分隔）
def seq2kmers(sequence, k):
    return " ".join([sequence[i:i + k] for i in range(len(sequence) - k + 1)])

# 文件路径定义
REF_FILE = "../data/sequence/language_model_sequence/case_analysis/case_analysis_38_401bp_num.txt"
ALT_FILE = "../data/sequence/language_model_sequence/case_analysis/ALT/case_analysis_38_ALT.txt"

for k in [3, 4, 5, 6]:
    print(f"\nProcessing k={k}")
    # 加载 k-mer 模型与分词器
    model_path = os.path.join(PRETRAIN_ROOT, str(k))
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    model.to(device)
    model.eval()

    # 读取序列
    ref_seqs = read_fasta(REF_FILE)
    alt_seqs = read_fasta(ALT_FILE)
    assert len(ref_seqs) == len(alt_seqs), "REF and ALT must have the same number of sequences"
    N = len(ref_seqs)
    H = model.config.hidden_size

    # 初始化输出数组
    ref_emb = np.zeros((N, 401, H), dtype=np.float32)
    alt_emb = np.zeros((N, 401, H), dtype=np.float32)

    # 内部函数：批量提取隐藏状态
    def extract_embeddings(seqs, emb_array):
        for idx, seq in enumerate(seqs):
            kmer_seq = seq2kmers(seq, k)
            inputs = tokenizer(kmer_seq, return_tensors='pt', truncation=True, max_length=401)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                hidden = model(**inputs).last_hidden_state[0].cpu().numpy()
            L = hidden.shape[0]
            emb_array[idx, :min(L, 401), :] = hidden[:401]

    # 分别提取
    extract_embeddings(ref_seqs, ref_emb)
    extract_embeddings(alt_seqs, alt_emb)

    # 保存 REF 和 ALT 特征
    ref_out = os.path.join(FEATURE_ROOT, f"DNABERT_unlabel_ref_k{k}.npy")
    alt_out = os.path.join(FEATURE_ROOT, f"DNABERT_unlabel_alt_k{k}.npy")
    np.save(ref_out, ref_emb)
    np.save(alt_out, alt_emb)
    print(f"Saved {ref_out} and {alt_out}")

    # 计算差值并保存
    diff = alt_emb - ref_emb
    diff_out = os.path.join(FEATURE_ROOT, f"DNABERT_unlabel_diff_k{k}.npy")
    np.save(diff_out, diff)
    print(f"Saved difference matrix to {diff_out}, shape = {diff.shape}")

print("All done.")

