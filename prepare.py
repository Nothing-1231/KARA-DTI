import os.path, sys
import pandas as pd
import numpy as np
import pickle
from tdc.multi_pred import DTI
from rdkit import Chem
import torch
import argparse
import subprocess
import os
from Bio import SeqIO
import random

# 添加命令行参数解析
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='davis',
                    help='which dataset to use: davis, KIBA, Drugbank, celegans, BingdingDB, human')
args = parser.parse_args()

# 根据命令行参数设置数据集
data_set = ['davis', 'KIBA', 'Drugbank', 'celegans', 'BingdingDB', 'human']

try:
    data_op = data_set.index(args.dataset)
    print(f"使用数据集: {args.dataset}, 索引: {data_op}")
except ValueError:
    print(f"错误: 数据集 {args.dataset} 不在支持列表中 {data_set}")
    sys.exit(1)


def run_cdhit_clustering(protein_sequences, threshold=0.4, output_dir='./cdhit_output'):
    """
    运行CD-HIT聚类
    threshold: 40% 序列一致性阈值
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. 保存所有蛋白质序列到FASTA文件
    fasta_file = os.path.join(output_dir, 'protein_sequences.fasta')
    with open(fasta_file, 'w') as f:
        for seq_id, seq in protein_sequences.items():
            f.write(f'>{seq_id}\n{seq}\n')

    # 2. 运行CD-HIT
    output_prefix = os.path.join(output_dir, 'clusters')
    cmd = f'cd-hit -i {fasta_file} -o {output_prefix} -c {threshold} -n 5 -M 8000 -T 4'
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"CD-HIT 运行失败: {e}")
        print("请确保 cd-hit 已安装并可在 PATH 中找到。")
        sys.exit(1)

    # 3. 读取聚类结果
    cluster_file = output_prefix + '.clstr'
    clusters = {}
    current_cluster = None

    with open(cluster_file, 'r') as f:
        for line in f:
            if line.startswith('>Cluster'):
                current_cluster = int(line.split()[1])
                clusters[current_cluster] = []
            elif line.strip():
                # 提取序列ID（去除长度信息等）
                seq_id = line.split('>')[1].split('...')[0]
                clusters[current_cluster].append(seq_id)

    return clusters


def generate_cluster_level_splits(data_df, clusters, split_type, drug_col='Drug_ID', target_col='Target_ID'):
    """
    基于CD-HIT聚类结果生成簇级别的冷启动分割
    返回 train_indices, test_indices
    """
    # 为每个蛋白质分配簇ID
    protein_to_cluster = {}
    for cluster_id, seqs in clusters.items():
        for seq_id in seqs:
            protein_to_cluster[seq_id] = cluster_id

    # 按簇划分训练/测试集
    unique_clusters = list(clusters.keys())
    random.shuffle(unique_clusters)

    if split_type == 'cold_target':
        # 将20%的簇放入测试集（可调整比例）
        test_cluster_ratio = 0.2
    elif split_type == 'cold_pair':
        test_cluster_ratio = 0.3  # 冷配对需要更多测试数据
    else:
        raise ValueError("split_type must be 'cold_target' or 'cold_pair'")

    num_test_clusters = int(len(unique_clusters) * test_cluster_ratio)
    test_clusters = set(unique_clusters[:num_test_clusters])

    # 根据簇分配划分数据
    train_indices = []
    test_indices = []

    for idx, row in data_df.iterrows():
        protein_id = row[target_col]
        cluster_id = protein_to_cluster.get(protein_id, -1)
        if cluster_id in test_clusters:
            test_indices.append(idx)
        else:
            train_indices.append(idx)

    # 对于 cold_pair，还需确保测试集中的药物不在训练集中
    if split_type == 'cold_pair':
        # 收集训练集中的药物ID
        train_drugs = set(data_df.loc[train_indices, drug_col].tolist())
        # 只保留那些药物ID不在训练集中的测试样本
        filtered_test = []
        for idx in test_indices:
            drug = data_df.loc[idx, drug_col]
            if drug not in train_drugs:
                filtered_test.append(idx)
        # 如果过滤后测试集太小，可适当调整，这里直接使用过滤后的
        print(f"Cold-pair: 原始测试样本 {len(test_indices)}, 过滤后 {len(filtered_test)}")
        test_indices = filtered_test

    return train_indices, test_indices


# 加载数据
print(f"加载数据集: {data_set[data_op]}")
data = DTI(name=data_set[data_op])

# 数据预处理（转换为对数形式，并设定阈值）
if data_op == 0:  # davis
    print("处理davis数据集...")
    data.convert_to_log(form="standard")
    thrshold = 7
elif data_op == 1:  # KIBA
    print("处理KIBA数据集...")
    data.convert_to_log(form="standard")
    thrshold = 9.0
elif data_op == 2:  # Drugbank
    print("处理Drugbank数据集...")
    data.convert_to_log(form="standard")
    thrshold = 3.5
elif data_op == 3:  # celegans
    print("处理celegans数据集...")
    data.convert_to_log(form="standard")
    thrshold = 3.5
elif data_op == 4:  # BindingDB
    print("处理BindingDB数据集...")
    data.convert_to_log(form="standard")
    thrshold = 3.5
elif data_op == 5:  # human
    print("处理human数据集...")
    data.convert_to_log(form="standard")
    thrshold = 9.0

base_path = f'./data/{data_set[data_op]}/'
os.makedirs(base_path, exist_ok=True)

print(f"开始生成4种数据分割方式...")

# 获取所有蛋白质序列（用于CD-HIT聚类）
# 注意：data.data 是完整的DataFrame，包含所有样本
all_proteins = data.data[['Target_ID', 'Target']].drop_duplicates().set_index('Target_ID')['Target'].to_dict()

# 预先运行CD-HIT聚类（只需运行一次，供split_type=2和3使用）
print("正在对蛋白质序列进行CD-HIT聚类（40%序列同一性）...")
clusters = run_cdhit_clustering(all_proteins, threshold=0.4, output_dir='./cdhit_output')
print(f"聚类完成，共 {len(clusters)} 个簇")

for split_type in range(4):
    print(f"\n正在生成分割方式 {split_type+1}...")

    if split_type == 0:      # random split
        split = data.get_split(method='random')
        path = base_path + 'split_s1/'
        split_name = '随机分割'
        # 直接使用TDC分割结果
        train_df = split['train']
        valid_df = split['valid']
        test_df = split['test']

    elif split_type == 1:    # cold-drug
        split = data.get_split(method='cold_split', column_name='Drug')
        path = base_path + 'split_s2/'
        split_name = '冷药物分割'
        train_df = split['train']
        valid_df = split['valid']
        test_df = split['test']

    elif split_type == 2:    # cold-target (使用CD-HIT簇级别)
        path = base_path + 'split_s3/'
        split_name = '冷靶标分割（CD-HIT）'
        # 使用簇级划分
        train_indices, test_indices = generate_cluster_level_splits(
            data.data, clusters, 'cold_target'
        )
        # 从训练集中随机抽取10%作为验证集
        random.shuffle(train_indices)
        split_point = int(len(train_indices) * 0.9)
        train_indices, valid_indices = train_indices[:split_point], train_indices[split_point:]
        # 构建DataFrame
        train_df = data.data.loc[train_indices]
        valid_df = data.data.loc[valid_indices]
        test_df = data.data.loc[test_indices]

    elif split_type == 3:    # cold-pair (使用CD-HIT簇级别 + 药物冷启动)
        path = base_path + 'split_s4/'
        split_name = '冷药物+冷靶标分割（CD-HIT）'
        train_indices, test_indices = generate_cluster_level_splits(
            data.data, clusters, 'cold_pair'
        )
        # 从训练集中随机抽取10%作为验证集
        random.shuffle(train_indices)
        split_point = int(len(train_indices) * 0.9)
        train_indices, valid_indices = train_indices[:split_point], train_indices[split_point:]
        train_df = data.data.loc[train_indices]
        valid_df = data.data.loc[valid_indices]
        test_df = data.data.loc[test_indices]

    # 创建目录
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"创建目录: {path}")

    print(f"{split_name} - 路径: {path}")
    print(f"训练集: {len(train_df)}, 验证集: {len(valid_df)}, 测试集: {len(test_df)}")

    # 构建映射字典（与原来一致）
    drug_id_sets = {}
    target_id_sets = {}
    re_drug_id_sets = {}
    re_target_id_sets = {}
    target_id2text = {}
    drug_id2mol = {}

    # 药物映射（遍历所有数据）
    all_drug_ids = pd.concat([train_df['Drug_ID'], valid_df['Drug_ID'], test_df['Drug_ID']]).unique()
    for i, drug_id in enumerate(all_drug_ids):
        drug_id_sets[i] = drug_id
        re_drug_id_sets[drug_id] = i
        # 获取SMILES（从原始数据中）
        drug_smiles = data.data[data.data['Drug_ID'] == drug_id]['Drug'].iloc[0]
        drug_id2mol[i] = drug_smiles

    # 靶标映射
    all_target_ids = pd.concat([train_df['Target_ID'], valid_df['Target_ID'], test_df['Target_ID']]).unique()
    for i, target_id in enumerate(all_target_ids):
        target_id_sets[i] = target_id
        re_target_id_sets[target_id] = i
        target_seq = data.data[data.data['Target_ID'] == target_id]['Target'].iloc[0]
        target_id2text[i] = target_seq

    print(f"药物数量: {len(drug_id_sets)}, 靶标数量: {len(target_id_sets)}")

    # 生成训练/验证/测试集文件（转换为0/1标签）
    def convert_to_binary(df, set_name):
        new_list = []
        for _, row in df.iterrows():
            drug_id = row['Drug_ID']
            target_id = row['Target_ID']
            y = row['Y']
            if y >= thrshold:
                new_list.append([re_drug_id_sets[drug_id], re_target_id_sets[target_id], 1])
            else:
                new_list.append([re_drug_id_sets[drug_id], re_target_id_sets[target_id], 0])
        pd.DataFrame(new_list).to_csv(path + f'{set_name}_data.csv', index=False, header=False)
        print(f"{set_name} 生成 {len(new_list)} 条")

    convert_to_binary(train_df, 'train')
    convert_to_binary(valid_df, 'valid')
    convert_to_binary(test_df, 'test')

print(f"\n所有4种数据分割方式已生成完成！")
