
import os.path,sys
import pandas as pd
import numpy as np
import pickle
from tdc.multi_pred import DTI
from rdkit import Chem
import torch
import argparse

# 添加命令行参数解析
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='davis', help='which dataset to use: davis, KIBA, Drugbank, celegans,BingdingDB,human')
args = parser.parse_args()

# 根据命令行参数设置数据集
data_set = ['davis','KIBA','Drugbank','celegans','BingdingDB','human']

# 找到用户选择的数据集索引
try:
    data_op = data_set.index(args.dataset)
    print(f"使用数据集: {args.dataset}, 索引: {data_op}")
except ValueError:
    print(f"错误: 数据集 {args.dataset} 不在支持列表中 {data_set}")
    sys.exit(1)

# 加载数据
print(f"加载数据集: {data_set[data_op]}")
data = DTI(name=data_set[data_op])

# 数据预处理
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
    thrshold = 3.5  # 根据Drugbank数据调整
elif data_op == 3:  # celegans
    print("处理celegans数据集...")
    data.convert_to_log(form="standard")
    thrshold = 3.5  # 根据celegans数据调整
elif data_op == 4:  
    print("处理BingdingDB数据集...")
    data.convert_to_log(form="standard")
    thrshold = 3.5  
elif data_op == 5:  
    print("处理human数据集...")
    data.convert_to_log(form="standard")
    thrshold = 9.0  
base_path = f'./data/{data_set[data_op]}/'

print(f"开始生成4种数据分割方式...")

for split_type in range(4):
    print(f"\n正在生成分割方式 {split_type+1}...")
    
    # split type : 0 for random, 1 for S2 split:cold Drug, 2 for S3 split:cold target, 3 for S4 split:cold drug+target
    if split_type == 0:
        split = data.get_split(method='random')
        path = base_path + 'split_s1/'
        split_name = '随机分割'
    elif split_type == 1:
        split = data.get_split(method='cold_split', column_name='Drug')
        path = base_path + 'split_s2/'
        split_name = '冷药物分割'
    elif split_type == 2:
        split = data.get_split(method='cold_split', column_name='Target')
        path = base_path + 'split_s3/'
        split_name = '冷靶标分割'
    else:
        split = data.get_split(method='cold_split', column_name=['Target','Drug'])
        path = base_path + 'split_s4/'
        split_name = '冷药物+冷靶标分割'

    # 创建目录
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"创建目录: {path}")

    print(f"{split_name} - 路径: {path}")

    # 获取分割数据
    train_drug_id = np.array(split['train']['Drug_ID']).tolist()
    train_target_id = np.array(split['train']['Target_ID']).tolist()
    train_drug = np.array(split['train']['Drug']).tolist()
    train_target = np.array(split['train']['Target']).tolist()
    train_y = np.array(split['train']['Y'])

    valid_drug_id = np.array(split['valid']['Drug_ID']).tolist()
    valid_target_id = np.array(split['valid']['Target_ID']).tolist()
    valid_drug = np.array(split['valid']['Drug']).tolist()
    valid_target = np.array(split['valid']['Target']).tolist()
    valid_y = np.array(split['valid']['Y'])

    test_drug_id = np.array(split['test']['Drug_ID']).tolist()
    test_target_id = np.array(split['test']['Target_ID']).tolist()
    test_drug = np.array(split['test']['Drug']).tolist()
    test_target = np.array(split['test']['Target']).tolist()
    test_y = np.array(split['test']['Y'])

    # 构建映射字典
    drug_id_sets = {}
    target_id_sets = {}
    re_drug_id_sets = {}
    re_target_id_sets = {}
    target_id2text = {}
    drug_id2mol = {}

    # 药物映射
    for ids, i in enumerate(train_drug_id):
        if i not in re_drug_id_sets:
            idx = len(drug_id_sets)
            drug_id_sets[idx] = i
            re_drug_id_sets[i] = idx
            drug_id2mol[idx] = train_drug[ids]
    for ids, i in enumerate(valid_drug_id):
        if i not in re_drug_id_sets:
            idx = len(drug_id_sets)
            drug_id_sets[idx] = i
            re_drug_id_sets[i] = idx
            drug_id2mol[idx] = valid_drug[ids]
    for ids, i in enumerate(test_drug_id):
        if i not in re_drug_id_sets:
            idx = len(drug_id_sets)
            drug_id_sets[idx] = i
            re_drug_id_sets[i] = idx
            drug_id2mol[idx] = test_drug[ids]

    # 靶标映射
    for ids, i in enumerate(train_target_id):
        if i not in re_target_id_sets:
            idx = len(target_id_sets)
            target_id_sets[idx] = i
            re_target_id_sets[i] = idx
            target_id2text[idx] = train_target[ids]
    for ids, i in enumerate(valid_target_id):
        if i not in re_target_id_sets:
            idx = len(target_id_sets)
            target_id_sets[idx] = i
            re_target_id_sets[i] = idx
            target_id2text[idx] = valid_target[ids]
    for ids, i in enumerate(test_target_id):
        if i not in re_target_id_sets:
            idx = len(target_id_sets)
            target_id_sets[idx] = i
            re_target_id_sets[i] = idx
            target_id2text[idx] = test_target[ids]

    print(f"药物数量: {len(drug_id_sets)}, 靶标数量: {len(target_id_sets)}")

    # 生成训练/验证/测试集文件
    new_train = []
    for (i, j, k) in zip(train_drug_id, train_target_id, train_y):
        if k >= thrshold:
            new_train.append([re_drug_id_sets[i], re_target_id_sets[j], 1])
        else:
            new_train.append([re_drug_id_sets[i], re_target_id_sets[j], 0])
    
    new_valid = []
    for (i, j, k) in zip(valid_drug_id, valid_target_id, valid_y):
        if k >= thrshold:
            new_valid.append([re_drug_id_sets[i], re_target_id_sets[j], 1])
        else:
            new_valid.append([re_drug_id_sets[i], re_target_id_sets[j], 0])
    
    new_test = []
    for (i, j, k) in zip(test_drug_id, test_target_id, test_y):
        if k >= thrshold:
            new_test.append([re_drug_id_sets[i], re_target_id_sets[j], 1])
        else:
            new_test.append([re_drug_id_sets[i], re_target_id_sets[j], 0])

    # 保存文件
    pd.DataFrame(new_train).to_csv(path + 'train_data.csv', index=False, header=False)
    pd.DataFrame(new_valid).to_csv(path + 'valid_data.csv', index=False, header=False)
    pd.DataFrame(new_test).to_csv(path + 'test_data.csv', index=False, header=False)
    
    print(f"生成完成: {len(new_train)}训练 + {len(new_valid)}验证 + {len(new_test)}测试")

print(f"\n所有4种数据分割方式已生成完成！")