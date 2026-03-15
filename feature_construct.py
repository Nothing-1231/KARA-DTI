import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool
import os
import pickle
import sys
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
current_dir = os.path.dirname(os.path.abspath(__file__))
code_dir = os.path.join(current_dir, 'code')
sys.path.insert(0, code_dir)

try:
    from layers import *
    print("成功导入layers模块")
except ImportError as e:
    print(f"导入layers失败: {e}")
    print("尝试直接导入...")
    # 如果还是失败，直接在这里定义必要的类
def main():
    """基于真实Davis数据集构建特征 - 修复RDKit兼容性"""
    print("=== 开始构建增强的MSI特征 ===")
    
    # 创建输出目录
    output_dir = 'data/BindingDB'
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出目录: {output_dir}")
    
    try:
        # 导入必要的库
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
            from rdkit.Chem import Lipinski, Crippen
            print("✓ 成功导入RDKit")
            # 检查可用的描述符
            print(f"RDKit版本: {Chem.rdBase.rdkitVersion}")
        except ImportError as e:
            print(f"✗ 无法导入RDKit: {e}")
            return
        
        # 1. 加载药物数据
        print("加载药物数据...")
        drugs_df = pd.read_csv('data/BindingDB/drugs.csv')
        print(f"药物数据: {drugs_df.shape}")
        
        # 2. 加载蛋白质数据
        print("加载蛋白质数据...")
        proteins_df = pd.read_csv('data/BindingDB/targets.csv')
        print(f"蛋白质数据: {proteins_df.shape}")
        
        # 3. 扩展的药物特征提取 - 修复兼容性
        print("提取扩展的药物分子特征...")
        drug_features = []
        drug_ids = []
        failed_drugs = 0
        
        for idx, row in drugs_df.iterrows():
            try:
                drug_id = row['drug_id']
                smiles = row['SMILES']
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    # 多种分子指纹和描述符
                    features = []
                    
                    # A. Morgan指纹 (ECFP-like)
                    morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
                    features.extend(morgan_fp)
                    
                    # B. RDKit拓扑指纹
                    rdkit_fp = AllChem.RDKFingerprint(mol, fpSize=512)
                    features.extend(rdkit_fp)
                    
                    # C. MACCS密钥
                    maccs_fp = AllChem.GetMACCSKeysFingerprint(mol)
                    features.extend(maccs_fp)
                    
                    # D. 物理化学描述符 - 使用兼容的版本
                    try:
                        # 基础描述符
                        base_descriptors = [
                            Descriptors.MolWt(mol),  # 分子量
                            Descriptors.MolLogP(mol),  # LogP
                            Descriptors.NumHDonors(mol),  # 氢键供体
                            Descriptors.NumHAcceptors(mol),  # 氢键受体
                            Descriptors.TPSA(mol),  # 极性表面积
                            Descriptors.NumRotatableBonds(mol),  # 可旋转键
                            Descriptors.NumAromaticRings(mol),  # 芳香环
                            Lipinski.NumHeteroatoms(mol),  # 杂原子数
                            Crippen.MolMR(mol),  # 摩尔折射率
                        ]
                        features.extend(base_descriptors)
                        
                        # 使用兼容的方式计算FractionCsp3
                        num_carbon = len([atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6])
                        if num_carbon > 0:
                            num_sp3_carbon = len([atom for atom in mol.GetAtoms() 
                                                if atom.GetAtomicNum() == 6 and atom.GetHybridization() == Chem.HybridizationType.SP3])
                            fraction_csp3 = num_sp3_carbon / num_carbon
                        else:
                            fraction_csp3 = 0
                        features.append(fraction_csp3)
                        
                        # 环信息
                        ring_info = [
                            rdMolDescriptors.CalcNumRings(mol),  # 环数
                            rdMolDescriptors.CalcNumAliphaticRings(mol),  # 脂肪环
                            rdMolDescriptors.CalcNumAromaticRings(mol),  # 芳香环
                            rdMolDescriptors.CalcNumHeterocycles(mol),  # 杂环
                        ]
                        features.extend(ring_info)
                        
                        # 原子组成
                        atom_features = [
                            Descriptors.HeavyAtomCount(mol),
                            Descriptors.HeavyAtomMolWt(mol),
                            rdMolDescriptors.CalcNumAmideBonds(mol),
                        ]
                        features.extend(atom_features)
                        
                    except Exception as desc_error:
                        print(f"描述符计算错误 {drug_id}: {desc_error}")
                        # 如果描述符计算失败，使用默认值
                        features.extend([0] * 17)  # 17个描述符的位置
                    
                    drug_features.append(np.array(features))
                    drug_ids.append(drug_id)
                    
                    # 每处理10个药物打印进度
                    if len(drug_features) % 10 == 0:
                        print(f"  已处理 {len(drug_features)} 个药物...")
                        
                else:
                    print(f"警告: 无法解析SMILES - 药物ID: {drug_id}")
                    failed_drugs += 1
            except Exception as e:
                print(f"处理药物 {row['drug_id']} 时出错: {e}")
                failed_drugs += 1
        
        if len(drug_features) > 0:
            drug_features = np.array(drug_features)
            print(f"成功提取 {len(drug_features)} 个药物特征, 失败: {failed_drugs}")
            print(f"药物特征维度: {drug_features.shape}")
        else:
            print("错误: 没有成功提取任何药物特征!")
            return
        
        # 4. 扩展的蛋白质特征提取
        print("提取扩展的蛋白质特征...")
        target_features = []
        target_ids = []
        failed_targets = 0
        
        # 尝试使用peptides库和自定义特征
        try:
            import peptides
            use_peptides = True
            print("✓ 使用peptides库提取蛋白质特征")
        except ImportError:
            use_peptides = False
            print("✗ peptides库未安装，使用简化特征提取")
        
        for idx, row in proteins_df.iterrows():
            try:
                target_id = row['protein_id']
                sequence = str(row['protein_fastas']).upper()  # 确保大写
                
                if pd.notna(sequence) and len(sequence) > 0:
                    features = []
                    
                    # A. 基础物理化学特征 (peptides库)
                    if use_peptides:
                        try:
                            peptide = peptides.Peptide(sequence)
                            physchem_features = [
                                peptide.aliphatic_index(),
                                peptide.boman(),
                                peptide.charge(pH=7.0),
                                peptide.hydrophobicity(scale="Eisenberg"),
                                peptide.instability_index(),
                                peptide.isoelectric_point(),
                                peptide.molar_extinction_coefficient(),
                                peptide.molecular_weight(),
                                peptide.optical_rotation(),
                            ]
                            features.extend(physchem_features)
                        except:
                            # 如果peptides失败，使用基础特征
                            features.extend([0] * 9)
                    
                    # B. 氨基酸组成特征 (类似iFeaturePro的AAC)
                    aac_features = calculate_amino_acid_composition(sequence)
                    features.extend(aac_features)
                    
                    # C. 二肽组成 (DPC)
                    dpc_features = calculate_dipeptide_composition(sequence)
                    features.extend(dpc_features)
                    
                    # D. 组成-转换-分布特征 (CTD)
                    ctd_features = calculate_ctd_descriptors(sequence)
                    features.extend(ctd_features)
                    
                    # E. 自相关特征
                    autocorrelation_features = calculate_autocorrelation(sequence)
                    features.extend(autocorrelation_features)
                    
                    # F. 序列长度和复杂性特征
                    complexity_features = calculate_sequence_complexity(sequence)
                    features.extend(complexity_features)
                    
                    # G. 二级结构倾向特征
                    ss_features = calculate_secondary_structure_propensity(sequence)
                    features.extend(ss_features)
                    
                    target_features.append(np.array(features))
                    target_ids.append(target_id)
                    
                    # 每处理50个蛋白质打印进度
                    if len(target_features) % 50 == 0:
                        print(f"  已处理 {len(target_features)} 个蛋白质...")
                        
                else:
                    print(f"警告: 无效的蛋白质序列 - 靶点ID: {target_id}")
                    failed_targets += 1
                    
            except Exception as e:
                print(f"处理蛋白质 {row['protein_id']} 时出错: {e}")
                failed_targets += 1
        
        if len(target_features) > 0:
            target_features = np.array(target_features)
            print(f"成功提取 {len(target_features)} 个靶点特征, 失败: {failed_targets}")
            print(f"靶点特征维度: {target_features.shape}")
        else:
            print("错误: 没有成功提取任何靶点特征!")
            return
        
        # 5. 构建字典结构
        print("构建字典结构...")
        
        # 药物字典
        drug_dict = {}
        for i, drug_id in enumerate(drug_ids):
            drug_dict[drug_id] = {
                'name': f'drug_{drug_id}',
                'smiles': drugs_df[drugs_df['drug_id'] == drug_id]['SMILES'].iloc[0],
                'feature_dim': drug_features.shape[1],
                'attribute': drug_features[i].tolist()[:20]  # 只保存前20个作为示例
            }
        
        # 靶点字典
        target_dict = {}
        for i, target_id in enumerate(target_ids):
            target_dict[target_id] = {
                'title': f'target_{target_id}',
                'sequence_length': len(str(proteins_df[proteins_df['protein_id'] == target_id]['protein_fastas'].iloc[0])),
                'feature_dim': target_features.shape[1],
                'attribute': target_features[i].tolist()[:20]  # 只保存前20个作为示例
            }
        
        # 特征字典
        feature_dict = {}
        idx = 0
        for drug_id in drug_ids:
            feature_dict[f'drug_{drug_id}'] = idx
            idx += 1
        for target_id in target_ids:
            feature_dict[f'target_{target_id}'] = idx
            idx += 1
        
        # 6. 保存文件
        print("保存特征文件...")
        
        # 保存npy文件
        np.save(os.path.join(output_dir, 'drug_features.npy'), drug_features)
        np.save(os.path.join(output_dir, 'target_features.npy'), target_features)
        print("✓ 保存npy特征文件")
        
        # 保存pkl文件
        with open(os.path.join(output_dir, 'drug_dict.pkl'), 'wb') as f:
            pickle.dump(drug_dict, f)
        with open(os.path.join(output_dir, 'target_dict.pkl'), 'wb') as f:
            pickle.dump(target_dict, f)
        with open(os.path.join(output_dir, 'feature_dict.pkl'), 'wb') as f:
            pickle.dump(feature_dict, f)
        print("✓ 保存pkl字典文件")
        
        # 7. 保存特征维度信息
        feature_info = {
            'drug_feature_types': {
                'morgan_fingerprint': 1024,
                'rdkit_fingerprint': 512,
                'maccs_keys': 167,
                'physicochemical_descriptors': 17,  # 更新为17个描述符
                'total_dimension': drug_features.shape[1]
            },
            'target_feature_types': {
                'physicochemical': 9,
                'amino_acid_composition': 20,
                'dipeptide_composition': 400,
                'ctd_descriptors': 39,
                'autocorrelation': 240,
                'sequence_complexity': 5,
                'secondary_structure': 3,
                'total_dimension': target_features.shape[1]
            }
        }
        
        with open(os.path.join(output_dir, 'feature_info.pkl'), 'wb') as f:
            pickle.dump(feature_info, f)
        print("✓ 保存特征信息文件")
        
        print("\n=== 增强特征构建完成 ===")
        print(f"药物特征: {drug_features.shape}")
        print(f"靶点特征: {target_features.shape}")
        print(f"特征信息已保存到 feature_info.pkl")
        
    except Exception as e:
        print(f"构建特征时出错: {e}")
        import traceback
        traceback.print_exc()

# ========== 蛋白质特征计算函数 ==========
# [保持原有的蛋白质特征计算函数不变]

def calculate_amino_acid_composition(sequence):
    """计算氨基酸组成 (AAC) - 20维"""
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    total_aa = len(sequence)
    aac = []
    
    for aa in amino_acids:
        count = sequence.count(aa)
        aac.append(count / total_aa if total_aa > 0 else 0)
    
    return aac

def calculate_dipeptide_composition(sequence):
    """计算二肽组成 (DPC) - 400维"""
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    dipeptides = [aa1 + aa2 for aa1 in amino_acids for aa2 in amino_acids]
    total_dipeptides = max(1, len(sequence) - 1)
    dpc = []
    
    for dp in dipeptides:
        count = 0
        for i in range(len(sequence) - 1):
            if sequence[i:i+2] == dp:
                count += 1
        dpc.append(count / total_dipeptides)
    
    return dpc

def calculate_ctd_descriptors(sequence):
    """计算组成-转换-分布特征 (CTD) - 39维"""
    # 按物理化学性质分组
    hydrophobic = 'ACFGILMPVWY'
    polar = 'NQST'
    positive = 'HKR'
    negative = 'DE'
    
    groups = [hydrophobic, polar, positive, negative]
    group_names = ['H', 'P', '+', '-']
    
    ctd_features = []
    
    # 组成 (Composition)
    total_aa = len(sequence)
    for group in groups:
        count = sum(1 for aa in sequence if aa in group)
        ctd_features.append(count / total_aa if total_aa > 0 else 0)
    
    # 转换 (Transition)
    for i in range(len(groups)):
        for j in range(i+1, len(groups)):
            trans_count = 0
            for k in range(len(sequence)-1):
                if (sequence[k] in groups[i] and sequence[k+1] in groups[j]) or \
                   (sequence[k] in groups[j] and sequence[k+1] in groups[i]):
                    trans_count += 1
            ctd_features.append(trans_count / max(1, len(sequence)-1))
    
    # 分布 (Distribution)
    for group in groups:
        positions = [i for i, aa in enumerate(sequence) if aa in group]
        if positions:
            for quartile in [0, 0.25, 0.5, 0.75, 1.0]:
                idx = int(quartile * (len(positions)-1))
                ctd_features.append(positions[idx] / len(sequence) if len(sequence) > 0 else 0)
        else:
            ctd_features.extend([0, 0, 0, 0, 0])
    
    return ctd_features

def calculate_autocorrelation(sequence, max_lag=5):
    """计算自相关特征"""
    # 使用不同的物理化学性质
    properties = {
        'hydrophobicity': {'A': 0.62, 'C': 0.29, 'D': -0.90, 'E': -0.74, 'F': 1.19,
                          'G': 0.48, 'H': -0.40, 'I': 1.38, 'K': -1.50, 'L': 1.06,
                          'M': 0.64, 'N': -0.78, 'P': 0.12, 'Q': -0.85, 'R': -2.53,
                          'S': -0.18, 'T': -0.05, 'V': 1.08, 'W': 0.81, 'Y': 0.26},
        'volume': {'A': 91.5, 'C': 117.7, 'D': 111.1, 'E': 138.4, 'F': 189.9,
                  'G': 66.4, 'H': 153.2, 'I': 168.8, 'K': 168.6, 'L': 167.9,
                  'M': 162.9, 'N': 114.1, 'P': 129.3, 'Q': 143.8, 'R': 173.4,
                  'S': 99.1, 'T': 122.1, 'V': 141.7, 'W': 227.8, 'Y': 193.6},
        'polarity': {'A': 8.1, 'C': 5.5, 'D': 13.0, 'E': 12.3, 'F': 5.2,
                    'G': 9.0, 'H': 10.4, 'I': 5.2, 'K': 11.3, 'L': 4.9,
                    'M': 5.7, 'N': 11.6, 'P': 8.0, 'Q': 10.5, 'R': 10.5,
                    'S': 9.2, 'T': 8.6, 'V': 5.9, 'W': 5.4, 'Y': 6.2}
    }
    
    autocorr_features = []
    n = len(sequence)
    
    for prop_name, prop_dict in properties.items():
        # 标准化性质值
        values = [prop_dict.get(aa, 0) for aa in sequence]
        mean_val = np.mean(values) if values else 0
        
        for lag in range(1, max_lag + 1):
            numerator = denominator = 0
            for i in range(n - lag):
                numerator += (values[i] - mean_val) * (values[i + lag] - mean_val)
                denominator += (values[i] - mean_val) ** 2
            
            if denominator > 0:
                autocorr = numerator / denominator
            else:
                autocorr = 0
            autocorr_features.append(autocorr)
    
    return autocorr_features

def calculate_sequence_complexity(sequence):
    """计算序列复杂性特征"""
    n = len(sequence)
    if n == 0:
        return [0, 0, 0, 0, 0]
    
    # 1. 香农熵
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    counts = [sequence.count(aa) for aa in amino_acids]
    proportions = [c / n for c in counts if c > 0]
    shannon_entropy = -sum(p * np.log2(p) for p in proportions) if proportions else 0
    
    # 2. Gini纯度
    gini = 1 - sum((c / n) ** 2 for c in counts)
    
    # 3. 重复模式比例
    unique_kmers = set()
    for i in range(n - 2):
        unique_kmers.add(sequence[i:i+3])
    kmer_diversity = len(unique_kmers) / max(1, n - 2)
    
    # 4. 低复杂性区域比例
    lc_regions = 0
    window_size = 10
    for i in range(0, n - window_size + 1, window_size):
        window = sequence[i:i+window_size]
        if len(set(window)) <= 3:  # 如果窗口中只有3种或更少的氨基酸
            lc_regions += 1
    lc_ratio = lc_regions / max(1, (n - window_size + 1) // window_size)
    
    # 5. 序列长度标准化
    normalized_length = n / 1000.0  # 假设平均长度1000
    
    return [shannon_entropy, gini, kmer_diversity, lc_ratio, normalized_length]

def calculate_secondary_structure_propensity(sequence):
    """计算二级结构倾向特征"""
    # Chou-Fasman参数 (简化版)
    helix_propensity = {'E': 1.51, 'A': 1.42, 'L': 1.21, 'H': 1.00, 'M': 1.45,
                       'Q': 1.11, 'W': 1.08, 'V': 1.06, 'F': 1.13, 'K': 1.16}
    
    sheet_propensity = {'V': 1.70, 'I': 1.60, 'Y': 1.47, 'F': 1.38, 'W': 1.37,
                       'L': 1.30, 'T': 1.19, 'C': 1.19, 'Q': 1.10, 'M': 1.05}
    
    coil_propensity = {'G': 1.00, 'N': 1.00, 'P': 1.00, 'S': 1.00, 'D': 1.00}
    
    n = len(sequence)
    if n == 0:
        return [0, 0, 0]
    
    helix_score = sum(helix_propensity.get(aa, 1.0) for aa in sequence) / n
    sheet_score = sum(sheet_propensity.get(aa, 1.0) for aa in sequence) / n
    coil_score = sum(coil_propensity.get(aa, 1.0) for aa in sequence) / n
    
    return [helix_score, sheet_score, coil_score]

if __name__ == "__main__":
    main()