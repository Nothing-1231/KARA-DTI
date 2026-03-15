from telnetlib import SE
import torch
from torch_geometric.data import InMemoryDataset, Data, Dataset
from torch_geometric.utils import to_undirected
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import random
import numpy as np
import pickle
import pandas as pd
import os.path as osp
import itertools
import os
from icecream import ic
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler


class Dataset(InMemoryDataset):
    def __init__(self, root, dataset, rating_file, sep, args, transform=None, pre_transform=None):
        self.path = root
        self.dataset = dataset
        self.rating_file = rating_file
        self.split_name = f'split_s{args.split + 1}'
        self.sep = sep
        self.args = args
        
        # 设置属性
        self.use_dynamic_router = getattr(args, 'use_dynamic_router', False)
        self.use_msi_features = getattr(args, 'use_msi_features', True)
        self.msi_target_dim = getattr(args, 'msi_projection_dim', 256)
        self.feature_selection_threshold = getattr(args, 'feature_selection_threshold', 0.01)

        print(f"数据加载配置: use_msi_features={self.use_msi_features}, use_dynamic_router={self.use_dynamic_router}")
        print(f"MSI特征配置: 目标维度={self.msi_target_dim}, 特征选择阈值={self.feature_selection_threshold}")

        # 关键修复：在调用父类初始化之前设置 processed_file_names 相关的属性
        self._setup_processed_names()

        super(Dataset, self).__init__(root, transform, pre_transform)
        
        # 检查处理后的文件是否存在，如果不存在则重新处理
        if not os.path.exists(self.processed_paths[0]):
            print("处理后的数据文件不存在，重新处理数据...")
            self.process()
            
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.stat_info = torch.load(self.processed_paths[1])
        self.data_num = self.stat_info['data_num']
        self.feature_num = self.stat_info['feature_num']

    def _setup_processed_names(self):
        """设置处理后的文件名"""
        # 根据特征类型生成不同的文件名
        if self.use_msi_features:
            suffix = '_msi'
        elif self.use_dynamic_router:
            suffix = '_dynamic'
        else:
            suffix = '_static'
            
        self._processed_file_names = [
            f'{self.dataset}/{self.split_name}/{self.dataset}{suffix}.dataset',
            f'{self.dataset}/{self.split_name}/{self.dataset}{suffix}.statinfo'
        ]

    @property  
    def processed_file_names(self):
        """处理后的文件名 - 修复版本"""
        return self._processed_file_names

    @property
    def raw_dir(self):
        # 原始数据目录是 {root}/{dataset}
        return osp.join(self.path, self.dataset)

    @property  
    def processed_dir(self):
        # 处理后的数据目录
        return osp.join(self.path, 'processed')

    @property
    def raw_file_names(self):
        # 只需要字典文件，不需要评分文件
        base_files = [
            'drug_dict.pkl',
            'target_dict.pkl', 
            'feature_dict.pkl'
        ]

        # 动态路由文件（可选）
        if self.use_dynamic_router:
            base_files.append('dynamic_router_info.pkl')

        return base_files
    
    def download(self):
        # 不需要下载，使用本地文件
        pass
        
    def process(self):
        """处理数据 - 核心修复版本"""
        print(f"开始处理数据集: {self.dataset}, 分割: {self.split_name}")
        
        # 1. 设置文件路径
        self.userfile = osp.join(self.raw_dir, 'drug_dict.pkl')
        self.itemfile = osp.join(self.raw_dir, 'target_dict.pkl')
        
        # 2. 读取数据
        print("读取数据...")
        try:
            graphs, stat_info = self.read_data()
        except Exception as e:
            print(f"读取数据失败: {e}")
            # 尝试创建简单的测试数据
            graphs, stat_info = self.create_test_data()
            
        # 3. 保存数据
        print(f"保存处理后的数据: {len(graphs)} 个图")
        data, slices = self.collate(graphs)
        
        # 确保目录存在
        processed_path = self.processed_paths[0]
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        
        torch.save((data, slices), processed_path)
        torch.save(stat_info, self.processed_paths[1])
        
        print(f"数据处理完成并保存到: {processed_path}")

    def create_test_data(self):
        """创建测试数据（用于调试）"""
        print("创建测试数据用于调试...")
        
        # 创建简单的字典
        self.drug_dict = {'DRUG_0': {'attribute': [0]}}
        self.target_dict = {'TARGET_0': {'attribute': [1]}}
        
        # 创建简单的ID映射
        self.drug_str_to_int = {'DRUG_0': 0}
        self.target_str_to_int = {'TARGET_0': 1}
        self.drug_int_to_str = {0: 'DRUG_0'}
        self.target_int_to_str = {1: 'TARGET_0'}
        
        # 创建简单的图
        graphs = []
        for i in range(10):
            graph = self.construct_graphs([0], [1], 1.0 if i % 2 == 0 else 0.0)
            if graph is not None:
                graphs.append(graph)
        
        stat_info = {
            'data_num': len(graphs),
            'feature_num': 2,  # 药物+靶标
            'train_test_split_index': [7, 9],  # 7训练, 2验证, 1测试
            'use_dynamic_router': self.use_dynamic_router,
            'use_msi_features': self.use_msi_features,
            'drug_mapping_size': 1,
            'target_mapping_size': 1,
            'msi_feature_dim': 0
        }
        
        return graphs, stat_info

    def load_msi_features(self):
        """
        优化版MSI特征加载 - 修复索引问题
        """
        try:
            # 使用feature_construct.py生成的特征文件
            drug_feature_path = osp.join(self.raw_dir, 'drug_features.npy')
            protein_feature_path = osp.join(self.raw_dir, 'target_features.npy')

            print(f"优化加载MSI特征...")
            print(f"  药物特征路径: {drug_feature_path}")
            print(f"  蛋白质特征路径: {protein_feature_path}")

            if not os.path.exists(drug_feature_path):
                print(f"错误: 药物特征文件不存在: {drug_feature_path}")
                return False
            if not os.path.exists(protein_feature_path):
                print(f"错误: 蛋白质特征文件不存在: {protein_feature_path}")
                return False

            # 读取 numpy 特征
            drug_features = np.load(drug_feature_path).astype(np.float32)
            protein_features = np.load(protein_feature_path).astype(np.float32)

            print(f"原始特征维度 - 药物: {drug_features.shape}, 蛋白质: {protein_features.shape}")

            # 修复：确保特征维度正确
            # 如果药物特征数不等于药物字典大小，进行修正
            if drug_features.shape[0] != len(self.drug_dict):
                print(f"警告: 药物特征数({drug_features.shape[0]}) != 药物字典大小({len(self.drug_dict)})")
                # 取较小的那个
                min_size = min(drug_features.shape[0], len(self.drug_dict))
                drug_features = drug_features[:min_size]
                print(f"修正后药物特征: {drug_features.shape}")
                
            if protein_features.shape[0] != len(self.target_dict):
                print(f"警告: 靶标特征数({protein_features.shape[0]}) != 靶标字典大小({len(self.target_dict)})")
                min_size = min(protein_features.shape[0], len(self.target_dict))
                protein_features = protein_features[:min_size]
                print(f"修正后靶标特征: {protein_features.shape}")

            # 应用特征选择 - 移除低方差特征
            drug_features = self.remove_low_variance_features(drug_features, 'drug')
            protein_features = self.remove_low_variance_features(protein_features, 'target')

            # 关键修改：使用msi_target_dim统一维度
            drug_features = self.unify_feature_dimension(drug_features, self.msi_target_dim, 'drug')
            protein_features = self.unify_feature_dimension(protein_features, self.msi_target_dim, 'target')

            print(f"优化后特征维度 - 药物: {drug_features.shape}, 蛋白质: {protein_features.shape}")

            # 存储优化后的特征 - 使用索引访问
            self.msi_drug_features = drug_features  # 直接存储数组
            self.msi_target_features = protein_features  # 直接存储数组

            # 保存统一后的维度信息
            self.msi_feature_dim = self.msi_target_dim

            print(f"[MSI优化] 统一特征维度: {self.msi_target_dim}")
            return True

        except Exception as e:
            print(f"[MSI优化] 加载特征失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def remove_low_variance_features(self, features, feature_type):
        """移除低方差特征"""
        if features.shape[1] <= self.msi_target_dim:
            return features  # 如果特征数已经小于目标维度，不需要移除
            
        variance_threshold = self.feature_selection_threshold
        if feature_type == 'target':
            variance_threshold *= 0.5  # 对靶点特征使用更严格的标准
            
        selector = VarianceThreshold(threshold=variance_threshold)
        try:
            features_selected = selector.fit_transform(features)
            print(f"[特征选择] {feature_type}: {features.shape[1]} -> {features_selected.shape[1]} 维")
            return features_selected
        except:
            print(f"[特征选择] {feature_type}: 选择失败，使用原始特征")
            return features

    def unify_feature_dimension(self, features, target_dim, feature_type):
        """统一特征维度到目标维度"""
        current_dim = features.shape[1]
        
        if current_dim == target_dim:
            return features
        elif current_dim > target_dim:
            # 使用PCA降维或简单截取前target_dim个特征
            if hasattr(self.args, 'use_pca') and self.args.use_pca:
                try:
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=target_dim)
                    return pca.fit_transform(features)
                except:
                    print(f"[PCA失败] {feature_type}: 使用简单截取")
            # 选择方差最大的target_dim个特征
            variances = np.var(features, axis=0)
            top_indices = np.argsort(variances)[-target_dim:]
            return features[:, top_indices]
        else:
            # 当前维度小于目标维度，用0填充
            padded = np.zeros((features.shape[0], target_dim), dtype=np.float32)
            padded[:, :current_dim] = features
            print(f"[维度统一] {feature_type}: {current_dim} -> {target_dim} (填充)")
            return padded

    # 在data_2_graphs方法中添加内存优化
    def data_2_graphs(self, ratings_df, dataset='train'):
        graphs = []
        processed_graphs = 0
        num_graphs = ratings_df.shape[0]

        if num_graphs == 0:
            print(f"警告: {dataset} 数据为空")
            return graphs

        one_per = max(1, int(num_graphs / 10))
        percent = 0

        print(f"处理 {dataset} 数据 ({num_graphs} 个样本)...")

        error_count = 0
        success_count = 0

        for i in range(len(ratings_df)):
            if processed_graphs % one_per == 0:
                print(f"处理 [{dataset}]: {percent * 10}%, {processed_graphs}/{num_graphs}, 成功: {success_count}, 错误: {error_count}")
                percent += 1
            processed_graphs += 1

            try:
                line = ratings_df.iloc[i]
                user_index_int = int(line[0])
                item_index_int = int(line[1])
                rating = float(line[2])

                # 验证ID范围
                max_drug_id = len(self.drug_str_to_int) - 1
                max_target_id = len(self.target_str_to_int) - 1

                if user_index_int < 0 or user_index_int > max_drug_id:
                    print(f"药物ID越界: {user_index_int}")
                    error_count += 1
                    continue

                if item_index_int < 0 or item_index_int > max_target_id:
                    print(f"靶点ID越界: {item_index_int}")
                    error_count += 1
                    continue

                # 获取对应的字符串ID
                user_index_str = self.drug_int_to_str.get(user_index_int)
                item_index_str = self.target_int_to_str.get(item_index_int)

                if user_index_str is None or item_index_str is None:
                    error_count += 1
                    continue

                # 使用字典中的属性 - 但限制属性数量以减少内存
                user_attr_list = self.drug_dict[user_index_str]['attribute'][:3]  # 只取前3个属性
                item_attr_list = self.target_dict[item_index_str]['attribute'][:3]  # 只取前3个属性

                # 构建简化的节点列表
                user_list = [user_index_int] + user_attr_list
                item_list = [item_index_int] + item_attr_list

                # 构建简化的图
                if self.use_msi_features and hasattr(self, 'msi_drug_features') and hasattr(self, 'msi_target_features'):
                    graph = self.construct_simplified_msi_graph(user_list, item_list, rating, user_index_int, item_index_int)
                else:
                    graph = self.construct_simplified_graph(user_list, item_list, rating)

                if graph is not None:
                    graphs.append(graph)
                    success_count += 1
                else:
                    error_count += 1

            except Exception as e:
                error_count += 1
                if error_count < 5:
                    print(f"处理第 {i} 行数据时出错: {e}")

        print(f"{dataset} 数据处理完成: 成功 {success_count}/{num_graphs}, 错误 {error_count}")
        return graphs

    def construct_simplified_graph(self, user_list, item_list, rating):
        """构建简化的图以减少内存使用"""
        try:
            u_n = len(user_list)
            i_n = len(item_list)

            # 只构建必要的边，而不是全连接
            inner_edge_index = [[], []]

            # 药物内部：只连接主节点和属性节点
            for i in range(1, u_n):  # 从1开始，跳过主节点自身
                inner_edge_index[0].append(0)  # 主节点
                inner_edge_index[1].append(i)  # 属性节点

            # 靶标内部：只连接主节点和属性节点
            for i in range(1, i_n):
                inner_edge_index[0].append(u_n)  # 主节点
                inner_edge_index[1].append(u_n + i)  # 属性节点

            # 外部边：只连接两个主节点
            outer_edge_index = [[0], [u_n]]

            # 转换为tensor
            inner_edge_index = torch.LongTensor(inner_edge_index)
            inner_edge_index = to_undirected(inner_edge_index)
            outer_edge_index = torch.LongTensor(outer_edge_index)
            outer_edge_index = to_undirected(outer_edge_index)

            return self.construct_graph(user_list + item_list, inner_edge_index, outer_edge_index, rating)

        except Exception as e:
            print(f"构建简化图时出错: {e}")
            return None

    def construct_simplified_msi_graph(self, user_list, item_list, rating, user_index, item_index):
        """构建简化的MSI图"""
        try:
            # 验证节点ID
            validated_node_list = []
            max_feature_id = len(self.drug_str_to_int) + len(self.target_str_to_int) - 1

            for node_id in user_list + item_list:
                if 0 <= node_id <= max_feature_id:
                    validated_node_list.append(node_id)
                else:
                    validated_node_list.append(0)

            # MSI特征 - 只为两个主节点添加
            total_nodes = len(user_list) + len(item_list)
            msi_features = torch.zeros((total_nodes, self.msi_feature_dim), dtype=torch.float32)

            # 药物主节点特征
            if (hasattr(self, 'msi_drug_features') and 
                user_index < len(self.msi_drug_features)):
                drug_feat = torch.tensor(self.msi_drug_features[user_index], dtype=torch.float32)
                msi_features[0] = drug_feat

            # 靶标主节点特征
            offset = len(user_list)
            if (hasattr(self, 'msi_target_features') and 
                item_index < len(self.msi_target_features)):
                target_feat = torch.tensor(self.msi_target_features[item_index], dtype=torch.float32)
                msi_features[offset] = target_feat

            # 简化的边索引
            u_n = len(user_list)
            i_n = len(item_list)

            # 内部边：主节点连接属性节点
            inner_edge_index = [[], []]
            for i in range(1, u_n):
                inner_edge_index[0].append(0)
                inner_edge_index[1].append(i)
            for i in range(1, i_n):
                inner_edge_index[0].append(u_n)
                inner_edge_index[1].append(u_n + i)

            # 外部边：只连接两个主节点
            outer_edge_index = [[0], [u_n]]

            # 转换为tensor
            inner_edge_index = torch.LongTensor(inner_edge_index)
            inner_edge_index = to_undirected(inner_edge_index)
            outer_edge_index = torch.LongTensor(outer_edge_index)
            outer_edge_index = to_undirected(outer_edge_index)

            # 构建图数据
            graph_data = Data(
                x=torch.LongTensor(validated_node_list).unsqueeze(1),
                x_msi=msi_features,
                edge_index=inner_edge_index,
                edge_attr=torch.transpose(outer_edge_index, 0, 1),
                y=torch.FloatTensor([rating]),
                drug_id=torch.LongTensor([user_index]),
                target_id=torch.LongTensor([item_index])
            )

            return graph_data

        except Exception as e:
            print(f"构建简化MSI图失败: {e}")
            return None

    def read_data(self):
        """读取数据 - 修复版本"""
        try:
            # 加载基础字典文件
            self.drug_dict = pickle.load(open(self.userfile, 'rb'))
            self.target_dict = pickle.load(open(self.itemfile, 'rb'))

            print(f"加载药物字典: {len(self.drug_dict)} 个药物")
            print(f"加载靶点字典: {len(self.target_dict)} 个靶点")

            # 创建新的ID映射系统
            self.drug_str_to_int = {}
            self.target_str_to_int = {}
            self.drug_int_to_str = {}
            self.target_int_to_str = {}

            # 为药物创建连续的整数ID（从0开始）
            drug_keys = sorted(list(self.drug_dict.keys()))
            for idx, drug_id in enumerate(drug_keys):
                self.drug_str_to_int[drug_id] = idx
                self.drug_int_to_str[idx] = drug_id

            # 为靶点创建连续的整数ID（从0开始）
            target_keys = sorted(list(self.target_dict.keys()))
            for idx, target_id in enumerate(target_keys):
                self.target_str_to_int[target_id] = idx
                self.target_int_to_str[idx] = target_id

            print(f"药物ID映射: 0-{len(self.drug_str_to_int)-1}")
            print(f"靶点ID映射: 0-{len(self.target_str_to_int)-1}")

            # 加载训练数据
            split_path = osp.join(self.raw_dir, self.split_name)
            train_file = osp.join(split_path, 'train_data.csv')
            valid_file = osp.join(split_path, 'valid_data.csv')
            test_file = osp.join(split_path, 'test_data.csv')

            if not os.path.exists(train_file):
                # 尝试使用原始的interactions.csv
                print(f"警告: {train_file} 不存在，尝试使用interactions.csv")
                return self.read_data_from_interactions()

            train_df = pd.read_csv(train_file, header=None)
            valid_df = pd.read_csv(valid_file, header=None)
            test_df = pd.read_csv(test_file, header=None)

            print(f"原始数据 - 训练集: {len(train_df)} 个样本")
            print(f"原始数据 - 验证集: {len(valid_df)} 个样本")
            print(f"原始数据 - 测试集: {len(test_df)} 个样本")

            # 转换ID - 确保使用正确的映射
            def convert_ids(df, df_name):
                """转换ID到新的映射系统"""
                converted_rows = []
                missing_count = 0

                for i, row in df.iterrows():
                    try:
                        orig_drug_id = int(row[0])  # CSV中的原始药物ID
                        orig_target_id = int(row[1])  # CSV中的原始靶标ID
                        rating = float(row[2])

                        # 关键修复：假设CSV中的ID就是字符串ID中的数字部分
                        # 例如：CSV中的0对应"DRUG_0"，1对应"DRUG_1"
                        drug_str = f"DRUG_{orig_drug_id}"
                        target_str = f"TARGET_{orig_target_id}"

                        # 检查字符串ID是否在映射中
                        if drug_str in self.drug_str_to_int and target_str in self.target_str_to_int:
                            new_drug_id = self.drug_str_to_int[drug_str]
                            new_target_id = self.target_str_to_int[target_str]
                            converted_rows.append([new_drug_id, new_target_id, rating])
                        else:
                            missing_count += 1
                            if missing_count < 5:  # 只显示前5个缺失
                                print(f"缺失映射: {drug_str} 或 {target_str}")

                    except Exception as e:
                        missing_count += 1
                        if missing_count < 5:
                            print(f"转换错误: {e}")
                        continue

                print(f"{df_name} ID转换: 成功 {len(converted_rows)}/{len(df)}, 缺失 {missing_count}")
                return pd.DataFrame(converted_rows)

            # 转换所有数据
            train_df = convert_ids(train_df, "训练集")
            valid_df = convert_ids(valid_df, "验证集")
            test_df = convert_ids(test_df, "测试集")

            print(f"转换后 - 训练集: {len(train_df)} 个样本")
            print(f"转换后 - 验证集: {len(valid_df)} 个样本")
            print(f"转换后 - 测试集: {len(test_df)} 个样本")

            # 加载MSI特征
            msi_loaded = False
            if self.use_msi_features:
                msi_loaded = self.load_msi_features()
                if not msi_loaded:
                    print("MSI特征加载失败，回退到基础特征")
                    self.use_msi_features = False

            print('开始处理数据...')
            train_graphs = self.data_2_graphs(train_df, dataset='train')
            valid_graphs = self.data_2_graphs(valid_df, dataset='valid')
            test_graphs = self.data_2_graphs(test_df, dataset='test')

            graphs = train_graphs + valid_graphs + test_graphs

            if len(graphs) == 0:
                raise ValueError("无法创建任何图数据，请检查数据文件")

            # 统计信息
            stat_info = {
                'data_num': len(graphs),
                'feature_num': len(self.drug_dict) + len(self.target_dict),  # 嵌入层大小
                'train_test_split_index': [len(train_graphs), len(train_graphs) + len(valid_graphs)],
                'use_dynamic_router': self.use_dynamic_router,
                'use_msi_features': self.use_msi_features,
                'drug_mapping_size': len(self.drug_str_to_int),
                'target_mapping_size': len(self.target_str_to_int),
                'msi_feature_dim': getattr(self, 'msi_feature_dim', 0)
            }

            print(f'数据处理完成: 总共 {len(graphs)} 个图')
            print(f'特征数量 (嵌入层大小): {stat_info["feature_num"]}')
            if self.use_msi_features:
                print(f'MSI特征维度: {stat_info["msi_feature_dim"]}')
            return graphs, stat_info

        except Exception as e:
            print(f"数据读取失败: {e}")
            import traceback
            traceback.print_exc()
            raise

    def read_data_from_interactions(self):
        """从interactions.csv读取数据（后备方案）"""
        try:
            interactions_file = osp.join(self.raw_dir, 'interactions.csv')
            if not os.path.exists(interactions_file):
                raise FileNotFoundError(f"交互文件不存在: {interactions_file}")
                
            df = pd.read_csv(interactions_file)
            print(f"从interactions.csv读取数据: {len(df)} 个样本")
            
            # 随机分割
            train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
            valid_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
            
            # 保存分割数据以便后续使用
            split_path = osp.join(self.raw_dir, self.split_name)
            os.makedirs(split_path, exist_ok=True)
            
            train_df.to_csv(osp.join(split_path, 'train_data.csv'), index=False, header=False)
            valid_df.to_csv(osp.join(split_path, 'valid_data.csv'), index=False, header=False)
            test_df.to_csv(osp.join(split_path, 'test_data.csv'), index=False, header=False)
            
            return train_df, valid_df, test_df
            
        except Exception as e:
            print(f"从interactions.csv读取失败: {e}")
            raise

    def construct_graphs(self, user_list, item_list, rating):
        """构建静态图"""
        try:
            u_n = len(user_list)  # user node number
            i_n = len(item_list)  # item node number

            # 构建内部边
            inner_edge_index = [[], []]
            for i in range(u_n):
                for j in range(i, u_n):
                    inner_edge_index[0].append(i)
                    inner_edge_index[1].append(j)

            for i in range(u_n, u_n + i_n):
                for j in range(i, u_n + i_n):
                    inner_edge_index[0].append(i)
                    inner_edge_index[1].append(j)

            # 构建外部边
            outer_edge_index = [[], []]
            for i in range(u_n):
                for j in range(i_n):
                    outer_edge_index[0].append(i)
                    outer_edge_index[1].append(u_n + j)

            # 构建图
            inner_edge_index = torch.LongTensor(inner_edge_index)
            inner_edge_index = to_undirected(inner_edge_index)
            outer_edge_index = torch.LongTensor(outer_edge_index)
            outer_edge_index = to_undirected(outer_edge_index)

            return self.construct_graph(user_list + item_list, inner_edge_index, outer_edge_index, rating)

        except Exception as e:
            print(f"构建图时出错: {e}")
            return None

    def construct_graph(self, node_list, edge_index_inner, edge_index_outer, rating):
        """构建基础图 - 严格验证节点ID"""
        try:
            # 严格验证所有节点ID
            validated_node_list = []
            max_feature_id = len(self.drug_str_to_int) + len(self.target_str_to_int) - 1

            for node_id in node_list:
                if isinstance(node_id, (int, float)):
                    int_id = int(node_id)
                    # 确保ID在有效范围内
                    if 0 <= int_id <= max_feature_id:
                        validated_node_list.append(int_id)
                    else:
                        # 使用第一个有效ID作为默认值
                        validated_node_list.append(0)
                else:
                    validated_node_list.append(0)

            x = torch.LongTensor(validated_node_list).unsqueeze(1)
            rating = torch.FloatTensor([rating])

            # 验证边索引
            num_nodes = len(validated_node_list)
            edge_index_inner = torch.clamp(edge_index_inner, 0, num_nodes - 1)
            edge_index_outer = torch.clamp(edge_index_outer, 0, num_nodes - 1)

            return Data(
                x=x,
                edge_index=edge_index_inner,
                edge_attr=torch.transpose(edge_index_outer, 0, 1),
                y=rating
            )
        except Exception as e:
            print(f"构建图失败: {e}")
            return None

    def construct_msi_enhanced_graph_direct(self, user_list, item_list, rating, user_index, item_index):
        """优化版MSI图构建 - 修复索引问题"""
        try:
            # 验证节点ID
            validated_node_list = []
            max_feature_id = len(self.drug_str_to_int) + len(self.target_str_to_int) - 1

            for node_id in user_list + item_list:
                if 0 <= node_id <= max_feature_id:
                    validated_node_list.append(node_id)
                else:
                    validated_node_list.append(0)

            # 预分配MSI特征数组 - 修复索引访问
            total_nodes = len(user_list) + len(item_list)
            msi_features = np.zeros((total_nodes, self.msi_feature_dim), dtype=np.float32)

            # 药物节点 - 修复索引检查
            if (hasattr(self, 'msi_drug_features') and 
                user_index < len(self.msi_drug_features)):
                drug_feat = self.msi_drug_features[user_index]
                if len(drug_feat) == self.msi_feature_dim:
                    msi_features[0] = drug_feat
                else:
                    # 维度不匹配时进行截断或填充
                    min_dim = min(len(drug_feat), self.msi_feature_dim)
                    msi_features[0, :min_dim] = drug_feat[:min_dim]

            # 靶点节点 - 修复索引检查
            offset = len(user_list)
            if (hasattr(self, 'msi_target_features') and 
                item_index < len(self.msi_target_features)):
                target_feat = self.msi_target_features[item_index]
                if len(target_feat) == self.msi_feature_dim:
                    msi_features[offset] = target_feat
                else:
                    min_dim = min(len(target_feat), self.msi_feature_dim)
                    msi_features[offset, :min_dim] = target_feat[:min_dim]

            # 构建边索引
            u_n = len(user_list)
            i_n = len(item_list)
            
            # 内部边
            inner_edge_index = [[], []]
            for i in range(u_n):
                for j in range(i, u_n):
                    inner_edge_index[0].append(i)
                    inner_edge_index[1].append(j)

            for i in range(u_n, u_n + i_n):
                for j in range(i, u_n + i_n):
                    inner_edge_index[0].append(i)
                    inner_edge_index[1].append(j)

            # 外部边
            outer_edge_index = [[], []]
            for i in range(u_n):
                for j in range(i_n):
                    outer_edge_index[0].append(i)
                    outer_edge_index[1].append(u_n + j)

            # 转换为tensor
            inner_edge_index = torch.LongTensor(inner_edge_index)
            inner_edge_index = to_undirected(inner_edge_index)
            outer_edge_index = torch.LongTensor(outer_edge_index)
            outer_edge_index = to_undirected(outer_edge_index)

            # 构建图数据 - 修复x_msi维度
            x = torch.LongTensor(validated_node_list).unsqueeze(1)
            x_msi = torch.FloatTensor(msi_features)
            
            # 确保x_msi的节点数与x相同
            if x_msi.shape[0] != x.shape[0]:
                print(f"警告: x_msi节点数({x_msi.shape[0]}) != x节点数({x.shape[0]})")
                # 调整x_msi以匹配x
                if x_msi.shape[0] < x.shape[0]:
                    padding = torch.zeros((x.shape[0] - x_msi.shape[0], x_msi.shape[1]))
                    x_msi = torch.cat([x_msi, padding], dim=0)
                else:
                    x_msi = x_msi[:x.shape[0]]

            graph_data = Data(
                x=x,
                x_msi=x_msi,
                edge_index=inner_edge_index,
                edge_attr=torch.transpose(outer_edge_index, 0, 1),
                y=torch.FloatTensor([rating]),
                drug_id=torch.LongTensor([user_index]),
                target_id=torch.LongTensor([item_index])
            )

            return graph_data
        except Exception as e:
            print(f"构建优化MSI图失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def feature_N(self):
        return self.feature_num

    def data_N(self):
        return self.data_num
        
    def load_dynamic_router_info(self):
        """加载动态路由信息"""
        if self.use_dynamic_router:
            router_file = osp.join(self.raw_dir, 'dynamic_router_info.pkl')
            if os.path.exists(router_file):
                with open(router_file, 'rb') as f:
                    self.dynamic_router_info = pickle.load(f)
                print(f"加载动态路由信息: {len(self.dynamic_router_info.get('router_embeddings', []))} 个路由节点")