from dataloader import Dataset
import argparse
from torch_geometric.data import DataLoader
from train import train
import torch
import random
import numpy as np
import time
import os

# 性能优化设置
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
torch.backends.cudnn.benchmark = True

# 支持的数据集列表
SUPPORTED_DATASETS = ['KIBA', 'davis', 'celegans', 'Drugbank','BindingDB','human']

parser = argparse.ArgumentParser(description='KARADTI - 药物靶点相互作用预测')
parser.add_argument('--dataset', type=str, default='BindingDB', 
                   choices=SUPPORTED_DATASETS,
                   help=f'选择数据集: {SUPPORTED_DATASETS}')
parser.add_argument('--split', type=int, default=0, 
                   help='数据分割方式: 0,1,2,3 分别对应 s1,s2,s3,s4')
parser.add_argument('--rating_file', type=str, default='implicit_ratings.csv')
parser.add_argument('--dim', type=int, default=64, help='嵌入维度')
parser.add_argument('--l2_weight', type=float, default=1e-5, help='L2正则化权重')
parser.add_argument('--lr', type=float, default=0.001, help='学习率')
parser.add_argument('--batch_size', type=int, default=256, help='批次大小')
parser.add_argument('--n_epoch', type=int, default=200, help='训练轮数')
parser.add_argument('--hidden_layer', type=int, default=256, help='隐藏层大小')
parser.add_argument('--num_user_features', type=int, default=-1)
parser.add_argument('--random_seed', type=int, default=2025, help='随机种子')
parser.add_argument('--inner_model', type=int, default=1, 
                   choices=[0, 1, 2], help='内部模型: 0=默认, 1=GCN, 2=GAT')
parser.add_argument('--cross_model', type=int, default=0,
                   choices=[0, 1, 2], help='跨域模型: 0=默认, 1=GCN, 2=GAT')
parser.add_argument('--use_dynamic_router', action='store_true', 
                   help='使用动态自适应路由器')
parser.add_argument('--use_dual_attention', action='store_true', 
                   default=False, help='使用DACMF双注意力机制')
parser.add_argument('--use_msi_features', type=lambda x: x.lower() == 'true', 
                   default=False, help='使用MSI多特征')

# 新增：投影维度相关参数
parser.add_argument('--msi_projection_dim', type=int, default=256,
                   help='MSI特征投影维度 (默认: 256)')
parser.add_argument('--use_msi_projection', type=lambda x: x.lower() == 'true',
                   default=True, help='是否使用MSI特征投影')
parser.add_argument('--feature_selection_threshold', type=float, default=0.01,
                   help='特征选择阈值 (移除低方差特征)')
parser.add_argument('--use_pca_projection', type=lambda x: x.lower() == 'true',
                   default=False, help='是否使用PCA进行投影')
parser.add_argument('--accumulation_steps', type=int, default=2,
                   help='梯度累积步数')

parser.add_argument('--num_runs', type=int, default=5, 
                   help='随机运行次数 (默认: 5次)')
parser.add_argument('--skip_existing', action='store_true',
                   help='跳过已存在的结果文件')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def setup_device():
    """设置设备"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.empty_cache()
        print(f"✓ 使用 GPU: {torch.cuda.get_device_name()}")
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✓ GPU内存: {gpu_memory:.1f} GB")
        return device
    else:
        device = torch.device('cpu')
        print("使用 CPU")
        return device

def get_dataset_config(dataset_name):
    """根据不同数据集调整配置"""
    configs = {
        'KIBA': {
            'batch_size': 256,
            'expected_samples': 116409,
            'recommended_projection_dim': 256
        },
        'davis': {
            'batch_size': 128,
            'expected_samples': 30056,
            'recommended_projection_dim': 128
        },
        'celegans': {
            'batch_size': 64,
            'expected_samples': 3916,
            'recommended_projection_dim': 64
        },
        'BindingDB': {
            'batch_size': 128,
            'expected_samples': 3916,
            'recommended_projection_dim': 128
        },
        'human': {
            'batch_size': 64,
            'expected_samples': 3916,
            'recommended_projection_dim': 64
        },
        'Drugbank': {
            'batch_size': 128,
            'expected_samples': 7306,
            'recommended_projection_dim': 128
        }
    }
    return configs.get(dataset_name, {'batch_size': 128, 'expected_samples': 10000, 'recommended_projection_dim': 128})

def validate_projection_config(args):
    """验证投影配置的合理性"""
    warnings = []
    
    # 检查MSI特征和投影的兼容性
    if args.use_msi_projection and not args.use_msi_features:
        warnings.append("⚠️  启用了MSI投影但未启用MSI特征，投影将不会生效")
    
    # 检查投影维度合理性
    if args.msi_projection_dim > 1024:
        warnings.append("⚠️  投影维度较大 (>1024)，可能增加计算负担")
    elif args.msi_projection_dim < 64:
        warnings.append("⚠️  投影维度较小 (<64)，可能丢失重要特征信息")
    
    # 检查数据集推荐的投影维度
    dataset_config = get_dataset_config(args.dataset)
    recommended_dim = dataset_config.get('recommended_projection_dim', 128)
    if args.msi_projection_dim != recommended_dim:
        warnings.append(f"💡 数据集 {args.dataset} 推荐投影维度为 {recommended_dim}")
    
    # 检查特征选择阈值
    if args.feature_selection_threshold < 0.001:
        warnings.append("⚠️  特征选择阈值过小，可能保留过多噪声特征")
    elif args.feature_selection_threshold > 0.1:
        warnings.append("⚠️  特征选择阈值过大，可能丢失重要特征")
    
    return warnings

def run_single_experiment(args, run_id):
    """运行单次实验"""
    separator = '=' * 70
    print(f'\n{separator}')
    print(f'🏃 开始第 {run_id+1}/{args.num_runs} 次运行')
    print(f'{separator}')
    
    start_time = time.time()
    
    try:
        # 数据加载
        dataset_path = '../data/'
        dataset = Dataset(dataset_path, args.dataset, args.rating_file, ',', args)
        
        # 自动调整特征数量
        feature_num = dataset.feature_N()
        if len(dataset) > 0:
            sample = dataset[0]
            max_node_id = sample.x.max().item()
            if max_node_id >= feature_num:
                feature_num = int(max_node_id) + 1
                print(f"📊 调整特征数量: {feature_num}")
        
        # 数据集分割
        train_index, val_index = dataset.stat_info['train_test_split_index']
        train_dataset = dataset[:train_index]
        val_dataset = dataset[train_index:val_index]
        test_dataset = dataset[val_index:]
        
        # 优化数据加载器
        n_workers = 4 if torch.cuda.is_available() else 0
        pin_memory = torch.cuda.is_available()
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            num_workers=n_workers, 
            shuffle=True,
            pin_memory=pin_memory,
            persistent_workers=True
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=args.batch_size, 
            num_workers=n_workers,
            pin_memory=pin_memory,
            persistent_workers=True
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=args.batch_size, 
            num_workers=n_workers,
            pin_memory=pin_memory,
            persistent_workers=True
        )
        
        datainfo = {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader,
            'feature_num': feature_num,
            'data_num': [len(train_dataset), len(val_dataset), len(test_dataset)],
            'device': setup_device(),
            'dataset_name': args.dataset,
            'run_id': run_id
        }
        
        print(f"📈 数据统计:")
        print(f"   - 训练集: {len(train_dataset):,} 样本")
        print(f"   - 验证集: {len(val_dataset):,} 样本") 
        print(f"   - 测试集: {len(test_dataset):,} 样本")
        print(f"   - 特征数量: {feature_num}")
        print(f"   - 批次大小: {args.batch_size}")
        
        # 投影配置信息
        if args.use_msi_features:
            print(f"🎯 投影配置:")
            print(f"   - 投影维度: {args.msi_projection_dim}")
            print(f"   - 使用投影: {args.use_msi_projection}")
            print(f"   - 特征选择阈值: {args.feature_selection_threshold}")
            print(f"   - PCA投影: {args.use_pca_projection}")
            print(f"   - 梯度累积: {args.accumulation_steps} 步")
        
        # 运行训练
        set_seed(args.random_seed + run_id)
        train_results = train(args, datainfo, run_id)
        
        end_time = time.time()
        duration = (end_time - start_time) / 60
        
        print(f"✅ 第 {run_id+1} 次运行完成, 耗时: {duration:.2f} 分钟")
        
        return True
        
    except Exception as e:
        end_time = time.time()
        duration = (end_time - start_time) / 60
        print(f"❌ 第 {run_id+1} 次运行失败: {e}")
        print(f"⏱️  运行时间: {duration:.2f} 分钟")
        import traceback
        traceback.print_exc()
        return False

def main():
    args = parser.parse_args()
    
    # 根据数据集调整配置
    dataset_config = get_dataset_config(args.dataset)
    if args.batch_size == 256:  # 如果使用默认值，则自动调整
        args.batch_size = dataset_config['batch_size']
    
    # 自动设置推荐的投影维度
    if args.msi_projection_dim == 256:  # 如果使用默认值
        recommended_dim = dataset_config.get('recommended_projection_dim', 128)
        args.msi_projection_dim = recommended_dim
    
    # 关键修改：确保dim和msi_projection_dim的一致性
    if args.use_msi_features:
        # 如果使用MSI特征，确保嵌入维度足够大以容纳投影特征
        if args.dim < 64:  # 如果嵌入维度太小
            args.dim = 128  # 增加到合适的值
            print(f"💡 自动调整嵌入维度到: {args.dim} 以支持MSI特征融合")
    
    separator = '=' * 70
    print(f"{separator}")
    print(f"🎯 KARADTI 实验配置")
    print(f"{separator}")
    print(f"📁 数据集: {args.dataset}")
    print(f"🔀 分割方式: s{args.split + 1}")
    print(f"🔄 运行次数: {args.num_runs}")
    print(f"🧠 内部模型: {args.inner_model} ({'默认' if args.inner_model == 0 else 'GCN' if args.inner_model == 1 else 'GAT'})")
    print(f"🌐 跨域模型: {args.cross_model} ({'默认' if args.cross_model == 0 else 'GCN' if args.cross_model == 1 else 'GAT'})")
    print(f"🚀 动态路由: {args.use_dynamic_router}")
    print(f"🎭 双注意力: {args.use_dual_attention}")
    print(f"📊 MSI特征: {args.use_msi_features}")
    
    # 投影配置显示
    if args.use_msi_features:
        print(f"🎯 投影维度: {args.msi_projection_dim}")
        print(f"🔧 使用投影: {args.use_msi_projection}")
        print(f"📏 特征选择: {args.feature_selection_threshold}")
        print(f"📐 PCA投影: {args.use_pca_projection}")
        print(f"📦 嵌入维度: {args.dim}")
    
    print(f"📦 批次大小: {args.batch_size}")
    print(f"📈 训练轮数: {args.n_epoch}")
    print(f"🎲 随机种子: {args.random_seed}")
    print(f"{separator}")
    
    # 验证配置
    warnings = validate_projection_config(args)
    if warnings:
        print("📋 配置警告:")
        for warning in warnings:
            print(f"   {warning}")
        print(f"{separator}")
    
    # 设置设备
    device = setup_device()
    
    # 运行多次实验
    successful_runs = 0
    for run_id in range(args.num_runs):
        success = run_single_experiment(args, run_id)
        if success:
            successful_runs += 1
        
        # 批次间暂停，让GPU冷却
        if run_id < args.num_runs - 1 and torch.cuda.is_available():
            print("⏸️  批次间暂停 5 秒...")
            time.sleep(5)
            torch.cuda.empty_cache()
    
    print(f"\n{separator}")
    print(f"🎉 所有实验完成!")
    print(f"✅ 成功运行: {successful_runs}/{args.num_runs}")
    
    # 最终配置总结
    if args.use_msi_features:
        print(f"🎯 使用的投影配置:")
        print(f"   - 投影维度: {args.msi_projection_dim}")
        print(f"   - 特征选择: {args.feature_selection_threshold}")
    
    print(f"{separator}")

if __name__ == '__main__':
    main()