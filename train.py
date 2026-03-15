import os.path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, log_loss, precision_score, recall_score, accuracy_score, \
    mean_squared_error, average_precision_score, r2_score
import pickle
import time
import math
import pandas as pd
from functools import wraps
import dill
from lifelines.utils import concordance_index
from sklearn.metrics import precision_recall_curve, auc

# 关键：导入 KARADTI 模型
from model import KARADTI

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss


crit = FocalLoss(alpha=2, gamma=2)  # alpha>1给正样本更多权重

def generate_result(raw, cur_step):
    val_auc, val_logloss, val_mse, val_ci, val_aupr, val_recall, val_precision, val_accuracy, val_r2 = raw
    return f'epoch:{cur_step},val_auc:{val_auc}, val_logloss:{val_logloss}, val_mse:{val_mse}, val_ci:{val_ci}, val_aupr:{val_aupr}, ' \
           f'val_recall:{val_recall}, val_precision:{val_precision}, val_accuracy:{val_accuracy}, val_r2:{val_r2}'


class EarlyStopping:
    def __init__(self, patience=15, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.best_result = ''

    def __call__(self, val_loss, model, path, indicatior, cur_step):
        if self.verbose:
            print("val_loss={}".format(val_loss))
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.indicator = generate_result(indicatior, cur_step)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0
            self.indicator = generate_result(indicatior, cur_step)

    def save_checkpoint(self, val_loss, model, path):
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        if self.verbose:
            print(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'model_checkpoint.pth')
        self.val_loss_min = val_loss


def extract_drug_target_ids(data):
    """从图数据中提取药物和靶点ID - 改进版本"""
    try:
        # 方法1: 检查是否有直接的drug_id和target_id属性
        if hasattr(data, 'drug_id') and hasattr(data, 'target_id'):
            drug_ids = data.drug_id
            target_ids = data.target_id
            
            # 确保是一维张量
            if drug_ids.dim() > 1:
                drug_ids = drug_ids.squeeze()
            if target_ids.dim() > 1:
                target_ids = target_ids.squeeze()
                
            return drug_ids.cpu().numpy(), target_ids.cpu().numpy()
        
        # 方法2: 从批次信息中提取
        elif hasattr(data, 'batch'):
            # 对于批次数据，需要更复杂的处理
            batch_size = data.batch.max().item() + 1 if hasattr(data, 'batch') else 1
            
            # 尝试从x中提取（假设第一个节点是药物，最后一个节点是靶点）
            drug_ids = []
            target_ids = []
            
            for i in range(batch_size):
                # 获取属于当前批次的节点
                if hasattr(data, 'batch'):
                    mask = data.batch == i
                    batch_nodes = data.x[mask]
                else:
                    batch_nodes = data.x
                
                if len(batch_nodes) >= 2:
                    # 假设第一个节点是药物，最后一个节点是靶点
                    drug_ids.append(batch_nodes[0].item())
                    target_ids.append(batch_nodes[-1].item())
                else:
                    drug_ids.append(0)
                    target_ids.append(0)
            
            return np.array(drug_ids), np.array(target_ids)
        
        # 方法3: 默认返回None
        else:
            return None, None
            
    except Exception as e:
        print(f"提取药物/靶点ID失败: {e}")
        return None, None


class StableBCELoss(nn.Module):
    """稳定的BCE损失，防止数值不稳定"""
    def __init__(self, label_smoothing=0.0):
        super().__init__()
        self.label_smoothing = label_smoothing
        
    def forward(self, input, target):
        # 标签平滑
        if self.label_smoothing > 0:
            target = target * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        
        # 使用clip防止数值溢出
        input = torch.clamp(input, 1e-7, 1 - 1e-7)
        loss = - (target * torch.log(input) + (1 - target) * torch.log(1 - input))
        return loss.mean()


def check_model_initialization(model):
    """检查模型参数初始化状态"""
    print("=== 模型参数初始化检查 ===")
    total_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
        
        # 检查参数值范围
        if param.requires_grad and 'weight' in name:
            print(f"{name}: shape={param.shape}, mean={param.data.mean().item():.6f}, "
                  f"std={param.data.std().item():.6f}")
    
    print(f"总参数: {total_params:,}, 可训练参数: {trainable_params:,}")
    print("=" * 50)


def train(args, data_info, t):
    train_loader = data_info['train']
    val_loader = data_info['val']
    test_loader = data_info['test']
    feature_num = data_info['feature_num']
    train_num, val_num, test_num = data_info['data_num']
    early_stop = EarlyStopping(verbose=getattr(args, 'verbose', False))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 投影维度配置
    msi_projection_dim = getattr(args, 'msi_projection_dim', 256)
    use_msi_projection = getattr(args, 'use_msi_projection', True)
    verbose = getattr(args, 'verbose', True)
    
    if verbose:
        print(f"=== 训练配置 ===")
        print(f"投影维度: {msi_projection_dim}")
        print(f"使用投影: {use_msi_projection}")
        print(f"MSI特征: {getattr(args, 'use_msi_features', True)}")
        print(f"动态路由: {getattr(args, 'use_dynamic_router', False)}")
        print(f"双注意力: {getattr(args, 'use_dual_attention', False)}")
        print(f"学习率: {args.lr}")
        print(f"批次大小: {args.batch_size}")
        print(f"L2权重: {args.l2_weight}")
        
        # 快速调试
        print("=== 数据调试 ===")
        for i, data in enumerate(train_loader):
            data = data.to(device)
            print(f"数据形状: x={data.x.shape}, y={data.y.shape}")
            print(f"y值: min={data.y.min().item():.3f}, max={data.y.max().item():.3f}, mean={data.y.mean().item():.3f}")
            
            # 检查MSI特征
            if hasattr(data, 'x_msi'):
                print(f"MSI特征形状: {data.x_msi.shape}")
                print(f"MSI特征范围: [{data.x_msi.min().item():.3f}, {data.x_msi.max().item():.3f}]")
            
            # 检查标签分布
            labels = data.y.cpu().numpy()
            pos_ratio = np.mean(labels)
            print(f"标签分布: 正样本比例={pos_ratio:.3f} ({np.sum(labels==1)}/{len(labels)})")
            break
    
    # 初始化模型
    dataset_path = f"./{args.dataset}"
    model = KARADTI(args, feature_num, device, dataset_path)
    model = model.to(device)

    # 检查模型初始化
    if verbose:
        check_model_initialization(model)
    
    # 优化器配置 - 添加更好的参数设置
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=args.l2_weight
    )
    
    # 添加学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',  # 监控loss最小化
        factor=0.5,  # 学习率乘以0.5
        patience=5,  # 5个epoch没改善就降低学习率
        verbose=True,
        min_lr=1e-6  # 最小学习率
    )
    
    # 使用稳定的损失函数
    crit = StableBCELoss(label_smoothing=0.05)  # 添加标签平滑防止过拟合

    # 梯度累积步数
    accumulation_steps = getattr(args, 'accumulation_steps', 2)
    
    if verbose:
        print('开始训练...')
    
    if not os.path.exists(f'./train_acc_{args.dataset}'):
        os.mkdir(f'./train_acc_{args.dataset}')
    
    # 记录是否使用MSI特征
    use_msi_features = getattr(args, 'use_msi_features', True)
    
    # 训练统计
    projection_stats_history = []
    
    with open(f'./train_acc_{args.dataset}/cross={args.cross_model}, inner={args.inner_model},split={args.split}, time={t}.txt', 'w') as f:
        for step in range(args.n_epoch):
            if early_stop.early_stop:
                if verbose:
                    print('超出耐心值，停止训练，模型已保存。\n')
                    print(f'最佳结果: {early_stop.indicator} \n\n')
                f.write(str(early_stop.indicator))
                break
            
            # 训练阶段
            loss_all = 0
            model.train()
            optimizer.zero_grad()

            gradient_norms = []
            
            for i, data in enumerate(train_loader):
                data = data.to(device)
                
                # 提取药物和靶点ID（用于MSI特征查找）
                drug_ids, target_ids = extract_drug_target_ids(data)
                # ============ 新增：KAN训练调试 ============
                if step == 0 and i == 0 and getattr(args, 'verbose', True):
                    print(f"\n[KAN训练调试] 第 {step} 轮, 批次 {i}")
                    print(f"  输入数据维度: x={data.x.shape}")
                    print(f"  批次信息: batch={data.batch.shape if hasattr(data, 'batch') else 'None'}")
                # ===========================================

                # 前向传播
                try:
                    if use_msi_features and drug_ids is not None and target_ids is not None:
                        output = model(data, drug_ids=drug_ids, protein_ids=target_ids)
                    else:
                        output = model(data)

                    # ============ 新增：KAN输出调试 ============
                    if step == 0 and i == 0 and getattr(args, 'verbose', True):
                        print(f"  KAN模型输出维度: {output.shape}")
                        print(f"  KAN模型输出值范围: [{output.min().item():.4f}, {output.max().item():.4f}]")

                        # 检查KAN预测器状态
                        if hasattr(model, 'kan_predictor'):
                            print(f"  KAN预测器参数统计:")
                            for name, param in model.kan_predictor.named_parameters():
                                if param.requires_grad:
                                    print(f"    {name}: shape={param.shape}, mean={param.data.mean().item():.6f}")
                    # ===========================================

                except Exception as e:
                    print(f"❌ 错误: {e}")
                    continue
                # 前向传播 - 添加详细的错误处理
                try:
                    if use_msi_features and drug_ids is not None and target_ids is not None:
                        output = model(data, drug_ids=drug_ids, protein_ids=target_ids)
                    else:
                        output = model(data)
                    
                    # 检查模型输出是否为None
                    if output is None:
                        print(f"❌ 第 {step} 轮, 批次 {i}: 模型返回 None，跳过此批次")
                        continue
                        
                except Exception as e:
                    print(f"❌ 第 {step} 轮, 批次 {i}: 模型前向传播错误: {e}")
                    print(f"   数据信息: x.shape={data.x.shape if data.x is not None else 'None'}")
                    print(f"   边索引: {data.edge_index.shape if hasattr(data, 'edge_index') and data.edge_index is not None else 'None'}")
                    continue
                
                label = data.y.to(device)
                
                # 检查输出和标签的形状
                if output.shape != label.shape:
                    try:
                        output = output.view_as(label)
                    except:
                        print(f"⚠️ 第 {step} 轮, 批次 {i}: 无法调整输出形状, output.shape={output.shape}, label.shape={label.shape}")
                        continue
                
                # 检查输出范围（添加sigmoid激活确保在[0,1]范围）
                output = torch.sigmoid(output)  # 确保输出在0-1之间用于二元分类
                
                if torch.isnan(output).any() or torch.isinf(output).any():
                    print(f"⚠️ 警告: 第 {step} 轮, 批次 {i} 输出包含NaN或Inf")
                    # 处理无效值
                    output = torch.clamp(output, 1e-7, 1-1e-7)
                    output[torch.isnan(output)] = 0.5
                    output[torch.isinf(output)] = 0.5
                
                # 计算损失
                try:
                    baseloss = crit(output, label)
                    
                    # 检查损失是否为NaN
                    if torch.isnan(baseloss) or torch.isinf(baseloss):
                        print(f"⚠️ 警告: 第 {step} 轮, 批次 {i} 损失为NaN或Inf")
                        # 使用一个较小的默认损失值
                        baseloss = torch.tensor(0.1, device=device, requires_grad=True)
                        
                except Exception as e:
                    print(f"❌ 第 {step} 轮, 批次 {i}: 计算损失失败: {e}")
                    # 使用默认损失值
                    baseloss = torch.tensor(0.1, device=device, requires_grad=True)
                
                loss = baseloss / accumulation_steps
                loss_all += data.num_graphs * loss.item() * accumulation_steps
                
                # 反向传播
                try:
                    loss.backward()
                except Exception as e:
                    print(f"❌ 第 {step} 轮, 批次 {i}: 反向传播错误: {e}")
                    optimizer.zero_grad()
                    continue
                
                # 每 accumulation_steps 个批次更新一次
                if (i + 1) % accumulation_steps == 0:
                    # 梯度监控
                    total_norm = 0
                    grad_found = False
                    for p in model.parameters():
                        if p.grad is not None:
                            grad_found = True
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    
                    if grad_found:
                        total_norm = total_norm ** 0.5
                        gradient_norms.append(total_norm)
                        
                        # 梯度裁剪
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    optimizer.zero_grad()
            
            # 处理剩余的梯度
            if (i + 1) % accumulation_steps != 0:
                try:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                except Exception as e:
                    print(f"❌ 第 {step} 轮: 梯度更新错误: {e}")
                    optimizer.zero_grad()

            cur_loss = loss_all / train_num if train_num > 0 else 0.0
            
            # 打印梯度信息
            if gradient_norms and step % 5 == 0:
                avg_grad_norm = np.mean(gradient_norms)
                max_grad_norm = np.max(gradient_norms)
                min_grad_norm = np.min(gradient_norms)
                print(f"Epoch {step}: 平均梯度范数={avg_grad_norm:.4f}, "
                      f"最大={max_grad_norm:.4f}, 最小={min_grad_norm:.4f}")

            # 评估阶段
            try:
                val_auc, val_logloss, val_mse, val_ci, val_aupr, val_recall, val_precision, val_accuracy, val_r2 = evaluate(
                    model, val_loader, device, use_msi_features)
                test_auc, test_logloss, test_mse, test_ci, test_aupr, test_recall, test_precision, test_accuracy, test_r2 = evaluate(
                    model, test_loader, device, use_msi_features)
            except Exception as e:
                print(f"❌ 第 {step} 轮: 评估错误: {e}")
                # 使用默认值继续训练
                val_auc = test_auc = 0.5
                val_logloss = test_logloss = 0.69  # log(2)
                val_mse = test_mse = 0.25
                val_ci = test_ci = 0.5
                val_aupr = test_aupr = 0.5
                val_recall = test_recall = 0.5
                val_precision = test_precision = 0.5
                val_accuracy = test_accuracy = 0.5
                val_r2 = test_r2 = 0.0
            
            # 更新学习率
            scheduler.step(val_logloss)  # 根据验证集logloss调整学习率
            current_lr = optimizer.param_groups[0]['lr']
            
            # 记录投影统计
            if hasattr(model, 'projection_stats'):
                projection_stats_history.append(model.projection_stats.copy())
                if verbose and step % 10 == 0:
                    print(f"[投影统计] 总融合: {model.projection_stats['total_fusions']}, "
                          f"药物投影使用: {model.projection_stats['drug_projection_used']}, "
                          f"靶点投影使用: {model.projection_stats['protein_projection_used']}")
            
            if step >= 10:
                early_stop(val_logloss, model, 
                          f'./model_acc_{args.dataset}/cross{args.cross_model}_inner{args.inner_model}_split{args.split}_proj{msi_projection_dim}',
                          (val_auc, val_logloss, val_mse, val_ci, val_aupr, val_recall, val_precision, val_accuracy, val_r2), step)
            
            # 输出训练进度
            print(
                'Epoch: {:03d}, Loss: {:.5f}, AUC: {:.5f}/{:.5f}, Logloss: {:.5f}/{:.5f}, MSE:{:.5f}/{:.5f}, CI:{:.5f}/{:.5f}, AUPR:{:.5f}/{:.5f}, Recall: {:.5f}/{:.5f}, Precision: {:.5f}/{:.5f}, Accuracy: {:.5f}/{:.5f}, R2: {:.5f}/{:.5f}, LR: {:.6f}'.
                    format(step, cur_loss, val_auc, test_auc, val_logloss, test_logloss, val_mse, test_mse, val_ci,
                           test_ci, val_aupr, test_aupr, val_recall, test_recall, val_precision, test_precision, 
                           val_accuracy, test_accuracy, val_r2, test_r2, current_lr))
            f.write(
                'Epoch: {:03d}, Loss: {:.5f}, AUC: {:.5f}/{:.5f}, Logloss: {:.5f}/{:.5f}, MSE:{:.5f}/{:.5f}, CI:{:.5f}/{:.5f}, AUPR:{:.5f}/{:.5f}, Recall: {:.5f}/{:.5f}, Precision: {:.5f}/{:.5f}, Accuracy: {:.5f}/{:.5f}, R2: {:.5f}/{:.5f}, LR: {:.6f}\n'.
                format(step, cur_loss, val_auc, test_auc, val_logloss, test_logloss, val_mse, test_mse, val_ci, test_ci,
                       val_aupr, test_aupr, val_recall, test_recall, val_precision, test_precision, 
                       val_accuracy, test_accuracy, val_r2, test_r2, current_lr))
        
        # 保存投影统计
        if projection_stats_history and verbose:
            stats_df = pd.DataFrame(projection_stats_history)
            stats_df.to_csv(f'./train_acc_{args.dataset}/projection_stats_split{args.split}.csv', index=False)
            print(f"投影统计已保存到: ./train_acc_{args.dataset}/projection_stats_split{args.split}.csv")


def evaluate(model, data_loader, device, use_msi_features=True):
    """评估模型"""
    model.eval()
    predictions = []
    labels = []
    
    batch_count = 0
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            
            # 提取药物和靶点ID
            drug_ids, target_ids = extract_drug_target_ids(data)
            
            # 根据是否使用MSI特征选择不同的调用方式
            try:
                if use_msi_features and drug_ids is not None and target_ids is not None:
                    output = model(data, drug_ids=drug_ids, protein_ids=target_ids)
                else:
                    output = model(data)
                    
                # 检查输出是否为None
                if output is None:
                    print(f"⚠️ 评估批次 {batch_count}: 模型返回 None，跳过此批次")
                    batch_count += 1
                    continue
                    
            except Exception as e:
                print(f"❌ 评估批次 {batch_count}: 模型前向传播错误: {e}")
                batch_count += 1
                continue
            
            # 应用sigmoid确保输出在0-1之间
            output = torch.sigmoid(output)
            
            # 检查输出范围
            if torch.isnan(output).any() or torch.isinf(output).any():
                print(f"⚠️ 警告: 评估批次 {batch_count} 输出包含NaN或Inf")
                output = torch.clamp(output, 1e-7, 1-1e-7)
                output[torch.isnan(output)] = 0.5
                output[torch.isinf(output)] = 0.5
            
            # 确保输出和标签都是一维的
            pred = output.cpu().numpy().flatten()
            label = data.y.cpu().numpy().flatten()
            
            predictions.extend(pred)
            labels.extend(label)
            batch_count += 1
    
    if len(predictions) == 0:
        print("❌ 警告: 没有有效的预测结果")
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    
    # 转换为numpy数组并确保是一维的
    predictions = np.array(predictions).flatten()
    labels = np.array(labels).flatten()
    
    # 检查预测值范围
    if np.isnan(predictions).any() or np.isinf(predictions).any():
        print("⚠️ 警告: 预测值包含NaN或Inf")
        predictions = np.clip(predictions, 1e-7, 1-1e-7)
        predictions[np.isnan(predictions)] = 0.5
    
    # 计算各种指标
    try:
        auc_score = roc_auc_score(labels, predictions)
    except:
        auc_score = 0.5
    
    try:
        # 防止logloss计算中的数值问题
        predictions_clipped = np.clip(predictions, 1e-7, 1-1e-7)
        logloss_score = log_loss(labels, predictions_clipped)
    except:
        logloss_score = 0.69  # log(2) 作为默认值
        
    mse_score = mean_squared_error(labels, predictions)
    
    # 计算CI（一致性指数）
    try:
        ci_score = concordance_index(labels, predictions)
    except:
        ci_score = 0.5
    
    # 计算AUPR
    try:
        precision_vals, recall_vals, _ = precision_recall_curve(labels, predictions)
        aupr_score = auc(recall_vals, precision_vals)
    except:
        aupr_score = 0.5
    
    # 计算二分类指标（将预测值转换为0/1）
    binary_predictions = (predictions > 0.5).astype(int)
    try:
        recall_score_val = recall_score(labels, binary_predictions, zero_division=0)
    except:
        recall_score_val = 0.5
        
    try:
        precision_score_val = precision_score(labels, binary_predictions, zero_division=0)
    except:
        precision_score_val = 0.5
        
    try:
        accuracy_score_val = accuracy_score(labels, binary_predictions)
    except:
        accuracy_score_val = 0.5
    
    # 计算R²
    try:
        r2_score_val = r2_score(labels, predictions)
    except:
        r2_score_val = 0.0
    
    return auc_score, logloss_score, mse_score, ci_score, aupr_score, recall_score_val, precision_score_val, accuracy_score_val, r2_score_val


def test_model(args, data_info, t):
    """测试训练好的模型"""
    test_loader = data_info['test']
    feature_num = data_info['feature_num']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 初始化模型
    dataset_path = f"./{args.dataset}"
    model = KARADTI(args, feature_num, device, dataset_path)
    model = model.to(device)
    
    # 加载训练好的模型
    model_path = f'./model_acc_{args.dataset}/cross{args.cross_model}_inner{args.inner_model}_split{args.split}_proj{getattr(args, "msi_projection_dim", 256)}/model_checkpoint.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"加载训练好的模型: {model_path}")
    else:
        # 尝试加载旧格式的模型路径
        old_model_path = f'./model_acc_{args.dataset}/corss {args.cross_model} inner {args.inner_model} on split{args.split}/model_checkpoint.pth'
        if os.path.exists(old_model_path):
            model.load_state_dict(torch.load(old_model_path))
            print(f"加载旧格式模型: {old_model_path}")
        else:
            print(f"未找到训练好的模型: {model_path}")
            return
    
    # 评估测试集
    use_msi_features = getattr(args, 'use_msi_features', True)
    test_auc, test_logloss, test_mse, test_ci, test_aupr, test_recall, test_precision, test_accuracy, test_r2 = evaluate(
        model, test_loader, device, use_msi_features)
    
    print(f"测试集结果:")
    print(f"AUC: {test_auc:.5f}")
    print(f"LogLoss: {test_logloss:.5f}")
    print(f"MSE: {test_mse:.5f}")
    print(f"CI: {test_ci:.5f}")
    print(f"AUPR: {test_aupr:.5f}")
    print(f"Recall: {test_recall:.5f}")
    print(f"Precision: {test_precision:.5f}")
    print(f"Accuracy: {test_accuracy:.5f}")
    print(f"R2: {test_r2:.5f}")
    
    # 保存测试结果
    if not os.path.exists('./test_results'):
        os.mkdir('./test_results')
    
    result_file = f'./test_results/test_results_split{args.split}_proj{getattr(args, "msi_projection_dim", 256)}.txt'
    with open(result_file, 'w') as f:
        f.write(f"投影配置: 维度={getattr(args, 'msi_projection_dim', 256)}, 使用投影={getattr(args, 'use_msi_projection', True)}\n")
        f.write(f"AUC: {test_auc:.5f}\n")
        f.write(f"LogLoss: {test_logloss:.5f}\n")
        f.write(f"MSE: {test_mse:.5f}\n")
        f.write(f"CI: {test_ci:.5f}\n")
        f.write(f"AUPR: {test_aupr:.5f}\n")
        f.write(f"Recall: {test_recall:.5f}\n")
        f.write(f"Precision: {test_precision:.5f}\n")
        f.write(f"Accuracy: {test_accuracy:.5f}\n")
        f.write(f"R2: {test_r2:.5f}\n")
    
    print(f"测试结果已保存到: {result_file}")


def debug_model_output(args, data_info):
    """调试模型输出"""
    train_loader = data_info['train']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_num = data_info['feature_num']
    
    # 初始化模型
    dataset_path = f"./{args.dataset}"
    model = KARADTI(args, feature_num, device, dataset_path)
    model = model.to(device)
    model.eval()
    
    print("=== 模型输出调试 ===")
    
    # 检查一个批次
    for i, data in enumerate(train_loader):
        data = data.to(device)
        
        # 提取药物和靶点ID
        drug_ids, target_ids = extract_drug_target_ids(data)
        
        print(f"批次 {i}:")
        print(f"  输入数据: x={data.x.shape}, y={data.y.shape}")
        print(f"  药物ID: {drug_ids}")
        print(f"  靶点ID: {target_ids}")
        
        # 前向传播
        with torch.no_grad():
            try:
                output = model(data, drug_ids=drug_ids, protein_ids=target_ids)
                if output is None:
                    print(f"  输出: None")
                else:
                    print(f"  输出形状: {output.shape}")
                    print(f"  输出值范围: [{output.min().item():.4f}, {output.max().item():.4f}]")
                    print(f"  输出均值: {output.mean().item():.4f}")
                    output_sigmoid = torch.sigmoid(output)
                    print(f"  Sigmoid后范围: [{output_sigmoid.min().item():.4f}, {output_sigmoid.max().item():.4f}]")
            except Exception as e:
                print(f"  前向传播错误: {e}")
        
        if i >= 2:  # 只检查前3个批次
            break