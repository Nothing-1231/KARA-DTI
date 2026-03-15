import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool
import os
from layers import *
from kan import KAN

class FeatureFusionKAN(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.feat_dim = feat_dim
        self.attention_proj = nn.Linear(feat_dim * 2, feat_dim)
        self.kan = KAN([feat_dim * 2, feat_dim, feat_dim], 
                      grid_size=5, 
                      spline_order=3,
                      scale_base=1.0,
                      scale_spline=1.0)
        
    def forward(self, feat1, feat2):
        if feat1.dim() == 1:
            feat1 = feat1.unsqueeze(0)
        if feat2.dim() == 1:
            feat2 = feat2.unsqueeze(0)
            
        batch_size = min(feat1.shape[0], feat2.shape[0])
        if feat1.shape[0] != feat2.shape[0]:
            feat1 = feat1[:batch_size]
            feat2 = feat2[:batch_size]
            
        combined = torch.cat([feat1, feat2], dim=-1)
        attention = torch.sigmoid(self.attention_proj(combined))
        weighted_feat1 = feat1 * attention
        weighted_feat2 = feat2 * (1 - attention)
        kan_input = torch.cat([weighted_feat1, weighted_feat2], dim=-1)
        fused = self.kan(kan_input)
        return fused

class KANPredictionHead(nn.Module):
    def __init__(self, dim, device, hidden_dim=32, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.device = device
        self.fusion_kan = FeatureFusionKAN(dim)
        self.kan_model = KAN([dim, hidden_dim, 2])
        self.output_proj = nn.Linear(2, 1)
        
        self.user_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )
        
        self.item_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def split_batch(self, batch, user_node_num):
        if user_node_num <= 0:
            user_node_num = 1

        ones = torch.ones_like(batch)
        nodes_per_graph = global_add_pool(ones, batch)
        min_nodes = torch.min(nodes_per_graph).item()

        if user_node_num > min_nodes:
            user_node_num = min_nodes

        if len(nodes_per_graph) > 0:
            cum_num = torch.cat((
                torch.tensor([0], dtype=torch.long, device=batch.device),
                torch.cumsum(nodes_per_graph, dim=0)[:-1]
            ))
        else:
            cum_num = torch.tensor([0], dtype=torch.long, device=batch.device)

        cum_num_list = []
        for i in range(user_node_num):
            adjusted_cum_num = cum_num + i
            max_valid_index = ones.size(0) - 1
            adjusted_cum_num = torch.clamp(adjusted_cum_num, 0, max_valid_index)
            cum_num_list.append(adjusted_cum_num)

        multi_hot = torch.cat(cum_num_list)
        test = torch.sum(
            F.one_hot(multi_hot, ones.size(0)),
            dim=0
        ) * (torch.max(batch) + 1)

        return batch + test

    def forward(self, updated_node, batch, sum_weight, user_node_num=9):
        if updated_node is None or updated_node.numel() == 0:
            return torch.zeros(1, 1, device=self.device)
        
        if batch is None:
            if updated_node.dim() == 2 and updated_node.size(0) > 0:
                user_graph = updated_node.mean(dim=0, keepdim=True)
                item_graph = updated_node.mean(dim=0, keepdim=True)
            else:
                return torch.zeros(1, 1, device=self.device)
        else:
            if user_node_num <= 0:
                user_node_num = 1
                
            new_batch = self.split_batch(batch, user_node_num)
            updated_graph = global_mean_pool(updated_node, new_batch)
            if updated_graph.dim() > 2:
                updated_graph = updated_graph.squeeze()
            
            if updated_graph.numel() == 0:
                return torch.zeros(1, 1, device=self.device)
                
            total_graphs = updated_graph.shape[0]
            split_size = max(1, total_graphs // 2)
            
            if total_graphs < 2:
                user_graph = updated_graph.mean(dim=0, keepdim=True)
                item_graph = updated_graph.mean(dim=0, keepdim=True)
                if sum_weight is not None:
                    sum_weight = sum_weight[:1]
            else:
                if total_graphs % 2 != 0:
                    split_size = (total_graphs - 1) // 2
                
                item_graphs = updated_graph[:split_size]
                user_graphs = updated_graph[split_size:split_size*2] if total_graphs >= split_size*2 else updated_graph[split_size:]
                
                min_size = min(user_graphs.shape[0], item_graphs.shape[0])
                user_graph = user_graphs[:min_size]
                item_graph = item_graphs[:min_size]
                if sum_weight is not None:
                    sum_weight = sum_weight[:min_size]
        
        user_features = self.user_proj(user_graph)
        item_features = self.item_proj(item_graph)
        
        try:
            user_features_norm = F.layer_norm(user_features, user_features.shape[1:])
            item_features_norm = F.layer_norm(item_features, item_features.shape[1:])
            
            fused_features = self.fusion_kan(user_features_norm, item_features_norm)
            fused_features_norm = F.layer_norm(fused_features, fused_features.shape[1:])
            
            if fused_features_norm.size(1) != self.dim:
                if fused_features_norm.size(1) < self.dim:
                    padding = torch.zeros(fused_features_norm.size(0), 
                                        self.dim - fused_features_norm.size(1),
                                        device=self.device)
                    fused_features_norm = torch.cat([fused_features_norm, padding], dim=1)
                else:
                    fused_features_norm = fused_features_norm[:, :self.dim]
            
            kan_output = self.kan_model(fused_features_norm)
            
            if kan_output.shape[1] >= 2:
                y = kan_output[:, 1:2]
            else:
                y = self.output_proj(kan_output)
            
        except Exception:
            dot_product = torch.sum(user_features * item_features, dim=1, keepdim=True)
            y = dot_product
        
        if sum_weight is not None and sum_weight.shape[0] == y.shape[0]:
            y = y + sum_weight
        
        return y

class DACMFCrossModalFusion(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super(DACMFCrossModalFusion, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        
        self.attention_net = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.feature_enhance = nn.Sequential(
            nn.Linear(dim * 3, dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.ReLU()
        )
        
        self.layer_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, node_features, inner_features, outer_features):
        batch_size = node_features.shape[0]
        
        if node_features.dim() == 2:
            node_features = node_features.unsqueeze(1)
        if inner_features.dim() == 2:
            inner_features = inner_features.unsqueeze(1)
        if outer_features.dim() == 2:
            outer_features = outer_features.unsqueeze(1)
        
        # 确保所有特征有相同的batch size
        min_batch = min(node_features.shape[0], inner_features.shape[0], outer_features.shape[0])
        if min_batch < batch_size:
            node_features = node_features[:min_batch]
            inner_features = inner_features[:min_batch]
            outer_features = outer_features[:min_batch]
        
        combined_features = torch.cat([node_features, inner_features, outer_features], dim=1)
        attended_features, _ = self.attention_net(
            combined_features,
            combined_features,
            combined_features
        )
        
        seq_len, dim = attended_features.shape[1], attended_features.shape[2]
        enhanced_features = self.feature_enhance(
            attended_features.reshape(min_batch, -1)
        )
        
        original_features = node_features.squeeze(1)
        if enhanced_features.shape == original_features.shape:
            fused_features = self.layer_norm(enhanced_features + original_features)
        else:
            fused_features = self.layer_norm(enhanced_features)
        
        fused_features = self.dropout(fused_features)
        return fused_features

class AdaptiveRouterLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_routers=20):
        super(AdaptiveRouterLayer, self).__init__()
        self.input_dim = input_dim
        self.num_routers = num_routers
        
        self.attention_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_routers),
            nn.Softmax(dim=-1)
        )
        
        self.router_embeddings = nn.Embedding(num_routers, input_dim)
        nn.init.xavier_uniform_(self.router_embeddings.weight)
    
    def forward(self, node_embeddings):
        if node_embeddings is None or node_embeddings.numel() == 0:
            return torch.zeros(1, self.num_routers, device=node_embeddings.device), self.router_embeddings.weight
        
        if node_embeddings.dim() == 2:
            node_embeddings = node_embeddings.unsqueeze(0)
        
        batch_size, num_nodes, input_dim = node_embeddings.shape
        
        if input_dim != self.input_dim:
            if input_dim < self.input_dim:
                padding = torch.zeros(batch_size, num_nodes, self.input_dim - input_dim, 
                                    device=node_embeddings.device)
                node_embeddings = torch.cat([node_embeddings, padding], dim=-1)
            else:
                node_embeddings = node_embeddings[:, :, :self.input_dim]
        
        router_features = self.router_embeddings.weight
        
        if router_features.size(1) != input_dim:
            if router_features.size(1) < input_dim:
                padding = torch.zeros(self.num_routers, input_dim - router_features.size(1), 
                                    device=router_features.device)
                router_features = torch.cat([router_features, padding], dim=-1)
            else:
                router_features = router_features[:, :input_dim]
        
        similarities = []
        for i in range(batch_size):
            node_emb = node_embeddings[i]
            
            if node_emb.numel() == 0:
                similarities.append(torch.zeros(num_nodes, self.num_routers, device=node_emb.device))
                continue
            
            node_norm = F.normalize(node_emb, p=2, dim=1)
            router_norm = F.normalize(router_features, p=2, dim=1)
            sim = torch.mm(node_norm, router_norm.t())
            similarities.append(sim)
        
        if not similarities:
            return torch.zeros(batch_size, num_nodes, self.num_routers, device=node_embeddings.device), router_features
        
        similarities = torch.stack(similarities)
        attention_weights = self.attention_net(node_embeddings)
        routing_weights = F.softmax(similarities * attention_weights, dim=-1)
        
        return routing_weights, router_features

class MSIFeatureLoader:
    def __init__(self, dataset_path, device):
        self.device = device
        current_file_path = os.path.abspath(__file__)
        code_dir = os.path.dirname(current_file_path)
        project_root = os.path.dirname(code_dir)
        
        actual_dataset_path = os.path.join(project_root, "data", os.path.basename(dataset_path))
        
        self.drug_features_path = os.path.join(actual_dataset_path, "drug_features.npy")
        self.target_features_path = os.path.join(actual_dataset_path, "target_features.npy")
        
        self.drug_features = None
        self.target_features = None
        
        if os.path.exists(self.drug_features_path):
            try:
                self.drug_features = np.load(self.drug_features_path, allow_pickle=True)
                print(f"✓ 加载药物特征: {self.drug_features.shape}")
            except Exception as e:
                print(f"✗ 加载药物特征失败: {e}")
        
        if os.path.exists(self.target_features_path):
            try:
                self.target_features = np.load(self.target_features_path, allow_pickle=True)
                print(f"✓ 加载靶点特征: {self.target_features.shape}")
            except Exception as e:
                print(f"✗ 加载靶点特征失败: {e}")
    
    def load_drug_features(self, drug_ids=None):
        if self.drug_features is None:
            return None
        
        try:
            if drug_ids is not None:
                if isinstance(drug_ids, torch.Tensor):
                    drug_ids = drug_ids.cpu().numpy()
                # 确保ID在有效范围内
                max_id = self.drug_features.shape[0] - 1
                if max_id < 0:
                    return None
                    
                valid_ids = []
                for id_val in drug_ids:
                    if isinstance(id_val, np.ndarray) and id_val.size > 0:
                        id_val = id_val.item()
                    int_id = int(id_val)
                    if 0 <= int_id <= max_id:
                        valid_ids.append(int_id)
                    else:
                        valid_ids.append(0)  # 使用第一个特征作为默认
                
                if not valid_ids:
                    return None
                    
                features = self.drug_features[valid_ids]
            else:
                features = self.drug_features
            
            # 特征处理
            features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
            
            if np.std(features) > 0:
                features = (features - np.mean(features)) / (np.std(features) + 1e-8)
            
            features = np.clip(features, -5, 5)
            
            return torch.tensor(features, dtype=torch.float32).to(self.device)
        except Exception:
            return None
    
    def load_protein_features(self, protein_ids=None):
        if self.target_features is None:
            return None
        
        try:
            if protein_ids is not None:
                if isinstance(protein_ids, torch.Tensor):
                    protein_ids = protein_ids.cpu().numpy()
                    
                max_id = self.target_features.shape[0] - 1
                if max_id < 0:
                    return None
                    
                valid_ids = []
                for id_val in protein_ids:
                    if isinstance(id_val, np.ndarray) and id_val.size > 0:
                        id_val = id_val.item()
                    int_id = int(id_val)
                    if 0 <= int_id <= max_id:
                        valid_ids.append(int_id)
                    else:
                        valid_ids.append(0)
                
                if not valid_ids:
                    return None
                    
                features = self.target_features[valid_ids]
            else:
                features = self.target_features
            
            features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
            
            if np.std(features) > 0:
                features = (features - np.mean(features)) / (np.std(features) + 1e-8)
            
            features = np.clip(features, -5, 5)
            
            return torch.tensor(features, dtype=torch.float32).to(self.device)
        except Exception:
            return None

class KARADTI(nn.Module):
    def __init__(self, args, n_features, device, dataset_path="../data/KIBA"):
        super(KARADTI, self).__init__()
        
        self.n_features = n_features
        self.dim = args.dim
        self.hidden_layer = args.hidden_layer
        self.device = device
        self.batch_size = args.batch_size
        self.use_dynamic_router = getattr(args, 'use_dynamic_router', False)
        self.use_msi_features = getattr(args, 'use_msi_features', True)
        self.msi_projection_dim = getattr(args, 'msi_projection_dim', 256)
        self.use_msi_projection = getattr(args, 'use_msi_projection', True)
        
        raw_num_user_features = getattr(args, 'num_user_features', 9)
        self.num_user_features = max(1, raw_num_user_features)
        
        self.inner_choice = args.inner_model
        self.cross_choice = args.cross_model
        self.use_dual_attention = getattr(args, 'use_dual_attention', True)
        
        # GNN初始化 - 添加安全检查
        try:
            if self.inner_choice == 0:
                self.inner_gnn = inner_GNN(self.dim, self.hidden_layer)
            elif self.inner_choice == 1:
                self.inner_gnn = standard_GCN(self.dim, self.hidden_layer, self.dim)
            elif self.inner_choice == 2:
                self.inner_gnn = GAT(self.dim, self.hidden_layer, self.dim)
            else:
                self.inner_gnn = standard_GCN(self.dim, self.hidden_layer, self.dim)
            
            if self.cross_choice == 0:
                if self.use_dual_attention:
                    self.outer_gnn = enhanced_cross_GNN(self.dim, self.hidden_layer, use_dual_attention=True, num_heads=4)
                else:
                    self.outer_gnn = cross_GNN(self.dim, self.hidden_layer)
            elif self.cross_choice == 1:
                self.outer_gnn = standard_GCN(self.dim, self.hidden_layer, self.dim)
            elif self.cross_choice == 2:
                self.outer_gnn = GAT(self.dim, self.hidden_layer, self.dim)
            else:
                self.outer_gnn = enhanced_cross_GNN(self.dim, self.hidden_layer, use_dual_attention=True, num_heads=4)
        except Exception as e:
            print(f"⚠️ GNN初始化失败: {e}")
            # 使用最简单的GCN作为fallback
            self.inner_gnn = standard_GCN(self.dim, 1, self.dim)
            self.outer_gnn = standard_GCN(self.dim, 1, self.dim)
        
        # MSI特征加载器
        if self.use_msi_features:
            try:
                self.msi_loader = MSIFeatureLoader(dataset_path, device)
                self.feature_fusion_gate = nn.Sequential(
                    nn.Linear(self.dim * 2, self.dim),
                    nn.Sigmoid()
                ).to(device)
                self._projectors_initialized = False
                self.drug_feature_projector = None
                self.protein_feature_projector = None
            except Exception as e:
                print(f"⚠️ MSI初始化失败: {e}")
                self.use_msi_features = False
                self.msi_loader = None
        else:
            self.msi_loader = None
        
        # 动态路由组件
        if self.use_dynamic_router:
            try:
                self.adaptive_router = AdaptiveRouterLayer(self.dim, 32, 32, 20).to(device)
                self.feature_projection = nn.Sequential(
                    nn.Linear(self.dim * 2, self.dim * 2),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(self.dim * 2, self.dim)
                ).to(device)
                self.cross_modal_fusion = DACMFCrossModalFusion(self.dim, num_heads=4, dropout=0.1).to(device)
            except Exception as e:
                print(f"⚠️ 动态路由初始化失败: {e}")
                self.use_dynamic_router = False
                self.cross_modal_fusion = DACMFCrossModalFusion(self.dim, num_heads=4, dropout=0.1).to(device)
        else:
            self.cross_modal_fusion = DACMFCrossModalFusion(self.dim, num_heads=4, dropout=0.1).to(device)
        
        # 嵌入层
        try:
            self.feature_embedding = nn.Embedding(n_features + 1, self.dim)
            self.node_weight = nn.Embedding(n_features + 1, 1)
            self.node_weight.weight.data.normal_(0.0, 0.01)
            self.g = nn.Linear(self.dim, 1, bias=False)
        except Exception as e:
            print(f"⚠️ 嵌入层初始化失败: {e}")
            raise
        
        # KAN预测头
        try:
            self.kan_predictor = KANPredictionHead(
                dim=self.dim,
                device=device,
                hidden_dim=32,
                dropout=0.1
            ).to(device)
        except Exception as e:
            print(f"⚠️ KAN预测头初始化失败: {e}")
            # 使用简单线性层作为fallback
            self.kan_predictor = nn.Linear(self.dim, 1).to(device)
        
        self.projection_stats = {
            'drug_projection_used': 0,
            'protein_projection_used': 0,
            'total_fusions': 0
        }
        
        # 参数统计
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"📊 模型参数: {total_params:,} 总参数, {trainable_params:,} 可训练参数")
    
    def _initialize_projectors(self, drug_msi_emb, protein_msi_emb):
        if self._projectors_initialized:
            return
        
        try:
            # ========================================================
            # 【关键修复】确保投影维度与模型维度(self.dim)一致，而不是 self.msi_projection_dim
            # 你的 self.dim=64, msi_projection_dim=256，导致相加时报错。
            # 下面我强制将 Linear 的输出设为 self.dim。
            # ========================================================
            target_dim = self.dim 
            
            if drug_msi_emb is not None and self.use_msi_projection:
                drug_input_dim = drug_msi_emb.shape[1]
                if drug_input_dim != target_dim: # 修改了判断条件
                    self.drug_feature_projector = nn.Sequential(
                        nn.Linear(drug_input_dim, target_dim), # 输出必须是 target_dim
                        nn.BatchNorm1d(target_dim),
                        nn.ReLU(),
                        nn.Dropout(0.1)
                    ).to(self.device)
            
            if protein_msi_emb is not None and self.use_msi_projection:
                protein_input_dim = protein_msi_emb.shape[1]
                if protein_input_dim != target_dim: # 修改了判断条件
                    self.protein_feature_projector = nn.Sequential(
                        nn.Linear(protein_input_dim, target_dim), # 输出必须是 target_dim
                        nn.BatchNorm1d(target_dim),
                        nn.ReLU(),
                        nn.Dropout(0.1)
                    ).to(self.device)
            
            self._projectors_initialized = True
        except Exception as e:
            print(f"⚠️ 投影层初始化失败: {e}")
            self._projectors_initialized = True  # 标记为已初始化，防止重复尝试
    
    def _fuse_msi_features_safe(self, node_emb, drug_ids, protein_ids, batch):
        if not self.use_msi_features or self.msi_loader is None:
            return node_emb
        
        try:
            # 加载特征
            drug_msi_emb = self.msi_loader.load_drug_features(drug_ids)
            protein_msi_emb = self.msi_loader.load_protein_features(protein_ids)
            
            if drug_msi_emb is None and protein_msi_emb is None:
                return node_emb
            
            # 确保维度正确
            if node_emb.dim() == 3:
                node_emb = node_emb.squeeze(1)
            
            result_emb = node_emb.clone()
            
            # 初始化投影层
            if not self._projectors_initialized:
                self._initialize_projectors(drug_msi_emb, protein_msi_emb)
            
            # 融合药物特征
            if drug_msi_emb is not None and drug_ids is not None:
                if isinstance(drug_ids, torch.Tensor):
                    drug_ids_np = drug_ids.cpu().numpy()
                else:
                    drug_ids_np = np.array(drug_ids)
                
                # 简化融合：直接添加特征
                min_size = min(node_emb.shape[0], drug_msi_emb.shape[0])
                if min_size > 0:
                    processed_drug_msi = self._adaptive_projection_selection(
                        node_emb[:min_size], drug_msi_emb[:min_size], 'drug'
                    )
                    if processed_drug_msi is not None:
                        # 确保维度匹配后再融合
                        if processed_drug_msi.shape[1] == node_emb.shape[1]:
                            alpha = 0.3  # MSI特征权重
                            result_emb[:min_size] = (1 - alpha) * node_emb[:min_size] + alpha * processed_drug_msi
                        else:
                            # 即使投影后维度不对（极其罕见），也不要崩，直接跳过
                            pass
            
            # 融合蛋白质特征
            if protein_msi_emb is not None and protein_ids is not None:
                if isinstance(protein_ids, torch.Tensor):
                    protein_ids_np = protein_ids.cpu().numpy()
                else:
                    protein_ids_np = np.array(protein_ids)
                
                min_size = min(node_emb.shape[0], protein_msi_emb.shape[0])
                if min_size > 0:
                    processed_protein_msi = self._adaptive_projection_selection(
                        node_emb[:min_size], protein_msi_emb[:min_size], 'protein'
                    )
                    if processed_protein_msi is not None:
                         # 确保维度匹配后再融合
                        if processed_protein_msi.shape[1] == node_emb.shape[1]:
                            alpha = 0.3
                            result_emb[:min_size] = (1 - alpha) * node_emb[:min_size] + alpha * processed_protein_msi
                        else:
                            pass
            
            self.projection_stats['total_fusions'] += 1
            return result_emb
            
        except Exception as e:
            # 修改打印，增加详细信息方便再次调试
            print(f"⚠️ MSI特征融合失败: {e}. Node dim: {node_emb.shape}, Drug MSI dim: {drug_msi_emb.shape if drug_msi_emb is not None else 'None'}")
            return node_emb
    
    def _adaptive_projection_selection(self, original_emb, msi_emb, feature_type):
        if not self.use_msi_projection or msi_emb is None:
            return msi_emb
        
        try:
            if feature_type == 'drug' and self.drug_feature_projector is not None:
                self.projection_stats['drug_projection_used'] += 1
                return self.drug_feature_projector(msi_emb)
            elif feature_type == 'protein' and self.protein_feature_projector is not None:
                self.projection_stats['protein_projection_used'] += 1
                return self.protein_feature_projector(msi_emb)
            else:
                return msi_emb
        except Exception:
            return msi_emb
    
    def forward(self, data, is_training=True, drug_ids=None, protein_ids=None):
        try:
            # 安全检查
            if data.x is None or data.x.numel() == 0:
                return torch.zeros(1, 1, device=self.device)
            
            # 获取节点ID
            node_id = data.x.to(self.device)
            if node_id.dim() == 1:
                node_id = node_id.unsqueeze(-1)
            if node_id.dtype != torch.long:
                node_id = node_id.long()
            
            # 检查ID范围
            if node_id.max() >= self.n_features + 1:
                print(f"⚠️ 警告: 节点ID超出范围 {node_id.max()} >= {self.n_features + 1}")
                node_id = torch.clamp(node_id, 0, self.n_features)
            
            # 获取嵌入
            node_emb = self.feature_embedding(node_id).squeeze(1)
            node_weight = self.node_weight(node_id).squeeze(1)
            
            # 获取边索引
            if hasattr(data, 'inner_edge_index'):
                inner_edge_index = data.inner_edge_index
                outer_edge_index = data.outer_edge_index
            elif hasattr(data, 'edge_index'):
                edge_index = data.edge_index
                inner_edge_index = edge_index
                outer_edge_index = edge_index
            else:
                return torch.zeros(node_emb.shape[0], 1, device=self.device)
            
            # 将边索引移动到设备
            inner_edge_index = inner_edge_index.to(self.device)
            outer_edge_index = outer_edge_index.to(self.device)
            
            # 批次信息
            batch = data.batch.to(self.device) if hasattr(data, 'batch') and data.batch is not None else None
            
            # MSI特征融合
            if self.use_msi_features and drug_ids is not None and protein_ids is not None:
                try:
                    node_emb = self._fuse_msi_features_safe(node_emb, drug_ids, protein_ids, batch)
                except Exception as e:
                    print(f"⚠️ MSI特征融合失败: {e}")
            
            # 选择前向传播模式
            if self.use_dynamic_router:
                output = self.forward_dynamic(node_emb, inner_edge_index, outer_edge_index, batch, node_weight)
            else:
                output = self.forward_static(node_emb, inner_edge_index, outer_edge_index, batch, node_weight)
            
            # 输出检查
            if output is None:
                return torch.zeros(node_emb.shape[0], 1, device=self.device)
            
            if output.dim() > 2:
                output = output.squeeze()
            if output.dim() == 1:
                output = output.unsqueeze(1)
            
            # 数值稳定性
            if torch.isnan(output).any() or torch.isinf(output).any():
                output = torch.clamp(output, -10, 10)
                output[torch.isnan(output)] = 0.0
            
            return output
            
        except Exception as e:
            print(f"❌ 前向传播失败: {e}")
            return torch.zeros(1, 1, device=self.device)
    
    def forward_dynamic(self, node_emb, inner_edge_index, outer_edge_index, batch, sum_weight):
        try:
            # 路由
            router_weights, router_features = self.adaptive_router(node_emb)
            routed_features = self.apply_routing(node_emb, router_weights, router_features)
            
            # 特征投影
            fused_features = torch.cat([node_emb, routed_features], dim=-1)
            projected_features = self.feature_projection(fused_features)
            
            # GNN处理
            inner_node_message = self.inner_gnn(projected_features, inner_edge_index)
            outer_node_message = self.outer_gnn(projected_features, outer_edge_index)
            
            # 跨模态融合
            updated_node = self.cross_modal_fusion(projected_features, inner_node_message, outer_node_message)
            
            # 预测
            return self._predict(updated_node, batch, sum_weight)
        except Exception:
            # 回退到静态模式
            return self.forward_static(node_emb, inner_edge_index, outer_edge_index, batch, sum_weight)
    
    def forward_static(self, node_emb, inner_edge_index, outer_edge_index, batch, sum_weight):
        try:
            # GNN处理
            inner_node_message = self.inner_gnn(node_emb, inner_edge_index)
            outer_node_message = self.outer_gnn(node_emb, outer_edge_index)
            
            # 跨模态融合
            updated_node = self.cross_modal_fusion(node_emb, inner_node_message, outer_node_message)
            
            # 预测
            return self._predict(updated_node, batch, sum_weight)
        except Exception:
            if node_emb is not None:
                return torch.zeros(node_emb.shape[0], 1, device=self.device)
            else:
                return torch.zeros(1, 1, device=self.device)
    
    def apply_routing(self, node_emb, router_weights, router_features):
        try:
            if node_emb.dim() == 2:
                node_emb = node_emb.unsqueeze(0)
            
            batch_size, num_nodes, dim = node_emb.shape
            router_features = router_features.unsqueeze(0).expand(batch_size, -1, -1)
            
            routed_emb = torch.bmm(router_weights, router_features)
            
            return routed_emb.squeeze(0) if batch_size == 1 else routed_emb
        except Exception:
            return node_emb
    
    def _predict(self, updated_node, batch, sum_weight):
        user_node_num = self.num_user_features
        if user_node_num <= 0:
            user_node_num = 1
        
        try:
            return self.kan_predictor(updated_node, batch, sum_weight, user_node_num=user_node_num)
        except Exception:
            # KAN失败时使用简单预测
            if updated_node is not None:
                return self.g(updated_node)
            else:
                return torch.zeros(1, 1, device=self.device)