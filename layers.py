import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing,GCNConv,GATConv
import numpy as np
import copy
import math
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.utils import softmax, add_self_loops
from torch_geometric.nn.inits import glorot, zeros
from torch.nn import Linear
import numpy as np
import time
from icecream import ic


################################################# Implement for gat #####################################

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, dropout=0.6)
        # On the Pubmed dataset, use `heads` output heads in `conv2`.
        self.conv2 = GATConv(hidden_channels , out_channels, heads=1,
                             concat=False, dropout=0.6)

    def forward(self, x, edge_index):
        x = x.squeeze()
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x
#GAT（图注意力网络）:
################################################# Implement for gcn#####################################
class standard_GCN(MessagePassing):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.act = nn.ReLU()

    def forward(self, x, edge_index, edge_weight=None):
        x = x.squeeze()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.act(self.conv1(x, edge_index, edge_weight=edge_weight))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return  x

#标准GCN（图卷积网络）:
################################################# Implement for original inner #####################################

class inner_GNN(MessagePassing):
    def __init__(self, dim, hidden_layer):
        super(inner_GNN, self).__init__(aggr='mean')

        #construct pairwise modeling network
        self.lin1 = nn.Linear(dim, hidden_layer)
        self.lin2 = nn.Linear(hidden_layer, dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=0.5)
    def forward(self, x, edge_index, edge_weight=None):
        # x has shape [N, dim]
        # edge_index has shape [2, E]
        #try:
        #    return self.propagate(edge_index, x=x, edge_weight=edge_weight)
        #except:
        # print("x.shape",x.shape)
        x = x.squeeze()
        # print("x.shape.new",x.shape)
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)

    def message(self, x_i, x_j, edge_weight):
        # x_i has shape [E, dim]
        # x_j has shape [E, dim]

        # pairwise analysis
        pairwise_analysis = x_i * x_j
        pairwise_analysis = self.lin1(pairwise_analysis)
        pairwise_analysis = self.act(pairwise_analysis)
        pairwise_analysis = self.lin2(pairwise_analysis)
        pairwise_analysis = self.drop(pairwise_analysis)
        #pairwise_analysis = x_i * x_j

        if edge_weight != None:
            interaction_analysis = pairwise_analysis * edge_weight.view(-1,1)
        else:
            interaction_analysis = pairwise_analysis

        return interaction_analysis

    def update(self, aggr_out):
        # aggr_out has shape [N, dim]
        return aggr_out
#内部GNN:深入分析分子层面的相互作用
################################################# Implement for original cross#####################################

class cross_GNN(MessagePassing):
    def __init__(self, dim, hidden_layer):
        super(cross_GNN, self).__init__(aggr='mean')

    def forward(self, x, edge_index, edge_weight=None):
        # x has shape [N, dim]
        # edge_index has shape [2, E]
        #try:
        #    return self.propagate(edge_index, x=x, edge_weight=edge_weight)
        #except:
        x = x.squeeze()
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)

    #def message(self, edge_index_i, x_i, x_j, edge_weight, num_nodes):
    def message(self, x_i, x_j, edge_weight):
        # x_i has shape [E, dim]
        # x_j has shape [E, dim]

        # pairwise analysis
        pairwise_analysis = x_i * x_j

        if edge_weight != None:
            interaction_analysis = pairwise_analysis * edge_weight.view(-1,1)
        else:
            interaction_analysis = pairwise_analysis

        return interaction_analysis

    def update(self, aggr_out):
        # aggr_out has shape [N, dim]
        return aggr_out
#跨域GNN:
################################################# DACMF Dual Attention Modules #####################################

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, d_v, d_model, h=4, dropout=.1):
        super(ScaledDotProductAttention, self).__init__()
        self.h = h
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        
        # 查询、键、值投影
        self.fc_q = nn.Linear(d_k, h * d_k)
        self.fc_k = nn.Linear(d_k, h * d_k)
        self.fc_p = nn.Linear(d_v, h * d_v)  # 用于药物→靶点
        self.fc_g = nn.Linear(d_v, h * d_v)  # 用于靶点→药物
        
        # 输出投影
        self.fc_o1 = nn.Linear(h * d_k, d_k)
        self.fc_o2 = nn.Linear(h * d_v, d_v)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys):
        '''
        queries: 药物特征 [batch_size, drug_seq_len, d_k]
        keys: 靶点特征 [batch_size, target_seq_len, d_k]
        '''
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        # 投影到多头空间
        u = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # [b, h, nq, d_k]
        v = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)     # [b, h, d_k, nk]
        p = self.fc_p(keys).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)     # [b, h, nk, d_v]
        g = self.fc_g(queries).view(b_s, nq, self.h, self.d_v).permute(0, 2, 1, 3)  # [b, h, nq, d_v]

        # 计算注意力分数
        att = torch.matmul(u, v) / np.sqrt(self.d_k)  # [b, h, nq, nk]
        
        # 双向注意力
        # 药物→靶点注意力
        att_p = torch.softmax(att, -1)
        att_p = self.dropout(att_p)
        P = torch.matmul(att_p, p).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)

        # 靶点→药物注意力  
        att_d = torch.softmax(att.permute(0, 1, 3, 2), -1)
        att_d = self.dropout(att_d)
        D = torch.matmul(att_d, g).permute(0, 2, 1, 3).contiguous().view(b_s, nk, self.h * self.d_v)

        # 输出投影
        P = self.fc_o1(P)  # 增强的靶点特征
        D = self.fc_o2(D)  # 增强的药物特征

        return P, D
#双注意力模块:
class DACMFDualAttention(nn.Module):
    """
    DACMF双注意力机制适配器
    用于增强跨域交互
    """
    def __init__(self, dim, hidden_layer, num_heads=4, dropout=0.1):
        super(DACMFDualAttention, self).__init__()
        self.dim = dim
        self.hidden_layer = hidden_layer
        
        # 双注意力核心
        self.dual_attention = ScaledDotProductAttention(
            d_k=dim, 
            d_v=dim, 
            d_model=dim, 
            h=num_heads, 
            dropout=dropout
        )
        
        # 融合层
        self.fusion_fc = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 残差连接和归一化
        self.layer_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, drug_features, target_features):
        """
        drug_features: 药物节点特征 [num_drug_nodes, dim]
        target_features: 靶点节点特征 [num_target_nodes, dim] 
        """
        # 确保是3D张量 [batch_size, seq_len, dim]
        if drug_features.dim() == 2:
            drug_features = drug_features.unsqueeze(0)  # [1, num_drug_nodes, dim]
            target_features = target_features.unsqueeze(0)  # [1, num_target_nodes, dim]
        
        # 应用双注意力
        target_enhanced, drug_enhanced = self.dual_attention(drug_features, target_features)
        
        # 交叉模态融合 (DACMF核心思想)
        # 药物视角: 原始特征 + 靶点关注的药物特征
        drug_fused = torch.cat([drug_features, drug_enhanced], dim=-1)
        # 靶点视角: 原始特征 + 药物关注的靶点特征
        target_fused = torch.cat([target_features, target_enhanced], dim=-1)
        
        # 融合处理
        drug_output = self.fusion_fc(drug_fused)
        target_output = self.fusion_fc(target_fused)
        
        # 残差连接和归一化
        drug_output = self.layer_norm(drug_output + drug_features)
        target_output = self.layer_norm(target_output + target_features)
        
        # 恢复原始形状
        if drug_output.shape[0] == 1:
            drug_output = drug_output.squeeze(0)  # [num_drug_nodes, dim]
            target_output = target_output.squeeze(0)  # [num_target_nodes, dim]
            
        return drug_output, target_output

################################################# Enhanced Cross GNN with Dual Attention #####################################

class enhanced_cross_GNN(MessagePassing):
    """
    增强的跨域GNN，融合DACMF双注意力机制
    """
    def __init__(self, dim, hidden_layer, use_dual_attention=True, num_heads=2,dropout=0.3):
        super(enhanced_cross_GNN, self).__init__(aggr='mean')
        
        # 修复：保存参数到正确的属性名
        self.use_dual_attention = use_dual_attention
        
        self.dim = dim
        self.hidden_layer = hidden_layer
        
        # DACMF双注意力组件
        if self.use_dual_attention:  # 使用正确的属性名
            self.dual_attention = DACMFDualAttention(
                dim=dim, 
                hidden_layer=hidden_layer, 
                num_heads=num_heads
            )
            
            # 融合权重（可学习）
            self.attention_weight = nn.Parameter(torch.tensor(0.7))
            
        # 原有的消息传递网络
        self.message_net = nn.Sequential(
            nn.Linear(dim * 2, hidden_layer),
            nn.ReLU(),
            nn.Linear(hidden_layer, dim),
            nn.Dropout(0.1)
        )

    def forward(self, x, edge_index, edge_weight=None):
        x = x.squeeze()
        
        # 应用双注意力（在消息传递之前）
        if self.use_dual_attention and hasattr(self, 'dual_attention'):
            # 分离药物和靶点节点
            drug_mask = torch.zeros(x.size(0), dtype=torch.bool)
            target_mask = torch.zeros(x.size(0), dtype=torch.bool)
            
            # 这里需要根据你的图结构来识别药物和靶点节点
            # 假设前一半是药物节点，后一半是靶点节点（根据你的数据调整）
            split_idx = x.size(0) // 2
            drug_mask[:split_idx] = True
            target_mask[split_idx:] = True
            
            drug_features = x[drug_mask]
            target_features = x[target_mask]
            
            # 应用双注意力
            enhanced_drug, enhanced_target = self.dual_attention(drug_features, target_features)
            
            # 融合回原始特征
            x_enhanced = x.clone()
            x_enhanced[drug_mask] = (1 - self.attention_weight) * drug_features + self.attention_weight * enhanced_drug
            x_enhanced[target_mask] = (1 - self.attention_weight) * target_features + self.attention_weight * enhanced_target
            
            # 使用增强的特征进行消息传递
            return self.propagate(edge_index, x=x_enhanced, edge_weight=edge_weight)
        else:
            # 原有的消息传递
            return self.propagate(edge_index, x=x, edge_weight=edge_weight)

    def message(self, x_i, x_j, edge_weight):
        # 原有的消息计算
        pairwise_analysis = x_i * x_j
        
        # 通过消息网络
        if hasattr(self, 'message_net'):
            message_output = self.message_net(torch.cat([x_i, x_j], dim=-1))
        else:
            message_output = pairwise_analysis

        if edge_weight is not None:
            interaction_analysis = message_output * edge_weight.view(-1, 1)
        else:
            interaction_analysis = message_output

        return interaction_analysis

    def update(self, aggr_out):
        return aggr_out
#让双方充分了解对方的需求和特点
################################################# Implement for Transformer#####################################

torch.manual_seed(1)
np.random.seed(1)

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, variance_epsilon=1e-12):

        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


class Embeddings(nn.Module):
    """Construct the embeddings from protein/target, position embeddings.
    """
    def __init__(self, vocab_size, hidden_size, max_position_size, dropout_rate):
        super(Embeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_size, hidden_size)

        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = words_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob):
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class SelfOutput(nn.Module):
    def __init__(self, hidden_size, hidden_dropout_prob):
        super(SelfOutput, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class Attention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob):
        super(Attention, self).__init__()
        self.self = SelfAttention(hidden_size, num_attention_heads, attention_probs_dropout_prob)
        self.output = SelfOutput(hidden_size, hidden_dropout_prob)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output

class Intermediate(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super(Intermediate, self).__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = F.relu(hidden_states)
        return hidden_states

class Output(nn.Module):
    def __init__(self, intermediate_size, hidden_size, hidden_dropout_prob):
        super(Output, self).__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class Encoder(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob):
        super(Encoder, self).__init__()
        self.attention = Attention(hidden_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob)
        self.intermediate = Intermediate(hidden_size, intermediate_size)
        self.output = Output(intermediate_size, hidden_size, hidden_dropout_prob)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class Encoder_MultipleLayers(nn.Module):
    def __init__(self, n_layer, hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob):
        super(Encoder_MultipleLayers, self).__init__()
        layer = Encoder(hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layer)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            #if output_all_encoded_layers:
            #    all_encoder_layers.append(hidden_states)
        #if not output_all_encoded_layers:
        #    all_encoder_layers.append(hidden_states)
        return hidden_stages



class transformer(nn.Sequential):
    def __init__(self, encoding, **config):
        super(transformer, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if encoding == 'drug':
            self.emb = Embeddings(config['input_dim_drug'], config['transformer_emb_size_drug'], 50, config['transformer_dropout_rate'])
            self.encoder = Encoder_MultipleLayers(config['transformer_n_layer_drug'],
                                                    config['transformer_emb_size_drug'],
                                                    config['transformer_intermediate_size_drug'],
                                                    config['transformer_num_attention_heads_drug'],
                                                    config['transformer_attention_probs_dropout'],
                                                    config['transformer_hidden_dropout_rate'])
        elif encoding == 'protein':
            self.emb = Embeddings(config['input_dim_protein'], config['transformer_emb_size_target'], 545, config['transformer_dropout_rate'])
            self.encoder = Encoder_MultipleLayers(config['transformer_n_layer_target'],
                                                    config['transformer_emb_size_target'],
                                                    config['transformer_intermediate_size_target'],
                                                    config['transformer_num_attention_heads_target'],
                                                    config['transformer_attention_probs_dropout'],
                                                    config['transformer_hidden_dropout_rate'])

    ### parameter v (tuple of length 2) is from utils.drug2emb_encoder
    def forward(self, v):
        e = v[0].long().to(self.device)
        e_mask = v[1].long().to(self.device)
        ex_e_mask = e_mask.unsqueeze(1).unsqueeze(2)
        ex_e_mask = (1.0 - ex_e_mask) * -10000.0

        emb = self.emb(e)
        encoded_layers = self.encoder(emb.float(), ex_e_mask.float())
        return encoded_layers[:,0]
# 在 layers.py 文件末尾添加
class AdaptiveRouterLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_routers=50):
        super(AdaptiveRouterLayer, self).__init__()
        self.input_dim = input_dim
        self.num_routers = num_routers
        
        # 注意力网络
        self.attention_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_routers),
            nn.Softmax(dim=-1)
        )
        
        # 可学习的路由节点嵌入
        self.router_embeddings = nn.Embedding(num_routers, input_dim)
        nn.init.xavier_uniform_(self.router_embeddings.weight)

    def forward(self, node_embeddings):
        # 使用 model.py 中的实现
        # 或者直接导入
        pass