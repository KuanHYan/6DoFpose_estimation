import torch
import torch.nn as nn
from mmdet.registry import MODELS


@MODELS.register_module()
class AttentionHead(nn.Module):
    def __init__(self, fea_channel=256, hidden_dim=512, num_heads=8,
                 posed_dim=6, topk=32,
                 dp=0.0, batch_first=True,
                 add_sa_module=True) -> None:
        super().__init__()
        # self.hidden_dim = hidden_dim = roi_feat_size**2 * num_heads
        self.hidden_dim = hidden_dim
        self.use_attention = num_heads > 0
        if self.use_attention:
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dp,
                batch_first=batch_first
            )
            self.feed_forward = nn.Sequential(
                # nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, 4*hidden_dim),
                nn.GELU(),  # 参数 approximate 用来控制是否使用一个更快的近似算法来计算GeLU函数。
                nn.Linear(4*hidden_dim, hidden_dim),
            )
        else:
            self.shared_net = nn.Sequential(
                nn.Linear(2*topk*hidden_dim, 4*hidden_dim),
                nn.ReLU(inplace=True),
            )
            self.fea_base = nn.Linear(4*hidden_dim, hidden_dim)
            self.fea_move = nn.Linear(4*hidden_dim, hidden_dim)
        self.add_sa_module = add_sa_module
        if self.add_sa_module:
            self.self_attention = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    hidden_dim, num_heads, dropout=dp, batch_first=batch_first),
                num_layers=2
            )
        else:
            self.self_attention = None
        self.flatten = nn.Sequential(
            nn.Conv2d(fea_channel, hidden_dim, 7, 1),
            nn.ReLU(inplace=True)
            # nn.AdaptiveAvgPool2d(1)
        )
        # self.score_linear = nn.Linear(hidden_dim, 1)
        self.topk = topk
        self.norm = nn.LayerNorm(hidden_dim)

    def _pad_and_create_mask(self, batch_roi_res):
        """
        Pad sequences to the same length and create attention mask
        
        Args:
            batch_roi_res: List of tensors with shape [L_i, hidden_dim] where L_i varies
            
        Returns:
            padded_sequences: Tensor of shape [batch_size, max_length, hidden_dim]
            attention_mask: Boolean tensor of shape [batch_size, max_length] 
                        where True indicates padded positions
        """
        # Find the maximum sequence length
        max_len = self.topk # max(seq.size(0) for seq in batch_roi_res)
        
        # Create padded sequences
        padded_sequences = []
        attention_masks = []
        
        for seq in batch_roi_res:
            seq_len = seq.size(0)
            hidden_dim = seq.size(1)
            
            # Pad with zeros
            if seq_len < max_len:
                pad_zeros = torch.zeros(max_len - seq_len, hidden_dim, 
                                    dtype=seq.dtype, device=seq.device)
                padded_seq = torch.cat([seq, pad_zeros], dim=0)
            else:
                padded_seq = seq[:self.topk, :]
                
            padded_sequences.append(padded_seq)
            
            # Create attention mask (True for valid positions, False for padding)
            mask = torch.ones(max_len, dtype=torch.bool, device=seq.device)
            mask[seq_len:] = False
            attention_masks.append(mask)
        
        padded_sequences = torch.stack(padded_sequences, dim=0)  # [batch_size, max_len, hidden_dim]
        attention_masks = torch.stack(attention_masks, dim=0)    # [batch_size, max_len]
        
        return padded_sequences, attention_masks

    def forward(self, roi_res, rois_per_img):
        bz = len(rois_per_img) // 2
        flatten_roi_res = self.flatten(roi_res).reshape(roi_res.shape[0], -1)
        assert flatten_roi_res.shape[-1] == self.hidden_dim
        batch_roi_res = flatten_roi_res.split(rois_per_img)
        # base_roi, move_roi = flatten_roi_res[:bz], flatten_roi_res[bz:]
        # Each roi is of shape [bz, L, hidden_dim]
        key, k_mask = self._pad_and_create_mask(batch_roi_res[:bz])
        query, q_mask = self._pad_and_create_mask(batch_roi_res[bz:])

        if self.use_attention:
            query = self.norm(query)
            key = self.norm(key)
            move_attn_output, attn_output_weights = \
                self.cross_attention.forward(
                    query, key, value=key, key_padding_mask=~k_mask)
            # attn_output_weights is of shape N, L, S
            # To predict base pose, query(N, L, D) is the key and value
            # NOTE: 不能用转置，会变为 S,L,N， 需要的是(N, S, L) * (N, L, D)
            base_attn_output = torch.matmul(attn_output_weights.transpose(-2, -1), query)
            if self.self_attention is not None:
                move_attn_output = self.self_attention.forward(move_attn_output)
            # attn_output is of shape [bz, L, hidden_dim]
            # attn_output_weights is of shape [bz, L_query, L_key]
            # TODO: how to select the top k scores and predict poses for concatenated rois
            # scores, indices = attn_output_weights.reshape(bz, -1).topk(k=self.topk, dim=1)
            # query_idx = indices // attn_output_weights.shape[1]
            # key_idx = indices % attn_output_weights.shape[2]
            
            # query = torch.gather(query, 1,
            #                      query_idx.unsqueeze(-1).expand(-1, -1, dim))
            # key = torch.gather(attn_output, 1,
            #                    key_idx.unsqueeze(-1).expand(-1, -1, dim))
            # query is of shape [bz, topk, hidden_dim]
            # key is of shape [bz, topk, hidden_dim]
            # key_regions = torch.cat([query, key], dim=1)
            query = query.mean(dim=1) + move_attn_output.mean(dim=1)
            query = self.norm(query)
            fea_move = self.feed_forward(query) + query
            key = key.mean(dim=1) + base_attn_output.mean(dim=1)
            key = self.norm(key)
            fea_base = self.feed_forward(key) + key
        else:
            fea = torch.cat([query, key], dim=1).reshape(bz, -1)
            shared_fea = self.shared_net(fea)
            fea_move = self.fea_move(shared_fea)
            fea_base = self.fea_base(shared_fea)
        return fea_base, fea_move
        # rot_mv = self.rotation_linear(fea_move)
        # tran_mv = self.translation_linear(fea_move)
        # rot_ba = self.rotation_linear(fea_base)
        # tran_ba = self.translation_linear(fea_base)
        # return torch.cat([rot_ba, rot_mv], dim=0), torch.cat([tran_ba, tran_mv], dim=0)
