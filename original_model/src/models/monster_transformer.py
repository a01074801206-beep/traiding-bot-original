import torch
import torch.nn as nn
import torch.nn.functional as F

class MonsterTransformer(nn.Module):
    def __init__(self, price_dim=15, macro_dim=7, nlp_dim=5, embed_dim=128, nhead=8, num_layers=3):
        """
        Multi-modal Transformer for Financial Time Series
        5D 텐서의 서로 다른 차원을 Cross-Attention으로 융합
        """
        super(MonsterTransformer, self).__init__()
        
        # 1. Feature Embedding: 각 차원을 동일한 embed_dim(128차원)으로 투사
        # RTX 4060의 FP32 연산에 최적화
        self.price_emb = nn.Linear(price_dim, embed_dim)
        self.macro_emb = nn.Linear(macro_dim, embed_dim)
        self.nlp_emb = nn.Linear(nlp_dim, embed_dim)
        
        # 2. Positional Encoding: 시계열 순서 정보 주입 (Transformer의 필수 요소)
        # Sequence Length는 DataLoader에서 결정 (예: 60일치 데이터)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 100, embed_dim)) # max_len=100 가정

        # 3. Transformer Encoder (각 채널의 내부 패턴 학습)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=512, dropout=0.1, batch_first=True
        )
        self.price_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.macro_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.nlp_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Cross-Attention: 데이터 융합의 핵심
        # 가격 데이터(Query)가 매크로와 뉴스(Key, Value) 중 어디에 반응하는지 계산
        self.cross_attn_macro = nn.MultiheadAttention(embed_dim, num_heads=nhead, batch_first=True)
        self.cross_attn_news = nn.MultiheadAttention(embed_dim, num_heads=nhead, batch_first=True)
        
        # 5. Final Prediction Head (수익률 예측)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim * 3, 256), # Price + Macro-Context + News-Context
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1) # Target Return (내일의 예상 수익률)
        )

    def forward(self, price_seq, macro_seq, nlp_seq):
        """
        Forward Pass (i5-14400에서 미니배치를 받아 RTX 4060에서 연산)
        Inputs: (Batch, Seq_Len, Dim) -> float32
        """
        batch_size, seq_len, _ = price_seq.size()
        
        # A. Embedding + Positional Encoding
        p_emb = self.price_emb(price_seq) + self.pos_encoder[:, :seq_len, :]
        m_emb = self.macro_emb(macro_seq) + self.pos_encoder[:, :seq_len, :]
        n_emb = self.nlp_emb(nlp_seq) + self.pos_encoder[:, :seq_len, :]
        
        # B. Transformer Encoding (내부 특징 추출)
        p_feat = self.price_transformer(p_emb) # 가격 시계열 특징
        m_feat = self.macro_transformer(m_emb) # 매크로 시계열 특징
        n_feat = self.nlp_transformer(n_emb) # 뉴스 시계열 특징
        
        # C. Cross-Attention: 5D 데이터 융합 (RTX 4060 CUDA 코어 풀가동)
        # 가격 정보에 매크로 정보를 주입
        macro_context, _ = self.cross_attn_macro(p_feat, m_feat, m_feat)
        # 가격 정보에 뉴스 정보를 주입
        news_context, _ = self.cross_attn_news(p_feat, n_feat, n_feat)
        
        # D. Concat & Predict
        # 마지막 타임스텝의 특징만 추출하여 합침
        last_price = p_feat[:, -1, :]
        last_macro = macro_context[:, -1, :]
        last_news = news_context[:, -1, :]
        
        combined = torch.cat([last_price, last_macro, last_news], dim=-1)
        
        output = self.fc(combined)
        return output