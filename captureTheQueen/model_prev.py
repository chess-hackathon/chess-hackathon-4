from torch import nn

import io
import torch
import torch.nn as nn
import chess.pgn
import numpy as np
import chess
from collections import OrderedDict, Counter
from policy_index import policy_index, policy_index_flipped


DEBUG_OUTPUT=False

import torch 
from torch import nn

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=11, patch_size=4, embed_dim=192):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, x):
        return self.mha(x, x, x)[0]

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformerTwoHeads(nn.Module):
    def __init__(self, in_channels=11, patch_size=4, embed_dim=192, num_heads=8, num_layers=6, num_classes1=3, num_classes2=1858):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, patch_size, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, (8 // patch_size) ** 2 + 1, embed_dim))
        
        self.transformer = nn.ModuleList([TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head1 = nn.Linear(embed_dim, num_classes1)
        self.head2 = nn.Linear(embed_dim, num_classes2)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        
        for layer in self.transformer:
            x = layer(x)
        
        x = self.norm(x)
        
        cls_token_final = x[:, 0]
        
        out1 = self.head1(cls_token_final)
        out2 = self.head2(cls_token_final)
        
        return out1, out2

def board_to_leela_input(board, expanded=True):
    '''
    board: chess.Board
    returns: tensor of shape (1, 19, 8, 8)
    '''
    flipped = False
    if board.turn == chess.BLACK:
        board = board.mirror()
        flipped = True
        
    our_pieces = board.occupied_co[chess.WHITE]
    their_pieces = board.occupied_co[chess.BLACK]

    full_board = (1 << 64) - 1

    planes = np.array([
        board.pawns & our_pieces,
        board.knights & our_pieces,
        board.bishops & our_pieces,
        board.rooks & our_pieces,
        board.queens & our_pieces,
        board.kings & our_pieces,
        board.pawns & their_pieces,
        board.knights & their_pieces,
        board.bishops & their_pieces,
        board.rooks & their_pieces,
        board.queens & their_pieces,
        board.kings & their_pieces,
        board.has_queenside_castling_rights(chess.WHITE) * full_board,
        board.has_kingside_castling_rights(chess.WHITE) * full_board,
        board.has_queenside_castling_rights(chess.BLACK) * full_board,
        board.has_kingside_castling_rights(chess.BLACK) * full_board,
        flipped  * full_board,
        0,
        1 * full_board,
    ], dtype="uint64")

    expanded_planes = np.unpackbits(planes.view("uint8")).reshape(1, 19, 8, 8)

    return expanded_planes if expanded else planes


def leela_policy_to_uci_moves(policy, flip):
    return dict(zip(policy_index_flipped if flip else policy_index, policy))


class Model(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = VisionTransformerTwoHeads(
            in_channels=model['in_channels'],
            patch_size=model['patch_size'],
            embed_dim=model['embed_dim'],
            num_heads=model['num_heads'],
            num_layers=model['num_layers'],
            num_classes1=model['num_classes1'],
            num_classes2=model['num_classes2']
        )
        self.criterion1 = nn.MSELoss()
        self.criterion2 = nn.MSELoss()

        self.board = None
        self.pgn = None
        self.policy = None

    def forward(self, x):
        return self.model(x)


    def score(self, pgn, move):
        '''
        pgn: string e.g. "1.e4 a6 2.Bc4 "
        move: string e.g. "a5 "
        '''

        if pgn != self.pgn:
            with torch.no_grad():
                self.pgn = pgn
                game = chess.pgn.read_game(io.StringIO(pgn))
                self.board = chess.Board()
                # catch board up on game to present
                for past_move in list(game.mainline_moves()):
                    self.board.push(past_move)
                
                tensor_input = torch.from_numpy(
                    board_to_leela_input(self.board).astype("float32")
                )
                q, policy_output = self.model.forward(tensor_input)
                policy = policy_output.numpy().flatten()
                self.policy = leela_policy_to_uci_moves(policy, flip=(self.board.turn == chess.BLACK))

                if DEBUG_OUTPUT:
                    print("Analyzed new board")
                    print(self.board.unicode())
                    print("Q: ", q)
                    print("Best moves:", Counter(self.policy).most_common(5))
        
        uci_move = self.board.parse_san(move).uci()

        if uci_move not in self.policy:
            print("Very strange!  Can't recognize move: ", move, uci_move)
        result = float(self.policy.get(uci_move, -1.0))
        if DEBUG_OUTPUT:
            print(uci_move, "->", result)
        return result
