import sys
import chess
import math
import random

class CTRLEngine:
    def __init__(self, depth=3):
        self.depth = depth
        self.piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000
        }

        # Таблиці позиційної цінності для грамотного розвитку фігур
        self.pst = {
            chess.PAWN: [
                 0,  0,  0,  0,  0,  0,  0,  0,
                 5, 10, 10,-20,-20, 10, 10,  5,
                 5, -5,-10,  0,  0,-10, -5,  5,
                 0,  0,  0, 25, 25,  0,  0,  0,
                 5,  5, 10, 30, 30, 10,  5,  5,
                10, 10, 20, 30, 30, 20, 10, 10,
                50, 50, 50, 50, 50, 50, 50, 50,
                 0,  0,  0,  0,  0,  0,  0,  0
            ],
            chess.KNIGHT: [
                -50,-40,-30,-30,-30,-30,-40,-50,
                -40,-20,  0,  5,  5,  0,-20,-40,
                -30,  5, 10, 15, 15, 10,  5,-30,
                -30,  0, 15, 20, 20, 15,  0,-30,
                -30,  5, 15, 20, 20, 15,  5,-30,
                -30,  0, 10, 15, 15, 10,  0,-30,
                -40,-20,  0,  0,  0,  0,-20,-40,
                -50,-40,-30,-30,-30,-30,-40,-50
            ],
            chess.BISHOP: [
                -20,-10,-10,-10,-10,-10,-10,-20,
                -10,  5,  0,  0,  0,  0,  5,-10,
                -10, 10, 10, 10, 10, 10, 10,-10,
                -10,  0, 10, 10, 10, 10,  0,-10,
                -10,  5,  5, 10, 10,  5,  5,-10,
                -10,  0,  5, 10, 10,  5,  0,-10,
                -10,  0,  0,  0,  0,  0,  0,-10,
                -20,-10,-10,-10,-10,-10,-10,-20
            ],
            chess.ROOK: [
                  0,  0,  0,  5,  5,  0,  0,  0,
                 -5,  0,  0,  0,  0,  0,  0, -5,
                 -5,  0,  0,  0,  0,  0,  0, -5,
                 -5,  0,  0,  0,  0,  0,  0, -5,
                 -5,  0,  0,  0,  0,  0,  0, -5,
                 -5,  0,  0,  0,  0,  0,  0, -5,
                  5, 10, 10, 10, 10, 10, 10,  5,
                  0,  0,  0,  0,  0,  0,  0,  0
            ],
            chess.QUEEN: [
                -20,-10,-10, -5, -5,-10,-10,-20,
                -10,  0,  0,  0,  0,  0,  0,-10,
                -10,  0,  5,  5,  5,  5,  0,-10,
                 -5,  0,  5,  5,  5,  5,  0, -5,
                  0,  0,  5,  5,  5,  5,  0, -5,
                -10,  5,  5,  5,  5,  5,  0,-10,
                -10,  0,  5,  0,  0,  0,  0,-10,
                -20,-10,-10, -5, -5,-10,-10,-20
            ],
            chess.KING: [
                 20, 30, 10,  0,  0, 10, 30, 20,
                 20, 20,  0,  0,  0,  0, 20, 20,
                -10,-20,-20,-20,-20,-20,-20,-10,
                -20,-30,-30,-40,-40,-30,-30,-20,
                -30,-40,-40,-50,-50,-40,-40,-30,
                -30,-40,-40,-50,-50,-40,-40,-30,
                -30,-40,-40,-50,-50,-40,-40,-30,
                -30,-40,-40,-50,-50,-40,-40,-30
            ]
        }

    def evaluate_board(self, board):
        if board.is_checkmate():
            return -99999 if board.turn == chess.WHITE else 99999
        if board.is_stalemate() or board.is_insufficient_material():
            return 0

        eval_score = 0
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece is not None:
                val = self.piece_values[piece.piece_type]
                sq_idx = sq if piece.color == chess.WHITE else chess.square_mirror(sq)
                pst_val = self.pst[piece.piece_type][sq_idx]
                total_val = val + pst_val

                if piece.color == chess.WHITE:
                    eval_score += total_val
                else:
                    eval_score -= total_val

        return eval_score

    def minimax(self, board, depth, alpha, beta, is_maximizing):
        if depth == 0 or board.is_game_over():
            return self.evaluate_board(board)

        if is_maximizing:
            max_eval = -math.inf
            for move in board.legal_moves:
                board.push(move)
                eval_score = self.minimax(board, depth - 1, alpha, beta, False)
                board.pop()
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = math.inf
            for move in board.legal_moves:
                board.push(move)
                eval_score = self.minimax(board, depth - 1, alpha, beta, True)
                board.pop()
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval

    def get_best_move(self, board):
        best_moves = []
        is_maximizing = board.turn == chess.WHITE

        if is_maximizing:
            best_eval = -math.inf
            for move in board.legal_moves:
                board.push(move)
                eval_score = self.minimax(board, self.depth - 1, -math.inf, math.inf, False)
                board.pop()
                if eval_score > best_eval:
                    best_eval = eval_score
                    best_moves = [move]
                elif eval_score == best_eval:
                    best_moves.append(move)
        else:
            best_eval = math.inf
            for move in board.legal_moves:
                board.push(move)
                eval_score = self.minimax(board, self.depth - 1, -math.inf, math.inf, True)
                board.pop()
                if eval_score < best_eval:
                    best_eval = eval_score
                    best_moves = [move]
                elif eval_score == best_eval:
                    best_moves.append(move)

        if best_moves:
            return random.choice(best_moves)
        elif list(board.legal_moves):
            return list(board.legal_moves)[0]

        return None


def main():
    engine = CTRLEngine(depth=3)
    board = chess.Board()

    while True:
        line = sys.stdin.readline()
        if not line:
            break
        line = line.strip()

        if line == "uci":
            print("id name c-t-r-l")
            print("id author You")
            # Усі необхідні UCI-параметри для lichess-bot:
            print("option name Move Overhead type spin default 0 min 0 max 10000")
            print("option name Threads type spin default 1 min 1 max 128")
            print("option name Hash type spin default 16 min 1 max 1024")
            print("option name SyzygyPath type string default <empty>")
            print("option name Ponder type check default false")
            print("option name MultiPV type spin default 1 min 1 max 500")
            print("option name UCI_AnalyseMode type check default false")
            print("option name UCI_Chess960 type check default false")
            print("option name UCI_Opponent type string default <empty>")
            print("option name UCI_EngineAbout type string default <empty>")
            print("option name UCI_ShowWDL type check default false")
            print("option name UCI_LimitStrength type check default false")
            print("option name UCI_Elo type spin default 1500 min 1350 max 2850")
            print("option name Skill Level type spin default 20 min 0 max 20")
            print("uciok")
            sys.stdout.flush()
        elif line == "isready":
            print("readyok")
            sys.stdout.flush()
        elif line.startswith("setoption"):
            pass
        elif line == "ucinewgame":
            board = chess.Board()
        elif line.startswith("position"):
            tokens = line.split()
            if "startpos" in tokens:
                board = chess.Board()
                if "moves" in tokens:
                    moves_idx = tokens.index("moves") + 1
                    for move in tokens[moves_idx:]:
                        board.push(chess.Move.from_uci(move))
            elif "fen" in tokens:
                fen_idx = tokens.index("fen")
                moves_idx = tokens.index("moves") if "moves" in tokens else len(tokens)
                fen_str = " ".join(tokens[fen_idx + 1:moves_idx])
                board = chess.Board(fen_str)
                if "moves" in tokens:
                    for move in tokens[moves_idx + 1:]:
                        board.push(chess.Move.from_uci(move))
        elif line.startswith("go"):
            best_move = engine.get_best_move(board)
            if best_move:
                print(f"bestmove {best_move.uci()}")
            sys.stdout.flush()
        elif line == "quit":
            break


if __name__ == "__main__":
    main()