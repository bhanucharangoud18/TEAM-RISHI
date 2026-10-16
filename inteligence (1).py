"""
Pi-Redactor competition adapter.

Public API:
    choose_move(board_fen: str, legal_moves: list[str], time_left_ms: int) -> str

This file is self-contained apart from python-chess and uses a compact
iterative-deepening negamax search with alpha-beta pruning, move ordering,
quiescence search, and a small transposition table.

Install dependency:
    pip install python-chess

If the competition runner provides only FEN + legal moves, this module uses
the supplied legal-move list as the source of truth for the root move choices.
"""

from __future__ import annotations

import math
import time
from typing import Dict, Optional, Tuple

try:
    import chess
except ImportError as exc:  # Fail clearly if the required dependency is missing.
    raise ImportError(
        "inteligence.py requires python-chess. Install it with: pip install python-chess"
    ) from exc


MATE = 100_000
INF = 200_000
MAX_QPLY = 12
PIECE_VALUE = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 335,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}
TT: Dict[tuple, tuple] = {}
TT_LIMIT = 120_000


def _piece_square(piece_type: int, square: int, color: bool, endgame: bool) -> int:
    """Lightweight positional terms; all scores are from White's perspective."""
    file_ = chess.square_file(square)
    rank = chess.square_rank(square)
    relative_rank = rank if color == chess.WHITE else 7 - rank
    center = 7 - (abs(2 * file_ - 7) + abs(2 * rank - 7)) // 2

    if piece_type == chess.PAWN:
        return relative_rank * (10 if endgame else 7) + center * 2
    if piece_type == chess.KNIGHT:
        return center * 8
    if piece_type == chess.BISHOP:
        return center * 4
    if piece_type == chess.ROOK:
        return relative_rank * 2
    if piece_type == chess.QUEEN:
        return center * 2
    if piece_type == chess.KING:
        return center * (7 if endgame else -5)
    return 0


def _evaluate(board: "chess.Board") -> int:
    """Static evaluation in centipawns, positive means White is better."""
    material = 0
    phase = 0
    pieces = []
    for square, piece in board.piece_map().items():
        value = PIECE_VALUE[piece.piece_type]
        material += value if piece.color == chess.WHITE else -value
        phase_weight = {
            chess.KNIGHT: 1, chess.BISHOP: 1, chess.ROOK: 2, chess.QUEEN: 4
        }.get(piece.piece_type, 0)
        phase += phase_weight
        pieces.append((square, piece))

    endgame = phase <= 8
    positional = 0
    for square, piece in pieces:
        term = _piece_square(piece.piece_type, square, piece.color, endgame)
        positional += term if piece.color == chess.WHITE else -term

    # Small mobility and king-safety nudges, without generating many extra moves.
    score = material + positional
    if board.is_check():
        score += -22 if board.turn == chess.WHITE else 22
    return score


def _terminal_score(board: "chess.Board", ply: int) -> Optional[int]:
    if board.is_checkmate():
        return -MATE + ply
    if (
        board.is_stalemate()
        or board.is_insufficient_material()
        or board.is_fifty_moves()
        or board.is_repetition(3)
    ):
        return 0
    return None


def _move_order_score(board: "chess.Board", move: "chess.Move", tt_move: Optional["chess.Move"]) -> int:
    if tt_move is not None and move == tt_move:
        return 10_000_000
    score = 0
    if board.is_capture(move):
        victim = chess.PAWN if board.is_en_passant(move) else board.piece_type_at(move.to_square)
        attacker = board.piece_type_at(move.from_square)
        score += 1_000_000 + 10 * PIECE_VALUE.get(victim, 0) - PIECE_VALUE.get(attacker, 0)
    if move.promotion:
        score += 800_000 + PIECE_VALUE.get(move.promotion, 0)
    if board.gives_check(move):
        score += 25_000
    return score


def _ordered_moves(board: "chess.Board", moves, tt_move=None):
    return sorted(
        moves,
        key=lambda move: _move_order_score(board, move, tt_move),
        reverse=True,
    )


def _check_time(deadline: float) -> None:
    if time.perf_counter() >= deadline:
        raise TimeoutError


def _quiescence(
    board: "chess.Board", alpha: int, beta: int, ply: int,
    deadline: float, qply: int = 0
) -> int:
    _check_time(deadline)
    terminal = _terminal_score(board, ply)
    if terminal is not None:
        return terminal

    in_check = board.is_check()
    if qply >= MAX_QPLY:
        # If checked, never use stand-pat; choose a conservative static leaf
        # after ensuring terminal checks above have been handled.
        if in_check:
            best = -INF
            for move in list(board.legal_moves):
                board.push(move)
                score = -_evaluate(board) if board.turn == chess.WHITE else _evaluate(board)
                board.pop()
                best = max(best, score)
            return best if best != -INF else -MATE + ply
        return _evaluate(board) if board.turn == chess.WHITE else -_evaluate(board)

    if not in_check:
        stand_pat_white = _evaluate(board)
        stand_pat = stand_pat_white if board.turn == chess.WHITE else -stand_pat_white
        if stand_pat >= beta:
            return stand_pat
        alpha = max(alpha, stand_pat)
        moves = [m for m in board.legal_moves if board.is_capture(m) or m.promotion]
    else:
        # In check, search all legal evasions; stand-pat is forbidden.
        moves = list(board.legal_moves)

    if not moves:
        return -MATE + ply if in_check else alpha

    for move in _ordered_moves(board, moves):
        board.push(move)
        score = -_quiescence(board, -beta, -alpha, ply + 1, deadline, qply + 1)
        board.pop()
        if score >= beta:
            return score
        if score > alpha:
            alpha = score
    return alpha


def _negamax(
    board: "chess.Board", depth: int, alpha: int, beta: int, ply: int,
    deadline: float
) -> int:
    _check_time(deadline)
    terminal = _terminal_score(board, ply)
    if terminal is not None:
        return terminal
    if depth <= 0:
        return _quiescence(board, alpha, beta, ply, deadline)

    alpha_original = alpha
    key = (board._transposition_key(), depth)
    entry = TT.get(key)
    tt_move = None
    if entry is not None:
        tt_score, flag, stored_move = entry
        tt_move = stored_move
        if flag == "exact":
            return tt_score
        if flag == "lower":
            alpha = max(alpha, tt_score)
        elif flag == "upper":
            beta = min(beta, tt_score)
        if alpha >= beta:
            return tt_score

    best_score = -INF
    best_move = None
    moves = _ordered_moves(board, list(board.legal_moves), tt_move)
    if not moves:
        return -MATE + ply if board.is_check() else 0

    for move in moves:
        board.push(move)
        score = -_negamax(board, depth - 1, -beta, -alpha, ply + 1, deadline)
        board.pop()

        if score > best_score:
            best_score, best_move = score, move
        alpha = max(alpha, score)
        if alpha >= beta:
            break

    flag = "exact"
    if best_score <= alpha_original:
        flag = "upper"
    elif best_score >= beta:
        flag = "lower"
    if len(TT) >= TT_LIMIT:
        TT.clear()
    TT[key] = (best_score, flag, best_move)
    return best_score


def choose_move(board_fen: str, legal_moves: list[str], time_left_ms: int) -> str:
    """Choose and return a legal move from legal_moves in UCI notation."""
    if not legal_moves:
        raise ValueError("legal_moves is empty: no legal move can be returned")

    # The competition's supplied list is authoritative. Preserve a safe fallback.
    fallback = legal_moves[0]
    board = chess.Board(board_fen)

    parsed = []
    for uci in legal_moves:
        try:
            move = chess.Move.from_uci(uci)
        except ValueError:
            continue
        if move in board.legal_moves:
            parsed.append(move)

    if not parsed:
        # Avoid returning a move that cannot be parsed; the runner should provide
        # legal UCI strings, but fail clearly if its input violates the contract.
        return fallback

    # Keep the root move selection restricted to the supplied list.
    allowed = {move.uci() for move in parsed}
    fallback = next((m for m in legal_moves if m in allowed), fallback)

    # Reserve clock for the caller/serialization overhead; clamp tiny budgets.
    budget_ms = max(1, int(time_left_ms))
    think_ms = max(1, min(250, int(budget_ms * 0.035)))
    if budget_ms <= 20:
        think_ms = max(1, budget_ms // 3)
    deadline = time.perf_counter() + think_ms / 1000.0

    best_uci = fallback
    try:
        # Iterative deepening ensures a completed shallower result is retained.
        for depth in range(1, 64):
            _check_time(deadline)
            alpha, beta = -INF, INF
            best_score = -INF
            depth_best = None
            root_moves = _ordered_moves(board, parsed)
            for move in root_moves:
                _check_time(deadline)
                board.push(move)
                score = -_negamax(board, depth - 1, -beta, -alpha, 1, deadline)
                board.pop()
                if score > best_score:
                    best_score, depth_best = score, move
                alpha = max(alpha, score)
            if depth_best is not None:
                best_uci = depth_best.uci()
            if abs(best_score) >= MATE - 100:
                break
    except TimeoutError:
        pass
    except Exception:
        # Always return a legal fallback on unexpected engine-side failures.
        return fallback

    return best_uci if best_uci in allowed else fallback


if __name__ == "__main__":
    # Minimal local smoke test.
    start = chess.Board()
    move = choose_move(start.fen(), [m.uci() for m in start.legal_moves], 1000)
    print(move)
