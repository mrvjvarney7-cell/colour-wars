"""Translated, scenario-for-scenario port of js/gameLogic.test.js.

Every test here mirrors one in the JS suite by name and intent, so the two
files should be read side by side. This is the cross-check required before
any training run: if these pass, the Python rules engine matches the
authoritative JS implementation exactly.

Run with: pytest python/colourwars/tests/test_game.py -v
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from colourwars.game import (  # noqa: E402
    CRITICAL_MASS,
    Cell,
    apply_move,
    count_cells_for_player,
    create_empty_board,
    create_game,
    get_critical_mass,
    get_neighbors,
    is_valid_move,
    placement_dots,
    play_move,
)

CM = 4  # fixed critical mass for every cell


def mid_game(state):
    """Tests that hand-build a mid-game board are simulating players who have
    already opened, so clear the per-player opening-move bonus for them."""
    for p in state.players:
        p.has_moved = True
    return state


def place_dots(board, r, c, player, times):
    """Repeatedly place single dots on one cell for `player`, returning the
    result of the final placement."""
    res = None
    b = board
    for _ in range(times):
        res = apply_move(b, r, c, player, 7, 7)
        b = res.board
    return res


# ---------- Critical mass is a fixed 4 everywhere ----------


def test_critical_mass_fixed_4_corners_edges_interior():
    assert CRITICAL_MASS == 4
    assert get_critical_mass(0, 0, 7, 7) == 4
    assert get_critical_mass(0, 6, 7, 7) == 4
    assert get_critical_mass(6, 0, 7, 7) == 4
    assert get_critical_mass(6, 6, 7, 7) == 4
    assert get_critical_mass(0, 3, 7, 7) == 4
    assert get_critical_mass(3, 0, 7, 7) == 4
    assert get_critical_mass(6, 3, 7, 7) == 4
    assert get_critical_mass(3, 6, 7, 7) == 4
    assert get_critical_mass(3, 3, 7, 7) == 4
    assert get_critical_mass(1, 1, 7, 7) == 4
    assert get_critical_mass(5, 2, 7, 7) == 4


def test_every_one_of_49_cells_reports_critical_mass_4():
    for r in range(7):
        for c in range(7):
            assert get_critical_mass(r, c, 7, 7) == 4


def test_neighbor_lookup_only_existing_orthogonal_neighbors():
    assert len(get_neighbors(0, 0, 7, 7)) == 2
    assert len(get_neighbors(0, 6, 7, 7)) == 2
    assert len(get_neighbors(6, 0, 7, 7)) == 2
    assert len(get_neighbors(6, 6, 7, 7)) == 2
    assert len(get_neighbors(0, 3, 7, 7)) == 3
    assert len(get_neighbors(3, 0, 7, 7)) == 3
    assert len(get_neighbors(3, 3, 7, 7)) == 4
    for (nr, nc) in get_neighbors(0, 0, 7, 7):
        assert 0 <= nr < 7 and 0 <= nc < 7
        assert abs(nr - 0) + abs(nc - 0) == 1


# ---------- Nothing explodes below 4, everything explodes at exactly 4 ----------


def test_no_cell_explodes_at_1_2_3_dots():
    for (r, c) in [(0, 0), (0, 6), (6, 0), (6, 6), (0, 3), (3, 0), (6, 3), (3, 6), (3, 3), (1, 1), (5, 2)]:
        res = place_dots(create_empty_board(7, 7), r, c, 0, 3)
        assert res.board[r][c].count == 3
        assert res.board[r][c].owner == 0
        assert len(res.steps) == 0


def test_corner_explodes_at_exactly_4_feeding_both_neighbors():
    res = place_dots(create_empty_board(7, 7), 0, 6, 0, 4)
    assert len(res.steps) == 1
    assert res.board[0][6].count == 0
    assert res.board[0][6].owner is None
    assert res.board[1][6].count == 1
    assert res.board[1][6].owner == 0
    assert res.board[0][5].count == 1
    assert res.board[0][5].owner == 0
    total = sum(cell.count for row in res.board for cell in row)
    assert total == 2


def test_edge_cell_explodes_at_exactly_4_feeding_all_three():
    res = place_dots(create_empty_board(7, 7), 0, 3, 0, 4)
    assert len(res.steps) == 1
    assert res.board[0][3].count == 0
    assert res.board[0][3].owner is None
    for (r, c) in [(0, 2), (0, 4), (1, 3)]:
        assert res.board[r][c].count == 1
        assert res.board[r][c].owner == 0
    total = sum(cell.count for row in res.board for cell in row)
    assert total == 3


def test_interior_cell_explodes_at_exactly_4_feeding_all_four():
    res = place_dots(create_empty_board(7, 7), 3, 3, 0, 4)
    assert len(res.steps) == 1
    assert res.board[3][3].count == 0
    assert res.board[3][3].owner is None
    for (r, c) in [(2, 3), (4, 3), (3, 2), (3, 4)]:
        assert res.board[r][c].count == 1
        assert res.board[r][c].owner == 0
    total = sum(cell.count for row in res.board for cell in row)
    assert total == 4


def test_every_corner_and_edge_explodes_at_4_without_crashing():
    for (r, c) in [(0, 0), (0, 6), (6, 0), (6, 6), (0, 3), (3, 0), (6, 3), (3, 6)]:
        res = place_dots(create_empty_board(7, 7), r, c, 0, 4)
        assert len(res.steps) == 1
        for rr in range(7):
            for cc in range(7):
                assert res.board[rr][cc].count >= 0
        nbrs = get_neighbors(r, c, 7, 7)
        delivered = sum(res.board[nr][nc].count for (nr, nc) in nbrs)
        assert delivered == len(nbrs)


def test_basic_placement_empty_cell_and_own_cell():
    board = create_empty_board(7, 7)
    r1 = apply_move(board, 3, 3, 0, 7, 7)
    assert r1.board[3][3].owner == 0
    assert r1.board[3][3].count == 1
    assert len(r1.steps) == 0
    r2 = apply_move(r1.board, 3, 3, 0, 7, 7)
    assert r2.board[3][3].count == 2


# ---------- Capture and chain reactions ----------


def test_exploding_into_opponent_cell_captures_regardless_of_owner():
    board = create_empty_board(7, 7)
    board[3][3] = Cell(owner=0, count=3)
    board[2][3] = Cell(owner=1, count=1)
    result = apply_move(board, 3, 3, 0, 7, 7)
    assert result.board[2][3].owner == 0
    assert result.board[2][3].count == 2


def test_explosion_pushing_neighbor_to_4_chains_further():
    board = create_empty_board(7, 7)
    board[3][3] = Cell(owner=0, count=3)
    board[3][4] = Cell(owner=0, count=3)
    result = apply_move(board, 3, 3, 0, 7, 7)
    assert len(result.steps) >= 2
    assert result.board[3][4].count == 0
    assert result.board[3][4].owner is None
    for (r, c) in [(2, 4), (4, 4), (3, 5)]:
        assert result.board[r][c].owner == 0
        assert result.board[r][c].count == 1


def test_corner_reaching_4_purely_from_cascade_explodes_same_move():
    board = create_empty_board(7, 7)
    board[0][6] = Cell(owner=1, count=3)
    board[1][6] = Cell(owner=0, count=3)
    result = apply_move(board, 1, 6, 0, 7, 7)
    assert len(result.steps) >= 2
    assert result.board[0][6].owner is None
    assert result.board[0][6].count == 0
    assert result.board[0][5].owner == 0


def test_chain_reaction_captures_cells_from_multiple_owners_across_waves():
    board = create_empty_board(7, 7)
    board[3][3] = Cell(owner=0, count=3)
    board[3][4] = Cell(owner=1, count=1)
    board[2][3] = Cell(owner=2, count=3)
    result = apply_move(board, 3, 3, 0, 7, 7)
    assert len(result.steps) >= 2
    assert result.board[3][4].owner == 0
    assert result.board[2][3].owner is None
    assert result.board[1][3].owner == 0


def test_no_cell_ever_rests_at_or_above_critical_mass():
    board = create_empty_board(7, 7)
    board[3][3] = Cell(owner=0, count=3)
    board[3][4] = Cell(owner=0, count=3)
    board[2][3] = Cell(owner=0, count=3)
    board[4][3] = Cell(owner=0, count=3)
    board[3][2] = Cell(owner=0, count=3)
    res = apply_move(board, 3, 3, 0, 7, 7)
    for r in range(7):
        for c in range(7):
            assert res.board[r][c].count < CM
            assert res.board[r][c].count >= 0


def test_does_not_explode_when_below_critical_mass():
    board = create_empty_board(7, 7)
    board[3][3] = Cell(owner=0, count=2)
    result = apply_move(board, 3, 3, 0, 7, 7)
    assert result.board[3][3].count == 3
    assert len(result.steps) == 0


# ---------- Move validation ----------


def test_empty_cell_is_valid_before_opening_invalid_after():
    board = create_empty_board(7, 7)
    assert is_valid_move(board, 2, 2, 0, False) is True
    assert is_valid_move(board, 2, 2, 1, False) is True
    assert is_valid_move(board, 2, 2, 0, True) is False


def test_own_cell_valid_opponent_cell_not():
    board = create_empty_board(7, 7)
    board[2][2] = Cell(owner=0, count=1)
    assert is_valid_move(board, 2, 2, 0, True) is True
    assert is_valid_move(board, 2, 2, 1, True) is False


def test_out_of_bounds_is_invalid():
    board = create_empty_board(7, 7)
    assert is_valid_move(board, -1, 0, 0, False) is False
    assert is_valid_move(board, 0, 7, 0, False) is False


def test_cannot_play_on_opponent_owned_cell_via_play_move():
    state = mid_game(create_game(2))
    state.board[3][3] = Cell(owner=1, count=1)
    r = play_move(state, 3, 3)
    assert r.state.current_player_index == 0
    assert r.state.board[3][3].count == 1


# ---------- Turn order, elimination and winning ----------


def test_turn_advances_to_next_player_after_a_move():
    state = create_game(3)
    r = play_move(state, 3, 3)
    assert r.state.current_player_index == 1


def test_board_dimensions_stay_7x7_by_default():
    state = create_game(2)
    assert len(state.board) == 7
    assert len(state.board[0]) == 7


def test_no_eliminations_before_every_player_has_had_one_turn():
    state = create_game(2, 7, 7)
    r = play_move(state, 0, 0)
    assert r.state.game_over is False
    assert r.state.players[0].active is True
    assert r.state.players[1].active is True


def test_player_with_zero_cells_eliminated_once_all_players_moved():
    state = mid_game(create_game(2, 7, 7))
    state.board[5][5] = Cell(owner=1, count=3)
    state.current_player_index = 1
    state.total_moves = 1
    r = play_move(state, 5, 5)
    assert r.state.players[0].active is False
    assert r.state.game_over is True
    assert r.state.winner == 1


def test_full_game_last_player_standing_wins_only_when_truly_last():
    state = mid_game(create_game(2, 7, 7))
    state.board[0][0] = Cell(owner=0, count=1)
    state.board[3][3] = Cell(owner=1, count=3)
    state.board[3][2] = Cell(owner=0, count=1)
    state.board[3][4] = Cell(owner=0, count=1)
    state.board[2][3] = Cell(owner=0, count=1)
    state.board[4][3] = Cell(owner=0, count=1)
    state.current_player_index = 1
    state.total_moves = 1
    r = play_move(state, 3, 3)
    assert r.state.players[0].active is True
    assert r.state.game_over is False


def test_4p_no_elimination_before_every_player_moved_once():
    state = create_game(4, 7, 7)
    state.total_moves = 2
    state.current_player_index = 0
    r = play_move(state, 0, 0)
    assert r.state.players[1].active is True
    assert r.state.game_over is False


def test_4p_turn_order_skips_eliminated_player():
    state = mid_game(create_game(4, 7, 7))
    state.board[3][3] = Cell(owner=0, count=3)
    state.board[2][3] = Cell(owner=1, count=1)
    state.board[6][6] = Cell(owner=2, count=1)
    state.board[0][6] = Cell(owner=3, count=1)
    state.board[0][0] = Cell(owner=0, count=1)
    state.current_player_index = 0
    state.total_moves = 4

    r1 = play_move(state, 3, 3)
    assert r1.state.players[1].active is False
    assert r1.state.game_over is False
    assert r1.state.current_player_index == 2

    r2 = play_move(r1.state, 6, 6)  # player 2's own cell
    assert r2.state.current_player_index == 3
    r3 = play_move(r2.state, 0, 6)  # player 3's own cell
    assert r3.state.current_player_index == 0


def test_4p_single_multiwave_cascade_eliminates_three_opponents():
    state = mid_game(create_game(4, 7, 7))
    state.board[3][3] = Cell(owner=0, count=3)
    state.board[0][0] = Cell(owner=0, count=1)
    state.board[2][3] = Cell(owner=1, count=1)
    state.board[3][4] = Cell(owner=1, count=1)
    state.board[4][3] = Cell(owner=2, count=3)
    state.board[3][2] = Cell(owner=3, count=1)
    state.board[5][3] = Cell(owner=3, count=1)
    state.current_player_index = 0
    state.total_moves = 3

    r = play_move(state, 3, 3)
    assert len(r.steps) >= 2
    assert count_cells_for_player(r.state.board, 1) == 0
    assert count_cells_for_player(r.state.board, 2) == 0
    assert count_cells_for_player(r.state.board, 3) == 0
    assert count_cells_for_player(r.state.board, 0) > 0
    for (rr, cc) in [(2, 3), (3, 4), (3, 2), (5, 3)]:
        assert r.state.board[rr][cc].owner == 0
    assert r.state.board[4][3].owner is None
    assert r.state.game_over is True
    assert r.state.winner == 0


def test_4p_game_does_not_falsely_end_while_two_or_more_hold_cells():
    state = mid_game(create_game(4, 7, 7))
    state.board[3][3] = Cell(owner=0, count=3)
    state.board[2][3] = Cell(owner=1, count=1)
    state.board[6][6] = Cell(owner=2, count=1)
    state.board[0][6] = Cell(owner=3, count=1)
    state.current_player_index = 0
    state.total_moves = 4
    r = play_move(state, 3, 3)
    assert r.state.players[1].active is False
    assert r.state.players[2].active is True
    assert r.state.players[3].active is True
    assert r.state.game_over is False
    assert r.state.winner is None


# ---------- Placement rules ----------


def test_after_opening_further_placements_restricted_to_own_cells():
    state = create_game(2, 7, 7)
    r1 = play_move(state, 0, 0)
    assert r1.state.current_player_index == 1
    assert is_valid_move(r1.state.board, 5, 5, 1, False) is True  # player 1's opening
    r2 = play_move(r1.state, 5, 5)
    assert r2.state.board[5][5].owner == 1
    assert r2.state.board[5][5].count == 3
    assert r2.state.current_player_index == 0

    # Player 0 has already opened - an unowned empty cell is no longer valid.
    assert is_valid_move(r2.state.board, 2, 6, 0, True) is False
    r3 = play_move(r2.state, 2, 6)
    assert r3.state.board[2][6].owner is None  # no-op
    assert r3.state.current_player_index == 0  # invalid move does not consume the turn

    # Their own opening cell is still a valid target - the 4th dot reaches
    # critical mass and detonates immediately.
    assert is_valid_move(r3.state.board, 0, 0, 0, True) is True
    r4 = play_move(r3.state, 0, 0)
    assert r4.state.board[0][0].count == 0
    assert r4.state.board[0][0].owner is None
    assert len(r4.steps) >= 1


# ---------- Opening-move rule ----------


def test_opening_move_first_placement_puts_3_dots():
    state = create_game(2, 7, 7)
    assert placement_dots(state, 0) == 3
    r = play_move(state, 3, 3)
    assert r.state.board[3][3].count == 3
    assert r.state.board[3][3].owner == 0
    assert len(r.steps) == 0


def test_opening_3_dots_never_detonates_even_on_corner():
    state = create_game(2, 7, 7)
    r = play_move(state, 0, 6)
    assert len(r.steps) == 0
    assert r.state.board[0][6].count == 3
    assert r.state.board[0][6].owner == 0
    assert r.state.board[1][6].owner is None
    assert r.state.board[0][5].owner is None


def test_opening_bonus_is_per_player():
    s = create_game(4, 7, 7)
    spots = [(1, 1), (1, 4), (4, 1), (4, 4)]
    for i in range(4):
        assert placement_dots(s, i) == 3
        res = play_move(s, spots[i][0], spots[i][1])
        assert res.state.board[spots[i][0]][spots[i][1]].count == 3
        assert res.state.board[spots[i][0]][spots[i][1]].owner == i
        s = res.state
    for j in range(4):
        assert placement_dots(s, j) == 1


def test_after_opening_later_move_on_own_cell_adds_exactly_1_dot_before_resolving():
    state = create_game(2, 7, 7)
    r1 = play_move(state, 1, 1)
    assert r1.state.board[1][1].count == 3
    r2 = play_move(r1.state, 4, 4)
    assert r2.state.board[4][4].count == 3
    assert placement_dots(r2.state, 0) == 1
    # Player 0's only owned cell is their opening cell - that is now their
    # only legal move, and the 4th dot reaches the threshold and detonates.
    r3 = play_move(r2.state, 1, 1)
    assert r3.state.board[1][1].count == 0
    assert len(r3.steps) >= 1


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
