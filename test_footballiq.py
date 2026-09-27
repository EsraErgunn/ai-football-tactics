# -*- coding: utf-8 -*-
"""
test_footballiq.py
==================
FootballIQ sistemi icin birim testler (pytest).

Calistirmak icin:
    pip install pytest numpy
    pytest test_footballiq.py -v

Her test bir kucuk fonksiyonu kontrol eder. Test adi "test_" ile baslar.
"""

import numpy as np

from footballiq_logic import (
    compute_voronoi_grid,
    heatmap_points_to_grid,
    is_ball_valid,
    resolve_goalkeepers_team_id,
    HEATMAP_GRID,
    PITCH_LENGTH,
    PITCH_WIDTH,
)


# ============================================================
# 1) heatmap_points_to_grid testleri
# ============================================================
def test_heatmap_empty_points():
    """Bos liste verince butun hucreler 0 olmali."""
    grid = heatmap_points_to_grid([])
    total = sum(sum(row) for row in grid)
    assert total == 0


def test_heatmap_single_point():
    """Tek nokta verince izgaradaki toplam tam olarak 1 olmali."""
    points = [[PITCH_LENGTH / 2, PITCH_WIDTH / 2]]   # sahanin ortasi
    grid = heatmap_points_to_grid(points)
    total = sum(sum(row) for row in grid)
    assert total == 1


def test_heatmap_grid_size():
    """Donen izgaranin boyutu HEATMAP_GRID ile ayni olmali."""
    grid = heatmap_points_to_grid([[100, 100]])
    rows = len(grid)
    cols = len(grid[0])
    assert rows == HEATMAP_GRID[1]
    assert cols == HEATMAP_GRID[0]


def test_heatmap_counts_all_points():
    """Uc nokta verince toplam say 3 olmali."""
    points = [[1000, 1000], [6000, 3500], [11000, 6000]]
    grid = heatmap_points_to_grid(points)
    total = sum(sum(row) for row in grid)
    assert total == 3


def test_heatmap_out_of_range_is_clipped():
    """Saha disindaki nokta da bir hucreye kirpilir, kaybolmaz."""
    points = [[999999, 999999]]   # cok buyuk, saha disinda
    grid = heatmap_points_to_grid(points)
    total = sum(sum(row) for row in grid)
    assert total == 1


# ============================================================
# 2) compute_voronoi_grid testleri
# ============================================================
def test_voronoi_returns_none_when_team_missing():
    """Takimlardan biri bossa fonksiyon None dondurmeli."""
    team_0 = np.array([[1000.0, 3500.0]])
    team_1 = np.empty((0, 2))
    result = compute_voronoi_grid(team_0, team_1)
    assert result is None


def test_voronoi_left_right_split():
    """Takim A solda, Takim B sagda: sol hucreler 0, sag hucreler 1 olmali."""
    team_0 = np.array([[1000.0, 3500.0]])    # sol
    team_1 = np.array([[11000.0, 3500.0]])   # sag
    grid = compute_voronoi_grid(team_0, team_1)
    assert grid is not None
    # En sol sutun Takim A (0), en sag sutun Takim B (1)
    assert grid[:, 0].mean() == 0      # sol kenar tamamen A
    assert grid[:, -1].mean() == 1     # sag kenar tamamen B


def test_voronoi_grid_shape():
    """Izgara sekli (satir=genislik, sutun=uzunluk) dogru olmali."""
    team_0 = np.array([[1000.0, 3500.0]])
    team_1 = np.array([[11000.0, 3500.0]])
    grid = compute_voronoi_grid(team_0, team_1)
    assert grid.shape == (40, 60)


# ============================================================
# 3) resolve_goalkeepers_team_id testleri
# ============================================================
def test_goalkeeper_near_team_a():
    """Kaleci Takim A merkezine yakinsa 0 dondurmeli."""
    players_xy = np.array([[1000, 3500], [1500, 3000], [11000, 3500], [10500, 4000]])
    players_class = np.array([0, 0, 1, 1])
    goalkeeper_xy = np.array([[500, 3500]])   # sol, Takim A'ya yakin
    result = resolve_goalkeepers_team_id(players_xy, players_class, goalkeeper_xy)
    assert result[0] == 0


def test_goalkeeper_near_team_b():
    """Kaleci Takim B merkezine yakinsa 1 dondurmeli."""
    players_xy = np.array([[1000, 3500], [1500, 3000], [11000, 3500], [10500, 4000]])
    players_class = np.array([0, 0, 1, 1])
    goalkeeper_xy = np.array([[11500, 3500]])  # sag, Takim B'ye yakin
    result = resolve_goalkeepers_team_id(players_xy, players_class, goalkeeper_xy)
    assert result[0] == 1


def test_goalkeeper_fallback_when_one_team_missing():
    """Bir takim hic yoksa fonksiyon guvenli sekilde 0 dondurmeli."""
    players_xy = np.array([[1000, 3500], [1500, 3000]])
    players_class = np.array([0, 0])   # sadece Takim A var
    goalkeeper_xy = np.array([[11500, 3500]])
    result = resolve_goalkeepers_team_id(players_xy, players_class, goalkeeper_xy)
    assert result[0] == 0


# ============================================================
# 4) is_ball_valid (top outlier filtresi) testleri
# ============================================================
def test_ball_first_position_always_valid():
    """Ilk top konumu her zaman gecerli (karsilastiracak onceki yok)."""
    assert is_ball_valid([6000, 3500], None)


def test_ball_small_move_is_valid():
    """Top kisa mesafe hareket ederse gecerli olmali."""
    last = np.array([6000, 3500])
    new = np.array([6100, 3550])   # 100 cm civari, normal
    assert is_ball_valid(new, last)


def test_ball_large_jump_is_rejected():
    """Top cok uzaga zipladiysa (hatali tespit) reddedilmeli."""
    last = np.array([1000, 1000])
    new = np.array([11000, 6000])   # binlerce cm sicrama, imkansiz
    assert not is_ball_valid(new, last)