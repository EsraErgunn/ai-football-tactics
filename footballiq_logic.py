# -*- coding: utf-8 -*-
"""
footballiq_logic.py
===================
FootballIQ'nun saf-numpy mantik fonksiyonlari, birim test icin izole edildi.
Bu fonksiyonlar main_api.py icindeki orijinallerin AYNISIDIR; sadece cv2,
torch, supervision gibi agir bagimliliklar olmadan test edilebilsinler diye
buraya tasinmistir.
"""

import numpy as np

# main_api.py'deki sabitlerle ayni
PITCH_LENGTH = 12000   # cm  (CONFIG.length)
PITCH_WIDTH = 7000     # cm  (CONFIG.width)
VORONOI_GRID = (60, 40)
HEATMAP_GRID = (60, 40)
BALL_OUTLIER_THRESHOLD_CM = 500
PROCESS_STRIDE = 3


def heatmap_points_to_grid(points, grid_cols=HEATMAP_GRID[0], grid_rows=HEATMAP_GRID[1]):
    """Ham (x, y) cm noktalarini yogunluk izgarasina cevirir."""
    grid = np.zeros((grid_rows, grid_cols), dtype=int)
    if not points:
        return grid.tolist()
    pts = np.array(points)
    xs = np.clip((pts[:, 0] / PITCH_LENGTH * grid_cols).astype(int), 0, grid_cols - 1)
    ys = np.clip((pts[:, 1] / PITCH_WIDTH * grid_rows).astype(int), 0, grid_rows - 1)
    np.add.at(grid, (ys, xs), 1)
    return grid.tolist()


def compute_voronoi_grid(team_0_xy, team_1_xy):
    """Her hucre icin 0 (Takim A daha yakin) veya 1 (Takim B daha yakin)."""
    if len(team_0_xy) == 0 or len(team_1_xy) == 0:
        return None
    gx = np.linspace(0, PITCH_LENGTH, VORONOI_GRID[0])
    gy = np.linspace(0, PITCH_WIDTH, VORONOI_GRID[1])
    grid = np.stack(np.meshgrid(gx, gy), axis=-1).reshape(-1, 2)
    d0 = np.min(np.linalg.norm(grid[:, None, :] - team_0_xy[None, :, :], axis=2), axis=1)
    d1 = np.min(np.linalg.norm(grid[:, None, :] - team_1_xy[None, :, :], axis=2), axis=1)
    return (d1 < d0).astype(int).reshape(VORONOI_GRID[1], VORONOI_GRID[0])


def resolve_goalkeepers_team_id(players_xy, players_class_id, goalkeepers_xy):
    """Kaleciyi takim merkezine (centroid) yakinligina gore esler."""
    players_xy = np.asarray(players_xy)
    players_class_id = np.asarray(players_class_id)
    goalkeepers_xy = np.asarray(goalkeepers_xy)
    team_0 = players_xy[players_class_id == 0]
    team_1 = players_xy[players_class_id == 1]
    if len(team_0) == 0 or len(team_1) == 0:
        return np.zeros(len(goalkeepers_xy), dtype=int)
    c0, c1 = team_0.mean(axis=0), team_1.mean(axis=0)
    return np.array(
        [0 if np.linalg.norm(g - c0) < np.linalg.norm(g - c1) else 1
         for g in goalkeepers_xy]
    )


def is_ball_valid(candidate_ball_xy, last_valid_ball_xy):
    """Top outlier filtresi: top iki islenen kare arasinda cok uzaga gitmez."""
    if last_valid_ball_xy is None:
        return True
    dist = np.linalg.norm(np.asarray(candidate_ball_xy) - np.asarray(last_valid_ball_xy))
    return dist <= BALL_OUTLIER_THRESHOLD_CM * PROCESS_STRIDE