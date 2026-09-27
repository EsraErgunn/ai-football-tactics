# -*- coding: utf-8 -*-
"""
FootballIQ - Yapay Zeka Motoru (Backend) - v2.0 "Near-Live"
===========================================================
Onceki surumun uzerine eklenen yetenekler (dokumandaki eksikler):

  1. WEBSOCKET CANLI AKIS (/ws/analyze/{video_id}):
     Backend videoyu kare kare islerken sonuclari BEKLEMEDEN frontend'e
     firlatir. Kullanici dashboard'da verilerin anlik aktigini gorur.
     Akis 3 tip mesaj icerir:
       - {"tip": "durum", ...}  -> "takim siniflandirici egitiliyor" gibi
       - {"tip": "kare",  ...}  -> her islenen kare icin radar + istatistik
       - {"tip": "ozet",  ...}  -> video bitince nihai rapor

  2. KARE BAZLI RADAR VERISI:
     Her "kare" mesajinda oyuncular [{"id": tracker_id, "takim": 0/1,
     "x": cm, "y": cm}] ve top {"x", "y"} olarak gider. Frontend bu
     noktalari HTML5 Canvas'a cizer, rengi "takim" alanina gore secer,
     "#5" gibi etiketleri "id" alanindan basar.

  3. ISI HARITASI GRID FORMATI:
     Nihai ozette ham koordinat listesine ek olarak 60x40'lik bir yogunluk
     izgarasi (grid) da donuyor. heatmap.js gibi kutuphaneler icin nokta
     listesi, kendi Canvas cizimin icin grid kullanilabilir.

  4. VORONOI IZGARASI (Toggle icin):
     Belirli araliklarla "kare" mesajina 60x40'lik sahiplik izgarasi
     ekleniyor (0 = Takim A, 1 = Takim B). Frontend'deki "Voronoi" butonu
     acildiginda bu izgara yari saydam renklerle radarin uzerine binebilir.

  5. AKIS MIMARISI:
     Once POST /upload ile video yuklenir (video_id doner), sonra frontend
     ws://sunucu/ws/analyze/{video_id} adresine baglanir ve akisi dinler.
     Eski tek atimlik POST /analyze endpoint'i de aynen calismaya devam
     ediyor (WebSocket istemeyen basit kullanim icin).

Calistirmak icin:
    pip install fastapi uvicorn python-multipart websockets
    uvicorn main_api:app --host 0.0.0.0 --port 8000

Hizli test (sunucusuz):
    python main_api.py
"""

import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
import uuid
from collections import deque
from datetime import datetime
from typing import Dict, Generator, List, Optional

import cv2
import numpy as np
import supervision as sv
import torch
from fastapi import (
    FastAPI,
    File,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from tqdm import tqdm

from inference import get_model
from sports.annotators.soccer import draw_pitch, draw_points_on_pitch
from sports.common.team import TeamClassifier
from sports.common.view import ViewTransformer
from sports.configs.soccer import SoccerPitchConfiguration

# ----------------------------------------------------------------------------
# API ANAHTARLARI
# Anahtarlar koda YAZILMAZ; proje klasorundeki .env dosyasindan (bkz.
# .env.example) veya sistem ortam degiskenlerinden okunur.
# ----------------------------------------------------------------------------
try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except ImportError:
    pass  # python-dotenv yoksa sadece sistem ortam degiskenleri kullanilir

os.environ["ONNXRUNTIME_EXECUTION_PROVIDERS"] = "[CPUExecutionProvider]"

ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY")
if not ROBOFLOW_API_KEY:
    sys.exit(
        "ROBOFLOW_API_KEY tanimli degil. .env.example dosyasini .env olarak "
        "kopyalayip kendi anahtarini yaz."
    )

torch.set_default_device("cpu")
DEVICE = "cpu"

# ----------------------------------------------------------------------------
# MODELLER (sunucu acilirken SADECE BIR KEZ yuklenir)
# ----------------------------------------------------------------------------
PLAYER_DETECTION_MODEL = get_model(
    model_id="football-players-detection-3zvbc/11", api_key=ROBOFLOW_API_KEY
)
FIELD_DETECTION_MODEL = get_model(
    model_id="football-field-detection-f07vi/14", api_key=ROBOFLOW_API_KEY
)

CONFIG = SoccerPitchConfiguration()

BALL_ID = 0
GOALKEEPER_ID = 1
PLAYER_ID = 2
REFEREE_ID = 3

# ----------------------------------------------------------------------------
# ANALIZ PARAMETRELERI
# SoccerPitchConfiguration santimetre kullanir (saha 12000 x 7000 cm).
# ----------------------------------------------------------------------------
CONTROL_DISTANCE_CM = 200        # 2.0 m: top bu mesafede ise oyuncu "kontrolde"
MIN_PASS_DISTANCE_CM = 300       # 3 m altindaki top hareketi pas sayilmaz.
                                 # (8 m cok yuksekti: futboldaki kisa paslarin cogu
                                 # 8 m altinda kaliyor ve sayilmiyordu. ID kayma
                                 # artiklarini artik ID koprusu temizledigi icin
                                 # bu esik gercek pas filtrelemek zorunda degil.)
MIN_CONTROL_FRAMES = 2           # Oyuncunun topu "aldi" sayilmasi icin gereken kare
                                 # (3 idi; stride'li 8 fps akista tek dokunus
                                 # paslarini kaciriyordu)

# --- ID KOPRUSU (kararli kimlik) -----------------------------------------------
# ByteTrack ayni oyuncuya yeni numara verdiginde, kopru "az once yakinimda
# kaybolan ayni takimdan numara var mi" diye bakar ve eski numarayi geri baglar.
ID_BRIDGE_BASE_RADIUS_CM = 150   # Eslestirme icin taban yaricap
ID_BRIDGE_CM_PER_FRAME = 120     # Kayip gecen her islenen kare basina ek yaricap
                                 # (sprint hizi ~9 m/s -> 8 fps'te ~110 cm/kare)
ID_BRIDGE_MAX_RADIUS_CM = 1200   # Yaricap ust siniri (12 m)
HOMOGRAPHY_WINDOW = 5            # 5 karelik homografi matrisi ortalamasi
KEYPOINT_CONF_THRESHOLD = 0.5
DETECTION_CONF = 0.3
CROP_STRIDE = 30
PROCESS_STRIDE = 3               # Ana donguda kare atlama (1 = tum kareler)
BALL_OUTLIER_THRESHOLD_CM = 500  # Kare basina; PROCESS_STRIDE ile olceklenir
VORONOI_STRIDE = 5               # Kac islenen karede bir Voronoi izgarasi gonderilsin
VORONOI_GRID = (60, 40)          # (uzunluk, genislik) hucre sayisi
HEATMAP_GRID = (60, 40)

# --- TAKIPCI (TRACKER) AYARLARI ------------------------------------------------
TRACKER_BACKEND = "bytetrack"    # "bytetrack" (varsayilan) | "botsort" (gorunum
                                 # tabanli re-ID; pip install boxmot gerektirir)
TRACK_ACTIVATION_THRESHOLD = 0.30
LOST_TRACK_SECONDS = 4.0         # Iz kaybolunca ID bu kadar saniye hafizada tutulur;
                                 # kisa ortusmelerde oyuncu yeni numara ALMAZ.
MIN_CONSECUTIVE_FRAMES = 1       # 1 = filtre kapali. DIKKAT: 3 denendi ve geri
                                 # tepti: hizli kosan gercek oyuncular elendi,
                                 # sabit duran kulube/personel kaldi. Hayalet
                                 # tespitler artik saha siniri filtresiyle eleniyor.
PITCH_MARGIN_CM = 350            # Kusbakisi konumu saha disina bu payin otesinde
                                 # dusen tespitler (yedek kulubesi, kenar personeli,
                                 # tribun) radara ve istatistige ALINMAZ.
                                 # (200 idi; homografinin en az guvenilir oldugu tac
                                 # cizgisinde gercek oyunculari ara ara eliyordu.)

# --- HIZLANDIRMA PARAMETRELERI -----------------------------------------------
FIELD_DETECT_EVERY = 1           # 1 = her karede (dogru sonuc). 3 denendi:
                                 # kamera pan yaparken homografi geride kaldi ve
                                 # koordinatlar kaydi. Hiz icin artirilabilir ama
                                 # SADECE kameranin cok az hareket ettigi videolarda.
TEAM_VOTE_FRAMES = 5             # Bir oyuncunun takimi bu kadar oyla KESINLESIR ve
                                 # sonrasinda o oyuncu icin SigLIP bir daha CALISMAZ.
                                 # CPU'daki en buyuk maliyet kalemi budur.

# --- RENDER (MP4 cikti) AYARLARI ----------------------------------------------
RADAR_INSET_RATIO = 0.30         # Mini radarin video genisligine orani

# Render modunda kullanilan cizim araclari (egitmenin stilinde):
# oyuncular takim renginde elips, top sari ucgen, etiketler "#id".
ELLIPSE_ANNOTATOR = sv.EllipseAnnotator(
    color=sv.ColorPalette.from_hex(["#00BFFF", "#FF1493"]), thickness=2
)
LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(["#00BFFF", "#FF1493"]),
    text_color=sv.Color.from_hex("#000000"),
)
TRIANGLE_ANNOTATOR = sv.TriangleAnnotator(
    color=sv.Color.from_hex("#FFD700"), base=25, height=21, outline_thickness=1
)


# ----------------------------------------------------------------------------
# YARDIMCI FONKSIYONLAR
# ----------------------------------------------------------------------------
class PlayerTracker:
    """
    Takipci adaptoru. Iki arka uc destekler:

    - "bytetrack": supervision'in ByteTrack'i, ama STRIDE'A GORE AYARLANMIS.
      Kritik nokta: frame_rate'e videonun gercek fps'i degil, ISLENEN kare
      hizi (fps / PROCESS_STRIDE) verilir. Boylece Kalman tahmini, kareler
      arasi gercek hareket miktarini dogru varsayar ve ID'ler kopmaz.

    - "botsort": boxmot kutuphanesinin BoT-SORT'u. Harekete ek olarak oyuncunun
      GORUNUMUNE (re-ID embedding) bakar; ortusme sonrasi ayni oyuncuyu taniyip
      eski ID'sini geri verir. CPU'da kare basina ek maliyet getirir.
      Kurulum: pip install boxmot  (re-ID agirligi ilk calismada otomatik iner)
    """

    def __init__(self, effective_fps: float):
        self.backend = TRACKER_BACKEND
        if self.backend == "botsort":
            try:
                from pathlib import Path as _Path
                from boxmot import BotSort
            except ImportError as e:
                raise RuntimeError(
                    "TRACKER_BACKEND='botsort' icin once: pip install boxmot"
                ) from e
            self.impl = BotSort(
                reid_weights=_Path("osnet_x0_25_msmt17.pt"),
                device="cpu",
                half=False,
            )
        else:
            kwargs = dict(
                track_activation_threshold=TRACK_ACTIVATION_THRESHOLD,
                lost_track_buffer=int(LOST_TRACK_SECONDS * 30),
                minimum_matching_threshold=0.8,
                frame_rate=max(1, int(round(effective_fps))),
                minimum_consecutive_frames=MIN_CONSECUTIVE_FRAMES,
            )
            try:
                self.impl = sv.ByteTrack(**kwargs)
            except TypeError:
                # Eski supervision surumlerinde minimum_consecutive_frames yok
                kwargs.pop("minimum_consecutive_frames", None)
                self.impl = sv.ByteTrack(**kwargs)
            self.impl.reset()

    def update(self, detections: sv.Detections, frame: np.ndarray) -> sv.Detections:
        if self.backend != "botsort":
            return self.impl.update_with_detections(detections)

        if len(detections) == 0:
            detections.tracker_id = np.array([], dtype=int)
            return detections
        dets = np.column_stack(
            [detections.xyxy, detections.confidence, detections.class_id]
        ).astype(np.float32)
        out = self.impl.update(dets, frame)
        tracker_ids = np.full(len(detections), -1, dtype=int)
        if out is not None and len(out) > 0:
            for row in out:
                src = int(row[7])  # boxmot, girdi tespit indeksini son sutunda verir
                if 0 <= src < len(detections):
                    tracker_ids[src] = int(row[4])
        detections.tracker_id = tracker_ids
        return detections[tracker_ids != -1]


class StableIDMapper:
    """
    ID KOPRUSU: Ham takipci (ByteTrack/BoT-SORT) numaralarini KARARLI numaralara
    cevirir. ByteTrack bir oyuncuyu kaybedip yeni numara actiginda, kopru son
    LOST_TRACK_SECONDS icinde yakinlarda kaybolan AYNI TAKIMDAN bir kararli
    numara arar; bulursa eski numarayi geri verir. Boylece:
      - Radar ve MP4 etiketlerinde numaralar 90'lara firlamaz,
      - Pas motoru "yeni oyuncu" yanilgisiyla gercek paslari yutmaz.
    Tum mesafeler kusbakisi saha koordinatinda (cm) olculur.
    """

    def __init__(self, effective_fps: float):
        self.raw_to_stable: Dict[int, int] = {}
        # stable_id -> {"xy": son konum, "team": takim, "frame": son gorulme}
        self.last_seen: Dict[int, dict] = {}
        self.next_id = 1
        self.max_gap_frames = max(2, int(LOST_TRACK_SECONDS * effective_fps))

    def update(
        self,
        raw_ids: np.ndarray,
        pitch_xy: np.ndarray,
        teams: np.ndarray,
        frame_idx: int,
    ) -> np.ndarray:
        stable_ids = np.zeros(len(raw_ids), dtype=int)
        used_this_frame = set()

        for i, raw in enumerate(raw_ids):
            raw = int(raw)
            sid = self.raw_to_stable.get(raw)

            if sid is None:
                # Yeni ham numara: yakin zamanda kaybolan kararli numara ara
                best_sid, best_dist = None, float("inf")
                for cand, info in self.last_seen.items():
                    if cand in used_this_frame:
                        continue
                    gap = frame_idx - info["frame"]
                    if gap <= 0 or gap > self.max_gap_frames:
                        continue
                    if info["team"] != int(teams[i]):
                        continue
                    allowed = min(
                        ID_BRIDGE_MAX_RADIUS_CM,
                        ID_BRIDGE_BASE_RADIUS_CM + ID_BRIDGE_CM_PER_FRAME * gap,
                    )
                    dist = float(np.linalg.norm(pitch_xy[i] - info["xy"]))
                    if dist <= allowed and dist < best_dist:
                        best_sid, best_dist = cand, dist
                sid = best_sid if best_sid is not None else self.next_id
                if best_sid is None:
                    self.next_id += 1
                self.raw_to_stable[raw] = sid

            # Ayni karede iki tespite ayni kararli numara verilmesin
            if sid in used_this_frame:
                sid = self.next_id
                self.next_id += 1
                self.raw_to_stable[raw] = sid

            used_this_frame.add(sid)
            stable_ids[i] = sid
            self.last_seen[sid] = {
                "xy": pitch_xy[i].copy(),
                "team": int(teams[i]),
                "frame": frame_idx,
            }

        # Hafiza temizligi: cok eski kayitlari at
        stale = [s for s, inf in self.last_seen.items()
                 if frame_idx - inf["frame"] > self.max_gap_frames * 3]
        for s in stale:
            self.last_seen.pop(s, None)
        return stable_ids


def resolve_goalkeepers_team_id(
    players: sv.Detections, goalkeepers: sv.Detections
) -> np.ndarray:
    """Kaleciyi takim merkezine (centroid) yakinligina gore esler."""
    goalkeepers_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    team_0 = players_xy[players.class_id == 0]
    team_1 = players_xy[players.class_id == 1]
    if len(team_0) == 0 or len(team_1) == 0:
        return np.zeros(len(goalkeepers), dtype=int)
    c0, c1 = team_0.mean(axis=0), team_1.mean(axis=0)
    return np.array(
        [0 if np.linalg.norm(g - c0) < np.linalg.norm(g - c1) else 1
         for g in goalkeepers_xy]
    )


def compute_voronoi_grid(
    team_0_xy: np.ndarray, team_1_xy: np.ndarray
) -> Optional[np.ndarray]:
    """
    Voronoi sahiplik izgarasi - CIZIM YOK, sadece matematik.
    Her hucre icin 0 (Takim A daha yakin) veya 1 (Takim B daha yakin) dondurur.
    Sekil: (VORONOI_GRID[1], VORONOI_GRID[0]) yani satir=genislik, sutun=uzunluk.
    """
    if len(team_0_xy) == 0 or len(team_1_xy) == 0:
        return None
    gx = np.linspace(0, CONFIG.length, VORONOI_GRID[0])
    gy = np.linspace(0, CONFIG.width, VORONOI_GRID[1])
    grid = np.stack(np.meshgrid(gx, gy), axis=-1).reshape(-1, 2)

    d0 = np.min(np.linalg.norm(grid[:, None, :] - team_0_xy[None, :, :], axis=2), axis=1)
    d1 = np.min(np.linalg.norm(grid[:, None, :] - team_1_xy[None, :, :], axis=2), axis=1)
    return (d1 < d0).astype(int).reshape(VORONOI_GRID[1], VORONOI_GRID[0])


def heatmap_points_to_grid(points: List[List[float]]) -> List[List[int]]:
    """
    Ham (x, y) cm noktalarini HEATMAP_GRID boyutunda yogunluk izgarasina cevirir.
    Frontend bu izgarayi dogrudan renklendirebilir (deger = o hucreye dusen nokta).
    """
    grid = np.zeros((HEATMAP_GRID[1], HEATMAP_GRID[0]), dtype=int)
    if not points:
        return grid.tolist()
    pts = np.array(points)
    xs = np.clip((pts[:, 0] / CONFIG.length * HEATMAP_GRID[0]).astype(int), 0, HEATMAP_GRID[0] - 1)
    ys = np.clip((pts[:, 1] / CONFIG.width * HEATMAP_GRID[1]).astype(int), 0, HEATMAP_GRID[1] - 1)
    np.add.at(grid, (ys, xs), 1)
    return grid.tolist()


def fit_team_classifier(video_path: str) -> TeamClassifier:
    """Seyrek karelerden forma kirpintilari toplar, takim siniflandiriciyi egitir."""
    frame_generator = sv.get_video_frames_generator(source_path=video_path, stride=CROP_STRIDE)
    crops = []
    for frame in tqdm(frame_generator, desc="Takim siniflandirici egitimi"):
        result = PLAYER_DETECTION_MODEL.infer(frame, confidence=DETECTION_CONF)[0]
        detections = sv.Detections.from_inference(result)
        detections = detections.with_nms(threshold=0.5, class_agnostic=True)
        detections = detections[detections.class_id == PLAYER_ID]
        crops += [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]

    if len(crops) < 10:
        raise RuntimeError("Takim siniflandirmasi icin yeterli oyuncu tespiti yapilamadi.")

    classifier = TeamClassifier(device=DEVICE)
    classifier.fit(crops)
    return classifier


# ----------------------------------------------------------------------------
# ANA ANALIZ AKISI (Generator)
# Her islenen kare icin bir "kare" sozlugu uretir (yield), en sonda "ozet".
# Hem WebSocket canli akisi hem de tek atimlik /analyze ayni motoru kullanir.
# ----------------------------------------------------------------------------
def analyze_video_stream(
    video_path: str, render_path: Optional[str] = None
) -> Generator[dict, None, None]:
    """
    render_path verilirse: islenen her kare, uzerine elips/etiket/ucgen ve mini
    radar cizilerek render_path'teki MP4'e yazilir (VideoSink). Verilmezse
    sadece veri akisi uretilir (eski davranis).
    """
    # --- 0) Takim siniflandirici egitimi (uzun surer, frontend'i bilgilendir) -
    yield {"tip": "durum", "mesaj": "Takim siniflandirici egitiliyor...", "asama": 1, "toplam_asama": 2}
    team_classifier = fit_team_classifier(video_path)
    yield {"tip": "durum", "mesaj": "Video analizi basladi", "asama": 2, "toplam_asama": 2}

    video_info = sv.VideoInfo.from_video_path(video_path)
    fps = video_info.fps if video_info.fps and video_info.fps > 0 else 25

    # Takipci, ISLENEN kare hizina gore kurulur (stride'in ID patlatmasini onler)
    effective_fps = fps / PROCESS_STRIDE
    tracker = PlayerTracker(effective_fps)
    id_bridge = StableIDMapper(effective_fps)  # kararli kimlik koprusu
    frame_generator = sv.get_video_frames_generator(video_path, stride=PROCESS_STRIDE)

    # --- VideoSink: kaynak videonun cozunurlugunu kopyala, FPS'i stride'a bol.
    # Boylece cikti MP4'un SURESI orijinalle ayni kalir (3 karede 1 yazdigimiz
    # icin fps/3 ile oynatmak zamani dogru tutar). PROCESS_STRIDE=1 yaparsan
    # egitmenin anlattigi tam akici (orijinal fps) sonucu alirsin.
    sink: Optional[sv.VideoSink] = None
    if render_path:
        render_info = sv.VideoInfo(
            width=video_info.width,
            height=video_info.height,
            fps=max(1, round(fps / PROCESS_STRIDE)),
        )
        sink = sv.VideoSink(target_path=render_path, video_info=render_info)
        sink.__enter__()

    M_queue: deque = deque(maxlen=HOMOGRAPHY_WINDOW)

    # Istatistik durum degiskenleri
    possession_frames = {0: 0, 1: 0}
    passes = {0: 0, 1: 0}
    turnovers = {0: 0, 1: 0}
    heatmap_points = {0: [], 1: []}
    ball_path: List[List[float]] = []
    voronoi_samples: List[float] = []

    last_holder_tracker_id: Optional[int] = None
    last_holder_team: Optional[int] = None
    last_holder_xy: Optional[np.ndarray] = None
    candidate_tracker_id: Optional[int] = None
    candidate_frames = 0
    last_valid_ball_xy: Optional[np.ndarray] = None

    # HIZLANDIRMA 1: Takim karari onbellegi. tracker_id -> takim (0/1).
    # Bir oyuncunun takimi TEAM_VOTE_FRAMES oyla kesinlesince SigLIP o oyuncu
    # icin bir daha calismaz (formasi mac ortasinda degismeyecegine gore).
    team_cache: Dict[int, int] = {}
    team_votes: Dict[int, List[int]] = {}

    # HIZLANDIRMA 2: Homografi onbellegi. Saha keypoint modeli her karede degil
    # FIELD_DETECT_EVERY karede bir calisir; arada son matris kullanilir.
    cached_transformer: Optional[ViewTransformer] = None

    processed_total = max(1, video_info.total_frames // PROCESS_STRIDE)
    frame_idx = 0

    try:
        for frame in frame_generator:
            frame_idx += 1
            video_time_sec = round(frame_idx * PROCESS_STRIDE / fps, 2)

            # --- 1) Oyuncu + top tespiti -----------------------------------------
            result = PLAYER_DETECTION_MODEL.infer(frame, confidence=DETECTION_CONF)[0]
            detections = sv.Detections.from_inference(result)

            ball_detections = detections[detections.class_id == BALL_ID]
            person_detections = detections[detections.class_id != BALL_ID].with_nms(
                threshold=0.5, class_agnostic=True
            )

            # --- 2) Takip (PlayerTracker: ayarli ByteTrack veya BoT-SORT) ---------
            person_detections = tracker.update(person_detections, frame)

            goalkeepers = person_detections[person_detections.class_id == GOALKEEPER_ID]
            players = person_detections[person_detections.class_id == PLAYER_ID]

            if len(players) == 0:
                continue

            # --- 3) Takim atamasi - ONBELLEKLI (HIZLANDIRMA 1) ------------------
            # Takimi kesinlesmis oyuncular icin SigLIP CALISMAZ; sadece yeni/
            # oylamasi suren tracker_id'ler icin tahmin yapilir. CPU'daki en
            # buyuk maliyet kalemi bu oldugu icin kazanc cok buyuktur.
            tr_list = players.tracker_id
            assigned = np.zeros(len(players), dtype=int)
            pending = []  # henuz kesinlesmemis oyuncularin indeksleri
            for i, tid in enumerate(tr_list):
                tid = int(tid)
                if tid in team_cache:
                    assigned[i] = team_cache[tid]
                else:
                    pending.append(i)
            if pending:
                pend_crops = [sv.crop_image(frame, players.xyxy[i]) for i in pending]
                preds = team_classifier.predict(pend_crops)
                for i, pred in zip(pending, preds):
                    tid = int(tr_list[i])
                    votes = team_votes.setdefault(tid, [])
                    votes.append(int(pred))
                    assigned[i] = int(pred)
                    if len(votes) >= TEAM_VOTE_FRAMES:
                        # Cogunluk oyu ile kesinlestir, oylamayi kapat
                        team_cache[tid] = int(np.bincount(votes).argmax())
                        assigned[i] = team_cache[tid]
                        team_votes.pop(tid, None)
            players.class_id = assigned
            if len(goalkeepers) > 0:
                goalkeepers.class_id = resolve_goalkeepers_team_id(players, goalkeepers)
                players = sv.Detections.merge([players, goalkeepers])

            # --- 4) Homografi - SEYREK HESAP (HIZLANDIRMA 2) ---------------------
            # Saha keypoint modeli her FIELD_DETECT_EVERY karede bir calisir;
            # kamera yavas dondugu icin aradaki karelerde son matris yeterlidir.
            # 5 karelik M_queue ortalamasi (egitmenin yontemi) aynen korunur.
            if cached_transformer is None or (frame_idx - 1) % FIELD_DETECT_EVERY == 0:
                field_result = FIELD_DETECTION_MODEL.infer(frame, confidence=DETECTION_CONF)[0]
                key_points = sv.KeyPoints.from_inference(field_result)
                mask = key_points.confidence[0] > KEYPOINT_CONF_THRESHOLD
                if np.sum(mask) >= 4:
                    frame_ref = key_points.xy[0][mask]
                    pitch_ref = np.array(CONFIG.vertices)[mask]
                    new_transformer = ViewTransformer(source=frame_ref, target=pitch_ref)
                    M_queue.append(new_transformer.m)
                    new_transformer.m = np.mean(np.array(M_queue), axis=0)
                    cached_transformer = new_transformer
            if cached_transformer is None:
                continue  # Henuz hic gecerli homografi kurulamadi
            transformer = cached_transformer

            # --- 5) Kusbakisi koordinatlar -----------------------------------------
            pitch_players_xy = transformer.transform_points(
                points=players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            )
            team_ids = players.class_id
            tracker_ids = players.tracker_id

            # --- 5.5) SAHA SINIRI FILTRESI (hayalet tespit cozumu) ----------------
            # Kusbakisi konumu sahanin (+pay) disina dusen herkes elenir:
            # yedek kulubesi, kenar personeli, tribun tespitleri radari ve
            # istatistikleri kirletemez. Dogru filtre budur; ardisik-kare
            # filtresi (MIN_CONSECUTIVE_FRAMES=3) tam tersini yapiyordu.
            in_pitch = (
                (pitch_players_xy[:, 0] >= -PITCH_MARGIN_CM)
                & (pitch_players_xy[:, 0] <= CONFIG.length + PITCH_MARGIN_CM)
                & (pitch_players_xy[:, 1] >= -PITCH_MARGIN_CM)
                & (pitch_players_xy[:, 1] <= CONFIG.width + PITCH_MARGIN_CM)
            )
            if not np.any(in_pitch):
                continue
            players = players[in_pitch]
            pitch_players_xy = pitch_players_xy[in_pitch]
            team_ids = players.class_id
            tracker_ids = players.tracker_id

            # --- 5.6) ID KOPRUSU: ham numaralari kararli numaralara cevir ----------
            # Bundan sonraki her sey (pas motoru, radar paketi, MP4 etiketleri)
            # kararli numaralari kullanir; ByteTrack'in numara sicramalari
            # disariya yansimaz.
            if tracker_ids is not None and len(tracker_ids) > 0:
                tracker_ids = id_bridge.update(
                    tracker_ids, pitch_players_xy, team_ids, frame_idx
                )

            pitch_ball_xy = None
            if len(ball_detections) > 0:
                best = int(np.argmax(ball_detections.confidence))
                ball_anchor = ball_detections.get_anchors_coordinates(
                    sv.Position.BOTTOM_CENTER
                )[best : best + 1]
                candidate_ball_xy = transformer.transform_points(points=ball_anchor)[0]

                # Outlier temizleme: islenen iki kare arasinda top en fazla
                # esik * stride kadar yol alabilir; fazlasi hatali tespittir.
                if (
                    last_valid_ball_xy is None
                    or np.linalg.norm(candidate_ball_xy - last_valid_ball_xy)
                    <= BALL_OUTLIER_THRESHOLD_CM * PROCESS_STRIDE
                ):
                    pitch_ball_xy = candidate_ball_xy
                    last_valid_ball_xy = candidate_ball_xy
                    ball_path.append(
                        [round(float(pitch_ball_xy[0]), 1), round(float(pitch_ball_xy[1]), 1)]
                    )

            # --- 6) Isi haritasi birikimi -------------------------------------------
            for xy, tid in zip(pitch_players_xy, team_ids):
                heatmap_points[int(tid)].append(
                    [round(float(xy[0]), 1), round(float(xy[1]), 1)]
                )

            # --- 7) PAS / TOPA SAHIP OLMA / TOP KAYBI -------------------------------
            pass_event = None  # Bu karede pas/top kaybi olduysa frontend'e bildir
            if pitch_ball_xy is not None:
                distances = np.linalg.norm(pitch_players_xy - pitch_ball_xy, axis=1)
                nearest_idx = int(np.argmin(distances))

                if distances[nearest_idx] <= CONTROL_DISTANCE_CM:
                    nearest_team = int(team_ids[nearest_idx])
                    nearest_tracker = (
                        int(tracker_ids[nearest_idx]) if tracker_ids is not None else -1
                    )

                    possession_frames[nearest_team] += 1

                    if nearest_tracker == candidate_tracker_id:
                        candidate_frames += 1
                    else:
                        candidate_tracker_id = nearest_tracker
                        candidate_frames = 1

                    if (
                        candidate_frames >= MIN_CONTROL_FRAMES
                        and nearest_tracker != last_holder_tracker_id
                    ):
                        if last_holder_tracker_id is not None and last_holder_xy is not None:
                            travel = np.linalg.norm(
                                pitch_players_xy[nearest_idx] - last_holder_xy
                            )
                            if nearest_team == last_holder_team:
                                if travel >= MIN_PASS_DISTANCE_CM:
                                    passes[nearest_team] += 1
                                    pass_event = {
                                        "olay": "pas",
                                        "takim": nearest_team,
                                        "kimden": last_holder_tracker_id,
                                        "kime": nearest_tracker,
                                        "mesafe_cm": round(float(travel), 1),
                                    }
                            else:
                                turnovers[last_holder_team] += 1
                                pass_event = {
                                    "olay": "top_kaybi",
                                    "kaybeden_takim": last_holder_team,
                                    "kazanan_takim": nearest_team,
                                }
                        last_holder_tracker_id = nearest_tracker
                        last_holder_team = nearest_team
                        last_holder_xy = pitch_players_xy[nearest_idx].copy()
                    elif nearest_tracker == last_holder_tracker_id:
                        last_holder_xy = pitch_players_xy[nearest_idx].copy()

            # --- 8) Anlik istatistikler ----------------------------------------------
            total_poss = possession_frames[0] + possession_frames[1]
            poss_a = round(possession_frames[0] / total_poss * 100, 1) if total_poss else 0.0

            # --- 9) Voronoi izgarasi (toggle icin, seyrek gonder) ----------------------
            voronoi_payload = None
            if frame_idx % VORONOI_STRIDE == 0:
                vgrid = compute_voronoi_grid(
                    pitch_players_xy[team_ids == 0], pitch_players_xy[team_ids == 1]
                )
                if vgrid is not None:
                    voronoi_samples.append(float(np.mean(vgrid == 0) * 100.0))
                    voronoi_payload = vgrid.tolist()

            # --- 9.5) RENDER: islenmis kareyi MP4'e yaz (sadece render modunda) -----
            if sink is not None:
                annotated = frame.copy()
                # Oyuncular: takim renginde elips + "#id" etiketi
                annotated = ELLIPSE_ANNOTATOR.annotate(scene=annotated, detections=players)
                id_labels = [
                    f"#{int(t)}" if t is not None else "?"
                    for t in (tracker_ids if tracker_ids is not None else [])
                ]
                if id_labels:
                    annotated = LABEL_ANNOTATOR.annotate(
                        scene=annotated, detections=players, labels=id_labels
                    )
                # Top: sari ucgen (en guvenilir tek tespit, kutusu genisletilmis)
                if len(ball_detections) > 0:
                    best_ball = ball_detections[[int(np.argmax(ball_detections.confidence))]]
                    best_ball.xyxy = sv.pad_boxes(xyxy=best_ball.xyxy, px=10)
                    annotated = TRIANGLE_ANNOTATOR.annotate(
                        scene=annotated, detections=best_ball
                    )
                # Mini radar: kusbakisi saha, alt-ortaya yari saydam bindirilir
                radar = draw_pitch(CONFIG)
                for team, hexcol in ((0, "00BFFF"), (1, "FF1493")):
                    pts = pitch_players_xy[team_ids == team]
                    if len(pts) > 0:
                        radar = draw_points_on_pitch(
                            config=CONFIG, xy=pts,
                            face_color=sv.Color.from_hex(hexcol),
                            edge_color=sv.Color.BLACK, radius=16, pitch=radar,
                        )
                if pitch_ball_xy is not None:
                    radar = draw_points_on_pitch(
                        config=CONFIG, xy=np.array([pitch_ball_xy]),
                        face_color=sv.Color.WHITE, edge_color=sv.Color.BLACK,
                        radius=10, pitch=radar,
                    )
                fh, fw = annotated.shape[:2]
                rw = int(fw * RADAR_INSET_RATIO)
                rh = int(radar.shape[0] * rw / radar.shape[1])
                radar_small = cv2.resize(radar, (rw, rh))
                x0, y0 = (fw - rw) // 2, fh - rh - 12
                roi = annotated[y0:y0 + rh, x0:x0 + rw]
                annotated[y0:y0 + rh, x0:x0 + rw] = cv2.addWeighted(
                    radar_small, 0.75, roi, 0.25, 0
                )
                sink.write_frame(annotated)

            # --- 10) KARE MESAJI: WebSocket'in frontend'e firlatacagi paket ------------
            yield {
                "tip": "kare",
                "kare": frame_idx,
                "ilerleme_yuzde": round(frame_idx / processed_total * 100, 1),
                "video_zamani_saniye": video_time_sec,
                "oyuncular": [
                    {
                        "id": int(tid) if tid is not None else -1,
                        "takim": int(team),
                        "x": round(float(xy[0]), 1),
                        "y": round(float(xy[1]), 1),
                    }
                    for xy, team, tid in zip(
                        pitch_players_xy,
                        team_ids,
                        tracker_ids if tracker_ids is not None else [None] * len(team_ids),
                    )
                ],
                "top": (
                    {"x": round(float(pitch_ball_xy[0]), 1), "y": round(float(pitch_ball_xy[1]), 1)}
                    if pitch_ball_xy is not None
                    else None
                ),
                "topu_kontrol_eden": {
                    "oyuncu_id": last_holder_tracker_id,
                    "takim": last_holder_team,
                },
                "olay": pass_event,
                "istatistik": {
                    "takim_A": {"pas": passes[0], "top_kaybi": turnovers[0], "topa_sahip_olma_yuzde": poss_a},
                    "takim_B": {"pas": passes[1], "top_kaybi": turnovers[1], "topa_sahip_olma_yuzde": round(100 - poss_a, 1) if total_poss else 0.0},
                },
                "voronoi_izgara": voronoi_payload,  # null degilse radarin uzerine bindir
            }
    finally:
        # Render modunda MP4 dosyasini kapat/finalize et. Kullanici baglantiyi
        # kapatsa bile (GeneratorExit) buraya dusulur, dosya bozuk kalmaz.
        if sink is not None:
            sink.__exit__(None, None, None)

    # =========================================================================
    # DONGU BITTI -> NIHAI OZET
    # =========================================================================
    total_possession = possession_frames[0] + possession_frames[1]
    if total_possession > 0:
        poss_a = round(possession_frames[0] / total_possession * 100, 1)
        poss_b = round(100 - poss_a, 1)
    else:
        poss_a = poss_b = 0.0

    avg_control_a = round(float(np.mean(voronoi_samples)), 1) if voronoi_samples else None

    yield {
        "tip": "ozet",
        "video_bilgisi": {
            "toplam_kare": video_info.total_frames,
            "islenen_kare_atlama": PROCESS_STRIDE,
            "fps": fps,
            "sure_saniye": round(video_info.total_frames / fps, 1),
        },
        "takim_A": {
            "pas": passes[0],
            "top_kaybi": turnovers[0],
            "topa_sahip_olma_yuzde": poss_a,
            "topa_sahip_olma_saniye": round(possession_frames[0] * PROCESS_STRIDE / fps, 1),
            "alan_kontrolu_yuzde": avg_control_a,
        },
        "takim_B": {
            "pas": passes[1],
            "top_kaybi": turnovers[1],
            "topa_sahip_olma_yuzde": poss_b,
            "topa_sahip_olma_saniye": round(possession_frames[1] * PROCESS_STRIDE / fps, 1),
            "alan_kontrolu_yuzde": round(100 - avg_control_a, 1) if avg_control_a is not None else None,
        },
        "isi_haritasi": {
            # heatmap.js icin nokta listesi (cm cinsinden, frontend olcekler)
            "noktalar": {"takim_A": heatmap_points[0], "takim_B": heatmap_points[1]},
            # Canvas ile dogrudan boyamak icin 60x40 yogunluk izgarasi
            "izgara": {
                "takim_A": heatmap_points_to_grid(heatmap_points[0]),
                "takim_B": heatmap_points_to_grid(heatmap_points[1]),
                "boyut": {"sutun": HEATMAP_GRID[0], "satir": HEATMAP_GRID[1]},
            },
        },
        "top_yorungesi": ball_path,
        "islenmis_video": bool(render_path),
        "birimler": "Tum koordinatlar santimetre; saha 12000x7000 cm (SoccerPitchConfiguration)",
    }


def process_video(video_path: str, render_path: Optional[str] = None) -> dict:
    """Tek atimlik kullanim: akisi sonuna kadar tuketir, sadece ozeti dondurur."""
    summary = None
    for message in analyze_video_stream(video_path, render_path=render_path):
        if message["tip"] == "ozet":
            summary = message
    if summary is None:
        raise RuntimeError("Analiz tamamlanamadi: ozet uretilemedi.")
    return summary


# ----------------------------------------------------------------------------
# FASTAPI SUNUCUSU
# ----------------------------------------------------------------------------
app = FastAPI(
    title="FootballIQ AI Motoru",
    description="Video yukle, taktiksel analizi canli (WebSocket) veya toplu (JSON) al.",
    version="2.0.0",
)

# Frontend baska bir adresten (orn. dosyayi cift tiklayip file:// ile) acilirsa
# tarayicinin /upload istegini engellememesi icin CORS izni.
# Uretimde "*" yerine kendi site adresini yazmalisin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dashboard'i dogrudan backend'den sun: index.html'i main_api.py'nin yanina koy,
# tarayicidan http://localhost:8000/app adresini ac. Boylece CORS derdi de olmaz.
FRONTEND_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")


@app.get("/app")
def serve_dashboard():
    if not os.path.exists(FRONTEND_PATH):
        raise HTTPException(status_code=404, detail="index.html bulunamadi. Dosyayi main_api.py ile ayni klasore koy.")
    return FileResponse(FRONTEND_PATH)


# Yuklenen videolarin gecici kaydi: video_id -> dosya yolu
# Not: Bu basit sozluk tek sunucu/tek islem icin yeterlidir. Coklu worker'a
# gecersen (uvicorn --workers 2+) Redis gibi paylasimli bir depoya tasinmali.
UPLOADS: Dict[str, str] = {}

# Tamamlanmis islenmis (render) videolar: video_id -> mp4 yolu
RENDERS: Dict[str, str] = {}

# Islenmis videolar artik gecici klasore degil, script'in yanindaki kalici
# "footballiq_ciktilar" klasorune yazilir. Boylece tarayici indirmese bile
# dosya her zaman diskte hazir durur ve sunucu yeniden baslasa da kaybolmaz.
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "footballiq_ciktilar")
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")


def save_upload_to_temp(file: UploadFile) -> str:
    """Path Traversal'a karsi guvenli kayit: isim uuid, kullanicidan sadece uzanti."""
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Desteklenmeyen dosya formati.")
    temp_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4().hex}{ext}")
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return temp_path


@app.get("/")
def health_check():
    return {"durum": "calisiyor", "cihaz": DEVICE, "kare_atlama": PROCESS_STRIDE}


@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """
    Adim 1: Frontend videoyu buraya yukler, karsiliginda video_id alir.
    Adim 2: ws://sunucu/ws/analyze/{video_id} adresine baglanip akisi dinler.
    """
    temp_path = save_upload_to_temp(file)
    video_id = uuid.uuid4().hex
    UPLOADS[video_id] = temp_path
    return {"video_id": video_id, "websocket": f"/ws/analyze/{video_id}"}


@app.websocket("/ws/analyze/{video_id}")
async def ws_analyze(websocket: WebSocket, video_id: str):
    """
    CANLI AKIS: Video islenirken her kare paketini aninda gonderir.
    Agir yapay zeka isi (senkron generator) event loop'u kilitlemesin diye
    her adim asyncio.to_thread ile ayri is parcaciginda calistirilir.
    """
    await websocket.accept()

    video_path = UPLOADS.get(video_id)
    if video_path is None or not os.path.exists(video_path):
        await websocket.send_json({"tip": "hata", "mesaj": "Gecersiz veya suresi dolmus video_id."})
        await websocket.close()
        return

    # Render modu: frontend WS adresine ?render=1 eklerse, analizle birlikte
    # cizimli (elips + etiket + sari ucgen + mini radar) bir MP4 de uretilir.
    # Dosya, script'in yanindaki footballiq_ciktilar klasorune kalici yazilir.
    render_on = websocket.query_params.get("render") == "1"
    render_path = (
        os.path.join(
            OUTPUT_DIR,
            f"analiz_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{video_id[:8]}.mp4",
        )
        if render_on
        else None
    )

    stream = analyze_video_stream(video_path, render_path=render_path)
    try:
        while True:
            # Senkron generator'un bir adimini thread'de calistir:
            message = await asyncio.to_thread(next, stream, None)
            if message is None:
                break  # Akis bitti
            if message["tip"] == "ozet" and render_on:
                # MP4 hazir: indirilebilir adresi kaydet ve ozete ekle
                RENDERS[video_id] = render_path
                message["islenmis_video_url"] = f"/render/{video_id}"
                message["islenmis_video_dosya"] = render_path  # diskteki kalici yol
            await websocket.send_json(message)
        await websocket.close()
    except WebSocketDisconnect:
        # Kullanici sayfayi kapatti; islemeyi durdur, kaynaklari birak.
        stream.close()
    except Exception as e:
        # HERHANGI bir analiz hatasi: terminale tam izi bas, UI'a ozetini gonder.
        # (Onceden sadece RuntimeError yakalaniyordu; baska hatalar baglantiyi
        #  sessizce dusurup frontend'de "baglanti hatasi" olarak gorunuyordu.)
        traceback.print_exc()
        try:
            await websocket.send_json(
                {"tip": "hata", "mesaj": f"{type(e).__name__}: {e}"}
            )
            await websocket.close()
        except Exception:
            pass  # Baglanti zaten kopmussa yapacak bir sey yok
    finally:
        # Analiz bitti veya kesildi: gecici videoyu temizle
        UPLOADS.pop(video_id, None)
        if os.path.exists(video_path):
            os.remove(video_path)


@app.get("/render/{video_id}")
def get_rendered_video(video_id: str):
    """Islenmis (cizimli) MP4'u dondurur; tarayicida oynatilabilir veya indirilebilir."""
    path = RENDERS.get(video_id)
    if path is None or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Islenmis video bulunamadi veya henuz hazir degil.")
    return FileResponse(path, media_type="video/mp4", filename="footballiq_analiz.mp4")


@app.post("/open-output-folder")
def open_output_folder():
    """
    Cikti klasorunu isletim sisteminin dosya yoneticisinde acar.
    (Sunucu kullanicinin kendi bilgisayarinda calistigi icin guvenli; uzak
    sunucuya tasirsan bu endpoint'i kaldir.)
    """
    try:
        if os.name == "nt":  # Windows
            os.startfile(OUTPUT_DIR)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":  # macOS
            subprocess.Popen(["open", OUTPUT_DIR])
        else:  # Linux
            subprocess.Popen(["xdg-open", OUTPUT_DIR])
        return {"durum": "acildi", "klasor": OUTPUT_DIR}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Klasor acilamadi: {e}")


@app.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):
    """
    ESKI TEK ATIMLIK KULLANIM: Videoyu yukle, islem bitince ozeti JSON al.
    (Canli akis istemeyen basit senaryolar ve curl testleri icin.)
    Uzun videolarda HTTP zaman asimi riski vardir; uretimde WebSocket'i tercih et.
    """
    temp_path = save_upload_to_temp(file)
    try:
        # Agir senkron isi thread'e at ki sunucu baska istekleri de karsilayabilsin
        return await asyncio.to_thread(process_video, temp_path)
    except RuntimeError as e:
        raise HTTPException(status_code=422, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# ----------------------------------------------------------------------------
# Sunucusuz hizli test:  python main_api.py
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    kare_sayisi = 0
    for mesaj in analyze_video_stream("08fd33_4.mp4"):
        if mesaj["tip"] == "durum":
            print(f"[DURUM] {mesaj['mesaj']}")
        elif mesaj["tip"] == "kare":
            kare_sayisi += 1
            if mesaj["olay"]:
                print(f"[OLAY] kare {mesaj['kare']}: {mesaj['olay']}")
            if kare_sayisi % 25 == 0:
                ist = mesaj["istatistik"]
                print(f"[%{mesaj['ilerleme_yuzde']}] "
                      f"A: pas={ist['takim_A']['pas']} poss={ist['takim_A']['topa_sahip_olma_yuzde']}% | "
                      f"B: pas={ist['takim_B']['pas']} poss={ist['takim_B']['topa_sahip_olma_yuzde']}%")
        elif mesaj["tip"] == "ozet":
            ozet = {k: v for k, v in mesaj.items() if k not in ("isi_haritasi", "top_yorungesi")}
            print(json.dumps(ozet, indent=2, ensure_ascii=False))
            print(f"Isi haritasi nokta sayisi: A={len(mesaj['isi_haritasi']['noktalar']['takim_A'])}, "
                  f"B={len(mesaj['isi_haritasi']['noktalar']['takim_B'])}")