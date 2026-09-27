# ⚽ FootballIQ — Yapay Zeka ile Canlı Futbol Taktik Analizi

FootballIQ, yayın kamerasıyla çekilmiş bir futbol maçı videosunu kare kare işleyip
oyuncuları, topu ve hakemleri tespit eden, takımları otomatik ayıran ve sahayı kuşbakışı
bir **taktik radara** dönüştüren bir bilgisayarlı görü (computer vision) projesidir.

Analiz sonuçları video işlenirken **WebSocket** üzerinden tarayıcıdaki panele anlık
olarak akar; kullanıcı analizin bitmesini beklemeden istatistiklerin canlı değiştiğini görür.

---

## ✨ Özellikler

| Özellik | Açıklama |
|---|---|
| 🎯 **Nesne tespiti** | Oyuncu, kaleci, hakem ve top tespiti (Roboflow YOLO modeli) |
| 🧭 **Saha homografisi** | Saha çizgilerindeki anahtar noktalar bulunur, görüntü koordinatları gerçek saha koordinatlarına (cm) çevrilir |
| 👕 **Otomatik takım ayrımı** | Forma renkleri SigLIP + UMAP + K-Means ile kümelenir, kaleciler takım merkezine yakınlığa göre atanır |
| 🔢 **Kararlı oyuncu ID'leri** | ByteTrack (veya isteğe bağlı BoT-SORT) + kendi yazdığımız "ID köprüsü" ile oyuncu numaraları kopmaz |
| 📡 **Canlı radar** | Her karede oyuncu ve top konumları mini sahada gösterilir |
| 📊 **İstatistikler** | Topa sahip olma (% ve saniye), pas sayısı, top kaybı, alan kontrolü |
| 🟦🟥 **Voronoi diyagramı** | Sahanın hangi bölgesini hangi takımın kontrol ettiği |
| 🔥 **Isı haritası** | Takımların sahada en çok bulunduğu bölgeler (60×40 ızgara) |
| 〰️ **Top yörüngesi** | Topun izlediği yol; hatalı tespitler aykırı değer filtresiyle elenir |
| 🎬 **İşlenmiş video çıktısı** | Oyuncu elipsleri, ID etiketleri ve mini radar çizilmiş MP4 dosyası |

---

## 🏗️ Mimari

```
┌──────────────┐   POST /upload    ┌──────────────────────────────────────────┐
│              │ ────────────────► │  FastAPI sunucusu  (football_ai.py)      │
│  index.html  │                   │                                          │
│  (tarayıcı   │  WS /ws/analyze/  │  1. Takım sınıflandırıcı eğitimi         │
│   paneli)    │ ◄──────────────── │  2. Kare kare: tespit → takip →          │
│              │  "durum" / "kare" │     homografi → takım → istatistik       │
│              │  / "ozet" mesajı  │  3. Özet rapor + işlenmiş MP4            │
└──────────────┘                   └──────────────────────────────────────────┘
```

1. Video `POST /upload` ile yüklenir ve bir `video_id` döner.
2. Panel `ws://localhost:8000/ws/analyze/{video_id}` adresine bağlanır.
3. Sunucu üç tip mesaj gönderir:
   - `durum` — "Takım sınıflandırıcı eğitiliyor…" gibi ilerleme bilgisi
   - `kare` — her işlenen kare için oyuncu/top konumları, anlık istatistikler, Voronoi ızgarası
   - `ozet` — video bitince nihai rapor (istatistikler, ısı haritası, top yörüngesi)

---

## 📁 Proje Yapısı

```
.
├── football_ai.py              # Ana yapay zeka motoru + FastAPI sunucusu
├── index.html                  # Canlı analiz paneli (arayüz)
├── footballiq_logic.py         # Test edilebilir saf NumPy fonksiyonları
├── test_footballiq.py          # Birim testler (pytest)
├── benchmark.py                # Performans testi (FPS, süre, RAM)
├── test_videolarini_indir.py   # Örnek maç videolarını indirir
├── requirements.txt            # Python bağımlılıkları
└── .env.example                # API anahtarı şablonu
```

Çalışma sırasında oluşan `footballiq_ciktilar/` klasörü (işlenmiş videolar), video
dosyaları ve `.env` GitHub'a yüklenmez (`.gitignore`).

---

## 📚 Veri ve Model Kaynakları

Bu projede **kendi topladığımız bir veri seti yoktur**; tüm veriler ve önceden eğitilmiş
modeller aşağıdaki açık kaynaklardan alınmıştır.

### Maç videoları

| Kaynak | Açıklama |
|---|---|
| [roboflow/sports](https://github.com/roboflow/sports) | Roboflow'un açık kaynak spor analizi deposunun örnek maç videoları (`0bfacc_0.mp4`, `2e57b9_0.mp4`, `08fd33_0.mp4`, `573e61_0.mp4`, `121364_0.mp4`). Deponun Google Drive bağlantılarından `test_videolarini_indir.py` betiği ile indirilir. |
| [DFL – Bundesliga Data Shootout (Kaggle)](https://www.kaggle.com/competitions/dfl-bundesliga-data-shootout) | Roboflow örnek videolarının asıl kaynağı: Deutsche Fußball Liga'nın (DFL) Kaggle yarışması için yayımladığı Bundesliga maç kayıtlarından kesilmiş klipler. |

> Videolar yalnızca eğitim/araştırma ve demo amacıyla kullanılmıştır. Telif hakları
> ilgili sahiplerine (DFL) aittir ve bu depoda paylaşılmaz.

### Önceden eğitilmiş modeller

| Model | Kaynak | Kullanım amacı |
|---|---|---|
| `football-players-detection-3zvbc/11` | [Roboflow Universe](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc) | Oyuncu, kaleci, hakem ve top tespiti |
| `football-field-detection-f07vi/14` | [Roboflow Universe](https://universe.roboflow.com/roboflow-jvuqo/football-field-detection-f07vi) | Saha anahtar noktaları (homografi için 32 nokta) |
| `google/siglip-base-patch16-224` | [Hugging Face](https://huggingface.co/google/siglip-base-patch16-224) | Forma görüntülerinden özellik çıkarımı (`sports` kütüphanesindeki `TeamClassifier` içinde) |
| `osnet_x0_25_msmt17` | [BoxMOT](https://github.com/mikel-brostrom/boxmot) | Yalnızca BoT-SORT takipçisi seçilirse, oyuncuyu görünümünden yeniden tanıma (re-ID) |

Modeller Roboflow Inference API'si üzerinden ilk çalıştırmada otomatik indirilir.

### Kullanılan kütüphaneler

- [Supervision](https://github.com/roboflow/supervision) — tespit, ByteTrack takibi ve çizim araçları
- [roboflow/sports](https://github.com/roboflow/sports) — saha konfigürasyonu, `ViewTransformer`, `TeamClassifier`
- [Roboflow Inference](https://github.com/roboflow/inference) — modellerin çalıştırılması
- [FastAPI](https://fastapi.tiangolo.com/) + Uvicorn — REST ve WebSocket sunucusu
- OpenCV, NumPy, PyTorch, Transformers, UMAP, scikit-learn

---

## 🚀 Kurulum

**Gereksinimler:** Python 3.12, ücretsiz bir [Roboflow](https://app.roboflow.com) hesabı.
Proje CPU üzerinde çalışacak şekilde ayarlanmıştır; GPU zorunlu değildir.

```bash
# 1. Depoyu klonla
git clone https://github.com/EsraErgunn/asensio.git
cd asensio

# 2. Sanal ortam oluştur
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 3. Bağımlılıkları kur
pip install -r requirements.txt
```

### API anahtarı

`.env.example` dosyasını `.env` adıyla kopyala ve kendi anahtarını yaz:

```env
ROBOFLOW_API_KEY=senin_roboflow_anahtarin
HF_TOKEN=                        # opsiyonel
```

Roboflow anahtarını **Roboflow → Settings → API Keys** sayfasından alabilirsin.
`.env` dosyası `.gitignore` içinde olduğu için GitHub'a yüklenmez.

### Örnek videoları indir

```bash
pip install gdown
python test_videolarini_indir.py
```

---

## ▶️ Kullanım

### Web paneli ile (önerilen)

```bash
uvicorn football_ai:app --host 0.0.0.0 --port 8000
```

Tarayıcıda **http://localhost:8000/app** adresini aç, "Video yükle ve analiz et"
butonuyla bir video seç. Radar ve istatistikler analiz sürerken canlı olarak güncellenir.
Panelde **Voronoi**, **Isı haritası** ve **Top yörüngesi** katmanları açılıp kapatılabilir.
Analiz bitince işlenmiş video `footballiq_ciktilar/` klasörüne kaydedilir.

### Terminalden (sunucusuz)

```bash
python football_ai.py
```

`football_ai.py` sonundaki video yolunu kendi videona göre değiştirebilirsin.

### API uç noktaları

| Yöntem | Yol | Açıklama |
|---|---|---|
| `GET` | `/app` | Analiz panelini açar |
| `GET` | `/` | Sağlık kontrolü |
| `POST` | `/upload` | Video yükler, `video_id` döner |
| `WS` | `/ws/analyze/{video_id}` | Canlı analiz akışı |
| `GET` | `/render/{video_id}` | İşlenmiş MP4 videosunu indirir |
| `POST` | `/analyze` | Tek seferlik analiz (WebSocket olmadan, kısa videolar için) |

---

## ⚙️ Önemli Ayarlar

`football_ai.py` başındaki sabitlerle analiz davranışı değiştirilebilir:

| Parametre | Varsayılan | Açıklama |
|---|---|---|
| `PROCESS_STRIDE` | `3` | Kaç karede bir işlem yapılacağı (1 = tüm kareler, daha yavaş ama daha akıcı) |
| `TRACKER_BACKEND` | `"bytetrack"` | `"botsort"` seçilirse görünüm tabanlı re-ID kullanılır (`pip install boxmot`) |
| `CONTROL_DISTANCE_CM` | `200` | Top bu mesafedeyse oyuncu topu kontrol ediyor sayılır |
| `MIN_PASS_DISTANCE_CM` | `300` | Bundan kısa top hareketleri pas sayılmaz |
| `PITCH_MARGIN_CM` | `350` | Saha dışındaki tespitler (yedek kulübesi, tribün) elenir |
| `DETECTION_CONF` | `0.3` | Tespit güven eşiği |

---

## 🧪 Testler ve Performans

```bash
# Birim testler (14 test: ısı haritası, Voronoi, kaleci ataması, top filtresi)
pip install pytest
pytest test_footballiq.py -v

# Performans testi: farklı PROCESS_STRIDE değerleri için FPS, süre ve RAM ölçer
pip install psutil
python benchmark.py 08fd33_0.mp4 --strides 1 3 5
```

---

## 🙏 Teşekkür

Bu proje, [Roboflow](https://roboflow.com)'un açık kaynak
[sports](https://github.com/roboflow/sports) deposu ve futbol analizi eğitim içerikleri
temel alınarak geliştirilmiştir. Canlı WebSocket akışı, kararlı ID köprüsü, pas/top kaybı
istatistikleri, Voronoi ve ısı haritası katmanları ile web paneli bu projeye eklenen
özelliklerdir.
