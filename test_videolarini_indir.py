# -*- coding: utf-8 -*-
"""
test_videolarini_indir.py
=========================
Egitmenin (Roboflow sports deposu) kullandigi ornek mac videolarini indirir.
Bu videolar herkese acik ve telif acisindan ornek/demo amaclidir.

Kullanim:
    pip install gdown
    python test_videolarini_indir.py

Videolar bu script'in bulundugu klasore iner. Inince football_ai.py icindeki
SOURCE_VIDEO_PATH'i indirdigin dosyayla degistirebilir veya dashboard'dan
dogrudan yukleyebilirsin.
"""

import os
import sys

try:
    import gdown
except ImportError:
    print("gdown kurulu degil. Once su komutu calistir:")
    print("    pip install gdown")
    sys.exit(1)

# Roboflow sports deposunun ornek videolari (Google Drive ID'leri).
# Kaynak: github.com/roboflow/sports  (acik kaynak, ornek veri)
VIDEOLAR = {
    "0bfacc_0.mp4": "12TqauVZ9tLAv8kWxTTBFWtgt2hNQ4_ZF",
    "2e57b9_0.mp4": "19PGw55V8aA6GZu5-Aac5_9mCy3fNxmEf",
    "08fd33_0.mp4": "1OG8K6wqUw9t7lp9ms1M48DxRhwTYciK-",
    "573e61_0.mp4": "1yYesfTQNAObltGjF54V1AyfssVGN70k7",
    "121364_0.mp4": "1vVwjW1dE1drIdd4ZSILfbCGPD4weoNiu",
}

print(f"{len(VIDEOLAR)} video indirilecek...\n")

for dosya_adi, drive_id in VIDEOLAR.items():
    if os.path.exists(dosya_adi):
        print(f"[ATLANDI] {dosya_adi} zaten var")
        continue
    print(f"[INIYOR]  {dosya_adi} ...")
    try:
        url = f"https://drive.google.com/uc?id={drive_id}"
        gdown.download(url, dosya_adi, quiet=False)
        print(f"[TAMAM]   {dosya_adi}\n")
    except Exception as e:
        print(f"[HATA]    {dosya_adi} indirilemedi: {e}\n")

print("Bitti. Inen dosyalar bu klasorde:")
for dosya_adi in VIDEOLAR:
    if os.path.exists(dosya_adi):
        boyut_mb = os.path.getsize(dosya_adi) / (1024 * 1024)
        print(f"  - {dosya_adi} ({boyut_mb:.1f} MB)")
