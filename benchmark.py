# -*- coding: utf-8 -*-
"""
benchmark.py
============
FootballIQ PERFORMANS TESTI (System Test Design - Performance Tests)

Bu betik bir videoyu farkli PROCESS_STRIDE degerleriyle isler ve her biri icin
GERCEK su olculeri verir:
    - islenen kare sayisi
    - toplam sure (saniye)
    - FPS (saniyede islenen kare)
    - RAM kullanimi (MB)

Sonucu hem terminale hem de tez icin hazir bir tablo olarak yazar.

KULLANIM:
    # main_api.py ile AYNI klasore koy.
    pip install psutil
    python benchmark.py 08fd33_4.mp4

    # Tek bir stride denemek istersen:
    python benchmark.py 08fd33_4.mp4 --strides 3

NOT: Bu betik main_api.py'deki analyze_video_stream fonksiyonunu kullanir.
PROCESS_STRIDE bir modul sabiti oldugu icin, her denemede modul icindeki
degeri gecici olarak degistiririz.
"""

import argparse
import time

try:
    import psutil
except ImportError:
    psutil = None  # RAM olcumu opsiyonel; yoksa "-" yazariz

# main_api.py'yi modul olarak iceri al
import football_ai as main_api


def get_ram_mb() -> float:
    """O anki islemin RAM kullanimini MB cinsinden dondurur."""
    if psutil is None:
        return -1.0
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)


def run_one_stride(video_path: str, stride: int) -> dict:
    """
    Videoyu tek bir stride degeriyle bastan sona isler, olculeri dondurur.
    Akistaki 'kare' mesajlarini sayar; agir cizim (render) YAPILMAZ, sadece
    saf analiz hizi olculur.
    """
    # main_api icindeki global PROCESS_STRIDE'i gecici degistir
    main_api.PROCESS_STRIDE = stride

    frame_count = 0
    peak_ram = 0.0

    start = time.perf_counter()
    for message in main_api.analyze_video_stream(video_path, render_path=None):
        if message["tip"] == "kare":
            frame_count += 1
            # Her 25 karede bir RAM olc (cok sik olcmek yavaslatir)
            if frame_count % 25 == 0:
                peak_ram = max(peak_ram, get_ram_mb())
    elapsed = time.perf_counter() - start

    fps = frame_count / elapsed if elapsed > 0 else 0.0
    return {
        "stride": stride,
        "frames": frame_count,
        "time_s": round(elapsed, 1),
        "fps": round(fps, 2),
        "ram_mb": round(peak_ram, 1) if peak_ram > 0 else -1.0,
    }


def print_markdown_table(results: list) -> None:
    """Tez icin hazir Markdown tablosu yazar."""
    print("\n" + "=" * 60)
    print("TEZ ICIN HAZIR TABLO (Markdown):")
    print("=" * 60)
    print("| Stride | Frames read | Time (s) | FPS | RAM (MB) |")
    print("|--------|-------------|----------|-----|----------|")
    for r in results:
        ram = r["ram_mb"] if r["ram_mb"] > 0 else "-"
        print(f"| {r['stride']} | {r['frames']} | {r['time_s']} | {r['fps']} | {ram} |")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="FootballIQ performans testi")
    parser.add_argument("video", help="Test edilecek video dosyasi (orn. 08fd33_4.mp4)")
    parser.add_argument(
        "--strides",
        type=int,
        nargs="+",
        default=[1, 3, 5],
        help="Denenecek PROCESS_STRIDE degerleri (varsayilan: 1 3 5)",
    )
    args = parser.parse_args()

    print(f"Video: {args.video}")
    print(f"Denenecek stride degerleri: {args.strides}")
    if psutil is None:
        print("UYARI: psutil kurulu degil, RAM olcumu atlanacak. (pip install psutil)")

    results = []
    for stride in args.strides:
        print(f"\n--- Stride {stride} ile isleniyor... (biraz surebilir) ---")
        try:
            r = run_one_stride(args.video, stride)
            results.append(r)
            print(
                f"  Bitti: {r['frames']} kare, {r['time_s']} s, "
                f"{r['fps']} FPS, RAM {r['ram_mb']} MB"
            )
        except Exception as e:
            print(f"  HATA (stride {stride}): {type(e).__name__}: {e}")

    if results:
        print_markdown_table(results)


if __name__ == "__main__":
    main()