import time
import csv
from datetime import datetime
from pathlib import Path

import sys

import numpy as np
import torch
import psutil
try:
    import GPUtil
except ImportError:
    GPUtil = None

import gradio as gr
from PIL import Image

import webbrowser
import threading

MODEL_NAME = "RealESRGAN_x4plus"
BASE_DIR = Path(__file__).resolve().parent

REALESRGAN_DIR = BASE_DIR / "Real-ESRGAN"
WEIGHTS = BASE_DIR / "weights" / "RealESRGAN_x4plus.pth"

LOG_PATH = Path("logs") / "perf_log.csv"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Import von Real-ESRGAN
sys.path.append(str(REALESRGAN_DIR))

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer


rrdbnet = RRDBNet(
    num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4
)

upsampler = RealESRGANer(
    scale=4,
    model_path=str(WEIGHTS),
    model=rrdbnet,
    tile=0,
    tile_pad=10,
    pre_pad=0,
    half=torch.cuda.is_available(),
    device=torch.device(device),
)

# Hilfsfunktionen für Begrenzung der Bildgröße
def get_vram_mb():
    if not torch.cuda.is_available():
        return None
    props = torch.cuda.get_device_properties(0)
    return props.total_memory / (1024 * 1024)

def choose_max_side(vram_mb: float | None) -> int:
    if vram_mb is None:
        return 640
    if vram_mb < 4500:
        return 640
    if vram_mb < 7000:
        return 960
    return 1280

def resize_max_side(image: Image.Image, max_side: int) -> Image.Image:
    w, h = image.size
    m = max(w, h)
    if m <= max_side:
        return image
    scale = max_side / m
    new_w = int(w * scale)
    new_h = int(h * scale)
    return image.resize((new_w, new_h))

# Leistungsmetriken
def get_gpu_metrics():
    data = {
        "gpu_name": None,
        "gpu_load_pct": None,
        "gpu_temp_c": None,
        "gpu_mem_used_mb": None,
        "gpu_mem_total_mb": None,
    }
    if GPUtil is None:
        return data

    gpus = GPUtil.getGPUs()
    if not gpus:
        return data

    gpu = gpus[0]
    data.update({
        "gpu_name": gpu.name,
        "gpu_load_pct": round(gpu.load * 100, 1),
        "gpu_temp_c": gpu.temperature,
        "gpu_mem_used_mb": round(gpu.memoryUsed, 1),
        "gpu_mem_total_mb": round(gpu.memoryTotal, 1),
    })
    return data


def append_log_row(row: dict):
    write_header = not LOG_PATH.exists()

    fieldnames = list(row.keys())
    with LOG_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# Bildverbesserung
def enhance(pil_img: Image.Image) -> Image.Image:
    if pil_img is None:
        raise gr.Error("Bitte ein Bild hochladen.")

    # Limit vom input für vram da output ~4x größer
    vram_mb = get_vram_mb()
    max_side = choose_max_side(vram_mb)
    pil_img = resize_max_side(pil_img, max_side)

    # PIL -> numpy (RGB)
    img_rgb = np.array(pil_img.convert("RGB"))

    ram = psutil.virtual_memory()
    ram_used_mb = (ram.total - ram.available) / (1024 * 1024)
    ram_total_mb = ram.total / (1024 * 1024)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start = time.perf_counter()

    img_bgr = img_rgb[:, :, ::-1]
    out_bgr, _ = upsampler.enhance(img_bgr, outscale=4)
    out_rgb = out_bgr[:, :, ::-1]

    if device == "cuda":
        torch.cuda.synchronize()

    infer_ms = (time.perf_counter() - start) * 1000
    out_pil = Image.fromarray(out_rgb)

    vram_peak_mb = None
    if device == "cuda":
        vram_peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    log_row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "device": device,
        "model": MODEL_NAME,
        "infer_ms": round(infer_ms, 2),
        "input_w": pil_img.size[0],
        "input_h": pil_img.size[1],
        "output_w": out_pil.size[0],
        "output_h": out_pil.size[1],
        "vram_peak_mb": round(vram_peak_mb, 1) if vram_peak_mb is not None else None,
        "ram_used_mb": round(ram_used_mb, 1),
        "ram_total_mb": round(ram_total_mb, 1),
        **get_gpu_metrics(),
    }
    append_log_row(log_row)

    return out_pil

# Interface mit Gradio
with gr.Blocks(title="Bildverbesserung (Real-ESRGAN)") as demo:
    gr.Markdown(
        """
# Bildverbesserung (Real-ESRGAN)
**So funktioniert's:** Bild hochladen → **Verbessern (x4)** → Ergebnis speichern.

Hinweis: Große Bilder brauchen viel RAM/VRAM (weil das Ergebnis ~4× größer wird).
"""
    )

    with gr.Row():
        inp = gr.Image(type="pil", label="Eingabebild")
        out = gr.Image(type="pil", label="Ergebnis")

    btn = gr.Button("Verbessern")
    btn.click(fn=enhance, inputs=inp, outputs=out)

    gr.Markdown("Hinweis: Bitte keine personenbezogenen Bilder verwenden, sofern keine Einwilligung vorliegt.")

def open_browser():
    webbrowser.open("http://127.0.0.1:7862")

if __name__ == "__main__":
    threading.Timer(1.0, open_browser).start()
    demo.launch(server_name="127.0.0.1", server_port=7862, inbrowser=False, share=False, prevent_thread_lock=False)