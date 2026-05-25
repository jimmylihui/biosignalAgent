from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps, ImageFilter

OUT = Path('/data1/jiahui/biosignal-agent/outputs/axis_ocr_compare')
OUT.mkdir(parents=True, exist_ok=True)


def make_plot(path: Path, dpi: int, fontsize: int, noise: float = 0.0, downscale: float = 1.0) -> None:
    t = np.linspace(0, 10, 1000)
    y = np.sin(2 * np.pi * 1.2 * t)
    fig, ax = plt.subplots(figsize=(8, 2.8), dpi=dpi)
    ax.plot(t, y, color='black', lw=1.4)
    ax.set_xlim(0, 10)
    ax.set_ylim(-1.2, 1.2)
    ax.set_xlabel('Time (s)', fontsize=fontsize)
    ax.set_xticks([0, 2, 4, 6, 8, 10])
    ax.tick_params(axis='both', labelsize=fontsize)
    ax.grid(True, alpha=.35)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    img = Image.open(path).convert('RGB')
    if downscale != 1.0:
        img = img.resize((max(1, int(img.width * downscale)), max(1, int(img.height * downscale))), Image.Resampling.BICUBIC)
    if noise > 0:
        arr = np.asarray(img).astype(np.float32)
        rng = np.random.default_rng(13)
        arr += rng.normal(0, noise, arr.shape)
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    img.save(path)


def axis_crop(src: Path, name: str, scale: int = 4) -> Path:
    img = Image.open(src).convert('RGB')
    w, h = img.size
    crop = img.crop((0, int(h * 0.48), w, h)).convert('L')
    crop = ImageOps.autocontrast(crop)
    crop = crop.filter(ImageFilter.SHARPEN)
    crop = crop.resize((crop.width * scale, crop.height * scale), Image.Resampling.LANCZOS)
    out = OUT / f'{name}_xaxis_x{scale}.png'
    crop.convert('RGB').save(out)
    return out


def numbers(text: str) -> list[float]:
    vals = []
    for tok in re.findall(r'[-+]?\d+(?:\.\d+)?', text):
        try:
            vals.append(float(tok))
        except ValueError:
            pass
    return vals


def infer_duration(vals: list[float]) -> tuple[float | None, str]:
    if len(vals) < 2:
        return None, 'too_few_numbers'
    uniq = sorted(set(vals))
    if len(uniq) > 20:
        return None, 'too_many_numbers_possible_hallucination'
    span = uniq[-1] - uniq[0]
    if not (0.1 <= span <= 3600):
        return None, 'span_out_of_range'
    return float(span), 'ok'


def ocr_tesseract(path: Path) -> str:
    best = ''
    for psm in ['6', '7', '11', '12']:
        proc = subprocess.run(['tesseract', str(path), 'stdout', '--psm', psm, '-c', 'tessedit_char_whitelist=0123456789.-+sSmM:/Time() '], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15)
        txt = proc.stdout.strip()
        if len(numbers(txt)) > len(numbers(best)):
            best = txt
    return best


def ocr_rapid(path: Path) -> str:
    from rapidocr_onnxruntime import RapidOCR
    engine = RapidOCR()
    result, _ = engine(str(path))
    if not result:
        return ''
    return ' '.join(str(item[1]) for item in result)


def ocr_easy(path: Path) -> str:
    import easyocr
    reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    result = reader.readtext(str(path), detail=0, paragraph=False, allowlist='0123456789.-+sSmM:/Time() ')
    return ' '.join(str(x) for x in result)


def main() -> None:
    plots = []
    specs = [
        ('synthetic_clear', 120, 10, 0, 1.0, 10.0),
        ('synthetic_small_font', 90, 7, 0, 1.0, 10.0),
        ('synthetic_lowres_noisy', 90, 7, 7.0, 0.55, 10.0),
    ]
    for name, dpi, fs, noise, downscale, expected in specs:
        img = OUT / f'{name}.png'
        make_plot(img, dpi, fs, noise=noise, downscale=downscale)
        plots.append({'name': name, 'source': str(img), 'crop': str(axis_crop(img, name)), 'expected_duration_s': expected})
    benchmarks = {
        'benchmark_ppg_grid': '/data1/jiahui/biosignal-agent/datasets/processed/digitization_benchmark/images/ppg_01_bidmc02_grid.png',
        'benchmark_scg_clean_more': '/data1/jiahui/biosignal-agent/datasets/processed/digitization_benchmark_more_10s/images/scg_00_b001_clean.png',
    }
    for name, src in benchmarks.items():
        crop = axis_crop(Path(src), name)
        plots.append({'name': name, 'source': src, 'crop': str(crop), 'expected_duration_s': None})

    engines = [('tesseract', ocr_tesseract), ('rapidocr', ocr_rapid), ('easyocr', ocr_easy)]
    rows = []
    for item in plots:
        crop = Path(item['crop'])
        for engine_name, fn in engines:
            try:
                text = fn(crop)
                vals = numbers(text)
                dur, status = infer_duration(vals)
                err = None if item['expected_duration_s'] is None or dur is None else abs(dur - item['expected_duration_s'])
                rows.append({**item, 'engine': engine_name, 'text': text, 'numbers': vals, 'duration_s': dur, 'status': status, 'duration_abs_error_s': err})
            except Exception as exc:
                rows.append({**item, 'engine': engine_name, 'text': '', 'numbers': [], 'duration_s': None, 'status': f'error:{type(exc).__name__}:{str(exc)[:120]}', 'duration_abs_error_s': None})
    summary = {}
    for engine_name, _ in engines:
        sub = [r for r in rows if r['engine'] == engine_name]
        synth = [r for r in sub if r['expected_duration_s'] is not None]
        false_positive = [r for r in sub if r['expected_duration_s'] is None and r['duration_s'] is not None]
        ok = [r for r in synth if r['duration_abs_error_s'] is not None and r['duration_abs_error_s'] <= 0.25]
        summary[engine_name] = {
            'synthetic_correct': len(ok),
            'synthetic_total': len(synth),
            'benchmark_false_positive_duration': len(false_positive),
            'statuses': {s: sum(1 for r in sub if r['status'] == s) for s in sorted(set(r['status'] for r in sub))},
        }
    report = {'plots': plots, 'summary': summary, 'rows': rows}
    out = OUT / 'axis_ocr_compare_results.json'
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({'out_json': str(out), 'summary': summary, 'rows': rows}, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    main()
