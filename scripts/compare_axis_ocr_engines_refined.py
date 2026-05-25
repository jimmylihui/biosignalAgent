from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.compare_axis_ocr_engines import OUT, make_plot


def make_crop(src: Path, name: str, y0_frac: float = 0.48, scale: int = 4) -> Path:
    img = Image.open(src).convert('RGB')
    w, h = img.size
    crop = img.crop((0, int(h * y0_frac), w, h)).convert('L')
    crop = ImageOps.autocontrast(crop)
    crop = crop.filter(ImageFilter.SHARPEN)
    crop = crop.resize((crop.width * scale, crop.height * scale), Image.Resampling.LANCZOS)
    out = OUT / f'{name}_xaxis_y{int(y0_frac*100)}_x{scale}.png'
    crop.convert('RGB').save(out)
    return out


def nums(text: str) -> list[float]:
    out=[]
    for tok in re.findall(r'[-+]?\d+(?:\.\d+)?', text):
        try: out.append(float(tok))
        except ValueError: pass
    return out


def duration_from(vals: list[float]) -> tuple[float | None, str]:
    vals=sorted(set(vals))
    if len(vals)<2: return None,'too_few_numbers'
    if len(vals)>12: return None,'too_many_numbers'
    span=vals[-1]-vals[0]
    if span<=0 or span>120: return None,'span_out_of_range'
    return float(span),'ok'


def rapid_boxes(path: Path):
    from rapidocr_onnxruntime import RapidOCR
    res,_=RapidOCR()(str(path))
    return res or []


def easy_boxes(path: Path):
    import easyocr
    reader=easyocr.Reader(['en'], gpu=False, verbose=False)
    res=reader.readtext(str(path), detail=1, paragraph=False, allowlist='0123456789.-+sSmM:/Time() ')
    return [(box,text,conf) for box,text,conf in res]


def box_center_y(box) -> float:
    return float(np.mean([p[1] for p in box]))


def box_text_numbers_bottom(boxes, image_height: int) -> tuple[str, list[float]]:
    # Keep numeric boxes in the lower x-axis band; this drops y-axis tick labels.
    picked=[]
    for box,text,conf in boxes:
        text=str(text).strip()
        if not nums(text):
            continue
        cy=box_center_y(box)
        if cy >= image_height * 0.55:
            picked.append((float(np.mean([p[0] for p in box])), text, float(conf)))
    picked.sort(key=lambda x:x[0])
    text=' '.join(t for _,t,_ in picked)
    return text, nums(text)


def tesseract_text(path: Path) -> str:
    # Use an extra lower crop for Tesseract because it has no boxes.
    img=Image.open(path).convert('RGB')
    w,h=img.size
    lower=img.crop((0,int(h*0.50),w,h))
    tmp=OUT/'_tesseract_lower_tmp.png'
    lower.save(tmp)
    best=''
    for psm in ['6','7','11','12']:
        p=subprocess.run(['tesseract',str(tmp),'stdout','--psm',psm],text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=15)
        txt=p.stdout.strip()
        if len(nums(txt))>len(nums(best)): best=txt
    return best


def main():
    tests=[]
    for name,dpi,fs,noise,down,expected in [
        ('synthetic_clear',120,10,0,1.0,10.0),
        ('synthetic_small_font',90,7,0,1.0,10.0),
        ('synthetic_lowres_noisy',90,7,7.0,0.55,10.0),
    ]:
        img=OUT/f'{name}_refined.png'
        make_plot(img,dpi,fs,noise=noise,downscale=down)
        crop=make_crop(img,name)
        tests.append({'name':name,'source':str(img),'crop':str(crop),'expected_duration_s':expected})
    for name,src in {
        'benchmark_ppg_grid':'/data1/jiahui/biosignal-agent/datasets/processed/digitization_benchmark/images/ppg_01_bidmc02_grid.png',
        'benchmark_scg_clean_more':'/data1/jiahui/biosignal-agent/datasets/processed/digitization_benchmark_more_10s/images/scg_00_b001_clean.png',
    }.items():
        crop=make_crop(Path(src),name)
        tests.append({'name':name,'source':src,'crop':str(crop),'expected_duration_s':None})
    rows=[]
    for t in tests:
        path=Path(t['crop'])
        h=Image.open(path).height
        # Tesseract
        try:
            text=tesseract_text(path); vals=nums(text); dur,status=duration_from(vals)
            rows.append({**t,'engine':'tesseract_lower','text':text,'numbers':vals,'duration_s':dur,'status':status,'duration_abs_error_s':None if dur is None or t['expected_duration_s'] is None else abs(dur-t['expected_duration_s'])})
        except Exception as e:
            rows.append({**t,'engine':'tesseract_lower','text':'','numbers':[],'duration_s':None,'status':f'error:{e}','duration_abs_error_s':None})
        # RapidOCR boxes
        try:
            text,vals=box_text_numbers_bottom(rapid_boxes(path),h); dur,status=duration_from(vals)
            rows.append({**t,'engine':'rapidocr_bottom_boxes','text':text,'numbers':vals,'duration_s':dur,'status':status,'duration_abs_error_s':None if dur is None or t['expected_duration_s'] is None else abs(dur-t['expected_duration_s'])})
        except Exception as e:
            rows.append({**t,'engine':'rapidocr_bottom_boxes','text':'','numbers':[],'duration_s':None,'status':f'error:{e}','duration_abs_error_s':None})
        # EasyOCR boxes
        try:
            text,vals=box_text_numbers_bottom(easy_boxes(path),h); dur,status=duration_from(vals)
            rows.append({**t,'engine':'easyocr_bottom_boxes','text':text,'numbers':vals,'duration_s':dur,'status':status,'duration_abs_error_s':None if dur is None or t['expected_duration_s'] is None else abs(dur-t['expected_duration_s'])})
        except Exception as e:
            rows.append({**t,'engine':'easyocr_bottom_boxes','text':'','numbers':[],'duration_s':None,'status':f'error:{type(e).__name__}:{str(e)[:100]}','duration_abs_error_s':None})
    summary={}
    for eng in sorted(set(r['engine'] for r in rows)):
        sub=[r for r in rows if r['engine']==eng]
        synth=[r for r in sub if r['expected_duration_s'] is not None]
        ok=[r for r in synth if r['duration_abs_error_s'] is not None and r['duration_abs_error_s']<=0.25]
        false=[r for r in sub if r['expected_duration_s'] is None and r['duration_s'] is not None]
        summary[eng]={'synthetic_correct':len(ok),'synthetic_total':len(synth),'benchmark_false_positive_duration':len(false),'statuses':{s:sum(1 for r in sub if r['status']==s) for s in sorted(set(r['status'] for r in sub))}}
    report={'summary':summary,'rows':rows,'tests':tests}
    out=OUT/'axis_ocr_compare_refined_results.json'
    out.write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print(json.dumps({'out_json':str(out),'summary':summary,'rows':rows},indent=2,ensure_ascii=False))

if __name__=='__main__': main()
