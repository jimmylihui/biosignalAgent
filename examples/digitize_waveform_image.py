from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from biosignal_agent.tools.digitize_tools import Signal_digitize_waveform_image, Signal_digitize_waveform_image_ml
from biosignal_agent.tools.digitize_unet_tools import Signal_digitize_waveform_image_unet


def main() -> None:
    parser = argparse.ArgumentParser(description="Digitize a clean single-trace waveform plot image into CSV.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--sampling-rate", type=float, default=None)
    parser.add_argument("--out-csv", default=None)
    parser.add_argument("--value-min", type=float, default=None)
    parser.add_argument("--value-max", type=float, default=None)
    parser.add_argument("--crop-left", type=int, default=0)
    parser.add_argument("--crop-right", type=int, default=0)
    parser.add_argument("--crop-top", type=int, default=0)
    parser.add_argument("--crop-bottom", type=int, default=0)
    parser.add_argument("--threshold", type=int, default=80)
    parser.add_argument("--method", choices=["rule", "ml", "unet"], default="rule")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--probability-threshold", type=float, default=0.5)
    parser.add_argument("--smooth-window", type=int, default=1)
    args = parser.parse_args()

    common = dict(
        image_path=args.image,
        sampling_rate=args.sampling_rate,
        out_csv=args.out_csv,
        value_min=args.value_min,
        value_max=args.value_max,
        crop_left=args.crop_left,
        crop_right=args.crop_right,
        crop_top=args.crop_top,
        crop_bottom=args.crop_bottom,
        smooth_window=args.smooth_window,
    )
    if args.method == "ml":
        result = Signal_digitize_waveform_image_ml(**common, model_path=args.model_path, probability_threshold=args.probability_threshold)
    elif args.method == "unet":
        result = Signal_digitize_waveform_image_unet(**common, model_path=args.model_path, probability_threshold=args.probability_threshold)
    else:
        result = Signal_digitize_waveform_image(**common, threshold=args.threshold)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
