from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description='Run DeepSeek-OCR-2 on an image for axis/tick OCR.')
    ap.add_argument('--image', required=True)
    ap.add_argument('--model-path', default='/data1/jiahui/biosignal-agent/models/DeepSeek-OCR-2')
    ap.add_argument('--prompt', default='Read all visible x-axis tick labels and x-axis text in this plot. Return only the labels/text, no explanation.')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--max-new-tokens', type=int, default=128)
    args = ap.parse_args()
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoTokenizer

    image = Image.open(args.image).convert('RGB')
    result = {'image': args.image, 'model_path': args.model_path, 'torch': torch.__version__, 'cuda': torch.cuda.is_available()}
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(args.device).eval()
        result['model_class'] = model.__class__.__name__
        if hasattr(model, 'infer'):
            prompt = args.prompt if '<image>' in args.prompt else '<image>\n' + args.prompt
            out = model.infer(tokenizer, prompt=prompt, image_file=args.image, output_path='/data1/jiahui/biosignal-agent/outputs/deepseek_ocr2_axis_tmp', base_size=1024, image_size=768, crop_mode=False, save_results=False, eval_mode=True)
            result.update({'method': 'infer', 'output': out})
        elif hasattr(model, 'chat'):
            out = model.chat(tokenizer, args.image, args.prompt, generation_config={'max_new_tokens': args.max_new_tokens, 'do_sample': False})
            result.update({'method': 'chat', 'output': out})
        else:
            result.update({'method': 'direct_loaded_no_known_api', 'known_attrs': [name for name in ['chat', 'generate', 'infer'] if hasattr(model, name)]})
    except Exception as exc:
        result['direct_error'] = repr(exc)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))

if __name__ == '__main__':
    main()
