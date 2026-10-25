"""Extract CLIP ViT-B/32 image embeddings locally (CPU)."""
import os, argparse, time
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)

def to_tensor(im):
    im = im.convert("RGB")
    im = im.resize((224, 224), Image.BICUBIC)
    x = torch.from_numpy(np.asarray(im, dtype="float32") / 255.0).permute(2, 0, 1)
    return (x - MEAN) / STD

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--bs", type=int, default=8)
    args = ap.parse_args()
    out = os.path.join(ROOT, "outputs", f"clip_{args.split}.npz")
    if os.path.exists(out):
        print("exists:", out); return
    df = pd.read_csv(os.path.join(ROOT, "data", f"{args.split}.csv"))
    img_dir = os.path.join(ROOT, "data", "images", args.split)
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    embs = np.zeros((len(df), 512), dtype=np.float32)
    t0 = time.time()
    for i in range(0, len(df), args.bs):
        batch = df.iloc[i:i+args.bs]
        xs = []
        for _, row in batch.iterrows():
            p = os.path.join(img_dir, row.image_file)
            try:
                with Image.open(p) as im:
                    xs.append(to_tensor(im))
            except Exception as e:
                print("ERR", p, repr(e)[:80])
                xs.append((torch.ones(3, 224, 224) - MEAN) / STD)
        x = torch.stack(xs)
        with torch.no_grad():
            z = model.visual_projection(model.vision_model(pixel_values=x).last_hidden_state[:, 0])
            z = z / z.norm(dim=-1, keepdim=True)
        embs[i:i+len(batch)] = z.numpy()
        if i % (args.bs * 25) == 0:
            dt = time.time() - t0
            rate = (i + len(batch)) / max(dt, 1e-6)
            print(f"{i+len(batch)}/{len(df)} rate={rate:.2f}/s eta={(len(df)-i)/max(rate,1e-6):.0f}s", flush=True)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, emb=embs, ids=df.trace_id.values)
    print("saved", out, f"{time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
