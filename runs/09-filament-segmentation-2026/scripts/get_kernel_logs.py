#!/usr/bin/env python3
"""Fetch Kaggle kernel logs via gRPC-web GetKernelLogs (no grpcio needed)."""
import os, sys, json, urllib.request

def tok():
    t = os.environ.get("KAGGLE_API_TOKEN", "")
    if os.path.isfile(t):
        t = open(t).read().strip()
    return t

def frame(payload: bytes) -> bytes:
    return b"\x00" + len(payload).to_bytes(4, "big") + payload

def slug_frame(slug: str, version: int = 0, offset: int = 0, from_end: bool = False) -> bytes:
    out = b"\x0a" + bytes([len(slug)]) + slug.encode()
    if version:
        out += b"\x10" + bytes([version])
    if offset:
        out += b"\x18" + bytes([offset])
    if from_end:
        out += b"\x20\x01"
    return out

def owner_slug_frame(slug: str, version: int = 0) -> bytes:
    owner, _, s = slug.partition("/")
    inner = b"\x0a" + bytes([len(owner)]) + owner.encode() + b"\x12" + bytes([len(s)]) + s.encode()
    out = b"\x0a" + bytes([len(inner)]) + inner
    if version:
        out += b"\x10" + bytes([version])
    return out

def call(method: str, body: bytes) -> None:
    req = urllib.request.Request(
        f"https://api.kaggle.com/v1/kernels.KernelsApiService/{method}",
        data=frame(body), method="POST",
        headers={"Authorization": f"Bearer {tok()}",
                 "Content-Type": "application/grpc-web+proto",
                 "x-grpc-web": "1", "x-user-agent": "kaggle-cli/2.2.4"})
    try:
        r = urllib.request.urlopen(req, timeout=30)
        data = r.read()
    except urllib.error.HTTPError as e:
        data = e.read()
        print(f"HTTP {e.code}")
    # parse grpc-web frames
    i = 0
    while i + 5 <= len(data):
        flag = data[i]; ln = int.from_bytes(data[i+1:i+5], "big")
        chunk = data[i+5:i+5+ln]; i += 5 + ln
        if flag & 0x80:
            print("TRAILERS:", chunk.decode("latin1"))
        else:
            print(f"FRAME flag={flag} len={ln}:")
            try:
                print(chunk.decode("utf-8")[:4000])
            except Exception:
                print(chunk[:2000])

if __name__ == "__main__":
    slug = sys.argv[1] if len(sys.argv) > 1 else "lilac1996/filament-magfilo-v1"
    ver = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    print("== GetKernelLogs (flat string ref)")
    call("GetKernelLogs", slug_frame(slug, ver, 0, True))
    print("== GetKernelLogs (owner/slug ref)")
    call("GetKernelLogs", owner_slug_frame(slug, ver))
    print("== GetKernel (flat string ref)")
    call("GetKernel", slug_frame(slug, ver))
