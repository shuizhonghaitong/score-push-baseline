#!/bin/bash
export HTTP_PROXY=http://172.28.4.16:18999 HTTPS_PROXY=http://172.28.4.16:18999
F=data/local/wheels311/rdkit-2025.3.6-cp311-cp311-manylinux_2_28_x86_64.whl
URL=https://files.pythonhosted.org/packages/b1/9d/2622e01e38736a68d84b34c7b955e7ce32f0d19901e79620b80b1217300e/rdkit-2025.3.6-cp311-cp311-manylinux_2_28_x86_64.whl
SHA=afc931d1be1a4ad2fc4f2f2732c10983b1a57fbde9c22870721beafe42be3ab0
SZ=36180899
for i in $(seq 1 60); do
  curl -sL -C - --max-time 300 -o "$F" "$URL"
  sz=$(stat -c %s "$F" 2>/dev/null || echo 0)
  if [ "$sz" -ge "$SZ" ]; then
    h=$(sha256sum "$F" | cut -d' ' -f1)
    if [ "$h" = "$SHA" ]; then echo "DONE $i sha ok"; exit 0; fi
    echo "attempt $i size ok sha BAD ($h) -> restart"
    mv "$F" "$F.bad$i"
  else
    echo "attempt $i have $sz/$SZ"
  fi
done
echo "FAILED after 60 attempts"; exit 1
