"""Extract (label, ns, svg, s-text) boxes from the Friedman/Ellsworth page HTML."""
import html as htmlmod
import json
import os
import re
import sys

def main():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(base, "data/friedman/page2.html")
    raw = open(path, encoding="utf-8").read()
    raw = re.sub(r"<!--.*?-->", "", raw, flags=re.S)
    boxes = re.findall(r'<div class="box">(.*?)</div></div>', raw, flags=re.S)
    items = []
    for box in boxes:
        m = re.search(r'<font size="\+3">(.*?)</font>', box, flags=re.S)
        if not m:
            continue
        head = m.group(1)
        svg = None
        sm = re.search(r'href="(square-[0-9]+[a-z]*\.svg)"', box)
        if sm:
            svg = sm.group(1)
        labeltxt = re.sub(r"<[^>]+>", "", head)
        labeltxt = htmlmod.unescape(labeltxt).strip()
        ns = []
        for part in labeltxt.split(","):
            part = part.strip()
            mm = re.match(r"^(\d+)\s*-\s*(\d+)$", part)
            if mm:
                ns.extend(range(int(mm.group(1)), int(mm.group(2)) + 1))
            elif part.isdigit():
                ns.append(int(part))
        stxt = ""
        stm = re.search(r'<font size="\+1">(.*?)</font>', box, flags=re.S)
        if stm:
            stxt = htmlmod.unescape(re.sub(r"<[^>]+>", " ", stm.group(1))).strip()
        if ns:
            items.append({"label": labeltxt, "ns": ns, "svg": svg, "s": stxt})
    out = os.path.join(base, "data/friedman/page_items.json")
    json.dump(items, open(out, "w"), indent=1)
    total = sum(len(g["ns"]) for g in items)
    print(f"{len(items)} boxes, {total} ns, {sum(1 for g in items if g['svg'])} with svg")
    nset = sorted(set(n for g in items for n in g["ns"]))
    print("n range:", nset[0], "-", nset[-1])

if __name__ == "__main__":
    main()
