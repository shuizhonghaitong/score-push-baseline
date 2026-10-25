"""Generate notebooks/submit/submit.py embedding a given prediction CSV.
Usage: python make_submit_nb.py <csv_path> [note_message]
"""
import sys

def main():
    csv_path = sys.argv[1]
    msg = sys.argv[2] if len(sys.argv) > 2 else "umud submit"
    csv_text = open(csv_path).read().strip("\n")
    lines = csv_text.split("\n")
    assert lines[0] == "image_id,pa_deg,fl_mm,mt_mm", f"bad header: {lines[0]}"
    n = len(lines) - 1
    print("embedding", n, "rows from", csv_path)
    q = chr(34) * 3  # triple double-quote
    body = (
        "# Auto-generated submission runner. Embeds a prediction CSV, writes it,\n"
        "# and submits it via the internal Kaggle API (avoids local GCS upload).\n"
        "import json, os, time, traceback\n"
        "\n"
        'COMP = "umud-challenge-muscle-architecture-in-ultrasound-data"\n'
        "MESSAGE = " + repr(msg) + "\n"
        "\n"
        "CSV_TEXT = " + q + "\n" + csv_text + "\n" + q + "\n"
        "\n"
        'os.makedirs("/kaggle/output", exist_ok=True)\n'
        'path = "/kaggle/submission.csv"\n'
        'with open(path, "w") as f:\n'
        '    f.write(CSV_TEXT + "\\n")\n'
        'print("wrote", path, "rows:", CSV_TEXT.count(chr(10)))\n'
        "\n"
        "try:\n"
        "    from kaggle.api.kaggle_api_extended import KaggleApi\n"
        "    api = KaggleApi()\n"
        "    api.authenticate()\n"
        "    res = api.competition_submit(path, f\"{MESSAGE} {time.strftime('%H%M%S')}\", COMP)\n"
        '    print("SUBMIT RESULT:", json.dumps(res, default=str), flush=True)\n'
        "except Exception:\n"
        "    traceback.print_exc()\n"
        '    print("SUBMIT FAILED", flush=True)\n'
        'print("ALL DONE", flush=True)\n'
    )
    with open("notebooks/submit/submit.py", "w") as f:
        f.write(body)
    print("wrote notebooks/submit/submit.py", len(body), "bytes")

if __name__ == "__main__":
    main()
