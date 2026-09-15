import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

KAT_DIR = os.path.join(os.path.dirname(__file__), "kat")


def parse_rsp(path):
    """Parse a CAVP .rsp file into a list of dicts (keys as in the file)."""
    entries = []
    cur = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("["):
                if cur:
                    entries.append(cur)
                    cur = {}
                continue
            m = re.match(r"(\w+)\s*=\s*(.*)", line)
            if m:
                k, v = m.group(1), m.group(2).strip()
                if k in cur:
                    entries.append(cur)
                    cur = {}
                cur[k] = v
    if cur:
        entries.append(cur)
    return [e for e in entries if "Msg" in e]


def kat_message(entry, len_key="Len"):
    n_bits = int(entry.get(len_key, 8 * len(entry["Msg"]) // 2))
    msg = bytes.fromhex(entry["Msg"]) if n_bits else b""
    assert len(msg) == n_bits // 8
    return msg
