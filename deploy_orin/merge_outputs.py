"""정합성 분할 결과(cons_c01..cNN.json)를 원래 순서대로 합쳐 consistency2000_out.json 하나로 만듦.

usage: python3 merge_outputs.py <결과 폴더> <분할 수> <출력 json>
"""
import json
import os
import sys


def main():
    d, n, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    resp = []
    for k in range(1, n + 1):
        r = json.load(open(os.path.join(d, f"cons_c{k:02d}.json")))["responses"]
        for x in r:
            x["batch_idx"] = len(resp)
            resp.append(x)
    json.dump(dict(input_file="chunks/cons_c01..c%02d.json" % n, responses=resp), open(out, "w"), ensure_ascii=False)
    print(f"{len(resp)}개 → {out}")


if __name__ == "__main__":
    main()
