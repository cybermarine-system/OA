"""VRSBench 데이터 점검: 분할 규모, 유형 분포, 이미지 무결성, 라벨 노이즈."""
import collections
import glob
import json
import os
import re
import sys
from os.path import join

from PIL import Image

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/home/user/ondeviceAI/molmo2/data/VRSBench"

test = json.load(open(join(ROOT, "VRSBench_EVAL_vqa.json")))
train = json.load(open(join(ROOT, "VRSBench_train.json")))
train_vqa = [x for x in train if "[vqa]" in x["conversations"][0]["value"]]
train_imgs = {x["image"] for x in train}
test_imgs = {x["image_id"] for x in test}

print("== 분할 규모 ==")
print(f"train: VQA {len(train_vqa)}쌍, 이미지(JSON 참조) {len(train_imgs)}장, "
      f"태그별 {dict(collections.Counter(re.search(r'\[(\w+)\]', x['conversations'][0]['value']).group(1) for x in train))}")
print(f"test : VQA {len(test)}쌍, 이미지(VQA 참조) {len(test_imgs)}장")
print(f"train/test 이미지 중복: {len(train_imgs & test_imgs)}")

print("\n== Annotations 기준 ==")
for sp in ["train", "val"]:
    fs = glob.glob(join(ROOT, f"ann/Annotations_{sp}/*.json"))
    ty = collections.Counter(q["type"] for f in fs for q in json.load(open(f))["qa_pairs"])
    print(f"{sp}: 이미지 {len(fs)}장, QA {sum(ty.values())}쌍, 유형 {len(ty)}개 {dict(ty.most_common())}")

print("\n== 이미지 무결성 ==")
for sp, ids in [("train", train_imgs), ("val", test_imgs)]:
    d = join(ROOT, f"images/Images_{sp}")
    files = os.listdir(d)
    missing = [i for i in ids if not os.path.exists(join(d, i))]
    sizes = collections.Counter(Image.open(join(d, f)).size for f in files)
    print(f"{sp}: 파일 {len(files)}장, 참조 누락 {len(missing)}, 해상도 {dict(sizes)}")

print("\n== test 라벨 점검 ==")
norm = lambda s: s.strip().lower().rstrip(".")
qty = [norm(x["ground_truth"]) for x in test if x["type"] == "object quantity"]
ex = [norm(x["ground_truth"]) for x in test if x["type"] == "object existence"]
print(f"quantity {len(qty)}건: 숫자 아님 {sum(not a.isdigit() for a in qty)}건 "
      f"(yes/no {sum(a in ('yes', 'no') for a in qty)}건), 1~3개 비율 {sum(a in ('1', '2', '3') for a in qty) / len(qty):.1%}")
print(f"existence {len(ex)}건: yes/no 아님 {sum(a not in ('yes', 'no') for a in ex)}건")
