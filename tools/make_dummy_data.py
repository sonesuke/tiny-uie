# -*- coding: utf-8 -*-
import json, os, random
random.seed(42)

SENTS = [
    ("ABC株式会社は1999年に山田太郎が設立した。", [(0,5,"ORG"),(12,16,"PER")], [(0,1,"founded_by")]),
    ("XYZ有限会社の創業者は佐藤花子である。", [(0,6,"ORG"),(9,12,"PER")], [(0,1,"founded_by")]),
]

def dump(path, n=20):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for _ in range(n):
            t, spans, rels = random.choice(SENTS)
            f.write(json.dumps({
                "text": t,
                "spans": [{"start": s, "end": e, "type": ty} for s,e,ty in spans],
                "relations": [{"head": h, "tail": ta, "type": rty} for h,ta,rty in rels]
            }, ensure_ascii=False) + "\n")

if __name__ == '__main__':
    dump('data/train.jsonl', n=80)
    dump('data/dev.jsonl', n=20)
    os.makedirs('data/schema', exist_ok=True)
    with open('data/schema/types.json','w',encoding='utf-8') as f:
        json.dump({
            "ORG": {"name":"組織名","definition":"法人・団体・企業などの正式名称"},
            "PER": {"name":"人名","definition":"個人の姓名"}
        }, f, ensure_ascii=False, indent=2)
    with open('data/schema/relations.json','w',encoding='utf-8') as f:
        json.dump({"founded_by": {"name":"創業者","definition":"設立者関係"}}, f, ensure_ascii=False, indent=2)
    print('dummy data written')