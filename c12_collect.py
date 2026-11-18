# -*- coding: utf-8 -*-
"""
Collect every C12 summary into one report: each concern -> the experiment that answers it -> result
per site set. Writes results/c12/C12_report.md and prints it. Reads only finished summaries.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SETS = ("dev", "select", "confirm", "test")


def load(root, s, name):
    p = Path(root) / s / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def tline(summ, tid):
    if not summ:
        return "-"
    t = next((t for t in summ["tests"] if t["id"] == tid), None)
    if not t:
        return "-"
    pc = t.get("page_ci95", [float("nan")] * 2)
    return (f"{t['diff'] * 100:+.1f} [{t['ci95'][0] * 100:+.1f},{t['ci95'][1] * 100:+.1f}] "
            f"page[{pc[0] * 100:+.1f},{pc[1] * 100:+.1f}] {t['sites_pos_neg'][0]}/{t['sites_pos_neg'][1]}")


def verdict(summ, tid):
    if not summ:
        return ""
    t = next((t for t in summ["tests"] if t["id"] == tid), None)
    if not t:
        return ""
    ok = t["ci95"][0] > 0 and t.get("page_ci95", [0])[0] > 0 and t["sites_pos_neg"][0] > t["sites_pos_neg"][1]
    return "PASS" if ok else "FAIL"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/c12")
    a = ap.parse_args()
    S = {s: load(a.root, s, "c12_summary.json") for s in SETS}
    G = {s: load(a.root, f"gen_{s}", "gen_summary.json") for s in SETS}
    L = []
    w = L.append
    w("# C12 report (PREREG_C12.md)\n")
    w("| set | n | pages | dense | perturb | code |")
    w("|---|---|---|---|---|---|")
    for s, v in S.items():
        if v:
            w(f"| {s} | {v['n']} | {v['n_pages']} | {v['dense'] or '-'} | {v['perturb']} | {v['code_hash']} |")

    rows = [
        ("이게 당연한 거 아냐? / BM25에서만 그런 거 아냐?", "D-1 dense 자기 vs 남의 관계 (주)", "D-1"),
        ("", "D-2 dense 글자만 vs BM25", "D-2"),
        ("", "D-2s 침묵 문항에서 dense 글자만 vs BM25", "D-2s"),
        ("", "D-4 dense 자기 관계 vs dense 글자만", "D-4"),
        ("", "D-5 dense 자기 관계 vs dense 부모", "D-5"),
        ("", "D-6 BM25+관계 vs dense 글자만", "D-6"),
        ("질문이 헤더 단어를 따라 쓴 데이터 편향?", "E-2 동의어로 바꾼 질문, BM25 자기 vs 남의 것 (주)", "E-2"),
        ("", "E-2p 실제로 바뀐 질문만", "E-2p"),
        ("", "E-3 바꾼 질문, dense 자기 vs 남의 것", "E-3"),
        ("기본 RAG(청킹)보다 나은가?", "G-1 노드+관계 vs 상위 청크, 64단어 (주)", "G-1"),
        ("", "G-1b BM25 노드 vs 상위 청크", "G-1b"),
        ("네 규칙 중 무엇이 핵심? (최적 구조)", "C-1 전체 vs 행만", "C-1_row"),
        ("", "C-1 전체 vs 열만", "C-1_col"),
        ("", "C-1 전체 vs 레이블만", "C-1_label"),
        ("", "C-1 전체 vs 제목만", "C-1_head"),
        ("", "C-2 행 빼면 손실", "C-2_row"),
        ("", "C-2 열 빼면 손실", "C-2_col"),
        ("", "C-2 레이블 빼면 손실", "C-2_label"),
        ("", "C-2 제목 빼면 손실", "C-2_head"),
        ("부모를 같이 붙이면?", "C-3 관계+부모 vs 관계", "C-3"),
        ("안 쓴 설명 관계(캡션, 속성 등)?", "B-2 확장 관계 vs 관계", "B-2"),
        ("기준 재현", "REF 자기 vs 남의 관계 (C8 R1)", "REF"),
    ]
    w("\n## 우려 → 실험 → 결과 (차이 %p [사이트 CI] page[사이트→페이지 CI] 사이트+/-)\n")
    w("| 우려 | 실험 | " + " | ".join(SETS) + " |")
    w("|---|---|" + "---|" * len(SETS))
    for concern, label, tid in rows:
        cells = []
        for s in SETS:
            v = S[s]
            c = tline(v, tid)
            vd = verdict(v, tid) if tid in ("D-1", "E-2", "G-1") else ""
            if vd:
                c += f" **{vd}**"
            cells.append(c)
        w(f"| {concern} | {label} | " + " | ".join(cells) + " |")

    w("\n## 생성형 리더 (F, 선택)\n")
    w("| 비교 | " + " | ".join(SETS) + " |")
    w("|---|" + "---|" * len(SETS))
    for tid, lab in (("F-1", "관계 vs BM25, F1 (주)"), ("F-2", "관계 vs 남의 관계, F1")):
        w(f"| {lab} | " + " | ".join(tline(G[s], tid) + (f" **{verdict(G[s], tid)}**" if tid == "F-1" and verdict(G[s], tid) else "") for s in SETS) + " |")

    w("\n## H: 관계 종류별 독립 비교, 자기 − 남의 것 (Recall@30 %p; 괄호는 정답에 그 관계가 있는 문항만)\n")
    names = next((list(v["types_H"]) for v in S.values() if v), [])
    w("| 종류 | 계열 | 존재율(test) | " + " | ".join(SETS) + " |")
    w("|---|---|---|" + "---|" * len(SETS))
    for nm in names:
        fam = next(v["types_H"][nm]["family"] for v in S.values() if v)
        pres = S["test"]["types_H"][nm]["present_at_gold"] if S["test"] else None
        cells = []
        for s in SETS:
            h = S[s]["types_H"][nm] if S[s] else None
            if not h or h["own_minus_other"] is None:
                cells.append("-")
                continue
            dp = h["own_minus_other_if_present"]
            cells.append(f"{h['own_minus_other'] * 100:+.1f} [{h['ci95'][0] * 100:+.1f},{h['ci95'][1] * 100:+.1f}]"
                         + (f" ({dp * 100:+.1f})" if dp is not None else ""))
        w(f"| {nm} | {fam} | {pres if pres is not None else '-'} | " + " | ".join(cells) + " |")
    w("\nH-1 (주): 존재율 10% 이상 의미 관계의 '존재 문항 차이' 중앙값 > 구조 관계의 같은 값 중앙값")
    for s in SETS:
        v = S[s]
        if not v:
            continue
        def med(xs):
            xs = sorted(x for x in xs if x is not None)
            return xs[len(xs) // 2] if xs else None
        sem = [h["own_minus_other_if_present"] for h in v["types_H"].values()
               if h["family"].startswith("semantic") and (h["present_at_gold"] or 0) >= 0.10]
        st = [h["own_minus_other_if_present"] for h in v["types_H"].values() if h["family"].startswith("structural")]
        ms, mt = med(sem), med(st)
        ok = ms is not None and mt is not None and ms > mt
        w(f"- {s}: semantic median {ms if ms is None else round(ms * 100, 1)} vs structural median "
          f"{mt if mt is None else round(mt * 100, 1)} → **{'PASS' if ok else 'FAIL'}**")
    w("\n겹침 (정답 노드, 의미 관계 토큰 중 구조 관계 안에 있는 비율):")
    for s in SETS:
        if S[s]:
            w(f"- {s}: " + json.dumps(S[s]["overlap_H"], ensure_ascii=False))

    w("\n## A: 질문-정답 연결 단어는 어디에 있나 (연결 단어가 있는 문항 중 비율)\n")
    cats = next((list(v["audit_A"]["among_questions_with_bridge_tokens"]) for v in S.values() if v), [])
    w("| 위치 | " + " | ".join(SETS) + " |")
    w("|---|" + "---|" * len(SETS))
    for c in cats:
        w(f"| {c} | " + " | ".join(str(S[s]["audit_A"]["among_questions_with_bridge_tokens"][c]) if S[s] else "-" for s in SETS) + " |")
    w("| 페이지 어디에도 없음 | " + " | ".join(str(S[s]["audit_A"]["nowhere_on_page"]) if S[s] else "-" for s in SETS) + " |")
    w("\n네 규칙이 놓친 문항 중 각 위치가 추가로 덮는 비율:")
    for s in SETS:
        if S[s]:
            w(f"- {s}: " + json.dumps(S[s]["audit_A"]["rules_missed_but_found_in"], ensure_ascii=False))
    w("\n정답 노드에 확장 관계가 있는 비율 (B-1):")
    for s in SETS:
        if S[s]:
            w(f"- {s}: " + json.dumps(S[s]["audit_A"]["present_at_gold"], ensure_ascii=False))
    w("\n질문 바꾸기 (E):")
    for s in SETS:
        if S[s] and S[s].get("perturb_E"):
            w(f"- {s}: " + json.dumps(S[s]["perturb_E"], ensure_ascii=False))
    w("\n청크 vs 노드 묶기, 정답 문자열 포함률 (G):")
    for s in SETS:
        if S[s]:
            w(f"- {s}: " + json.dumps(S[s]["chunk_G_ans_in_ctx"], ensure_ascii=False))

    text = "\n".join(L) + "\n"
    out = Path(a.root) / "C12_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"[saved] {out.resolve()}")


if __name__ == "__main__":
    main()
