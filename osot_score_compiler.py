#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OSOT 播音分谱双向编译器 & 前沿 TTS/指挥家协议转换器 (osot_score_compiler.py)
==========================================================================
打通 OSOT 阶段一/二（标注程序 JSON）与阶段三（可控语音合成 & LLM 指挥家训练）的原创编译工具。

四大核心编译目标（开源社区目前无现成对应工具）：
1. arXiv:2610.29123 中文字符级内联韵律标记序列：
   - 四级停顿（‚ / ‥ / … / 。）、能量（↑ / ↓）、音高（↗ / ↘）、字长拉伸/压缩（→ / ←）
   - 可直接作为 Fun-CosyVoice 3.0 字符级韵律 LoRA 微调训练文本。
2. ACL 2026 BatonVoice 风格「指挥家—乐团」结构化 JSON 声学分谱：
   - 整合篇章级（节奏六型/体裁）、句级（语势五形态/勒让德系数/补气三态）、字级（停连四法/十类重音/虚实声）；
   - 按《03号文档》多引擎路由规则，自动为每个短语推荐最优合成引擎：
     Fun-CosyVoice-3.0 / IndexTTS-2.5 (emo_alpha <= 0.85) / FlowTTS-GRPO。
3. LLM「指挥家」SFT / DPO 偏好对齐训练样本 (.jsonl)：
   - 输入机器初标 JSON (rejected) + 人工专家终稿 JSON (chosen)，自动构造标准 DPO 偏好对，
     用于消除大模型规划意图与声学执行间的“播音腔/机械停顿”幻觉。
4. 逆向解析器 (--parse-script)：
   - 将播音员在记事本编辑的符号稿（【主重】〔次重〕 / // ^ [气声] [虚声]）逆向解析为标准 OSOT v0.2 JSON。
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# 1. 统一读取器（兼容 v0.1 MVP / v0.2 扩展 / 工具 01 三层解耦 JSON）
# ---------------------------------------------------------------------------
def normalize_osot_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
    meta = {
        "id": doc.get("id", doc.get("source_id", "untitled")),
        "title": doc.get("title", "未命名播音稿"),
        "global_rhythm": doc.get("global_rhythm", "舒缓"),
        "genre_style": doc.get("genre_style", "文艺朗诵"),
    }
    phrases_meta = []
    tokens_out: List[Dict[str, Any]] = []

    if "tier1_objective_acoustic" in doc:
        t1 = doc["tier1_objective_acoustic"].get("tokens", [])
        t2 = doc.get("tier2_algorithmic_hypothesis", {}).get("tokens", [])
        t3 = doc.get("tier3_human_gold", {}).get("tokens", [])
        phrases_meta = doc.get("tier2_algorithmic_hypothesis", {}).get("phrases", [])
        for i, a in enumerate(t1):
            h = t2[i] if i < len(t2) else {}
            g = t3[i] if i < len(t3) else {}
            stress = g.get("human_stress", "UNVERIFIED")
            if stress == "UNVERIFIED":
                stress = h.get("candidate_stress", "S0")
            pause = g.get("human_pause", "UNVERIFIED")
            if pause == "UNVERIFIED":
                pause = h.get("candidate_pause", {}).get("level", "B0")
            vq = g.get("human_voice_quality", "UNVERIFIED")
            if vq == "UNVERIFIED":
                vq = h.get("candidate_voice_quality", "Modal")
            tokens_out.append(
                {
                    "index": i,
                    "char": a.get("char", ""),
                    "punct_after": a.get("punct_after", ""),
                    "start_ms": a.get("start_ms", 0),
                    "end_ms": a.get("end_ms", 0),
                    "duration_ms": a.get("duration_ms", 200),
                    "gap_after_ms": a.get("gap_after_ms", 0),
                    "stress": stress,
                    "pause_after": pause,
                    "voice_quality": vq,
                    "is_elongated": h.get("is_elongated", a.get("duration_ms", 200) >= 380),
                    "f0_macro_mean_st": a.get("f0_macro_mean_st", 0.0),
                    "v02_tags": g.get("knowledge_tags", []),
                }
            )
    else:
        for i, t in enumerate(doc.get("tokens", [])):
            feat = t.get("features", {})
            dur = int(feat.get("duration_ms", max(40, int(t.get("end_ms", 200)) - int(t.get("start_ms", 0)))))
            gap = int(feat.get("gap_after_ms", 0))
            stress_raw = str(t.get("stress", "none"))
            stress = {"primary": "S2", "secondary": "S1", "none": "S0"}.get(stress_raw, stress_raw)
            pause_raw = str(t.get("pause_after", "none"))
            pause = {"long": "B3", "short": "B2", "connect": "B0_CONNECT", "none": "B0"}.get(pause_raw, pause_raw)
            v02 = t.get("v02", {})
            tokens_out.append(
                {
                    "index": i,
                    "char": t.get("char", ""),
                    "punct_after": t.get("punct_after", ""),
                    "start_ms": int(t.get("start_ms", 0)),
                    "end_ms": int(t.get("end_ms", 200)),
                    "duration_ms": dur,
                    "gap_after_ms": gap,
                    "stress": stress,
                    "pause_after": pause,
                    "voice_quality": v02.get("voice_quality", t.get("voice_quality", "Modal")),
                    "pause_type": v02.get("pause_type", ""),
                    "stress_type": v02.get("stress_type", ""),
                    "breath_action": v02.get("breath_action", ""),
                    "sentence_contour": v02.get("sentence_contour", ""),
                    "is_elongated": dur >= 380,
                    "f0_macro_mean_st": float(feat.get("f0_range_st", 0.0)),
                    "v02_tags": t.get("knowledge_tags", []),
                }
            )

    return {"meta": meta, "tokens": tokens_out, "phrases_meta": phrases_meta}


# ---------------------------------------------------------------------------
# 2. 编译目标 A：arXiv:2610.29123 中文字符级内联韵律标记序列
# ---------------------------------------------------------------------------
def compile_inline_prosody_markup(norm: Dict[str, Any]) -> str:
    """
    按照 arXiv:2610.29123 规范嵌入内联 Unicode 韵律控制符：
    - 停顿：‚ (B1微停/虚连) | ‥ (B2短停) | … (B3长停) | 。 (极长句群停顿 >=700ms)
    - 能量/重音：↑ (S2主重音) | ↓ (轻虚弱读)
    - 音高走势：↗ (高扬/上山点) | ↘ (低沉/落潮点)
    - 字长：→ (显著拖音延长) | ← (极短促压缩 <=95ms)
    - 发声态扩展标签：[气声] / [虚声] / [喉化]
    """
    out: List[str] = []
    prev_vq = "Modal"
    for t in norm["tokens"]:
        vq = t.get("voice_quality", "Modal")
        if vq != prev_vq and vq in ("Breathy", "Soft", "Creaky", "气声", "虚声", "喉化"):
            vq_map = {"Breathy": "[气声]", "Soft": "[虚声]", "Creaky": "[喉化]",
                      "气声": "[气声]", "虚声": "[虚声]", "喉化": "[喉化]"}
            out.append(vq_map.get(vq, ""))
        prev_vq = vq

        prefix = ""
        if t["stress"] == "S2":
            prefix += "↑"
        elif vq in ("Soft", "Breathy", "虚声", "气声") and t["stress"] == "S0":
            prefix += "↓"

        if t.get("f0_macro_mean_st", 0.0) >= 2.0:
            prefix += "↗"
        elif t.get("f0_macro_mean_st", 0.0) <= -2.0:
            prefix += "↘"

        suffix = ""
        if t.get("is_elongated") or t.get("duration_ms", 200) >= 380:
            suffix += "→"
        elif t.get("duration_ms", 200) <= 95:
            suffix += "←"

        out.append(f"{prefix}{t['char']}{suffix}")

        p = t["pause_after"]
        gap = t.get("gap_after_ms", 0)
        if p == "B3" and gap >= 700:
            out.append("。")
        elif p == "B3":
            out.append("…")
        elif p == "B2":
            out.append("‥")
        elif p == "B1":
            out.append("‚")
        elif t.get("punct_after") and p != "B0_CONNECT":
            # 保留非直连处原标点语义提示
            pass
    return "".join(out)


# ---------------------------------------------------------------------------
# 3. 编译目标 B：ACL 2026 BatonVoice 风格结构化分谱 + 多引擎智能路由
# ---------------------------------------------------------------------------
def compile_batonvoice_director_score(norm: Dict[str, Any]) -> Dict[str, Any]:
    tokens = norm["tokens"]
    phrases_meta = norm["phrases_meta"]
    segments: List[Dict[str, Any]] = []

    # 按 B2/B3 切分短语
    cuts: List[Tuple[int, int]] = []
    if phrases_meta:
        for pm in phrases_meta:
            cuts.append((int(pm["token_start"]), int(pm["token_end"])))
    else:
        s = 0
        for i, t in enumerate(tokens):
            if t["pause_after"] in ("B2", "B3") or i == len(tokens) - 1:
                cuts.append((s, i))
                s = i + 1

    for idx, (s, e) in enumerate(cuts):
        sub_toks = tokens[s : e + 1]
        plain = "".join(x["char"] + x.get("punct_after", "") for x in sub_toks)
        inline = compile_inline_prosody_markup({"meta": norm["meta"], "tokens": sub_toks, "phrases_meta": []})

        has_non_modal = any(x.get("voice_quality") in ("Breathy", "Soft", "Creaky", "气声", "虚声", "喉化") for x in sub_toks)
        s2_cnt = sum(1 for x in sub_toks if x["stress"] == "S2")
        elong_cnt = sum(1 for x in sub_toks if x.get("is_elongated"))

        # 《03号文档》多引擎动态路由决策规则
        if has_non_modal or elong_cnt >= 2:
            engine = "FlowTTS-GRPO (声学速度场气声/拖音精修)"
            engine_params = {"cfg_scale": 1.2, "phonation_reward_guided": True}
        elif s2_cnt >= 2 or norm["meta"]["genre_style"] in ("文艺朗诵", "影视配音"):
            engine = "IndexTTS-2.5 (音色-情感双路解耦)"
            engine_params = {"emo_alpha": 0.80, "note": "严控 emo_alpha <= 0.85 防音色漂移"}
        else:
            engine = "Fun-CosyVoice-3.0 + CharProsody-LoRA (稳健长句/低CER)"
            engine_params = {"lora_alpha": 0.85, "inline_prosody_enabled": True}

        leg = phrases_meta[idx].get("legendre_yushi") if idx < len(phrases_meta) else None
        rhythm = phrases_meta[idx].get("rhythm") if idx < len(phrases_meta) else None

        segments.append(
            {
                "segment_id": idx + 1,
                "plain_text": plain,
                "inline_prosody_text": inline,
                "sentence_contour": leg["yushi_cn"] if leg else (sub_toks[0].get("sentence_contour") or "待标定"),
                "legendre_vector": (
                    [leg["c0_level"], leg["c1_slope"], leg["c2_curvature"], leg["c3_inflection"]] if leg else None
                ),
                "rhythm_profile": rhythm or {"mode_cn": norm["meta"]["global_rhythm"]},
                "recommended_tts_engine": engine,
                "engine_params": engine_params,
                "qa_gate": {
                    "max_cer": 0.02,
                    "min_speaker_sim": 0.80,
                    "enable_ast_local_inpaint": True,
                },
            }
        )

    return {
        "protocol": "osot-batonvoice-director-v1.0",
        "compiled_at": datetime.now().isoformat(timespec="seconds"),
        "global_directive": norm["meta"],
        "segments": segments,
    }


# ---------------------------------------------------------------------------
# 4. 编译目标 C：LLM「指挥家」SFT / DPO 偏好对齐训练对 (.jsonl)
# ---------------------------------------------------------------------------
def build_conductor_dpo_record(auto_doc: Dict[str, Any], gold_doc: Dict[str, Any]) -> Dict[str, Any]:
    norm_rej = normalize_osot_doc(auto_doc)
    norm_cho = normalize_osot_doc(gold_doc)
    plain_text = "".join(t["char"] + t.get("punct_after", "") for t in norm_cho["tokens"])

    prompt = (
        f"你是 OSOT 中文专业播音导演（指挥家）。请根据体裁「{norm_cho['meta']['genre_style']}」"
        f"与全篇节奏「{norm_cho['meta']['global_rhythm']}」，为以下播音稿生成字级内联韵律分谱：\n{plain_text}"
    )
    chosen = json.dumps(
        {
            "inline_prosody": compile_inline_prosody_markup(norm_cho),
            "director_score": compile_batonvoice_director_score(norm_cho)["segments"],
        },
        ensure_ascii=False,
    )
    rejected = json.dumps(
        {
            "inline_prosody": compile_inline_prosody_markup(norm_rej),
            "director_score": compile_batonvoice_director_score(norm_rej)["segments"],
        },
        ensure_ascii=False,
    )
    return {
        "id": norm_cho["meta"]["id"],
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
    }


# ---------------------------------------------------------------------------
# 5. 逆向解析器：播音符号稿 -> 标准 OSOT v0.2 JSON
# ---------------------------------------------------------------------------
def parse_annotated_script_to_json(script_text: str, title: str = "逆向解析播音稿") -> Dict[str, Any]:
    """支持解析 【主重音】、〔次重音〕、// (长停)、/ (短停)、^ (连读)、[气声]/[虚声]/[喉化]。"""
    tokens: List[Dict[str, Any]] = []
    cur_vq = "Modal"
    i = 0
    n = len(script_text)
    t_cursor = 0

    while i < n:
        ch = script_text[i]
        if ch.isspace():
            i += 1
            continue
        # 发声态标签 [气声] / [虚声] / [喉化] / [实声]
        if ch == "[":
            end_b = script_text.find("]", i)
            if end_b != -1:
                tag = script_text[i + 1 : end_b]
                cur_vq = {"气声": "Breathy", "虚声": "Soft", "喉化": "Creaky", "实声": "Modal"}.get(tag, cur_vq)
                i = end_b + 1
                continue
        # 停连符号作用于前一个字
        if script_text.startswith("//", i):
            if tokens:
                tokens[-1]["pause_after"] = "long"
            i += 2
            continue
        if ch == "/":
            if tokens:
                tokens[-1]["pause_after"] = "short"
            i += 1
            continue
        if ch == "^":
            if tokens:
                tokens[-1]["pause_after"] = "connect"
            i += 1
            continue
        # 主/次重音包裹块
        if ch in ("【", "〔"):
            close = "】" if ch == "【" else "〕"
            stress_val = "primary" if ch == "【" else "secondary"
            end_c = script_text.find(close, i + 1)
            if end_c != -1:
                inner = script_text[i + 1 : end_c]
                for c in inner:
                    if c.strip():
                        tokens.append(
                            {
                                "index": len(tokens),
                                "char": c,
                                "punct_after": "",
                                "start_ms": t_cursor,
                                "end_ms": t_cursor + 220,
                                "stress": stress_val,
                                "pause_after": "none",
                                "v02": {"voice_quality": cur_vq},
                                "knowledge_tags": [],
                            }
                        )
                        t_cursor += 230
                i = end_c + 1
                continue
        # 标点挂靠到前一个字
        if ch in "，。！？、；：,.!?;:":
            if tokens:
                tokens[-1]["punct_after"] += ch
            i += 1
            continue
        # 普通汉字
        tokens.append(
            {
                "index": len(tokens),
                "char": ch,
                "punct_after": "",
                "start_ms": t_cursor,
                "end_ms": t_cursor + 200,
                "stress": "none",
                "pause_after": "none",
                "v02": {"voice_quality": cur_vq},
                "knowledge_tags": [],
            }
        )
        t_cursor += 210
        i += 1

    return {
        "id": "parsed_script",
        "title": title,
        "schema_version": "osot-v0.2",
        "tokens": tokens,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="OSOT 播音分谱双向编译器 & TTS/指挥家协议转换器")
    parser.add_argument("--json", dest="json_path", help="输入 OSOT JSON（v0.1 / v0.2 / 3-tier）")
    parser.add_argument("--auto-json", dest="auto_json", help="可选：机器初标 JSON（与 --json 配对生成 DPO JSONL）")
    parser.add_argument("--parse-script", dest="script_path", help="可选：逆向解析带符号播音稿 .txt")
    parser.add_argument("--out-dir", dest="out_dir", default="osot_compiled_out", help="输出目录")
    parser.add_argument("--demo", action="store_true", help="运行内置全流程编译与逆向解析自测")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.demo:
        demo_script = "让暴风雨 / 来得【更猛烈】些吧！// [气声]夜色^渐〔深〕。"
        parsed = parse_annotated_script_to_json(demo_script, title="OSOT分谱编译器演示稿")
        norm = normalize_osot_doc(parsed)
        inline_txt = compile_inline_prosody_markup(norm)
        baton = compile_batonvoice_director_score(norm)
        # 构造一份未精标版本用于演示 DPO 生成
        auto_mock = json.loads(json.dumps(parsed, ensure_ascii=False))
        for tok in auto_mock["tokens"]:
            tok["stress"] = "none"
        dpo_rec = build_conductor_dpo_record(auto_mock, parsed)

        (out_dir / "demo_parsed_v02.json").write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")
        (out_dir / "demo_inline_2610_29123.txt").write_text(inline_txt + "\n", encoding="utf-8")
        (out_dir / "demo_batonvoice_score.json").write_text(json.dumps(baton, ensure_ascii=False, indent=2), encoding="utf-8")
        (out_dir / "demo_conductor_dpo.jsonl").write_text(json.dumps(dpo_rec, ensure_ascii=False) + "\n", encoding="utf-8")

        print("=== OSOT 分谱编译与逆向解析自测完成 ===")
        print("输入符号稿 :", demo_script)
        print("内联韵律串 :", inline_txt)
        for seg in baton["segments"]:
            print(f"  片段#{seg['segment_id']} 「{seg['plain_text']}」 -> 路由引擎: {seg['recommended_tts_engine']}")
        print(f"输出目录   : {out_dir}")
        return

    if args.script_path:
        raw = Path(args.script_path).read_text(encoding="utf-8")
        parsed = parse_annotated_script_to_json(raw, title=Path(args.script_path).stem)
        out_p = out_dir / f"{Path(args.script_path).stem}_v02.json"
        out_p.write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"已逆向解析为标准 v0.2 JSON: {out_p}")
        return

    if not args.json_path:
        parser.error("请提供 --json、--parse-script 或使用 --demo 运行自测。")

    gold_doc = json.loads(Path(args.json_path).read_text(encoding="utf-8"))
    norm = normalize_osot_doc(gold_doc)
    stem = Path(args.json_path).stem

    inline_txt = compile_inline_prosody_markup(norm)
    baton = compile_batonvoice_director_score(norm)
    (out_dir / f"{stem}_inline_prosody.txt").write_text(inline_txt + "\n", encoding="utf-8")
    (out_dir / f"{stem}_batonvoice_score.json").write_text(json.dumps(baton, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.auto_json:
        auto_doc = json.loads(Path(args.auto_json).read_text(encoding="utf-8"))
        dpo_rec = build_conductor_dpo_record(auto_doc, gold_doc)
        (out_dir / f"{stem}_conductor_dpo.jsonl").write_text(json.dumps(dpo_rec, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"已额外生成 LLM 指挥家 DPO 训练样本: {stem}_conductor_dpo.jsonl")

    print(f"编译完成！结果已输出至 {out_dir}")


if __name__ == "__main__":
    main()
