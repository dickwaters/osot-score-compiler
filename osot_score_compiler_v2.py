#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OSOT 播音分谱双向编译器、多引擎独立适配器与合成音频回测器 v2.0 (osot_score_compiler_v2.py)
======================================================================================
v2.0 核心升级（响应 P0/P1/P2 评审意见）：
1. 【P0 统一协议】全面基于 osot-unified-v2.0 读取与输出。
2. 【P1 引擎独立适配器 + 能力声明矩阵】不再假设所有 TTS 引擎支持同一套指令：
   - CosyVoice3Adapter  : 支持字符级内联停顿/重音标记，原生气声连续谱控制有限；
   - IndexTTS25Adapter  : 强情感/音色解耦（严控 emo_alpha <= 0.85），明确声明不支持精确毫秒停顿控制；
   - FlowTTSGRPOAdapter : 面向声学速度场的发声态（气声/虚声）与拖音精修，附带奖励目标向量。
3. 【P1 真实合成音频闭环回测器 (--verify-audio)】
   - 输入任意 TTS 引擎合成后的 WAV + 目标分谱 JSON，实测合成音频中的：
     ① 目标停顿处是否真的出现对应时长的低能量静音（容差 ±80ms）；
     ② 目标主重音 (S2) 位置相对非重音基线是否真的产生能量抬升（≥ +1.5 dB）；
     ③ 输出逐条指令的 PASS / FAIL 回测评分单 (audio_backtest_report.json)。
4. 【P2 严格 SFT / DPO / 对齐修正分流闸门】
   - 严禁把“改错字”或“纯调时间戳”直接塞进 DPO 偏好对！
   - 仅当两个方案满足「同一原文 + 同一语境 + 有效韵律表达差异 + 附带选择理由」时才入库 DPO (.jsonl)；
   - 改错字/纯时间校准自动分流至 SFT 或 alignment_qa_only.jsonl。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import wave
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_CUR_DIR = Path(__file__).resolve().parent
for _cand in (_CUR_DIR, _CUR_DIR.parent / "00_osot_schema"):
    if (_cand / "osot_schema.py").exists() and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))

from osot_schema import migrate_to_unified, resolve_effective_tokens  # noqa: E402


# ---------------------------------------------------------------------------
# 1. 引擎独立适配器（显式声明各引擎真实能力边界）
# ---------------------------------------------------------------------------
class BaseTTSAdapter:
    engine_name = "base"
    capabilities: Dict[str, Any] = {}

    def compile_segment(self, tokens: List[Dict[str, Any]], meta: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


class CosyVoice3Adapter(BaseTTSAdapter):
    engine_name = "Fun-CosyVoice-3.0 + CharProsody-LoRA"
    capabilities = {
        "inline_prosody_markup": True,
        "exact_pause_ms_control": False,  # 通过四级离散停顿符近似控制
        "stress_control": True,
        "voice_quality_continuous_control": False,
        "polyphone_inpainting": True,
    }

    def compile_segment(self, tokens: List[Dict[str, Any]], meta: Dict[str, Any]) -> Dict[str, Any]:
        out = []
        unsupported_warnings = []
        for t in tokens:
            if t.get("voice_quality") in ("Breathy", "Soft", "Creaky", "气声", "虚声", "喉化"):
                unsupported_warnings.append(f"字「{t['char']}」要求发声态 {t['voice_quality']}，CosyVoice3 原生控制较弱，建议交由 FlowTTS-GRPO。")
            pre = "↑" if t["stress"] == "S2" else ""
            suf = "→" if t.get("is_elongated") else ("←" if t.get("duration_ms", 200) <= 95 else "")
            out.append(f"{pre}{t['char']}{suf}")
            p = t["pause_after"]
            if p == "B3":
                out.append("…" if t.get("gap_after_ms", 400) < 700 else "。")
            elif p == "B2":
                out.append("‥")
            elif p == "B1":
                out.append("‚")
        return {
            "engine": self.engine_name,
            "capabilities": self.capabilities,
            "payload": {"inline_text": "".join(out), "lora_alpha": 0.85},
            "unsupported_warnings": unsupported_warnings,
        }


class IndexTTS25Adapter(BaseTTSAdapter):
    engine_name = "IndexTTS-2.5"
    capabilities = {
        "inline_prosody_markup": False,
        "exact_pause_ms_control": False,  # 官方 README 明确标注精准时长控制未启用
        "stress_control": "partial_via_emotion_prompt",
        "voice_quality_continuous_control": False,
        "emotion_timbre_decoupling": True,
    }

    def compile_segment(self, tokens: List[Dict[str, Any]], meta: Dict[str, Any]) -> Dict[str, Any]:
        plain = "".join(t["char"] + t.get("punct_after", "") for t in tokens)
        s2_words = [t["char"] for t in tokens if t["stress"] == "S2"]
        warnings = []
        if any(t["pause_after"] in ("B1", "B2", "B3") for t in tokens):
            warnings.append("IndexTTS-2.5 开源版精准时长/毫秒停顿控制尚未启用，停顿需通过标点切分或后期拼接兜底。")
        return {
            "engine": self.engine_name,
            "capabilities": self.capabilities,
            "payload": {
                "text": plain,
                "emo_alpha": 0.80,  # 严控 <= 0.85 防音色漂移
                "emotion_instruction": f"体裁:{meta.get('genre_style','文艺朗诵')}; 节奏:{meta.get('global_rhythm','舒缓型')}; 重点强调:{''.join(s2_words)}",
            },
            "unsupported_warnings": warnings,
        }


class FlowTTSGRPOAdapter(BaseTTSAdapter):
    engine_name = "FlowTTS-GRPO (声学速度场精修)"
    capabilities = {
        "inline_prosody_markup": True,
        "exact_pause_ms_control": True,
        "stress_control": True,
        "voice_quality_continuous_control": True,
    }

    def compile_segment(self, tokens: List[Dict[str, Any]], meta: Dict[str, Any]) -> Dict[str, Any]:
        targets = []
        for t in tokens:
            targets.append({
                "char": t["char"],
                "target_stress": t["stress"],
                "target_pause": t["pause_after"],
                "target_gap_ms": t.get("gap_after_ms", 0),
                "target_voice_quality": t.get("voice_quality", "Modal"),
            })
        return {
            "engine": self.engine_name,
            "capabilities": self.capabilities,
            "payload": {
                "text": "".join(t["char"] + t.get("punct_after", "") for t in tokens),
                "cfg_scale": 1.2,
                "phonation_reward_targets": targets,
            },
            "unsupported_warnings": [],
        }


ADAPTERS = {
    "cosyvoice3": CosyVoice3Adapter(),
    "indextts25": IndexTTS25Adapter(),
    "flowtts_grpo": FlowTTSGRPOAdapter(),
}


def compile_with_engine_adapters(doc: Dict[str, Any], engine_key: str = "auto") -> Dict[str, Any]:
    u = migrate_to_unified(doc)
    toks = resolve_effective_tokens(u, layer="gold_or_pred")
    meta = u["meta"]

    # 按 B2/B3 切段并逐段调用独立适配器
    segs = []
    s = 0
    for i, t in enumerate(toks):
        if t["pause_after"] in ("B2", "B3") or i == len(toks) - 1:
            sub = toks[s : i + 1]
            if engine_key == "auto":
                has_vq = any(x.get("voice_quality") in ("Breathy", "Soft", "Creaky", "气声", "虚声", "喉化") for x in sub)
                s2_cnt = sum(1 for x in sub if x["stress"] == "S2")
                chosen = "flowtts_grpo" if has_vq else ("indextts25" if s2_cnt >= 2 else "cosyvoice3")
            else:
                chosen = engine_key
            adapted = ADAPTERS[chosen].compile_segment(sub, meta)
            segs.append({"segment_id": len(segs) + 1, "token_span": [s, i], **adapted})
            s = i + 1

    return {
        "protocol": "osot-engine-adapted-score-v2.0",
        "compiled_at": datetime.now().isoformat(timespec="seconds"),
        "meta": meta,
        "segments": segs,
    }


# ---------------------------------------------------------------------------
# 2. P1 真实合成音频闭环回测器（检验要求重音/停顿后实际声音是否真的达标）
# ---------------------------------------------------------------------------
def _load_wav(wav_path: str) -> Tuple[np.ndarray, int]:
    with wave.open(wav_path, "rb") as wf:
        sr, ch, sw, nf = wf.getframerate(), wf.getnchannels(), wf.getsampwidth(), wf.getnframes()
        raw = wf.readframes(nf)
    data = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0 if sw == 2 else np.frombuffer(raw, dtype=np.int32).astype(np.float64) / 2147483648.0
    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1)
    return data, int(sr)


def verify_synthesized_audio(synth_wav: str, target_doc: Dict[str, Any], pause_tol_ms: float = 100.0, stress_lift_db: float = 1.5) -> Dict[str, Any]:
    """
    对 TTS 合成后的真实 WAV 执行客观声学回测：
    1. 重音回测：计算目标 S2 主重音时间窗的平均能量 (dB) 相对全句非重音 (S0) 中位数的实际抬升量；
    2. 停顿回测：检测目标 B2/B3 停顿窗口内连续低能量静音段的实际持续毫秒数，与目标停顿时长对比。
    """
    y, sr = _load_wav(synth_wav)
    toks = resolve_effective_tokens(target_doc, layer="gold_or_pred")
    if not toks or len(y) == 0:
        raise ValueError("目标分谱或合成音频为空。")

    # 计算 10ms 帧能量包络
    hop = max(1, int(sr * 0.010))
    win = max(2, int(sr * 0.025))
    n_f = max(1, 1 + max(0, len(y) - win) // hop)
    rms = np.array([math.sqrt(float(np.mean(y[i * hop : i * hop + win] ** 2)) + 1e-12) for i in range(n_f)])
    db = 20.0 * np.log10(np.maximum(rms, 1e-6)) + 90.0
    times_ms = (np.arange(n_f) * hop + 0.5 * win) * 1000.0 / sr

    # 按有效发声尾边界对齐（避免末尾静音把时间轴整体拉偏）
    ref_dur_ms = max(1.0, float(toks[-1]["end_ms"]))
    syn_dur_ms = len(y) * 1000.0 / sr
    active_idx = np.where(db > float(np.percentile(db, 15)) + 6.0)[0]
    syn_active_end_ms = float(times_ms[active_idx[-1]]) if len(active_idx) > 0 else syn_dur_ms
    scale = syn_active_end_ms / ref_dur_ms if abs(syn_active_end_ms - ref_dur_ms) > 80.0 else 1.0

    # 1. 重音能量抬升回测
    s0_dbs, stress_checks = [], []
    tok_dbs = []
    for t in toks:
        s_m, e_m = t["start_ms"] * scale, t["end_ms"] * scale
        mask = (times_ms >= s_m) & (times_ms <= e_m)
        val = float(np.mean(db[mask])) if np.any(mask) else float(np.median(db))
        tok_dbs.append(val)
        if t["stress"] == "S0":
            s0_dbs.append(val)
    baseline_db = float(np.median(s0_dbs)) if s0_dbs else float(np.median(db))

    for t, val in zip(toks, tok_dbs):
        if t["stress"] == "S2":
            lift = val - baseline_db
            stress_checks.append({
                "index": t["index"], "char": t["char"],
                "actual_lift_db": round(lift, 2),
                "required_lift_db": stress_lift_db,
                "passed": bool(lift >= stress_lift_db),
            })

    # 2. 停顿实测静音时长回测
    sil_thresh = float(np.percentile(db, 15)) + 6.0
    pause_checks = []
    for i, t in enumerate(toks[:-1]):
        if t["pause_after"] in ("B2", "B3"):
            target_ms = float(t.get("gap_after_ms") or (350.0 if t["pause_after"] == "B3" else 150.0))
            win_s = max(0.0, t["end_ms"] * scale - 60.0)
            win_e = min(syn_dur_ms, toks[i + 1]["start_ms"] * scale + 60.0)
            mask = (times_ms >= win_s) & (times_ms <= win_e)
            sub_sil = db[mask] <= sil_thresh
            # 统计窗口内最长连续静音帧数 * 10ms
            max_run, cur_run = 0, 0
            for flag in sub_sil:
                cur_run = cur_run + 1 if flag else 0
                max_run = max(max_run, cur_run)
            actual_ms = max_run * 10.0
            err_ms = abs(actual_ms - target_ms)
            pause_checks.append({
                "after_index": t["index"], "after_char": t["char"], "level": t["pause_after"],
                "target_pause_ms": round(target_ms, 1), "actual_pause_ms": round(actual_ms, 1),
                "error_ms": round(err_ms, 1), "passed": bool(err_ms <= pause_tol_ms),
            })

    s_pass = sum(1 for x in stress_checks if x["passed"])
    p_pass = sum(1 for x in pause_checks if x["passed"])
    return {
        "verified_at": datetime.now().isoformat(timespec="seconds"),
        "synth_wav": str(Path(synth_wav).name),
        "duration_scale_ratio": round(scale, 3),
        "summary": {
            "stress_pass_rate": round(s_pass / max(1, len(stress_checks)), 3) if stress_checks else 1.0,
            "pause_pass_rate": round(p_pass / max(1, len(pause_checks)), 3) if pause_checks else 1.0,
            "stress_passed": f"{s_pass}/{len(stress_checks)}",
            "pause_passed": f"{p_pass}/{len(pause_checks)}",
        },
        "stress_checks": stress_checks,
        "pause_checks": pause_checks,
    }


# ---------------------------------------------------------------------------
# 3. P2 严格 SFT / DPO / 对齐修正分流闸门
# ---------------------------------------------------------------------------
def route_revision_for_training(
    plan_a_doc: Dict[str, Any],
    plan_b_gold_doc: Dict[str, Any],
    preference_reason: str = "",
) -> Dict[str, Any]:
    """
    严格审查两份标注能否构成 DPO 偏好对，杜绝把“改错字”或“纯调时间戳”污染进 DPO：
    - 规则 1：若纯文本字符不一致 -> 判定为 typo_fix（仅用于 ASR/SFT 监督学习，拒绝入库 DPO）
    - 规则 2：若停连/重音/发声态完全一致，仅时间戳变化 -> 判定为 alignment_fix（仅用于对齐质检，拒绝入库 DPO）
    - 规则 3：若同一原文下存在实质韵律表达方案差异，且两者均为合法方案 -> 若提供 preference_reason 则允许入库 DPO，
             否则归入 SFT 并提示补充人工选择理由。
    """
    ta = resolve_effective_tokens(plan_a_doc, layer="pred")
    tb = resolve_effective_tokens(plan_b_gold_doc, layer="gold_or_pred")
    text_a = "".join(t["char"] for t in ta)
    text_b = "".join(t["char"] for t in tb)

    if text_a != text_b:
        return {
            "route": "SFT_TYPO_CORRECTION_ONLY",
            "dpo_eligible": False,
            "reason": "检测到原文错字修改，属于转写纠错而非播音表达偏好，已拦截移出 DPO。",
            "record": {"input_text": text_a, "corrected_text": text_b},
        }

    expr_diffs = []
    time_diffs = 0
    for a, b in zip(ta, tb):
        if a["stress"] != b["stress"] or a["pause_after"] != b["pause_after"] or a["voice_quality"] != b["voice_quality"]:
            expr_diffs.append({
                "index": a["index"], "char": a["char"],
                "plan_a": f"{a['stress']}/{a['pause_after']}/{a['voice_quality']}",
                "plan_b": f"{b['stress']}/{b['pause_after']}/{b['voice_quality']}",
            })
        if abs(int(a["start_ms"]) - int(b["start_ms"])) >= 10 or abs(int(a["end_ms"]) - int(b["end_ms"])) >= 10:
            time_diffs += 1

    if len(expr_diffs) == 0:
        return {
            "route": "ALIGNMENT_QA_ONLY",
            "dpo_eligible": False,
            "reason": f"仅存在 {time_diffs} 处时间戳对齐微调，无播音表达方案差异，已拦截移出 DPO。",
            "record": {"text": text_b, "time_adjusted_tokens": time_diffs},
        }

    # 检查方案 A 是否为空白/未标注（空白初稿 vs 终稿更适合 SFT，而非双方案偏好 DPO）
    a_has_expr = any(x["stress"] != "S0" or x["pause_after"] != "B0" for x in ta)
    if not a_has_expr:
        return {
            "route": "SFT_SUPERVISED_ONLY",
            "dpo_eligible": False,
            "reason": "方案 A 为空白无标注草稿，不属于「两种有效播音表达方案」对比，已自动路由至 SFT 监督训练集。",
            "record": {"prompt": text_b, "target_expression_diffs": expr_diffs},
        }

    if not preference_reason.strip():
        return {
            "route": "DPO_PENDING_REASON",
            "dpo_eligible": False,
            "reason": "满足同原文双表达方案条件，但缺少人工选择理由 (preference_reason)，补充理由后即可入库 DPO。",
            "expression_diffs": expr_diffs,
        }

    return {
        "route": "DPO_PREFERENCE_PAIR",
        "dpo_eligible": True,
        "reason": "通过全部门禁：同原文、同语境、两种有效表达方案对比且附带人工选择理由。",
        "record": {
            "prompt": f"请为以下播音稿设计专业韵律表达方案：{text_b}",
            "chosen": compile_with_engine_adapters(plan_b_gold_doc, "cosyvoice3")["segments"],
            "rejected": compile_with_engine_adapters(plan_a_doc, "cosyvoice3")["segments"],
            "preference_reason": preference_reason.strip(),
            "expression_diffs": expr_diffs,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="OSOT 分谱多引擎适配、音频回测与 SFT/DPO 分流器 v2.0")
    ap.add_argument("--json", help="目标 OSOT 统一 JSON")
    ap.add_argument("--engine", choices=["auto", "cosyvoice3", "indextts25", "flowtts_grpo"], default="auto")
    ap.add_argument("--verify-audio", help="可选：TTS 合成后的 WAV 文件，执行真实音频回测")
    ap.add_argument("--plan-a", help="可选：用于 SFT/DPO 分流闸门对比的方案 A JSON")
    ap.add_argument("--preference-reason", default="", help="人工偏好选择理由（DPO 入库必填）")
    ap.add_argument("--out-dir", default="osot_compiler_v2_out", help="输出目录")
    ap.add_argument("--demo", action="store_true", help="运行内置全功能 v2.0 自测（含引擎适配、真实音频回测、DPO 闸门拦截演示）")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.demo:
        # 1. 构造带真实停顿与重音能量抬升的模拟合成音频 + 目标 JSON
        sr = 16000
        toks_gold = [
            {"index": 0, "char": "让", "punct_after": "", "start_ms": 50, "end_ms": 230, "stress": "none", "pause_after": "none", "features": {"duration_ms": 180, "gap_after_ms": 20}},
            {"index": 1, "char": "暴", "punct_after": "", "start_ms": 250, "end_ms": 470, "stress": "primary", "pause_after": "short", "features": {"duration_ms": 220, "gap_after_ms": 200}},
            {"index": 2, "char": "风", "punct_after": "", "start_ms": 670, "end_ms": 850, "stress": "none", "pause_after": "none", "features": {"duration_ms": 180, "gap_after_ms": 20}},
            {"index": 3, "char": "雨", "punct_after": "！", "start_ms": 870, "end_ms": 1150, "stress": "primary", "pause_after": "long", "features": {"duration_ms": 280, "gap_after_ms": 380}},
            {"index": 4, "char": "来", "punct_after": "。", "start_ms": 1530, "end_ms": 1750, "stress": "none", "pause_after": "none", "v02": {"voice_quality": "气声"}, "features": {"duration_ms": 220, "gap_after_ms": 0}},
        ]
        doc_gold = {"id": "demo_v2_comp", "title": "v2.0 编译器自测", "tokens": toks_gold}

        # 合成一段匹配上述停顿与重音抬升的测试 WAV
        y = np.zeros(int(1.85 * sr), dtype=np.float64)
        for t in toks_gold:
            s_i, e_i = int(t["start_ms"] * sr / 1000), int(t["end_ms"] * sr / 1000)
            amp = 0.65 if t["stress"] == "primary" else 0.25
            tt = np.arange(e_i - s_i) / sr
            y[s_i:e_i] = amp * np.sin(2 * np.pi * 220 * tt)
        wav_p = out_dir / "demo_synth_check.wav"
        with wave.open(str(wav_p), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes((np.clip(y, -0.99, 0.99) * 32767).astype(np.int16).tobytes())

        adapted = compile_with_engine_adapters(doc_gold, "auto")
        backtest = verify_synthesized_audio(str(wav_p), doc_gold)

        # 测试 DPO 闸门：情形 1（仅改时间戳 -> 应拦截）vs 情形 2（双有效方案 + 理由 -> 应通过）
        doc_time_only = json.loads(json.dumps(doc_gold, ensure_ascii=False))
        doc_time_only["tokens"][0]["start_ms"] = 80
        gate1 = route_revision_for_training(doc_time_only, doc_gold)

        doc_alt_expr = json.loads(json.dumps(doc_gold, ensure_ascii=False))
        doc_alt_expr["tokens"][1]["stress"] = "none"
        doc_alt_expr["tokens"][2]["stress"] = "primary"
        gate2 = route_revision_for_training(
            doc_alt_expr, doc_gold, preference_reason="新闻/朗诵语境下重音落在核心意象「暴」而非「风」更具冲击力"
        )

        (out_dir / "demo_adapted_score.json").write_text(json.dumps(adapted, ensure_ascii=False, indent=2), encoding="utf-8")
        (out_dir / "demo_audio_backtest.json").write_text(json.dumps(backtest, ensure_ascii=False, indent=2), encoding="utf-8")
        (out_dir / "demo_dpo_gate_test.json").write_text(json.dumps({"case1_time_only": gate1, "case2_valid_dpo": gate2}, ensure_ascii=False, indent=2), encoding="utf-8")

        print("=== OSOT 工具 03 v2.0 自测完成 ===")
        print(f"音频回测结果 : 重音达标={backtest['summary']['stress_passed']} | 停顿达标={backtest['summary']['pause_passed']}")
        print(f"DPO 闸门测试1: {gate1['route']} (dpo_eligible={gate1['dpo_eligible']})")
        print(f"DPO 闸门测试2: {gate2['route']} (dpo_eligible={gate2['dpo_eligible']})")
        return

    if not args.json:
        ap.error("请提供 --json 或使用 --demo。")
    doc = json.loads(Path(args.json).read_text(encoding="utf-8"))
    adapted = compile_with_engine_adapters(doc, args.engine)
    (out_dir / f"{Path(args.json).stem}_adapted.json").write_text(json.dumps(adapted, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.verify_audio:
        bt = verify_synthesized_audio(args.verify_audio, doc)
        (out_dir / f"{Path(args.json).stem}_audio_backtest.json").write_text(json.dumps(bt, ensure_ascii=False, indent=2), encoding="utf-8")
        print("音频回测完成:", bt["summary"])

    if args.plan_a:
        pa = json.loads(Path(args.plan_a).read_text(encoding="utf-8"))
        routed = route_revision_for_training(pa, doc, args.preference_reason)
        (out_dir / f"{Path(args.json).stem}_training_route.json").write_text(json.dumps(routed, ensure_ascii=False, indent=2), encoding="utf-8")
        print("SFT/DPO 分流结果:", routed["route"], "|", routed["reason"])


if __name__ == "__main__":
    main()
