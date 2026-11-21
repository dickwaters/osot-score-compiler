#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OSOT 统一数据协议 (osot-unified-v2.0) 公共模块
=============================================
供工具 01、工具 02、工具 03 及前端工作台共享，确保：
1. 全链路统一使用同一份 JSON 数据结构 (osot-unified-v2.0)；
2. 严格隔离三层数据：
   - tier1_acoustic      : 客观声学物理测量（不可人工篡改）
   - tier2_machine_pred  : 机器初始预测 + 置信度 + 白箱特征归因
   - tier3_human_gold    : 人工修正终稿 + 完整 edit_history（机器初判 -> 人工修改理由）
3. 向下兼容 D:\\otos\\osot-annotator v0.1 MVP 导出的单层 JSON（自动无损迁移）。
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = "osot-unified-v2.0"

STRESS_TO_UNIFIED = {"primary": "S2", "secondary": "S1", "none": "S0", "S2": "S2", "S1": "S1", "S0": "S0"}
PAUSE_TO_UNIFIED = {
    "long": "B3",
    "short": "B2",
    "connect": "B0_CONNECT",
    "none": "B0",
    "B3": "B3",
    "B2": "B2",
    "B1": "B1",
    "B0_CONNECT": "B0_CONNECT",
    "B0": "B0",
}


def empty_unified_doc(
    doc_id: str = "untitled",
    title: str = "未命名播音稿",
    speaker_id: str = "spk_unknown",
    speaker_group: str = "unspecified",
    genre_style: str = "文艺朗诵",
    global_rhythm: str = "舒缓型",
    audio_wav: str = "",
    duration_sec: float = 0.0,
) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "meta": {
            "id": doc_id,
            "title": title,
            "speaker_id": speaker_id,
            "speaker_group": speaker_group,  # pro | normal | unspecified
            "genre_style": genre_style,
            "global_rhythm": global_rhythm,
            "audio_wav": audio_wav,
            "duration_sec": round(float(duration_sec), 3),
        },
        "raw_text": "",
        "tier1_acoustic": {
            "immutable_notice": "客观物理测量层：严禁人工直接覆盖，仅在对齐时间戳变更后由声学管线重算。",
            "analysis_meta": {},
            "curves": {},
            "tokens": [],
        },
        "tier2_machine_pred": {
            "model_version": "",
            "tokens": [],
            "phrases": [],
        },
        "tier3_human_gold": {
            "annotator_id": "",
            "review_status": "pending",  # pending | reviewed | arbitrated
            "tokens": [],
            "edit_history": [],
        },
    }


def append_edit_history(
    doc: Dict[str, Any],
    token_index: int,
    field: str,
    old_val: Any,
    new_val: Any,
    change_type: str = "expression_choice",  # expression_choice | alignment_fix | typo_fix
    editor: str = "human",
    reason: str = "",
) -> None:
    """在 tier3_human_gold.edit_history 中追加不可丢弃的修改轨迹。"""
    if old_val == new_val:
        return
    doc.setdefault("tier3_human_gold", {}).setdefault("edit_history", []).append(
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "token_index": int(token_index),
            "field": field,
            "from": old_val,
            "to": new_val,
            "change_type": change_type,
            "editor": editor,
            "reason": reason,
        }
    )


def migrate_to_unified(doc: Dict[str, Any], default_speaker_id: str = "spk_unknown") -> Dict[str, Any]:
    """将任意历史格式（osot-annotator v0.1 / 工具01 v1.0 / v0.2）无损迁移为 osot-unified-v2.0。"""
    if doc.get("schema_version") == SCHEMA_VERSION and "tier1_acoustic" in doc:
        doc.setdefault("meta", {}).setdefault("speaker_id", default_speaker_id)
        doc.setdefault("tier3_human_gold", {}).setdefault("edit_history", [])
        return doc

    # 兼容工具 01 v1.0 (osot-3tier-prosody-v1.0)
    if "tier1_objective_acoustic" in doc:
        u = empty_unified_doc(
            doc_id=doc.get("source_id", doc.get("id", "untitled")),
            title=doc.get("title", "未命名"),
            speaker_id=doc.get("speaker_id", default_speaker_id),
            duration_sec=float(doc.get("duration_sec", 0.0)),
        )
        t1_old = doc["tier1_objective_acoustic"]
        t2_old = doc.get("tier2_algorithmic_hypothesis", {})
        t3_old = doc.get("tier3_human_gold", {})
        u["tier1_acoustic"]["analysis_meta"] = doc.get("analysis_meta", {})
        u["tier1_acoustic"]["curves"] = t1_old.get("curves", {})
        u["tier1_acoustic"]["tokens"] = t1_old.get("tokens", [])
        u["tier2_machine_pred"]["tokens"] = t2_old.get("tokens", [])
        u["tier2_machine_pred"]["phrases"] = t2_old.get("phrases", [])
        u["tier3_human_gold"]["annotator_id"] = t3_old.get("annotator_id", "")
        u["tier3_human_gold"]["review_status"] = t3_old.get("review_status", "pending")
        u["tier3_human_gold"]["tokens"] = t3_old.get("tokens", [])
        u["raw_text"] = "".join(
            t.get("char", "") + t.get("punct_after", "") for t in u["tier1_acoustic"]["tokens"]
        )
        return u

    # 兼容 osot-annotator v0.1 / v0.2 单层 JSON
    tokens = doc.get("tokens", [])
    u = empty_unified_doc(
        doc_id=doc.get("id", "untitled"),
        title=doc.get("title", "未命名"),
        speaker_id=doc.get("speaker_id", default_speaker_id),
        speaker_group=doc.get("speaker_group", "unspecified"),
        genre_style=doc.get("genre_style", "文艺朗诵"),
        global_rhythm=doc.get("global_rhythm", "舒缓型"),
        duration_sec=float(doc.get("duration_sec", 0.0)),
    )
    u["raw_text"] = "".join(t.get("char", "") + t.get("punct_after", "") for t in tokens)

    for i, t in enumerate(tokens):
        feat = t.get("features", {})
        s_ms, e_ms = int(t.get("start_ms", 0)), int(t.get("end_ms", 0))
        dur_ms = int(feat.get("duration_ms", max(1, e_ms - s_ms)))
        gap_ms = int(feat.get("gap_after_ms", 0))
        u["tier1_acoustic"]["tokens"].append(
            {
                "index": i,
                "char": t.get("char", ""),
                "punct_after": t.get("punct_after", ""),
                "start_ms": s_ms,
                "end_ms": e_ms,
                "duration_ms": dur_ms,
                "gap_after_ms": gap_ms,
                "energy_db": float(feat.get("energy_db", 0.0)),
                "f0_local_span_st": float(feat.get("f0_range_st", 0.0)),
                "f0_macro_mean_st": 0.0,
            }
        )
        s_uni = STRESS_TO_UNIFIED.get(str(t.get("stress", "none")), "S0")
        p_uni = PAUSE_TO_UNIFIED.get(str(t.get("pause_after", "none")), "B0")
        v02 = t.get("v02", {})
        u["tier2_machine_pred"]["tokens"].append(
            {
                "index": i,
                "char": t.get("char", ""),
                "candidate_stress": s_uni,
                "stress_prob": float(feat.get("prominence_score", 0.0)),
                "candidate_pause": {"level": p_uni, "confidence": 0.5, "reason": "migrated_from_v0.1"},
                "candidate_voice_quality": v02.get("voice_quality", "Modal"),
                "voice_quality_confidence": 0.5,
            }
        )
        u["tier3_human_gold"]["tokens"].append(
            {
                "index": i,
                "char": t.get("char", ""),
                "human_stress": s_uni,
                "human_stress_semantic_type": v02.get("stress_type", ""),
                "human_pause": p_uni,
                "human_pause_method": v02.get("pause_type", ""),
                "human_voice_quality": v02.get("voice_quality", "UNVERIFIED"),
                "human_breath_action": v02.get("breath_action", ""),
                "human_sentence_contour": v02.get("sentence_contour", ""),
                "knowledge_tags": list(t.get("knowledge_tags", [])),
            }
        )
    return u


def resolve_effective_tokens(doc: Dict[str, Any], layer: str = "gold_or_pred") -> List[Dict[str, Any]]:
    """
    统一读取视图：
    - layer='pred'         : 严格只读 tier2_machine_pred（机器初始预测）
    - layer='gold'         : 严格只读 tier3_human_gold（人工终稿）
    - layer='gold_or_pred' : 优先人工终稿，未审校 (UNVERIFIED) 时回退机器预测
    """
    u = migrate_to_unified(doc)
    t1 = u["tier1_acoustic"]["tokens"]
    t2 = u["tier2_machine_pred"]["tokens"]
    t3 = u["tier3_human_gold"]["tokens"]
    out: List[Dict[str, Any]] = []
    for i, a in enumerate(t1):
        p = t2[i] if i < len(t2) else {}
        g = t3[i] if i < len(t3) else {}

        p_stress = p.get("candidate_stress", "S0")
        g_stress = g.get("human_stress", "UNVERIFIED")
        p_pause = (p.get("candidate_pause") or {}).get("level", "B0")
        g_pause = g.get("human_pause", "UNVERIFIED")
        p_vq = p.get("candidate_voice_quality", "Modal")
        g_vq = g.get("human_voice_quality", "UNVERIFIED")

        if layer == "pred":
            stress, pause, vq = p_stress, p_pause, p_vq
        elif layer == "gold":
            stress = g_stress if g_stress != "UNVERIFIED" else "S0"
            pause = g_pause if g_pause != "UNVERIFIED" else "B0"
            vq = g_vq if g_vq != "UNVERIFIED" else "Modal"
        else:
            stress = g_stress if g_stress != "UNVERIFIED" else p_stress
            pause = g_pause if g_pause != "UNVERIFIED" else p_pause
            vq = g_vq if g_vq != "UNVERIFIED" else p_vq

        out.append(
            {
                **a,
                "stress": STRESS_TO_UNIFIED.get(str(stress), "S0"),
                "pause_after": PAUSE_TO_UNIFIED.get(str(pause), "B0"),
                "voice_quality": vq,
                "stress_prob": p.get("stress_prob", 0.0),
                "is_elongated": p.get("is_elongated", a.get("duration_ms", 200) >= 380),
                "human_stress_semantic_type": g.get("human_stress_semantic_type", ""),
                "human_pause_method": g.get("human_pause_method", ""),
                "human_breath_action": g.get("human_breath_action", ""),
                "human_sentence_contour": g.get("human_sentence_contour", ""),
                "knowledge_tags": g.get("knowledge_tags", []),
            }
        )
    return out
