# manual_eval_gsm8k.py
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def _get_gold_raw(doc: dict) -> Optional[str]:
    # GSM8K 常见：answer / target
    for k in ["answer", "target", "gold", "label"]:
        if k in doc and doc[k] is not None:
            return str(doc[k])
    return None


def _normalize_num_str(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    s = str(s)

    # gold 常见格式："... #### 42"
    if "####" in s:
        s = s.split("####")[-1]

    s = s.strip()
    nums = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", s)
    if not nums:
        return None
    return nums[-1].replace(",", "")


def extract_pred_answer(resp_text: Optional[str]) -> Optional[str]:
    if resp_text is None:
        return None
    t = str(resp_text)

    # 优先抓 ####
    if "####" in t:
        cand = t.split("####")[-1]
        v = _normalize_num_str(cand)
        if v is not None:
            return v

    return _normalize_num_str(t)


def exact_match_num(pred: Optional[str], gold: Optional[str]) -> int:
    if pred is None or gold is None:
        return 0
    return 1 if pred == gold else 0


def load_jsonl(path: str) -> List[dict]:
    data = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def index_by_doc_id(data: List[dict]) -> Dict[int, dict]:
    out = {}
    for ex in data:
        if "doc_id" not in ex:
            continue
        out[int(ex["doc_id"])] = ex
    return out


def compute_em_for_ids(
    by_id: Dict[int, dict],
    doc_ids: List[int],
) -> Dict[int, int]:
    em = {}
    for did in doc_ids:
        ex = by_id[did]
        doc = ex.get("doc", {}) or {}
        gold = _normalize_num_str(_get_gold_raw(doc))

        resp = None
        try:
            resp = ex["resps"][0][0]
        except Exception:
            resp = None

        pred = extract_pred_answer(resp)
        em[did] = exact_match_num(pred, gold)
    return em


def stats_first_n(em_by_id: Dict[int, int], doc_ids: List[int], first_n: int) -> dict:
    first_ids = doc_ids[:first_n]
    cnt1 = sum(em_by_id[i] for i in first_ids)
    cnt0 = len(first_ids) - cnt1
    return {"n_checked": len(first_ids), "exact_match_1": cnt1, "exact_match_0": cnt0}


def select_first_k_correct(em_by_id: Dict[int, int], doc_ids: List[int], k: int) -> List[int]:
    out = []
    for did in doc_ids:
        if em_by_id.get(did, 0) == 1:
            out.append(did)
        if len(out) >= k:
            break
    return out


def run_manual_eval_and_select(
    big_path: str,
    small_path: str,
    out_dir: Path,
    select_source: str = "big",
    max_examples: int = 100,
    first_n: int = 100,
    write_debug_first: int = 200,
    overwrite: bool = False,
) -> Tuple[List[int], dict]:
    """
    读取 big/small samples jsonl，对某个 source（big 或 small）的输出做手写 EM，
    统计前 first_n 的 EM 分布，并挑出前 max_examples 个 EM=1 的 doc_id。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    selected_path = out_dir / "selected_doc_ids.json"
    stats_path = out_dir / "stats_firstN.json"
    debug_path = out_dir / f"em_debug_first{write_debug_first}.json"

    if selected_path.exists() and not overwrite:
        # 直接复用
        with open(selected_path, "r") as f:
            selected = json.load(f)
        with open(stats_path, "r") as f:
            stats = json.load(f) if stats_path.exists() else {}
        return selected, stats

    big_by_id = index_by_doc_id(load_jsonl(big_path))
    small_by_id = index_by_doc_id(load_jsonl(small_path))
    common_ids = sorted(set(big_by_id) & set(small_by_id))

    if select_source not in ("big", "small"):
        raise ValueError("select_source must be 'big' or 'small'")

    source_by_id = big_by_id if select_source == "big" else small_by_id

    em_by_id = compute_em_for_ids(source_by_id, common_ids)
    stats = stats_first_n(em_by_id, common_ids, first_n=first_n)

    selected = select_first_k_correct(em_by_id, common_ids, k=max_examples)

    # write outputs
    with open(stats_path, "w") as f:
        json.dump(
            {**stats, "select_source": select_source, "note": "Manual EM via number extraction."},
            f,
            indent=2,
        )
    with open(selected_path, "w") as f:
        json.dump(selected, f, indent=2)

    # debug
    dbg = []
    for did in common_ids[:write_debug_first]:
        ex = source_by_id[did]
        doc = ex.get("doc", {}) or {}
        gold = _normalize_num_str(_get_gold_raw(doc))
        resp = ex.get("resps", [[None]])[0][0]
        pred = extract_pred_answer(resp)
        dbg.append({"doc_id": did, "em": em_by_id[did], "gold": gold, "pred": pred})
    with open(debug_path, "w") as f:
        json.dump(dbg, f, indent=2)

    return selected, stats
