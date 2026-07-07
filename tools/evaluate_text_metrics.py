#!/usr/bin/env python3
"""Text-answer metrics for Ground3D-LMM predictions (paper Sec. 5.1).

Two independent, selectable stages (``--stages``) — these are the **paper** text/numeric metrics
(segmentation mIoU is the separate Step-1 evaluator, ``tools/test.py``):

* ``ape``   — an LLM extracts numbers (in metres) from the **4 numeric tasks**, then computes
              **APE** ``|s_hat - s|/s * 100%`` and **delta@1.25** success ``max(s_hat/s, s/s_hat) <= 1.25``.
* ``judge`` — a rubric LLM judge scores **Hallucination** + **Completeness** (0-10) on **all** tasks.

Both LLMs run via **HuggingFace transformers** (the weights you pass to ``--llm_model_path``) —
**no vLLM / server required**. Run one stage or both (``--stages ape``, ``--stages judge``,
``--stages all``).

Numeric (APE/delta) tasks: ``distance_estimation``, ``grounded_dimension_reasoning``,
``scale_estimation``, ``scale_comparison_size``. Extraction shapes / parsing corner-cases and the
APE/delta rules follow the reference implementation (per-task JSON, missing ``pred`` -> 0.0, GT==0
pairs skipped, ``pred==0`` -> delta=inf / APE=100, tau=1.25, mean-over-pairs then macro-average).

Predictions may be nested single-turn ``{task_level: {key: {sub_task: [ {question, gt_answer,
pred_answer}, ... ]}}}`` or multi-turn ``{multi_turn_qa_data: [ [ {...}, ... ], ... ]}`` and spread
across sub-task sub-directories — both are handled (recursive discovery + task-aware flatten).

Examples::

    python tools/evaluate_text_metrics.py --input preds/ --output out.json --stages ape \
        --llm_model_path Qwen/Qwen3-4B-Instruct-2507 --gpus 0,1,2,3
    python tools/evaluate_text_metrics.py --input preds/ --output out.json --stages judge ...
    python tools/evaluate_text_metrics.py --selftest        # APE/delta unit test (no LLM, no data)

Deps: ``transformers`` + ``torch`` (for the LLM). No nltk/rouge/sklearn needed.
"""
import os
import re
import json
import math
import argparse
from pathlib import Path
from statistics import mean
from typing import List, Dict, Any, Optional, Tuple

# ---- the only 4 tasks that get LLM number-extraction + APE/delta ----
NUMERIC_TASKS = {
    "distance_estimation",
    "grounded_dimension_reasoning",
    "scale_estimation",
    "scale_comparison_size",
}

_SPECIAL = re.compile(r"<\|im_end\|>|<\|endoftext\|>|</?p>|<SEG>|</?b>|<name>|<sentence>")


def clean_text(t: str) -> str:
    """Strip grounding/special tokens so the judge sees the natural-language answer."""
    return _SPECIAL.sub(" ", t or "").strip()


# =========================== task-aware flatten ===========================
def flatten_with_task(data) -> List[Dict[str, Any]]:
    """Collect every ``{question?, gt_answer, pred_answer}`` sample, tagging each with ``_task``
    (the enclosing dict key — the sub-task for single-turn, ``multi_turn_qa_data`` for multi-turn)."""
    out: List[Dict[str, Any]] = []

    def walk(x, key=None):
        if isinstance(x, dict):
            if "pred_answer" in x and "gt_answer" in x:
                s = dict(x)
                s["_task"] = (x.get("meta_info", {}) or {}).get("group") or key
                out.append(s)
            else:
                for k, v in x.items():
                    walk(v, k)
        elif isinstance(x, list):
            for v in x:
                walk(v, key)

    walk(data)
    return out


_KNOWN_TASKS = NUMERIC_TASKS | {
    "functional_part_grounding", "functional_object_grounding",
    "relative_position_forward_reasoning", "relative_depth_forward",
    "existence_verification", "multi_turn_qa_data",
}


def _task_from_path(path: Path) -> Optional[str]:
    for part in path.parts[::-1]:
        if part in _KNOWN_TASKS:
            return part
    return None


def gather_samples(input_path: Path) -> List[Dict[str, Any]]:
    files = [input_path] if input_path.is_file() else sorted(input_path.rglob("*.json"))
    if not files:
        raise FileNotFoundError(f"No .json prediction files found under {input_path}")
    samples: List[Dict[str, Any]] = []
    for f in files:
        try:
            data = json.load(open(f, encoding="utf-8"))
        except Exception as e:
            print(f"[WARN] skipping {f}: {e}")
            continue
        path_task = _task_from_path(f)
        for s in flatten_with_task(data):
            if not s.get("_task") or s["_task"] not in _KNOWN_TASKS:
                s["_task"] = path_task or s.get("_task")
            samples.append(s)
    return samples


# =========================== local LLM (transformers) ===========================
class LocalLLM:
    """One HF causal LM, sharded across the given GPUs (device_map='auto'); deterministic chat."""

    def __init__(self, model_path: str, gpus: str = ""):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if gpus:
            os.environ["CUDA_VISIBLE_DEVICES"] = gpus
        self.torch = torch
        print(f"[LLM] loading {model_path} (transformers, no vLLM) ...")
        self.tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True).eval()

    def chat(self, system: str, user: str, max_new_tokens: int = 512) -> str:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = self.tok(text, return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False,
                                      repetition_penalty=1.05)
        return self.tok.decode(out[0][ids["input_ids"].shape[-1]:], skip_special_tokens=True)


# =========================== LLM number extraction (Sec. 2) ===========================
EXTRACTION_SYSTEM_PROMPT = """You are a precise extraction assistant. You will receive a task type, a question, a ground-truth answer (gt_answer), and a predicted answer (pred_answer). Your job is to extract numerical values (in meters) according to the task type and return a single JSON object.

TASK TYPES AND EXTRACTION RULES:

1) distance_estimation
   - Extract exactly one distance in meters from each answer.
   - Output: {"extracted_gt": {"distance_m": <float>}, "extracted_pred": {"distance_m": <float>}}
   - If a value is missing in an answer, use 0.0 for that key so the metric can penalize.

2) grounded_dimension_reasoning AND scale_estimation
   - Extract length_m, width_m, thickness_m (in that order: length x width x thickness) when the question asks for dimensions/size.
   - If the question asks only for one dimension (e.g. only thickness, only width, only length), extract only that dimension for both gt and pred.
   - If the question asks for distance from the camera (e.g. "how far from the camera"), also extract "distance_from_camera_m" for both answers.
   - Output: {"extracted_gt": {"length_m"?: float, "width_m"?: float, "thickness_m"?: float, "distance_from_camera_m"?: float}, "extracted_pred": {...}}
   - Use only the keys that apply to the question. If a value is present in gt but missing in pred, use 0.0 for pred.

3) scale_comparison_size
   - Extract dimensions for TWO objects: object1 (first mentioned) and object2 (second mentioned).
   - Each object: [length_m, width_m, thickness_m] in that order.
   - Output: {"extracted_gt": {"object1": [l, w, t], "object2": [l, w, t]}, "extracted_pred": {"object1": [l, w, t], "object2": [l, w, t]}}
   - If pred does not contain numbers for an object, use [0.0, 0.0, 0.0] for that object.

RULES:
- Return ONLY valid JSON. No markdown, no code fences, no explanation.
- All numbers in meters as floats.
- If gt has a value and pred does not, set the pred value to 0.0 for that key."""


def build_extraction_user(task: str, question: str, gt: str, pred: str) -> str:
    return (f"Task: {task}\n\nQuestion: {question}\n\ngt_answer: {gt}\n\npred_answer: {pred}\n\n"
            "Extract the required numbers and return a single JSON object with keys extracted_gt and "
            "extracted_pred as specified for this task. Output only the JSON.")


def parse_extraction(resp: str, task: str) -> Tuple[Optional[dict], Optional[dict]]:
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", (resp or "").strip(), flags=re.S).strip()
    blob = re.search(r"\{.*\}", s, re.S)
    for cand in (s, blob.group(0) if blob else None):
        if not cand:
            continue
        try:
            d = json.loads(cand)
            if isinstance(d, dict) and "extracted_gt" in d and "extracted_pred" in d:
                return d["extracted_gt"], d["extracted_pred"]
        except Exception:
            pass
    if task == "distance_estimation":   # regex fallback — distance only
        nums = re.findall(r"(\d+\.?\d*)\s*(?:m|meters?|metres?)", resp or "")
        if len(nums) >= 2:
            return {"distance_m": float(nums[0])}, {"distance_m": float(nums[1])}
        if len(nums) == 1:
            return {"distance_m": float(nums[0])}, {"distance_m": 0.0}
    return None, None


def fill_missing_with_zero(eg: dict, ep: dict) -> dict:
    ep = dict(ep or {})
    for k, v in (eg or {}).items():
        if k not in ep or ep[k] is None:
            ep[k] = [0.0, 0.0, 0.0] if isinstance(v, list) else 0.0
    return ep


# =========================== APE / delta (Sec. 3) ===========================
def _f(x):
    try:
        return float(x)
    except Exception:
        return None


def get_pairs_for_task(task: str, eg: dict, ep: dict) -> List[Tuple[float, float]]:
    pairs: List[Tuple[float, float]] = []
    if task == "distance_estimation":
        g = _f((eg or {}).get("distance_m"))
        if g is not None:
            p = _f((ep or {}).get("distance_m", 0.0))
            pairs.append((g, p if p is not None else 0.0))
    elif task in ("grounded_dimension_reasoning", "scale_estimation"):
        for k, gv in (eg or {}).items():
            if isinstance(gv, list):
                continue
            g = _f(gv)
            if g is None:
                continue
            p = _f((ep or {}).get(k, 0.0))
            pairs.append((g, p if p is not None else 0.0))
    elif task == "scale_comparison_size":
        for obj in ("object1", "object2"):
            gl = (eg or {}).get(obj, [0.0, 0.0, 0.0])
            pl = (ep or {}).get(obj, [0.0, 0.0, 0.0])
            if not (isinstance(gl, list) and len(gl) == 3):
                gl = [0.0, 0.0, 0.0]
            if not (isinstance(pl, list) and len(pl) == 3):
                pl = [0.0, 0.0, 0.0]
            for i in range(3):
                g = _f(gl[i])
                if g is None:
                    continue
                p = _f(pl[i])
                pairs.append((g, p if p is not None else 0.0))
    return pairs


def ape_delta_from_pairs(pairs: List[Tuple[float, float]], tau: float = 1.25) -> Optional[Dict[str, float]]:
    apes, ratios, succ = [], [], []
    for g, p in pairs:
        if g == 0:                      # GT==0 pairs are EXCLUDED (not scored)
            continue
        apes.append(abs((g - p) / g) * 100.0)
        r = math.inf if p == 0 else max(p / g, g / p)
        ratios.append(r)
        succ.append(1.0 if r <= tau else 0.0)
    if not apes:
        return None
    return {"ape_percent": round(mean(apes), 4),
            "delta_ratio": (math.inf if math.inf in ratios else round(mean(ratios), 4)),
            "delta_success": round(mean(succ), 4),
            "n_pairs": len(apes)}


# =========================== stage: ape ===========================
def stage_ape(samples, llm: "LocalLLM", tau: float) -> Dict[str, Any]:
    from tqdm import tqdm
    per_task: Dict[str, List[Dict[str, float]]] = {}
    n_fail = 0
    numeric = [s for s in samples if s.get("_task") in NUMERIC_TASKS]
    print(f"[ape] {len(numeric)} numerical-task samples (of {len(samples)})")
    for s in tqdm(numeric, desc="extract+ape"):
        task = s["_task"]
        resp = llm.chat(EXTRACTION_SYSTEM_PROMPT,
                        build_extraction_user(task, s.get("question", ""),
                                              s.get("gt_answer", ""), s.get("pred_answer", "")))
        eg, ep = parse_extraction(resp, task)
        if eg is None:
            n_fail += 1
            continue
        ep = fill_missing_with_zero(eg, ep)
        res = ape_delta_from_pairs(get_pairs_for_task(task, eg, ep), tau=tau)
        if res is not None:
            per_task.setdefault(task, []).append(res)
    summary: Dict[str, Any] = {"tau": tau, "extraction_failures": n_fail, "per_task": {}}
    all_qa: List[Dict[str, float]] = []
    for task, lst in per_task.items():
        all_qa += lst
        summary["per_task"][task] = {
            "n": len(lst),
            "APE_percent": round(mean(x["ape_percent"] for x in lst), 4),
            "delta@1.25_success_rate": round(mean(x["delta_success"] for x in lst), 4),
        }
    if all_qa:
        summary["overall"] = {
            "n": len(all_qa),
            "APE_percent": round(mean(x["ape_percent"] for x in all_qa), 4),
            "delta@1.25_success_rate": round(mean(x["delta_success"] for x in all_qa), 4),
        }
    return summary


# =========================== stage: judge (paper: hallucination + completeness) ===========================
JUDGE_SYSTEM = ("You are an evaluation assistant for 3D indoor-scene question answering. Given a "
                "question, a ground-truth answer, and a model's predicted answer, score the prediction. "
                "Return ONLY a JSON object, nothing else.")
JUDGE_USER = """Question: {q}
Ground-truth answer: {gt}
Predicted answer: {pred}

Score each criterion on a 0-10 scale (paper rubric):
- hallucination: 10 = no unsupported/fabricated claims about the scene (fully grounded in the ground truth); 0 = mostly hallucinated.
- completeness: 10 = the prediction addresses every quantity/object the question asks for; 0 = misses them.

Return exactly: {{"hallucination": <0-10>, "completeness": <0-10>}}"""


def stage_judge(samples, llm: "LocalLLM") -> Dict[str, Any]:
    from tqdm import tqdm
    h_sum = c_sum = 0.0
    n = 0
    for s in tqdm(samples, desc="judge"):
        resp = llm.chat(JUDGE_SYSTEM, JUDGE_USER.format(q=clean_text(s.get("question", "")),
                                                        gt=clean_text(s.get("gt_answer", "")),
                                                        pred=clean_text(s.get("pred_answer", ""))),
                        max_new_tokens=64)
        try:
            m = re.search(r"\{.*\}", resp, re.S)
            d = json.loads(m.group(0))
            h, c = float(d["hallucination"]), float(d["completeness"])
        except Exception:
            continue
        h_sum += h
        c_sum += c
        n += 1
    if n == 0:
        return {"n_judged": 0}
    return {"hallucination": round(h_sum / n, 4), "completeness": round(c_sum / n, 4), "n_judged": n}


# =========================== stage: GM-delta (paper, supplementary) ===========================
def _mask_iou(gt, pred) -> Optional[float]:
    """Mean instance IoU for one QA's masks (gt/pred each [n_obj, N] bool), matching the seg metric."""
    import numpy as np

    def to2d(m):
        m = np.asarray(m, dtype=object) if isinstance(m, list) else np.asarray(m)
        if m.dtype == object:
            m = np.stack([np.asarray(r) for r in m]) if m.ndim else np.asarray(m.item())
        m = np.asarray(m)
        return m[None, :] if m.ndim == 1 else m

    g, p = to2d(gt).astype(bool), to2d(pred).astype(bool)
    k = min(g.shape[0], p.shape[0])
    if k == 0:
        return None
    g, p = g[:k], p[:k]
    valid = g.sum(-1) > 0
    if valid.sum() == 0:
        return None
    ious = (g & p).sum(-1) / ((g | p).sum(-1) + 1e-8)
    return float(ious[valid].mean())


def stage_gmdelta(input_path: Path, mask_dump_dir: Path, llm: "LocalLLM",
                  tasks, iou_thr: float = 0.3, tau: float = 1.25) -> Dict[str, Any]:
    """GM-delta (Grounded-Measurement): a prediction succeeds iff **mask IoU > iou_thr AND numeric
    delta <= tau**. **Auto-discovers** every ``<dataset>/.../<numeric task>/`` prediction directory
    under ``--input`` (both ScanNet and ScanNet++), lines each up with the mirrored per-QA mask dump
    under ``--mask_dump_dir`` (produced by a Step-1 run with ``--mask-dump``), and reports GM-delta
    **per (dataset, task) + overall**. ``tasks`` = set of sub-tasks to include (e.g. the 4 numeric)."""
    import numpy as np
    from collections import defaultdict
    from tqdm import tqdm
    pred_files = [input_path] if input_path.is_file() else sorted(input_path.rglob("*.json"))
    base = input_path if input_path.is_dir() else input_path.parent
    groups: Dict[Path, List[Path]] = defaultdict(list)
    for pf in pred_files:
        groups[pf.parent].append(pf)

    per_group: Dict[str, Any] = {}
    tot = {"n": 0, "succ": 0, "iou_ok": 0, "delta_ok": 0}
    for tdir in sorted(groups, key=str):
        task = tdir.name
        if task not in tasks:                      # only the requested (numeric) tasks
            continue
        try:
            rel = tdir.relative_to(base)
        except ValueError:
            rel = Path(".")
        mdir = mask_dump_dir / rel                 # mirror of the prediction tree
        ds = next((p for p in tdir.parts if p in ("scannet", "scannetpp")), "data")
        g = {"n": 0, "succ": 0, "iou_ok": 0, "delta_ok": 0}
        for pf in tqdm(groups[tdir], desc=f"gm-δ {ds}/{task}", leave=False):
            mp, mg = mdir / f"{pf.stem}_qa_pred_masks.npy", mdir / f"{pf.stem}_qa_gt_masks.npy"
            if not (mp.exists() and mg.exists()):
                continue
            samples = [s for s in flatten_with_task(json.load(open(pf, encoding="utf-8")))
                       if s.get("_task") in (task, None)]
            pred_masks, gt_masks = list(np.load(mp, allow_pickle=True)), list(np.load(mg, allow_pickle=True))
            for i, s in enumerate(samples):
                if i >= len(pred_masks) or i >= len(gt_masks):
                    break
                iou = _mask_iou(gt_masks[i], pred_masks[i])
                if iou is None:
                    continue
                eg, ep = parse_extraction(
                    llm.chat(EXTRACTION_SYSTEM_PROMPT,
                             build_extraction_user(task, s.get("question", ""),
                                                   s.get("gt_answer", ""), s.get("pred_answer", ""))), task)
                res = (ape_delta_from_pairs(get_pairs_for_task(task, eg, fill_missing_with_zero(eg, ep)), tau=tau)
                       if eg is not None else None)
                delta_ok = (res is not None and res["delta_success"] >= 1.0)
                iou_ok = iou > iou_thr
                g["n"] += 1
                g["iou_ok"] += int(iou_ok)
                g["delta_ok"] += int(delta_ok)
                g["succ"] += int(iou_ok and delta_ok)
        if g["n"]:
            per_group[f"{ds}/{task}"] = {
                "n": g["n"],
                "GM_delta_success_rate": round(g["succ"] / g["n"], 4),
                "mask_iou>thr_rate": round(g["iou_ok"] / g["n"], 4),
                "delta<=tau_rate": round(g["delta_ok"] / g["n"], 4)}
            for k in tot:
                tot[k] += g[k]
    out: Dict[str, Any] = {"iou_thr": iou_thr, "tau": tau, "per_group": per_group}
    if tot["n"]:
        out["overall"] = {"n": tot["n"], "GM_delta_success_rate": round(tot["succ"] / tot["n"], 4)}
    else:
        out["error"] = ("no joined (scene, task) groups — run the numeric sub-tasks with --mask-dump, "
                        "then point --input at the prediction root and --mask_dump_dir at the mask root")
    return out


# =========================== self-test (reference fixtures T1-T4) ===========================
def selftest() -> int:
    cases = [
        ("distance_estimation", {"distance_m": 0.899}, {"distance_m": 3.5}, 289.3215, 3.8932, 0.0),
        ("grounded_dimension_reasoning", {"length_m": 1.78, "width_m": 0.78, "thickness_m": 0.53},
         {"length_m": 2.1, "width_m": 0.9, "thickness_m": 0.7}, 21.8125, 1.2181, 0.6667),
        ("scale_estimation", {"length_m": 1.78, "width_m": 0.78, "thickness_m": 0.53},
         {"length_m": 2.0, "width_m": 0.8, "thickness_m": 0.7}, 15.6664, 1.1567, 0.6667),
        ("scale_comparison_size", {"object1": [0.13, 0.09, 0.05], "object2": [0.61, 0.51, 0.09]},
         {"object1": [0.0, 0.0, 0.0], "object2": [0.0, 0.0, 0.0]}, 100.0, math.inf, 0.0),
    ]
    ok = True
    for task, eg, ep, ape_e, ratio_e, succ_e in cases:
        r = ape_delta_from_pairs(get_pairs_for_task(task, eg, ep))
        pa = abs(r["ape_percent"] - ape_e) < 1e-2
        pr = (r["delta_ratio"] == math.inf) if ratio_e == math.inf else abs(r["delta_ratio"] - ratio_e) < 1e-2
        ps = abs(r["delta_success"] - succ_e) < 1e-3
        ok &= pa and pr and ps
        print(f"  {task:30} ape={r['ape_percent']:<10} delta={r['delta_ratio']:<8} "
              f"succ={r['delta_success']}  -> {'OK' if (pa and pr and ps) else 'FAIL'}")
    print("SELFTEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# =========================== main ===========================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", help="prediction .json file OR directory (searched recursively)")
    ap.add_argument("--output", default="eval_text_metrics.json")
    ap.add_argument("--stages", default="ape,judge",
                    help="comma list of: ape,judge,gmdelta  (or 'all' = ape,judge)")
    ap.add_argument("--tau", type=float, default=1.25, help="delta success threshold (paper: 1.25)")
    ap.add_argument("--mask_dump_dir", help="(gmdelta) dir with <scene>_qa_pred_masks.npy from a "
                                            "Step-1 run launched with --mask-dump")
    ap.add_argument("--gmdelta_task", default="all",
                    help="(gmdelta) 'all' = the 4 numeric tasks (auto-discovered across datasets), "
                         "or a single task name")
    ap.add_argument("--iou_thr", type=float, default=0.3, help="(gmdelta) mask IoU threshold")
    ap.add_argument("--llm_model_path", default="Qwen/Qwen3-4B-Instruct-2507",
                    help="HF text LLM for ape-extraction + judge (transformers; no vLLM)")
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--selftest", action="store_true", help="run the APE/delta unit test and exit")
    args = ap.parse_args()

    if args.selftest:
        raise SystemExit(selftest())
    if not args.input:
        ap.error("--input is required (unless --selftest)")

    stages = {"ape", "judge"} if args.stages == "all" else {x.strip() for x in args.stages.split(",")}
    samples = gather_samples(Path(args.input))
    if args.max_samples > 0:
        samples = samples[:args.max_samples]
    print(f"Loaded {len(samples)} samples; stages={sorted(stages)}")

    llm = LocalLLM(args.llm_model_path, args.gpus) if (stages & {"ape", "judge", "gmdelta"}) else None
    results: Dict[str, Any] = {"num_samples": len(samples), "stages": sorted(stages)}
    if "ape" in stages:
        results["numeric_ape_delta"] = stage_ape(samples, llm, args.tau)
    if "judge" in stages:
        results["llm_judge"] = stage_judge(samples, llm)
    if "gmdelta" in stages:
        if not args.mask_dump_dir:
            ap.error("--stages gmdelta requires --mask_dump_dir (per-QA masks from a Step-1 --mask-dump run)")
        gtasks = NUMERIC_TASKS if args.gmdelta_task == "all" else {args.gmdelta_task}
        results["gm_delta"] = stage_gmdelta(Path(args.input), Path(args.mask_dump_dir), llm,
                                            gtasks, iou_thr=args.iou_thr, tau=args.tau)

    print("\n================ Results ================")
    print(json.dumps(results, indent=2, default=str))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(args.output, "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=str)
    print(f"\nSaved -> {args.output}")


if __name__ == "__main__":
    main()
