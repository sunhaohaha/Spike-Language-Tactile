from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd

MODALITIES = ("materials", "shape", "temperature", "weight")

PUBLIC_SYSTEM_PROMPT = """
You are an evaluator for multimodal tactile-language classification.
Given candidate labels, support examples, and query samples, select one candidate label for each query sample.
Return only result lines in the required format. Do not provide explanations.
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Public redacted evaluation pipeline for multimodal tactile-language "
            "classification using a chat completion API."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=Path("test/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("test/outputs/llm_tactile_eval_public"))
    parser.add_argument("--model-name", type=str, default=os.environ.get("LLM_MODEL_NAME", "llm-chat-model"))
    parser.add_argument("--base-url", type=str, default=os.environ.get("LLM_BASE_URL", ""))
    parser.add_argument("--api-key", type=str, default="")
    parser.add_argument("--system-prompt-file", type=Path, default=None)
    parser.add_argument("--train-per-class", type=int, default=30)
    parser.add_argument("--val-per-class", type=int, default=10)
    parser.add_argument("--support-per-class", type=int, default=5)
    parser.add_argument("--samples-per-class-per-group", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--test-class-mode", choices=("open_set", "closed_set"), default="open_set")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def save_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_system_prompt(prompt_file: Path | None) -> Tuple[str, str]:
    if prompt_file is None:
        return PUBLIC_SYSTEM_PROMPT, "default_public_prompt"
    prompt_text = prompt_file.read_text(encoding="utf-8").strip()
    if not prompt_text:
        raise ValueError(f"Empty system prompt file: {prompt_file}")
    return prompt_text, "external_prompt_file"


def build_label_aliases(labels: Sequence[str]) -> Dict[str, str]:
    unique_labels = sorted(set(map(str, labels)))
    return {label: f"class_{index:03d}" for index, label in enumerate(unique_labels, start=1)}


def redact_label(label: object, aliases: Dict[str, str]) -> str:
    label_str = str(label)
    if label_str in aliases:
        return aliases[label_str]
    if label_str == "Unknown_Parse_Error":
        return "parse_error"
    return "non_candidate_output"


def load_data(data_dir: Path) -> Dict[str, pd.DataFrame]:
    data: Dict[str, pd.DataFrame] = {}
    for split in ("train", "test"):
        for modality in MODALITIES:
            file_path = data_dir / f"{modality}_X_{split}.xlsx"
            data[f"{modality}_{split}"] = pd.read_excel(file_path, header=None)
        data[f"y_{split}"] = pd.read_excel(data_dir / f"Y_{split}.xlsx", header=None)
    return data


def build_records(raw: Dict[str, pd.DataFrame], split: str, indices: Sequence[int]) -> List[Dict[str, str]]:
    labels = raw[f"y_{split}"][0].astype(str)
    records: List[Dict[str, str]] = []
    for index in indices:
        record = {"row_index": str(int(index)), "label": str(labels.iloc[index])}
        for modality in MODALITIES:
            record[modality] = str(raw[f"{modality}_{split}"].iloc[index, 0])
        records.append(record)
    return records


def get_class_lists(raw: Dict[str, pd.DataFrame]) -> Tuple[List[str], List[str], List[str]]:
    known_classes = sorted(raw["y_train"][0].astype(str).unique().tolist())
    test_classes = sorted(raw["y_test"][0].astype(str).unique().tolist())
    open_classes = [label for label in test_classes if label not in known_classes]
    return known_classes, open_classes, known_classes + open_classes


def split_train_validation(
    labels: pd.Series,
    train_per_class: int,
    val_per_class: int,
    seed: int,
) -> Tuple[List[int], List[int]]:
    rng = random.Random(seed)
    train_indices: List[int] = []
    val_indices: List[int] = []
    labels = labels.astype(str)

    for label in sorted(labels.unique().tolist()):
        class_indices = [int(index) for index, value in enumerate(labels.tolist()) if value == label]
        rng.shuffle(class_indices)
        required_count = train_per_class + val_per_class
        if len(class_indices) < required_count:
            raise ValueError(f"Class {label!r} has {len(class_indices)} samples, but {required_count} are required.")
        train_part = sorted(class_indices[:train_per_class])
        val_part = sorted(class_indices[train_per_class:required_count])
        train_indices.extend(train_part)
        val_indices.extend(val_part)
    return sorted(train_indices), sorted(val_indices)


def select_support_records(records: Sequence[Dict[str, str]], support_per_class: int, seed: int) -> List[Dict[str, str]]:
    if support_per_class <= 0:
        return list(records)

    rng = random.Random(seed)
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for record in records:
        grouped[record["label"]].append(record)

    selected: List[Dict[str, str]] = []
    for label in sorted(grouped.keys()):
        class_records = sorted(grouped[label], key=lambda item: int(item["row_index"]))
        rng.shuffle(class_records)
        selected.extend(sorted(class_records[:support_per_class], key=lambda item: int(item["row_index"])))
    return selected


def select_test_indices(y_test: pd.Series, known_classes: Sequence[str], mode: str) -> List[int]:
    if mode == "closed_set":
        known_set = set(known_classes)
        return [index for index, label in enumerate(y_test.astype(str).tolist()) if label in known_set]
    return list(range(len(y_test)))


def build_balanced_groups(
    query_records: Sequence[Dict[str, str]],
    class_order: Sequence[str],
    samples_per_class: int,
) -> List[List[Dict[str, str]]]:
    if samples_per_class <= 0:
        raise ValueError("samples_per_class_per_group must be greater than 0.")

    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for record in query_records:
        grouped[record["label"]].append(record)

    missing = [label for label in class_order if label not in grouped]
    if missing:
        raise ValueError(f"The query set is missing the following classes: {missing}")

    for label in grouped:
        grouped[label] = sorted(grouped[label], key=lambda item: int(item["row_index"]))

    group_count = max((len(grouped[label]) + samples_per_class - 1) // samples_per_class for label in class_order)
    groups: List[List[Dict[str, str]]] = []
    for group_index in range(group_count):
        group_records: List[Dict[str, str]] = []
        start = group_index * samples_per_class
        end = start + samples_per_class
        for label in class_order:
            group_records.extend(grouped[label][start:end])
        if group_records:
            groups.append(group_records)
    return groups


def format_record(record: Dict[str, str]) -> List[str]:
    return [
        f"- materials modality: {record['materials']}",
        f"- shape modality: {record['shape']}",
        f"- temperature modality: {record['temperature']}",
        f"- weight modality: {record['weight']}",
    ]


def build_prompt(
    all_classes: Sequence[str],
    support_records: Sequence[Dict[str, str]],
    query_records: Sequence[Dict[str, str]],
) -> Tuple[str, Dict[str, Dict[str, str]]]:
    support_by_class: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for record in support_records:
        support_by_class[record["label"]].append(record)

    lines: List[str] = []
    lines.append("Candidate labels. The answer for each query must be copied exactly from this list:")
    for label in all_classes:
        lines.append(f"- {label}")

    lines.append("")
    lines.append("Support examples:")
    for label in all_classes:
        if label not in support_by_class:
            continue
        lines.append(f"Label: {label}")
        sorted_records = sorted(support_by_class[label], key=lambda item: int(item["row_index"]))
        for sample_index, record in enumerate(sorted_records, start=1):
            lines.append(f"Support sample {sample_index}:")
            lines.extend(format_record(record))
            lines.append("")

    lines.append("Query samples:")
    query_mapping: Dict[str, Dict[str, str]] = {}
    for index, record in enumerate(query_records, start=1):
        test_id = f"test{index:03d}"
        lines.append(f"Sample {test_id}:")
        lines.extend(format_record(record))
        lines.append("")
        query_mapping[test_id] = record

    lines.append("Return only result lines in the following format:")
    for index in range(1, len(query_records) + 1):
        lines.append(f"Sample test{index:03d} -> label_name")
    return "\n".join(lines), query_mapping


def normalize_label(text: str) -> str:
    text = str(text).strip().strip("`").strip('"').strip("'").rstrip(".;")
    for source, target in {" ": "", "\t": "", "\n": ""}.items():
        text = text.replace(source, target)
    return text


def canonicalize_prediction(text: str, allowed_labels: Sequence[str]) -> str:
    normalized = normalize_label(text)
    label_map = {normalize_label(label): label for label in allowed_labels}
    if normalized in label_map:
        return label_map[normalized]
    for normalized_label, label in label_map.items():
        if normalized and (normalized in normalized_label or normalized_label in normalized):
            return label
    return str(text).strip() or "Unknown_Parse_Error"


def parse_model_output(text: str, allowed_labels: Sequence[str]) -> Dict[str, str]:
    predictions: Dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        match = re.match(r"^(?:Sample\s*)?(?:test[_-]?)?0*(\d+)\s*(?:->|:)\s*(.+?)\s*$", line, flags=re.IGNORECASE)
        if match:
            test_id = f"test{int(match.group(1)):03d}"
            predictions[test_id] = canonicalize_prediction(match.group(2), allowed_labels)
    return predictions


def evaluate_group(
    group_index: int,
    query_mapping: Dict[str, Dict[str, str]],
    predictions: Dict[str, str],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for test_id, record in query_mapping.items():
        predicted_label = predictions.get(test_id, "Unknown_Parse_Error")
        true_label = record["label"]
        rows.append(
            {
                "group_index": group_index,
                "test_id": test_id,
                "true_label": true_label,
                "predicted_label": predicted_label,
                "correct": predicted_label == true_label,
            }
        )
    return rows


def compute_summary(rows: Sequence[Dict[str, object]], label_order: Sequence[str]) -> Dict[str, object]:
    total = len(rows)
    correct = sum(1 for row in rows if bool(row["correct"]))
    per_class_rows = []
    for label in label_order:
        label_rows = [row for row in rows if row["true_label"] == label]
        if not label_rows:
            continue
        label_correct = sum(1 for row in label_rows if bool(row["correct"]))
        per_class_rows.append(
            {
                "label": label,
                "total": len(label_rows),
                "correct": label_correct,
                "accuracy": label_correct / len(label_rows),
            }
        )
    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else None,
        "per_class_accuracy": per_class_rows,
    }


def redact_rows(rows: Sequence[Dict[str, object]], aliases: Dict[str, str]) -> List[Dict[str, object]]:
    redacted: List[Dict[str, object]] = []
    for row in rows:
        redacted.append(
            {
                "group_index": row["group_index"],
                "test_id": row["test_id"],
                "true_label": redact_label(row["true_label"], aliases),
                "predicted_label": redact_label(row["predicted_label"], aliases),
                "correct": row["correct"],
            }
        )
    return redacted


def redact_summary(summary: Dict[str, object], aliases: Dict[str, str]) -> Dict[str, object]:
    redacted = dict(summary)
    per_class = []
    for row in summary.get("per_class_accuracy", []):
        row_dict = dict(row)
        row_dict["label"] = redact_label(row_dict["label"], aliases)
        per_class.append(row_dict)
    redacted["per_class_accuracy"] = per_class
    redacted["redacted"] = True
    return redacted


def save_confusion_matrix(
    rows: Sequence[Dict[str, object]],
    label_order: Sequence[str],
    output_csv: Path,
) -> None:
    labels = list(label_order)
    for row in rows:
        for key in ("true_label", "predicted_label"):
            label = str(row[key])
            if label not in labels:
                labels.append(label)

    matrix = pd.DataFrame(0, index=labels, columns=labels)
    for row in rows:
        matrix.loc[str(row["true_label"]), str(row["predicted_label"])] += 1
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(output_csv, encoding="utf-8")


def public_manifest(
    args: argparse.Namespace,
    known_classes: Sequence[str],
    open_classes: Sequence[str],
    fixed_test_classes: Sequence[str],
    train_indices: Sequence[int],
    val_indices: Sequence[int],
    test_indices: Sequence[int],
    groups: Sequence[Sequence[Dict[str, str]]],
    prompt_source: str,
    system_prompt: str,
    aliases: Dict[str, str],
) -> Dict[str, object]:
    return {
        "release_type": "public_redacted_evaluation_pipeline",
        "redacted_outputs": True,
        "seed": args.seed,
        "train_per_class": args.train_per_class,
        "val_per_class": args.val_per_class,
        "support_per_class": args.support_per_class,
        "samples_per_class_per_group": args.samples_per_class_per_group,
        "test_class_mode": args.test_class_mode,
        "known_class_count": len(known_classes),
        "open_class_count": len(open_classes),
        "candidate_class_count": len(fixed_test_classes),
        "train_indices_count": len(train_indices),
        "validation_indices_count": len(val_indices),
        "test_indices_count": len(test_indices),
        "group_count": len(groups),
        "modalities": list(MODALITIES),
        "system_prompt_source": prompt_source,
        "system_prompt_sha256": sha256_text(system_prompt),
        "model_interface": "chat_completion_api",
        "class_aliases": sorted(set(aliases.values())),
        "note": "Private prompts, credentials, full prompt logs, raw responses, exact labels, paths, and class mappings are not saved.",
    }


def save_prompt_audit(
    group_dir: Path,
    system_prompt: str,
    user_prompt: str,
    support_records: Sequence[Dict[str, str]],
    query_mapping: Dict[str, Dict[str, str]],
    aliases: Dict[str, str],
) -> None:
    group_dir.mkdir(parents=True, exist_ok=True)
    support_counts = Counter(record["label"] for record in support_records)
    query_counts = Counter(record["label"] for record in query_mapping.values())
    support_counts_out = {redact_label(label, aliases): count for label, count in support_counts.items()}
    query_counts_out = {redact_label(label, aliases): count for label, count in query_counts.items()}
    save_json(
        group_dir / "prompt_audit.json",
        {
            "redacted": True,
            "system_prompt_sha256": sha256_text(system_prompt),
            "user_prompt_sha256": sha256_text(user_prompt),
            "user_prompt_character_count": len(user_prompt),
            "support_size": len(support_records),
            "query_size": len(query_mapping),
            "support_class_counts": support_counts_out,
            "query_class_counts": query_counts_out,
            "full_prompt_saved": False,
        },
    )


def get_client(api_key: str, base_url: str):
    resolved_key = api_key or os.environ.get("LLM_API_KEY") or ""
    resolved_base_url = base_url or os.environ.get("LLM_BASE_URL") or ""
    if not resolved_key:
        raise RuntimeError("No API key was provided. Set LLM_API_KEY or pass --api-key locally.")
    if not resolved_base_url:
        raise RuntimeError("No base URL was provided. Set LLM_BASE_URL or pass --base-url locally.")
    from openai import OpenAI

    return OpenAI(api_key=resolved_key, base_url=resolved_base_url)


def call_model(
    client,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
) -> Tuple[str, Dict[str, object]]:
    started_at = datetime.now(timezone.utc)
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        max_tokens=max_tokens,
        stream=False,
    )
    finished_at = datetime.now(timezone.utc)
    message = response.choices[0].message
    content = message.content or ""
    usage = getattr(response, "usage", None)
    timing = {
        "api_started_at_utc": started_at.isoformat(),
        "api_finished_at_utc": finished_at.isoformat(),
        "api_duration_seconds": time.perf_counter() - start,
    }
    if usage is not None:
        timing["prompt_tokens"] = getattr(usage, "prompt_tokens", None)
        timing["completion_tokens"] = getattr(usage, "completion_tokens", None)
        timing["total_tokens"] = getattr(usage, "total_tokens", None)
    return content, timing


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    system_prompt, prompt_source = load_system_prompt(args.system_prompt_file)
    raw = load_data(args.data_dir)
    known_classes, open_classes, all_test_classes = get_class_lists(raw)
    fixed_test_classes = known_classes if args.test_class_mode == "closed_set" else all_test_classes
    label_aliases = build_label_aliases(fixed_test_classes)
    y_train = raw["y_train"][0].astype(str)
    y_test = raw["y_test"][0].astype(str)
    train_indices, val_indices = split_train_validation(y_train, args.train_per_class, args.val_per_class, args.seed)
    test_indices = select_test_indices(y_test, known_classes, args.test_class_mode)
    train_records = build_records(raw, "train", train_indices)
    support_records = select_support_records(train_records, args.support_per_class, args.seed)
    test_records = build_records(raw, "test", test_indices)
    groups = build_balanced_groups(test_records, fixed_test_classes, args.samples_per_class_per_group)

    save_json(
        args.output_dir / "manifest.json",
        public_manifest(
            args=args,
            known_classes=known_classes,
            open_classes=open_classes,
            fixed_test_classes=fixed_test_classes,
            train_indices=train_indices,
            val_indices=val_indices,
            test_indices=test_indices,
            groups=groups,
            prompt_source=prompt_source,
            system_prompt=system_prompt,
            aliases=label_aliases,
        ),
    )

    client = None if args.dry_run else get_client(args.api_key, args.base_url)
    all_rows: List[Dict[str, object]] = []
    group_summaries: List[Dict[str, object]] = []

    for group_index, group_records in enumerate(groups, start=1):
        group_dir = args.output_dir / "groups" / f"group_{group_index:02d}"
        user_prompt, query_mapping = build_prompt(fixed_test_classes, support_records, group_records)
        save_prompt_audit(
            group_dir=group_dir,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            support_records=support_records,
            query_mapping=query_mapping,
            aliases=label_aliases,
        )

        if args.dry_run:
            group_summary = {
                "group_index": group_index,
                "status": "dry_run",
                "query_size": len(group_records),
                "redacted": True,
            }
            save_json(group_dir / "summary.json", group_summary)
            group_summaries.append(group_summary)
            continue

        content, timing = call_model(
            client=client,
            model_name=args.model_name,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=args.max_tokens,
        )
        save_json(
            group_dir / "raw_response_audit.json",
            {
                "raw_response_saved": False,
                "raw_response_sha256": sha256_text(content),
                "raw_response_character_count": len(content),
            },
        )
        predictions = parse_model_output(content, fixed_test_classes)
        rows = evaluate_group(group_index, query_mapping, predictions)
        summary = compute_summary(rows, fixed_test_classes)
        summary["group_index"] = group_index
        summary["timing"] = timing
        rows_to_save = redact_rows(rows, label_aliases)
        summary_to_save = redact_summary(summary, label_aliases)
        save_json(group_dir / "predictions.json", rows_to_save)
        save_json(group_dir / "summary.json", summary_to_save)
        all_rows.extend(rows)
        group_summaries.append(summary_to_save)

    if args.dry_run:
        save_json(
            args.output_dir / "summary.json",
            {
                "status": "dry_run",
                "redacted": True,
                "group_count": len(groups),
                "support_size": len(support_records),
                "test_size": len(test_records),
            },
        )
        return

    overall_summary = compute_summary(all_rows, fixed_test_classes)
    overall_summary["status"] = "completed"
    overall_summary["group_count"] = len(groups)
    overall_summary["group_summaries"] = group_summaries
    rows_to_save = redact_rows(all_rows, label_aliases)
    summary_to_save = redact_summary(overall_summary, label_aliases)
    matrix_label_order = [label_aliases[label] for label in fixed_test_classes]
    save_json(args.output_dir / "all_predictions.json", rows_to_save)
    save_json(args.output_dir / "summary.json", summary_to_save)
    save_confusion_matrix(rows_to_save, matrix_label_order, args.output_dir / "confusion_matrix.csv")
    pd.DataFrame(summary_to_save["per_class_accuracy"]).to_csv(
        args.output_dir / "per_class_accuracy.csv", index=False, encoding="utf-8"
    )


if __name__ == "__main__":
    main()
