"""Prepare and audit the RAM high-resolution readout case split.

This is a read-only dataset audit.  It does not load labels into a model and
does not touch the RAM test split.  The existing identity-information probe
used SHA256(case) % 5 == 0 as its case-held-out set.  We preserve that
holdout and split only the old fit set into calibration (modulo 5 == 1) and
fit (modulo 5 in {2, 3, 4}).

The RAM archive contains case files but no validated patient/subject mapping.
Consequently the manifest explicitly reports case-disjoint, patient identity
unknown.  Filename prefixes are audited as candidates only and are never
treated as patient IDs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def case_hash(case: str) -> int:
    return int(hashlib.sha256(case.encode("utf-8")).hexdigest()[:8], 16)


def assign_probe_group(case: str) -> str:
    # Preserve the previous identity-information audit's modulo-5 holdout.
    bucket = case_hash(case) % 5
    if bucket == 0:
        return "probe_holdout"
    if bucket == 1:
        return "probe_calibration"
    return "probe_fit"


def file_digest_rows(root: Path, split: str, cases: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for case in cases:
        image = root / "images" / f"{case}.bmp"
        mask = root / "masks" / split / f"{case}.npy"
        if not image.is_file():
            raise FileNotFoundError(image)
        if not mask.is_file():
            raise FileNotFoundError(mask)
        rows.append(
            {
                "case": case,
                "image_sha256": sha256_file(image),
                "mask_sha256": sha256_file(mask),
            }
        )
    return rows


def duplicate_sha_values(rows_a: list[dict[str, str]], rows_b: list[dict[str, str]], key: str) -> list[str]:
    left = {row[key] for row in rows_a}
    right = {row[key] for row in rows_b}
    return sorted(left & right)


def candidate_groups(cases: list[str], extractor) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for case in cases:
        grouped[extractor(case)].append(case)
    return {key: sorted(value) for key, value in sorted(grouped.items())}


def group_span_audit(groups: dict[str, list[str]], labels: dict[str, str]) -> dict[str, object]:
    spans = {
        key: sorted({labels[case] for case in cases})
        for key, cases in groups.items()
    }
    cross = {key: value for key, value in spans.items() if len(value) > 1}
    return {
        "group_count": len(groups),
        "cross_probe_group_count": len(cross),
        "cross_probe_groups": cross,
        "interpretation": "candidate filename grouping only; not validated subject identity",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("G:/gutou/YOLO+SAM/data/remote_variants/RAM-W600/BoneSegmentation"),
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("G:/gutou/YOLO+SAM/outputs/ram_highres_readout_20260928"),
    )
    args = parser.parse_args()
    root = args.dataset_root.resolve()
    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=False)

    split_cases: dict[str, list[str]] = {}
    rows_by_split: dict[str, list[dict[str, str]]] = {}
    for split in ("train", "val"):
        mask_dir = root / "masks" / split
        cases = sorted(path.stem for path in mask_dir.glob("*.npy"))
        if len(set(cases)) != len(cases):
            raise RuntimeError(f"duplicate case names in {split}")
        split_cases[split] = cases
        rows_by_split[split] = file_digest_rows(root, split, cases)

    labels: dict[str, str] = {
        case: assign_probe_group(case) for case in split_cases["train"]
    }
    groups: dict[str, list[str]] = {
        group: sorted(case for case, label in labels.items() if label == group)
        for group in ("probe_fit", "probe_calibration", "probe_holdout")
    }
    if set().union(*groups.values()) != set(split_cases["train"]):
        raise RuntimeError("probe groups do not cover train cases exactly")
    if any(set(groups[a]) & set(groups[b]) for a in groups for b in groups if a < b):
        raise RuntimeError("probe groups overlap")

    train_rows = rows_by_split["train"]
    val_rows = rows_by_split["val"]
    train_val_image_dupes = duplicate_sha_values(train_rows, val_rows, "image_sha256")
    train_val_mask_dupes = duplicate_sha_values(train_rows, val_rows, "mask_sha256")

    prefix_groups = candidate_groups(split_cases["train"], lambda case: case.split("_", 1)[0])
    study_groups = candidate_groups(
        split_cases["train"], lambda case: "_".join(case.split("_")[:2])
    )
    prefix_audit = group_span_audit(prefix_groups, labels)
    study_audit = group_span_audit(study_groups, labels)

    source_artifact = Path(
        "G:/gutou/YOLO+SAM/outputs/whitebox_adam_20260922/SPLIT_INTEGRITY.json"
    )
    source_artifact_sha = sha256_file(source_artifact) if source_artifact.is_file() else None
    audit = {
        "status": "COMPLETE",
        "created_by": "scripts/prepare_ram_highres_probe_split.py",
        "scope": "RAM-W600 train425/val69 only; test not accessed",
        "dataset_root": str(root),
        "raw_dataset_unchanged": True,
        "case_counts": {split: len(cases) for split, cases in split_cases.items()},
        "case_order_sha256": {
            split: hashlib.sha256("\n".join(cases).encode("utf-8")).hexdigest()
            for split, cases in split_cases.items()
        },
        "train_val_content_audit": {
            "identical_image_sha256_across_train_val": train_val_image_dupes,
            "identical_mask_sha256_across_train_val": train_val_mask_dupes,
            "same_case_ids_across_train_val": sorted(
                set(split_cases["train"]) & set(split_cases["val"])
            ),
        },
        "patient_identity": {
            "status": "UNKNOWN",
            "validated_subject_mapping_found": False,
            "metadata_search_scope": "RAM-W600 local BoneSegmentation tree and project manifests",
            "filename_prefixes_used_as_patient_ids": False,
            "limitation": "The manifest guarantees case-disjoint groups only; patient-disjointness cannot be claimed.",
            "prior_integrity_artifact": str(source_artifact),
            "prior_integrity_artifact_sha256": source_artifact_sha,
            "prior_integrity_claim": "patient independence UNKNOWN; not inferred solely from filenames",
        },
        "candidate_filename_group_audit": {
            "first_token_before_underscore": prefix_audit,
            "first_two_tokens": study_audit,
            "warning": "Candidate groups span probe subsets and their semantics are unverified; they were not used to reassign cases.",
        },
        "probe_split_rule": {
            "hash": "int(SHA256(case UTF-8)[:8],16)",
            "probe_holdout": "hash % 5 == 0 (preserved from information_v2)",
            "probe_calibration": "hash % 5 == 1 (new inner split of old fit set)",
            "probe_fit": "hash % 5 in {2,3,4} (new inner split of old fit set)",
            "old_identity_information_fit": "hash % 5 != 0",
            "old_identity_information_holdout": "hash % 5 == 0",
            "seed": None,
            "random_sampling": False,
        },
        "cross_probe_disjointness": {
            "case_id_intersections": {
                f"{left}__{right}": sorted(set(groups[left]) & set(groups[right]))
                for left in groups
                for right in groups
                if left < right
            },
            "all_intersections_empty": True,
        },
        "file_hashes_computed": True,
        "test_accessed": False,
    }

    manifest = {
        "status": "COMPLETE",
        "manifest_type": "RAM high-resolution feature readout probe split",
        "dataset": "RAM-W600",
        "dataset_root": str(root),
        "split_semantics": {
            "probe_fit": "fit readout parameters/statistics only",
            "probe_calibration": "lock diagnostic low-FPR operating points only",
            "probe_holdout": "training-internal case holdout; no tuning after evaluation",
            "val": "locked development confirmation only; previously used for development",
        },
        "patient_disjoint": False,
        "case_disjoint": True,
        "patient_identity_status": "UNKNOWN",
        "groups": groups,
        "val_cases": split_cases["val"],
        "file_rows": {
            "train": train_rows,
            "val": val_rows,
        },
        "file_row_hashes": {
            "train_rows_sha256": hashlib.sha256(
                json.dumps(train_rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "val_rows_sha256": hashlib.sha256(
                json.dumps(val_rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
        },
        "leakage_controls": [
            "No GT-derived feature, H0 flag, boundary distance, or subject label is in the split rule.",
            "The old modulo-5 case holdout is preserved exactly.",
            "Calibration is drawn only from the old fit cases.",
            "No test files were read.",
            "Patient-level disjointness is not claimed because no validated subject mapping was found.",
        ],
        "split_audit_path": str(out_root / "SPLIT_AUDIT.json"),
    }
    (out_root / "SPLIT_AUDIT.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (out_root / "PROBE_SPLIT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "counts": {key: len(value) for key, value in groups.items()},
                "val": len(split_cases["val"]),
                "train_val_image_duplicates": len(train_val_image_dupes),
                "train_val_mask_duplicates": len(train_val_mask_dupes),
                "patient_disjoint": False,
                "out_root": str(out_root),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
