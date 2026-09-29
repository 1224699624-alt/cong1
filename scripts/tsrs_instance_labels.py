#!/usr/bin/env python3
"""Decode and audit anonymous TSRS instance labels.

The TSRS labels are per-image anonymous instance annotations.  An index or
colour is therefore only meaningful inside one image; this module deliberately
does not assign a fixed anatomical identity to an index or colour.

Decoding policy
--------------
* ``P``, ``L``, ``I`` and the other supported single-channel Pillow modes use
  the raw integer value as the label, with integer ``0`` as background.
* ``RGB`` and ``RGBA`` use the complete RGB tuple as the label, with
  ``(0, 0, 0)`` as background.  RGBA alpha is ignored for identity and is
  reported by the audit as an ambiguity QC signal instead of being guessed.
* Every non-background value is split with 8-connected components.  A
  disconnected repetition of one value is therefore recovered as separate
  anonymous instances.  A connected region is kept as one instance because
  its semantic multiplicity cannot be inferred from pixels alone.

The binary foreground union is always ``decoded > 0``.  The decoder checks
that this union is unchanged before returning it.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image

try:  # OpenCV is already a repository dependency and is the fast path.
    import cv2  # type: ignore
except Exception:  # pragma: no cover - exercised only in minimal environments.
    cv2 = None


_SINGLE_CHANNEL_MODES = {
    "1",
    "L",
    "P",
    "I",
    "I;16",
    "I;16B",
    "I;16L",
    "I;16N",
}
_RGB_MODES = {"RGB", "RGBA"}
_IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
_CONNECTIVITY = 8


def _load_label(path: str | Path) -> tuple[Path, str, np.ndarray, dict[str, Any]]:
    """Load a label image while keeping the Pillow mode and metadata."""

    label_path = Path(path)
    if not label_path.is_file():
        raise FileNotFoundError(f"TSRS label does not exist: {label_path}")
    with Image.open(label_path) as image:
        mode = image.mode
        values = np.asarray(image).copy()
        info = dict(image.info)
    return label_path, mode, values, info


def _connected_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Return 8-connected component IDs and their count for a boolean mask."""

    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError(f"connected-component input must be 2-D, got {binary.shape}")
    if not np.any(binary):
        return np.zeros(binary.shape, dtype=np.int32), 0

    if cv2 is not None:
        count, labels = cv2.connectedComponents(
            binary.astype(np.uint8, copy=False), connectivity=_CONNECTIVITY
        )
        return np.asarray(labels, dtype=np.int32), int(count - 1)

    try:
        from scipy import ndimage  # type: ignore

        labels, count = ndimage.label(
            binary, structure=np.ones((3, 3), dtype=np.uint8)
        )
        return np.asarray(labels, dtype=np.int32), int(count)
    except Exception:  # pragma: no cover - fallback for a bare NumPy/Pillow env.
        labels = np.zeros(binary.shape, dtype=np.int32)
        height, width = binary.shape
        next_id = 0
        neighbours = (
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        )
        for y, x in zip(*np.nonzero(binary)):
            if labels[y, x] != 0:
                continue
            next_id += 1
            labels[y, x] = next_id
            stack = [(int(y), int(x))]
            while stack:
                cy, cx = stack.pop()
                for dy, dx in neighbours:
                    ny, nx = cy + dy, cx + dx
                    if (
                        0 <= ny < height
                        and 0 <= nx < width
                        and binary[ny, nx]
                        and labels[ny, nx] == 0
                    ):
                        labels[ny, nx] = next_id
                        stack.append((ny, nx))
        return labels, next_id


def _rgb_codes(rgb: np.ndarray) -> np.ndarray:
    """Pack uint8 RGB tuples into sortable integer codes."""

    channels = np.asarray(rgb)
    if channels.ndim != 3 or channels.shape[-1] != 3:
        raise ValueError(f"RGB array must have shape HxWx3, got {channels.shape}")
    # Pillow's RGB/RGBA modes are uint8.  Cast explicitly so bit operations do
    # not overflow on a platform-dependent integer dtype.
    channels = channels.astype(np.uint32, copy=False)
    return (
        (channels[..., 0] << np.uint32(16))
        | (channels[..., 1] << np.uint32(8))
        | channels[..., 2]
    )


def _code_to_rgb(code: int) -> tuple[int, int, int]:
    return ((int(code) >> 16) & 255, (int(code) >> 8) & 255, int(code) & 255)


def _prepare_values(
    mode: str, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray, list[int], int, str, list[int] | list[int]]:
    """Normalize a Pillow array to sortable category codes.

    Returns ``(categories, foreground, unique_codes, background_code,
    background_definition, shape)``.  ``categories`` is a 2-D integer array;
    for RGB/RGBA it contains packed full-RGB tuple codes.
    """

    if mode in _SINGLE_CHANNEL_MODES:
        if values.ndim != 2:
            raise ValueError(f"mode {mode} must decode to a 2-D array, got {values.shape}")
        categories = values.astype(np.int64, copy=False)
        background_code = 0
        background_definition = "single-channel index/value 0"
    elif mode in _RGB_MODES:
        if values.ndim != 3 or values.shape[-1] < 3:
            raise ValueError(f"mode {mode} must decode to HxWx3/4, got {values.shape}")
        categories = _rgb_codes(values[..., :3]).astype(np.int64, copy=False)
        background_code = 0
        background_definition = "full RGB tuple (0, 0, 0); RGBA alpha is ignored for identity"
    else:
        raise ValueError(
            f"unsupported TSRS label mode {mode!r}; supported single-channel modes "
            f"are {sorted(_SINGLE_CHANNEL_MODES)} plus RGB/RGBA"
        )

    unique_codes = np.unique(categories).astype(np.int64, copy=False).tolist()
    foreground = categories != background_code
    shape = [int(dim) for dim in categories.shape]
    return categories, foreground, [int(code) for code in unique_codes], int(background_code), background_definition, shape


def _decode_categories(
    categories: np.ndarray,
    unique_codes: Sequence[int],
    background_code: int,
    foreground: np.ndarray,
) -> tuple[np.ndarray, dict[int, int]]:
    """Split every non-background category into deterministic instance IDs."""

    decoded = np.zeros(categories.shape, dtype=np.int32)
    component_counts: dict[int, int] = {}
    next_instance = 1
    for code in unique_codes:
        if int(code) == background_code:
            continue
        mask = categories == int(code)
        labels, count = _connected_components(mask)
        component_counts[int(code)] = int(count)
        for component_id in range(1, int(count) + 1):
            decoded[labels == component_id] = next_instance
            next_instance += 1

    if not np.array_equal(decoded > 0, foreground):
        raise RuntimeError("anonymous instance remapping changed the binary foreground union")
    return decoded, component_counts


def decode_anonymous_instances(path: str | Path) -> np.ndarray:
    """Decode one TSRS label into per-image anonymous ``int32`` instance IDs.

    The returned IDs start at 1 for each image and have no cross-image meaning.
    Background is zero.  Every non-background raw value is split using
    8-connected components, and the binary foreground union is preserved.
    """

    _, mode, values, _ = _load_label(path)
    categories, foreground, unique_codes, background_code, _, _ = _prepare_values(mode, values)
    decoded, _ = _decode_categories(categories, unique_codes, background_code, foreground)
    return decoded


def _serial_value(mode: str, code: int) -> int | list[int]:
    if mode in _RGB_MODES:
        return list(_code_to_rgb(code))
    return int(code)


def _value_key(mode: str, code: int) -> str:
    if mode in _RGB_MODES:
        return ",".join(str(item) for item in _code_to_rgb(code))
    return str(int(code))


def _alpha_qc(
    mode: str,
    values: np.ndarray,
    categories: np.ndarray,
    unique_codes: Sequence[int],
) -> dict[str, Any]:
    """Describe alpha ambiguity without making an alpha-based repair."""

    if mode != "RGBA":
        return {
            "alpha_present": False,
            "alpha_unique_values": [],
            "alpha_zero_non_background_pixel_count": 0,
            "alpha_nonzero_background_pixel_count": 0,
            "alpha_tuple_conflict_count": 0,
            "alpha_ambiguity": False,
            "alpha_policy": "no alpha channel; background follows the mode policy",
        }

    alpha = values[..., 3]
    alpha_unique = [int(value) for value in np.unique(alpha).tolist()]
    zero_non_bg = int(np.count_nonzero((alpha == 0) & (categories != 0)))
    nonzero_bg = int(np.count_nonzero((alpha != 0) & (categories == 0)))
    tuple_conflicts = 0
    for code in unique_codes:
        alpha_values = np.unique(alpha[categories == int(code)])
        if alpha_values.size > 1:
            tuple_conflicts += 1
    ambiguous = bool(zero_non_bg or nonzero_bg or tuple_conflicts)
    return {
        "alpha_present": True,
        "alpha_unique_values": alpha_unique,
        "alpha_zero_non_background_pixel_count": zero_non_bg,
        "alpha_nonzero_background_pixel_count": nonzero_bg,
        "alpha_tuple_conflict_count": int(tuple_conflicts),
        "alpha_ambiguity": ambiguous,
        "alpha_policy": (
            "alpha is ignored for identity; RGB (0, 0, 0) is background; "
            "review flagged alpha/RGB conflicts before treating transparency as labels"
        ),
    }


def audit_instance_label(path: str | Path) -> dict[str, Any]:
    """Audit one label and return JSON/CSV-friendly QC statistics.

    The report intentionally distinguishes recoverable repeated values
    (disconnected components) from connected regions whose semantic instance
    count is unknowable.  No connected same-colour region is split further.
    """

    label_path, mode, values, image_info = _load_label(path)
    categories, foreground, unique_codes, background_code, background_definition, shape = _prepare_values(mode, values)
    decoded, component_counts = _decode_categories(
        categories, unique_codes, background_code, foreground
    )
    non_background_codes = [
        int(code) for code in unique_codes if int(code) != background_code
    ]
    multi_component_codes = [
        code for code in non_background_codes if component_counts.get(code, 0) > 1
    ]
    single_component_codes = [
        code for code in non_background_codes if component_counts.get(code, 0) == 1
    ]
    recoverable_extra_instances = sum(
        max(component_counts.get(code, 0) - 1, 0) for code in non_background_codes
    )

    first_channel_groups: dict[int, list[int | list[int]]] = {}
    foreground_first_channel_groups: dict[int, list[int | list[int]]] = {}
    if mode in _RGB_MODES:
        for include_background, target in (
            (True, first_channel_groups),
            (False, foreground_first_channel_groups),
        ):
            codes_for_groups = unique_codes if include_background else non_background_codes
            grouped: defaultdict[int, list[int | list[int]]] = defaultdict(list)
            for code in codes_for_groups:
                color = _serial_value(mode, int(code))
                assert isinstance(color, list)
                grouped[int(color[0])].append(color)
            target.update(
                {
                    int(red): sorted(colors, key=lambda color: tuple(color))
                    for red, colors in grouped.items()
                    if len(colors) > 1
                }
            )

    alpha = _alpha_qc(mode, values, categories, unique_codes)
    raw_values = [_serial_value(mode, code) for code in unique_codes]
    component_count_by_value = {
        _value_key(mode, code): int(component_counts.get(code, 0))
        for code in non_background_codes
    }
    palette_transparency = image_info.get("transparency") if mode == "P" else None
    report: dict[str, Any] = {
        "path": str(label_path),
        "filename": label_path.name,
        "mode": mode,
        "shape": shape,
        "height": int(shape[0]),
        "width": int(shape[1]),
        "background_definition": background_definition,
        "background_value": (
            list(_code_to_rgb(background_code)) if mode in _RGB_MODES else int(background_code)
        ),
        "raw_unique_values": raw_values,
        "raw_unique_value_count": int(len(unique_codes)),
        "raw_unique_non_background_value_count": int(len(non_background_codes)),
        # Explicit aliases make the CSV and downstream audits easy to read.
        "unique_color_index_count": int(len(unique_codes)),
        "unique_non_background_color_index_count": int(len(non_background_codes)),
        "foreground_pixel_count": int(np.count_nonzero(foreground)),
        "background_pixel_count": int(np.count_nonzero(~foreground)),
        "anonymous_instance_count": int(decoded.max(initial=0)),
        "instance_count": int(decoded.max(initial=0)),
        "component_count_by_value": component_count_by_value,
        "multi_component_value_count": int(len(multi_component_codes)),
        "multi_component_value_keys": [
            _value_key(mode, code) for code in multi_component_codes
        ],
        "multi_component_image": bool(multi_component_codes),
        "recoverable_duplicate_color_count": int(len(multi_component_codes)),
        "recoverable_duplicate_color_instance_count": int(recoverable_extra_instances),
        "recoverable_duplicate_instance_count": int(recoverable_extra_instances),
        # A connected same-value region is deliberately left unsplit.  This is
        # a conservative QC candidate count, not a claim that every such
        # region contains multiple semantic instances.
        # A single connected value is the normal representation of one
        # instance. It is a QC candidate for hidden semantic multiplicity,
        # but the pixels do not provide an evidence-based count of hidden
        # instances. Keep that candidate count separate from actual decoder
        # ambiguity, which remains zero unless a human review supplies it.
        "same_color_connected_qc_candidate_count": int(len(single_component_codes)),
        "same_color_connected_ambiguity_count": 0,
        "same_color_connected_unrecoverable_count": 0,
        "same_color_connected_ambiguity": False,
        "same_color_connected_policy": (
            "single connected regions are retained; semantic multiplicity cannot "
            "be inferred automatically"
        ),
        "first_channel_collision_count": int(len(first_channel_groups)),
        "first_channel_collision_color_count": int(
            sum(len(colors) for colors in first_channel_groups.values())
        ),
        "first_channel_collision": bool(first_channel_groups),
        "first_channel_collision_groups": first_channel_groups,
        "first_channel_collision_foreground_count": int(
            len(foreground_first_channel_groups)
        ),
        "first_channel_collision_foreground": bool(foreground_first_channel_groups),
        "first_channel_collision_foreground_groups": foreground_first_channel_groups,
        "binary_union_preserved": bool(
            np.array_equal(decoded > 0, foreground)
        ),
        "decoded_dtype": str(decoded.dtype),
        "palette_transparency_metadata": palette_transparency is not None,
        **alpha,
    }
    return report


def _resolve_split_dir(dataset_root: Path, split: str) -> Path:
    candidates = (dataset_root / f"{split}_labels", dataset_root / split)
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    rendered = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"could not find {split!r} label directory; checked {rendered}")


def _iter_label_paths(split_dir: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in split_dir.iterdir()
            if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
        ),
        key=lambda path: path.name,
    )


def _histogram(values: Iterable[Any]) -> dict[str, int]:
    counts = Counter(str(value) for value in values)
    return {key: int(counts[key]) for key in sorted(counts)}


def audit_dataset(
    dataset_root: str | Path,
    splits: Sequence[str] = ("train", "val"),
    limit_per_split: int | None = None,
) -> dict[str, Any]:
    """Audit train/val labels under a dataset root.

    ``test`` is rejected explicitly so an accidental test read cannot be
    hidden by a broad directory glob.
    """

    root = Path(dataset_root)
    normalized_splits = tuple(str(split).lower() for split in splits)
    if not normalized_splits:
        raise ValueError("at least one split is required")
    forbidden = [split for split in normalized_splits if split == "test"]
    if forbidden or any(split not in {"train", "val"} for split in normalized_splits):
        raise ValueError(
            f"TSRS instance audit only permits train/val; received {normalized_splits}"
        )
    if limit_per_split is not None and limit_per_split < 0:
        raise ValueError("limit_per_split must be non-negative or omitted")

    records: list[dict[str, Any]] = []
    split_counts: dict[str, dict[str, Any]] = {}
    for split in normalized_splits:
        split_dir = _resolve_split_dir(root, split)
        paths = _iter_label_paths(split_dir)
        if limit_per_split is not None:
            paths = paths[:limit_per_split]
        if not paths:
            raise FileNotFoundError(f"no label images found under {split_dir}")
        split_records: list[dict[str, Any]] = []
        for path in paths:
            record = audit_instance_label(path)
            record["split"] = split
            try:
                record["relative_path"] = str(path.relative_to(root))
            except ValueError:
                record["relative_path"] = str(path)
            split_records.append(record)
            records.append(record)
        split_counts[split] = _summarize_records(split_records)

    summary = _summarize_records(records)
    summary.update(
        {
            "schema_version": 1,
            "dataset_root": str(root),
            "splits": list(normalized_splits),
            "test_read": False,
            "split_counts": split_counts,
            "records": records,
            "policy": {
                "single_channel_background": 0,
                "rgb_background": [0, 0, 0],
                "rgb_identity": "complete RGB tuple",
                "rgba_identity": "complete RGB tuple; alpha ignored for identity and audited",
                "connected_components": "8-connected, applied independently per non-background value",
                "instance_ids": "renumbered from 1 independently within each image",
                "binary_union": "decoded instance_id > 0 equals raw non-background foreground",
                "test_split": "forbidden during this audit",
            },
        }
    )
    return summary


def _summarize_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    modes = Counter(str(record["mode"]) for record in records)
    return {
        "image_count": int(len(records)),
        "mode_distribution": {key: int(modes[key]) for key in sorted(modes)},
        "unique_color_index_count_distribution": _histogram(
            record["unique_color_index_count"] for record in records
        ),
        "unique_non_background_count_distribution": _histogram(
            record["unique_non_background_color_index_count"] for record in records
        ),
        "anonymous_instance_count_distribution": _histogram(
            record["anonymous_instance_count"] for record in records
        ),
        "multi_component_value_image_count": int(
            sum(bool(record["multi_component_image"]) for record in records)
        ),
        "multi_component_value_count": int(
            sum(int(record["multi_component_value_count"]) for record in records)
        ),
        "recoverable_duplicate_color_instance_count": int(
            sum(
                int(record["recoverable_duplicate_color_instance_count"])
                for record in records
            )
        ),
        "same_color_connected_ambiguity_image_count": int(
            sum(bool(record["same_color_connected_ambiguity"]) for record in records)
        ),
        "same_color_connected_qc_candidate_count": int(
            sum(int(record["same_color_connected_qc_candidate_count"]) for record in records)
        ),
        "same_color_connected_ambiguity_count": int(
            sum(int(record["same_color_connected_ambiguity_count"]) for record in records)
        ),
        "first_channel_collision_image_count": int(
            sum(bool(record["first_channel_collision"]) for record in records)
        ),
        "first_channel_collision_count": int(
            sum(int(record["first_channel_collision_count"]) for record in records)
        ),
        "first_channel_collision_color_count": int(
            sum(int(record["first_channel_collision_color_count"]) for record in records)
        ),
        "alpha_ambiguity_image_count": int(
            sum(bool(record["alpha_ambiguity"]) for record in records)
        ),
        "binary_union_failure_count": int(
            sum(not bool(record["binary_union_preserved"]) for record in records)
        ),
    }


_CSV_FIELDS = [
    "split",
    "path",
    "relative_path",
    "filename",
    "mode",
    "height",
    "width",
    "raw_unique_value_count",
    "raw_unique_non_background_value_count",
    "unique_color_index_count",
    "unique_non_background_color_index_count",
    "anonymous_instance_count",
    "foreground_pixel_count",
    "background_pixel_count",
    "multi_component_value_count",
    "multi_component_image",
    "recoverable_duplicate_color_instance_count",
    "same_color_connected_ambiguity_count",
    "same_color_connected_ambiguity",
    "first_channel_collision_count",
    "first_channel_collision_color_count",
    "first_channel_collision",
    "first_channel_collision_foreground_count",
    "alpha_present",
    "alpha_ambiguity",
    "alpha_unique_values",
    "alpha_zero_non_background_pixel_count",
    "alpha_nonzero_background_pixel_count",
    "alpha_tuple_conflict_count",
    "binary_union_preserved",
    "background_definition",
    "raw_unique_values",
    "component_count_by_value",
]


def write_audit_outputs(
    summary: Mapping[str, Any], output_dir: str | Path
) -> tuple[Path, Path]:
    """Write one JSON summary and one per-image CSV table."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / "tsrs_instance_label_audit.json"
    csv_path = destination / "tsrs_instance_label_audit.csv"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for record in summary.get("records", []):
            row = {
                field: json.dumps(record[field], ensure_ascii=False, sort_keys=True)
                if isinstance(record.get(field), (dict, list, tuple))
                else record.get(field, "")
                for field in _CSV_FIELDS
            }
            writer.writerow(row)
    return json_path, csv_path


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit TSRS train/val label modes and anonymous instance decoding."
    )
    parser.add_argument(
        "--dataset-root",
        "--root",
        type=Path,
        default=Path("data/raw/TSRS_RSNA-Epiphysis"),
        help="TSRS root containing train_labels/ and val_labels/.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "val"),
        default=("train", "val"),
        help="Splits to audit; test is intentionally not an accepted choice.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/final_prior_v1/audit"),
        help="Directory for tsrs_instance_label_audit.json/csv.",
    )
    parser.add_argument(
        "--limit-per-split",
        type=int,
        default=None,
        help="Optional smoke-test limit; omit for the complete split audit.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    summary = audit_dataset(
        args.dataset_root,
        splits=args.splits,
        limit_per_split=args.limit_per_split,
    )
    json_path, csv_path = write_audit_outputs(summary, args.output_dir)
    print(
        json.dumps(
            {
                "images": summary["image_count"],
                "modes": summary["mode_distribution"],
                "first_channel_collision_images": summary[
                    "first_channel_collision_image_count"
                ],
                "multi_component_images": summary[
                    "multi_component_value_image_count"
                ],
                "json": str(json_path),
                "csv": str(csv_path),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
