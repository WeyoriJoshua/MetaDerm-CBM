
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd


ID_RE = re.compile(r"^IMD\d+$", re.I)


def norm(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def marked(x):
    s = norm(x).lower()
    return s in {"x", "1", "1.0", "true", "yes", "y", "present", "p"}


def find_metadata_file(root: Path):
    candidates = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".xlsx", ".xls", ".csv", ".txt"}:
            continue
        low = p.name.lower()
        score = 0
        if "ph2_dataset" in low:
            score += 20
        if "metadata" in low:
            score += 10
        if p.suffix.lower() in {".xlsx", ".xls"}:
            score += 3
        candidates.append((score, p))

    if not candidates:
        raise FileNotFoundError(
            "No PH2 metadata spreadsheet/table found under the supplied root."
        )

    candidates.sort(key=lambda x: (-x[0], len(str(x[1]))))
    return candidates[0][1]


def _looks_useful(df):
    if df is None or df.empty:
        return False
    joined = " ".join(
        df.astype(str).fillna("").to_numpy().ravel().tolist()
    )
    return bool(re.search(r"\bIMD\d+\b", joined, flags=re.I))


def read_raw_table(path: Path):
    suffix = path.suffix.lower()

    if suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path, header=None, dtype=object)
        if _looks_useful(df):
            print("metadata parser: Excel")
            return df

    encodings = [
        "utf-8-sig",
        "utf-8",
        "utf-16",
        "utf-16-le",
        "utf-16-be",
        "cp1252",
        "latin-1",
    ]

    # Try normal delimited parsing first.
    separators = [
        ("semicolon", ";"),
        ("tab", "\t"),
        ("comma", ","),
        ("pipe", "|"),
    ]

    for encoding in encodings:
        for sep_name, sep in separators:
            try:
                df = pd.read_csv(
                    path,
                    sep=sep,
                    header=None,
                    dtype=object,
                    engine="python",
                    encoding=encoding,
                    on_bad_lines="skip",
                )
                if df.shape[1] >= 2 and _looks_useful(df):
                    print(
                        f"metadata parser: delimiter={sep_name}, "
                        f"encoding={encoding}, columns={df.shape[1]}"
                    )
                    return df
            except Exception:
                pass

    # Automatic delimiter detection.
    for encoding in encodings:
        try:
            df = pd.read_csv(
                path,
                sep=None,
                header=None,
                dtype=object,
                engine="python",
                encoding=encoding,
                on_bad_lines="skip",
            )
            if df.shape[1] >= 2 and _looks_useful(df):
                print(
                    f"metadata parser: automatic delimiter, "
                    f"encoding={encoding}, columns={df.shape[1]}"
                )
                return df
        except Exception:
            pass

    # PH2 mirrors sometimes package the original table as a whitespace
    # or fixed-width text document.
    for encoding in encodings:
        try:
            df = pd.read_csv(
                path,
                sep=r"\s{2,}",
                header=None,
                dtype=object,
                engine="python",
                encoding=encoding,
                on_bad_lines="skip",
            )
            if df.shape[1] >= 2 and _looks_useful(df):
                print(
                    f"metadata parser: multi-space, "
                    f"encoding={encoding}, columns={df.shape[1]}"
                )
                return df
        except Exception:
            pass

        try:
            df = pd.read_fwf(
                path,
                header=None,
                dtype=object,
                encoding=encoding,
            )
            if df.shape[1] >= 2 and _looks_useful(df):
                print(
                    f"metadata parser: fixed-width, "
                    f"encoding={encoding}, columns={df.shape[1]}"
                )
                return df
        except Exception:
            pass

    # Last-resort diagnostic: show the first readable lines so the
    # user can send them back without manually opening the file.
    preview = []
    used_encoding = None
    for encoding in encodings:
        try:
            text = path.read_text(encoding=encoding)
            preview = text.splitlines()[:15]
            used_encoding = encoding
            break
        except Exception:
            continue

    print("\nCould not parse PH2_dataset.txt automatically.")
    if used_encoding:
        print(f"Readable encoding detected: {used_encoding}")
        print("First 15 lines:")
        for line in preview:
            print(repr(line))

    raise RuntimeError(f"Could not parse metadata table: {path}")


def find_id_rows(raw: pd.DataFrame):
    rows = []
    for i, row in raw.iterrows():
        for j, value in enumerate(row.tolist()):
            s = norm(value)
            # Handle cells containing extra text around IMD IDs.
            m = re.search(r"\b(IMD\d+)\b", s, flags=re.I)
            if m:
                rows.append((i, j, m.group(1).upper()))
                break

    if not rows:
        raise RuntimeError(
            "No PH2 image identifiers (IMDxxx) were found in the metadata table."
        )
    return rows


def header_contexts(raw: pd.DataFrame, first_data_row: int):
    header = raw.iloc[:first_data_row].copy()
    header_ffill = header.ffill(axis=1)

    contexts = {}
    for j in range(raw.shape[1]):
        parts = []
        for frame in (header, header_ffill):
            for value in frame.iloc[:, j].tolist():
                s = norm(value)
                if s and s not in parts:
                    parts.append(s)
        contexts[j] = " | ".join(parts)
    return contexts


def cols_matching(contexts, required):
    out = []
    for j, context in contexts.items():
        low = context.lower()
        if all(term.lower() in low for term in required):
            out.append(j)
    return out


def decode_diagnosis(row, contexts):
    checks = [
        ("common nevus", 0, ("common", "nevus")),
        ("atypical nevus", 0, ("atypical", "nevus")),
        ("melanoma", 1, ("melanoma",)),
    ]

    active = []
    for name, binary, terms in checks:
        cols = cols_matching(contexts, terms)
        if any(marked(row.iloc[j]) for j in cols):
            active.append((name, binary))

    if len(active) == 1:
        return active[0]

    # Fallback for textual diagnosis cells or rows.
    row_text = " | ".join(norm(v) for v in row.tolist()).lower()
    if "melanoma" in row_text:
        return "melanoma", 1
    if "atypical" in row_text and "nev" in row_text:
        return "atypical nevus", 0
    if "common" in row_text and "nev" in row_text:
        return "common nevus", 0

    for j, context in contexts.items():
        if "diagnos" not in context.lower():
            continue
        s = norm(row.iloc[j]).lower()
        if "melanoma" in s:
            return "melanoma", 1
        if "atypical" in s and "nev" in s:
            return "atypical nevus", 0
        if "common" in s and "nev" in s:
            return "common nevus", 0

    return "", np.nan


def decode_marked_state(row, contexts, parent_terms, states):
    hits = []
    for canonical, state_terms in states:
        for j, context in contexts.items():
            low = context.lower()
            if not all(t.lower() in low for t in parent_terms):
                continue
            if not all(t.lower() in low for t in state_terms):
                continue
            if marked(row.iloc[j]):
                hits.append(canonical)
                break
    return hits[0] if len(hits) == 1 else ""


def decode_single_token(row, contexts, parent_terms, token_map):
    for j, context in contexts.items():
        low = context.lower()
        if not all(t.lower() in low for t in parent_terms):
            continue
        token = norm(row.iloc[j]).lower()
        if token in token_map:
            return token_map[token]
    return ""


def decode_concept(row, contexts, concept):
    if concept == "pigment_network":
        v = decode_marked_state(
            row, contexts, ("pigment", "network"),
            [("typical", ("typical",)),
             ("atypical", ("atypical",))],
        )
        return v or decode_single_token(
            row, contexts, ("pigment", "network"),
            {"t": "typical", "typical": "typical",
             "at": "atypical", "atypical": "atypical"},
        )

    if concept == "dots_and_globules":
        v = decode_marked_state(
            row, contexts, ("dots",),
            [("absent", ("absent",)),
             ("typical", ("typical",)),
             ("atypical", ("atypical",))],
        )
        return v or decode_single_token(
            row, contexts, ("dots",),
            {"a": "absent", "absent": "absent",
             "t": "typical", "typical": "typical",
             "at": "atypical", "atypical": "atypical"},
        )

    if concept == "streaks":
        v = decode_marked_state(
            row, contexts, ("streak",),
            [("absent", ("absent",)),
             ("present", ("present",))],
        )
        return v or decode_single_token(
            row, contexts, ("streak",),
            {"a": "absent", "absent": "absent",
             "p": "present", "present": "present"},
        )

    if concept == "regression_structures":
        v = decode_marked_state(
            row, contexts, ("regression",),
            [("absent", ("absent",)),
             ("present", ("present",))],
        )
        return v or decode_single_token(
            row, contexts, ("regression",),
            {"a": "absent", "absent": "absent",
             "p": "present", "present": "present"},
        )

    if concept == "blue_whitish_veil":
        v = decode_marked_state(
            row, contexts, ("veil",),
            [("absent", ("absent",)),
             ("present", ("present",))],
        )
        return v or decode_single_token(
            row, contexts, ("veil",),
            {"a": "absent", "absent": "absent",
             "p": "present", "present": "present"},
        )

    raise KeyError(concept)


def discover_images(root: Path):
    chosen = {}
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".bmp", ".jpg", ".jpeg", ".png"}:
            continue

        # PH2 image paths sometimes contain the IMD id in a parent folder,
        # while the actual filename may carry a suffix.
        matches = re.findall(r"\b(IMD\d+)\b", str(p), flags=re.I)
        if not matches:
            continue
        image_id = matches[-1].upper()

        low = str(p).lower()
        score = 0
        if "dermoscopic" in low:
            score += 10
        if "original" in low:
            score += 3
        if any(x in low for x in ["lesion", "mask", "segmentation"]):
            score -= 10

        if image_id not in chosen or score > chosen[image_id][0]:
            chosen[image_id] = (score, p.resolve())

    return {k: v[1] for k, v in chosen.items()}


def main():
    parser = argparse.ArgumentParser(
        description="Audit PH2 and create canonical MetaDerm external metadata."
    )
    parser.add_argument("--ph2-root", required=True)
    parser.add_argument(
        "--output",
        default="data_external/ph2/ph2_canonical.csv",
    )
    args = parser.parse_args()

    root = Path(args.ph2_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(root)

    metadata = find_metadata_file(root)
    raw = read_raw_table(metadata)
    id_rows = find_id_rows(raw)
    first_data_row = min(i for i, _, _ in id_rows)
    contexts = header_contexts(raw, first_data_row)
    images = discover_images(root)

    print("metadata:", metadata)
    print("metadata shape:", raw.shape)
    print("rows containing IMD IDs:", len(id_rows))
    print("dermoscopic image candidates:", len(images))

    print("\nRelevant metadata header contexts:")
    for j, context in contexts.items():
        low = context.lower()
        if any(
            t in low
            for t in [
                "diagnos", "nevus", "melanoma", "pigment",
                "dot", "glob", "streak", "regression", "veil"
            ]
        ):
            print(f"  col {j}: {context}")

    records = []
    for i, _id_col, image_id in id_rows:
        row = raw.iloc[i]
        diagnosis, label = decode_diagnosis(row, contexts)

        rec = {
            "image_id": image_id,
            "image_path": str(images.get(image_id, "")),
            "clinical_diagnosis": diagnosis,
            "label": label,
        }

        for concept in [
            "pigment_network",
            "dots_and_globules",
            "streaks",
            "regression_structures",
            "blue_whitish_veil",
        ]:
            rec[concept] = decode_concept(row, contexts, concept)

        records.append(rec)

    df = pd.DataFrame(records).drop_duplicates("image_id")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print("\nCanonical PH2 audit")
    print("rows:", len(df))
    print("images found:", int(df["image_path"].astype(bool).sum()))
    print("diagnoses parsed:", int(df["label"].notna().sum()))

    if df["label"].notna().any():
        print("\nDiagnosis distribution:")
        print(
            df.dropna(subset=["label"])
            .groupby(["clinical_diagnosis", "label"])
            .size()
            .to_string()
        )

    print("\nCompatible concept annotations parsed:")
    for c in [
        "pigment_network",
        "dots_and_globules",
        "streaks",
        "regression_structures",
        "blue_whitish_veil",
    ]:
        valid = df[c].fillna("").astype(str).str.len() > 0
        print(
            f"  {c}: {int(valid.sum())} "
            f"{df.loc[valid, c].value_counts().to_dict()}"
        )

    missing_images = df[
        df["image_path"].fillna("").astype(str).str.len() == 0
    ]["image_id"].tolist()
    if missing_images:
        print("\nWARNING: missing images:", missing_images[:20])

    if len(df) != 200:
        print(
            f"\nWARNING: expected 200 PH2 lesions; parsed {len(df)}."
        )

    if int(df["label"].notna().sum()) != 200:
        print(
            "\nWARNING: diagnosis parsing is incomplete. "
            "Do NOT run external evaluation yet."
        )

    print("\nsaved:", out)


if __name__ == "__main__":
    main()
