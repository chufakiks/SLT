#!/usr/bin/env python3
"""
Link cropped OpenASL videos to TSV metadata using (yid, start, end).
"""

import csv
from pathlib import Path

def parse_filename(filename):
    """
    Expected format:
    <yid>-HH:MM:SS.mmm-HH:MM:SS.mmm.mp4
    """
    if not filename.endswith(".mp4"):
        return None

    name = filename[:-4]
    parts = name.split("-")

    if len(parts) != 3:
        return None

    return {
        "yid": parts[0],
        "start": parts[1],
        "end": parts[2],
    }

def link_cropped_videos_to_tsv(cropped_video_dir, tsv_path, output_csv=None):
    print(f"Reading TSV file from {tsv_path}...")

    # Load TSV and build lookup
    tsv_rows = []
    tsv_lookup = {}

    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for idx, row in enumerate(reader):
            tsv_rows.append(row)
            key = (row["yid"], row["start"], row["end"])
            tsv_lookup[key] = (idx, row)

    print(f"Loaded {len(tsv_rows)} TSV entries")

    video_files = list(Path(cropped_video_dir).glob("*.mp4"))
    print(f"Found {len(video_files)} cropped videos")

    mappings = []

    for video_path in video_files:
        info = parse_filename(video_path.name)
        if info is None:
            print(f"Warning: Unrecognized filename format: {video_path.name}")
            continue

        key = (info["yid"], info["start"], info["end"])

        if key not in tsv_lookup:
            print(f"Warning: No TSV match for {video_path.name}")
            continue

        row_idx, tsv_row = tsv_lookup[key]

        mappings.append({
            "cropped_video_path": str(video_path),
            "cropped_video_filename": video_path.name,
            "tsv_row_index": row_idx,
            "vid": tsv_row["vid"],
            "yid": tsv_row["yid"],
            "start": tsv_row["start"],
            "end": tsv_row["end"],
            "raw_text": tsv_row["raw-text"],
            "tokenized_text": tsv_row["tokenized-text"],
            "gloss": tsv_row.get("gloss", ""),
            "split": tsv_row["split"],
        })

    print(f"\nSuccessfully mapped {len(mappings)} videos to TSV entries")

    if output_csv and mappings:
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            fieldnames = mappings[0].keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(mappings)

        print(f"Saved mapping to {output_csv}")

    return mappings

def main():
    cropped_video_dir = "/work3/s235253/openaslcropeed"
    tsv_path = "/home/user/SLT/openasl-v1.0.tsv"
    output_csv = "/work3/s235253cropped_videos_mapping.csv"

    mappings = link_cropped_videos_to_tsv(
        cropped_video_dir,
        tsv_path,
        output_csv,
    )

    print("\nSample mappings:")
    for row in mappings[:5]:
        print(row["cropped_video_filename"], "→ TSV row", row["tsv_row_index"])

if __name__ == "__main__":
    main()