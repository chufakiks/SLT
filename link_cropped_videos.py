#!/usr/bin/env python3
"""
Script to link cropped videos from OpenASL's crop_video to the TSV metadata file.
This creates a mapping between cropped video files and their corresponding TSV entries.
"""

import os
import re
import csv
from pathlib import Path

def extract_row_index_from_filename(filename):
    """
    Extract the TSV row index from the cropped video filename.
    Expected format: {hash}-{row_index}_{time1}_{time2}.mp4
    Example: 00001167060947993015-3717_3_38.mp4 -> 3717
    """
    # Remove .mp4 extension
    name = filename.replace('.mp4', '')

    # Pattern: {anything}-{digits}_{digits}_{digits}
    match = re.search(r'-(\d+)_\d+_\d+$', name)
    if match:
        return int(match.group(1))

    return None

def link_cropped_videos_to_tsv(cropped_video_dir, tsv_path, output_csv=None):
    """
    Create a mapping between cropped videos and TSV entries.

    Args:
        cropped_video_dir: Directory containing cropped .mp4 files
        tsv_path: Path to the openasl-v1.0.tsv file
        output_csv: Optional path to save the mapping as CSV

    Returns:
        List of dictionaries with video files linked to TSV metadata
    """
    # Read the TSV file
    print(f"Reading TSV file from {tsv_path}...")
    tsv_rows = []
    with open(tsv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            tsv_rows.append(row)
    print(f"Loaded {len(tsv_rows)} entries from TSV")

    # Find all cropped video files
    video_files = list(Path(cropped_video_dir).glob('*.mp4'))
    print(f"Found {len(video_files)} video files in {cropped_video_dir}")

    # Create mapping
    mappings = []
    for video_path in video_files:
        filename = video_path.name
        row_idx = extract_row_index_from_filename(filename)

        if row_idx is not None:
            # Row index is 0-based
            if row_idx < len(tsv_rows):
                tsv_row = tsv_rows[row_idx]
                mappings.append({
                    'cropped_video_path': str(video_path),
                    'cropped_video_filename': filename,
                    'tsv_row_index': row_idx,
                    'vid': tsv_row['vid'],
                    'yid': tsv_row['yid'],
                    'start': tsv_row['start'],
                    'end': tsv_row['end'],
                    'raw_text': tsv_row['raw-text'],
                    'tokenized_text': tsv_row['tokenized-text'],
                    'gloss': tsv_row.get('gloss', ''),
                    'split': tsv_row['split']
                })
            else:
                print(f"Warning: Row index {row_idx} from {filename} exceeds TSV length")
        else:
            print(f"Warning: Could not extract row index from {filename}")

    print(f"\nSuccessfully mapped {len(mappings)} videos to TSV entries")

    # Save to CSV if requested
    if output_csv and mappings:
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['cropped_video_path', 'cropped_video_filename', 'tsv_row_index',
                         'vid', 'yid', 'start', 'end', 'raw_text', 'tokenized_text',
                         'gloss', 'split']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(mappings)
        print(f"Saved mapping to {output_csv}")

    return mappings

def main():
    # Configuration
    cropped_video_dir = "/home/user/SLT/newdir"
    tsv_path = "/home/user/SLT/openasl-v1.0.tsv"
    output_csv = "/home/user/SLT/cropped_videos_mapping.csv"

    # Create the mapping
    mappings = link_cropped_videos_to_tsv(
        cropped_video_dir=cropped_video_dir,
        tsv_path=tsv_path,
        output_csv=output_csv
    )

    # Display sample results
    print("\n" + "="*80)
    print("Sample mappings:")
    print("="*80)
    for row in mappings[:5]:  # Show first 5
        print(f"\nVideo: {row['cropped_video_filename']}")
        print(f"  TSV Row: {row['tsv_row_index']}")
        print(f"  Video ID: {row['vid']}")
        print(f"  YouTube ID: {row['yid']}")
        print(f"  Time Range: {row['start']} - {row['end']}")
        print(f"  Text: {row['raw_text']}")
        print(f"  Split: {row['split']}")

    return mappings

if __name__ == "__main__":
    main()
