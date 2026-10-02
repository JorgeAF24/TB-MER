from __future__ import annotations

import re
import os
from pathlib import Path

import pandas as pd

# Use regular expressions to parse the emotion and transcription files
# The emotion files have lines in the format: [start_time - end_time] utterance_id emotion
# Use the parenthesis to save the start_time, end_time, utterance_id and emotion as groups for later extraction
EMO_PATTERN = re.compile(
    r"\[(.*?) - (.*?)\]\s+"
    r"(\S+)\s+"
    r"(\w+)\s+"
)

# The transcription files have lines in the format: utterance_id [speaker] text
# Use the parenthesis to save the utterance_id, speaker and text as groups for later extraction
TRANS_PATTERN = re.compile(
    r"(\S+)\s+\[(.*?)\]\:\s+(.*)"
)
def parse_transcription_file(path: Path):

    utterances = {}

    with open(path, "r", encoding="utf-8", errors="ignore") as f:

        for line in f:

            m = TRANS_PATTERN.match(line.strip())

            if m is None:
                continue

            utterance_id = m.group(1)
            text = m.group(3)

            utterances[utterance_id] = text

    return utterances

def parse_emotion_file(path: Path):

    records = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:

        for line in f:

            m = EMO_PATTERN.match(line)

            if m is None:
                continue

            records.append(
                {
                    "start_time": float(m.group(1)),
                    "end_time": float(m.group(2)),
                    "utterance_id": m.group(3),
                    "emotion": m.group(4)
                }
            )

    return records

def parse_session(session_dir: Path):

    records = []

    emo_dir = session_dir / "dialog" / "EmoEvaluation"

    trans_dir = session_dir / "dialog" / "transcriptions"

    video_dir = session_dir / "dialog" / "avi" / "DivX"

    for emo_file in emo_dir.glob("*.txt"):

        # Use .stem to extract only the name without extension of the file which is the dialogue ID 
        dialogue_id = emo_file.stem

        trans_file = trans_dir / f"{dialogue_id}.txt"

        if not trans_file.exists():
            continue

        texts = parse_transcription_file(trans_file)

        emotion_records = parse_emotion_file(emo_file)

        video_path = video_dir / f"{dialogue_id}.avi"

        for r in emotion_records:

            utterance_id = r["utterance_id"]

            records.append(
                {
                    "dialogue_id": dialogue_id,
                    "utterance_id": utterance_id,
                    "emotion": r["emotion"],
                    "start_time": r["start_time"],
                    "end_time": r["end_time"],
                    "text": texts.get(utterance_id, ""),
                    "video_path": str(video_path)
                }
            )

    return records
def parse_iemocap(root_dir):

    all_records = []

    for session_dir in sorted(Path(root_dir).glob("Session*")):

        print(f"Processing {session_dir.name}")

        all_records.extend(parse_session(session_dir))

    return pd.DataFrame(all_records)

if __name__ == "__main__":

    df = parse_iemocap("data/IEMOCAP")

    print(df.head())

    print(f"Total utterances: {len(df)}")

    print(df["emotion"].value_counts())

    print(sorted(df["emotion"].unique()))

    os.makedirs("data/IEMOCAP_metadata",exist_ok=True)

    df.to_csv("data/IEMOCAP_metadata/metadata.csv",index=False)
# Running args: python src/datasets/iemocap_parser.py