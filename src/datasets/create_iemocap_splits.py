from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


VALID_EMOTIONS = {
    "ang",
    "hap",
    "exc",
    "sad",
    "neu",
    "fru"
}


def get_session(dialogue_id: str) -> str:
    """
    Ses01F_impro01 -> Session1
    """
    return f"Session{int(dialogue_id[3:5])}"


def main():

    metadata_path = Path("data") / "IEMOCAP" / "metadata" / "iemocap_metadata.csv"
    

    output_dir = metadata_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(metadata_path)

    print(f"Raw utterances: {len(df)}")

    # --------------------------------------------------
    # Keep only standard 6-class protocol
    # --------------------------------------------------

    df = df[df["emotion"].isin(VALID_EMOTIONS)].copy()

    print(f"Filtered utterances: {len(df)}")

    # --------------------------------------------------
    # Session assignment
    # --------------------------------------------------

    df["session"] = df["dialogue_id"].apply(get_session)

    # --------------------------------------------------
    # Test = Session5
    # --------------------------------------------------

    test_df = df[df["session"] == "Session5"].copy()

    train_dev_df = df[df["session"] != "Session5"].copy()

    # --------------------------------------------------
    # Dialogue-level split
    # --------------------------------------------------

    dialogues = train_dev_df["dialogue_id"].unique()

    train_dialogues, dev_dialogues = train_test_split(
        dialogues,
        test_size=0.10,
        random_state=42,
        shuffle=True
    )

    train_df = train_dev_df[
        train_dev_df["dialogue_id"].isin(train_dialogues)
    ].copy()

    dev_df = train_dev_df[
        train_dev_df["dialogue_id"].isin(dev_dialogues)
    ].copy()

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    train_path = output_dir / "iemocap_train.csv"
    dev_path = output_dir / "iemocap_dev.csv"
    test_path = output_dir / "iemocap_test.csv"

    train_df.to_csv(train_path, index=False)
    dev_df.to_csv(dev_path, index=False)
    test_df.to_csv(test_path, index=False)

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    print()
    print("=" * 60)

    print("Train utterances:", len(train_df))
    print("Dev utterances  :", len(dev_df))
    print("Test utterances :", len(test_df))

    print()

    print("Train dialogues:", train_df["dialogue_id"].nunique())
    print("Dev dialogues  :", dev_df["dialogue_id"].nunique())
    print("Test dialogues :", test_df["dialogue_id"].nunique())

    print()

    print("Train distribution")
    print(train_df["emotion"].value_counts())

    print()

    print("Dev distribution")
    print(dev_df["emotion"].value_counts())

    print()

    print("Test distribution")
    print(test_df["emotion"].value_counts())

    print("=" * 60)

    print("Dialogues in both train and dev:", set(train_df["dialogue_id"]) & set(dev_df["dialogue_id"]))
    print("Dialogues in both train and test:", set(train_df["dialogue_id"]) & set(test_df["dialogue_id"]))


if __name__ == "__main__":
    main()
# Running args: python src/datasets/create_iemocap_splits.py