"""Load, validate, inspect, and split the BIPA pronunciation dataset."""

from __future__ import annotations

import argparse
import logging
import sys
import unicodedata
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download
from sklearn.model_selection import StratifiedGroupKFold


DATASET_ID = "thiagomonteles/BIPA"
DATASET_FILE = "por_bz_com_dialeto_55K.tsv"
DATASET_REVISION = "d1e81df1565756229bc20d6841716a7a5560fc80"
SPLIT_SEED = 42
DIALECTS = (
    "[brasil]",
    "[rio-de-janeiro]",
    "[são-paulo]",
    "[regiao-sul]",
    "[regiao-nordeste]",
    "[regiao-centro-oeste]",
)
DIALECT_ALIASES = {
    "brazil": "[brasil]",
    "brazil general": "[brasil]",
    "rio de janeiro": "[rio-de-janeiro]",
    "sao paulo": "[são-paulo]",
    "southern brazil": "[regiao-sul]",
    "south region": "[regiao-sul]",
    "south": "[regiao-sul]",
    "northeast brazil": "[regiao-nordeste]",
    "northeast region": "[regiao-nordeste]",
    "central west brazil": "[regiao-centro-oeste]",
    "center west brazil": "[regiao-centro-oeste]",
    "center west region": "[regiao-centro-oeste]",
    "central west region": "[regiao-centro-oeste]",
    "rural central brazil": "[regiao-centro-oeste]",
}


def normalize_dialect(label: str) -> str:
    if label in DIALECTS:
        return label
    normalized = unicodedata.normalize("NFKD", label)
    normalized = "".join(
        char for char in normalized if not unicodedata.combining(char)
    )
    normalized = " ".join(
        normalized.casefold()
        .replace("_", " ")
        .replace("-", " ")
        .replace("(", "")
        .replace(")", "")
        .split()
    )
    try:
        return DIALECT_ALIASES[normalized]
    except KeyError as exc:
        raise ValueError(
            f"Rótulo de dialeto não reconhecido no dataset: {label!r}"
        ) from exc


def _load_and_clean_dataset(
    input_path: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    if input_path is None:
        dataset_path = hf_hub_download(
            repo_id=DATASET_ID,
            filename=DATASET_FILE,
            repo_type="dataset",
            revision=DATASET_REVISION,
        )
    else:
        dataset_path = str(input_path)

    frame = pd.read_csv(dataset_path, sep="\t", dtype=str, keep_default_na=False)
    source_columns = {
        "dialeto": "dialect_label",
        "palavra": "word",
        "pronuncia": "pronunciation",
    }
    cleaned_columns = {
        "dialect": "dialect_label",
        "word": "word",
        "pronunciation": "pronunciation",
    }
    if set(source_columns).issubset(frame.columns):
        frame = frame.rename(columns=source_columns)
    elif set(cleaned_columns).issubset(frame.columns):
        frame = frame.rename(columns=cleaned_columns)
    else:
        missing = set(source_columns).difference(frame.columns)
        raise ValueError(
            f"Colunas ausentes no TSV do BIPA: {', '.join(sorted(missing))}"
        )
    frame["dialect"] = frame["dialect_label"].map(normalize_dialect)
    blank_rows = frame["word"].str.strip().eq("") | frame[
        "pronunciation"
    ].str.strip().eq("")
    blank_count = int(blank_rows.sum())
    frame = frame.loc[~blank_rows].copy()
    before_deduplication = len(frame)
    frame = frame.drop_duplicates(["dialect", "word", "pronunciation"])
    duplicate_count = before_deduplication - len(frame)
    frame = frame[["dialect", "word", "pronunciation"]].reset_index(drop=True)

    validate_dataset(frame)
    return frame, {
        "source_rows": int(len(blank_rows)),
        "blank_rows_removed": blank_count,
        "duplicate_rows_removed": duplicate_count,
    }


def load_dataset(input_path: Path | None = None) -> pd.DataFrame:
    """Load the pinned public dataset or a local TSV and return cleaned rows."""
    frame, _ = _load_and_clean_dataset(input_path)
    return frame


def validate_dataset(frame: pd.DataFrame) -> None:
    """Raise an explicit error if the cleaned dataset violates core invariants."""
    required = {"dialect", "word", "pronunciation"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            f"Colunas ausentes no dataset processado: {', '.join(sorted(missing))}"
        )
    if frame.empty:
        raise ValueError("O dataset não contém linhas válidas após a limpeza.")
    if frame[["word", "pronunciation"]].apply(
        lambda column: column.str.strip().eq("").any()
    ).any():
        raise ValueError("O dataset processado contém palavra ou pronúncia vazia.")
    unknown_dialects = set(frame["dialect"]).difference(DIALECTS)
    if unknown_dialects:
        raise ValueError(
            f"Dialetos não reconhecidos no dataset: {sorted(unknown_dialects)}"
        )
    if frame.duplicated(["dialect", "word", "pronunciation"]).any():
        raise ValueError("O dataset processado ainda contém duplicatas exatas.")


def create_splits(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Create group-safe 75/15/10 splits stratified by dialect and word length."""
    validate_dataset(frame)
    strata = frame["dialect"] + "|" + frame["word"].str.len().astype(str)
    groups = pd.factorize(
        pd.MultiIndex.from_frame(frame[["dialect", "word"]])
    )[0]
    test_splitter = StratifiedGroupKFold(
        n_splits=10, shuffle=True, random_state=SPLIT_SEED
    )
    remaining_indices, test_indices = next(
        test_splitter.split(frame[["word"]], strata, groups)
    )
    remaining = frame.iloc[remaining_indices].reset_index(drop=True)
    remaining_strata = (
        remaining["dialect"] + "|" + remaining["word"].str.len().astype(str)
    )
    remaining_groups = pd.factorize(
        pd.MultiIndex.from_frame(remaining[["dialect", "word"]])
    )[0]
    validation_splitter = StratifiedGroupKFold(
        n_splits=6, shuffle=True, random_state=SPLIT_SEED
    )
    train_indices, validation_indices = next(
        validation_splitter.split(
            remaining[["word"]], remaining_strata, remaining_groups
        )
    )
    splits = {
        "train": remaining.iloc[train_indices].reset_index(drop=True),
        "validation": remaining.iloc[validation_indices].reset_index(drop=True),
        "test": frame.iloc[test_indices].reset_index(drop=True),
    }
    pair_sets = {
        name: set(zip(part["dialect"], part["word"], strict=False))
        for name, part in splits.items()
    }
    split_names = tuple(splits)
    for index, first in enumerate(split_names):
        for second in split_names[index + 1 :]:
            if pair_sets[first] & pair_sets[second]:
                raise RuntimeError(f"Vazamento entre {first} e {second}.")
    return splits


def print_dataset_report(
    frame: pd.DataFrame, splits: dict[str, pd.DataFrame], cleaning: dict[str, int]
) -> None:
    pair_counts = frame.groupby("dialect")["word"].nunique()
    row_counts = frame.groupby("dialect").size()
    word_lengths = frame["word"].str.len()
    references_per_pair = frame.groupby(["dialect", "word"]).size()

    print("Verificações do dataset BIPA: OK")
    print(f"Registros de origem: {cleaning['source_rows']}")
    print(f"Linhas vazias removidas: {cleaning['blank_rows_removed']}")
    print(f"Duplicatas exatas removidas: {cleaning['duplicate_rows_removed']}")
    print(f"Registros válidos: {len(frame)}")
    print(f"Pares únicos dialeto-palavra: {len(references_per_pair)}")
    print(
        "Comprimento das palavras (caracteres): "
        f"mínimo={word_lengths.min()}, "
        f"mediana={word_lengths.median():g}, "
        f"máximo={word_lengths.max()}"
    )
    print(
        "Pronúncias de referência por par: "
        f"mínimo={references_per_pair.min()}, "
        f"mediana={references_per_pair.median():g}, "
        f"máximo={references_per_pair.max()}"
    )
    print("\nPor dialeto:")
    for dialect in DIALECTS:
        if dialect in row_counts:
            print(
                f"  {dialect}: {row_counts[dialect]} registros, "
                f"{pair_counts[dialect]} pares únicos"
            )

    print("\nPartições reproduzíveis (semente 42):")
    for name, part in splits.items():
        pair_count = part[["dialect", "word"]].drop_duplicates().shape[0]
        print(f"  {name}: {len(part)} registros, {pair_count} pares")
    print("Verificação de vazamento entre partições: OK")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analisa e valida o dataset BIPA sem carregar o modelo."
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="TSV local com as colunas dialeto, palavra e pronuncia.",
    )
    parser.add_argument(
        "--save-cleaned",
        type=Path,
        help="Salva o dataset limpo em TSV, se desejado.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    frame, cleaning = _load_and_clean_dataset(args.input)
    splits = create_splits(frame)
    print_dataset_report(frame, splits, cleaning)
    if args.save_cleaned:
        args.save_cleaned.parent.mkdir(parents=True, exist_ok=True)
        frame.rename(
            columns={
                "dialect": "dialeto",
                "word": "palavra",
                "pronunciation": "pronuncia",
            }
        ).to_csv(args.save_cleaned, sep="\t", index=False, encoding="utf-8")
        print(f"\nDataset limpo salvo em: {args.save_cleaned.resolve()}")


if __name__ == "__main__":
    main()
