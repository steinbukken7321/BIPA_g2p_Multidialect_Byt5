"""Prepare BIPA data and run model training or inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import secrets
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd
from huggingface_hub import hf_hub_download
from sklearn.model_selection import StratifiedGroupKFold

from bipa_model import (
    BASE_MODEL,
    BASE_MODEL_REVISION,
    BIPAG2PModel,
    PUBLISHED_MODEL,
    PUBLISHED_MODEL_REVISION,
)


LOGGER = logging.getLogger("bipa")
PROJECT_DIR = Path(__file__).resolve().parent
DATASET_REPO = "thiagomonteles/BIPA"
DATASET_FILE = "por_bz_com_dialeto_55K.tsv"
DATASET_REVISION = "d1e81df1565756229bc20d6841716a7a5560fc80"
EXPECTED_EXAMPLES = 292_862
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
SPLIT_NAMES = ("train", "validation", "test")
DIALECTS = (
    "[brasil]",
    "[rio-de-janeiro]",
    "[são-paulo]",
    "[regiao-sul]",
    "[regiao-nordeste]",
    "[regiao-centro-oeste]",
)

_DIALECT_ALIASES = {
    "brazil": "[brasil]",
    "brazil (general)": "[brasil]",
    "rio de janeiro": "[rio-de-janeiro]",
    "sao paulo": "[são-paulo]",
    "southern brazil": "[regiao-sul]",
    "south region": "[regiao-sul]",
    "south": "[regiao-sul]",
    "northeast brazil": "[regiao-nordeste]",
    "northeast region": "[regiao-nordeste]",
    "central-west brazil": "[regiao-centro-oeste]",
    "center-west brazil": "[regiao-centro-oeste]",
    "center-west region": "[regiao-centro-oeste]",
    "central-west region": "[regiao-centro-oeste]",
    "central west brazil": "[regiao-centro-oeste]",
    "rural central brazil": "[regiao-centro-oeste]",
}


def normalize_dialect(label: str) -> str:
    normalized = unicodedata.normalize("NFKD", label)
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    normalized = " ".join(normalized.casefold().replace("_", " ").split())
    try:
        return _DIALECT_ALIASES[normalized]
    except KeyError as exc:
        raise ValueError(
            f"Rótulo de dialeto não reconhecido: {label!r}. "
            "Confira o rótulo no TSV antes de atualizar o mapeamento."
        ) from exc


def load_and_clean_dataset(source: Path) -> pd.DataFrame:
    frame = pd.read_csv(source, sep="\t", dtype=str, keep_default_na=False)
    required = {"dialeto", "palavra", "pronuncia"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            f"Colunas ausentes: {', '.join(sorted(missing))}. "
            f"Encontradas: {', '.join(frame.columns)}"
        )
    frame = frame.rename(
        columns={
            "dialeto": "dialect_label",
            "palavra": "word",
            "pronuncia": "pronunciation",
        }
    )
    frame = frame.loc[:, ["dialect_label", "word", "pronunciation"]].copy()
    frame["dialect"] = frame["dialect_label"].map(normalize_dialect)
    frame = frame.drop(columns="dialect_label")
    frame = frame[
        frame["word"].str.strip().ne("")
        & frame["pronunciation"].str.strip().ne("")
    ].copy()
    frame = frame.drop_duplicates(["dialect", "word", "pronunciation"])
    frame = frame.sort_values(
        ["dialect", "word", "pronunciation"], kind="stable"
    ).reset_index(drop=True)
    if len(frame) != EXPECTED_EXAMPLES:
        LOGGER.warning(
            "Dataset após deduplicação: %s exemplos; o artigo informa %s.",
            f"{len(frame):,}",
            f"{EXPECTED_EXAMPLES:,}",
        )
    return frame


def create_splits(frame: pd.DataFrame, seed: int) -> dict[str, pd.DataFrame]:
    """Group by dialect-word and stratify by dialect plus word length."""
    strata = frame["dialect"] + "|" + frame["word"].str.len().astype(str)
    groups = pd.factorize(pd.MultiIndex.from_frame(frame[["dialect", "word"]]))[0]
    outer = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
    remaining_indices, test_indices = next(
        outer.split(frame[["word"]], strata, groups)
    )
    remaining = frame.iloc[remaining_indices].reset_index(drop=True)
    remaining_strata = (
        remaining["dialect"] + "|" + remaining["word"].str.len().astype(str)
    )
    remaining_groups = pd.factorize(
        pd.MultiIndex.from_frame(remaining[["dialect", "word"]])
    )[0]
    validation_splitter = StratifiedGroupKFold(
        n_splits=6, shuffle=True, random_state=seed
    )
    train_indices, validation_indices = next(
        validation_splitter.split(
            remaining[["word"]], remaining_strata, remaining_groups
        )
    )

    splits = {
        "train": remaining.iloc[train_indices],
        "validation": remaining.iloc[validation_indices],
        "test": frame.iloc[test_indices],
    }
    pair_sets = {
        name: set(zip(part["dialect"], part["word"], strict=False))
        for name, part in splits.items()
    }
    for index, first in enumerate(SPLIT_NAMES):
        for second in SPLIT_NAMES[index + 1 :]:
            if pair_sets[first] & pair_sets[second]:
                raise RuntimeError(f"Vazamento entre {first} e {second}.")
    return {
        name: part.sort_values(
            ["dialect", "word", "pronunciation"], kind="stable"
        ).reset_index(drop=True)
        for name, part in splits.items()
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_data(seed: int | None) -> None:
    used_seed = seed if seed is not None else secrets.randbits(32)
    LOGGER.info("Seed usada para criar as partições: %s", used_seed)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    source = Path(
        hf_hub_download(
            repo_id=DATASET_REPO,
            filename=DATASET_FILE,
            repo_type="dataset",
            revision=DATASET_REVISION,
        )
    )
    frame = load_and_clean_dataset(source)
    splits = create_splits(frame, used_seed)
    split_counts: dict[str, dict[str, int]] = {}
    for name, part in splits.items():
        part.to_csv(PROCESSED_DIR / f"{name}.tsv", sep="\t", index=False)
        split_counts[name] = {
            "examples": len(part),
            "dialect_word_pairs": int(
                part[["dialect", "word"]].drop_duplicates().shape[0]
            ),
        }

    manifest: dict[str, Any] = {
        "dataset_repo": DATASET_REPO,
        "dataset_file": DATASET_FILE,
        "dataset_revision": DATASET_REVISION,
        "dataset_sha256": sha256_file(source),
        "base_model": BASE_MODEL,
        "base_model_revision": BASE_MODEL_REVISION,
        "seed": used_seed,
        "deduplicated_examples": len(frame),
        "split_method": (
            "StratifiedGroupKFold over deduplicated examples, grouped by "
            "(dialect, word), stratified by dialect and word length."
        ),
        "splits": split_counts,
        "paper_reported_examples": {
            "train": 219_921,
            "validation": 43_699,
            "test": 29_242,
        },
        "reproduction_note": (
            "Reported split sizes imply 75%/15%/10%; exact split membership "
            "cannot be reconstructed because the authors did not publish indices "
            "or a random seed."
        ),
    }
    (PROCESSED_DIR / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    LOGGER.info("Exemplos únicos: %s", f"{len(frame):,}")
    for name, counts in split_counts.items():
        LOGGER.info(
            "%s: %s exemplos, %s pares dialeto-palavra",
            name,
            f"{counts['examples']:,}",
            f"{counts['dialect_word_pairs']:,}",
        )


def read_split(name: str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"{name}.tsv"
    if not path.exists():
        raise FileNotFoundError(
            f"Não encontrei {path}. Rode primeiro: "
            "python reproduce_bipa.py prepare"
        )
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def resolve_project_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_DIR / candidate


def run_training(args: argparse.Namespace) -> None:
    train_frame = read_split("train")
    validation_frame = read_split("validation")
    test_frame = read_split("test")
    model = BIPAG2PModel.from_base_model()
    model.train(
        train_frame=train_frame,
        validation_frame=validation_frame,
        test_frame=test_frame,
        output_dir=resolve_project_path(args.output_dir),
        seed=args.seed,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
    )


def run_inference(args: argparse.Namespace) -> None:
    revision = args.revision
    if revision is None and args.model == PUBLISHED_MODEL:
        revision = PUBLISHED_MODEL_REVISION
    model = BIPAG2PModel.from_pretrained(args.model, revision=revision)

    if args.split:
        metrics = model.evaluate(
            read_split(args.split),
            resolve_project_path(args.output_dir) / args.split,
            batch_size=args.batch_size,
        )
        LOGGER.info(
            "Resultado salvo em %s (Minimum PER macro %.2f%%)",
            (resolve_project_path(args.output_dir) / args.split).resolve(),
            metrics["min_per"],
        )
        return

    if not args.text or not args.dialect:
        raise ValueError("Passe --text e --dialect, ou use --split para avaliar.")
    inputs = [f"{args.dialect} {word}" for word in args.text]
    predictions = model.infer(inputs, batch_size=args.batch_size)
    for word, prediction in zip(args.text, predictions, strict=True):
        print(f"{args.dialect} {word}\t{prediction}")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepara o BIPA e executa treino ou inferência do modelo G2P."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser(
        "prepare", help="Baixa e divide o dataset sem vazamento entre partições."
    )
    prepare_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed opcional para repetir a divisão; sem ela, uma seed aleatória é gerada.",
    )

    train_parser = subparsers.add_parser(
        "train", help="Treina ByT5-small a partir do modelo base."
    )
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--output-dir", default="runs/bipa_byt5")
    train_parser.add_argument("--train-batch-size", type=int, default=96)
    train_parser.add_argument("--eval-batch-size", type=int, default=16)

    inference_parser = subparsers.add_parser(
        "inference", help="Gera pronúncias ou avalia um split com um checkpoint."
    )
    inference_parser.add_argument("--model", default=PUBLISHED_MODEL)
    inference_parser.add_argument("--revision")
    inference_parser.add_argument("--text", nargs="+", help="Uma ou mais palavras.")
    inference_parser.add_argument(
        "--dialect", choices=DIALECTS, help="Etiqueta de dialeto, por exemplo [brasil]."
    )
    inference_parser.add_argument(
        "--split", choices=("validation", "test"), help="Avalia este split preparado."
    )
    inference_parser.add_argument("--batch-size", type=int, default=16)
    inference_parser.add_argument("--output-dir", default="runs/inference")
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = make_parser().parse_args()
    if args.command == "prepare":
        prepare_data(args.seed)
    elif args.command == "train":
        run_training(args)
    elif args.command == "inference":
        if args.split and (args.text or args.dialect):
            raise ValueError("Use --split ou --text/--dialect, não ambos.")
        run_inference(args)


if __name__ == "__main__":
    main()
