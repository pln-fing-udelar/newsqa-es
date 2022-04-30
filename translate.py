#!/usr/bin/env python
from __future__ import annotations

import argparse
import itertools
import os
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, PreTrainedTokenizerBase


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="cnn_stories/cnn/stories")
    parser.add_argument("--output-path", default="cnn_stories/cnn/translated")

    parser.add_argument("--model-name", default="Helsinki-NLP/opus-mt-en-es")
    parser.add_argument("--device", default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    parser.add_argument("--batch-size", type=int, default=46)
    parser.add_argument("--num-workers", type=int,
                        default=len(os.sched_getaffinity(0)) // max(torch.cuda.device_count(), 1))
    return parser.parse_args()


def collate(instances: Sequence[Mapping[str, Any]],
            tokenizer: PreTrainedTokenizerBase | None = None) -> Mapping[str, Any]:
    keys = next(iter(instances), {})
    batch = {k: [instance[k] for instance in instances] for k in keys}

    if tokenizer:
        for k in keys:
            stack = batch[k]

            if k == "line":
                tokenized = tokenizer(stack, truncation=True, padding=True, return_tensors="pt")
                batch[f"{k}_ids"] = tokenized["input_ids"]
                batch[f"{k}_mask"] = tokenized["attention_mask"]

    return batch


class TextFileLineDataset(Dataset):
    def __getitem__(self, k: tuple[str, int]) -> Mapping[str, Any]:  # noqa
        path, line_number = k
        with open(path, encoding="utf-8") as file:
            for i, line in enumerate(file):
                if i == line_number:
                    return {"line": line.strip(), "path": path}


class TextFileLineSampler(Sampler):
    def __init__(self, paths: Iterable[str]) -> None:
        super().__init__(paths)  # noqa
        self.paths = paths

    def __iter__(self) -> Iterator[tuple[str, int]]:
        paths, self.paths = itertools.tee(self.paths)
        for path in paths:
            with open(path, encoding="utf-8") as file:
                for i, line in enumerate(file):
                    if line.strip():
                        yield path, i


def _write_file_maybe(folder: str, filename: str | None, lines: Iterable[str]) -> None:
    if filename is not None:
        with open(os.path.join(folder, filename), "w", encoding="utf-8") as file:
            file.write("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    model.eval()

    dataset = TextFileLineDataset()
    paths = (os.path.join(args.data_path, filename)
             for filename in os.listdir(args.data_path)
             if not os.path.exists(os.path.join(args.output_path, filename)))  # We only compute the missing files.
    sampler = TextFileLineSampler(paths)
    data_loader = DataLoader(dataset, sampler=sampler, batch_size=args.batch_size, num_workers=args.num_workers,
                             pin_memory=True, collate_fn=lambda instances: collate(instances, tokenizer=tokenizer))

    file_count = len(next(os.walk(args.data_path))[2])  # More efficient than `listdir` as it doesn't sort.
    done_file_count = len(next(os.walk(args.output_path))[2])

    with torch.inference_mode(), \
            tqdm(initial=done_file_count, total=file_count, desc="Translating", unit="file") as progress:
        last_filename = None
        translated_lines = None

        for batch in data_loader:
            batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            generated_token_ids = model.generate(batch["line_ids"], attention_mask=batch["line_mask"])
            translated_text = tokenizer.batch_decode(generated_token_ids, skip_special_tokens=True)

            for translated_line, path in zip(translated_text, batch["path"]):
                filename = os.path.basename(path)

                if last_filename != filename:  # A new file starts.
                    _write_file_maybe(args.output_path, last_filename, translated_lines)

                    last_filename = filename
                    translated_lines = []

                    progress.update()

                translated_lines.append(translated_line)

        _write_file_maybe(args.output_path, last_filename, translated_lines)

        progress.update()


if __name__ == "__main__":
    main()
