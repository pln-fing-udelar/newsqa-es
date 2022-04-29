#!/usr/bin/env python
from __future__ import annotations

import itertools
import os
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Any

import torch
from torch.utils.data import DataLoader, IterableDataset
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, PreTrainedTokenizerBase

DATA_PATH = "cnn_stories/cnn/stories"
OUTPUT_PATH = "cnn_stories/cnn/translated"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_WORKERS = len(os.sched_getaffinity(0)) // max(torch.cuda.device_count(), 1)

BATCH_SIZE = 63

MODEL_NAME = "Helsinki-NLP/opus-mt-en-es"


class CNNStoryDataset(IterableDataset):
    def __init__(self, paths: Iterable[str], tokenizer: PreTrainedTokenizerBase | None = None) -> None:
        self.paths = paths
        self.tokenizer = tokenizer

    def __iter__(self) -> Iterator[Mapping[str, Any]]:
        paths, self.paths = itertools.tee(self.paths)
        for path in paths:
            with open(path, encoding="utf-8") as file:
                for line in file:
                    if line := line.strip():
                        yield {"line": line, "path": path}

    def collate(self, instances: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
        keys = next(iter(instances), {})
        batch = {k: [instance[k] for instance in instances] for k in keys}

        if self.tokenizer:
            for k in keys:
                stack = batch[k]

                if k == "line":
                    tokenized = self.tokenizer(stack, truncation=True, padding=True, return_tensors="pt")
                    batch[f"{k}_ids"] = tokenized["input_ids"]
                    batch[f"{k}_mask"] = tokenized["attention_mask"]

        return batch


def _write_file_maybe(filename: str | None, lines: Iterable[str]) -> None:
    if filename is not None:
        with open(os.path.join(OUTPUT_PATH, filename), "w", encoding="utf-8") as file:
            file.write("\n".join(lines) + "\n")


def main() -> None:
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model.eval()

    paths = (os.path.join(DATA_PATH, filename)
             for filename in os.listdir(DATA_PATH)
             if not os.path.exists(os.path.join(OUTPUT_PATH, filename)))  # We only compute the missing files.
    dataset = CNNStoryDataset(paths=paths, tokenizer=tokenizer)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True,
                             collate_fn=dataset.collate)

    file_count = len(next(os.walk(DATA_PATH))[2])  # More efficient to count files than `listdir` as it doesn't sort.
    done_file_count = len(next(os.walk(OUTPUT_PATH))[2])

    with torch.inference_mode(), \
            tqdm(initial=done_file_count, total=file_count, desc="Translating", unit="file") as progress:
        last_filename = None
        translated_lines = None

        for batch in data_loader:
            batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            generated_token_ids = model.generate(batch["line_ids"], attention_mask=batch["line_mask"])
            translated_text = tokenizer.batch_decode(generated_token_ids, skip_special_tokens=True)

            for translated_line, path in zip(translated_text, batch["path"]):
                filename = os.path.basename(path)

                if last_filename != filename:  # A new file starts.
                    _write_file_maybe(last_filename, translated_lines)

                    last_filename = filename
                    translated_lines = []

                    progress.update()

                translated_lines.append(translated_line)

        _write_file_maybe(last_filename, translated_lines)

        progress.update()


if __name__ == "__main__":
    main()
