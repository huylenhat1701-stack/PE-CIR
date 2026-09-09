"""Download reachable CIRR validation images from the public NLVR2 URL list."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image

from pic2word.data import load_cirr_split

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 Chrome/126 Safari/537.36"
)
MAX_IMAGE_BYTES = 25 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class DownloadFailure:
    image_id: str
    url: str
    error: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/cirr"))
    parser.add_argument(
        "--nlvr-json",
        type=Path,
        default=Path("data/nlvr2_url_source/dev.json"),
    )
    parser.add_argument(
        "--hashes",
        type=Path,
        default=Path("data/nlvr2_url_source/dev_hashes.json"),
    )
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--failure-log",
        type=Path,
        default=Path("reports/cirr_url_download_failures.json"),
    )
    return parser


def load_nlvr_urls(path: Path) -> dict[str, str]:
    """Map NLVR2 image IDs to their original public image URLs."""

    urls: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                entry = json.loads(line)
                base_id = "-".join(entry["identifier"].split("-")[:3])
                urls.setdefault(f"{base_id}-img0", entry["left_url"])
                urls.setdefault(f"{base_id}-img1", entry["right_url"])
            except (KeyError, TypeError, json.JSONDecodeError) as error:
                raise ValueError(f"Invalid NLVR2 JSON entry at line {line_number}") from error
    return urls


def average_hash(image: Image.Image, *, hash_size: int = 8) -> str:
    """Reproduce ``imagehash.average_hash`` used by the official downloader."""

    pixels = np.asarray(
        image.convert("L").resize((hash_size, hash_size), Image.Resampling.LANCZOS)
    )
    bits = pixels > pixels.mean()
    return f"{int(''.join('1' if value else '0' for value in bits.flat), 2):016x}"


def _read_response(url: str, timeout: float) -> bytes:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=timeout) as response:
        if response.status != 200:
            raise HTTPError(url, response.status, "non-200 response", response.headers, None)
        content = response.read(MAX_IMAGE_BYTES + 1)
    if len(content) > MAX_IMAGE_BYTES:
        raise ValueError("response exceeds the 25 MiB image limit")
    return content


def download_one(
    image_id: str,
    url: str,
    expected_hash: str,
    destination: Path,
    timeout: float,
) -> DownloadFailure | None:
    """Download, decode, and hash-check one image before saving it."""

    if destination.is_file():
        try:
            with Image.open(destination) as image:
                if average_hash(image) == expected_hash:
                    return None
        except (OSError, ValueError):
            pass
    try:
        content = _read_response(url, timeout)
        with Image.open(BytesIO(content)) as image:
            image.load()
            actual_hash = average_hash(image)
        if actual_hash != expected_hash:
            raise ValueError(f"hash mismatch: expected {expected_hash}, received {actual_hash}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = destination.with_suffix(destination.suffix + ".part")
        temporary_path.write_bytes(content)
        temporary_path.replace(destination)
        return None
    except (HTTPError, URLError, TimeoutError, OSError, ValueError) as error:
        return DownloadFailure(image_id, url, f"{type(error).__name__}: {error}")


def main() -> int:
    args = build_parser().parse_args()
    if args.workers <= 0 or args.timeout <= 0:
        raise ValueError("workers and timeout must be greater than zero")
    dataset = load_cirr_split(args.dataset_root)
    urls = load_nlvr_urls(args.nlvr_json)
    hashes = json.loads(args.hashes.read_text(encoding="utf-8"))
    image_ids = list(dataset.candidate_ids[: args.limit])
    unmapped = [
        image_id
        for image_id in image_ids
        if image_id not in urls or f"{image_id}.png" not in hashes
    ]
    if unmapped:
        raise ValueError(f"NLVR2 URL/hash data is missing {len(unmapped)} required IDs")

    failures: list[DownloadFailure] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                download_one,
                image_id,
                urls[image_id],
                hashes[f"{image_id}.png"],
                dataset.path_for(image_id),
                args.timeout,
            ): image_id
            for image_id in image_ids
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            failure = future.result()
            if failure is not None:
                failures.append(failure)
            if completed == 1 or completed % 100 == 0 or completed == len(futures):
                print(f"Processed {completed}/{len(futures)} images; failures={len(failures)}")

    failure_path = args.failure_log.resolve()
    failure_path.parent.mkdir(parents=True, exist_ok=True)
    failure_path.write_text(
        json.dumps([asdict(failure) for failure in failures], indent=2),
        encoding="utf-8",
    )
    available = len(dataset.candidate_ids) - len(dataset.missing_images())
    print(f"Available CIRR images: {available}/{len(dataset.candidate_ids)}")
    print(f"Failure log: {failure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
