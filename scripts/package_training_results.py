"""Package existing run evidence with an honest inventory of missing/empty files."""
import argparse
import hashlib
import json
import re
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

import yaml

REQUIRED = ("config.yaml", "train.log", "metrics.json", "per_query_predictions.json",
            "constraint_scores.json", "best_checkpoint.pt", "seed.txt", "git_commit.txt")
OPTIONAL = ("run_protocol.json", "server_config.yaml", "server_preflight.json",
            "b5_override_config.yaml", "fashioniq_metrics.json",
            "fashioniq_per_query_predictions.json", "fashioniq_constraint_scores.json")


def inspect(path):
    if not path.is_file():
        return dict(status="missing")
    size = path.stat().st_size
    result = dict(status="present", bytes=size)
    if not size:
        result["status"] = "empty"
        return result
    if path.suffix == ".json":
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data is None or data == [] or data == {}:
                result["status"] = "empty_placeholder"
        except (ValueError, UnicodeError):
            result["status"] = "invalid_json"
    if path.name == "git_commit.txt":
        value = path.read_text(encoding="utf-8").strip()
        if not re.fullmatch(r"[a-fA-F0-9]{40}|[a-fA-F0-9]{64}", value):
            result["status"] = "unknown_or_invalid_commit"
    return result


def package(runs, mining_dir, output):
    runs = [r.resolve() for r in runs]
    if len({r.name for r in runs}) != len(runs):
        raise ValueError("Run folder names must be distinct")
    if any(not r.is_dir() for r in runs) or not mining_dir.is_dir():
        raise ValueError("A run or mining directory does not exist")
    # No fabricated data, no source edits, no implicit collection of unrelated files.
    report = dict(status="EVIDENCE_INVENTORY_NOT_SUBMISSION_CERTIFICATION", runs={}, mining={},
        notes=["File presence does not establish benchmark validity or audit completion.",
               "Counterfactual evaluation is not CIRR/Fashion-IQ retrieval evaluation.",
               "No missing predictions, checkpoints or Git revisions are fabricated.",
               "The existing B5 manual override record is preserved when present.",
               "Historical data-quality and split-overlap limitations still apply."])
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "x", ZIP_DEFLATED) as archive:
        for run in runs:
            inventory = {}
            for name in (*REQUIRED, *OPTIONAL):
                path = run / name
                status = inspect(path)
                if name in REQUIRED or status["status"] != "missing":
                    inventory[name] = status
                if path.is_file():
                    if path.is_symlink() or not path.resolve().is_relative_to(run):
                        raise ValueError(f"Refusing indirect file: {path}")
                    with path.open("rb") as f:
                        status["sha256"] = hashlib.file_digest(f, "sha256").hexdigest()
                    archive.write(path, f"runs/{run.name}/{name}",
                                  compress_type=ZIP_STORED if path.suffix == ".pt" else ZIP_DEFLATED)
            report["runs"][run.name] = dict(source=str(run), files=inventory)
        audit = mining_dir / "mining_audit.csv"
        report["mining"]["mining_audit.csv"] = inspect(audit)
        if audit.is_file():
            archive.write(audit, "mining/mining_audit.csv")
        yaml_path = mining_dir / "mining_config.yaml"
        json_path = mining_dir / "mining_config.json"
        if yaml_path.is_file():
            report["mining"]["mining_config.yaml"] = inspect(yaml_path)
            archive.write(yaml_path, "mining/mining_config.yaml")
        elif json_path.is_file():
            config = json.loads(json_path.read_text(encoding="utf-8"))
            if not isinstance(config, dict):
                raise ValueError("Mining config must be a mapping")
            archive.writestr("mining/mining_config.yaml", yaml.safe_dump(config, sort_keys=False))
            archive.write(json_path, "mining/mining_config.original.json")
            report["mining"]["mining_config.yaml"] = dict(status="converted_from_json",
                source=str(json_path.resolve()), note="Same values, YAML representation; not a new mining run.")
        else:
            report["mining"]["mining_config.yaml"] = dict(status="missing")
        archive.writestr("inventory.json", json.dumps(report, indent=2))
        archive.writestr("READ_ME.txt",
            "Existing B1/B4/B5 training evidence. Inspect inventory.json before submission.\n"
            "Empty [] prediction files are placeholders, not completed benchmark results.\n"
            "This package does not certify CIRR/Fashion-IQ evaluation or human audit completion.\n"
            "Unknown Git commit remains unknown; the source was transferred as a ZIP.\n"
            "B5 manual override, if present, is recorded in its run_protocol.json.\n")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs="+", type=Path, required=True)
    p.add_argument("--mining-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    report = package(a.runs, a.mining_dir, a.output)
    for name, run in report["runs"].items():
        print(name)
        for filename, entry in run["files"].items():
            print(f"  {filename}: {entry['status']}")
    print("Mining:", json.dumps(report["mining"], indent=2))
    print(f"DONE: {a.output.resolve()}")
    print("Check inventory.json; missing/empty files mean further work is required.")


if __name__ == "__main__":
    main()
