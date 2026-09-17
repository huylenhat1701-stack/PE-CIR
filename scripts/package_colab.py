"""Create the source-only Colab bundle."""

from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

INCLUDE = ("src", "scripts", "configs", "tests", "pyproject.toml", "README.md")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    output = root / "output" / "CF-PE-CIR-colab-code.zip"
    output.parent.mkdir(exist_ok=True)
    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        for name in INCLUDE:
            path = root / name
            if path.is_file():
                archive.write(path, path.relative_to(root).as_posix())
                continue
            for child in path.rglob("*"):
                if (
                    child.is_file()
                    and "__pycache__" not in child.parts
                    and not any(part.endswith(".egg-info") for part in child.parts)
                ):
                    archive.write(child, child.relative_to(root).as_posix())
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
