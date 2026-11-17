#!/usr/bin/env python3
"""Restore split archives to a separate directory and verify every raw member."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]


class Parts:
    def __init__(self, paths):
        self.paths = iter(paths)
        self.handle = None

    def read(self, size):
        result = bytearray()
        while len(result) < size:
            if self.handle is None:
                try:
                    self.handle = next(self.paths).open("rb")
                except StopIteration:
                    break
            block = self.handle.read(size - len(result))
            if block:
                result.extend(block)
            else:
                self.handle.close()
                self.handle = None
        return bytes(result)

    def close(self):
        if self.handle is not None:
            self.handle.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="*", help="Archive names; default: all four datasets")
    parser.add_argument("--output", type=Path, default=ROOT / ".reproduction/restored")
    args = parser.parse_args()
    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    manifests = sorted((ROOT / "datasets").glob("*.manifest.json"))
    if args.names:
        manifests = [p for p in manifests if p.name[:-len(".manifest.json")] in args.names]
        if len(manifests) != len(set(args.names)):
            raise SystemExit("unknown or duplicate dataset name")
    count = 0
    for manifest in manifests:
        data = json.loads(manifest.read_text())
        expected = {row["path"]: row for row in data["members"]}
        paths = []
        for part in data["parts"]:
            path = ROOT / part["path"]
            h = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    h.update(block)
            if h.hexdigest() != part["sha256"]:
                raise SystemExit("corrupt archive part: " + str(path))
            paths.append(path)
        reader = Parts(paths)
        seen = set()
        try:
            with tarfile.open(fileobj=reader, mode="r|gz") as archive:
                for member in archive:
                    relative = Path(member.name)
                    if not member.isfile() or relative.is_absolute() or ".." in relative.parts or member.name not in expected:
                        raise SystemExit("unexpected archive member: " + member.name)
                    target = destination / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    h = hashlib.sha256()
                    mode = "xb" if not target.exists() else None
                    output = target.open(mode) if mode else None
                    try:
                        stream = archive.extractfile(member)
                        for block in iter(lambda: stream.read(1024 * 1024), b""):
                            h.update(block)
                            if output:
                                output.write(block)
                    finally:
                        if output:
                            output.close()
                    if h.hexdigest() != expected[member.name]["sha256"]:
                        raise SystemExit("raw member checksum mismatch: " + member.name)
                    if mode is None:
                        existing = hashlib.sha256()
                        with target.open("rb") as handle:
                            for block in iter(lambda: handle.read(1024 * 1024), b""):
                                existing.update(block)
                        if existing.hexdigest() != h.hexdigest():
                            raise SystemExit("existing restored file differs; use a fresh --output: " + str(target))
                    seen.add(member.name)
                    count += 1
        finally:
            reader.close()
        if seen != set(expected):
            raise SystemExit("missing archive members")
        print(data["name"], "PASS", len(seen), flush=True)
    print(json.dumps({"restored_files": count, "output": str(destination)}, indent=2))


if __name__ == "__main__":
    main()
