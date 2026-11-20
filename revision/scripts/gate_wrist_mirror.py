"""Copy registered native RGB-D review sequences and verify both images at every exposure."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import time

REMOTE = "/home/uav/paper_revision_20261117/revision/data/round_gate_feedback/formal"
METHODS = ("proposed", "oracle", "vision", "strain", "fixed")


def mirror(folder):
    optics = folder / "optics"
    index = optics / "index.jsonl"
    marker = optics / "pixels-verified.json"
    index_sha = hashlib.sha256(index.read_bytes()).hexdigest()
    if marker.exists() and json.loads(marker.read_text())["index_sha256"] == index_sha:
        return
    frames = [json.loads(line) for line in index.read_text().splitlines()]
    expected = {}
    for row in frames:
        fid = row["frame"]
        for key, filename in (("rgb", f"pixels/r{fid:06d}.jpg"), ("depth", f"pixels/d{fid:06d}.png")):
            if row[key] != filename or filename in expected:
                raise ValueError("Unexpected camera frame identifier")
            expected[filename] = row[key+"_sha256"]
    (optics / "pixels").mkdir(exist_ok=True)
    command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "nuc-uav",
               "tar", "-C", REMOTE+"/"+folder.name+"/optics", "-cf", "-", "pixels"]
    seen = set()
    total = 0
    with subprocess.Popen(command, stdout=subprocess.PIPE) as child:
        try:
            with tarfile.open(mode="r|", fileobj=child.stdout, bufsize=131072) as stream:
                for member in stream:
                    if member.isdir() and member.name.rstrip("/") == "pixels":
                        continue
                    if not member.isfile() or member.name not in expected or member.name in seen or member.size > 2*1024*1024:
                        raise ValueError("Unregistered camera asset: "+member.name)
                    data = stream.extractfile(member).read()
                    if hashlib.sha256(data).hexdigest() != expected[member.name]:
                        raise IOError("Camera image fingerprint mismatch: "+member.name)
                    target = optics / member.name
                    part = target.with_suffix(target.suffix+".part")
                    part.write_bytes(data)
                    part.replace(target)
                    seen.add(member.name)
                    total += len(data)
        finally:
            child.stdout.close()
        if child.wait() != 0 or seen != set(expected):
            raise IOError("Incomplete native RGB-D stream for "+folder.name)
    completed = {"case": folder.name, "frames": len(frames), "images_verified": len(seen),
                 "bytes": total, "index_sha256": index_sha}
    temp = marker.with_suffix(".part")
    temp.write_text(json.dumps(completed,indent=2)+"\n")
    temp.replace(marker)
    print(folder.name, len(frames), total, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", type=Path)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    while True:
        verified = 0
        for method in METHODS:
            folder = args.stage/"formal"/("26112131-differential-occluded-"+method)
            if not (folder/"optics/index.jsonl").is_file():
                continue
            try:
                mirror(folder)
                verified += 1
            except (OSError, subprocess.SubprocessError, tarfile.TarError) as error:
                if not args.watch:
                    raise
                print(folder.name, error,flush=True)
        if verified == len(METHODS) or not args.watch:
            return
        time.sleep(55)


if __name__=="__main__":
    main()
