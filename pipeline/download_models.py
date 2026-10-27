"""Download public model weights; verify every new file against its published SHA256."""
import argparse
import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import ProxyHandler, build_opener

ROOT = Path(__file__).resolve().parent / "models"


def fetch(model, owner="Qwen"):
    opener = build_opener(ProxyHandler({}))
    url = f"https://modelscope.cn/api/v1/models/{owner}/{model}/repo/files?Revision=master&Recursive=true"
    with opener.open(url, timeout=30) as response:
        metadata = json.load(response)
    if metadata.get("Code") != 200:
        raise RuntimeError(str(metadata)[:200])
    files = [f for f in metadata["Data"]["Files"] if f["Type"] == "blob" and "/" not in f["Path"]
             and f["Path"].endswith((".json", ".py", ".safetensors", ".model", ".jinja", ".yml", ".txt"))]
    destination = ROOT / model
    destination.mkdir(parents=True, exist_ok=True)
    def one(info):
        path = destination / info["Path"]
        if path.exists() and path.stat().st_size == info["Size"]:
            return
        part = path.with_name(path.name + ".part")
        source = f"https://modelscope.cn/models/{owner}/{model}/resolve/master/{info['Path']}"
        subprocess.run(["curl", "--noproxy", "*", "--fail", "--silent", "--show-error", "--location",
                        "--retry", "3", "--connect-timeout", "10", "--max-time", "3600", "--continue-at", "-",
                        "--output", str(part), source], check=True)
        if part.stat().st_size != info["Size"]:
            raise ValueError("下载长度不符: " + str(path))
        digest = hashlib.sha256()
        with part.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        if info.get("Sha256") and digest.hexdigest() != info["Sha256"]:
            raise ValueError("SHA256 不符: " + str(path))
        part.replace(path)
        print("VERIFIED", model, path.name, info["Size"], flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(one, files))
    (destination / "download-manifest.json").write_text(json.dumps(files, ensure_ascii=False, indent=2))
    print("MODEL_READY", model, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="*", default=["Qwen3-VL-Embedding-2B"])
    parser.add_argument("--owner", default="Qwen")
    args = parser.parse_args()
    for name in args.models:
        fetch(name, args.owner)
