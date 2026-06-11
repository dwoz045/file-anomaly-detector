import pathlib
import shutil

clean_dir = pathlib.Path("data/test_env/clean_large")
clean_dir.mkdir(parents=True, exist_ok=True)

copied = 0
for f in pathlib.Path("/usr/bin").iterdir():
    if f.is_file() and not f.is_symlink():
        try:
            shutil.copy2(f, clean_dir / f.name)
            copied += 1
        except (PermissionError, OSError):
            continue

print(f"Copied {copied} clean files")