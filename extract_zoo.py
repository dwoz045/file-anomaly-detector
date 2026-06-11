import pathlib
import zipfile

binaries_dir = pathlib.Path("data/thezoo/malware/Binaries")
output_dir = pathlib.Path("data/test_env/malicious_large")
output_dir.mkdir(parents=True, exist_ok=True)

extracted = 0
failed = 0

for family_dir in sorted(binaries_dir.iterdir()):
    if not family_dir.is_dir():
        continue
    zips = list(family_dir.glob("*.zip"))
    if not zips:
        continue
    for zip_path in zips:
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                for member in zf.namelist():
                    family_output = output_dir / family_dir.name
                    family_output.mkdir(exist_ok=True)
                    zf.extract(member, path=family_output, pwd=b"infected")
                    extracted += 1
        except Exception:
            failed += 1
            continue

print(f"Extracted {extracted} samples into {output_dir}")
print(f"Failed: {failed}")

print("\nSamples per family:")
for d in sorted(output_dir.iterdir()):
    if d.is_dir():
        count = sum(1 for f in d.rglob("*") if f.is_file())
        print(f"  {d.name}: {count} file(s)")