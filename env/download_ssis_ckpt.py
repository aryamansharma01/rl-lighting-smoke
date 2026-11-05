"""Download SSIS / SSISv2 checkpoints from the Google Drive folder linked in the SSIS README,
skipping the SOBA dataset zip. Places them where the SSIS configs expect them."""
import argparse
from pathlib import Path

import gdown

FOLDER = "https://drive.google.com/drive/folders/1MKxyq3R6AUeyLai9i9XWzG2C_n5f0ppP"
DEST = {
    "model_ssis_final.pth": "tools/output/SSIS_MS_R_101_bifpn_with_offset_class",
    "model_ssisv2_final.pth": "tools/output/SSISv2_MS_R_101_bifpn_with_offset_class_maskiouv2_da_bl",
}

ap = argparse.ArgumentParser()
ap.add_argument("--ssis", required=True, type=Path)
args = ap.parse_args()

files = gdown.download_folder(FOLDER, skip_download=True, quiet=True)
print("folder contents:", [f.path for f in files])
for f in files:
    name = Path(f.path).name
    if name in DEST:
        out = args.ssis / DEST[name] / name
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists() and out.stat().st_size > 1e6:
            print("exists", out); continue
        gdown.download(id=f.id, output=str(out), quiet=False)
        print("saved", out, out.stat().st_size / 1e6, "MB")
