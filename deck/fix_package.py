"""Strip content-type overrides for parts that were never written.

    .venv/bin/python deck/fix_package.py deck.pptx

pptxgenjs emits one `<Override PartName="/ppt/slideMasters/slideMasterN.xml">`
per slide while writing exactly one master. An override naming a non-existent
part violates the OPC spec (ECMA-376 Part 2 §10.1.2.3); PowerPoint's tolerance
for it is version-dependent, so the entries are removed rather than trusted.

Rewrites the archive in place, leaving every other part byte-identical.
"""

from __future__ import annotations

import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

OVERRIDE = re.compile(r'<Override\s+PartName="([^"]+)"[^>]*/>')


def fix(path: Path) -> int:
    with zipfile.ZipFile(path) as z:
        entries = [(i, z.read(i.filename)) for i in z.infolist()]
        names = {i.filename for i in z.infolist()}

    ct_name = "[Content_Types].xml"
    ct = next(data for info, data in entries if info.filename == ct_name).decode("utf8")

    removed = []

    def keep(match: re.Match) -> str:
        part = match.group(1).lstrip("/")
        if part in names:
            return match.group(0)
        removed.append(part)
        return ""

    fixed = OVERRIDE.sub(keep, ct)
    if not removed:
        print("no dangling overrides; package left untouched")
        return 0

    tmp = Path(tempfile.mkstemp(suffix=".pptx")[1])
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as out:
        for info, data in entries:
            out.writestr(info, fixed.encode("utf8") if info.filename == ct_name else data)
    shutil.move(str(tmp), str(path))

    print(f"removed {len(removed)} dangling content-type override(s):")
    for r in removed[:5]:
        print(f"  {r}")
    if len(removed) > 5:
        print(f"  ... and {len(removed) - 5} more")
    return 0


if __name__ == "__main__":
    sys.exit(fix(Path(sys.argv[1])))
