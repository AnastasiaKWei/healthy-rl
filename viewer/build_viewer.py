"""Inject exported transcript JSON into the viewer template."""
import sys
from pathlib import Path

tpl, data, out = (Path(p) for p in sys.argv[1:4])
html = tpl.read_text()
blob = data.read_text()
# `</script>` inside the payload would close the tag early. JSON treats \/ as /,
# so escaping the slash is lossless.
blob = blob.replace("</", "<\\/")
assert "__DATA__" in html, "template has no __DATA__ placeholder"
out.write_text(html.replace("__DATA__", blob))
print(f"{out} -> {out.stat().st_size/1e6:.2f} MB")
