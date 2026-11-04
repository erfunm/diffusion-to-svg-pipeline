import re
import xml.etree.ElementTree as ET
from html import escape

# A conservative allowlist aligned with competition spirit (paths only).
ALLOWED_TAGS = {"svg", "path", "g"}
# Allowed attributes for <svg> and <path>
ALLOWED_ATTRS = {
    "svg": {"xmlns","width","height","viewBox","version"},
    "path": {"d","fill","stroke","stroke-width","fill-rule","opacity"}
}

MAX_BYTES = 10_000

def sanitize_svg(svg: str, width: int = 512, height: int = 512) -> str:
    """Very conservative sanitizer: keeps only <svg>,<g>,<path> and whitelisted attributes.
    Strips styles, scripts, hrefs, and external refs. Also enforces size <= 10KB.
    """
    # Normalize whitespace
    svg = re.sub(r"\s+", " ", svg).strip()

    # Remove styles, scripts, images, foreignObject, and comments
    svg = re.sub(r"<!--.*?-->", "", svg)
    svg = re.sub(r"<\s*(style|script|image|foreignObject)[^>]*>.*?<\s*/\s*\1\s*>", "", svg, flags=re.I)

    # Extract paths using a simple regex, discard other tags
    paths = re.findall(r"<\s*path\b[^>]*>", svg, flags=re.I)
    path_elems = []
    for p in paths:
        attrs = {}
        for attr, val in re.findall(r"(\w[\w-]*)\s*=\s*\"([^\"]*)\"", p):
            tag = "path"
            if attr in ALLOWED_ATTRS[tag]:
                attrs[attr] = val
        if "d" in attrs:
            # Basic clean of d
            d = attrs["d"]
            d = re.sub(r"[^MmLlHhVvCcSsQqTtAaZz0-9,\.\-\s]", "", d)
            attrs["d"] = d
            # Defaults
            if "fill" not in attrs:
                attrs["fill"] = "#000000"
            # Rebuild element
            attr_str = " ".join(f"{k}=\"{escape(v)}\"" for k,v in attrs.items())
            path_elems.append(f"<path {attr_str}/>")

    # Build root svg
    view_box = f"0 0 {width} {height}"
    body = "".join(path_elems)
    out = f"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\" viewBox=\"{view_box}\">{body}</svg>"

    # Enforce max bytes
    if len(out.encode("utf-8")) > MAX_BYTES:
        # Truncate paths until size fits
        keep = []
        for pe in path_elems:
            trial = f"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\" viewBox=\"{view_box}\">{''.join(keep+[pe])}</svg>"
            if len(trial.encode("utf-8")) <= MAX_BYTES:
                keep.append(pe)
            else:
                break
        out = f"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\" viewBox=\"{view_box}\">{''.join(keep)}</svg>"
    return out

SVG_NS = "{http://www.w3.org/2000/svg}"

def validate_svg(svg: str) -> list[str]:
    """Independent check of a final SVG against the constraints. Returns a list of
    violations (empty list == valid): size <= MAX_BYTES, parseable XML, svg root,
    and only ALLOWED_TAGS / ALLOWED_ATTRS. Does not modify anything.
    """
    problems = []
    size = len(svg.encode("utf-8"))
    if size > MAX_BYTES:
        problems.append(f"size {size} > {MAX_BYTES} bytes")
    try:
        root = ET.fromstring(svg)
    except ET.ParseError as e:
        return problems + [f"not well-formed XML: {e}"]
    for el in root.iter():
        if not el.tag.startswith(SVG_NS):
            problems.append(f"element outside SVG namespace: {el.tag}")
            continue
        tag = el.tag[len(SVG_NS):]
        if tag not in ALLOWED_TAGS:
            problems.append(f"disallowed tag <{tag}>")
            continue
        allowed = ALLOWED_ATTRS.get(tag, set())
        for attr in el.attrib:
            if attr not in allowed:
                problems.append(f"disallowed attribute {attr!r} on <{tag}>")
    if root.tag != SVG_NS + "svg":
        problems.append(f"root element is {root.tag}, expected svg")
    return problems
