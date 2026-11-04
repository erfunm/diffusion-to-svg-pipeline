from kaggle_pkg.svg_sanitizer import MAX_BYTES, sanitize_svg, validate_svg

def test_disallowed_tags_and_attrs_removed():
    dirty = (
        '<svg xmlns="http://www.w3.org/2000/svg" onload="alert(1)">'
        '<style>path{fill:red}</style><script>alert(1)</script>'
        '<image href="http://example.com/x.png"/><foreignObject><div>x</div></foreignObject>'
        '<rect x="0" y="0" width="10" height="10"/><text>hello</text>'
        '<path d="M 0 0 L 10 0 L 10 10 Z" fill="#ff0000" style="x" onclick="y" href="z"/>'
        '</svg>'
    )
    out = sanitize_svg(dirty)
    for bad in ("style", "script", "image", "foreignObject", "rect", "text", "onload", "onclick", "href", "http://example"):
        assert bad not in out.replace('xmlns="http://www.w3.org/2000/svg"', "")
    assert '<path d="M 0 0 L 10 0 L 10 10 Z" fill="#ff0000"/>' in out
    assert validate_svg(out) == []

def test_size_cap_enforced_keeping_leading_paths():
    paths = "".join(f'<path d="M {i} 0 L {i} 500 L {i + 1} 500 Z" fill="#123456"/>' for i in range(500))
    raw = f'<svg xmlns="http://www.w3.org/2000/svg">{paths}</svg>'
    assert len(raw.encode()) > MAX_BYTES
    out = sanitize_svg(raw)
    assert len(out.encode()) <= MAX_BYTES
    assert '<path d="M 0 0 L 0 500 L 1 500 Z"' in out  # earliest (most important) paths survive
    assert validate_svg(out) == []

def test_validate_flags_violations():
    assert any("disallowed tag" in p for p in validate_svg('<svg xmlns="http://www.w3.org/2000/svg"><rect/></svg>'))
    assert any("disallowed attribute" in p for p in validate_svg('<svg xmlns="http://www.w3.org/2000/svg"><path d="M0 0" style="x"/></svg>'))
    big = '<svg xmlns="http://www.w3.org/2000/svg">' + '<path d="M 0 0"/>' * 1000 + "</svg>"
    assert any("size" in p for p in validate_svg(big))
    assert any("well-formed" in p for p in validate_svg("<svg"))

def test_heuristic_model_output_is_valid():
    from kaggle_pkg import Model
    for prompt in ["a red circle", "blue square and a star", "something else entirely"]:
        assert validate_svg(Model().predict(prompt)) == []
