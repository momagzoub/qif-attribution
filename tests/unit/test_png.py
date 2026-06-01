from qif_attribution.generation.png import PNG_SIGNATURE, write_solid_rgb_png


def test_write_solid_rgb_png_writes_valid_signature(tmp_path) -> None:
    output = tmp_path / "solid.png"

    write_solid_rgb_png(output, width=3, height=2, rgb=(1, 2, 3))

    assert output.read_bytes().startswith(PNG_SIGNATURE)
