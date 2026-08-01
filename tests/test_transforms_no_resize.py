from pathlib import Path

import pytest
import torch
import yaml
from PIL import Image

from semtrace.data.transforms import ProtocolTransform, SmallImageError


def test_main_protocol_does_not_call_resize(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden_resize(*args: object, **kwargs: object) -> None:
        raise AssertionError("resize must not be used by the main protocol")

    monkeypatch.setattr(Image.Image, "resize", forbidden_resize)
    transform = ProtocolTransform(crop_size=128, training=False, small_image_policy="skip")

    output = transform(Image.new("RGB", (160, 144), "white"))

    assert output.shape == (3, 128, 128)
    assert output.dtype == torch.float32


def test_main_yaml_explicitly_disables_resize() -> None:
    config_path = Path(__file__).parents[1] / "configs" / "protocol" / "genimage_sdv14.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["preprocessing"]["allow_resize"] is False


def test_small_image_skip_is_explicit() -> None:
    transform = ProtocolTransform(crop_size=128, training=False, small_image_policy="skip")

    with pytest.raises(SmallImageError):
        transform(Image.new("RGB", (127, 128)))


def test_reflect_padding_is_opt_in() -> None:
    transform = ProtocolTransform(crop_size=128, training=False, small_image_policy="reflect")

    output = transform(Image.new("RGB", (120, 124), "white"))

    assert output.shape == (3, 128, 128)
