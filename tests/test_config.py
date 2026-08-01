from semtrace.config import compose_config


def test_config_composition_supports_protocol_and_data_root_overrides() -> None:
    config = compose_config(
        "stage3_detector",
        ["protocol=forensynths_progan4", "data.root=/datasets/forensynths"],
    )

    assert config.protocol.name == "forensynths_progan4"
    assert config.data.root == "/datasets/forensynths"
    assert config.model.name == "dinov3_vitb16"
    assert config.training.target_global_batch_size == 128
