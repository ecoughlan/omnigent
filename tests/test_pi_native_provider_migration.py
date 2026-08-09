"""Fork provider preservation must respect upstream inference bindings and scopes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from omnigent.errors import OmnigentError
from omnigent.harnesses.pi_native import credentials as creds


@pytest.fixture
def global_agent(tmp_path: Path) -> Path:
    root = tmp_path / "global"
    root.mkdir()
    (root / "models.json").write_text(
        json.dumps(
            {
                "providers": {
                    name: {
                        "baseUrl": "https://personal.example/v1",
                        "api": "openai-completions",
                        "apiKey": "test-personal",
                        "models": [{"id": "personal-model"}],
                    }
                    for name in ("personal", "omnigent-openai")
                }
            }
        )
    )
    (root / "auth.json").write_text('{"openai": {"type": "api_key", "key": "test"}}')
    (root / "settings.json").write_text('{"theme": "light"}')
    return root


@pytest.mark.parametrize("allowlisted", [False, True])
@pytest.mark.parametrize(
    ("family", "wire_api", "expected_id"),
    [
        ("anthropic", None, "omnigent"),
        ("openai", "responses", "omnigent-openai"),
        ("openai", "chat", "omnigent-completions"),
    ],
)
def test_bound_launch_keeps_reserved_identity_and_literal_model(
    tmp_path: Path,
    global_agent: Path,
    allowlisted: bool,
    family: str,
    wire_api: str | None,
    expected_id: str,
) -> None:
    model = "omnigent/private-model[large]"
    family_config = {
        "base_url": "https://bound.example/v1",
        "api_key": "test-bound",
        "models": {"default": model},
    }
    if wire_api is not None:
        family_config["wire_api"] = wire_api
    binding = {"provider": "company", "default_model": model}
    if allowlisted:
        binding["model_allowlist"] = [model]
    config = {
        "providers": {"company": {"kind": "gateway", family: family_config}},
        "inference": {"harnesses": {"pi-native": binding}},
    }
    provider = creds.resolve_pi_native_provider(config_loader=lambda: config)
    assert provider is not None
    # Upstream uses one primary reserved ID without an allowlist; allowlisted
    # bindings group models by wire protocol under separate reserved IDs.
    provider_id = expected_id if allowlisted else "omnigent"
    assert provider.provider_id == provider_id
    assert provider.model == model
    managed = tmp_path / "managed"
    env, args, _ = creds.pi_native_provider_launch(
        managed, provider, selection=model, global_agent_dir=global_agent
    )
    assert env["OMNIGENT_PI_INFERENCE_BOUND"] == "1"
    assert args[:4] == ["--provider", provider_id, "--model", f"{provider_id}/{model}"]
    rendered = json.loads((managed / "models.json").read_text())
    assert list(rendered["providers"]) == [provider_id]
    assert rendered["providers"][provider_id]["models"][0]["id"] == model
    assert not (managed / "auth.json").exists()
    settings = json.loads((managed / "settings.json").read_text())
    assert settings["theme"] == "light"
    if allowlisted:
        assert settings["enabledModels"] == [f"{provider_id}/{model}"]
        with pytest.raises(OmnigentError, match="not in the configured model list"):
            creds.resolve_pi_native_provider(model="personal-model", config_loader=lambda: config)


def test_legacy_curated_launch_preserves_globals_without_widening_scope(
    tmp_path: Path, global_agent: Path
) -> None:
    config = {
        "providers": {
            "company": {
                "kind": "gateway",
                "default": True,
                "openai": {
                    "base_url": "https://bound.example/v1",
                    "api_key": "test-bound",
                    "wire_api": "chat",
                    "models": {
                        "default": "fast",
                        "fast": "vendor/fast",
                        "strong": "vendor/strong",
                    },
                },
            }
        }
    }
    provider = creds.resolve_pi_native_provider(config_loader=lambda: config)
    assert provider is not None and provider.curated_models
    assert provider.provider_id == "company"
    managed = tmp_path / "managed"
    env, args, _ = creds.pi_native_provider_launch(
        managed,
        provider,
        selection="company/vendor/strong",
        global_agent_dir=global_agent,
    )
    assert "OMNIGENT_PI_INFERENCE_BOUND" not in env
    assert args[:4] == ["--provider", "company", "--model", "company/vendor/strong"]
    rendered = json.loads((managed / "models.json").read_text())
    assert set(rendered["providers"]) == {"company", "personal", "omnigent-openai"}
    settings = json.loads((managed / "settings.json").read_text())
    assert settings["enabledModels"] == ["company/vendor/fast", "company/vendor/strong"]
    assert (managed / "auth.json").read_bytes() == (global_agent / "auth.json").read_bytes()
    assert not (managed / "auth.json").is_symlink()
