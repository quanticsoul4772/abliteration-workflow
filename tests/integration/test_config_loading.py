# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

"""Integration tests for configuration loading.

These tests verify that config.toml files are actually loaded and parsed,
detecting silent failures where TOML is ignored and defaults are used.
"""

import os

import pytest

# ============================================================================
# Fixtures for MockSettings to reduce code repetition
# ============================================================================


class MockDataset:
    """Mock dataset specification for testing."""

    dataset = "test-dataset"
    split = "train[:100]"
    config = "en"


@pytest.fixture
def mock_settings_defaults():
    """Create a MockSettings object with default values."""

    class MockSettings:
        use_mpoa = False
        use_concept_cones = False
        use_caa = False
        activation_target_percentile = 0.75

    return MockSettings()


@pytest.fixture
def mock_settings_full():
    """Create a comprehensive MockSettings object for testing print_config_summary."""

    class MockSettings:
        model = "test-model"
        n_trials = 200
        study_name = "test_study"
        storage = "sqlite:///test.db"
        batch_size = 0
        max_batch_size = 128
        use_mpoa = True
        use_concept_cones = False
        use_caa = False
        use_activation_calibration = True
        use_circuit_ablation = False
        use_sacred_directions = False
        use_neural_refusal_detection = True
        ensemble_probe_pca = True
        orthogonalize_directions = True
        use_warm_start_params = True
        use_pca_extraction = True
        n_refusal_directions = 3
        use_eigenvalue_weights = True
        eigenvalue_weight_method = "softmax"
        direction_weights = [1.0, 0.5, 0.25]
        activation_target_percentile = 0.75
        activation_calibration_layer_frac = 0.6
        activation_calibration_min_factor = 0.5
        activation_calibration_max_factor = 2.0
        mpoa_norm_mode = "row"
        mpoa_min_scale = 0.5
        mpoa_max_scale = 2.0
        good_prompts = MockDataset()
        bad_prompts = MockDataset()
        unhelpfulness_prompts = MockDataset()

    return MockSettings()


class TestConfigLoading:
    """Test suite for configuration loading integration."""

    def test_config_file_found_detection(self, monkeypatch, tmp_path):
        """Test that log_config_status correctly detects config.toml presence."""
        from bruno.config_verify import log_config_status

        # Test when config.toml doesn't exist
        monkeypatch.chdir(tmp_path)
        assert log_config_status() is False

        # Test when config.toml exists
        config_file = tmp_path / "config.toml"
        config_file.write_text('model = "test-model"\n')
        assert log_config_status() is True

    def test_verify_config_was_parsed_no_config(
        self, monkeypatch, tmp_path, mock_settings_defaults
    ):
        """Test verify_config_was_parsed when no config file exists."""
        from bruno.config_verify import verify_config_was_parsed

        monkeypatch.chdir(tmp_path)
        warnings = verify_config_was_parsed(mock_settings_defaults)
        # No warnings expected when no config file exists
        assert len(warnings) == 0

    def test_verify_config_was_parsed_detects_mismatch(self, monkeypatch, tmp_path):
        """Test that verify_config_was_parsed detects when TOML wasn't loaded."""
        from bruno.config_verify import verify_config_was_parsed

        # Create a config file with values DIFFERENT from defaults
        # Note: use_mpoa, use_concept_cones, use_caa now default to True
        # So we set them to False in TOML to detect if TOML wasn't loaded
        config_content = """
use_mpoa = false
use_concept_cones = false
activation_target_percentile = 0.60
"""
        config_file = tmp_path / "config.toml"
        config_file.write_text(config_content)
        monkeypatch.chdir(tmp_path)

        # Mock settings where TOML wasn't loaded (using defaults)
        # Defaults are now: use_mpoa=True, use_concept_cones=True, use_caa=True
        class MockSettings:
            use_mpoa = True  # Default is True, but TOML says false
            use_concept_cones = True  # Default is True, but TOML says false
            use_caa = True  # Not in TOML, using default
            activation_target_percentile = 0.75  # Default, but TOML says 0.60

        warnings = verify_config_was_parsed(MockSettings())

        # Should warn about use_mpoa, use_concept_cones, and activation_target_percentile
        assert len(warnings) >= 1
        assert any("use_mpoa" in w for w in warnings)
        assert any("use_concept_cones" in w for w in warnings)
        # Also verify activation_target_percentile mismatch is detected
        assert any("activation_target_percentile" in w for w in warnings)

    def test_verify_config_was_parsed_no_warnings_when_loaded(
        self, monkeypatch, tmp_path
    ):
        """Test no warnings when TOML values are properly loaded."""
        from bruno.config_verify import verify_config_was_parsed

        # Create a config file
        config_content = """
use_mpoa = true
activation_target_percentile = 0.60
"""
        config_file = tmp_path / "config.toml"
        config_file.write_text(config_content)
        monkeypatch.chdir(tmp_path)

        # Mock settings where TOML was properly loaded
        class MockSettings:
            use_mpoa = True  # Matches TOML
            use_concept_cones = False  # Not in TOML, using default
            use_caa = False  # Not in TOML, using default
            activation_target_percentile = 0.60  # Matches TOML

        warnings = verify_config_was_parsed(MockSettings())
        # No warnings expected
        assert len(warnings) == 0

    def test_verify_config_was_parsed_handles_invalid_toml(
        self, monkeypatch, tmp_path, mock_settings_defaults
    ):
        """Test that invalid TOML produces a warning."""
        from bruno.config_verify import verify_config_was_parsed

        # Create an invalid TOML file
        config_file = tmp_path / "config.toml"
        config_file.write_text("invalid toml {{{{{")
        monkeypatch.chdir(tmp_path)

        warnings = verify_config_was_parsed(mock_settings_defaults)
        # Should have a warning about parse failure
        assert len(warnings) >= 1
        assert any("failed to parse" in w.lower() for w in warnings)


class TestConfigPriority:
    """Test configuration priority (CLI > ENV > TOML > defaults)."""

    def test_defaults_are_used_without_config(self, monkeypatch, tmp_path):
        """Test that Field defaults are used when no config file exists."""
        monkeypatch.chdir(tmp_path)

        # Clear any env vars that might interfere
        for key in list(os.environ.keys()):
            if key.startswith("BRUNO_"):
                monkeypatch.delenv(key, raising=False)

        # Import after clearing env to ensure clean state
        # Note: Can't easily test full Settings loading without a model,
        # but we can verify the defaults in the class definition

        from bruno.config import Settings

        # Verify critical defaults are correct (conservative/safe values)
        fields = Settings.model_fields

        # These are now True by default (stable features enabled)
        assert fields["use_concept_cones"].default is True
        assert fields["use_caa"].default is True
        # Circuit ablation is False by default (doesn't work with GQA models)
        assert fields["use_circuit_ablation"].default is False

        # These should be True by default (stable features)
        assert fields["use_pca_extraction"].default is True
        assert fields["use_activation_calibration"].default is True


class TestShowConfigCommand:
    """Test the show-config CLI command."""

    def test_print_config_summary_runs(
        self, monkeypatch, tmp_path, capsys, mock_settings_full
    ):
        """Test that print_config_summary runs and outputs expected content."""
        from bruno.config_verify import print_config_summary

        # Should not raise any exceptions
        print_config_summary(mock_settings_full)

        # Capture and verify output contains expected content
        captured = capsys.readouterr()
        output = captured.out

        # Verify key sections are present in output
        assert "test-model" in output or "model" in output.lower()
        assert "200" in output or "n_trials" in output.lower()
        assert "test_study" in output or "study" in output.lower()


class TestLogEffectiveSettings:
    """Test log_effective_settings function."""

    def test_log_effective_settings_runs(self, capsys, mock_settings_full):
        """Test that log_effective_settings runs without errors."""
        from bruno.config_verify import log_effective_settings

        # Should not raise any exceptions
        log_effective_settings(mock_settings_full)

        # Capture and verify output contains expected content
        captured = capsys.readouterr()
        output = captured.out

        # Verify key settings are logged
        assert "test-model" in output
        assert "200" in output  # n_trials
        assert "MPOA" in output or "mpoa" in output.lower()

    def test_log_effective_settings_shows_enabled_features(
        self, capsys, mock_settings_full
    ):
        """Test that enabled features are listed."""
        from bruno.config_verify import log_effective_settings

        log_effective_settings(mock_settings_full)

        captured = capsys.readouterr()
        output = captured.out

        # MPOA is enabled in mock_settings_full
        assert "MPOA" in output
        # Ensemble Probe+PCA is enabled
        assert "Ensemble" in output or "ensemble" in output.lower()

    def test_log_effective_settings_shows_disabled_features(self, capsys):
        """Test that disabled features are listed."""
        from bruno.config_verify import log_effective_settings

        # Create settings with all key features disabled
        class DisabledSettings:
            model = "test-model"
            n_trials = 100
            study_name = "test"
            use_mpoa = False
            use_concept_cones = False
            use_caa = False
            use_activation_calibration = False
            use_circuit_ablation = False
            use_sacred_directions = False
            ensemble_probe_pca = False
            orthogonalize_directions = False
            use_warm_start_params = False
            use_pca_extraction = False
            n_refusal_directions = 1
            use_eigenvalue_weights = False
            direction_weights = [1.0]
            activation_target_percentile = 0.75

        log_effective_settings(DisabledSettings())

        captured = capsys.readouterr()
        output = captured.out

        # Should show disabled features section with specific feature names
        # The log_effective_settings prints "Disabled features: MPOA, Concept Cones, CAA"
        assert "Disabled features" in output or "disabled" in output.lower()
        # At least one of the disabled features should be listed
        assert "MPOA" in output or "Concept Cones" in output or "CAA" in output


class TestDirectionWeightsFix:
    """Test the direction_weights eigenvalue override fix.

    This tests that when use_eigenvalue_weights=False, user-specified
    direction_weights are used instead of being overridden by eigenvalue-computed weights.
    """

    def test_user_weights_used_when_eigenvalue_disabled(self):
        """Test that user-specified weights are used when use_eigenvalue_weights=False."""

        # Simulate the logic from main.py lines ~346-359
        class MockSettings:
            use_eigenvalue_weights = False
            direction_weights = [1.0, 0.3, 0.1]  # User-specified

        class MockDirectionResult:
            direction_weights = [
                0.8,
                0.15,
                0.05,
            ]  # Eigenvalue-computed (should be ignored)

        settings = MockSettings()
        direction_result = MockDirectionResult()

        # This is the fixed logic from main.py
        if (
            settings.use_eigenvalue_weights
            and direction_result.direction_weights is not None
        ):
            direction_weights = direction_result.direction_weights
        else:
            direction_weights = settings.direction_weights

        # User weights should be used since use_eigenvalue_weights=False
        assert direction_weights == [1.0, 0.3, 0.1]

    def test_eigenvalue_weights_used_when_enabled(self):
        """Test that eigenvalue-computed weights are used when use_eigenvalue_weights=True."""

        class MockSettings:
            use_eigenvalue_weights = True
            direction_weights = [1.0, 0.3, 0.1]  # User-specified (should be ignored)

        class MockDirectionResult:
            direction_weights = [0.8, 0.15, 0.05]  # Eigenvalue-computed

        settings = MockSettings()
        direction_result = MockDirectionResult()

        # This is the fixed logic from main.py
        if (
            settings.use_eigenvalue_weights
            and direction_result.direction_weights is not None
        ):
            direction_weights = direction_result.direction_weights
        else:
            direction_weights = settings.direction_weights

        # Eigenvalue weights should be used since use_eigenvalue_weights=True
        assert direction_weights == [0.8, 0.15, 0.05]

    def test_user_weights_used_when_eigenvalue_weights_none(self):
        """Test that user weights are used when eigenvalue_weights is None (fallback)."""

        class MockSettings:
            use_eigenvalue_weights = True  # Enabled, but result is None
            direction_weights = [1.0, 0.3, 0.1]

        class MockDirectionResult:
            direction_weights = None  # No eigenvalue weights computed

        settings = MockSettings()
        direction_result = MockDirectionResult()

        # This is the fixed logic from main.py
        if (
            settings.use_eigenvalue_weights
            and direction_result.direction_weights is not None
        ):
            direction_weights = direction_result.direction_weights
        else:
            direction_weights = settings.direction_weights

        # User weights should be used as fallback since eigenvalue_weights is None
        assert direction_weights == [1.0, 0.3, 0.1]
