# Configuration Examples

Example configuration files for different model sizes and use cases.

## Available Configurations

- **config.32b.toml** - Qwen2.5-Coder-32B-Instruct configuration
  - Requires: 80GB GPU, 200GB disk
  - C4 streaming enabled, 200 trials recommended
  - v1.2.0+: `cache_weights=true` now supported for 32B models

- **config-qwen32b.toml** - Alternative Qwen 32B configuration
  - Similar to config.32b.toml with different parameter settings

## Usage

Copy an example configuration to the root directory:

```bash
cp configs/config.32b.toml config.toml
```

Then customize as needed and run Bruno:

```bash
bruno --model Qwen/Qwen2.5-Coder-32B-Instruct
```

## Verify Configuration

Before running a long experiment, verify your configuration is loaded correctly:

```bash
# Show effective settings without running abliteration
bruno show-config --model Qwen/Qwen2.5-32B-Instruct
```

Bruno will also log at startup whether config.toml was found and warn if TOML values may not have been loaded.

## Configuration Priority

Settings cascade: **CLI > Environment > config.toml > Defaults**

**Important:** CLI default values (including `None`) override config.toml!

Best practice: Either pass ALL settings via CLI, OR use config.toml with no conflicting CLI args.

See `config.default.toml` for all available settings and documentation.
