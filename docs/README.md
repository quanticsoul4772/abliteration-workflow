# Bruno Documentation Organization

This directory contains planning documents, implementation records, and archived documentation.

## Active Documents

### Planning & Implementation
- **IMPLEMENTATION_PLAN.md** - Primary technical roadmap (testing, security, CI/CD)
- **NEXT_LEVEL_IMPROVEMENTS.md** - Enhancement proposals with realistic impact estimates
- **RUNPOD_32B_PLAN.md** - A100 80GB setup plan for 32B models
- **ARCHITECTURE.md** - System architecture overview
- **STREAMING_OPPORTUNITIES.md** - Dataset streaming analysis

## Archive

Historical documents moved to `archive/` for reference:
- **IMPROVEMENT_PLAN.md** - Superseded by NEXT_LEVEL_IMPROVEMENTS
- **INNOVATIVE_IMPROVEMENTS.md** - Superseded by NEXT_LEVEL_IMPROVEMENTS
- **DEPLOYMENT_GUIDE.md** - Consolidated into WORKFLOW.md
- **GPU_PCA_IMPLEMENTATION.md** - GPU PCA optimization completed (v1.0.1)
- **PCA_OPTIMIZATION_PLAN.md** - PCA optimization planning

## Organization Principles

**Root level** (`/`): User-facing documentation only
- README.md - Quick start and overview
- ROADMAP.md - Vision and future direction
- WORKFLOW.md - Cloud GPU comprehensive guide
- QUICK_REFERENCE.md - Command cheatsheet
- CLAUDE.md - AI assistant guide
- LESSONS_LEARNED.md - Troubleshooting and gotchas

**docs/** directory: Planning and implementation tracking
- Active planning documents
- Implementation records
- Archive of superseded documents

**configs/** directory: Example configuration files
- Production-ready TOML configs for different model sizes
- GPU-specific optimization templates

**examples/** directory: Example applications
- Chat interface (`chat_app.py`)
- Custom direction extraction examples

**claudedocs/** directory: Claude Code specific documentation
- Technical analyses (error handling, caching)
- Implementation notes (C4 streaming, layer-wise cache)
- Claude-generated reports

## Version History

| Version | Key Features |
|---------|-------------|
| v1.0.0 | Initial release with Optuna optimization |
| v1.0.1 | GPU-accelerated PCA (15-20x faster) |
| v1.1.0 | C4 dataset streaming (~50GB disk savings) |
| v1.2.0 | Layer-wise weight caching (55-75% memory reduction) |
