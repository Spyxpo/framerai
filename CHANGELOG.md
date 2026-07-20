# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Continuous integration workflows for the model, backend, and website.
- CodeQL security analysis and a scheduled stale issue workflow.
- Dependabot configuration for pip, npm, and GitHub Actions updates.
- Issue and pull request templates, CODEOWNERS, and community health files.
- Contributor guide, code of conduct, security policy, project guide, and roadmap.

## [1.0.0] - 2026-02-16

### Added

- Initial public release of FramerAI.
- Transformer backbone with RoPE, SwiGLU, and RMSNorm for text and code.
- Vision encoder for image understanding.
- Diffusion module for text-to-image generation.
- Spatial-temporal video generator.
- Multimodal projector aligning vision and language spaces.
- Knowledge distillation pipeline using local open-source teacher models.
- Express backend with REST endpoints and WebSocket streaming.
- React web frontend for chat and generation.

[Unreleased]: https://github.com/Spyxpo/framerai/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/Spyxpo/framerai/releases/tag/v1.0.0
