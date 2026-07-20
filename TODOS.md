# Roadmap and TODOs

This is the working checklist for FramerAI. Items are grouped by area. Larger
items are tracked as GitHub issues; check them off here when the matching issue
is closed. New ideas are welcome through a feature request.

Legend: `[ ]` open, `[x]` done, `[~]` in progress.

## Testing and quality

- [ ] Add a `pytest` suite covering the tokenizer, transformer forward pass, and generation utilities.
- [ ] Add unit tests for the backend routes and WebSocket service (Jest or Vitest with supertest).
- [ ] Add component tests for the website chat flow (Vitest and Testing Library).
- [ ] Add end-to-end smoke tests that boot the backend and exercise the core endpoints.
- [ ] Add code coverage reporting to CI and publish a coverage badge.
- [ ] Introduce pre-commit hooks running ruff and lightweight JS checks.

## Model and training

- [ ] Publish reproducible training configs for tiny, small, medium, and large sizes.
- [ ] Add evaluation harness with standard benchmarks and a results table in the docs.
- [ ] Add checkpoint resume and safe interruption to `build.py` training.
- [ ] Support gradient checkpointing and mixed precision flags end to end.
- [ ] Add ONNX and safetensors export validation with a round-trip test.
- [ ] Document and validate multi-GPU and distributed training paths.

## Backend

- [ ] Add request validation and consistent error responses across all routes.
- [ ] Add rate limiting and payload size limits to generation endpoints.
- [ ] Add structured logging and a request id for traceability.
- [ ] Add a Dockerfile and a docker-compose file for local backend plus model serving.
- [ ] Add OpenAPI or a documented schema for the REST API.

## Frontend

- [ ] Add loading, empty, and error states to the chat and generation views.
- [ ] Add accessibility passes for keyboard navigation and screen readers.
- [ ] Add a settings panel for model size, temperature, and sampling controls.
- [ ] Add persistence of conversations to local storage.
- [ ] Add a production build and static hosting guide.

## Documentation

- [ ] Expand GUIDE.md with architecture diagrams for each module.
- [ ] Add a distillation tutorial that walks through a full run on a single GPU.
- [ ] Add a troubleshooting page for common CUDA, VRAM, and dependency issues.
- [ ] Add API reference documentation generated from source.

## DevOps and CI/CD

- [ ] Add a release workflow that tags versions and drafts release notes.
- [ ] Add container image publishing to a registry on release.
- [ ] Add caching for Python dependencies to speed up CI.
- [ ] Add a workflow that validates documentation links.

## Community and governance

- [ ] Enable GitHub Discussions and seed categories for questions and ideas.
- [ ] Add a curated list of good first issues for new contributors.
- [ ] Add a maintainers file describing review ownership and release duties.

## Done

- [x] Add continuous integration for model, backend, and website.
- [x] Add CodeQL analysis and stale issue automation.
- [x] Add Dependabot for pip, npm, and GitHub Actions.
- [x] Add issue and pull request templates and CODEOWNERS.
- [x] Add contributor guide, code of conduct, and security policy.
