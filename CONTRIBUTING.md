# Contributing

## Branches

Branch from **`dev`**, not `master`. `master` is behind and does not contain the
model code. Open pull requests against `dev` unless you are deliberately
patching the older tree.

## Getting set up

```bash
cp .env.example .env
docker compose up -d minio createbuckets rabbitmq   # dependencies only
set -a && source .env && set +a
cargo run --bin ai_oxide
```

You need GStreamer development libraries on the host for `src/vod.rs` to build:

```bash
# macOS
brew install gstreamer gst-plugins-base gst-plugins-good

# Debian / Ubuntu
sudo apt install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
```

ONNX Runtime comes in through `ort` and does not need separate installation for
the CPU execution provider.

## Before you push

```bash
cargo fmt --all
cargo clippy --all-targets -- -D warnings
cargo build --release
cargo test
```

There is very little test coverage, so `cargo build --release` passing is not
much of a guarantee. Say in your pull request what you actually exercised — a
real archive, a real raster, a real video.

## Style

- Standard `rustfmt`. No hand-formatting.
- Errors are `thiserror` enums, propagated with `?`. Do not add `unwrap()` or
  `expect()` to a request path; they are acceptable only at startup, where
  failing fast on bad configuration is the intent.
- Instrument new async entry points with `#[instrument]` and keep logs
  structured — fields, not interpolated strings.
- **Stream, do not buffer.** This is the central constraint of the codebase.
  Anything that reads a whole object into memory needs an explicit size bound,
  as `decompress` has at 16 MiB per entry.
- Bound your concurrency with a semaphore rather than spawning per item.

## Configuration

Read configuration from the environment, via a `from_env()` constructor that
panics on a missing required variable. Document every new variable in
`.env.example` with a placeholder.

Be careful with `Default` implementations that carry credentials. `S3::default()`
falls back to `minioadmin` / `minioadmin` for the compose stack's benefit. That
is convenient locally and a hazard anywhere else, because a misconfigured
deployment silently gets a well-known credential instead of an error. Prefer
`from_env()` on any path that could run in production, and do not add new
credential-bearing defaults.

## Queue conventions

- Queues are **durable**; messages survive a broker restart.
- Consume with manual ack and ack only after the response is published, so a
  crash mid-job redelivers rather than loses the work.
- Prefetch stays at 1. Scale by adding replicas, not by increasing prefetch.
- A response echoes the request's `metadata` untouched, so the caller can
  correlate, and carries an explicit `success` flag rather than relying on
  absence.

## Things worth fixing

- **Almost no tests.** `resize_padded`, the zip entry-name sanitisation and the
  GeoTIFF coordinate maths are all pure enough to unit test and worth covering
  first.
- **`master` is stale.** The branch divergence should be resolved rather than
  left to grow; `dev` has been ahead for months.
- **`src/main.rs` hardcodes the decompression queue names** as consts, so the
  binary does one job. Several workers in one crate would want these
  parameterised.
- **The telemetry service name is `"zip-decompress"`** regardless of what the
  binary does, which is misleading in aggregated logs.
- **`infer.rs` returns `Box<dyn Error>`** rather than a typed error like the
  rest of the crate.
- **The CPU execution provider is hardcoded.** GPU inference would need the
  provider list to be configurable.

## Dependency licensing

This repository is MIT. Most dependencies are MIT or Apache-2.0 and compatible.

Two to keep in mind:

- **GStreamer is LGPL**, and individual plugins vary — some are GPL, and some
  codecs carry patent obligations in some jurisdictions. Dynamic linking keeps
  LGPL satisfiable, but the plugin set you ship determines your actual
  obligations. Audit it before distributing binaries.
- **`cog3pio` is pinned to a git revision**, not a crates.io release. That makes
  the build depend on a third-party GitHub repository staying available. Check
  its licence before relying on it further, and consider vendoring.

Do not add GPL or AGPL crates.

## Secrets

Never commit credentials. Configuration goes in the environment; `.env` is
git-ignored. CI secrets belong in GitHub Actions secrets and are referenced as
`${{ secrets.NAME }}` — the existing workflow does this correctly.

If you commit a secret by accident, treat it as compromised and rotate it.
Removing it in a later commit does not remove it from history.

## License

Contributions are accepted under the [MIT License](LICENSE) covering this
repository.
