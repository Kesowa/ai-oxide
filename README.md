# ai-oxide

Rust worker processes for the ARU drone-data platform. Each one consumes jobs
from a RabbitMQ queue, streams data in and out of S3-compatible object storage,
and publishes a result back onto a response queue. The pieces here cover archive
decompression, GeoTIFF handling, video transcoding and ONNX model inference.

The design goal throughout is to stream rather than buffer: drone deliverables
are large, so a worker should process a multi-gigabyte archive or raster without
holding it in memory.

> **Branches.** The default branch `master` is behind. Active development
> happens on **`dev`**, currently 34 commits ahead with the model code
> (`src/model/`) that `master` does not have. Read `dev` for the current state.
> Other live branches: `bboxes`, `f32_mobilenet`, `fix/bounding-box-accuracy`,
> `gdal`.

## Modules

| Module | Purpose |
| --- | --- |
| `src/lib.rs` | streaming zip decompression with a bounded concurrency pool |
| `src/main.rs` | the decompression worker: RabbitMQ consume → S3 → publish |
| `src/infer.rs` | ONNX Runtime inference via [`ort`](https://ort.pyke.io/), with letterbox resize |
| `src/geotiff.rs` | GeoTIFF reading, built on `cog3pio` |
| `src/vod.rs` | video-on-demand transcoding through GStreamer |
| `src/utils.rs` | S3 bucket, object and directory abstractions |
| `src/telemetry.rs` | structured JSON logging via `tracing` and bunyan |
| `src/model/` | **`dev` only** — RetinaNet and MobileNet heads, CLAHE preprocessing |

## How a worker runs

The decompression worker in `src/main.rs` is the reference shape:

1. Declare a durable request queue (`file.decompress.req`) and a durable
   response queue (`file.decompress.res`).
2. Set prefetch to 1, so a worker takes one job at a time and more replicas
   means more throughput.
3. Consume with manual acknowledgement. A job is acked only after its response
   is published, so a crash mid-job returns the work to the queue.
4. Stream the archive out of S3, expanding entries and uploading each one back
   under a sibling prefix, up to 32 concurrent uploads governed by a semaphore.
5. Publish a JSON response carrying the original metadata and a success flag.

Entries larger than 16 MiB uncompressed are rejected rather than buffered, and
entry names are sanitised before use to prevent a zip-slip path escape.

## Requirements

- Rust 2021 edition, stable toolchain
- GStreamer development libraries, for `src/vod.rs`
- ONNX Runtime — supplied by `ort`; CPU execution provider by default
- A RabbitMQ instance and an S3-compatible object store

## Running it

A `docker-compose.yaml` brings up MinIO, the bucket bootstrap, RabbitMQ and the
worker together. That is the quickest path:

```bash
cp .env.example .env     # then edit for your setup
docker compose up
```

MinIO's console is on port 9001 and RabbitMQ's management UI on 15672, both with
the compose file's development credentials.

To run the worker directly against your own infrastructure:

```bash
cp .env.example .env
set -a && source .env && set +a
cargo run --release --bin ai_oxide
```

## Configuration

All configuration is environment variables — see `.env.example`.

| Variable | Required | Purpose |
| --- | --- | --- |
| `AWS_ACCESS_KEY_ID` | yes | object store access key |
| `AWS_SECRET_ACCESS_KEY` | yes | object store secret key |
| `S3_BUCKET` | yes | bucket to read from and write to |
| `S3_ENDPOINT` | yes | object store endpoint URL |
| `S3_PATH_STYLE` | no | `true` for MinIO, `false` for AWS S3. Defaults to `false` |
| `S3_REGION` | no | defaults to `ap-south-1` |
| `AMQP_ADDR` | no | defaults to `amqp://172.17.0.1:5672/%2f` |
| `RUST_LOG` | no | tracing filter. Defaults to `info` |

`S3::from_env()` panics if any required variable is missing — the worker fails
at startup rather than running misconfigured.

### A note on the credential defaults

`S3::default()` falls back to `minioadmin` / `minioadmin` against
`http://localhost:9000`. Those are MinIO's documented defaults, present so the
compose stack works out of the box — they are not Kesowa credentials and there
is nothing to rotate. But a silent fallback to a well-known credential is worth
being deliberate about: production paths should use `S3::from_env()`, which
fails loudly instead. See [CONTRIBUTING.md](CONTRIBUTING.md#configuration).

## Building the image

`.github/workflows/docker-publish.yml` builds and publishes a container image on
push. The Dockerfile is multi-stage: it compiles against the GStreamer and ONNX
Runtime development packages, then copies the binary into a slim runtime layer.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Security reports go through
[SECURITY.md](SECURITY.md), not public issues.

## License

Released under the [MIT License](LICENSE). © Kesowa Infinite Ventures.

Note that GStreamer is **LGPL**, and some GStreamer plugins are GPL or carry
patent obligations. Linking dynamically, as this crate does, keeps LGPL
obligations satisfiable, but check the licence of any plugin set you ship. See
[CONTRIBUTING.md](CONTRIBUTING.md#dependency-licensing).
