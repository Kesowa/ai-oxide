# Security Policy

## Reporting a vulnerability

Please report security issues privately rather than opening a public issue.

- Use GitHub's [private vulnerability reporting](https://github.com/Kesowa/ai-oxide/security/advisories/new)
  on this repository, or
- email **security@kesowa.com** with the details.

Include what you did, what happened, and what you expected, with enough detail
to reproduce. We will acknowledge your report and tell you whether we intend to
fix it, and when.

Please only test against infrastructure you own.

## Threat model

These are **internal worker processes**, not public services. They listen on no
port. A worker's only inputs are:

1. **JSON job messages** from a RabbitMQ queue, which is inside the trust
   boundary.
2. **Objects fetched from S3** — archives, rasters and video — which are
   ultimately **uploaded by users** and are therefore untrusted.

That second input is where the risk concentrates: a worker parses attacker-
influenced archives, images and video with native code. Findings there are the
most valuable.

## In scope

- **Zip-slip / path traversal** through archive entry names. `decompress`
  calls `sanitized_name()` and rejects entries without one; a way past that is
  a real finding.
- **Resource exhaustion from a crafted archive** — a zip bomb, a deeply nested
  archive, or an entry whose declared size disagrees with its actual size. The
  16 MiB per-entry cap is the current defence; a bypass is in scope.
- **Memory-safety or panic-based denial of service** in the GeoTIFF, image,
  video or ONNX paths, reachable from a crafted input file.
- **Unsafe deserialization.** An ONNX model file is untrusted input if it can
  come from outside; so is a malformed job message.
- **Writing outside the intended S3 prefix**, or any way to make a worker read
  or overwrite an object it should not.
- **Credential exposure** — a credential reaching a log line, an error message,
  or a response payload.
- **Committed secrets** anywhere in the repository or its git history.
- **Vulnerable dependencies** in `Cargo.toml` or `Cargo.lock`.
- **Supply chain**: `cog3pio` is pinned to a git revision rather than a
  published crate, so a problem with that dependency is relevant here.

## Out of scope

- The `minioadmin` / `minioadmin` values in `S3::default()` and
  `docker-compose.yaml`. These are MinIO's published defaults for the local
  development stack, not Kesowa credentials. We know they are there; the risk of
  a silent fallback is documented in the README and CONTRIBUTING. A concrete
  path by which a *production* deployment reaches those defaults would be in
  scope.
- The absence of authentication on worker processes, which have no listening
  socket to authenticate.
- Findings in RabbitMQ, MinIO, ONNX Runtime or GStreamer themselves — report
  those upstream, though a heads-up is welcome.
- `unwrap()` and `expect()` in startup configuration handling. Failing fast on
  invalid configuration is deliberate.

## Hardening notes for operators

- Give each worker an object-store credential scoped to the one bucket it needs,
  not a root key.
- Do not expose the RabbitMQ or MinIO ports from `docker-compose.yaml` beyond
  localhost. That compose file is a development convenience and is not a
  production configuration.
- Workers trust the queue. Anyone who can publish to `file.decompress.req` can
  direct a worker at any object key the worker's credential can reach, so
  restrict who can publish.
