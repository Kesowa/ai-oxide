FROM debian:trixie-slim AS build
RUN apt-get update && apt-get install --no-install-recommends -y ca-certificates libssl-dev curl git pkg-config cargo rustc g++-11
WORKDIR /build
COPY . .
RUN cargo build --release \
    && cp /build/target/release/ai_oxide /usr/bin/

FROM debian:trixie-slim
RUN apt-get update && apt-get install --no-install-recommends -y  ca-certificates
COPY --from=build --chmod=755 /usr/bin/ai_oxide /ai_oxide
ENTRYPOINT [ "/ai_oxide" ]
