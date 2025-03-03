FROM debian:trixie-slim AS build
RUN apt-get update && apt-get install --no-install-recommends -y ca-certificates libssl-dev curl git pkg-config cargo rustc 
WORKDIR /build
COPY . .
RUN cargo build --release \
    && cp /build/target/release/zip_decompress /usr/bin/

FROM debian:trixie-slim
RUN apt-get update && apt-get install --no-install-recommends -y  ca-certificates
COPY --from=build --chmod=755 /usr/bin/zip_decompress /zip_decompress
ENTRYPOINT [ "/zip_decompress" ]
