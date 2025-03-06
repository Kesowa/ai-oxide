FROM debian:trixie-slim AS build
RUN apt-get update && apt-get install --no-install-recommends -y ca-certificates libssl-dev curl git pkg-config cargo rustc g++-11 libglib2.0-dev libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev

WORKDIR /build
RUN git clone https://gitlab.freedesktop.org/gstreamer/gst-plugins-rs.git && cd gst-plugins-rs && git checkout main

WORKDIR /build/gst-plugins-rs
RUN cargo build --release -p gst-plugin-hlssink3 -p gst-plugin-aws --lib \
    && cp /build/gst-plugins-rs/target/release/*.so /usr/lib/x86_64-linux-gnu/gstreamer-1.0/ 

WORKDIR /build/ai_oxide
COPY . .
RUN cargo build --release \
    && cp ./target/release/ai_oxide /usr/bin/

FROM debian:trixie-slim
RUN apt-get update && apt-get install --no-install-recommends -y gstreamer1.0-plugins-bad gstreamer1.0-plugins-base gstreamer1.0-plugins-ugly gstreamer1.0-libav ca-certificates
COPY --from=build /usr/lib/x86_64-linux-gnu/gstreamer-1.0/*.so /usr/lib/x86_64-linux-gnu/gstreamer-1.0/ 
COPY --from=build --chmod=755 /usr/bin/ai_oxide /ai_oxide
ENTRYPOINT [ "/ai_oxide" ]
