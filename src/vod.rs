use crate::utils::S3;
use gst::{prelude::*, Element, ElementFactory, Pad, Pipeline};
use gstreamer::{self as gst, PadProbeReturn, PadProbeType};
use serde::Serialize;
use std::path::Path;
use thiserror::Error;
use tokio_stream::StreamExt;
use tracing::{debug, error, info, warn};

pub fn attach_tee<T: GstBinExt>(pipeline: &T, src_pad: &Pad) -> Result<(Pad, Pad), VodError> {
    let tee = ElementFactory::make("tee")
        .name("tee")
        .property("allow-not-linked", true)
        .build()?;
    pipeline.add(&tee)?;
    let sink = tee.static_pad("sink").ok_or(VodError::Impossible)?;
    src_pad.link(&sink).or(Err(VodError::Impossible))?;
    let pads = tee
        .request_pad_simple("src_%u")
        .zip(tee.request_pad_simple("src_%u"));
    tee.sync_state_with_parent()?;
    pads.ok_or(VodError::Impossible)
}

pub fn attach_video<T: GstBinExt>(pipeline: &T, src_pad: &Pad) -> Result<(), VodError> {
    let queue = ElementFactory::make("queue").build()?;
    let convert = ElementFactory::make("videoconvert").build()?;
    let scale = ElementFactory::make("videoscale").build()?;
    let scale_caps = gst::Caps::builder("video/x-raw")
        // .field("width", 1280)
        // .field("height", 720)
        .build();
    let capsfilter = ElementFactory::make("capsfilter").build()?;
    capsfilter.set_property("caps", scale_caps);
    let rate = ElementFactory::make("videorate").build()?;
    let enc = ElementFactory::make("x264enc")
        .property_from_str("speed-preset", "slow")
        .property_from_str("tune", "zerolatency")
        .property_from_str("psy-tune", "film")
        .build()?;
    let parse = ElementFactory::make("h264parse").build()?;
    let elements = &[&queue, &convert, &scale, &rate, &capsfilter, &enc, &parse];
    pipeline.add_many(elements)?;
    Element::link_many(elements)?;

    {
        let src_pad = parse.static_pad("src");
        let sink_pad = pipeline.by_name("hls_sink").and_then(|sink| {
            sink.pad_template("video")
                .and_then(|template| sink.request_pad(&template, None, None))
        });
        src_pad
            .zip(sink_pad)
            .and_then(|(src_pad, sink_pad)| src_pad.link(&sink_pad).ok())
            .ok_or(VodError::Impossible)?;
    }

    queue
        .static_pad("sink")
        .and_then(|sink_pad| src_pad.link(&sink_pad).ok())
        .ok_or(VodError::Impossible)?;

    for e in elements {
        e.sync_state_with_parent()?;
    }
    Ok(())
}

pub fn attach_thumb<T: GstBinExt + Send + Sync>(
    pipeline: &T,
    src_pad: Pad,
    sink: Element,
) -> Result<(), VodError> {
    let valve = ElementFactory::make("valve")
        .property("drop", false)
        .property_from_str("drop-mode", "forward-sticky-events")
        .build()?;
    let queue = ElementFactory::make("queue").name("thumb_queue").build()?;
    let convert = ElementFactory::make("videoconvert").build()?;
    let enc = ElementFactory::make("pngenc").name("thumb_enc").build()?;
    sink.set_property("name", "thumb_sink");
    let elements = &[&valve, &queue, &convert, &enc, &sink];
    pipeline.add_many(elements)?;
    Element::link_many(elements)?;
    valve
        .static_pad("sink")
        .and_then(|sink_pad| src_pad.link(&sink_pad).ok())
        .ok_or(VodError::Impossible)?;
    for e in elements {
        e.sync_state_with_parent()?;
    }
    let block_pad = sink.static_pad("sink").ok_or(VodError::Impossible)?;
    block_pad
        .add_probe(
            PadProbeType::BLOCK_DOWNSTREAM | PadProbeType::BUFFER,
            move |pad, probe_info| {
                valve.set_property("drop", true);
                probe_info.id.take().map(|id| pad.remove_probe(id));
                PadProbeReturn::Remove
            },
        )
        .ok_or(VodError::Impossible)?;
    Ok(())
}

pub fn attach_audio(pipeline: &Pipeline, src_pad: &Pad) -> Result<(), VodError> {
    let queue = ElementFactory::make("queue").name("audio_queue").build()?;
    let convert = ElementFactory::make("audioconvert").build()?;
    let resample = ElementFactory::make("audioresample").build()?;
    let enc = ElementFactory::make("avenc_aac").build()?;
    let elements = &[&queue, &convert, &resample, &enc];
    pipeline.add_many(elements)?;
    Element::link_many(elements)?;

    {
        let src_pad = enc.static_pad("src");
        let sink_pad = pipeline.by_name("hls_sink").and_then(|sink| {
            sink.pad_template("audio")
                .and_then(|template| sink.request_pad(&template, None, None))
        });
        src_pad
            .zip(sink_pad)
            .and_then(|(src_pad, sink_pad)| src_pad.link(&sink_pad).ok())
            .ok_or(VodError::Impossible)?;
    }

    queue
        .static_pad("sink")
        .and_then(|sink_pad| src_pad.link(&sink_pad).ok())
        .ok_or(VodError::Impossible)?;

    for e in elements {
        e.sync_state_with_parent()?;
    }
    Ok(())
}

pub fn attach_text<T: GstBinExt>(
    pipeline: &T,
    src_pad: &Pad,
    sink: Element,
) -> Result<(), VodError> {
    let queue = ElementFactory::make("queue").build()?;
    let convert = ElementFactory::make("srtenc").build()?;
    sink.set_property("name", "srt_sink");
    let elements = &[&queue, &convert, &sink];
    pipeline.add_many(elements)?;
    Element::link_many(elements)?;
    queue
        .static_pad("sink")
        .and_then(|sink_pad| src_pad.link(&sink_pad).ok())
        .ok_or(VodError::Impossible)?;
    for e in elements {
        e.sync_state_with_parent()?;
    }
    Ok(())
}

#[derive(Error, Debug)]
pub enum VodError {
    #[error("failed to build element")]
    Element(#[from] gst::glib::BoolError),

    #[error("gst broke")]
    Gst(#[from] gst::glib::Error),

    #[error("pipeline failed: {0}")]
    Pipeline(&'static str),

    #[error("impossible")]
    Impossible,
}

#[derive(Serialize, Default)]
struct Video {
    srt: Option<String>,
    hls: Option<String>,
    thumb: Option<String>,
}

async fn transcode_video(s3: S3, key: String) -> Result<Video, VodError> {
    gst::init()?;

    let path = Path::new(&key);
    let dir_path = path
        .parent()
        .zip(path.file_stem())
        .map(|(dir, name)| dir.join(name))
        .ok_or(VodError::Impossible)?;

    let hls_dir = dir_path.to_string_lossy().to_string();
    let srt_path = dir_path.join("subtitle.srt").to_string_lossy().to_string();
    let thumb_path = dir_path.join("thumb.png").to_string_lossy().to_string();
    let thumb_clone = thumb_path.clone();
    let srt_clone = srt_path.clone();

    let pipeline = gst::Pipeline::default();

    let src = ElementFactory::make("awss3src")
        .property("force-path-style", &s3.path_style)
        .property("access-key", &s3.access)
        .property("secret-access-key", &s3.secret)
        .property(
            "uri",
            format!("s3://{}/{}/{}", &s3.region, &s3.bucket, &key),
        )
        .build()?;
    let decode = ElementFactory::make("decodebin").build()?;

    let sink = ElementFactory::make("awss3hlssink")
        .name("hls_sink")
        .property("bucket", &s3.bucket)
        .property("force-path-style", &s3.path_style)
        .property("access-key", &s3.access)
        .property("secret-access-key", &s3.secret)
        .property("region", &s3.region)
        .property("key-prefix", &hls_dir)
        .build()?;
    let hlssink = sink
        .dynamic_cast_ref::<gst::ChildProxy>()
        .ok_or(VodError::Impossible)?;
    hlssink.set_child_property("hlssink::max-files", 0u32);
    hlssink.set_child_property("hlssink::playlist-length", 0u32);
    // hlssink.set_child_property("hlssink::target-duration", 10u32);
    hlssink.set_child_property("hlssink::location", "%05d.ts");
    hlssink.set_child_property_from_value(
        "hlssink::playlist-type",
        &gst::glib::Type::from_name("GstHlsSink3PlaylistType")
            .and_then(|gtype| gst::glib::Value::deserialize("vod", gtype).ok())
            .ok_or(VodError::Impossible)?,
    );
    hlssink.set_child_property("hlssink::playlist-location", "index.m3u8");

    pipeline.add_many(&[&src, &decode, &sink])?;
    src.link(&decode)?;

    let pipeline_clone = pipeline.downgrade();

    decode.connect_pad_added(move |_dbin, src_pad| {
        let Some(pipeline) = pipeline_clone.upgrade() else {
            warn!("pipeline upgrade failed");
            return;
        };
        let Some(cap_name) = src_pad
            .current_caps()
            .and_then(|caps| caps.structure(0).map(|s| s.name()))
        else {
            warn!("decode cap name not found");
            return;
        };

        match cap_name.as_str() {
            "video/x-raw" => {
                info!("got video stream!");
                if pipeline.by_name("tee").is_some() {
                    warn!("got another video stream, skipping!");
                    return;
                }
                let (video_src, thumb_src) = match attach_tee(&pipeline, src_pad) {
                    Ok(pads) => pads,
                    Err(err) => {
                        error!("failed to create tee: {err:?}");
                        return;
                    }
                };
                match s3_sink(&s3, &thumb_clone) {
                    Ok(sink) => match attach_thumb(&pipeline, thumb_src, sink) {
                        Ok(_) => {}
                        Err(err) => {
                            error!("failed to create thumbnail: {err:?}");
                        }
                    },
                    Err(err) => {
                        error!("failed to create thumb_sink: {err:?}");
                    }
                };
                match attach_video(&pipeline, &video_src) {
                    Ok(_) => {}
                    Err(err) => {
                        error!("failed to create video: {err:?}");
                    }
                };
            }
            "audio/x-raw" => {
                info!("got audio stream!");
                if pipeline.by_name("audio_queue").is_some() {
                    warn!("got another audio stream, skipping!");
                    return;
                }
                match attach_audio(&pipeline, src_pad) {
                    Ok(_) => {}
                    Err(err) => {
                        error!("failed to attach audio: {err:?}");
                    }
                };
            }
            "text/x-raw" => {
                info!("got subtitle stream!");
                match s3_sink(&s3, &srt_clone) {
                    Ok(sink) => match attach_text(&pipeline, src_pad, sink) {
                        Ok(_) => {}
                        Err(err) => {
                            error!("failed to create subtitle: {err:?}");
                        }
                    },
                    Err(err) => {
                        error!("failed to create srt_sink: {err:?}");
                    }
                };
            }
            unknown => {
                warn!("unknown caps found: {unknown}");
            }
        };
    });

    pipeline
        .set_state(gst::State::Playing)
        .map_err(|_| VodError::Pipeline("failed to start"))?;

    let mut is_srt = false;
    let mut is_hls = false;
    let mut is_thumb = false;

    let bus = pipeline.bus().ok_or(VodError::Pipeline("fetch bus"))?;
    let mut messages = bus.stream();
    while let Some(msg) = messages.next().await {
        match msg.view() {
            gst::MessageView::Eos(eos) => {
                is_hls = pipeline.by_name("hls_sink").is_some();
                is_srt = pipeline.by_name("srt_sink").is_some();
                is_thumb = pipeline.by_name("thumb_sink").is_some();
                info!("END OF STREAM!!! {:?}", eos.src().map(|s| s.path_string()));
                info!("HLS: {} SRT: {} THUMB: {}", is_hls, is_srt, is_thumb);
                break;
            }
            gst::MessageView::Error(err) => {
                pipeline
                    .set_state(gst::State::Null)
                    .map_err(|_| VodError::Pipeline("failed to stop"))?;
                error!("{:?}", err.src().map(|s| s.path_string()));
                error!("{:?}", err.error());
                error!("{:?}", err.debug());
            }
            gst::MessageView::StateChanged(s) => {
                debug!(
                    "State changed from {:?}: {:?} -> {:?} ({:?})",
                    s.src().map(|s| s.path_string()),
                    s.old(),
                    s.current(),
                    s.pending(),
                );
            }
            _ => (),
        }
    }
    pipeline
        .set_state(gst::State::Null)
        .map_err(|_| VodError::Pipeline("failed to stop"))?;
    Ok(Video {
        srt: is_srt.then_some(srt_path),
        hls: is_hls.then_some(hls_dir + "/index.m3u8"),
        thumb: is_thumb.then_some(thumb_path),
    })
}

fn s3_sink(s3: &S3, location: &str) -> Result<Element, VodError> {
    let sink = ElementFactory::make("awss3sink")
        .property("force-path-style", &s3.path_style)
        .property("access-key", &s3.access)
        .property("secret-access-key", &s3.secret)
        .property(
            "uri",
            format!("s3://{}/{}/{}", &s3.region, &s3.bucket, location),
        )
        .build()?;
    Ok(sink)
}
