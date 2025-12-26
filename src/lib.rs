use std::{sync::Arc, time::Duration};

use rc_zip_tokio::ReadZipStreaming;
use serde::Deserialize;
use thiserror::Error;
use tokio::{io::AsyncReadExt, sync::Semaphore};
use tracing::{error, info};

pub mod geotiff;
pub mod infer;
pub mod model;
pub mod telemetry;
pub mod utils;
pub mod vod;

#[derive(Error, Debug)]
pub enum InferError {
    #[error("s3 error")]
    S3(#[from] s3::error::S3Error),
    #[error("io error")]
    IO(#[from] std::io::Error),
    #[error("zip error")]
    Zip(#[from] rc_zip_tokio::rc_zip::error::Error),
    #[error("impossible")]
    Impossible,
    #[error("invalid request: {0}")]
    InvalidRequest(String),
}
pub async fn inference(
    in_file: impl utils::S3Object,
    out_file: impl utils::S3Object,
    infer: InferType,
) -> Result<(), InferError> {
    info!("running inference for {:?}...", infer);
    todo!();
}

#[derive(Clone, Copy, Deserialize, Debug)]
#[serde(rename_all = "kebab-case")]
pub enum InferType {
    Violence,
    Deepforest,
    Thermal,
    Rooftopseg,
    PeopleCount,
}

impl InferType {
    pub fn output_ext(&self) -> &'static str {
        match self {
            InferType::Violence => "csv",
            InferType::Deepforest => "geojson",
            InferType::Thermal => "geojson",
            InferType::Rooftopseg => "geojson",
            InferType::PeopleCount => "csv",
        }
    }
}

#[cfg(test)]
mod test {}
