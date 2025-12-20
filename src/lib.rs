use std::{sync::Arc, time::Duration};

use rc_zip_tokio::ReadZipStreaming;
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
pub enum ZipError {
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
pub async fn decompress(
    obj: impl utils::S3Object,
    dir: impl utils::S3Dir + 'static,
) -> Result<(), ZipError> {
    const CONCURRENCY: usize = 32;
    info!("decompressing...");
    let step_size = obj.len().await? / 100;
    let data = obj.reader().await?;
    let semaphore = Arc::new(Semaphore::new(CONCURRENCY));
    let dir = Arc::new(dir);
    let first_entry = data
        .stream_zip_entries_throwing_caution_to_the_wind()
        .await?;
    let mut entry = first_entry;
    let mut prev_i = 0;
    let mut total_read = 0;
    loop {
        total_read += entry.entry().compressed_size;
        let i = total_read / step_size;
        let dir = dir.clone();
        let semaphore = semaphore.clone();
        if i > prev_i {
            prev_i = i;
            info!(
                "Progress: {}% | Uploads: {}",
                i,
                CONCURRENCY - semaphore.available_permits()
            );
        }
        if entry.entry().uncompressed_size > 0xffffff {
            return Err(std::io::Error::new(
                std::io::ErrorKind::OutOfMemory,
                format!("entry size too big: {}", entry.entry().uncompressed_size),
            )
            .into());
        }
        let permit = semaphore.acquire_owned().await.map_err(|e| {
            error!("Failed to acquire permit, this shouldn't happen! {e:?}");
            ZipError::Impossible
        })?;
        let file_name = entry
            .entry()
            .sanitized_name()
            .ok_or(ZipError::InvalidRequest(format!(
                "Invalid file name for entry at: {total_read}"
            )))?
            .to_string();
        let mut file_data = Vec::with_capacity(entry.entry().uncompressed_size as usize);
        entry.read_to_end(&mut file_data).await?;
        tokio::spawn(async move {
            let mut errored = false;
            for i in 0..5 {
                match dir.create_obj(file_name.clone(), file_data.clone()).await {
                    Ok(_) => {
                        errored = false;
                        break;
                    }
                    Err(err) => {
                        error!("failed to upload file: {err:?}");
                        error!("retrying after a timeout, attempt {} out of {}", i + 1, 5);
                        tokio::time::sleep(Duration::from_secs(5)).await;
                        errored = true;
                    }
                }
            }
            if errored {
                error!("upload failed too many times, aborting...");
                permit.semaphore().close();
            }
            drop(permit);
        });
        match entry.finish().await? {
            Some(next) => entry = next,
            None => break,
        }
    }
    _ = semaphore
        .acquire_many_owned(CONCURRENCY as u32)
        .await
        .map_err(|e| {
            error!("Failed to acquire permit, this shouldn't happen! {e:?}");
            ZipError::Impossible
        })?;
    Ok(())
}

#[cfg(test)]
mod test {
    use std::sync::LazyLock;

    use crate::{
        decompress,
        telemetry::{get_subscriber, init_subscriber},
        utils::{
            Dir, Object, S3, create_bucket,
            mock::{MockDir, MockObject},
        },
    };

    static TRACING: LazyLock<()> = LazyLock::new(|| {
        let subscriber = get_subscriber("zip-decompress".into(), "info".into(), std::io::stdout);
        init_subscriber(subscriber);
    });

    #[tokio::test(flavor = "multi_thread")]
    async fn decompress_zip_fs() {
        LazyLock::force(&TRACING);
        let zip = MockObject::new("./test.zip");
        {
            tokio::fs::remove_dir_all("./test2").await.ok();
            tokio::fs::create_dir("./test2").await.ok();
        }
        let dir = MockDir::new("./test2");
        decompress(zip, dir).await.unwrap();
    }
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn decompress_zip_s3() {
        LazyLock::force(&TRACING);
        let s3 = S3::default();
        let bucket = create_bucket(s3).unwrap();
        let dir = Dir::new(*bucket.clone(), "test");
        let zip = Object::new(*bucket, "test.zip");
        decompress(zip, dir).await.unwrap();
    }
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn dont_decompress_zip_s3() {
        LazyLock::force(&TRACING);
        let s3 = S3::default();
        let bucket = create_bucket(s3).unwrap();
        let zip = Object::new(*bucket.clone(), "garbage.zip");
        let dir = Dir::new(*bucket, "garbage");
        let res = decompress(zip, dir).await;
        assert!(res.is_err());
    }
}
