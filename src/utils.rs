use s3::{creds::Credentials, error::S3Error, Bucket, Region};
use serde::Deserialize;
use tokio_stream::StreamExt;
use tokio_util::io::StreamReader;

#[derive(Deserialize, Clone, Debug)]
pub struct S3 {
    pub bucket: String,
    pub endpoint: String,
    pub path_style: bool,
    pub access: String,
    pub secret: String,
    pub region: String,
}

impl S3 {
    pub fn from_env() -> Self {
        Self {
            access: std::env::var("AWS_ACCESS_KEY_ID").expect("access key not found"),
            secret: std::env::var("AWS_SECRET_ACCESS_KEY").expect("secret not found"),
            bucket: std::env::var("S3_BUCKET").expect("bucket not found"),
            endpoint: std::env::var("S3_ENDPOINT").expect("endpoint not found"),
            path_style: std::env::var("S3_PATH_STYLE")
                .ok()
                .and_then(|p| p.parse().ok())
                .unwrap_or_default(),
            region: std::env::var("S3_REGION").unwrap_or("ap-south-1".into()),
        }
    }
}

impl Default for S3 {
    fn default() -> Self {
        Self {
            access: "minioadmin".into(),
            secret: "minioadmin".into(),
            bucket: "aru".into(),
            endpoint: "http://localhost:9000".into(),
            path_style: true,
            region: "ap-south-1".into(),
        }
    }
}

pub fn create_bucket(s3: S3) -> Result<Box<Bucket>, S3Error> {
    let region = Region::Custom {
        region: s3.region,
        endpoint: s3.endpoint,
    };
    let credentials = Credentials {
        access_key: Some(s3.access),
        secret_key: Some(s3.secret),
        security_token: None,
        session_token: None,
        expiration: None,
    };
    let mut bucket = Bucket::new(&s3.bucket, region, credentials)?;
    if s3.path_style {
        bucket = bucket.with_path_style();
    }
    Ok(bucket)
}
pub struct Object {
    bucket: Bucket,
    key: String,
}

impl Object {
    pub fn new(bucket: Bucket, key: impl ToString) -> Self {
        Self {
            bucket,
            key: key.to_string(),
        }
    }
}

pub struct Dir {
    bucket: Bucket,
    prefix: String,
}

impl Dir {
    pub fn new(bucket: Bucket, prefix: impl ToString) -> Self {
        Self {
            bucket,
            prefix: prefix.to_string(),
        }
    }
}

pub trait S3Object: Send + Sync {
    fn range(
        &self,
        start: u64,
        end: u64,
    ) -> impl std::future::Future<Output = Result<Vec<u8>, std::io::Error>> + Send;
    fn len(&self) -> impl std::future::Future<Output = Result<u64, std::io::Error>> + Send;
    fn reader(
        &self,
    ) -> impl std::future::Future<Output = Result<impl tokio::io::AsyncRead + Unpin, std::io::Error>>
           + Send;
}

pub trait S3Dir: Send + Sync {
    fn create_obj(
        &self,
        key: String,
        data: Vec<u8>,
    ) -> impl std::future::Future<Output = Result<impl S3Object, std::io::Error>> + Send;
}

impl S3Dir for Dir {
    async fn create_obj(
        &self,
        key: String,
        data: Vec<u8>,
    ) -> Result<impl S3Object, std::io::Error> {
        let key = format!("{}/{}", self.prefix, key);
        self.bucket
            .put_object(&key, data.as_ref())
            .await
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

        Ok(Object::new(self.bucket.clone(), key))
    }
}

impl S3Object for Object {
    async fn range(&self, start: u64, end: u64) -> Result<Vec<u8>, std::io::Error> {
        let data = self
            .bucket
            .get_object_range(&self.key, start, Some(end - 1))
            .await
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        Ok(data.to_vec())
    }

    async fn len(&self) -> Result<u64, std::io::Error> {
        let response_data = self
            .bucket
            .head_object(&self.key)
            .await
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

        let head_object = match response_data {
            (head_object, 200) => Ok(head_object),
            _ => Err(std::io::Error::new(std::io::ErrorKind::Other, "Woops")),
        }?;
        let object_len = head_object.content_length.unwrap_or(0) as u64;
        Ok(object_len)
    }

    async fn reader(&self) -> Result<impl tokio::io::AsyncRead + Unpin, std::io::Error> {
        let stream = self
            .bucket
            .get_object_stream(&self.key)
            .await
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

        let stream = stream.bytes.map(|chunk| match chunk {
            Ok(chunk) => Ok(chunk),
            Err(err) => Err(std::io::Error::new(std::io::ErrorKind::Other, err)),
        });
        let read = StreamReader::new(stream);
        Ok(read)
    }
}

#[cfg(test)]
pub mod mock {
    use std::{
        os::unix::fs::MetadataExt,
        path::Path,
        sync::atomic::{self, Ordering},
        time::Duration,
    };

    use tokio::io::{AsyncReadExt, AsyncSeekExt, BufReader};
    use tracing::instrument;

    use super::{S3Dir, S3Object};

    #[derive(Debug)]
    pub struct MockObject {
        path: String,
    }
    impl MockObject {
        pub fn new(path: impl ToString) -> Self {
            Self {
                path: path.to_string(),
            }
        }
    }
    impl S3Object for MockObject {
        #[instrument]
        async fn range(&self, start: u64, end: u64) -> Result<Vec<u8>, std::io::Error> {
            let mut file = tokio::fs::File::open(&self.path).await?;
            file.seek(std::io::SeekFrom::Start(start)).await?;
            let mut buf = vec![0; (end - start) as usize];
            file.read_exact(&mut buf).await?;
            Ok(buf)
        }

        #[instrument]
        async fn len(&self) -> Result<u64, std::io::Error> {
            let metadata = tokio::fs::metadata(&self.path).await?;
            Ok(metadata.size())
        }

        #[instrument]
        async fn reader(&self) -> Result<impl tokio::io::AsyncRead + Unpin, std::io::Error> {
            let file = tokio::fs::File::open(&self.path).await?;
            let read = BufReader::new(file);
            Ok(read)
        }
    }

    pub struct MockDir {
        path: String,
    }
    impl MockDir {
        pub fn new(path: impl ToString) -> Self {
            Self {
                path: path.to_string(),
            }
        }
    }

    static STEPS: atomic::AtomicU64 = atomic::AtomicU64::new(0);
    impl S3Dir for MockDir {
        async fn create_obj(
            &self,
            key: String,
            data: Vec<u8>,
        ) -> Result<impl S3Object, std::io::Error> {
            let step = STEPS.fetch_add(1, Ordering::Relaxed);
            tokio::time::sleep(Duration::from_millis(100)).await;
            if step % 13 == 0 {
                return Err(std::io::Error::new(std::io::ErrorKind::Other, "oops!"));
            }
            let path = format!("{}/{}", self.path, key.to_string());
            if let Some(dir) = Path::new(&path).parent() {
                tokio::fs::create_dir_all(dir).await?;
            }
            tokio::fs::write(&path, data).await?;
            Ok(MockObject::new(path))
        }
    }
}
