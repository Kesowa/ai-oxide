use ai_oxide::{
    decompress,
    telemetry::{get_subscriber, init_subscriber},
    utils::{create_bucket, Dir, Object, S3},
    ZipError,
};
use lapin::{
    options::{
        BasicAckOptions, BasicConsumeOptions, BasicPublishOptions, BasicQosOptions,
        QueueDeclareOptions,
    },
    types::FieldTable,
    BasicProperties, Connection, ConnectionProperties,
};
use serde::{Deserialize, Serialize};
use tokio_stream::StreamExt;
use tracing::{error, info, instrument};

const REQ_QUEUE: &str = "file.decompress.req";
const RES_QUEUE: &str = "file.decompress.res";
const TAG: &str = "";

#[tokio::main]
async fn main() {
    if std::env::var("RUST_LOG").is_err() {
        unsafe { std::env::set_var("RUST_LOG", "info") };
    }

    let subscriber = get_subscriber(
        "zip-decompress".into(),
        std::env::var("RUST_LOG").unwrap(),
        std::io::stdout,
    );

    init_subscriber(subscriber);

    let addr = std::env::var("AMQP_ADDR").unwrap_or_else(|_| "amqp://172.17.0.1:5672/%2f".into());
    let s3 = S3::from_env();

    let options = ConnectionProperties::default();

    let connection = Connection::connect(&addr, options)
        .await
        .expect("rabbitmq connection failed");
    info!("CONNECTED");
    let req_channel = connection
        .create_channel()
        .await
        .expect("failed to create request channel");

    let req_queue = req_channel
        .queue_declare(
            REQ_QUEUE,
            QueueDeclareOptions {
                durable: true,
                ..Default::default()
            },
            FieldTable::default(),
        )
        .await
        .expect("failed to create request queue");
    info!(?req_queue, "Declared request queue");

    let res_channel = connection
        .create_channel()
        .await
        .expect("failed to create response channel");
    let res_queue = res_channel
        .queue_declare(
            RES_QUEUE,
            QueueDeclareOptions {
                durable: true,
                ..Default::default()
            },
            FieldTable::default(),
        )
        .await
        .expect("failed to create response queue");
    info!(?res_queue, "Declared response queue");

    req_channel
        .basic_qos(1, BasicQosOptions::default())
        .await
        .expect("failed to set qos");

    let mut consumer = req_channel
        .basic_consume(
            REQ_QUEUE,
            TAG,
            BasicConsumeOptions {
                no_ack: false,
                ..Default::default()
            },
            FieldTable::default(),
        )
        .await
        .expect("cannot create consumer");

    while let Some(delivery) = consumer.next().await {
        let delivery = match delivery {
            Ok(d) => d,
            Err(err) => {
                error!("error in delivery: {err:?}");
                continue;
            }
        };
        let req = match serde_json::from_slice::<TranscodeRequest>(&delivery.data) {
            Ok(r) => r,
            Err(err) => {
                error!("unable to decode message: {err:?}");
                continue;
            }
        };
        let s3 = s3.clone();
        info!("started zip decompression!");
        info!("{req:?}");
        let zip = decompress_zip(s3, req.file).await;
        info!("zip decompressed successfully!");
        let res = TranscodeResponse {
            metadata: req.metadata,
            success: zip.is_ok(),
            zip: zip.unwrap_or_default(),
        };
        let res = match serde_json::to_vec(&res) {
            Ok(r) => r,
            Err(err) => {
                error!("response serialization failed! {err:?}");
                continue;
            }
        };
        let pub_conf = res_channel
            .basic_publish(
                "",
                RES_QUEUE,
                BasicPublishOptions::default(),
                &res,
                BasicProperties::default()
                    .with_content_type("application/json".into())
                    .with_reply_to(REQ_QUEUE.into()),
            )
            .await;
        let conf = match pub_conf {
            Ok(c) => c.await,
            Err(err) => {
                error!("unable to publish message: {err:?}");
                continue;
            }
        };
        conf.ok();
        info!("response sent!");
        match delivery.ack(BasicAckOptions::default()).await {
            Ok(_) => {}
            Err(err) => error!("failed to ack message: {err:?}"),
        }
        info!("ack sent!");
    }
}

#[derive(Deserialize, Debug)]
struct TranscodeRequest {
    file: String,
    metadata: AruMetadata,
}

#[derive(Serialize)]
struct TranscodeResponse {
    metadata: AruMetadata,
    zip: Zip,
    success: bool,
}

type AruMetadata = serde_json::Value;

#[derive(Serialize, Default)]
struct Zip {
    zip: Option<String>,
}

#[instrument(skip(s3))]
async fn decompress_zip(s3: S3, key: String) -> Result<Zip, ZipError> {
    let bucket = create_bucket(s3)?;
    let dir_key = key
        .strip_suffix(".zip")
        .ok_or(ZipError::InvalidRequest("invalid file extension".into()))?;
    let dir = Dir::new(*bucket.clone(), dir_key);
    let zip = Object::new(*bucket, &key);
    decompress(zip, dir).await?;
    Ok(Zip {
        zip: Some(dir_key.into()),
    })
}
