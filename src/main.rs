use ai_oxide::{
    InferError, InferType, inference,
    telemetry::{get_subscriber, init_subscriber},
    utils::{Object, S3, create_bucket},
};
use lapin::{
    BasicProperties, Connection, ConnectionProperties,
    options::{
        BasicAckOptions, BasicConsumeOptions, BasicPublishOptions, BasicQosOptions,
        QueueDeclareOptions,
    },
    types::FieldTable,
};
use serde::{Deserialize, Serialize};
use tokio_stream::StreamExt;
use tracing::{error, info, instrument};
use uuid::Uuid;

const REQ_QUEUE: &str = "rooftop.infer.req";
const RES_QUEUE: &str = "rooftop.infer.res";
const TAG: &str = "";

#[tokio::main]
async fn main() {
    if std::env::var("RUST_LOG").is_err() {
        unsafe { std::env::set_var("RUST_LOG", "info") };
    }

    let subscriber = get_subscriber(
        "ai-infer".into(),
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
        let req = match serde_json::from_slice::<InferenceRequest>(&delivery.data) {
            Ok(r) => r,
            Err(err) => {
                error!("unable to decode message: {err:?}");
                continue;
            }
        };
        let s3 = s3.clone();
        info!("started zip decompression!");
        info!("{req:?}");
        let inference = run_inference(s3, req.file, req.infer).await;
        info!("zip decompressed successfully!");
        let res = InferenceResponse {
            metadata: req.metadata,
            success: inference.is_ok(),
            inference: inference.unwrap_or_default(),
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
struct InferenceRequest {
    file: String,
    infer: InferType,
    metadata: AruMetadata,
}

#[derive(Serialize)]
struct InferenceResponse {
    metadata: AruMetadata,
    inference: Inference,
    success: bool,
}

type AruMetadata = serde_json::Value;

#[derive(Serialize, Default)]
struct Inference {
    inference: Option<String>,
}

#[instrument(skip(s3))]
async fn run_inference(s3: S3, key: String, infer: InferType) -> Result<Inference, InferError> {
    let bucket = create_bucket(s3)?;
    let dir_key = key
        .strip_suffix(".tif")
        .ok_or(InferError::InvalidRequest("invalid file extension".into()))?;
    let in_file = Object::new(*bucket.clone(), &key);
    let out_file_key = format!("vector/{}.{}", Uuid::new_v4(), infer.output_ext());
    let out_file = Object::new(*bucket, &out_file_key);
    inference(in_file, out_file, infer).await?;
    Ok(Inference {
        inference: Some(dir_key.into()),
    })
}
