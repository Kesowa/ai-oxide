use std::io::Cursor;

use bytes::Bytes;
use cog3pio::io::geotiff::read_geotiff;
use ndarray::Array3;
use object_store::path::Path;
use object_store::{parse_url, GetResult, ObjectStore};
use tokio;
use url::Url;

async fn example() {
    let cog_url: &str =
        "https://github.com/cogeotiff/rio-tiler/raw/6.4.0/tests/fixtures/cog_nodata_nan.tif";
    let tif_url: Url = Url::parse(cog_url).unwrap();
    let (store, location): (Box<dyn ObjectStore>, Path) = parse_url(&tif_url).unwrap();

    let stream: Cursor<Bytes> = {
        let result: GetResult = store.get(&location).await.unwrap();
        let bytes: Bytes = result.bytes().await.unwrap();
        Cursor::new(bytes)
    };

    // Read GeoTIFF into an ndarray::Array
    let arr: Array3<f32> = read_geotiff::<f32, _>(stream).unwrap();
    assert_eq!(arr.dim(), (1, 549, 549));
    assert_eq!(arr[[0, 500, 500]], 0.13482364);
}

#[tokio::test]
async fn test_example() {}
