use gdal::{raster::RasterBand, Dataset, Metadata};
use ndarray::Axis;
use std::{collections::HashMap, path::Path};

/* Set the following ENV Vars
export GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR
export CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff"
export CPL_VSIL_CURL_CACHE_SIZE=200000000  # 200 MB
export VSI_CACHE=TRUE
export VSI_CACHE_SIZE=268435456  # ~256 MB
export GDAL_NUM_THREADS=ALL_CPUS
export GDAL_HTTP_MERGE_CONSECUTIVE_RANGES=YES
export GDAL_HTTP_MULTIPLEX=YES
export GDAL_CACHEMAX=512MB
export GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR
export GDAL_READDIR_LIMIT_ON_OPEN=100
*/
pub fn example() {
    let path = Path::new("/vsicurl/http://localhost:9000/aru/raster/Ortho_25cm.tif");
    // let path = Path::new("./ortho.tiff");
    let dataset = Dataset::open(path).unwrap();
    println!(
        "dataset metadata: {:#?}",
        dataset.metadata().collect::<Vec<_>>()
    );
    println!("dataset bands: {:?}", dataset.raster_count());
    println!("dataset size: {:?}", dataset.raster_size());
    println!("dataset projection: {:?}", dataset.projection());
    let band_map = HashMap::<String, RasterBand>::from_iter(
        dataset
            .rasterbands()
            .filter_map(|r| r.ok())
            .map(|r| (r.color_interpretation().name(), r)),
    );
    println!("{:?}", band_map.keys());
    let (r, g, b, a) = (
        band_map.get("Red").unwrap(),
        band_map.get("Green").unwrap(),
        band_map.get("Blue").unwrap(),
        band_map.get("Alpha").unwrap(),
    );
    let block_size = r.block_size();
    let raster_size = r.size();
    let (x_blocks, y_blocks) = (
        (raster_size.0 + block_size.0 - 1) / block_size.0,
        (raster_size.1 + block_size.1 - 1) / block_size.1,
    );
    for x in 0..x_blocks {
        for y in 0..y_blocks {
            let block = (x, y);
            let (red, green, blue, alpha) = (
                r.read_block::<u8>(block).unwrap().to_array().unwrap(),
                g.read_block::<u8>(block).unwrap().to_array().unwrap(),
                b.read_block::<u8>(block).unwrap().to_array().unwrap(),
                a.read_block::<u8>(block).unwrap().to_array().unwrap(),
            );
            let img_arr = ndarray::stack![Axis(2), red, green, blue, alpha];
            let img_arr = img_arr.flatten_with_order(ndarray::Order::RowMajor);
            let img = image::RgbaImage::from_raw(
                block_size.0 as u32,
                block_size.1 as u32,
                img_arr.as_slice().unwrap().to_vec(),
            )
            .unwrap();
            img.save(format!("output/{x}.{y}.png")).ok();
        }
    }
}

#[test]
fn test_example() {
    let _ = example();
}
