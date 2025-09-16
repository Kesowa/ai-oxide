use gdal::{raster::RasterBand, Dataset, Metadata};
use ndarray::{s, Array3, Axis};
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
export AWS_S3_ENDPOINT=127.0.0.1:9000
export AWS_HTTPS=NO
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin
export AWS_VIRTUAL_HOSTING=FALSE
*/
pub fn example() {
    // let path = Path::new("/vsis3/aru/raster/Ortho_25cm.tif");
    let path = Path::new("./clippedv2.tif");
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
    let bands = [
        band_map.get("Red").unwrap(),
        band_map.get("Green").unwrap(),
        band_map.get("Blue").unwrap(),
        band_map.get("Alpha").unwrap(),
    ];
    // Final block size that will be generated for inferencing
    let block_size = (512, 512);
    // Inverse of the step size ratio. 1 = no overlap. 2 = 50% overlap. 3 = 67% overlap. etc.
    let overlap = 1;
    let raster_size = bands[0].size();
    for x in (0..raster_size.0).step_by(block_size.0 / overlap) {
        for y in (0..raster_size.1).step_by(block_size.1 / overlap) {
            // Boundary condition
            // If the raster size is not a multiple of the block size, it will not be able to fill an entire block at the extents (the rightmost and bottommost edges).
            // In that case, read the incomplete window and overlay it onto the proper block sized channel, snapped to the top-left corner
            let window_size = (
                block_size.0.min(raster_size.0 - x),
                block_size.1.min(raster_size.1 - y),
            );
            let buffer_size = block_size;
            let offset = (x as isize, y as isize);

            let mut img_arr: Array3<u8> = Array3::zeros((block_size.0, block_size.1, bands.len()));

            for (index, band) in bands.iter().enumerate() {
                let window = band
                    .read_as::<u8>(offset, window_size, window_size, None)
                    .unwrap()
                    .to_array()
                    .unwrap();
                let mut channel = img_arr.index_axis_mut(Axis(2), index);
                // Crop the channel to align with the window size at the top-left corner
                let mut view = channel.slice_mut(s![0..window_size.1, 0..window_size.0]);
                view += &window;
            }

            // Convert 3D array into pixel-interleaved linear array
            let img_arr = img_arr.flatten_with_order(ndarray::Order::RowMajor);
            let img = image::RgbaImage::from_raw(
                buffer_size.0 as u32,
                buffer_size.1 as u32,
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
