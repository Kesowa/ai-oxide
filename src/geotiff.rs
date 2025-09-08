use gdal::{raster::RasterBand, Dataset, Metadata};
use ndarray::Axis;
use std::{collections::HashMap, path::Path};

pub fn example() {
    // let path = Path::new(
    //     "/vsicurl/https://cdn-dev.kesowa.com/aru/raster/d8b5d21b-9e00-4016-af65-6cebf3016329.tif",
    // );
    let path = Path::new("./ortho.tiff");
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
                r.read_block::<u8>(block)
                    .unwrap()
                    .to_array()
                    .unwrap()
                    .into_flat(),
                g.read_block::<u8>(block)
                    .unwrap()
                    .to_array()
                    .unwrap()
                    .into_flat(),
                b.read_block::<u8>(block)
                    .unwrap()
                    .to_array()
                    .unwrap()
                    .into_flat(),
                a.read_block::<u8>(block)
                    .unwrap()
                    .to_array()
                    .unwrap()
                    .into_flat(),
            );
            let img_arr = ndarray::stack![Axis(0), red, green, blue, alpha];
            let img_arr = img_arr.flatten_with_order(ndarray::Order::ColumnMajor);
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
