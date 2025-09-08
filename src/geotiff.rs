use gdal::raster::RasterBand;
use gdal::{Dataset, Metadata};
use std::path::Path;

pub fn example() {
    // let path = Path::new(
    //     "/vsicurl/https://cdn-dev.kesowa.com/aru/raster/d8b5d21b-9e00-4016-af65-6cebf3016329.tif",
    // );
    let path = Path::new("./ortho.tiff");
    let dataset = Dataset::open(path).unwrap();
    println!("dataset description: {:?}", dataset.description());

    let rasterband: RasterBand = dataset.rasterband(1).unwrap();
    println!("rasterband description: {:?}", rasterband.description());
    println!(
        "rasterband color: {:?}",
        rasterband.color_interpretation().name()
    );
    let size = rasterband.size();
    println!("rasterband size: {:?}", size);
    println!("rasterband overviews: {:?}", rasterband.overview_count());
    for i in 0..rasterband.overview_count().unwrap() {
        let overview = rasterband.overview(i as usize).unwrap();
        let o_size = overview.size();
        let data = overview.read_block::<u8>((0, 0)).unwrap();
        let block_size = overview.block_size();
        image::save_buffer(
            format!("overview_{i}.png"),
            data.data(),
            block_size.0.try_into().unwrap(),
            block_size.1.try_into().unwrap(),
            image::ColorType::L8,
        )
        .unwrap();
        println!("overview {i}: {:?}, {}x", o_size, size.0 / o_size.0);
    }
}

#[test]
fn test_example() {
    let _ = example();
}
