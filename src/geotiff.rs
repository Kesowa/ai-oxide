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
    for i in -1..rasterband.overview_count().unwrap() {
        let overview = if i == -1 {
            &rasterband
        } else {
            &rasterband.overview(i as usize).unwrap()
        };
        let o_size = overview.size();
        println!("overview {i}: {:?}, {}x", o_size, size.0 / o_size.0);
        let block_size = overview.block_size();
        let x_blocks = (o_size.0 + block_size.0 - 1) / block_size.0;
        let y_blocks = (o_size.1 + block_size.1 - 1) / block_size.1;
        let out_dir = format!("overview_{i}.png");
        std::fs::create_dir(&out_dir).unwrap();

        for x in 0..x_blocks {
            for y in 0..y_blocks {
                let data = overview.read_block::<u8>((x, y)).unwrap();
                image::save_buffer(
                    format!("{out_dir}/{x}.{y}.png"),
                    data.data(),
                    block_size.0.try_into().unwrap(),
                    block_size.1.try_into().unwrap(),
                    image::ColorType::L8,
                )
                .unwrap();
            }
        }
    }
}

#[test]
fn test_example() {
    let _ = example();
}
