use std::collections::HashMap;
use std::path::Path;

use gdal::Dataset;
use gdal::raster::RasterBand;
use gdal::spatial_ref::CoordTransform;
use gdal::spatial_ref::SpatialRef;
use ndarray::Array3;
use ndarray::Axis;
use ndarray::s;
use thiserror::Error;

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

/// Convert pixel (x, y) from a dataset into geocoordinates (lon, lat).
pub fn pixel_to_geocoord(
    transform: [f64; 6],
    spatial_ref: SpatialRef,
    pixel_x: f64,
    pixel_y: f64,
) -> gdal::errors::Result<(f64, f64)> {
    // Apply affine transformation (convert pixel -> projected coordinate system)
    let geo_x = transform[0] + pixel_x * transform[1] + pixel_y * transform[2];
    let geo_y = transform[3] + pixel_x * transform[4] + pixel_y * transform[5];

    // Now, reproject into WGS84 (lat/lon) if needed
    let wgs84 = SpatialRef::from_epsg(4326)?; // WGS84
    let coord_transform = CoordTransform::new(&spatial_ref, &wgs84)?;

    let mut x = [geo_x];
    let mut y = [geo_y];
    let mut z = [];

    coord_transform.transform_coords(&mut x, &mut y, &mut z)?;

    Ok((x[0], y[0])) // lon, lat
}

pub struct Ortho {
    dataset: Dataset,
    block_size: (usize, usize),
    overlap: usize,
}

impl Ortho {
    const CHANNELS: [&'static str; 3] = ["Red", "Green", "Blue"];
    pub fn new(path: &Path, block_size: (u32, u32), overlap: u32) -> Result<Self, GeoError> {
        let dataset = Dataset::open(path)?;
        let band_map = HashMap::<String, RasterBand>::from_iter(
            dataset
                .rasterbands()
                .filter_map(|r| r.ok())
                .map(|r| (r.color_interpretation().name(), r)),
        );
        for channel in Self::CHANNELS {
            if band_map.get(channel).is_none() {
                return Err(GeoError::Band(channel));
            }
        }
        Ok(Self {
            dataset,
            block_size: (block_size.0 as usize, block_size.1 as usize),
            overlap: overlap as usize,
        })
    }

    pub fn iter(&self) -> impl Iterator<Item = Result<((usize, usize), Array3<u8>), GeoError>> {
        let mut band_map = HashMap::<String, RasterBand>::from_iter(
            self.dataset
                .rasterbands()
                .filter_map(|r| r.ok())
                .map(|r| (r.color_interpretation().name(), r)),
        );
        let bands = Self::CHANNELS.map(|ch| band_map.remove(ch).unwrap());
        let block_size = self.block_size;
        let overlap = self.overlap;
        let raster_size = bands[0].size();
        let my_iter = (0..raster_size.0)
            .step_by(block_size.0 / overlap)
            .flat_map(move |x| {
                (0..raster_size.1)
                    .step_by(block_size.1 / overlap)
                    .map(move |y| (x, y))
            });
        let my_iter =
            my_iter.map(move |(x, y)| img_from_bands(x, y, &bands, block_size, raster_size));
        my_iter
    }
}

fn img_from_bands(
    x: usize,
    y: usize,
    bands: &[RasterBand<'_>; 3],
    block_size: (usize, usize),
    raster_size: (usize, usize),
) -> Result<((usize, usize), Array3<u8>), GeoError> {
    let window_size = (
        block_size.0.min(raster_size.0 - x),
        block_size.1.min(raster_size.1 - y),
    );
    let offset = (x as isize, y as isize);

    let mut img_arr: Array3<u8> = Array3::zeros((bands.len(), block_size.0, block_size.1));

    for (index, band) in bands.iter().enumerate() {
        let window = band
            .read_as::<u8>(offset, window_size, window_size, None)?
            .to_array()?;
        let mut channel = img_arr.index_axis_mut(Axis(0), index);
        // Crop the channel to align with the window size at the top-left corner
        let mut view = channel.slice_mut(s![0..window_size.1, 0..window_size.0]);
        view += &window;
    }
    Ok(((x, y), img_arr))
}

#[derive(Error, Debug)]
pub enum GeoError {
    #[error("gdal error")]
    GDAL(#[from] gdal::errors::GdalError),
    #[error("band not found in tif: {0}")]
    Band(&'static str),
}

#[cfg(test)]
mod test {
    use std::path::Path;

    use crate::{geotiff::Ortho, model::utils::ndarray3_to_image};
    #[test]
    fn test_geotiff_ortho() {
        let path = Path::new("./ortho.tiff");
        let ortho = Ortho::new(path, (512, 512), 1).unwrap();
        for block in ortho.iter() {
            let ((x, y), img_arr) = block.unwrap();

            let img = ndarray3_to_image(img_arr);
            img.save(format!("output/{x}.{y}.png")).ok();
        }
    }
}
