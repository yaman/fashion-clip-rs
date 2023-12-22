use std::io::Cursor;

use image::io::Reader as ImageReader;
use image::{imageops::FilterType, ImageBuffer, Rgb};
use ndarray::{Array4, ArrayBase, CowArray, CowRepr, Dim};

#[allow(dead_code)]
pub struct CLIPImageProcessor {
    do_resize: bool,
    size: Option<(usize, usize)>, // Represents 'shortest_edge' or (height, width)
    resample: FilterType,         // Use image crate's FilterType for resampling
    do_center_crop: bool,
    crop_size: Option<(u32, u32)>, // (height, width)
    do_rescale: bool,
    rescale_factor: f32,
    do_normalize: bool,
    image_mean: Vec<f32>, // Length corresponds to the number of channels
    image_std: Vec<f32>,  // Length corresponds to the number of channels
    do_convert_rgb: bool,
}

#[allow(dead_code)]
impl CLIPImageProcessor {
    pub fn new() -> CLIPImageProcessor {
        CLIPImageProcessor {
            do_resize: true,
            size: Some((224, 224)), // Default size, can be changed based on shortest_edge logic
            resample: FilterType::CatmullRom, // Using CatmullRom as a default resampling filter
            do_center_crop: true,
            crop_size: Some((224, 224)), // Default crop size
            do_rescale: true,
            rescale_factor: 1.0 / 255.0, // Default rescale factor
            do_normalize: true,
            image_mean: vec![0.48145466, 0.4578275, 0.40821073], // Default image mean
            image_std: vec![0.26862954, 0.26130258, 0.27577711], // Default image std
            do_convert_rgb: true,
        }
    }

    pub fn resize(&self, image: &ImageBuffer<Rgb<u8>, Vec<u8>>) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
        if let Some((width, height)) = self.size {
            // Resize the image using the specified resampling filter
            image::imageops::resize(image, width as u32, height as u32, self.resample)
        } else {
            // If size is not specified, return the original image
            image.clone()
        }
    }

    pub fn center_crop(
        &self,
        image: &ImageBuffer<Rgb<u8>, Vec<u8>>,
    ) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
        if let Some((crop_width, crop_height)) = self.crop_size {
            let (width, height) = image.dimensions();

            // Calculate the cropping coordinates
            let left = (width.saturating_sub(crop_width)) / 2;
            let top = (height.saturating_sub(crop_height)) / 2;

            // Perform cropping
            image::imageops::crop_imm(image, left, top, crop_width, crop_height).to_image()
        } else {
            // If crop_size is None, return the original image
            image.clone()
        }
    }

    pub fn rescale(&self, image: &ImageBuffer<Rgb<u8>, Vec<u8>>) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
        if self.do_rescale {
            let mut scaled_image = image.clone();
            for (_, _, pixel) in scaled_image.enumerate_pixels_mut() {
                *pixel = image::Rgb([
                    (pixel[0] as f32 * self.rescale_factor).min(255.0) as u8,
                    (pixel[1] as f32 * self.rescale_factor).min(255.0) as u8,
                    (pixel[2] as f32 * self.rescale_factor).min(255.0) as u8,
                ]);
            }
            scaled_image
        } else {
            // If do_rescale is false, return the original image
            image.clone()
        }
    }

    pub fn normalize(&self,
        image: &ImageBuffer<Rgb<u8>, Vec<u8>>,
        pixels: &mut ArrayBase<CowRepr<'_, f32>, Dim<[usize; 4]>>,
        index: usize,
        mean: Vec<f32>,
        std: Vec<f32>,
    ) {
        for (x, y, pixel) in image.clone().enumerate_pixels_mut(){
            pixels[[index, 0, x as usize, y as usize]] =
                (pixel.0[0] as f32 / 255.0 - mean[0]) / std[0];
            pixels[[index, 1, x as usize, y as usize]] =
                (pixel.0[1] as f32 / 255.0 - mean[1]) / std[1];
            pixels[[index, 2, x as usize, y as usize]] =
                (pixel.0[2] as f32 / 255.0 - mean[2]) / std[2];
        }
    }

    pub fn preprocess(
        &self,
        images_bytes: Vec<u8>,
    ) -> ArrayBase<CowRepr<'_, f32>, Dim<[usize; 4]>> {
        let images_bytes = vec![images_bytes];
        let mut pixels: ArrayBase<CowRepr<'_, f32>, Dim<[usize; 4]>> =
            CowArray::from(Array4::<f32>::zeros(Dim([
                images_bytes.len(),
                3,
                self.size.unwrap().0,
                self.size.unwrap().1,
            ])));

        for (index, image_bytes) in images_bytes.iter().enumerate() {
            let image = ImageReader::new(Cursor::new(image_bytes))
                .with_guessed_format()
                .unwrap()
                .decode()
                .unwrap();
            // Step 1: Convert to RGB
            let mut processed_image = image.clone().into_rgb8();

            // Step 2: Resize (if enabled)
            if self.do_resize {
                processed_image = self.resize(&processed_image);
            }

            // Step 3: Center crop (if enabled)
            if self.do_center_crop {
                processed_image = self.center_crop(&processed_image);
            }

            // Step 4: Rescale (if enabled)
            if self.do_rescale {
                processed_image = self.rescale(&processed_image);
            }

            // Step 5: Normalize (if enabled)
            if self.do_normalize {
                self.normalize(
                    &processed_image,
                    &mut pixels,
                    index,
                    self.image_mean.clone(),
                    self.image_std.clone(),
                );
            }
        }

        pixels
    }
}

impl Default for CLIPImageProcessor {
    fn default() -> Self {
        Self::new()
    }
}
