#[cfg(test)]
mod tests {
    use embed_rs::clip_image_processor::CLIPImageProcessor;
    use image::{open, ImageBuffer, Rgb};

    fn get_image() -> ImageBuffer<Rgb<u8>, Vec<u8>> {
        let img_path = "tests/data/test_image.jpg";
        // Replace with the path to your image file
        let img = open(img_path).unwrap();
        let img_buffer: ImageBuffer<Rgb<u8>, _> = img.to_rgb8();
        img_buffer
    }

    #[test]
    fn test_resize_image() {
        let img_buffer = get_image();
        let processor = CLIPImageProcessor::default();

        let resized_img = processor.resize(&img_buffer);

        assert_eq!(resized_img.width(), 224);
        assert_eq!(resized_img.height(), 224);
    }

    #[test]
    fn test_center_crop() {
        let img_buffer = get_image();
        let processor = CLIPImageProcessor::default();

        let cropped_img = processor.center_crop(&img_buffer);

        assert_eq!(cropped_img.width(), 224);
        assert_eq!(cropped_img.height(), 224);
    }

    #[test]
    fn test_rescale_image() {
        let img = ImageBuffer::from_fn(2, 2, |x, y| Rgb([(x * 50 + y * 50) as u8, 0, 0]));
        let processor = CLIPImageProcessor::default();
        
        let rescaled_img = processor.rescale(&img);

        assert_eq!(rescaled_img[(0, 0)], Rgb([0, 0, 0])); // 0 / 255 = 0
        assert_eq!(rescaled_img[(1, 0)], Rgb([50 / 255, 0, 0])); // 50 / 255 ≈ 0.196, rounded to nearest u8
        assert_eq!(rescaled_img[(0, 1)], Rgb([50 / 255, 0, 0])); // 50 / 255 ≈ 0.196, rounded to nearest u8
        assert_eq!(rescaled_img[(1, 1)], Rgb([100 / 255, 0, 0])); // 100 / 255 ≈ 0.392, rounded to nearest u8
    }

//    #[test]
//     fn test_preprocess() {
//         // Create a test image
//         let img = ImageBuffer::from_fn(256, 256, |x, y| {
//             if (x as i32 - y as i32).abs() < 3 {
//                 Rgb([0, 0, 0])
//             } else {
//                 Rgb([255, 255, 255])
//             }
//         });

//         // Convert the image to bytes
//         let mut img_bytes: Vec<u8> = Vec::new();
//         img.write_to(&mut img_bytes, image::ImageOutputFormat::Png).unwrap();

//         // Create a ClipImageProcessor
//         let clip_image_processor = CLIPImageProcessor::default();

//         // Call the preprocess function
//         let result = clip_image_processor.preprocess(&vec![img_bytes]);

//         // Check the result
//         assert_eq!(result.dim(), [1, 3, 224, 224]);
//     }
    // #[test]
    // fn test_normalize_image_default_values() {
    //     let img = ImageBuffer::from_fn(2, 2, |_, _| Rgb([128, 128, 128]));
    //     let processor = CLIPImageProcessor::default();

    //     let normalized_img = processor.normalize(&img);

    //     assert_eq!(normalized_img[(0, 0)], Rgb([19, 43, 87]));
    // }
}
