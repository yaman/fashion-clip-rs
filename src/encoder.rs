
use crate::args::Args;
use std::io::Cursor;
use std::sync::Arc;

use ndarray::{Array2, Array4, ArrayD, CowArray, Dim};
use ort::session::Session;
use ort::{Environment, GraphOptimizationLevel, SessionBuilder, Value};
use tokenizers::tokenizer::Tokenizer;

use itertools::Itertools;
use tokenizers::{PaddingDirection, PaddingParams, PaddingStrategy};
extern crate image;
use image::io::Reader as ImageReader;

use image::imageops::FilterType;
use image::GenericImageView;
use ndarray::IxDyn;

extern crate num_cpus;


pub struct EncoderService {
    tokenizer: Tokenizer,
    encoder: Session,
    vision_size: usize,
}

impl EncoderService {
    pub fn new(
        environment: &Arc<Environment>,
        args: Args,
    ) -> Result<EncoderService, Box<dyn std::error::Error + Send + Sync>> {

        let model_path = "fashion-clip-onnx/model.onnx";
        let tokenizer_path = "fashion-clip-onnx/tokenizer.json";

        let mut tokenizer = Tokenizer::from_file(tokenizer_path)?;
        tokenizer.with_padding(Some(PaddingParams {
            strategy: if args.pad_token_sequence {
                PaddingStrategy::Fixed(77)
            } else {
                PaddingStrategy::BatchLongest
            },
            direction: PaddingDirection::Right,
            pad_to_multiple_of: None,
            pad_id: 0,
            pad_type_id: 0,
            pad_token: "[PAD]".to_string(),
        }));

        let num_cpus = num_cpus::get();
        let encoder = SessionBuilder::new(environment)?
            .with_parallel_execution(true)?
            .with_intra_threads(num_cpus as i16)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_model_from_file(model_path)?;

        Ok(EncoderService {
            tokenizer,
            encoder,
            vision_size: args.input_image_size,
        })
    }

    pub fn _process_text(
        &self,
        text: &Vec<String>,
    ) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error + Send + Sync>> {
        let preprocessed = self.tokenizer.encode_batch(text.clone(), true)?;

        let image_batch_size = 10;
        let num_channels = 3;
        let height = 224;
        let width = 224;

        let shape = IxDyn(&[image_batch_size, num_channels, height, width]);
        let pixel_values = ArrayD::<f32>::zeros(shape);
        let pixel_values = CowArray::from(pixel_values);

        let input_ids_vector: Vec<i64> = preprocessed
            .iter()
            .map(|i| i.get_ids().iter().map(|b| *b as i64).collect())
            .concat();

        let input_ids_vector = CowArray::from(Array2::from_shape_vec(
            (text.len(), input_ids_vector.len() / text.len()),
            input_ids_vector,
        )?)
        .into_dyn();

        let attention_mask_vector: Vec<i64> = preprocessed
            .iter()
            .map(|i| i.get_attention_mask().iter().map(|b| *b as i64).collect())
            .concat();

        let attention_mask_vector = CowArray::from(Array2::from_shape_vec(
            (text.len(), attention_mask_vector.len() / text.len()),
            attention_mask_vector,
        )?)
        .into_dyn();

        let session = &self.encoder;
        let outputs = session.run(vec![
            Value::from_array(session.allocator(), &input_ids_vector)?,
            Value::from_array(session.allocator(), &pixel_values)?,
            Value::from_array(session.allocator(), &attention_mask_vector)?,
        ])?;

        let output_text_embed_index = 2;
        let binding = outputs[output_text_embed_index].try_extract()?;
        let embeddings = binding.view();

        let seq_len = embeddings.shape().get(1).ok_or("not")?;

        Ok(embeddings
            .iter().copied()
            .chunks(*seq_len)
            .into_iter()
            .map(|b| b.collect())
            .collect())
    }

    pub fn _process_image(
        &self,
        images_bytes: &Vec<Vec<u8>>,
    ) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error + Send + Sync>> {
        let mean = [0.48145466, 0.4578275, 0.40821073]; // CLIP Dataset
        let std = [0.26862954, 0.261_302_6, 0.275_777_1];

        let mut pixels = CowArray::from(Array4::<f32>::zeros(Dim([
            images_bytes.len(),
            3,
            self.vision_size,
            self.vision_size,
        ])));
        for (index, image_bytes) in images_bytes.iter().enumerate() {
            let image = ImageReader::new(Cursor::new(image_bytes))
                .with_guessed_format()?
                .decode()?;
            let image = image.resize_exact(
                self.vision_size as u32,
                self.vision_size as u32,
                FilterType::CatmullRom,
            );
            for (x, y, pixel) in image.pixels() {
                pixels[[index, 0, x as usize, y as usize]] =
                    (pixel.0[0] as f32 / 255.0 - mean[0]) / std[0];
                pixels[[index, 1, x as usize, y as usize]] =
                    (pixel.0[1] as f32 / 255.0 - mean[1]) / std[1];
                pixels[[index, 2, x as usize, y as usize]] =
                    (pixel.0[2] as f32 / 255.0 - mean[2]) / std[2];
            }
        }

        let session = &self.encoder;
        let outputs = session.run(vec![Value::from_array(
            session.allocator(),
            &pixels.into_dyn(),
        )?])?;
        let binding = outputs[0].try_extract()?;
        let embeddings = binding.view();

        let seq_len = embeddings.shape().get(1).unwrap();

        Ok(embeddings
            .iter().copied()
            .chunks(*seq_len)
            .into_iter()
            .map(|b| b.collect())
            .collect())
    }
}

