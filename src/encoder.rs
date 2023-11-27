// External crates
extern crate image;
extern crate num_cpus;

// Standard library
use std::io::Cursor;
use std::sync::Arc;

// Crate modules
use crate::args::Args;

// External libraries
use image::imageops::FilterType;
use image::io::Reader as ImageReader;
use image::GenericImageView;
use ndarray::{Array2, Array4, CowArray, Dim};
use ort::session::Session;
use ort::{Environment, GraphOptimizationLevel, SessionBuilder, Value};
use tokenizers::tokenizer::Tokenizer;

// Other
use itertools::Itertools;

pub struct EncoderService {
    tokenizer: Tokenizer,
    encoder: Session,
    vision_size: usize,
}

impl EncoderService {
    pub fn new(
        environment: &Arc<Environment>,
        args: Args,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let model_path = "models/text/model.onnx";
        let tokenizer_path = "sentence-transformers/clip-ViT-B-32-multilingual-v1";

        let mut tokenizer = Tokenizer::from_pretrained(tokenizer_path, None)?;

        tokenizer.with_padding(Some(tokenizers::PaddingParams {
            strategy: tokenizers::PaddingStrategy::BatchLongest,
            direction: tokenizers::PaddingDirection::Right,
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
    ) -> Result<Vec<f32>, Box<dyn std::error::Error + Send + Sync>> {
        let preprocessed = self.tokenizer.encode(text.clone(), true)?;

        let input_ids_vector: Vec<i64> = preprocessed
            .get_ids()
            .iter()
            .map(|b| *b as i64)
            .collect::<Vec<i64>>();

        let ids_shape = (text.len(), input_ids_vector.len() / text.len());
        let input_ids_vector =
            CowArray::from(Array2::from_shape_vec(ids_shape, input_ids_vector)?).into_dyn();

        let attention_mask_vector: Vec<i64> = preprocessed
            .get_attention_mask()
            .iter()
            .map(|b| *b as i64)
            .collect::<Vec<i64>>();

        let mask_shape = (text.len(), attention_mask_vector.len() / text.len());
        let attention_mask_vector =
            CowArray::from(Array2::from_shape_vec(mask_shape, attention_mask_vector)?).into_dyn();

        let session = &self.encoder;
        let outputs = session.run(vec![
            Value::from_array(session.allocator(), &input_ids_vector)?,
            Value::from_array(session.allocator(), &attention_mask_vector)?,
        ])?;

        let output_text_embed_index = 0;
        let binding = outputs[output_text_embed_index].try_extract()?;
        let embeddings = binding.view();

        let seq_len = embeddings
            .shape()
            .first()
            .ok_or("cannot find seq_len with index 0 in text embeddings")?;


        let embeddings: Vec<f32> = embeddings
            .iter()
            .copied()
            .chunks(*seq_len)
            .into_iter()
            .flatten()
            .collect();
        Ok(embeddings)
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

        let seq_len = embeddings
            .shape()
            .get(1)
            .ok_or("cannot find seq_len with index 1 for image embeddings")?;

        Ok(embeddings
            .iter()
            .copied()
            .chunks(*seq_len)
            .into_iter()
            .map(|b| b.collect())
            .collect())
    }
}
