use std::io::Cursor;
use std::sync::Arc;
use tonic::{Request, Response, Status};
pub mod encoder {
    tonic::include_proto!("encoder");
}
use encoder::encoder_server::Encoder;
use encoder::{Embedding, EncodeTextRequest, EncoderResponse};

use ndarray::{Array2, Array4, CowArray, Dim};
use ort::session::Session;
use ort::{Environment, GraphOptimizationLevel, SessionBuilder, Value};
use tokenizers::tokenizer::Tokenizer;

use itertools::Itertools;
use tokenizers::{PaddingDirection, PaddingParams, PaddingStrategy};
extern crate image;
use image::io::Reader as ImageReader;

use image::imageops::FilterType;
use image::GenericImageView;

pub struct EncoderService {
    tokenizer: Tokenizer,
    encoder: Session,
    vision_mode: bool,
    vision_size: usize,
}

impl EncoderService {
    fn new(
        environment: &Arc<Environment>,
        vision_mode: bool,
        pad_token_sequence: bool,
        input_image_size: usize,
    ) -> Result<EncoderService, Box<dyn std::error::Error + Send + Sync>> {
        let model_path = if vision_mode {
            "clip_visual.onnx"
        } else {
            "clip_textual.onnx"
        };
        let tokenizer_path = "fashion-clip-onnx/tokenizer.json";

        let mut tokenizer = Tokenizer::from_file(tokenizer_path)?;
        tokenizer.with_padding(Some(PaddingParams {
            strategy: if pad_token_sequence {
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
            vision_mode,
            vision_size: input_image_size,
        })
    }

    pub fn _process_text(
        &self,
        text: &Vec<String>,
    ) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error + Send + Sync>> {
        for input_info in &self.encoder.inputs {
            println!("Input Name: {}", input_info.name);
            println!("Input Type: {:?}", input_info.input_type);
        }

        for output_info in &self.encoder.outputs {
            println!("Output Name: {}", output_info.name);
            println!("Output Type: {:?}", output_info.output_type);
        }

        let preprocessed = self.tokenizer.encode_batch(text.clone(), true)?;
        let tokens_vector: Vec<_> = preprocessed
            .iter()
            .map(|i| i.get_ids().iter().map(|b| *b as i64).collect::<Vec<_>>())
            .concat();

        let ids = CowArray::from(Array2::from_shape_vec(
            (text.len(), tokens_vector.len() / text.len()),
            tokens_vector,
        )?)
        .into_dyn();
        let attention_mask_vector: Vec<_> = preprocessed
            .iter()
            .map(|i| {
                i.get_attention_mask()
                    .iter()
                    .map(|b| *b as i64)
                    .collect::<Vec<_>>()
            })
            .concat();

        let _mask = CowArray::from(Array2::from_shape_vec(
            (text.len(), attention_mask_vector.len() / text.len()),
            attention_mask_vector,
        )?)
        .into_dyn();

        let session = &self.encoder;
        let outputs: Vec<Value<'_>> = session.run(vec![
            Value::from_array(session.allocator(), &ids)?,
            // Value::from_array(session.allocator(), &mask)?,
        ])?;
        let binding = outputs[0].try_extract()?;
        let embeddings = binding.view();

        let seq_len = embeddings.shape().get(1).ok_or("not")?;

        Ok(embeddings
            .iter()
            .map(|s| *s)
            .chunks(*seq_len)
            .into_iter()
            .map(|b| b.collect())
            .collect())
    }

    pub fn _process_image(
        &self,
        images_bytes: &Vec<Vec<u8>>,
    ) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error + Send + Sync>> {
        let mean = vec![0.48145466, 0.4578275, 0.40821073]; // CLIP Dataset
        let std = vec![0.26862954, 0.26130258, 0.27577711];

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
            .iter()
            .map(|s| *s)
            .chunks(*seq_len)
            .into_iter()
            .map(|b| b.collect())
            .collect())
    }
}

#[tonic::async_trait]
impl Encoder for EncoderService {
    async fn encode_text(
        &self,
        request: Request<EncodeTextRequest>,
    ) -> Result<Response<EncoderResponse>, Status> {
        if self.vision_mode {
            return Err(Status::invalid_argument("wrong model is loaded"));
        }
        let texts = &request.get_ref().texts;
        return match self._process_text(texts) {
            Ok(d) => {
                let embedding = d.into_iter().map(|i| Embedding { point: i }).collect();
                Ok(Response::new(EncoderResponse { embedding }))
            }
            Err(e) => Err(Status::internal(format!("{:?}", e))),
        };
    }
    async fn encode_image(
        &self,
        request: tonic::Request<encoder::EncodeImageRequest>,
    ) -> Result<Response<EncoderResponse>, Status> {
        if !self.vision_mode {
            return Err(Status::invalid_argument("wrong model is loaded"));
        }
        let images = &request.get_ref().images;
        return match self._process_image(images) {
            Ok(d) => {
                let embedding = d.into_iter().map(|i| Embedding { point: i }).collect();
                Ok(Response::new(EncoderResponse { embedding }))
            }
            Err(e) => Err(Status::internal(format!("{:?}", e))),
        };
    }
}
pub fn create_encoder_service(
    environment: &Arc<Environment>,
    vision_mode: bool,
    pad_token_sequence: bool,
    input_image_size: usize,
) -> Result<EncoderService, Box<dyn std::error::Error + Send + Sync>> {
    EncoderService::new(
        environment,
        vision_mode,
        pad_token_sequence,
        input_image_size,
    )
}
