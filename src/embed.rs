use std::error::Error;
use std::io::Cursor;

use image::io::Reader as ImageReader;
use image::GenericImageView;
use image::{imageops::FilterType, DynamicImage};
use itertools::Itertools;
use ndarray::{Array, Array2, Array4, ArrayBase, CowArray, CowRepr, Dim, IxDynImpl, OwnedRepr};
use ort::{Environment, ExecutionProvider, GraphOptimizationLevel, Session, SessionBuilder, Value};
use tokenizers::{Encoding, Tokenizer};

pub struct EmbedText {
    session: Session,
    tokenizer: Tokenizer,
}

impl EmbedText {
    pub fn new(
        text_model_path: &str,
        text_model_for_tokenizer: &str,
    ) -> Result<Self, Box<dyn Error + Send + Sync>> {
        let session = create_session(text_model_path)?;
        let tokenizer = Self::create_tokenizer(text_model_for_tokenizer)?;
        Ok(EmbedText { session, tokenizer })
    }

    pub fn encode(
        &self,
        text: &String,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error + Send + Sync>> {
        if text.is_empty() {
            return Err("No text provided".into());
        }
        let preprocessed = self.tokenizer.encode(text.clone(), true)?;

        let binding = vec![text.to_string()];
        let input_ids_vector = Self::get_input_ids_vector(preprocessed.clone(), &binding)?;

        let binding = vec![text.to_string()];
        let attention_mask_vector = Self::get_attention_mask_vector(preprocessed, &binding)?;

        let session = &self.session;

        // Input name: input_ids, shape: [0, 0]
        // Output name: embedding, shape: [512]
        let outputs = session.run(vec![
            Value::from_array(session.allocator(), &input_ids_vector)?,
            Value::from_array(session.allocator(), &attention_mask_vector)?,
        ])?;

        let text_embed_index = 0;
        let embeddings = try_extract(outputs, text_embed_index)?;
        Ok(embeddings)
    }

    fn create_tokenizer(
        text_model_for_tokenizer: &str,
    ) -> Result<Tokenizer, Box<dyn Error + Send + Sync>> {
        let mut tokenizer = Tokenizer::from_pretrained(text_model_for_tokenizer, None)?;
        tokenizer.with_padding(Some(tokenizers::PaddingParams {
            strategy: tokenizers::PaddingStrategy::BatchLongest,
            direction: tokenizers::PaddingDirection::Right,
            pad_to_multiple_of: None,
            pad_id: 0,
            pad_type_id: 0,
            pad_token: "[PAD]".to_string(),
        }));
        Ok(tokenizer)
    }

    fn get_attention_mask_vector(
        preprocessed: Encoding,
        text: &Vec<String>,
    ) -> Result<ArrayBase<CowRepr<'_, i64>, Dim<IxDynImpl>>, Box<dyn Error + Send + Sync>> {
        let attention_mask_vector: Vec<i64> = preprocessed
            .get_attention_mask()
            .iter()
            .map(|b| *b as i64)
            .collect::<Vec<i64>>();
        let mask_shape = (text.len(), attention_mask_vector.len() / text.len());
        let attention_mask_vector =
            CowArray::from(Array2::from_shape_vec(mask_shape, attention_mask_vector)?).into_dyn();
        Ok(attention_mask_vector)
    }

    fn get_input_ids_vector(
        preprocessed: Encoding,
        text: &Vec<String>,
    ) -> Result<ArrayBase<CowRepr<'_, i64>, Dim<IxDynImpl>>, Box<dyn Error + Send + Sync>> {
        let input_ids_vector: Vec<i64> = preprocessed
            .get_ids()
            .iter()
            .map(|b| *b as i64)
            .collect::<Vec<i64>>();
        let ids_shape = (text.len(), input_ids_vector.len() / text.len());
        let input_ids_vector =
            CowArray::from(Array2::from_shape_vec(ids_shape, input_ids_vector)?).into_dyn();
        Ok(input_ids_vector)
    }
}

pub struct EmbedImage {
    session: Session,
    image_width: usize,
    image_height: usize,
}

impl EmbedImage {
    pub fn new(
        model_path: &str,
        image_width: &usize,
        image_height: &usize,
    ) -> Result<Self, Box<dyn Error + Send + Sync>> {
        let session = create_session(model_path)?;
        Ok(Self {
            session,
            image_width: *image_width,
            image_height: *image_height,
        })
    }

    pub fn encode(
        &self,
        images_bytes: &Vec<Vec<u8>>,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error + Send + Sync>> {
        let mean = [0.48145466, 0.4578275, 0.40821073]; // CLIP Dataset
        let std = [0.26862954, 0.26130258, 0.27577711];

        let mut pixels = CowArray::from(Array4::<f32>::zeros(Dim([
            images_bytes.len(),
            3,
            self.image_width,
            self.image_height,
        ])));
        for (index, image_bytes) in images_bytes.iter().enumerate() {
            let image = ImageReader::new(Cursor::new(image_bytes))
                .with_guessed_format()?
                .decode()?;
            let image = self.to_rgb8(image);

            let image = self.crop(image);

            normalize(image, &mut pixels, index, mean, std);
        }

        let fixed_dim_array: ArrayBase<OwnedRepr<i64>, Dim<[usize; 1]>> = Array::from_vec(vec![0]);
        let dynamic_dim_array: ArrayBase<CowRepr<_>, _> = fixed_dim_array
            .into_shape((1, 1))
            .unwrap()
            .into_dyn()
            .into();
        let session = &self.session;
        let binding = pixels.into_dyn();
        let outputs = session.run(vec![
            Value::from_array(session.allocator(), &dynamic_dim_array)?,
            Value::from_array(session.allocator(), &binding)?,
            Value::from_array(session.allocator(), &dynamic_dim_array)?,
        ])?;

        let binding = outputs[3].try_extract()?;
        let embeddings = binding.view();

        let seq_len = embeddings.shape().get(1).unwrap();

        let embeddings: Vec<Vec<f32>> = embeddings
            .iter()
            .copied()
            .chunks(*seq_len)
            .into_iter()
            .map(|b| b.collect())
            .collect();
        let embedding = embeddings[0].clone();
        Ok(embedding)
    }

    fn crop(&self, image: DynamicImage) -> DynamicImage {
        let (width, height) = image.dimensions();

        // Calculate the coordinates for the top-left corner of the crop rectangle
        let start_x = (width - self.image_width as u32) / 2;
        let start_y = (height - self.image_height as u32) / 2;
        image.crop_imm(
            start_x,
            start_y,
            self.image_width as u32,
            self.image_height as u32,
        )
    }

    fn to_rgb8(&self, image: DynamicImage) -> DynamicImage {
        let image = DynamicImage::ImageRgb8(image.into_rgb8());
        image.resize_exact(
            self.image_width as u32,
            self.image_height as u32,
            FilterType::CatmullRom,
        )
    }
}

fn normalize(image: DynamicImage, pixels: &mut ArrayBase<CowRepr<'_, f32>, Dim<[usize; 4]>>, index: usize, mean: [f32; 3], std: [f32; 3]) {
    for (x, y, pixel) in image.pixels() {
        pixels[[index, 0, x as usize, y as usize]] =
            (pixel.0[0] as f32 / 255.0 - mean[0]) / std[0];
        pixels[[index, 1, x as usize, y as usize]] =
            (pixel.0[1] as f32 / 255.0 - mean[1]) / std[1];
        pixels[[index, 2, x as usize, y as usize]] =
            (pixel.0[2] as f32 / 255.0 - mean[2]) / std[2];
    }
}

fn create_session(model_path: &str) -> Result<Session, Box<dyn Error + Send + Sync>> {
    let environment = Environment::builder()
        .with_name("embed-rs")
        .with_execution_providers([ExecutionProvider::CPU(Default::default())])
        .build()?
        .into_arc();
    let num_cpus = num_cpus::get();
    let session = SessionBuilder::new(&environment)?
        .with_parallel_execution(true)?
        .with_intra_threads(num_cpus as i16)?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_model_from_file(model_path)?;
    Ok(session)
}

fn try_extract(
    outputs: Vec<Value<'_>>,
    embed_index: usize,
) -> Result<Vec<f32>, Box<dyn Error + Send + Sync>> {
    let binding = outputs[embed_index].try_extract()?;
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
