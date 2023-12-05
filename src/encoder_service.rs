use autometrics::autometrics;
use embed_rs::embed::{EmbedImage, EmbedText};
use tonic::{Request, Response, Status};

pub mod encoder {
    tonic::include_proto!("encoder");
}

use self::encoder::{
    encoder_server::Encoder, EncodeImageRequest, EncodeTextRequest, EncoderResponse,
};
use autometrics::objectives::{Objective, ObjectiveLatency, ObjectivePercentile};

const API_SLO: Objective = Objective::new("embed-rs")
    .success_rate(ObjectivePercentile::P99_9)
    .latency(ObjectiveLatency::Ms10, ObjectivePercentile::P99);

pub struct EncoderService {
    pub embed_text: EmbedText,
    pub embed_image: EmbedImage,
}

#[tonic::async_trait]
impl Encoder for EncoderService {
    #[autometrics(objective = API_SLO)]
    async fn encode_text(
        &self,
        request: Request<EncodeTextRequest>,
    ) -> Result<Response<EncoderResponse>, Status> {
        let texts = &request.get_ref().texts;
        if texts.is_empty() {
            return Err(Status::invalid_argument("No text provided"));
        }
        return match self.embed_text.encode(texts) {
            Ok(d) => {
                let embedding = d.into_iter().flat_map(|i| vec![i]).collect();
                Ok(Response::new(EncoderResponse { embedding }))
            }
            Err(e) => Err(Status::internal(format!("{:?}", e))),
        };
    }

    #[autometrics(objective = API_SLO)]
    async fn encode_image(
        &self,
        request: Request<EncodeImageRequest>,
    ) -> Result<Response<EncoderResponse>, Status> {
        let image = &request.get_ref().image;
        return match self.embed_image.encode(&vec![image.clone() as Vec<u8>]) {
            Ok(d) => {
                let embedding = d.clone().into_iter().flat_map(|i| vec![i]).collect();
                Ok(Response::new(EncoderResponse { embedding }))
            }
            Err(e) => Err(Status::internal(format!("{:?}", e))),
        };
    }
}
