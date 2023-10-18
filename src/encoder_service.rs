use tonic::{Request, Response, Status};

pub mod encoder {
    tonic::include_proto!("encoder");
}

use crate::EncoderService;

use self::encoder::{encoder_server::Encoder, EncodeTextRequest, EncoderResponse, Embedding, EncodeImageRequest};


#[tonic::async_trait]
impl Encoder for EncoderService {
    async fn encode_text(
        &self,
        request: Request<EncodeTextRequest>,
    ) -> Result<Response<EncoderResponse>, Status> {
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
        request: Request<EncodeImageRequest>,
    ) -> Result<Response<EncoderResponse>, Status> {
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