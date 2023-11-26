use autometrics::autometrics;
use tonic::{Request, Response, Status};

pub mod encoder {
    tonic::include_proto!("encoder");
}

use crate::EncoderService;

use self::encoder::{
    encoder_server::Encoder, EncodeImageRequest, EncodeTextRequest, EncoderResponse,
};
use autometrics::objectives::{Objective, ObjectiveLatency, ObjectivePercentile};

const API_SLO: Objective = Objective::new("embed-rs")
    .success_rate(ObjectivePercentile::P99_9)
    .latency(ObjectiveLatency::Ms10, ObjectivePercentile::P99);

#[tonic::async_trait]
impl Encoder for EncoderService {
    #[autometrics(objective = API_SLO)]
    async fn encode_text(
        &self,
        request: Request<EncodeTextRequest>,
    ) -> Result<Response<EncoderResponse>, Status> {
        let texts = &request.get_ref().texts;
        return match self._process_text(texts) {
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
        // return dummy response without image or anythin
        Ok(Response::new(EncoderResponse { embedding: vec![] }))
    }
}
