namespace py jclaw

struct PingRequest {
  1: optional string note
}

struct PingResponse {
  1: bool ok
  2: string message
}

service JClawService {
  PingResponse ping(1: PingRequest request)
}

