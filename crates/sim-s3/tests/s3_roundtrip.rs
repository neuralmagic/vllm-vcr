//! Live round trip against an S3-compatible endpoint through the OpenSSL
//! HttpClient: PUT a file, GET it back, compare bytes. Needs credentials and
//! an endpoint in the environment (AWS_ENDPOINT_URL, AWS_ACCESS_KEY_ID,
//! AWS_SECRET_ACCESS_KEY, AWS_REGION) plus S3_ROUNDTRIP_BUCKET; run with
//! `cargo test -p sim-s3 --test s3_roundtrip -- --ignored`.

use sim_s3::TraceUri;

#[tokio::test]
#[ignore]
async fn put_then_get_through_openssl_client() {
    let bucket = std::env::var("S3_ROUNDTRIP_BUCKET").expect("S3_ROUNDTRIP_BUCKET");
    let scratch = tempfile::tempdir().expect("tempdir");
    let payload = format!("vllm-vcr s3 roundtrip {}", std::process::id());
    let local = scratch.path().join("probe.txt");
    std::fs::write(&local, &payload).expect("write probe");

    let uri: TraceUri = format!(
        "s3://{bucket}/vllm-vcr-roundtrip/{}.txt",
        std::process::id()
    )
    .parse()
    .expect("uri");
    uri.upload(&local).await.expect("PUT");
    let fetched = uri.materialize(scratch.path()).await.expect("GET");
    assert_ne!(
        fetched, local,
        "GET must land in scratch, not the source file"
    );
    assert!(fetched.starts_with(scratch.path()));
    assert_eq!(std::fs::read_to_string(&fetched).expect("read"), payload);
}
