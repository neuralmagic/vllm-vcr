"""Dump diffusion-stage request and response frames from vLLM-Omni's own encoder.

The output is the ground truth for
crates/sim-protocol/tests/fixtures/omni_v<version>_diffusion_<name>.hex. Run it in
the rig's head container, where vllm-omni is installed:

    kubectl -n inference-sim exec -i omni-rig -c head -- python3 - < deploy/omni/encode_groundtruth.py

It prints one `<name> <hex>` line per frame.
"""

from PIL import Image
from vllm_omni.diffusion.stage_diffusion_client import StageDiffusionClient
from vllm_omni.distributed.omni_connectors.utils.serialization import OmniMsgpackEncoder
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.metrics.utils import diffusion_scheduler_waiting_metrics
from vllm_omni.outputs import OmniRequestOutput

encoder = OmniMsgpackEncoder()


def add_request(request_id: str, prompt: dict, params: OmniDiffusionSamplingParams) -> bytes:
    return encoder.encode(
        {
            "type": "add_request",
            "request_id": request_id,
            "prompt": prompt,
            "sampling_params": StageDiffusionClient._sampling_params_to_dict(params),
            "kv_sender_info": None,
            "kv_transfer_params": None,
            "payload_sender_info": None,
        }
    )


image = Image.new("RGB", (6, 4), (200, 30, 30))
mask = Image.new("L", (6, 4), 255)

frames = {
    # /v1/images/generations with size, n, and num_inference_steps set.
    "t2i": add_request(
        "img_gen-0123456789abcdef-01234567",
        {
            "prompt": "a red fox",
            "modalities": ["image"],
            "mm_processor_kwargs": {"target_h": 512, "target_w": 768},
            "height": 512,
            "width": 768,
        },
        OmniDiffusionSamplingParams(
            num_outputs_per_prompt=2, width=768, height=512, num_inference_steps=30, seed=7
        ),
    ),
    # /v1/images/generations with nothing but a prompt.
    "bare": add_request(
        "img_gen-fedcba9876543210-76543210",
        {"prompt": "a red fox", "modalities": ["image"]},
        OmniDiffusionSamplingParams(seed=7),
    ),
    # /v1/images/edits with two input images and a mask.
    "edit": add_request(
        "img_edit-0123456789abcdef-89abcdef",
        {
            "prompt": "make it night",
            "modalities": ["image"],
            "multi_modal_data": {"image": [image, image], "mask_image": mask},
            "mm_processor_kwargs": {"target_h": 4, "target_w": 6},
            "height": 4,
            "width": 6,
        },
        OmniDiffusionSamplingParams(
            width=6, height=4, width_not_provided=True, height_not_provided=True, seed=7
        ),
    ),
    "abort": encoder.encode({"type": "abort", "request_ids": ["img_gen-a", "img_gen-b"]}),
    # A finished request, with the metrics DiffusionEngine.step_streaming attaches.
    "result": encoder.encode(
        {
            "type": "result",
            "output": OmniRequestOutput.from_diffusion(
                request_id="img_gen-0123456789abcdef-01234567",
                images=[image, image],
                prompt={"prompt": "a red fox"},
                metrics={
                    "preprocess_time_ms": 0.25,
                    "diffusion_engine_exec_time_ms": 1500.5,
                    "output_ready_wait_time_ms": 0.0,
                    "postprocess_time_ms": 12.75,
                    **diffusion_scheduler_waiting_metrics(3),
                    "scheduler_queue_wait_ms": 420.0,
                    "vae_decode_time_ms": 30.5,
                    "forward_time_ms": 1040.0,
                },
            ),
        }
    ),
    # The lifecycle marker a stage sends when a request leaves its queue.
    "result_started": encoder.encode(
        {
            "type": "result",
            "output": OmniRequestOutput.from_diffusion(
                request_id="img_gen-0123456789abcdef-01234567",
                images=[],
                finished=False,
                custom_output={"_diffusion_request_lifecycle": "started"},
            ),
        }
    ),
    "error": encoder.encode(
        {
            "type": "error",
            "request_id": "img_gen-fedcba9876543210-76543210",
            "error": "height and width must be divisible by 16",
            "status_code": 400,
            "error_type": "BadRequestError",
            "metrics": diffusion_scheduler_waiting_metrics(1),
        }
    ),
    # Sent alone when a request is aborted: it names no request.
    "abort_metrics": encoder.encode({"type": "metrics", "metrics": diffusion_scheduler_waiting_metrics(2)}),
}

for name, data in frames.items():
    print(name, data.hex())
