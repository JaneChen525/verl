"""NaVIDA stateless prompt builder (byte-faithful to eval_vllm_navida.py).

Each decision rebuilds [system, user(intro + sampled-8 history frames + bridge +
current frame + tail)]. Images are forwarded as env_server JPEG base64 DIRECTLY
(no decode/re-encode) so train==rollout==eval pixels match (avoids double-JPEG).
"""
SYSTEM_PROMPT = "You are a helpful assistant."

PROMPT_TEMPLATE = (
    "Imagine you are a robot programmed for navigation tasks. "
    "You have been given a video of historical observations and an image of the current observation. "
    "Your assigned task is: '{}'. Analyze this series of images to decide your next move, "
    "which could involve turning left or right by a specific degree or moving forward a certain distance."
)
# eval hardcodes these two text pieces around the images:
NAVIDA_INTRO = ("Imagine you are a robot programmed for navigation tasks. "
                "You have been given a video of historical observations")
NAVIDA_BRIDGE = "and an image of the current observation"
K_HISTORY = 8


def navida_tail(instruction: str) -> str:
    return PROMPT_TEMPLATE.format(instruction).split("current observation")[1]


def uniform_sample_with_ends(data: list, n: int) -> list:
    if len(data) <= n:
        return data
    idx = [round(i * (len(data) - 1) / (n - 1)) for i in range(n)]
    return [data[i] for i in idx]


def build_navida_messages_b64(instruction: str, b64_buffer: list, k_history: int = K_HISTORY):
    """b64_buffer = list of env_server JPEG base64 strings, [-1] = current frame.

    Returns OpenAI-format messages (data-URL images). Matches eval: if only the
    first frame exists, the single history slot is the current frame.
    """
    current = b64_buffer[-1]
    historic = uniform_sample_with_ends(b64_buffer[:-1], k_history) if len(b64_buffer) > 1 else [current]

    def img(b):
        return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}}

    content = [{"type": "text", "text": NAVIDA_INTRO}]
    content += [img(b) for b in historic]
    content.append({"type": "text", "text": NAVIDA_BRIDGE})
    content.append(img(current))
    content.append({"type": "text", "text": navida_tail(instruction)})
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ]
