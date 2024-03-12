import re

from flytekit import ImageSpec, Resources, task
from flytekit.extras.accelerators import T4

from .clone_voice import language_codes

language_translation_image = ImageSpec(
    name="language_translation",
    registry="samhitaalla",
    packages=[
        "transformers==4.36.2",
        "torch==2.2.1",
        "accelerate==0.27.2",
        "bitsandbytes==0.43.0",
        "flytekit==1.10.7",
    ],
    cuda="12.1.0",
    cudnn="8",
    python_version="3.11",
)

if language_translation_image.is_container():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


@task(
    cache=True,
    cache_version="1",
    container_image=language_translation_image,
    requests=Resources(gpu="1", mem="10Gi", cpu="1"),
    accelerator=T4,
)
def translate_text(translate_from: str, translate_to: str, input: str) -> str:
    if translate_to not in language_codes:
        raise ValueError(f"{translate_to} language isn't supported by Coqui TTS model.")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.2",
        quantization_config=quantization_config,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

    messages = [
        {
            "role": "user",
            "content": f"""Just generate the translated text without any sort of explanations and notes, and ensure the output is written in English.
            Translate this text from English to Spanish:
            What is your favourite condiment?""",
        },
        {
            "role": "assistant",
            "content": "¿Cuál es tu condimento favorito?",
        },
        {
            "role": "user",
            "content": f"""Translate this text from {translate_from} to {translate_to}: 
            {input}""",
        },
    ]

    model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)

    generated_text = tokenizer.batch_decode(
        generated_ids[:, model_inputs.shape[1] :], skip_special_tokens=True
    )[0]

    try:
        extracted_text = re.search(
            r'"([^"]*)"',
            generated_text,
        ).group(1)
        return extracted_text
    except:
        return generated_text
