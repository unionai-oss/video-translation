from flytekit import ImageSpec, Resources, task

from .clone_voice import language_codes as clone_voice_language_codes

language_codes = {
    "Afrikaans": "af",
    "Amharic": "am",
    "Arabic": "ar",
    "Asturian": "ast",
    "Azerbaijani": "az",
    "Bashkir": "ba",
    "Belarusian": "be",
    "Bulgarian": "bg",
    "Bengali": "bn",
    "Breton": "br",
    "Bosnian": "bs",
    "Catalan; Valencian": "ca",
    "Cebuano": "ceb",
    "Czech": "cs",
    "Welsh": "cy",
    "Danish": "da",
    "German": "de",
    "Greeek": "el",
    "English": "en",
    "Spanish": "es",
    "Estonian": "et",
    "Persian": "fa",
    "Fulah": "ff",
    "Finnish": "fi",
    "French": "fr",
    "Western Frisian": "fy",
    "Irish": "ga",
    "Gaelic; Scottish Gaelic": "gd",
    "Galician": "gl",
    "Gujarati": "gu",
    "Hausa": "ha",
    "Hebrew": "he",
    "Hindi": "hi",
    "Croatian": "hr",
    "Haitian; Haitian Creole": "ht",
    "Hungarian": "hu",
    "Armenian": "hy",
    "Indonesian": "id",
    "Igbo": "ig",
    "Iloko": "ilo",
    "Icelandic": "is",
    "Italian": "it",
    "Japanese": "ja",
    "Javanese": "jv",
    "Georgian": "ka",
    "Kazakh": "kk",
    "Central Khmer": "km",
    "Kannada": "kn",
    "Korean": "ko",
    "Luxembourgish; Letzeburgesch": "lb",
    "Ganda": "lg",
    "Lingala": "ln",
    "Lao": "lo",
    "Lithuanian": "lt",
    "Latvian": "lv",
    "Malagasy": "mg",
    "Macedonian": "mk",
    "Malayalam": "ml",
    "Mongolian": "mn",
    "Marathi": "mr",
    "Malay": "ms",
    "Burmese": "my",
    "Nepali": "ne",
    "Dutch; Flemish": "nl",
    "Norwegian": "no",
    "Northern Sotho": "ns",
    "Occitan (post 1500)": "oc",
    "Oriya": "or",
    "Panjabi; Punjabi": "pa",
    "Polish": "pl",
    "Pushto; Pashto": "ps",
    "Portuguese": "pt",
    "Romanian; Moldavian; Moldovan": "ro",
    "Russian": "ru",
    "Sindhi": "sd",
    "Sinhala; Sinhalese": "si",
    "Slovak": "sk",
    "Slovenian": "sl",
    "Somali": "so",
    "Albanian": "sq",
    "Serbian": "sr",
    "Swati": "ss",
    "Sundanese": "su",
    "Swedish": "sv",
    "Swahili": "sw",
    "Tamil": "ta",
    "Thai": "th",
    "Tagalog": "tl",
    "Tswana": "tn",
    "Turkish": "tr",
    "Ukrainian": "uk",
    "Urdu": "ur",
    "Uzbek": "uz",
    "Vietnamese": "vi",
    "Wolof": "wo",
    "Xhosa": "xh",
    "Yiddish": "yi",
    "Yoruba": "yo",
    "Chinese": "zh",
    "Zulu": "zu",
}

language_translation_image = ImageSpec(
    name="language_translation",
    builder="ucimage",
    packages=[
        "transformers==4.36.2",
        "torch==2.2.1",
        "accelerate==0.27.2",
        "bitsandbytes==0.43.0",
        "flytekit==1.10.7",
        "sentencepiece==0.2.0",
        "nltk==3.8.1",
        "unionai==0.1.5",
    ],
)


@task(
    cache=True,
    cache_version="2",
    container_image=language_translation_image,
    requests=Resources(mem="10Gi", cpu="3"),
)
def translate_text(translate_from: str, translate_to: str, input: str) -> str:
    import nltk
    from nltk import sent_tokenize
    from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

    if translate_to not in clone_voice_language_codes:
        raise ValueError(f"{translate_to} language isn't supported by Coqui TTS model.")

    if translate_to not in language_codes:
        raise ValueError(f"{translate_to} language isn't supported by M2M100 model.")

    model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_1.2B")
    tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_1.2B")

    tokenizer.src_lang = language_codes[translate_from]

    nltk.download("punkt")
    result = []
    for sentence in sent_tokenize(input):
        encoded_input = tokenizer(sentence, return_tensors="pt")

        generated_tokens = model.generate(
            **encoded_input,
            forced_bos_token_id=tokenizer.get_lang_id(language_codes[translate_to]),
        )
        output = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        result += output

    return " ".join(result).strip()
