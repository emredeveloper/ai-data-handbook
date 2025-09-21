import os
import jiwer
from transformers import pipeline
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

# Doğru kabul edilen referans metin
reference = "Apple telefonları tanıttı, herkes de bir şeyler söyledi, tamam ama bak şimdi. Anladık çok ince telefon yapmışsın, sanki başkası daha önce yapmamış gibi. Ben bu telefonu almaya kalksam yaklaşık 44 bin lira vergi vereceğim. Telefona 98 bin lira verdikten sonra benim sadece tek kanlarım olacak. Ben onbirden devam kardeş."

"""Yerel fine-tuned modeli ve tokenizer yolunu script konumuna göre kur"""
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.join(script_dir, "Whisper Turkish Fine Tuning", "whisper-small-turkish")
ckpt_dir = os.path.join(base_dir, "checkpoint-10")

# Windows'ta backslash sorunlarına karşı güvenli hale getir (opsiyonel)
base_dir = os.path.normpath(base_dir)
ckpt_dir = os.path.normpath(ckpt_dir)

if not os.path.isdir(ckpt_dir):
    raise FileNotFoundError(f"Checkpoint klasörü bulunamadı: {ckpt_dir}")

# Modelleri yükle ve transkribe et (yerel fine-tuned model + üst klasördeki tokenizer)
pipe_emre = pipeline("automatic-speech-recognition", model=ckpt_dir, tokenizer=base_dir)
# Dil/ görev sabitle (Türkçe transcribe)
pipe_emre.model.config.forced_decoder_ids = pipe_emre.tokenizer.get_decoder_prompt_ids(language="turkish", task="transcribe")
transcription_emre = pipe_emre("iphone-air.mp3", generate_kwargs={
    "task": "transcribe",
    "language": "turkish",
    "no_repeat_ngram_size": 2,
    "repetition_penalty": 1.2,
    "do_sample": False,
    "max_new_tokens": 100,
    "num_beams": 5,
    "length_penalty": 1.2,
    "min_length": 5,
})["text"]

pipe_openai = pipeline("automatic-speech-recognition", model="openai/whisper-small")
pipe_openai.model.config.forced_decoder_ids = pipe_openai.tokenizer.get_decoder_prompt_ids(language="turkish", task="transcribe")
transcription_openai = pipe_openai("iphone-air.mp3", generate_kwargs={
    "task": "transcribe",
    "language": "turkish",
    "no_repeat_ngram_size": 2,
    "repetition_penalty": 1.2,
    "do_sample": False,
    "max_new_tokens": 100,
    "num_beams": 5,
    "length_penalty": 1.2,
    "min_length": 5,
})["text"]

# WER hesapla (normalize ederek)
transform = jiwer.Compose([
    jiwer.Strip(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
])
ref_n = transform(reference)
hyp_emre_n = transform(transcription_emre)
hyp_openai_n = transform(transcription_openai)
wer_emre = jiwer.wer(ref_n, hyp_emre_n)
wer_openai = jiwer.wer(ref_n, hyp_openai_n)

# Sonuçları yazdır
print(f"Emredeveloper modelinin WER değeri: {wer_emre:.4f}")
print(f"OpenAI Whisper modelinin WER değeri: {wer_openai:.4f}")

# Karşılaştırma
if wer_emre < wer_openai:
    print("Emredeveloper modeli daha iyi performans gösterdi.")
elif wer_openai < wer_emre:
    print("OpenAI Whisper modeli daha iyi performans gösterdi.")
else:
    print("İki model de aynı performansı gösterdi.")