import dwani 
import os 
import tempfile 
dwani.api_key = os.getenv("DWANI_API_KEY")
dwani.api_base = os.getenv("DWANI_API_BASE_URL")



text = "ಧ್ವನಿ ಭಾರತೀಯ ಭಾಷೆಗಳಲ್ಲಿ 1 ಬಿಲಿಯನ್ ಬಳಕೆದಾರರನ್ನು ತಲುಪಲು ಮತ್ತು ಸಮಸ್ಯೆಗಳನ್ನು ಪರಿಹರಿಸಲು ಸಹಾಯ ಮಾಡಲು AI ಅನ್ನು ನಿರ್ಮಿಸುತ್ತಿದೆ."
language = "kannada"
audio_file
try:
    response = dwani.Audio.speech(
        input=text,
        response_format="mp3",
        language=language
    )
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
        temp_file.write(response)
        return temp_file.name

result = dwani.Translate.run_translate(sentences=sentences, src_lang=src_lang, tgt_lang=tgt_lang)

result = dwani.ASR.transcribe(file_path=audio_file, language=language)