import configparser
from openai import OpenAI
from pydub import AudioSegment
import os
import logging

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("transcription.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("transcription")

# Load configurations from config file
config = configparser.ConfigParser()
config.read("config.ini")

def compress_and_convert_audio(file_path, max_size_mb=20):
    """
    Compress and convert an audio file to MP3 format if it exceeds the maximum size.
    
    Args:
        file_path (str): Path to the input audio file
        max_size_mb (int, optional): Maximum file size in MB. Defaults to 20.
        
    Returns:
        str: Path to the compressed/converted audio file
    """
    logger.info(f"Starting compression of audio file: {file_path}")
    audio = AudioSegment.from_file(file_path)
    compressed_file_path = "compressed_audio.mp3"
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    
    if file_size_mb > max_size_mb:
        logger.info(f"Compressing audio file from {file_size_mb:.2f} MB to under {max_size_mb} MB.")
        
        # Estimate the required bitrate reduction factor
        bitrate_reduction_factor = file_size_mb / max_size_mb
        new_bitrate = int(192 / bitrate_reduction_factor)  # 192 kbps is a reasonable starting point
        bitrate = f"{new_bitrate}k"
        audio.export(compressed_file_path, format="mp3", bitrate=bitrate)
        logger.debug(f"Audio compressed with bitrate: {bitrate}")
    else:
        logger.info(f"Audio file size is {file_size_mb:.2f} MB, converting without compression.")
        audio.export(compressed_file_path, format="mp3")
    
    final_size_mb = os.path.getsize(compressed_file_path) / (1024 * 1024)
    logger.info(f"Compression completed. Final size: {final_size_mb:.2f} MB")
    return compressed_file_path

def transcrever_audio(file_path, client):
    """
    Transcribe an audio file to text using OpenAI's Whisper model.
    
    Args:
        file_path (str): Path to the audio file
        client (OpenAI): OpenAI client instance
        
    Returns:
        str: Transcribed text
    """
    logger.info(f"Starting transcription of audio file: {file_path}")
    audio_file = open(file_path, "rb")
    client = OpenAI(api_key=config["OPENAI"]["api_key"])
    
    try:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language="pt",
            temperature=0.5,
            prompt="Transcreva o áudio.",
        )
        logger.info("Audio transcription completed successfully")
        return response.text
    except Exception as e:
        logger.error(f"Error during audio transcription: {e}")
        raise

def formatar_texto(transcricao, client):
    """
    Format transcribed text to correct grammar and coherence issues.
    
    Args:
        transcricao (str): Raw transcription text
        client (OpenAI): OpenAI client instance
        
    Returns:
        str: Formatted and corrected text
    """
    logger.info("Starting text formatting")
    prompt = f"Por favor, corrija erros de coerência e gramática e formate o texto a seguir:\n\n{transcricao}"

    try:
        response = client.chat.completions.create(
            model="o1-2024-12-17",
            messages=[
                {
                    "role": "system",
                    "content": "Você é um assistente útil que corrige e formata textos.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        logger.info("Text formatting completed successfully")
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error during text formatting: {e}")
        logger.warning("Returning original text due to formatting error")
        return transcricao

def salvar_texto_em_arquivo(texto, caminho_arquivo):
    """
    Save text to a file.
    
    Args:
        texto (str): Text to save
        caminho_arquivo (str): Output file path
    """
    try:
        with open(caminho_arquivo, "w", encoding="utf-8") as file:
            file.write(texto)
        logger.info(f"Text saved successfully to file: {caminho_arquivo}")
    except Exception as e:
        logger.error(f"Error saving text to file {caminho_arquivo}: {e}")
        raise

if __name__ == "__main__":
    logger.info("Starting transcription process")
    audio_file_path = "audio.m4a"
    output_file_path = "transcricao_formatada.txt"
    max_size_mb = 25
    
    try:
        client = OpenAI(api_key=config["OPENAI"]["api_key"])
        logger.info(f"Processing audio file: {audio_file_path}")

        compressed_audio_file_path = compress_and_convert_audio(audio_file_path, max_size_mb)
        logger.info(f"Audio compressed and converted to: {compressed_audio_file_path}")
        
        transcricao = transcrever_audio(compressed_audio_file_path, client)
        logger.debug(f"Raw transcription length: {len(transcricao)} characters")
        
        texto_formatado = formatar_texto(transcricao, client)
        logger.debug(f"Formatted text length: {len(texto_formatado)} characters")

        salvar_texto_em_arquivo(texto_formatado, output_file_path)
        logger.info(f"Formatted text saved to: {output_file_path}")
        logger.info("Transcription process completed successfully")
    except Exception as e:
        logger.critical(f"Transcription process failed: {e}")