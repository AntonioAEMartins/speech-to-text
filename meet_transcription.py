import base64
import configparser
from openai import OpenAI
from pydub import AudioSegment
import os
import math
import concurrent.futures
import tempfile
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
logger = logging.getLogger("meet_transcription")

# Load configurations from config.ini
config = configparser.ConfigParser()
config.read("config.ini")


def compress_and_convert_audio_to_mp4(file_path, max_size_mb=25):
    """
    Convert an audio file to MP4 (AAC) ensuring the final size doesn't exceed 'max_size_mb'.
    
    Calculates an appropriate bitrate and attempts to export via ffmpeg, considering
    a 1.5MB overhead. If needed, reduces sample rate and channels to meet size requirements.
    
    Args:
        file_path (str): Path to the original audio file
        max_size_mb (int, optional): Maximum file size in MB (including 1.5MB overhead). Defaults to 25.
        
    Returns:
        str: Path to the converted MP4 file
        
    Raises:
        ValueError: If the overhead is greater than or equal to the maximum allowed size
    """
    # Overhead to consider
    overhead_mb = 1.5
    target_size_mb = max_size_mb - overhead_mb
    if target_size_mb <= 0:
        raise ValueError("Overhead greater than or equal to maximum allowed size.")

    target_size_bytes = target_size_mb * 1024 * 1024

    # Load original audio
    audio = AudioSegment.from_file(file_path)
    duration_seconds = len(audio) / 1000.0

    # Output file
    compressed_file_path = "compressed_audio.mp4"

    # Calculate required bitrate (in kbps)
    # Formula: file_size_bytes ≈ (bitrate_kbps * 1000 / 8) * duration_seconds
    required_bitrate_kbps = (target_size_bytes * 8) / (duration_seconds * 1000)

    # Typical limits for AAC audio
    min_bitrate_kbps = 16  # 16 kbps is very low, but works
    max_bitrate_kbps = 320  # 320 kbps is a high value for AAC

    # Ensure bitrate stays within limits
    chosen_bitrate = max(
        min_bitrate_kbps,
        min(int(math.floor(required_bitrate_kbps)), max_bitrate_kbps),
    )

    def export_mp4(audio_segment, out_path, bitrate_kbps, sr=None, channels=None):
        """
        Export audio segment to MP4 with specified parameters.
        
        Args:
            audio_segment: The audio segment to export
            out_path: Output file path
            bitrate_kbps: Audio bitrate in kbps
            sr: Sample rate (optional)
            channels: Number of channels (optional)
            
        Returns:
            float: Final file size in MB
        """
        # Modify sample rate or channels if specified
        if sr is not None:
            audio_segment = audio_segment.set_frame_rate(sr)
        if channels is not None:
            audio_segment = audio_segment.set_channels(channels)

        audio_segment.export(
            out_path,
            format="mp4",
            bitrate=f"{bitrate_kbps}k",
            codec="aac",  # force AAC codec
            parameters=["-profile:a", "aac_low"],  # AAC LC profile
        )
        return os.path.getsize(out_path) / (1024 * 1024)  # MB

    # First attempt
    final_size_mb = export_mp4(audio, compressed_file_path, chosen_bitrate)
    logger.info(f"Attempt with {chosen_bitrate} kbps => {final_size_mb:.2f} MB")

    # If still above limit, reduce sample rate and/or channels
    if final_size_mb > max_size_mb:
        # Try reducing sample rate and channels to mono (16000 Hz, 1 channel)
        chosen_bitrate = max(
            min_bitrate_kbps, min(chosen_bitrate, 128)
        )  # reduce bitrate further
        final_size_mb = export_mp4(
            audio, compressed_file_path, chosen_bitrate, sr=16000, channels=1
        )
        logger.info(
            f"Attempt reducing SR to 16000 Hz, mono and {chosen_bitrate} kbps => {final_size_mb:.2f} MB"
        )

    # Final message
    if final_size_mb > max_size_mb:
        logger.warning(
            f"Final file is {final_size_mb:.2f} MB, above the limit of {max_size_mb} MB!"
        )
    else:
        logger.info(f"OK! Final .mp4 file = {final_size_mb:.2f} MB (<= {max_size_mb} MB)")

    return compressed_file_path


def split_audio_into_chunks(file_path, num_chunks=10):
    """
    Split an audio file into approximately equal parts.
    
    Args:
        file_path (str): Path to the audio file
        num_chunks (int, optional): Number of parts to split into. Defaults to 10.
        
    Returns:
        list: List of AudioSegment objects representing each chunk
    """
    audio = AudioSegment.from_file(file_path)
    duration = len(audio)
    chunk_length = duration // num_chunks
    chunks = []
    for i in range(num_chunks):
        start = i * chunk_length
        # For the last chunk, go to the end
        end = (i + 1) * chunk_length if i < num_chunks - 1 else duration
        chunks.append(audio[start:end])
    
    logger.info(f"Audio file split into {num_chunks} chunks")
    return chunks


def transcrever_audio(file_path, client, num_chunks=10, max_workers=5):
    """
    Transcribe audio using OpenAI's Whisper by dividing it into chunks and processing concurrently.
    
    Args:
        file_path (str): Path to the audio file
        client (OpenAI): OpenAI client instance
        num_chunks (int, optional): Number of chunks to divide the audio into. Defaults to 10.
        max_workers (int, optional): Maximum number of threads to use. Defaults to 5.
        
    Returns:
        str: Complete audio transcription
    """
    logger.info(f"Starting transcription of {file_path} with {num_chunks} chunks and {max_workers} workers")
    audio_chunks = split_audio_into_chunks(file_path, num_chunks=num_chunks)
    transcricao_total = ["" for _ in range(num_chunks)]  # Pre-allocate list to maintain order

    def transcrever_chunk(idx, chunk):
        """
        Transcribe an individual audio chunk.
        
        Args:
            idx (int): Index of the chunk
            chunk (AudioSegment): Audio segment to transcribe
        """
        logger.debug(f"Starting transcription of chunk {idx + 1}/{num_chunks}")
        # Create a temporary file for the chunk
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
            chunk.export(tmp_file.name, format="mp4", codec="aac")
            tmp_file_path = tmp_file.name

        # Transcribe the chunk
        try:
            with open(tmp_file_path, "rb") as audio_file:
                response = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language="pt",
                    temperature=0.5,
                    prompt="Transcreva o áudio.",
                )
                transcricao_total[idx] = response.text
                logger.info(f"Transcription of part {idx + 1}/{num_chunks} completed.")
        except Exception as e:
            logger.error(f"Error in transcription of part {idx + 1}: {e}")
            transcricao_total[idx] = ""
        finally:
            # Remove the temporary file
            os.remove(tmp_file_path)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(transcrever_chunk, idx, chunk)
            for idx, chunk in enumerate(audio_chunks)
        ]
        # Wait for completion and handle exceptions
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                logger.error(f"A thread generated an exception: {exc}")

    # Join all transcriptions in the correct order
    transcricao_completa = " ".join(transcricao_total)
    logger.info("Full transcription completed successfully")
    return transcricao_completa


def corrigir_nomes_frameworks(transcricao, client, num_chunks=10, max_workers=5):
    """
    Correct framework or library names in the transcription by dividing the text
    into chunks and processing corrections concurrently.
    
    Args:
        transcricao (str): Transcription text
        client (OpenAI): OpenAI client instance
        num_chunks (int, optional): Number of parts to divide the text. Defaults to 10.
        max_workers (int, optional): Maximum number of threads to use. Defaults to 5.
        
    Returns:
        str: Corrected text
    """
    logger.info(f"Starting framework name correction with {num_chunks} chunks and {max_workers} workers")
    
    def split_text_in_chunks(text, num_chunks=10):
        """
        Split the string 'text' into approximately equal parts.
        
        Args:
            text (str): The text to split
            num_chunks (int, optional): Number of chunks to create. Defaults to 10.
            
        Returns:
            list: List of text chunks
        """
        words = text.split()
        total_words = len(words)
        chunk_size = math.ceil(total_words / num_chunks)
        chunks = []
        for i in range(num_chunks):
            start = i * chunk_size
            end = start + chunk_size
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
        return chunks

    transcricao_chunks = split_text_in_chunks(transcricao, num_chunks=num_chunks)
    corrected_chunks = ["" for _ in range(num_chunks)]  # Pre-allocate list to maintain order
    logger.debug(f"Text split into {num_chunks} chunks for correction")

    def corrigir_chunk(idx, chunk):
        """
        Correct an individual text chunk.
        
        Args:
            idx (int): Index of the chunk
            chunk (str): Text chunk to correct
        """
        if not chunk.strip():
            corrected_chunks[idx] = ""
            return

        prompt = (
            "Revise a seguinte transcrição de uma reunião e corrija apenas os nomes de frameworks ou "
            "bibliotecas que foram possivelmente transcritos incorretamente. Não altere nenhum outro "
            "conteúdo do texto.\n\n"
            f"{chunk}"
        )

        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Você é um assistente que corrige apenas nomes de frameworks e bibliotecas em um texto."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )
            corrected_text_part = response.choices[0].message.content
            corrected_chunks[idx] = corrected_text_part
            logger.info(f"Part {idx + 1}/{num_chunks} corrected.")
        except Exception as e:
            logger.error(f"Error in correction of part {idx + 1}: {e}")
            corrected_chunks[idx] = chunk  # Keep original in case of error

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(corrigir_chunk, idx, chunk)
            for idx, chunk in enumerate(transcricao_chunks)
        ]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                logger.error(f"A thread generated an exception: {exc}")

    # Join all corrected parts in the same order
    transcricao_corrigida = " ".join(corrected_chunks)
    logger.info("Framework name correction completed successfully")
    return transcricao_corrigida


def salvar_texto_em_arquivo(texto, caminho_arquivo):
    """
    Save text to a file.
    
    Args:
        texto (str): Text to save
        caminho_arquivo (str): Path to the destination file
    """
    with open(caminho_arquivo, "w", encoding="utf-8") as file:
        file.write(texto)
    logger.info(f"Text saved to file: {caminho_arquivo}")


def main():
    """
    Main function to process audio transcription with framework name correction.
    
    Coordinates the audio compression, transcription and correction processes,
    managing file paths and OpenAI API interactions.
    """
    logger.info("Starting meet transcription process")
    audio_file_path = "audio.m4a"  # Path to original audio file
    compressed_audio_file_path = "compressed_audio.mp4"  # Path to compressed audio file
    transcricao_file_path = "transcricao.txt"  # Path to save initial transcription
    output_file_path = "transcricao_corrigida.txt"  # Path to save corrected transcription
    max_size_mb = 25  # Maximum allowed size for compressed audio file
    num_chunks = 10  # Number of chunks to divide audio/text
    max_workers = 5  # Number of threads for concurrency

    api_key = config["OPENAI"]["api_key"]

    # Initialize OpenAI client with API key
    client = OpenAI(api_key=api_key)
    logger.info(f"Processing audio file: {audio_file_path}")

    # Step 1: Compress and convert audio, if necessary
    compressed_audio_file_path = compress_and_convert_audio_to_mp4(
        audio_file_path, max_size_mb
    )
    logger.info(f"Audio compressed and converted to: {compressed_audio_file_path}")

    # Step 2: Transcribe audio (using OpenAI's Whisper) concurrently
    transcricao = transcrever_audio(
        compressed_audio_file_path,
        client,
        num_chunks=num_chunks,
        max_workers=max_workers
    )
    logger.info("Transcription completed.")

    # Save transcription to file
    salvar_texto_em_arquivo(transcricao, transcricao_file_path)
    logger.info(f"Transcription saved to: {transcricao_file_path}")

    # Read transcription from file
    try:
        with open("transcricao.txt", "r") as file:
            transcricao = file.read()
        logger.info("Transcription read from file successfully")
    except Exception as e:
        logger.error(f"Error reading transcription from file: {e}")
        # If error, use the transcription we already have

    # Step 3: Correct framework/library names in transcription concurrently
    transcricao_corrigida = corrigir_nomes_frameworks(
        transcricao,
        client,
        num_chunks=num_chunks,
        max_workers=max_workers
    )
    logger.info("Corrections applied to transcription.")

    # Step 4: Save corrected transcription to file
    salvar_texto_em_arquivo(transcricao_corrigida, output_file_path)
    logger.info(f"Corrected transcription saved to: {output_file_path}")
    logger.info("Meet transcription process completed successfully")


if __name__ == "__main__":
    main()
