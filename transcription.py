import configparser
from openai import OpenAI
from pydub import AudioSegment
import os

config = configparser.ConfigParser()
config.read("config.ini")

def compress_and_convert_audio(file_path, max_size_mb=20):
    audio = AudioSegment.from_file(file_path)
    compressed_file_path = "compressed_audio.mp3"
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    
    if file_size_mb > max_size_mb:
        print(f"Compressing audio file from {file_size_mb:.2f} MB to under {max_size_mb} MB.")
        
        # Estimate the required bitrate reduction factor
        bitrate_reduction_factor = file_size_mb / max_size_mb
        new_bitrate = int(192 / bitrate_reduction_factor)  # 192 kbps is a reasonable starting point
        bitrate = f"{new_bitrate}k"
        audio.export(compressed_file_path, format="mp3", bitrate=bitrate)
    else:
        print(f"Audio file size is {file_size_mb:.2f} MB, converting without compression.")
        audio.export(compressed_file_path, format="mp3")
    
    return compressed_file_path

def transcrever_audio(file_path, client):
    audio_file = open(file_path, "rb")
    client = OpenAI(api_key=config["OPENAI"]["api_key"])
    response = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
        language="pt",
        temperature=0.5,
        # prompt="Transcreva a entrevista, separando as falas dos entrevistados em 'Entrevistador' e 'Entrevistado'.",
        prompt="Transcreva o áudio.",
    )
    return response.text

def formatar_texto(transcricao, client):
    # prompt = f"Por favor, corrija erros de coerência e gramática e formate o texto a seguir, separando as falas do entrevistador (normalmente quem estará fazendo perguntas) e do gerente do restaurante:\n\n{transcricao}"
    prompt = f"Por favor, corrija erros de coerência e gramática e formate o texto a seguir:\n\n{transcricao}"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "Você é um assistente útil que corrige e formata textos.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )
    return response.choices[0].message.content

def salvar_texto_em_arquivo(texto, caminho_arquivo):
    with open(caminho_arquivo, "w", encoding="utf-8") as file:
        file.write(texto)

if __name__ == "__main__":
    audio_file_path = "audio.m4a"
    output_file_path = "transcricao_formatada.txt"
    max_size_mb = 25
    client = OpenAI(api_key=config["OPENAI"]["api_key"])

    compressed_audio_file_path = compress_and_convert_audio(audio_file_path, max_size_mb)
    transcricao = transcrever_audio(compressed_audio_file_path, client)
    texto_formatado = formatar_texto(transcricao, client)

    salvar_texto_em_arquivo(texto_formatado, output_file_path)
    print(f"Texto formatado foi salvo em: {output_file_path}")