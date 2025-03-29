import base64
import configparser
from openai import OpenAI  # Certifique-se de que esta importação está correta conforme a sua biblioteca OpenAI
from pydub import AudioSegment
import os
import math
import concurrent.futures
import tempfile

# Carrega as configurações do arquivo config.ini
config = configparser.ConfigParser()
config.read("config.ini")


def compress_and_convert_audio_to_mp4(file_path, max_size_mb=25):
    """
    Converte um arquivo de áudio para MP4 (AAC) tentando garantir que o tamanho final
    não exceda 'max_size_mb' (descontando overhead de 1.5MB).
    Faz um cálculo aproximado do bitrate e tenta exportar via ffmpeg.

    :param file_path: Caminho do arquivo de áudio original.
    :param max_size_mb: Tamanho máximo em MB (inclui overhead de 1.5 MB).
    :return: Caminho para o arquivo MP4 convertido.
    """
    # Overhead a ser considerado
    overhead_mb = 1.5
    target_size_mb = max_size_mb - overhead_mb
    if target_size_mb <= 0:
        raise ValueError("Overhead maior ou igual ao tamanho máximo permitido.")

    target_size_bytes = target_size_mb * 1024 * 1024

    # Carrega áudio original
    audio = AudioSegment.from_file(file_path)
    duration_seconds = len(audio) / 1000.0

    # Arquivo de saída
    compressed_file_path = "compressed_audio.mp4"

    # ----------------------------------------------------------
    # Passo 1: Cálculo aproximado de bitrate necessário (em kbps)
    # ----------------------------------------------------------
    #
    # Fórmula:
    #   tamanho_final_bytes ≈ (bitrate_kbps * 1000 / 8) * duração_segundos
    #
    # Isole o bitrate_kbps:
    #   bitrate_kbps = (tamanho_final_bytes * 8) / (duração_segundos * 1000)
    #

    required_bitrate_kbps = (target_size_bytes * 8) / (duration_seconds * 1000)

    # Limites típicos para áudio AAC
    min_bitrate_kbps = 16  # 16 kbps é muito baixo, mas funciona
    max_bitrate_kbps = 320  # 320 kbps é um valor bem alto para AAC

    # Garante que o bitrate fique dentro desses limites
    chosen_bitrate = max(
        min_bitrate_kbps,
        min(int(math.floor(required_bitrate_kbps)), max_bitrate_kbps),
    )

    # ----------------------------------------------------------
    # Passo 2: Exportar usando pydub/ffmpeg com codec AAC e esse bitrate
    # ----------------------------------------------------------
    def export_mp4(audio_segment, out_path, bitrate_kbps, sr=None, channels=None):
        # Se quisermos alterar amostragem ou canais, podemos fazer:
        if sr is not None:
            audio_segment = audio_segment.set_frame_rate(sr)
        if channels is not None:
            audio_segment = audio_segment.set_channels(channels)

        audio_segment.export(
            out_path,
            format="mp4",
            bitrate=f"{bitrate_kbps}k",
            codec="aac",  # força usar AAC
            parameters=["-profile:a", "aac_low"],  # perfil AAC LC
        )
        return os.path.getsize(out_path) / (1024 * 1024)  # MB

    # Primeira tentativa
    final_size_mb = export_mp4(audio, compressed_file_path, chosen_bitrate)
    print(f"Tentativa com {chosen_bitrate} kbps => {final_size_mb:.2f} MB")

    # ----------------------------------------------------------
    # Passo 3: Se ainda estiver acima do limite, reduz a sample rate e/ou canais
    # ----------------------------------------------------------
    if final_size_mb > max_size_mb:
        # Tenta reduzir sample rate e canais para mono (isso também impacta a compressão)
        # Exemplo: 16000 Hz, 1 canal
        chosen_bitrate = max(
            min_bitrate_kbps, min(chosen_bitrate, 128)
        )  # reduz um pouco o bitrate
        final_size_mb = export_mp4(
            audio, compressed_file_path, chosen_bitrate, sr=16000, channels=1
        )
        print(
            f"Tentativa reduzindo SR p/16000 Hz, mono e {chosen_bitrate} kbps => {final_size_mb:.2f} MB"
        )

    # Mensagem final
    if final_size_mb > max_size_mb:
        print(
            f"Aviso: Arquivo final tem {final_size_mb:.2f} MB, acima do limite de {max_size_mb} MB!"
        )
    else:
        print(f"OK! Arquivo .mp4 final = {final_size_mb:.2f} MB (<= {max_size_mb} MB)")

    return compressed_file_path


def split_audio_into_chunks(file_path, num_chunks=10):
    """
    Divide o arquivo de áudio em 'num_chunks' partes aproximadamente iguais.

    :param file_path: Caminho do arquivo de áudio.
    :param num_chunks: Número de partes para dividir.
    :return: Lista de objetos AudioSegment representando cada chunk.
    """
    audio = AudioSegment.from_file(file_path)
    duration = len(audio)
    chunk_length = duration // num_chunks
    chunks = []
    for i in range(num_chunks):
        start = i * chunk_length
        # Para o último chunk, vai até o final
        end = (i + 1) * chunk_length if i < num_chunks - 1 else duration
        chunks.append(audio[start:end])
    return chunks


def transcrever_audio(file_path, client, num_chunks=10, max_workers=5):
    """
    Usa o Whisper da OpenAI para transcrever o áudio dividindo em 'num_chunks' partes,
    processando os chunks de forma concorrente.

    :param file_path: Caminho do arquivo de áudio.
    :param client: Instância do cliente OpenAI.
    :param num_chunks: Número de partes para dividir o áudio.
    :param max_workers: Número máximo de threads a serem usadas.
    :return: Transcrição completa do áudio.
    """
    audio_chunks = split_audio_into_chunks(file_path, num_chunks=num_chunks)
    transcricao_total = ["" for _ in range(num_chunks)]  # Pré-aloca a lista para manter a ordem

    def transcrever_chunk(idx, chunk):
        # Cria um arquivo temporário para o chunk
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
            chunk.export(tmp_file.name, format="mp4", codec="aac")
            tmp_file_path = tmp_file.name

        # Transcreve o chunk
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
                print(f"Transcrição da parte {idx + 1}/{num_chunks} concluída.")
        except Exception as e:
            print(f"Erro na transcrição da parte {idx + 1}: {e}")
            transcricao_total[idx] = ""
        finally:
            # Remove o arquivo temporário
            os.remove(tmp_file_path)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(transcrever_chunk, idx, chunk)
            for idx, chunk in enumerate(audio_chunks)
        ]
        # Opcional: aguarda a conclusão e trata exceções
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                print(f"Uma thread gerou uma exceção: {exc}")

    # Junta todas as transcrições na ordem correta
    transcricao_completa = " ".join(transcricao_total)
    return transcricao_completa


def corrigir_nomes_frameworks(transcricao, client, num_chunks=10, max_workers=5):
    """
    Corrige nomes de frameworks ou bibliotecas na transcrição, dividindo o texto em 'num_chunks' partes,
    processando as correções de forma concorrente.

    :param transcricao: Texto da transcrição.
    :param client: Instância do cliente OpenAI.
    :param num_chunks: Número de partes para dividir o texto.
    :param max_workers: Número máximo de threads a serem usadas.
    :return: Texto corrigido.
    """
    def split_text_in_chunks(text, num_chunks=10):
        """
        Divide a string 'text' em 'num_chunks' partes aproximadamente iguais.
        Retorna uma lista com cada pedaço.
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
    corrected_chunks = ["" for _ in range(num_chunks)]  # Pré-aloca a lista para manter a ordem

    def corrigir_chunk(idx, chunk):
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
            print(f"Parte {idx + 1}/{num_chunks} corrigida.")
        except Exception as e:
            print(f"Erro na correção da parte {idx + 1}: {e}")
            corrected_chunks[idx] = chunk  # Mantém o original em caso de erro

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(corrigir_chunk, idx, chunk)
            for idx, chunk in enumerate(transcricao_chunks)
        ]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                print(f"Uma thread gerou uma exceção: {exc}")

    # Juntar todos os pedaços corrigidos na mesma ordem
    transcricao_corrigida = " ".join(corrected_chunks)
    return transcricao_corrigida


def salvar_texto_em_arquivo(texto, caminho_arquivo):
    """
    Salva o texto fornecido em um arquivo.

    :param texto: Texto a ser salvo.
    :param caminho_arquivo: Caminho para o arquivo de destino.
    """
    with open(caminho_arquivo, "w", encoding="utf-8") as file:
        file.write(texto)


def main():
    audio_file_path = "audio.m4a"  # Caminho para o arquivo de áudio original
    compressed_audio_file_path = "compressed_audio.mp4"  # Caminho para o arquivo de áudio comprimido
    transcricao_file_path = "transcricao.txt"  # Caminho para salvar a transcrição inicial
    output_file_path = "transcricao_corrigida.txt"  # Caminho para salvar a transcrição corrigida
    max_size_mb = 25  # Tamanho máximo permitido para o arquivo de áudio comprimido
    num_chunks = 10  # Número de chunks para dividir o áudio/texto
    max_workers = 5  # Número de threads para concorrência

    api_key = config["OPENAI"]["api_key"]

    # Inicializa o cliente OpenAI com a chave da API
    client = OpenAI(api_key=api_key)

    # Passo 1: Comprimir e converter o áudio, se necessário
    # Descomente se quiser realmente comprimir/converter para MP4.
    # compressed_audio_file_path = compress_and_convert_audio_to_mp4(
    #     audio_file_path, max_size_mb
    # )

    # Passo 2: Transcrever o áudio (usando Whisper da OpenAI) de forma concorrente
    # transcricao = transcrever_audio(
    #     compressed_audio_file_path,
    #     client,
    #     num_chunks=num_chunks,
    #     max_workers=max_workers
    # )
    # print("Transcrição concluída.")

    # Salvar a transcrição em um arquivo
    # salvar_texto_em_arquivo(transcricao, transcricao_file_path)
    # print(f"Transcrição salva em: {transcricao_file_path}")

    with open("transcricao.txt", "r") as file:
        transcricao = file.read()

    # Passo 3: Corrigir nomes de frameworks ou bibliotecas na transcrição de forma concorrente
    transcricao_corrigida = corrigir_nomes_frameworks(
        transcricao,
        client,
        num_chunks=num_chunks,
        max_workers=max_workers
    )
    print("Correções aplicadas na transcrição.")

    # Passo 4: Salvar a transcrição corrigida em um arquivo
    salvar_texto_em_arquivo(transcricao_corrigida, output_file_path)
    print(f"Transcrição corrigida foi salva em: {output_file_path}")


if __name__ == "__main__":
    main()
