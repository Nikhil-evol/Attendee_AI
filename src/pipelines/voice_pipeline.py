import io

import numpy as np
import librosa
import streamlit as st


def _load_audio_bytes(audio_bytes, sr=16000):
    if not audio_bytes:
        return None, None

    try:
        audio, _ = librosa.load(io.BytesIO(audio_bytes), sr=sr, mono=True)
        if audio.size == 0:
            return None, None
        return audio, sr
    except Exception:
        return None, None


def _normalize_vector(vector):
    vec = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(vec)
    return vec / (norm + 1e-9)


def _audio_embedding(audio):
    if audio is None or audio.size == 0:
        return None

    mel = librosa.feature.melspectrogram(y=audio, sr=16000, n_fft=1024, hop_length=512, n_mels=128)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    embedding = np.mean(log_mel, axis=1)
    return _normalize_vector(embedding)


def get_voice_embedding(audio_bytes):
    try:
        encoder = load_voice_encoder()

        audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)
        wav = preprocess_wav(audio)
        embedding = encoder.embed_utterance(wav)
        return embedding.tolist()
    except Exception as e:
        st.error('Voice recog error')
        return None
    

def identify_speaker(new_embedding, candidates_dict, threshold=0.65):
    if new_embedding is None or not candidates_dict:
        return None, 0.0
    
    best_sid = None
    best_score = -1.0

    for sid, stored_embedding in candidates_dict.items():
        if stored_embedding:
            similarity = np.dot(new_embedding, stored_embedding)
            if similarity> best_score:
                best_score = similarity
                best_sid = sid

    if best_score >= threshold:
        return best_sid, best_score
    
    return None, best_score



def process_bulk_audio(audio_bytes, candidates_dict, threshold=0.65):

    try:
        encoder = load_voice_encoder()

        audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)
        segments = librosa.effects.split(audio, top_db=30)

        identified_results = {}


        for start, end in segments:

            if (end-start) < sr * 0.5:
                continue
            segment_audio = audio[start:end]
            wav = preprocess_wav(segment_audio)
            embedding = encoder.embed_utterance(wav)


            sid, score = identify_speaker(embedding, candidates_dict, threshold)

            if sid:
                if sid not in identified_results or score > identified_results[sid]:
                    identified_results[sid] = score

        return identified_results
    except Exception as e:
        st.error('Bulk process error')
        return {}