import os
import numpy as np
try:
    import librosa
except ImportError:
    pass # Managed by requirements, may not be loaded yet

from pydantic import BaseModel, Field

# FIX: The Pydantic model must be explicitly wrapped in a BaseModel class.
# If these fields were left loose, structured_llm.invoke() would crash with a NameError.
class SongFeatures(BaseModel):
    valence: float = Field(default=0.5, description="Musical positiveness or happiness (0.0 to 1.0)")
    tempo: float = Field(default=120.0, description="Estimated tempo in beats per minute (BPM) (e.g., 80.0 to 180.0)")
    danceability: float = Field(default=0.5, description="Suitability for dancing (0.0 to 1.0)")
    energy: float = Field(default=0.5, description="Intensity and activity (0.0 to 1.0)")
    loudness: float = Field(default=-10.0, description="Overall loudness in decibels (dB), typically -60 to 0")
    speechiness: float = Field(default=0.0, description="Presence of spoken words (0.0 to 1.0)")
    acousticness: float = Field(default=0.0, description="Probability that the track is acoustic (0.0 to 1.0)")
    instrumentalness: float = Field(default=0.0, description="Probability that the track contains no vocals (0.0 to 1.0)")
    liveness: float = Field(default=0.0, description="Probability that the track was performed live (0.0 to 1.0)")
    keywords: list[str] = Field(default_factory=list, description="Extraction of genres (pop, rap, rock, latin, r&b, edm), artist names, or mood keywords found in the prompt.")

def parse_text_to_features(prompt: str, api_key: str) -> dict:
    '''
    Uses LangChain and Groq to parse a user's natural language vibe description
    into numerical Spotify features.
    '''
    from langchain_groq import ChatGroq
    
    try:
        # Initialize Groq LLM
        llm = ChatGroq(
            temperature=0, 
            groq_api_key=api_key, 
            model_name="llama-3.1-8b-instant"
        )
        
        # Bind output structure
        structured_llm = llm.with_structured_output(SongFeatures)
        
        system_instruction = (
            "You are an expert musicologist. The user will describe a vibe, mood, or context "
            "for a song they want to hear. Output the Spotify audio features that best represent this description. "
            "For example, 'high energy workout music' should have high energy, tempo, and danceability. "
            "IMPORTANT: If the user mentions a specific genre (like K-pop, Rock, EDM, Rap, R&B, Latin) or a specific artist, "
            "include them in the 'keywords' list."
        )
        
        result = structured_llm.invoke(f"{system_instruction}\\n\\nUser Description: {prompt}")
        
        # Convert pydantic model to dictionary
        return result.model_dump()
        
    except Exception as e:
        raise Exception(f"Failed to parse text query using Groq: {e}")

def parse_audio_to_features(audio_file_path: str) -> dict:
    '''
    Uses Librosa to extract basic matching features from a humming file.
    Currently extracts Tempo (BPM) and RMS Energy.
    '''
    try:
        # Load audio file (resample to 22050 for speed)
        y, sr = librosa.load(audio_file_path, sr=22050)
        
        # 1. Estimate Tempo
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo_array = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
        tempo = float(tempo_array[0])
        
        # 2. Estimate Energy (Compute RMS, take 90th percentile to ignore silence)
        rms = librosa.feature.rms(y=y)[0]
        energy_raw = float(np.percentile(rms, 90))
        
        # Map raw RMS energy (typically 0.0 to ~0.5 for voice) to [0, 1] range loosely
        # This is a heuristic mapping for humming to Spotify energy bounds
        energy_scaled = min(1.0, max(0.0, energy_raw * 3.0))
        
        return {
            'tempo': tempo,
            'energy': energy_scaled
        }
        
    except Exception as e:
        raise Exception(f"Failed to process audio file: {e}")
