from pydantic import BaseModel, Field

class SongFeatures(BaseModel):
    """
    Structured schema for LLM response mapping natural language to Spotify audio features.
    Used for the 'Vibe Search' feature.
    """
    valence: float = Field(description="Musical positiveness or happiness (0.0 to 1.0)")
    tempo: float = Field(description="Estimated tempo in beats per minute (BPM) (e.g., 80.0 to 180.0)")
    keywords: list[str] = Field(description="Extraction of genres, artist names, or mood keywords found in the prompt.")

def parse_text_to_features(prompt: str, api_key: str) -> dict:
    """
    Translates user's descriptions into numerical audio features using Groq LLM.
    Leverages Llama-3.1 for zero-shot 'Vibe to Feature' mapping.
    """
    from langchain_groq import ChatGroq
    
    try:
        # Initialize Groq LLM with structured output binding
        llm = ChatGroq(
            temperature=0, 
            groq_api_key=api_key, 
            model_name="llama-3.1-8b-instant"
        )
        
        structured_llm = llm.with_structured_output(SongFeatures)
        
        system_instruction = (
            "You are an expert musicologist. Output Spotify audio features (0.0-1.0) and BPM "
            "that best match the user's mood description."
        )
        
        result = structured_llm.invoke(f"{system_instruction}\n\nUser Description: {prompt}")
        return result.model_dump()
        
    except Exception as e:
        raise Exception(f"Failed to parse text query using Groq: {e}")
