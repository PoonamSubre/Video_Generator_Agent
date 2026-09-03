import logging
import wave

from google.genai import types

from .. import tingting_brand as brand

logger = logging.getLogger(__name__)


# Gemini TTS returns 24kHz 16-bit mono PCM.
_SAMPLE_RATE = 24000
_SAMPLE_WIDTH = 2
_CHANNELS = 1


def generate_speech(text: str, session_id: str, scene_number: int) -> str | None:
    """
    Generate warm, child-friendly narration audio and save it locally as WAV.

    Args:
        text (str): The narrator line to synthesize.
        session_id (str): The current session ID for output pathing.
        scene_number (int): The current scene number.

    Returns:
        str | None: Local path of the generated WAV file, or None on failure.
    """
    try:
        logger.info(
            f"Synthesizing narration for scene {scene_number}: '{text[:50]}...'"
        )
        response = brand.genai_client().models.generate_content(
            model=brand.TTS_MODEL,
            contents=(
                "Say this in a warm, gentle, cheerful voice for toddlers, "
                f"speaking slowly and clearly: {text}"
            ),
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=brand.TTS_VOICE
                        )
                    )
                ),
            ),
        )

        pcm_data = response.candidates[0].content.parts[0].inline_data.data
        out_path = brand.scene_dir(session_id, scene_number) / "narration.wav"
        with wave.open(str(out_path), "wb") as wf:
            wf.setnchannels(_CHANNELS)
            wf.setsampwidth(_SAMPLE_WIDTH)
            wf.setframerate(_SAMPLE_RATE)
            wf.writeframes(pcm_data)

        logger.info(f"Narration saved to {out_path}")
        return str(out_path)
    except Exception as e:
        logger.error(f"Error generating speech for scene {scene_number}: {e}")
        return None
