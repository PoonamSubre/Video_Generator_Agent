"""Central Ting Ting brand configuration.

Single source of truth for models, local output storage and the brand
style/character text that is programmatically appended to every Imagen and
Veo prompt (so visual consistency never depends on an LLM copying it).
"""

import os
import pathlib

from dotenv import load_dotenv

# All configuration and credentials live in .env (gitignored) - nothing is
# hardcoded. Values below are only last-resort fallbacks if a key is absent.
_PROJECT_ROOT_FOR_ENV = pathlib.Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT_FOR_ENV / ".env")

# --- Models (override in .env without touching code) ---
# Gemini 3.x Flash models are served on the Vertex GLOBAL endpoint
# (GOOGLE_CLOUD_LOCATION=global).
LLM_MODEL = os.getenv("TINGTING_LLM_MODEL", "gemini-3.7-flash")
# Imagen 4 is not enabled on this project; Gemini native image gen is.
IMAGE_MODEL = os.getenv("TINGTING_IMAGE_MODEL", "gemini-2.5-flash-image")
VIDEO_MODEL = os.getenv("TINGTING_VIDEO_MODEL", "veo-3.0-generate-001")
TTS_MODEL = os.getenv("TINGTING_TTS_MODEL", "gemini-2.5-flash-preview-tts")
TTS_VOICE = os.getenv("TINGTING_TTS_VOICE", "Leda")  # warm, youthful

_client = None
_video_client = None


def _is_unusable(client) -> bool:
    """True if the client's underlying HTTP session was closed (this happens
    when adk web hot-reloads the app mid-run)."""
    if client is None:
        return True
    try:
        httpx_client = client._api_client._httpx_client
        return bool(httpx_client is not None and httpx_client.is_closed)
    except Exception:
        return False


def genai_client():
    """Shared lazily-created google-genai client (env must be loaded first).
    Self-healing: recreated if the previous instance was closed."""
    global _client
    if _is_unusable(_client):
        from google import genai

        _client = genai.Client()
    return _client


def video_genai_client():
    """Client used for Veo. If GOOGLE_API_KEY is set, Veo calls go through the
    Gemini API instead of Vertex (useful while the org blocks Veo on Vertex).
    Otherwise a Vertex client pinned to a Veo-serving region, since Veo is
    regional and is NOT available on the global endpoint."""
    global _video_client
    if _is_unusable(_video_client):
        from google import genai

        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key:
            _video_client = genai.Client(vertexai=False, api_key=api_key)
        else:
            _video_client = genai.Client(
                vertexai=True,
                project=os.getenv("GOOGLE_CLOUD_PROJECT"),
                location=os.getenv("TINGTING_VIDEO_LOCATION", "us-central1"),
            )
    return _video_client

# --- Local storage (no GCS) ---
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
OUTPUT_ROOT = pathlib.Path(
    os.getenv("TINGTING_OUTPUT_DIR", str(_PROJECT_ROOT / "output"))
)

# Optional pre-rendered brand assets. If these files exist they are
# prepended/appended to every episode by the post-production stitcher.
BRAND_ASSETS_DIR = _PROJECT_ROOT / "assets" / "brand"
BRAND_INTRO_PATH = BRAND_ASSETS_DIR / "intro.mp4"
BRAND_OUTRO_PATH = BRAND_ASSETS_DIR / "outro.mp4"

# Font used for on-screen number overlays (ffmpeg drawtext).
OVERLAY_FONT = os.getenv(
    "TINGTING_OVERLAY_FONT", "C:/Windows/Fonts/comicbd.ttf"
)


def session_dir(session_id: str) -> pathlib.Path:
    """Return (and create) the local output directory for a session."""
    path = OUTPUT_ROOT / session_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def scene_dir(session_id: str, scene_number: int) -> pathlib.Path:
    """Return (and create) the local output directory for a scene."""
    path = session_dir(session_id) / f"scene_{scene_number}"
    path.mkdir(parents=True, exist_ok=True)
    return path


# --- Ting Ting master visual style (appended to every image/video prompt) ---
MASTER_STYLE = """
STYLE: Original high-quality 3D preschool animation for the Ting Ting brand.
Warm, adorable, polished 3D look with soft rounded shapes, expressive cute
characters, large readable eyes, friendly faces, smooth gentle animation.
Bright cheerful foreground colors on soft pastel backgrounds, no neon.
Soft morning daylight, warm cheerful atmosphere, subtle shadows, premium
children's TV render quality. Colorful magical garden world: flowers, soft
green grass, rounded trees, fluffy clouds, small picnic table, bright blue sky.
Eye-level child perspective, medium shots and close-ups, slow gentle camera,
important objects centered and unobstructed.
Completely original visual identity - do not imitate any existing children's
animation franchise or YouTube channel.
MOOD: joyful, curious, safe, musical, educational, playful.
AVOID: fast cuts, rapid camera movement, flashing lights, chaotic action,
frightening elements, visual clutter, extra background objects.
"""

# --- Character sheet (appended to every image/video prompt) ---
CHARACTER_SHEET = """
CHARACTERS (identical design in every shot, never change proportions, colors,
clothing, eyes or scale):
- TING TING: cute preschool kid character with a personality of a 3-4 year
  old. Round face, large expressive eyes, small button nose, happy warm
  smile, soft rounded body, short brown hair, signature yellow t-shirt with
  a small golden bell logo and blue shorts. Friendly, curious, energetic.
- BOBO: cute friendly light-brown teddy-bear character, small rounded body,
  soft face, large expressive eyes, red neck scarf, playful and curious.
- MIMI: small cute yellow bird chick character, rounded fluffy body, large
  expressive eyes, tiny orange beak, sweet cheerful face, gentle movements.
Keep character scale consistent relative to one another.
"""

# Global continuity line injected into every Veo prompt.
VIDEO_CONTINUITY = """
Same characters with identical design, clothing and colors as the reference
image. Gentle preschool pacing: slow readable movement, characters point at
objects when counting, warm smiles, no fast camera moves.
Cheerful preschool music: soft xylophone, ukulele, gentle piano, soft bells,
light hand percussion. Warm child-friendly voices, clear slow pronunciation.
No aggressive bass, no loud or frightening sound effects.
"""
