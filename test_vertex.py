import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

def test_connectivity():
    print(f"Project: {os.getenv('GOOGLE_CLOUD_PROJECT')}")
    print(f"Location: {os.getenv('GOOGLE_CLOUD_LOCATION')}")
    print("Testing Vertex AI connectivity...")
    client = genai.Client(
        vertexai=True,
        project=os.getenv("GOOGLE_CLOUD_PROJECT"),
        location=os.getenv("GOOGLE_CLOUD_LOCATION")
    )
    
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents="test"
        )
        print("Success! Response text:", response.text)
    except Exception as e:
        print("Failure!")
        print(f"Error type: {type(e)}")
        print(f"Error: {e}")

if __name__ == "__main__":
    test_connectivity()
