from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from preprocess_api_sample.app.services.text_similarity  import dialogue_processor

# Initialize FastAPI
app = FastAPI()

class DialogueRequest(BaseModel):
    prompt: str

@app.post("/find-similar")
async def get_similar_dialogue(request: DialogueRequest):
    """API Endpoint to find the most similar dialogue."""
    if not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    result = dialogue_processor.find_most_similar(request.prompt)
    return {"prompt": request.prompt, "similar_dialogue": result}


# Run the API with: uvicorn main:app --reload
