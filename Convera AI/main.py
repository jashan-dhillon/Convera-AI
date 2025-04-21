from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from agent import PersonalAgent, load_config
from fastapi.middleware.cors import CORSMiddleware

# Define app only once
app = FastAPI()

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production (e.g., ["https://your-convera-ai-web.netlify.app"])
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up templates
templates = Jinja2Templates(directory="templates")

# Load config and initialize agent
config = load_config()
agent = PersonalAgent(config)

# Derive agent name based on profession
agent_name = f"{config['profession'].capitalize().replace(' ', '')}Buddy"  # e.g., "SoftwareDeveloperBuddy"
profession = config['profession']

@app.get("/convera", response_class=HTMLResponse)
async def convera(request: Request):
    return templates.TemplateResponse("convera.html", {"request": request})

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    intro = agent.introduce()
    return templates.TemplateResponse("index.html", {
        "request": request,
        "intro": intro,
        "agent_name": agent_name,
        "profession": profession
    })

class Query(BaseModel):
    message: str

@app.post("/query")
async def query(query: Query):
    user_input = query.message
    if user_input.lower() == 'exit':
        return {"response": "Goodbye!"}
    response = agent.handle_query(user_input)
    return {"response": response}