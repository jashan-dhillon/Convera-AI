# import json
# import requests

# class PersonalAgent:
#     def __init__(self, config):
#         self.profession = config['profession']
#         self.skills = config['preferences']['skills']
#         self.tone = config['preferences']['tone']
#         self.experience_level = config['metadata']['experience_level']
#         self.domain = config['metadata']['domain']
#         self.ollama_url = "http://localhost:11434/api/generate"

#     def introduce(self):
#         return f"Hello! I’m your {self.tone} AI assistant for {self.profession}s. I can help with anything related to {self.domain}—just ask!"

#     def handle_query(self, user_input):
#         prompt = (
#             f"You are a {self.tone} AI assistant for a {self.profession} with {self.experience_level} experience. "
#             f"Your skills include {', '.join(self.skills)}. "
#             f"The user has asked: '{user_input}'. "
#             f"Provide a helpful, concise response tailored to their question, staying within the {self.domain} domain."
#         )

#         try:
#             response = requests.post(
#                 self.ollama_url,
#                 json={
#                     "model": "mistral",
#                     "prompt": prompt,
#                     "stream": False
#                 }
#             )
#             response.raise_for_status()
#             return response.json()["response"].strip()
#         except requests.RequestException as e:
#             return f"Oops, something went wrong with the AI: {str(e)}"

# # Load JSON
# try:
#     with open('preferences.json', 'r') as file:
#         config = json.load(file)
# except FileNotFoundError:
#     print("Error: preferences.json not found!")
#     exit(1)
# except json.JSONDecodeError:
#     print("Error: Invalid JSON format in preferences.json!")
#     exit(1)

# agent = PersonalAgent(config)
# print(agent.introduce())

# while True:
#     user_input = input("Ask me anything about software development (or 'exit' to quit): ").strip()
#     if user_input.lower() == 'exit':
#         print("Goodbye!")
#         break
#     response = agent.handle_query(user_input)
#     print(response)



import json
import requests

class PersonalAgent:
    def __init__(self, config):
        self.profession = config['profession']
        self.skills = config['preferences']['skills']
        self.tone = config['preferences']['tone']
        self.experience_level = config['metadata']['experience_level']
        self.domain = config['metadata']['domain']
        self.ollama_url = "http://localhost:11434/api/generate"

    def introduce(self):
        return f"Hello! I’m your {self.tone} AI assistant for {self.profession}s. I can help with anything related to {self.domain}—just ask!"

    def handle_query(self, user_input):
        prompt = (
            f"You are a {self.tone} AI assistant for a {self.profession} with {self.experience_level} experience. "
            f"Your skills include {', '.join(self.skills)}. "
            f"The user has asked: '{user_input}'. "
            f"Provide a helpful, concise response tailored to their question, staying within the {self.domain} domain."
        )
        try:
            response = requests.post(
                self.ollama_url,
                json={"model": "mistral", "prompt": prompt, "stream": False}
            )
            response.raise_for_status()
            return response.json()["response"].strip()
        except requests.RequestException as e:
            return f"Oops, something went wrong with the AI: {str(e)}"

def load_config():
    try:
        with open('preferences.json', 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        raise Exception("Error: preferences.json not found!")
    except json.JSONDecodeError:
        raise Exception("Error: Invalid JSON format in preferences.json!")