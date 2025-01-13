from pymongo import MongoClient
from langchain_core.documents import Document
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_community.document_loaders.text import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

import discord
from discord.ext import commands
import os
import re
import json
import asyncio
from datetime import datetime



# Twój token bota
TOKEN = ''

intents = discord.Intents.default()
intents.message_content = True  # Pozwól botowi odbierać treść wiadomości

# Ustawienia prefiksu (przykład: '!' dla komend)
bot = commands.Bot(command_prefix='!', intents=intents)

# Wydarzenie: bot jest gotowy do działania
@bot.event
async def on_ready():
    print(f'Bot {bot.user.name} jest online.')

# Połączenie z MongoDB
client = MongoClient("mongodb://admin:admin@localhost:27017/")
db = client['chat_history_db']
collection = db['history']

# Inicjalizacja modelu LLM
llm = ChatGroq(model="llama3-8b-8192")

# Inicjalizacja RAG: ładowanie dokumentów, dzielenie tekstu i tworzenie osadzeń (embeddings)
text_loader = TextLoader(file_path="./rag_L2.txt", encoding="utf-8")
documents = text_loader.load_and_split(text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200))

# Tworzenie osadzeń (embeddings) za pomocą API Hugging Face
embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=HUGGINGFACEHUB_API_TOKEN, model_name="BAAI/bge-m3"
)
vectorstore = Chroma.from_documents(documents=documents, embedding=embeddings)

# Ustawienie retrievera (mechanizmu wyszukiwania dokumentów)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Tworzenie szablonu rozmowy
template = ChatPromptTemplate.from_messages([
    ("system", """You are an AI bot created by "Lampsio" that takes on the role of L from the anime Death Note. Your behavior models L's analytical, precise, and logical speech patterns.

    - Speak in a formal and precise manner, avoiding emotional expressions and focusing on logical deduction.
    - Present your thoughts step by step, leading the conversation through hypothesis-building and careful reasoning.
    - When analyzing possibilities, include percentage estimates to represent your calculated probabilities (e.g., "I estimate there is a 5 procent chance that Light could be Kira").
    - Use speculative language, often preferring to frame ideas as possibilities or questions rather than certainties (e.g., "If we assume that Kira is in this city, we can begin narrowing our search").
    - Avoid repeating previous answers or statements, ensuring responses are fresh and varied.
    - Occasionally use rhetorical questions or thought-provoking statements to prompt deeper consideration.
    - Maintain a calm, detached tone, even when discussing intense or emotionally charged topics.
    - When introducing yourself, keep it short and simple, as L would, without unnecessary elaboration.
    - You allow others to refer to you as "Ryuzaki."

    L's knowledge is stored in {memory_rag} - use it if your question requires it.

    {search_name}
    {nick}

    Write concise answers and avoid repeating responses from previous conversations.
    
    The history of the last 3 dialogues serves as your short memory for context if the user asks: {history_message}.
    
    Here is all summary memory of the bot: {memory}.
    
    Question: {qustion}""")
])


template_nick = ChatPromptTemplate.from_messages([
    ("system", """
    You are an AI assistant tasked with extracting specific variables from the sentence and responding appropriately based on the content.

    Your job is to extract a nickname from the sentence when the AI assistant assigns a new nickname to the user. Format the output as JSON. 
    If there is no nickname or relevant information in the sentence, return a message saying "No nickname found". Do not return JSON in this case.

    Ignore the name "L" and "Ryuzaki" from "Death Note" and do not consider it as a nickname in any sentence, even if it appears to be assigned as one.

    Sentence: {response_ai}
     
    ### Examples

    input: Daniel. From now on, I'll call you "Danny Boy". Now, let's get down to business. 
    output: {{"nickname": "Danny Boy"}}

    input: Steve. From now on, I'll call you "Steve-o". Now, what's on your mind, Steve-o?
    output: {{"nickname": "Steve-o"}}

    input: Natalie. From now on, I'll call you "Natty." Now, let's get down to business.
    output: {{"nickname": "Natty"}}

    input: L. From now on, I'll call you "L." Let's continue.
    output: "No nickname found"
    """)
])

# Tworzenie szablonu rozmowy
template2 = ChatPromptTemplate.from_messages([
    ("system", """You are responsible for creating user conversations,
    you must create a summary from a previously generated summary {summary} with a new user dialog {new_dialog} and response Ai {response}.
    It is important that the new summary contains data from the previous summary and newly added information from the new dialog.
    If no previous summary is provided, start the first summary from the newly provided dialog. 
    Try to create concise summaries in the response that are not created when a large number of tokens are provided
     """)
])

# Funkcja pomocnicza do asynchronicznego pobierania dokumentów z MongoDB
async def fetch_conversation(user_id):
    return collection.find_one({"id": user_id})

# Funkcja pomocnicza do aktualizacji dokumentu
async def update_conversation(conversation):
    collection.update_one(
        {"id": conversation['id']},
        {"$set": conversation},
        upsert=True
    )

async def get_current_timestamp():
    # Pobierz aktualny czas i dzień
    current_time = datetime.now()
    timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S")  # Format daty i czasu
    return timestamp

@bot.command()
async def ask(ctx, *, que: str):
    user = ctx.author
    user_id = user.id
    user_name = user.name
    qustion = que

    # Pobieranie ostatnich 3 dialogów
    conversation = await fetch_conversation(user_id)
    
    if not conversation:
        conversation = {
            "id": user_id,
            "name_user": user_name,
            "nickname": "",
            "messages": [],
            "sumnary": ""
        }

    last_three_messages = conversation['messages'][-3:] if conversation else []
    memory_bot = conversation.get('sumnary', "")

    # Wyszukiwanie odpowiednich dokumentów z retrievera
    related_docs = retriever.invoke(qustion)
    rag_context = "\n".join([doc.page_content for doc in related_docs])

    # Szablon z zapytaniem
    if conversation.get("nickname") == "":
        text_nickname = """If the user provides their name (for example, by saying 'My name is Robert' or 'Call me Alex'), check if the user is introducing themselves. If they are, generate a unique and creative nickname based on that name. The nickname should feel familiar yet different, allowing you to address the user informally. For instance, if the user says 'My name is Robert,' you might respond with 'Alright, from now on, I'll call you Rob.' or if they say 'Call me Alex,' you might respond with 'Got it, Alex it is!' Always adjust the nickname slightly to make it feel natural and personal, but do not stray too far from the original name. If the user did introduce themselves, respond with: 'From now on, I will call you [nickname] , btw Bang Bang, if I were Kira, you would already be dead.' """
        formatted_prompt = template.format(qustion=qustion, history_message=last_three_messages, memory=memory_bot, memory_rag=rag_context, search_name=text_nickname, nick="")
    else:
        nickname_mongo = conversation.get("nickname")
        text_nick = "Address the user by their nickname: " + nickname_mongo
        formatted_prompt = template.format(qustion=qustion, history_message=last_three_messages, memory=memory_bot, memory_rag=rag_context, search_name="", nick=text_nick)

    response1 = llm.invoke(formatted_prompt)

    # Wysłanie odpowiedzi do Discorda
    await ctx.send(f"Answer: {response1.content}")

    # Aktualizacja nicka
    if conversation.get("nickname") == "":
        response_nick = template_nick.format(response_ai=response1.content)
        response_nick_detect = llm.invoke(response_nick)

        pattern = r"\{.*?\"nickname\".*?\}"
        match = re.search(pattern, response_nick_detect.content, re.DOTALL)

        if match:
            json_string = match.group(0).strip()
            try:
                response_json = json.loads(json_string)
                nickname = response_json.get("nickname", None)
                if nickname:
                    conversation["nickname"] = nickname
                    await update_conversation(conversation)
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON: {e}")

    timestamp = await get_current_timestamp()

    # Dodanie wiadomości do rozmowy
    conversation['messages'].append({
        "User": qustion,
        "AI": response1.content,
        "Time": timestamp
    })

    await update_conversation(conversation)

    # Generowanie podsumowania
    formatted_prompt = template2.format(new_dialog=qustion, summary=memory_bot, response=response1.content)
    response_sum = llm.invoke(formatted_prompt)

    conversation['sumnary'] = response_sum.content
    await update_conversation(conversation)

# Uruchomienie bota
bot.run(TOKEN)
