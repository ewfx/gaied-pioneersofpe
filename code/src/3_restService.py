import os
import email
import quopri
import PyPDF2
from typing import List, Tuple, Optional
from transformers import pipeline, AutoModel, AutoTokenizer
from pydantic import BaseModel, Field, ValidationError
import json
from bs4 import BeautifulSoup
import torch
from PIL import Image
import pytesseract
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
import tempfile
from typing import List, Dict
from pydantic import BaseModel
from typing import Optional
import shutil
import logging
import faiss  # Import Faiss
import uvicorn  # Import uvicorn
from contextlib import asynccontextmanager # Import lifespan
import requests # For making API calls to Gemini
from typing import Any
from dotenv import load_dotenv

load_dotenv(override=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler to initialize and cleanup Faiss.
    """
    initialize_faiss()
    yield
    # Here you would typically clean up resources, but Faiss doesn't require explicit cleanup.
    # If you had other resources (e.g., connection pools), you'd close them here.
    logger.info("Application shutdown: Faiss index saved.") # Added log
    if faiss_index:
        faiss.write_index(faiss_index, faiss_index_path)

app = FastAPI(lifespan=lifespan) # Register lifespan


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load pre-trained models
try:
    embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_model = AutoModel.from_pretrained(embedding_model_name)
    embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
    image_classification_model = pipeline("image-classification", model="google/vit-base-patch16-224")
    logger.info(f"Using embedding model: {embedding_model_name}")
except Exception as e:
    msg = f"Error loading models: {e}"
    logger.error(msg)
    raise

# Global Faiss index and dimension
faiss_index: faiss.Index = None
embedding_dimension = 384  # For all-MiniLM-L6-v2, adjust if you change the model
faiss_index_path = "C:/Users/venka/OneDrive/Desktop/Hackathon/Data/db/faiss_index.bin"  # Path to save/load Faiss indexn
#os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'path/to/your/google_credentials.json' #set this

# Environment variables for API keys
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
#GROQ_REASONING_MODEL = 'deepseek-r1-distill-llama-70b' #os.environ.get("GROQ_REASONING_MODEL", "default-model")  # Default value provided


def initialize_faiss():
    """Initializes the Faiss index."""
    global faiss_index
    try:
        if os.path.exists(faiss_index_path):
            faiss_index = faiss.read_index(faiss_index_path)
            logger.info(f"Faiss index loaded from {faiss_index_path}")
        else:
            faiss_index = faiss.IndexFlatL2(embedding_dimension)
            logger.info("Faiss index initialized")
    except Exception as e:
        msg = f"Error initializing Faiss index: {e}"
        logger.error(msg)
        raise



# Pydantic Model for Classification Result
class ClassificationResult(BaseModel):
    file_path: str = Field(..., description="Path to the input file")
    file_type: str = Field(..., description="Type of the input file (e.g., eml, pdf, jpeg)")
    subject: Optional[str] = Field(None, description="Subject of the email (if applicable)")
    content: Optional[str] = Field(
        None, description="Extracted text content from the file (if applicable)"
    )
    request_type: str = Field(..., description="Type of request")
    sub_request_type: str = Field(..., description="Sub-type of request")
    summary: str = Field(..., description="A brief summary of the content")
    confidence_score: float = Field(
        ..., description="Confidence score of the classification"
    )
    status: str = Field(..., description="Status of processing (e.g., success, duplicate, error)")
    message: str = Field(..., description="Descriptive message about the processing status")
    attachments: Optional[List[str]] = Field(
        None, description="List of extracted attachment file paths (if applicable)"
    )



# File processing functions (same as before)
def extract_text_from_email(
    email_path: str,
) -> Tuple[str, List[str], Optional[str]]:
    text = ""
    attachments = []
    subject = None
    try:
        with open(email_path, "rb") as f:
            msg = email.message_from_binary_file(f)
        subject = msg.get("Subject")
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = part.get("Content-Disposition")
                if content_type == "text/plain":
                    charset = part.get_content_charset()
                    try:
                        text += part.get_payload(decode=True).decode(
                            charset or "utf-8", errors="ignore"
                        )
                    except Exception as e:
                        msg = f"Error decoding text part: {e}, charset: {charset}"
                        logger.error(msg)
                        text += part.get_payload(decode=True).decode(
                            "utf-8", errors="ignore"
                        )
                elif content_type == "text/html":
                    charset = part.get_content_charset()
                    try:
                        html_text = part.get_payload(decode=True).decode(
                            charset or "utf-8", errors="ignore"
                        )
                        soup = BeautifulSoup(html_text, "html.parser")
                        text += soup.get_text()
                    except Exception as e:
                        msg = f"Error decoding html part: {e}, charset: {charset}"
                        logger.error(msg)
                        text += part.get_payload(decode=True).decode(
                            "utf-8", errors="ignore"
                        )
                elif content_disposition:
                    filename = part.get_filename()
                    if filename:
                        filename = quopri.decodestring(filename).decode(
                            "utf-8", errors="ignore"
                        )
                        attachment_path = os.path.join("attachments", filename)
                        if not os.path.exists("attachments"):
                            os.makedirs("attachments")
                        try:
                            with open(attachment_path, "wb") as attachment_file:
                                attachment_file.write(part.get_payload(decode=True))
                            attachments.append(attachment_path)
                        except Exception as e:
                            msg = f"Error saving attachment {filename}: {e}"
                            logger.error(msg)
                # Check if the attachment is an image and extract text using OCR
                elif content_type.startswith("image"):
                    filename = part.get_filename()
                    if filename:
                        attachment_path = os.path.join("attachments", filename)
                        if not os.path.exists("attachments"):
                            os.makedirs("attachments")
                        try:
                            with open(attachment_path, "wb") as attachment_file:
                                attachment_file.write(part.get_payload(decode=True))
                            extracted_text = extract_text_from_image(attachment_path)
                            text += f"Image Text: {extracted_text}"  # Add extracted text
                            attachments.append(attachment_path)
                        except Exception as e:
                            msg = f"Error processing image attachment {filename}: {e}"
                            logger.error(msg)
    except Exception as e:
        msg = f"Error processing email: {e}"
        logger.error(msg)
        raise
    return text, attachments, subject


def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() or ""
    except Exception as e:
        msg = f"Error processing PDF: {e}"
        logger.error(msg)
        raise
    return text



# Use the data provided for more accurate classification
training_data = [
    {
        "request_type": "Adjustment",
        "sub_request_types": [
            "AU Transfer",
            "Closing Notice",
            "Reallocation Fees",
            "Amendment Fees",
            "Reallocation Principal"
        ],
        "description": "Requests related to adjustments to loan accounts",
        "keywords": ["adjust", "transfer", "reallocate", "amend", "close"]
    },
    {
        "request_type": "Commitment Change",
        "sub_request_types": [
            "Cashless Roll",
            "Decrease",
            "Increase"
        ],
        "description": "Requests to modify loan commitment amounts",
        "keywords": ["commitment", "increase", "decrease", "roll", "change"]
    },
    {
        "request_type": "Fee Payment",
        "sub_request_types": [
            "Ongoing Fee",
            "Letter of Credit Fee"
        ],
        "description": "Requests related to fee payments",
        "keywords": ["fee", "payment", "charge", "credit fee"]
    },
    {
        "request_type": "Money Movement-Inbound",
        "sub_request_types": [
            "Principal",
            "Interest",
            "Principal + Interest",
            "Principal+Interest+Fee"
        ],
        "description": "Requests for incoming fund transfers",
        "keywords": ["deposit", "transfer in", "funding", "inbound", "payment"]
    },
    {
        "request_type": "Money Movement-Outbound",
        "sub_request_types": [
            "Timebound",
            "Foreign Currency"
        ],
        "description": "Requests for outgoing fund transfers",
        "keywords": ["withdraw", "transfer out", "outbound", "payment out"]
    }
]

def classify_text(text: str) -> Tuple[str, str, float]:
    """
    Classifies the input text using a combination of methods, including Gemini and DeepSeek.

    Args:
        text (str): The text to classify.

    Returns:
        Tuple[str, str, float]: A tuple containing the request type, sub-request type, and confidence score.
    """
    try:
        # 1. Keyword and Cosine Similarity (Baseline)
        best_match_key = None
        best_match_score = 0.0
        for data_item in training_data:
            for keyword in data_item["keywords"]:
                query_embedding = get_embedding(text).flatten()
                keyword_embedding = get_embedding(keyword).flatten()
                similarity = cosine_similarity(query_embedding.unsqueeze(0), keyword_embedding.unsqueeze(0))[0, 0]
                if similarity > best_match_score:
                    best_match_score = similarity
                    best_match_key = data_item["request_type"]

        baseline_request_type = "Unknown"
        baseline_sub_request_type = "Unknown"
        baseline_confidence_score = 0.0

        if best_match_key and best_match_score > 0.6:  # Threshold for baseline
            for data_item in training_data:
                if data_item["request_type"] == best_match_key:
                    baseline_request_type = data_item["request_type"]
                    baseline_sub_request_type = data_item["sub_request_types"][0]
                    baseline_confidence_score = best_match_score

        # 2. Gemini Classification
        gemini_request_type, gemini_sub_request_type, gemini_confidence_score = classify_with_gemini(text)

        # 3. DeepSeek Classification
        openai_request_type, openai_sub_request_type, openai_confidence_score = classify_with_openai(text) 





        # 4. Combine Results (Voting with Confidence)
        results = [
            ("baseline", baseline_request_type, baseline_sub_request_type, baseline_confidence_score),
            ("gemini", gemini_request_type, gemini_sub_request_type, gemini_confidence_score),
            ("openai", openai_request_type, openai_sub_request_type, openai_confidence_score), 
        ]

        # Determine the final classification based on the highest confidence
        best_result = max(results, key=lambda x: x[3])  # Get result with max confidence
        final_request_type = best_result[1]
        final_sub_request_type = best_result[2]
        final_confidence_score = best_result[3]

        logger.info(f"Baseline: {baseline_request_type}, {baseline_confidence_score}, Gemini: {gemini_request_type}, {gemini_confidence_score}, OpenAI: {openai_request_type}, {openai_confidence_score}, Final: {final_request_type}, {final_confidence_score}") 
        return final_request_type, final_sub_request_type, final_confidence_score

    except Exception as e:
        msg = f"Error classifying text: {e}"
        logger.error(msg)
        return "Error", "Error", 0.0



#def classify_with_gemini(text: str) -> Tuple[str, str, float]:
    """
    Classifies the text using the Gemini model.

    Args:
        text (str): The text to classify.

    Returns:
        Tuple[str, str, float]: request type, sub-request type, confidence.
    """
    if not GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY is not set. Skipping Gemini classification.")
        return "Unknown", "Unknown", 0.0

    gemini_prompt = f"""
    Classify the following text into one of the following request types and sub-request types.
    Text: {text}
    
    Available Request Types:
    {', '.join(set(item['request_type'] for item in training_data))}
    
    For each Request Type, here are the Sub-Request Types:
    {json.dumps({item['request_type']: item['sub_request_types'] for item in training_data}, indent=2)}
    
    Respond with a JSON object containing "request_type", "sub_request_type", and "confidence" (0.0 to 1.0).
    Ensure the response is a valid JSON object. Do not include any additional text or formatting.
    Example Response:
    {{
        "request_type": "Adjustment",
        "sub_request_type": "AU Transfer",
        "confidence": 0.95
    }}
    """
    try:
        response = requests.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
            headers={"Content-Type": "application/json"},
            params={"key": GEMINI_API_KEY},
            json={
                "contents": [{"parts": [{"text": gemini_prompt}]}],
                "generationConfig": {
                    "temperature": 0.1,  # Lower temperature for more deterministic output
                    "maxOutputTokens": 256
                }
            }
        )

        if response.status_code == 200:
            gemini_response = response.json()
            logger.info(f"Gemini API response: {gemini_response}")
            candidates = gemini_response.get("candidates", [])
            if candidates:
                content = candidates[0].get("content", {})
                parts = content.get("parts", [])
                if parts:
                    text_response = parts[0].get("text", "")
                    # Attempt to parse the response as JSON
                    try:
                        json_response = json.loads(text_response)
                        request_type = json_response.get("request_type", "Unknown")
                        sub_request_type = json_response.get("sub_request_type", "Unknown")
                        confidence = float(json_response.get("confidence", 0.0))
                        return request_type, sub_request_type, confidence
                    except json.JSONDecodeError as e:
                        logger.error(f"Gemini response was not valid JSON: {text_response}. Error: {e}")
                        # Attempt to extract JSON from the response (if it's embedded in text)
                        try:
                            start_index = text_response.find('{')
                            end_index = text_response.rfind('}')
                            if start_index != -1 and end_index != -1:
                                json_str = text_response[start_index:end_index + 1]
                                json_response = json.loads(json_str)
                                request_type = json_response.get("request_type", "Unknown")
                                sub_request_type = json_response.get("sub_request_type", "Unknown")
                                confidence = float(json_response.get("confidence", 0.0))
                                logger.info(f"Extracted JSON: {json_response} from Gemini response.")
                                return request_type, sub_request_type, confidence
                            else:
                                logger.warning("Could not extract valid JSON from Gemini response.")
                                return "Unknown", "Unknown", 0.0
                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to extract and parse JSON from Gemini response: {text_response}. Error: {e}")
                            return "Unknown", "Unknown", 0.0
                else:
                    logger.warning("Gemini response had no parts.")
                    return "Unknown", "Unknown", 0.0
            else:
                logger.warning("Gemini response had no candidates.")
                return "Unknown", "Unknown", 0.0
        else:
            logger.error(f"Gemini API error: {response.status_code}, {response.text}")
            return "Unknown", "Unknown", 0.0
    except Exception as e:
        logger.error(f"Error calling Gemini API: {e}")
        return "Unknown", "Unknown", 0.0

def classify_with_gemini(text: str) -> Tuple[str, str, float]:
    """
    Classifies the text using the Gemini model.

    Args:
        text (str): The text to classify.

    Returns:
        Tuple[str, str, float]: request type, sub-request type, confidence.
    """
    if not GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY is not set. Skipping Gemini classification.")
        return "Unknown", "Unknown", 0.0

    gemini_prompt = f"""
    Classify the following text into one of the following request types and sub-request types.
    Text: {text}
    
    Available Request Types:
    {', '.join(set(item['request_type'] for item in training_data))}
    
    For each Request Type, here are the Sub-Request Types:
    {json.dumps({item['request_type']: item['sub_request_types'] for item in training_data}, indent=2)}
    
    Respond with a JSON object containing "request_type", "sub_request_type", and "confidence" (0.0 to 1.0).
    Ensure the response is a valid JSON object. Do not include any additional text or formatting.
    Example Response:
    {{
        "request_type": "Adjustment",
        "sub_request_type": "AU Transfer",
        "confidence": 0.95
    }}
    """
    try:
        response = requests.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent", 
            headers={"Content-Type": "application/json"},
            params={"key": GEMINI_API_KEY},
            json={
                "contents": [{"parts": [{"text": gemini_prompt}]}],
                "generationConfig": {
                    "temperature": 0.1,  # Lower temperature for more deterministic output
                    "maxOutputTokens": 256
                }
            }
        )

        if response.status_code == 200:
            gemini_response = response.json()
            logger.info(f"Gemini API response: {gemini_response}")
            candidates = gemini_response.get("candidates", [])
            if candidates:
                content = candidates[0].get("content", {})
                parts = content.get("parts", [])
                if parts:
                    text_response = parts[0].get("text", "")
                    # Attempt to parse the response as JSON
                    try:
                        json_response = json.loads(text_response)
                        request_type = json_response.get("request_type", "Unknown")
                        sub_request_type = json_response.get("sub_request_type", "Unknown")
                        confidence = float(json_response.get("confidence", 0.0))
                        return request_type, sub_request_type, confidence
                    except json.JSONDecodeError as e:
                       # logger.error(f"Gemini response was not valid JSON: {text_response}. Error: {e}")
                        # Attempt to extract JSON from the response (if it's embedded in text)
                        try:
                            start_index = text_response.find('{')
                            end_index = text_response.rfind('}')
                            if start_index != -1 and end_index != -1:
                                json_str = text_response[start_index:end_index + 1]
                                json_response = json.loads(json_str)
                                request_type = json_response.get("request_type", "Unknown")
                                sub_request_type = json_response.get("sub_request_type", "Unknown")
                                confidence = float(json_response.get("confidence", 0.0))
                                #logger.info(f"Extracted JSON: {json_response} from Gemini response.")
                                return request_type, sub_request_type, confidence
                            else:
                                #logger.warning("Could not extract valid JSON from Gemini response.")
                                return "Unknown", "Unknown", 0.0
                        except json.JSONDecodeError as e:
                            #logger.error(f"Failed to extract and parse JSON from Gemini response: {text_response}. Error: {e}")
                            return "Unknown", "Unknown", 0.0
                else:
                    logger.warning("Gemini response had no parts.")
                    return "Unknown", "Unknown", 0.0
            else:
                logger.warning("Gemini response had no candidates.")
                return "Unknown", "Unknown", 0.0
        else:
            logger.error(f"Gemini API error: {response.status_code}, {response.text}")
            return "Unknown", "Unknown", 0.0
    except Exception as e:
        logger.error(f"Error calling Gemini API: {e}")
        return "Unknown", "Unknown", 0.0


def classify_with_openai(text: str) -> Tuple[str, str, float]:
    """
    Classifies the text using the OpenAI API.

    Args:
        text (str): The text to classify.

    Returns:
        Tuple[str, str, float]:  request type, sub-request type, confidence.
    """
    if not OPENAI_API_KEY:
        logger.warning("OPENAI_API_KEY is not set. Skipping OpenAI classification.")
        return "Unknown", "Unknown", 0.0

    openai_prompt = f"""
    Classify the following text into one of the available request types and sub-request types.
    Text: {text}

    Available Request Types:
    {', '.join(set(item['request_type'] for item in training_data))}

    For each Request Type, here are the Sub-Request Types:
    {json.dumps({item['request_type']: item['sub_request_types'] for item in training_data}, indent=2)}
    
    Respond with a JSON object containing "request_type", "sub_request_type", and "confidence" (0.0 to 1.0).
     Ensure the response is a valid JSON object. Do not include any additional text or formatting.
    Example Response:
    {{
        "request_type": "Adjustment",
        "sub_request_type": "AU Transfer",
        "confidence": 0.92
    }}
    """
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": 'gpt-4o-mini',
                "messages": [
                    {"role": "user", "content": openai_prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 256,
            },
        )
        if response.status_code == 200:
            openai_response = response.json()
            choices = openai_response.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                content = message.get("content", "")
                try:
                    json_response = json.loads(content)
                    request_type = json_response.get("request_type", "Unknown")
                    sub_request_type = json_response.get("sub_request_type", "Unknown")
                    confidence = float(json_response.get("confidence", 0.0))
                    return request_type, sub_request_type, confidence
                except json.JSONDecodeError as e:
                    logger.error(f"OpenAI response was not valid JSON: {content}. Error: {e}")
                    return "Unknown", "Unknown", 0.0
            else:
                logger.warning("OpenAI response had no choices.")
                return "Unknown", "Unknown", 0.0
        else:
            logger.error(f"OpenAI API error: {response.status_code}, {response.text}")
            return "Unknown", "Unknown", 0.0
    except Exception as e:
        logger.error(f"Error calling OpenAI API: {e}")
        return "Unknown", "Unknown", 0.0

def get_embedding(text: str) -> torch.Tensor:
    inputs = embedding_tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = embedding_model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :]
    return embeddings



def is_duplicate_request(text: str, threshold: float = 0.8) -> bool:
    """
    Checks if the given text is a duplicate of any text already processed using Faiss.

    Args:
        text (str): The text to check for duplication.
        threshold (float, optional): The similarity threshold. Defaults to 0.8.

    Returns:
        bool: True if the text is a duplicate, False otherwise.
    """
    global faiss_index
    if faiss_index is None:
        return False  # No vectors in the database yet

    try:
        text_embedding = get_embedding(text).numpy()
        # Ensure the embedding is 2D
        if text_embedding.ndim == 1:
            text_embedding = text_embedding.reshape(1, -1)

        D, I = faiss_index.search(text_embedding, 1)  # Search for the nearest neighbor
        if len(D) > 0 and len(D[0]) > 0:
            similarity = 1 - D[0][0]  # Convert L2 distance to similarity
            logger.info(f"Similarity: {similarity}")
            return similarity > threshold
        else:
            return False  # Return false if no results are found

    except Exception as e:
        msg = f"Error checking for duplicates: {e}"
        logger.error(msg)
        return False  # Return False in case of an error



def add_to_faiss(embedding: torch.Tensor, file_path: str):
    """Adds the embedding to the Faiss index."""
    global faiss_index
    if faiss_index is None:
        raise ValueError("Faiss index not initialized.")

    try:
        embedding_np = embedding.numpy()
        # Ensure the embedding is 2D
        if embedding_np.ndim == 1:
            embedding_np = embedding_np.reshape(1, -1)
        faiss_index.add(embedding_np)
        faiss.write_index(faiss_index, faiss_index_path)  # Save index
        logger.info(f"Added embedding for {file_path} to Faiss index and saved to {faiss_index_path}")
    except Exception as e:
        msg = f"Error adding to Faiss index: {e}"
        logger.error(msg)
        raise



def create_summary(text: str) -> str:
    try:
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        summary = summarizer(
            text, max_length=130, min_length=30, do_sample=False
        )[0]["summary_text"]
        return summary
    except Exception as e:
        msg = f"Error creating summary: {e}"
        logger.error(msg)
        return "Summary generation failed."


def extract_text_from_image(image_path: str) -> str:
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        logger.info(
            f"content from image is ************************************\n{text}"
        )
        return text
    except Exception as e:
        msg = f"Error extracting text from image: {e}"
        logger.error(msg)
        return ""



def process_file(file_path: str) -> ClassificationResult:
    file_type = os.path.splitext(file_path)[1].lower()
    text = ""
    attachments = []
    subject = None
    try:
        logger.info(f"Processing file: {file_path}")
        if file_type == ".eml":
            text, attachments, subject = extract_text_from_email(file_path)
            request_type, sub_request_type, confidence_score = classify_text(text)
            summary = create_summary(text)
        elif file_type == ".pdf":
            text = extract_text_from_pdf(file_path)
            request_type, sub_request_type, confidence_score = classify_text(text)
            summary = create_summary(text)
        elif file_type == ".jpeg" or file_type == ".jpg":
            text= extract_text_from_image(file_path)
            request_type, sub_request_type, confidence_score = classify_text(text)
            summary = f"Image contains: {text[:50] if text else 'No text'}"
        else:
            msg = f"Unsupported file type: {file_path}"
            logger.error(msg)
            raise ValueError(msg)

        if text and is_duplicate_request(text):
            result = ClassificationResult(
                file_path=file_path,
                file_type=file_type,
                subject=subject,
                content="",
                request_type="Duplicate",
                sub_request_type="Duplicate",
                summary="",
                confidence_score=1.0,
                status="duplicate",
                message="Duplicate request",
                attachments=None,
            )
            logger.info(f"File {file_path} is a duplicate.")
            return result

        if text:
            text_embedding = get_embedding(text)
            add_to_faiss(text_embedding, file_path)  # Add to Faiss index
            logger.info(f"Stored embedding for file: {file_path} in Faiss.")

        result = ClassificationResult(
            file_path=file_path,
            file_type=file_type,
            subject=subject,
            content=text,
            request_type=request_type,
            sub_request_type=sub_request_type,
            summary=summary,
            confidence_score=confidence_score,
            status="success",
            message="Processed successfully",
            attachments=attachments,
        )
        logger.info(f"Successfully processed file: {file_path}")
        return result
    except (ValueError, ValidationError, Exception) as e:
        msg = f"Error processing file: {file_path} - {e}"
        logger.error(msg)
        result = ClassificationResult(
            file_path=file_path,
            file_type=file_type,
            subject=subject,
            content="",
            request_type="Error",
            sub_request_type="Error",
            summary="",
            confidence_score=0.0,
            status="error",
            message=str(e),
            attachments=None,
        )
        return result



@app.post("/process_files/", response_model=List[ClassificationResult])
async def process_files_endpoint(folder_path: str = Form(...)):
    """
    Processes files from a folder.

    Args:
        folder_path (str): Path to the folder containing the files.

    Returns:
        List[ClassificationResult]: A list of ClassificationResult objects.
    """
    results = []
    if not folder_path:
        msg = "Folder path must be provided."
        logger.error(msg)
        raise HTTPException(status_code=400, detail=msg)
    if not os.path.exists(folder_path):
        msg = f"Folder not found: {folder_path}"
        logger.error(msg)
        raise HTTPException(status_code=400, detail=msg)
    file_paths = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith((".eml", ".pdf", ".jpeg", ".jpg"))
    ]
    if not file_paths:
        msg = f"No .eml, .pdf, or .jpeg files found in folder: {folder_path}"
        logger.error(msg)
        raise HTTPException(status_code=400, detail=msg)

    logger.info(f"Processing files from folder: {folder_path}")
    # Initialize Faiss index
    initialize_faiss()
    processed_files = set()
    for file_path in file_paths:
        if file_path not in processed_files:
            result = process_file(file_path)
            results.append(result)
            processed_files.add(file_path)
        else:
            logger.info(f"File {file_path} already processed, skipping")
    logger.info(f"Processed {len(results)} files from folder: {folder_path}")
    return JSONResponse(content=[result.model_dump() for result in results])



if __name__ == "__main__":
    import uvicorn

    # Add a log message when the server starts
    logger.info("Starting FastAPI application")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_config=None)
