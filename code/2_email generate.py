import os
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple
from io import BytesIO
from transformers import pipeline
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load the summarization pipeline
try:
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
except Exception as e:
    logger.error(f"Error loading summarization pipeline: {e}")
    raise


def create_email(
    subject: str,
    body_text: str,
    body_html: str,
    image_path: str,
    sender_email: str,
    recipient_email: str,
    email_count: int,
) -> str:
    """
    Creates an email message with a subject, text body, HTML body, and an embedded image.
    The image is generated, not loaded from a file.  The body text and html are now generated
    by an LLM for more realistic content.

    Args:
        subject (str): The subject of the email.
        body_text (str): The text version of the email body.  Now generated.
        body_html (str): The HTML version of the email body.  Now generated.
        image_path (str): The path to save the image.  This is now only used for the function name.
        sender_email (str): The sender's email address.
        recipient_email (str): The recipient's email address.
        email_count (int): Counter for email.

    Returns:
        str: The path to the created EML file.
    """

    # Create the email message
    message = MIMEMultipart("related")
    message["Subject"] = subject
    message["From"] = sender_email
    message["To"] = recipient_email
    message.preamble = "This is a multi-part message in MIME format."

    # Create the text part
    msg_text = MIMEText(body_text, "plain")
    message.attach(msg_text)

    # Create the HTML part
    msg_html = MIMEText(body_html, "html")
    message.attach(msg_html)

    # Generate the image and add it to the email
    try:
        img = generate_image(email_count)
        img_bytes = BytesIO()
        img.save(img_bytes, format="PNG")  # Save the image to a BytesIO object
        img_bytes.seek(0)  # Reset the buffer position to the beginning

        # Create the image part
        msg_image = MIMEImage(img_bytes.read(), _subtype="png") # explicitly set subtype
        msg_image.add_header("Content-ID", "<image1>")
        msg_image.add_header(
            "Content-Disposition",
            "inline",
            filename="transaction_image.png",
        )
        message.attach(msg_image)
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        # If there's an error with the image, continue without it.
        # You might want to add a placeholder image or send an email without the image.

    # Save the email to an EML file
    email_file_path = f"email_{email_count}.eml"
    try:
        with open(email_file_path, "wb") as f:
            f.write(message.as_bytes())
        logger.info(f"Email {email_file_path} created successfully.")
    except Exception as e:
        msg = f"Error saving email: {e}"
        logger.error(msg)
        raise

    return email_file_path


def generate_image(email_count: int) -> Image.Image:
    """
    Generates a sample image for the email.

    Returns:
        Image.Image: The generated image.
    """
    width, height = 400, 200
    img = Image.new("RGB", (width, height), color=(240, 240, 240))  # Light gray background
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("arial.ttf", 16)  # Use Arial font.  If not found, use a default.
    try:
        # Draw title
        title_text = "Loan Transaction Details"
        title_x = (width - draw.textlength(title_text, font=font)) / 2
        draw.text((title_x, 10), title_text, fill=(0, 0, 0), font=font)  # Black text

        # Draw transaction details
        transaction_details = [
            f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Amount: ${1000 + email_count * 100}",  # Varying amount
            "From: Sender Bank",
            "To: Recipient Bank",
            f"Request ID: {email_count}",
        ]
        y = 40
        for detail in transaction_details:
            detail_x = 20
            draw.text((detail_x, y), detail, fill=(50, 50, 50), font=font)  # Dark gray text
            y += 20
    except Exception as e:
        logger.error(f"Error drawing on image: {e}")
        return img
    return img



def generate_realistic_email_content(email_count: int) -> Tuple[str, str]:
    """
    Generates more realistic email content using a language model.

    Args:
        email_count: The email counter.

    Returns:
        Tuple[str, str]: A tuple containing the text and HTML versions of the email body.
    """
    prompt = f"""
    Generate a realistic email for a loan servicing transaction between two banks.
    The email should include details such as the date, amount, sending bank, receiving bank,
    and a unique request ID.  The request ID should be {email_count}.  The amount should vary.
    The tone should be professional and formal.  Do not include salutations or closings.
    Do not include a subject.  Just generate the body of the email.  The amount should be a realistic
    loan servicing amount, and should be different for each email. Do not include any introductory
    or concluding phrases. The date should be today.  The banks should be different fictitious banks.
    """

    # Generate the email text using the summarization pipeline (which can generate text)
    try:
        generated_text = summarizer(prompt, max_length=600, min_length=50, do_sample=True)[0][
            "summary_text"
        ]
    except Exception as e:
        logger.error(f"Error generating email content: {e}")
        generated_text = f"Error generating email content.  Default text.  Request ID: {email_count}"
    # Basic HTML conversion (you can expand on this as needed)
    generated_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <title>Loan Servicing Transaction</title>
    </head>
    <body>
    <div style="font-family: Arial, sans-serif; color: #333;">
    {generated_text.replace('.', '.<br><br>')}
    <img src="cid:image1" alt="Transaction Details" style="margin-top: 20px; width: 400px; height: 200px;">
    </div>
    </body>
    </html>
    """
    return generated_text, generated_html



def create_sample_emails():
    """
    Creates multiple sample emails with different subjects and body content related to loan servicing transactions.
    """
    sender_email = "sender@example.com"
    recipient_email = "recipient@example.com"
    image_path = "transaction_image.png"  # won't actually save to a file.

    # Create a directory to store the emails if it doesn't exist
    if not os.path.exists("emails"):
        os.makedirs("emails")
    os.chdir("emails")

    for i in range(5):  # Create 5 sample emails
        subject = f"Loan Servicing Transaction - Request {i + 1}"
        body_text, body_html = generate_realistic_email_content(i + 1)
        email_file_path = create_email(
            subject, body_text, body_html, image_path, sender_email, recipient_email, i
        )
        print(f"Email {i + 1} created: {email_file_path}")
    os.chdir("..")  # change back to the root directory
    print("Sample emails created successfully in the 'emails' directory.")



if __name__ == "__main__":
    create_sample_emails()

