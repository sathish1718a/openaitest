from flask import Flask, render_template, request
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError
from pypdf import PdfReader
import markdown
import os
import uuid

load_dotenv()

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"

os.makedirs("uploads", exist_ok=True)

client = OpenAI()

SYSTEM_PROMPT = """
You are a professional document analysis assistant.

Rules:
- When a document is provided, analyze it carefully
- Provide a clear summary
- Highlight key points
- Mention risks, conclusions, or action items if relevant
- Use headings, bullet points, and structured format
"""

messages = [
    {"role": "system", "content": SYSTEM_PROMPT}
]

def extract_pdf_text(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text.strip()

@app.route("/", methods=["GET", "POST"])
def chat():
    global messages
    error = None

    if request.method == "POST":
        user_input = request.form.get("message")
        pdf_file = request.files.get("pdf")

        # 📄 PDF upload handling
        if pdf_file and pdf_file.filename.endswith(".pdf"):
            filename = f"{uuid.uuid4()}.pdf"
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            pdf_file.save(filepath)

            pdf_text = extract_pdf_text(filepath)

            messages.append({
                "role": "system",
                "content": f"Analyze the following PDF content:\n\n{pdf_text[:12000]}"
            })

            user_input = "Please summarize and analyze this PDF."

        if user_input:
            messages.append({"role": "user", "content": user_input})

            try:
                response = client.responses.create(
                    model="gpt-5-nano",
                    input=messages
                )

                reply_text = response.output_text
                reply_html = markdown.markdown(
                    reply_text,
                    extensions=["fenced_code", "tables"]
                )

                messages.append({
                    "role": "assistant",
                    "content": reply_html
                })

            except RateLimitError:
                error = "⏳ Rate limit reached. Please wait 20–30 seconds."
                messages.pop()

    return render_template("index.html", messages=messages, error=error)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)