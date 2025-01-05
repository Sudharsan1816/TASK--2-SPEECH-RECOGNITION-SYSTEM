# Import necessary libraries
import nltk
from transformers import pipeline
from nltk.tokenize import sent_tokenize

# Download NLTK data
nltk.download("punkt")

# Function to summarize using Hugging Face Transformers
def summarize_text_huggingface(text, max_length=100, min_length=30):
    # Initialize the summarization pipeline
    summarizer = pipeline("summarization")
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]['summary_text']

# Function to summarize using NLTK (extractive summarization)
def summarize_text_nltk(text, num_sentences=3):
    sentences = sent_tokenize(text)
    if len(sentences) <= num_sentences:
        return "Text is too short to summarize."
    return " ".join(sentences[:num_sentences])

# Main script to input text and show summaries
if __name__ == "_main_":
    # Input: Prompt user to enter the text
    input_text = input("Enter the text you want to summarize:\n")

    # Check if input is empty
    if not input_text.strip():
        print("Error: No text provided.")
    else:
        print("\nInput Text:\n")
        print(input_text)
        
        # Summarize using Hugging Face
        print("\nAbstractive Summary (Hugging Face):\n")
        print(summarize_text_huggingface(input_text))
        
        # Summarize using NLTK
        print("\nExtractive Summary (NLTK):\n")
        print(summarize_text_nltk(input_text, num_sentences=2))
