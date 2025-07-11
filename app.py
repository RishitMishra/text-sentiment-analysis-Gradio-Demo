import gradio as gr
from transformers import pipeline

# Load sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

def analyze_sentiment(text):
    result = sentiment_pipeline(text)[0]
    label = result['label']
    score = result['score']
    return f"Sentiment: {label} (Confidence: {score:.2f})"

# Create Gradio interface
iface = gr.Interface(fn=analyze_sentiment,
                     inputs=gr.Textbox(lines=4, placeholder="Type a review here..."),
                     outputs="text",
                     title="Sentiment Analyzer",
                     description="A simple sentiment analysis demo using Hugging Face transformers.")

iface.launch(share=True)
