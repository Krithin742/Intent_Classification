from intent_classifier import IntentPredictor
import json

# Load conversations
with open('input_conversations.json', 'r') as f:
    conversations = json.load(f)

# Initialize predictor
predictor = IntentPredictor(
    model_name="google/flan-t5-base",
    max_messages=10
)

# Run predictions
predictions = predictor.predict_batch(
    conversations,
    output_json="predictions.json",
    output_csv="predictions.csv"
)