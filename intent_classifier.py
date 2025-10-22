"""
Multi-Turn Intent Classification System
Classifies customer intent from WhatsApp-style conversations
"""

import json
import pandas as pd
import re
from typing import List, Dict, Tuple
from transformers import pipeline
import torch


class ConversationPreprocessor:
    """Handles cleaning and formatting of conversation data"""

    def __init__(self, max_messages: int = 10):
        """
        Args:
            max_messages: Maximum number of recent messages to consider
        """
        self.max_messages = max_messages

    def clean_text(self, text: str) -> str:
        """Clean individual message text"""
        # Remove emojis
        text = re.sub(r'[^\w\s\.,!?\-\'\"]+', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        # Convert to lowercase for consistency
        text = text.lower()
        return text.strip()

    def format_conversation(self, messages: List[Dict]) -> str:
        """Convert message list to formatted conversation text"""
        # Take last N messages for efficiency
        recent_messages = messages[-self.max_messages:] if len(messages) > self.max_messages else messages

        conversation_parts = []
        for msg in recent_messages:
            sender = msg['sender']
            text = self.clean_text(msg['text'])
            conversation_parts.append(f"{sender}: {text}")

        return "\n".join(conversation_parts)

    def preprocess_conversation(self, conversation: Dict) -> Dict:
        """Preprocess entire conversation object"""
        processed = {
            'conversation_id': conversation['conversation_id'],
            'conversation_text': self.format_conversation(conversation['messages']),
            'original_messages': conversation['messages'],
            'num_messages': len(conversation['messages']),
            'last_user_message': self._get_last_user_message(conversation['messages'])
        }
        return processed

    def _get_last_user_message(self, messages: List[Dict]) -> str:
        """Extract the last message from user"""
        for msg in reversed(messages):
            if msg['sender'] == 'user':
                return self.clean_text(msg['text'])
        return ""


class IntentClassifier:
    """Classifies conversation intent using prompt-based LLM approach"""

    INTENT_LABELS = [
        "Book Appointment",
        "Product Inquiry",
        "Pricing Negotiation",
        "Support Request",
        "Follow-Up"
    ]

    def __init__(self, model_name: str = "google/flan-t5-base"):
        """
        Initialize classifier with specified model
        Args:
            model_name: HuggingFace model identifier
        """
        print(f"Loading model: {model_name}...")
        self.model_name = model_name

        # Use text2text-generation for FLAN-T5
        self.classifier = pipeline(
            "text2text-generation",
            model=model_name,
            device=0 if torch.cuda.is_available() else -1,
            max_length=50
        )
        print("Model loaded successfully!")

    def create_prompt(self, conversation_text: str, last_user_message: str) -> str:
        """Create classification prompt for the model"""
        prompt = f"""Analyze this customer service conversation and determine the customer's final intent.

Conversation:
{conversation_text}

The customer's last message was: "{last_user_message}"

Choose ONE intent from these options:
1. Book Appointment - Customer wants to schedule a meeting, visit, or appointment
2. Product Inquiry - Customer is asking about products, services, or availability
3. Pricing Negotiation - Customer is discussing, negotiating, or asking about prices/discounts
4. Support Request - Customer needs help with an issue, problem, or technical support
5. Follow-Up - Customer is following up on a previous conversation or request

What is the customer's final intent? Answer with just the intent name."""

        return prompt

    def classify(self, conversation_text: str, last_user_message: str) -> Tuple[str, str]:
        """
        Classify conversation intent
        Returns: (intent, rationale)
        """
        prompt = self.create_prompt(conversation_text, last_user_message)

        # Generate prediction
        result = self.classifier(prompt, max_new_tokens=30, temperature=0.1)[0]['generated_text']

        # Extract intent from response
        predicted_intent = self._extract_intent(result)

        # Generate rationale
        rationale = self._generate_rationale(conversation_text, last_user_message, predicted_intent)

        return predicted_intent, rationale

    def _extract_intent(self, model_output: str) -> str:
        """Extract intent label from model output"""
        model_output = model_output.lower().strip()

        # Check for each intent keyword
        if "appointment" in model_output or "book" in model_output or "schedule" in model_output or "visit" in model_output:
            return "Book Appointment"
        elif "price" in model_output or "negotiat" in model_output or "discount" in model_output or "cost" in model_output:
            return "Pricing Negotiation"
        elif "support" in model_output or "help" in model_output or "issue" in model_output or "problem" in model_output:
            return "Support Request"
        elif "follow" in model_output or "update" in model_output or "status" in model_output:
            return "Follow-Up"
        else:
            return "Product Inquiry"  # Default fallback

    def _generate_rationale(self, conversation_text: str, last_user_message: str, intent: str) -> str:
        """Generate human-readable rationale for the classification"""
        rationales = {
            "Book Appointment": f"The customer requested scheduling or arranging a meeting/visit. Last message: '{last_user_message}'",
            "Product Inquiry": f"The customer was inquiring about products, services, or availability. Last message: '{last_user_message}'",
            "Pricing Negotiation": f"The customer discussed pricing, budget, or negotiation. Last message: '{last_user_message}'",
            "Support Request": f"The customer requested help or reported an issue. Last message: '{last_user_message}'",
            "Follow-Up": f"The customer followed up on a previous conversation or request. Last message: '{last_user_message}'"
        }
        return rationales.get(intent, f"Intent classified as {intent}")


class IntentPredictor:
    """Main predictor that orchestrates preprocessing and classification"""

    def __init__(self, model_name: str = "google/flan-t5-base", max_messages: int = 10):
        """
        Initialize predictor
        Args:
            model_name: HuggingFace model to use
            max_messages: Max recent messages to consider
        """
        self.preprocessor = ConversationPreprocessor(max_messages=max_messages)
        self.classifier = IntentClassifier(model_name=model_name)

    def predict_single(self, conversation: Dict) -> Dict:
        """Predict intent for a single conversation"""
        # Preprocess
        processed = self.preprocessor.preprocess_conversation(conversation)

        # Classify
        intent, rationale = self.classifier.classify(
            processed['conversation_text'],
            processed['last_user_message']
        )

        return {
            'conversation_id': processed['conversation_id'],
            'predicted_intent': intent,
            'rationale': rationale
        }

    def predict_batch(self, conversations: List[Dict], output_json: str = "predictions.json",
                      output_csv: str = "predictions.csv") -> List[Dict]:
        """
        Predict intents for multiple conversations
        Args:
            conversations: List of conversation objects
            output_json: Path to save JSON output
            output_csv: Path to save CSV output
        Returns:
            List of predictions
        """
        print(f"Processing {len(conversations)} conversations...")
        predictions = []

        for i, conv in enumerate(conversations):
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(conversations)} conversations")

            prediction = self.predict_single(conv)
            predictions.append(prediction)

        # Save outputs
        self._save_predictions(predictions, output_json, output_csv)

        print(f"✅ Predictions saved to {output_json} and {output_csv}")
        return predictions

    def _save_predictions(self, predictions: List[Dict], json_path: str, csv_path: str):
        """Save predictions in both JSON and CSV formats"""
        # Save JSON
        with open(json_path, 'w') as f:
            json.dump(predictions, f, indent=2)

        # Save CSV
        df = pd.DataFrame(predictions)
        df.to_csv(csv_path, index=False)


# Example usage
if __name__ == "__main__":
    # Sample input data
    sample_conversations = [
        {
            "conversation_id": "conv_001",
            "messages": [
                {"sender": "user", "text": "Hi, I'm looking for a 2BHK in Dubai"},
                {"sender": "agent", "text": "Great! Any specific area in mind?"},
                {"sender": "user", "text": "Preferably Marina or JVC"},
                {"sender": "agent", "text": "What's your budget?"},
                {"sender": "user", "text": "Max 120k. Can we do a site visit this week?"}
            ]
        },
        {
            "conversation_id": "conv_002",
            "messages": [
                {"sender": "user", "text": "What amenities do you have?"},
                {"sender": "agent", "text": "We have pool, gym, parking, and 24/7 security"},
                {"sender": "user", "text": "Do you have units with balconies?"}
            ]
        }
    ]

    # Load from JSON file
    # with open('input_conversations.json', 'r') as f:
    #     conversations = json.load(f)

    # Initialize predictor
    predictor = IntentPredictor(model_name="google/flan-t5-base", max_messages=10)

    # Run predictions
    predictions = predictor.predict_batch(
        sample_conversations,
        output_json="predictions.json",
        output_csv="predictions.csv"
    )

    # Print results
    for pred in predictions:
        print(f"\n{pred['conversation_id']}: {pred['predicted_intent']}")
        print(f"Rationale: {pred['rationale']}")