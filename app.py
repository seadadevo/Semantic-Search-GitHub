import pandas as pd
from flask import Flask, request, jsonify
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os

app = Flask(__name__)
app.secret_key = 'e332c75bc8de5a684596e55242f9beb5c1cff28d8dc771618a90e82af17a2610'

# Check if model and data files exist
model_path = 'my_model.h5'
data_path = 'book_data.csv'

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

if not os.path.exists(data_path):
    raise FileNotFoundError(f"Data file not found: {data_path}")

# Load the trained model
try:
    model = keras.models.load_model(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    raise

# Load book data
try:
    data = pd.read_csv(data_path, encoding='ISO-8859-1')
    data.dropna(subset=['book_desc'], inplace=True)
    print("Data loaded successfully.")
except Exception as e:
    print(f"Error loading data: {e}")
    raise

# Prepare tokenizer
try:
    max_words = 1000
    max_len = 100
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(data['book_desc'])
    print("Tokenizer prepared successfully.")
except Exception as e:
    print(f"Error preparing tokenizer: {e}")
    raise

# Prepare label encoders
label_encoder_title = LabelEncoder()
label_encoder_author = LabelEncoder()
try:
    data['book_title_encoded'] = label_encoder_title.fit_transform(data['book_title'])
    data['book_author_encoded'] = label_encoder_author.fit_transform(data['book_authors'])
    print("Label encoders prepared successfully.")
except Exception as e:
    print(f"Error preparing label encoders: {e}")
    raise

@app.route("/get-response", methods=["POST"])
def get_response():
    user_input = request.json.get('description')
    if not user_input:
        return jsonify({"error": "No description provided"}), 400
    
    try:
        # Tokenize and pad the user input
        user_sequence = tokenizer.texts_to_sequences([user_input])
        user_padded = pad_sequences(user_sequence, maxlen=max_len)
        
        # Predict using the model
        predictions = model.predict(user_padded)
        
        # Get top 10 predictions for title and author
        top_title_indices = predictions[0][0].argsort()[-10:][::-1]
        top_author_indices = predictions[1][0].argsort()[-10:][::-1]
        
        response = []
        for i in range(10):
            title_index = top_title_indices[i]
            author_index = top_author_indices[i]
            rating = float(predictions[2][0][0])  # Use the same rating for simplicity
            
            title = label_encoder_title.inverse_transform([title_index])[0]
            author = label_encoder_author.inverse_transform([author_index])[0]
            
            response.append({
                "title": title,
                "author": author,
                "rating": rating
            })
        
        return jsonify(response)
    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

if __name__ == "__main__":
    try:
        app.run(debug=True, use_reloader=False)  # Added use_reloader=False to prevent multiple instances
    except Exception as e:
        print(f"Error starting Flask app: {e}")
