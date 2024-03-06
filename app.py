from flask import Flask, request, jsonify
import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import time

# Initialize Flask application
app = Flask(__name__)

# Preload and cache the SentenceTransformer model and FAISS index
model = SentenceTransformer('all-MiniLM-L6-v2')
# Assume embeddings and index are loaded here as before

# Function to load and preprocess the JSON data
def load_data(file_path):
    with open(file_path) as f:
        return json.load(f)

# Similar to your original create_passages function
def create_passages(data):
    passages = []
    # Extract chicken items
    for item_id, item_info in data["Chicken"].items():
        name, price, details = item_info
        passage = f"Name: {name}, Price: {price}, Calories: {details['nutritionalInfo']['kcal']}, Fat: {details['nutritionalInfo']['fat']}, Protein: {details['nutritionalInfo']['protein']}, Allergens: {', '.join(details['nutritionalInfo']['allergens'])}, Available: {details['available']}"
        passages.append(passage)

    # Extract menu items
    for menu_id, menu_info in data["Menus"].items():
        name = menu_info["name"]
        price = menu_info["price"]
        contents = []
        for item in menu_info["contents"]:
            if isinstance(item, list):  # Checks if item is a list
                contents.append(f"{item[0]} x {item[1]}")
            elif isinstance(item, dict):  # Checks if item is a dictionary
                if 'from' in item:
                    contents.append(f"from: {item['from']} size: {item.get('size', 'N/A')}")
                elif 'choose' in item:
                    contents.append(f"choose from: {item['from']}, options: {item.get('choose', 'N/A')}")
        content_str = ", ".join(contents)
        passage = f"Menu: {name}, Price: {price}, Contents: {content_str}"
        passages.append(passage)

    return passages



# Assuming embeddings are generated here as before
# Function to generate embeddings from passages
def generate_embeddings(passages):
    embeddings = model.encode(passages, convert_to_tensor=False)  # convert_to_tensor=False ensures returning numpy array
    print(f"Embeddings shape: {embeddings.shape}")  # Debug: Print the shape to verify
    return embeddings


# Initialize the FAISS index with the embeddings
def create_faiss_index(embeddings):
    if embeddings.ndim == 2:
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings.astype(np.float32))
        return index
    else:
        raise ValueError("Embeddings are not a 2D array. Check the embeddings generation.")

# Assume embeddings are added to index here as before

# Preprocess data and create FAISS index
data = load_data('menu.json')
passages = create_passages(data)
embeddings = generate_embeddings(passages)  # Ensure embeddings are generated here
index = create_faiss_index(embeddings)  # Create FAISS index with the generated embeddings

# Adjusted search_index function
def search_index(query, k=5):
    query_vec = model.encode([query], convert_to_tensor=True).cpu().numpy()
    if query_vec.ndim == 1:
        query_vec = query_vec.reshape(1, -1)
    D, I = index.search(query_vec, k)
    return [passages[i] for i in I[0]]

# Optimized query handler
def handle_query(query):
    search_results = search_index(query, k=5)
    response = {"query": query, "results": search_results}
    return response

@app.route('/query', methods=['POST'])
def query_menu():
    data = request.json
    query = data.get('query', '')
    start_time = time.time()
    response = handle_query(query)
    end_time = time.time()
    time_taken = end_time - start_time
    response['time_taken'] = f"{time_taken:.4f} seconds"
    return jsonify({"response": response})

@app.route('/')
def hello_world():
    return 'Hello, World!'

if __name__ == '__main__':
    app.run(debug=True)



'''Un optimized Code'''

'''
from flask import Flask, request, jsonify
from transformers import RagTokenizer, RagTokenForGeneration, RagRetriever
import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import time 

#initilize flask application
app = Flask(__name__)

# Load and preprocess the JSON data
def load_data(file_path):
    with open(file_path) as f:
        return json.load(f)

# Function to create passages from the data
def create_passages(data):
    passages = []
    # Extract chicken items
    for item_id, item_info in data["Chicken"].items():
        name, price, details = item_info
        passage = f"Name: {name}, Price: {price}, Calories: {details['nutritionalInfo']['kcal']}, Fat: {details['nutritionalInfo']['fat']}, Protein: {details['nutritionalInfo']['protein']}, Allergens: {', '.join(details['nutritionalInfo']['allergens'])}, Available: {details['available']}"
        passages.append(passage)

    # Extract menu items
    for menu_id, menu_info in data["Menus"].items():
        name = menu_info["name"]
        price = menu_info["price"]
        contents = []
        for item in menu_info["contents"]:
            if isinstance(item, list):  # Checks if item is a list
                contents.append(f"{item[0]} x {item[1]}")
            elif isinstance(item, dict):  # Checks if item is a dictionary
                if 'from' in item:
                    contents.append(f"from: {item['from']} size: {item.get('size', 'N/A')}")
                elif 'choose' in item:
                    contents.append(f"choose from: {item['from']}, options: {item.get('choose', 'N/A')}")
        content_str = ", ".join(contents)
        passage = f"Menu: {name}, Price: {price}, Contents: {content_str}"
        passages.append(passage)

    return passages


data = load_data('menu.json')
passages = create_passages(data)

# Encode passages and create FAISS index
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(passages)
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings.astype(np.float32))

# Function to search in the index
def search_index(query, k=5):
    query_vec = model.encode([query])
    D, I = index.search(query_vec.astype(np.float32), k)
    return [passages[i] for i in I[0]]

#query handler
def handle_query(query):
    # Use the search_index function to find relevant passages
    search_results = search_index(query, k=5)  # Adjust k as needed
    
    # Format the results for the response
    # Depending on your application's needs, you might format these results differently
    response = {"query": query, "results": search_results}
    
    return response


#create a route
@app.route('/query', methods=['POST'])
def query_menu():
    data = request.json
    query = data.get('query', '')
    start_time = time.time()
    response = handle_query(query)
    end_time = time.time()
    time_taken = end_time - start_time
    response['time_taken'] = f"{time_taken:.4f} seconds"
    return jsonify({"response": response})


@app.route('/')
def hello_world():
    return 'Hello, World!'

if __name__ == '__main__':
    app.run(debug=True)

'''
