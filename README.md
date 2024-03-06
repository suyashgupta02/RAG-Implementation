# Menu Search API

This project provides a RESTful API for searching through a menu database using natural language queries. It leverages the Flask framework for the API, Sentence Transformers for generating textual embeddings, and FAISS for efficient similarity search in the embedding space. The project is designed to preprocess menu data, generate embeddings, and create a FAISS index to facilitate quick and relevant search results based on user queries.

## Features

- Flask-based web API
- Natural language processing via Sentence Transformers
- Efficient search using FAISS
- Support for complex queries related to menu items and nutritional information

## Setting Up

Before running the application, you need to prepare your menu data in a JSON file named menu.json. The JSON structure should follow the expected format as indicated by the code (with "Chicken" and "Menus" categories).

## Installation

Ensure you have Python 3.6+ installed on your system. Then, install the required dependencies by running:

## Usage

curl -X POST http://127.0.0.1:5000/query -H 'Content-Type: application/json' -d '{"query":"Find me a chicken dish with less than 500 calories."}'

```bash
pip install flask sentence_transformers faiss-cpu numpy
flask run




