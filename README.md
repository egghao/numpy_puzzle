# NumPy Neural Network Practice

An interactive web application for practicing NumPy implementations of common neural network operations. This project helps you understand the fundamentals of deep learning by implementing various neural network components from scratch using NumPy.

## Features

- Interactive code editor with syntax highlighting
- Real-time code execution
- Multiple difficulty levels (Easy, Medium, Hard)
- Various categories of neural network operations:
  - Activation Functions
  - Linear Layers
  - Convolution Operations
  - Pooling Layers
  - Regularization Techniques
  - Normalization Layers
  - Loss Functions
- Test cases for each implementation
- Filter questions by difficulty and category

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- A modern web browser
- Node.js and npm (optional, for serving the frontend)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd numpy-neural-network-practice
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

1. Start the Python backend server:
```bash
python server.py
```
The server will start on `http://localhost:5000`

2. Serve the frontend:

Option 1: Using Python's built-in HTTP server:
```bash
# In a new terminal
python -m http.server 8000
```
Then open `http://localhost:8000` in your browser

Option 2: Using Node.js http-server (if installed):
```bash
# Install http-server globally if you haven't already
npm install -g http-server

# Start the server
http-server -p 8000
```

Option 3: Simply open the `index.html` file in your web browser

## Usage

1. Browse the available questions in the left panel
2. Filter questions by difficulty and category using the filter button
3. Select a question to view its description, template, and test cases
4. Implement the solution in the code editor
5. Click "Run Code" to test your implementation
6. Check the output panel for results or error messages

## Project Structure

```
.
├── index.html          # Main HTML file
├── styles.css          # CSS styles
├── script.js           # Frontend JavaScript
├── server.py          # Python backend server
└── requirements.txt    # Python dependencies
```

## Contributing

Feel free to submit issues and enhancement requests!

## License

[MIT License](LICENSE) 