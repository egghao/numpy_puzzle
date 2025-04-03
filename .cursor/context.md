# Project: NumPy Puzzle

## 1. Purpose
An interactive web application designed for practicing and learning NumPy implementations of common neural network operations. Users can write Python code using NumPy to solve specific problems (e.g., implementing ReLU, Convolution, Pooling) and test their solutions against predefined test cases.

## 2. Technology Stack
*   **Frontend:** Static HTML (`public/index.html`), CSS (`public/styles.css`), and Vanilla JavaScript (`public/script.js`). It uses the CodeMirror library for the interactive code editor.
*   **Backend:** Python Flask application (`api/index.py`) designed to run as a serverless function (configured for Vercel).
*   **Core Computation:** NumPy is used for all the numerical operations within the user's submitted code and potentially within the test case generation/evaluation (though test cases seem pre-defined in the JS).
*   **Deployment:** Configured for Vercel (`vercel.json`).
*   **Prerequisites:** Python 3.8 or higher, pip, Vercel CLI, and a modern web browser.

## 3. Project Structure
*   `api/`: Contains the backend Flask application (`index.py`).
*   `public/`: Contains all static frontend assets (HTML, CSS, JS, images).
*   `requirements.txt`: Python dependencies (Flask, Flask-Cors, NumPy).
*   `vercel.json`: Vercel deployment configuration.
*   `README.md`: Project description, setup, and usage instructions.
*   `.cursor/`: Directory intended for Cursor-specific context files (like this one).

## 4. Question Categories and Difficulty Levels
*   **Categories:**
    *   Activation Functions (ReLU, Sigmoid, Tanh, etc.)
    *   Linear Layers (Forward and backward propagation)
    *   Convolution Operations (2D convolution, padding, stride)
    *   Pooling Layers (Max pooling, average pooling)
    *   Regularization Techniques (Dropout, L1/L2)
    *   Normalization Layers (Batch normalization)
    *   Loss Functions (Cross-entropy, MSE)
*   **Difficulty Levels:**
    *   Easy: Basic operations and simple implementations
    *   Medium: More complex operations requiring deeper understanding
    *   Hard: Advanced implementations with multiple components

## 5. Frontend Logic (`public/script.js`)
*   **Data:** Holds a large array `questions` containing all puzzle data (ID, title, description, difficulty, category, status, code template, and multiple test cases with input strings and expected output strings).
*   **UI Components:**
    *   Code Editor (CodeMirror instance).
    *   Resizable and collapsible side panel (`#questionsPanel`) listing filterable/searchable questions. Panel width is saved to `localStorage`.
    *   Main content area (`.code-view`) displaying either a welcome screen or the selected question's details (description, template, test case descriptions).
    *   Tabbed view showing Input/Expected Output/Actual Output for the first three test cases.
    *   Live search bar (`#searchBar`) in the questions panel header.
    *   Filter button and dropdown for difficulty/category.
    *   "Run Code" button.
    *   Theme toggle (Light/Dark), defaulting to Dark mode.
*   **Functionality:**
    *   Loads saved question statuses, test outputs, and panel width from `localStorage` on startup.
    *   Populates and dynamically filters the question list based on user selections (difficulty, category) and live search terms.
    *   Displays a welcome screen initially; hides it and shows question details when a question is selected.
    *   Loads the selected question's data into the UI, including saved test outputs if available.
    *   Formats NumPy array strings from the `questions` data and API responses for cleaner display and comparison.
    *   Sends the code from the editor and the relevant test case input string(s) to the backend API upon clicking "Run Code". Handles up to 3 test cases per question.
    *   Receives results from the backend.
    *   Displays the actual output, compares it visually (via text color and formatting) to the expected output (green for match, yellow for mismatch, red for error).
    *   Updates the status (`pending`, `ongoing`, `completed`) of the question in the list and saves it to `localStorage`.
    *   Saves test case outputs to `localStorage` after running code.
    *   Manages theme persistence using `localStorage`, defaulting to dark mode.
    *   Handles panel resizing and collapsing.
    *   Updates scroll indicators for I/O areas.

## 6. Backend Logic (`api/index.py`)
*   **Framework:** Flask.
*   **Endpoints:**
    *   `/api/run_code` (POST): Accepts JSON containing `code` (string) and optionally `test_input` (string).
        *   Executes the provided `code` string within a restricted Python environment using `exec`. Only specific built-ins and the `numpy` and `math` modules are allowed.
        *   If `test_input` is provided, it identifies the first user-defined function in the `code`, constructs a call string (e.g., `func_name(test_input_value)`), handling `np.random.seed` separately, and evaluates it using `eval`.
        *   Captures `stdout` generated during execution.
        *   Returns JSON `{ "status": "success" | "error", "output": "captured_stdout_or_error_message" }`.
    *   `/api/health` (GET): Returns `{"status": "healthy"}`.
*   **Security:** Attempts to provide a secure execution environment by limiting available globals and built-ins during `exec`.

## 7. Interaction Flow
1.  Page loads, defaults to dark mode, loads saved statuses/outputs/panel width.
2.  Welcome screen is shown in the right pane.
3.  User selects a question in the resizable/collapsible left panel (list is filterable/searchable).
4.  Frontend hides the welcome screen, shows the code view, displays the question details, and loads the template into the CodeMirror editor. Saved outputs are displayed if present.
5.  User writes/modifies the NumPy code.
6.  User clicks "Run Code".
7.  Frontend iterates through the first 1-3 test cases for the selected question. For each:
    *   It sends the *entire* current editor code and the specific `test_input` string to the `/api/run_code` backend endpoint.
8.  Backend receives the request, executes the *entire* code using `secure_exec`, evaluates the function call with the `test_input`, and captures the result/error.
9.  Backend responds with the status and output for that test case.
10. Frontend receives the response, displays the formatted output in the corresponding tab, colors it based on formatted comparison, and repeats for other test cases.
11. Frontend updates the overall status of the question (`ongoing` or `completed`) based on the results of all run test cases.
12. The updated status and the outputs for each test case are saved to `localStorage`.
13. The question list in the left panel is refreshed to show the updated status badge. 