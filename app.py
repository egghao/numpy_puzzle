from flask import Flask, request, jsonify, send_from_directory
import numpy as np
import io
import sys
import traceback
from contextlib import redirect_stdout

app = Flask(__name__)

# Capture stdout to get print statements
def capture_output(func):
    f = io.StringIO()
    with redirect_stdout(f):
        func()
    return f.getvalue()

def run_code_safely(code):
    # Create a new namespace for the code execution
    namespace = {'np': np}
    
    try:
        # Execute the code
        exec(code, namespace)
        
        # Get the last defined function
        functions = [name for name, obj in namespace.items() if callable(obj)]
        if not functions:
            return "No function defined in the code."
        
        last_function = functions[-1]
        func = namespace[last_function]
        
        # Get the test cases from the code
        test_cases = []
        if 'test_cases' in namespace:
            test_cases = namespace['test_cases']
        
        # Run the test cases
        results = []
        for test_case in test_cases:
            try:
                # Execute the test case
                result = eval(test_case['input'], namespace)
                expected = eval(test_case['output'])
                
                # Compare results
                if np.array_equal(result, expected):
                    results.append(f"Test case passed: {test_case['description']}")
                else:
                    results.append(f"Test case failed: {test_case['description']}\nExpected: {expected}\nGot: {result}")
            except Exception as e:
                results.append(f"Error in test case: {str(e)}")
        
        return "\n".join(results)
    
    except Exception as e:
        return f"Error: {str(e)}\n{traceback.format_exc()}"

@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('.', path)

@app.route('/run_code', methods=['POST'])
def run_code():
    code = request.json.get('code', '')
    output = run_code_safely(code)
    return jsonify({'output': output})

if __name__ == '__main__':
    app.run(debug=True, port=5000) 