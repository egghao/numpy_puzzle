from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import io
import sys
import contextlib
from typing import Dict, Any
import traceback
import math

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*", "allow_headers": ["Content-Type"]}})

def secure_exec(code: str, test_input: str = None) -> Dict[str, Any]:
    """
    Securely execute the provided code in a restricted environment.
    
    Args:
        code (str): The Python code to execute
        test_input (str, optional): The test input to use
        
    Returns:
        dict: A dictionary containing the execution result and any output
    """
    # Create string buffer to capture output
    output_buffer = io.StringIO()
    
    # Create a dictionary for local variables
    local_dict = {}
    
    # Prepare the restricted globals
    restricted_globals = {
        'np': np,
        'math': math,
        '__builtins__': {
            '__import__': __import__,
            'abs': abs,
            'all': all,
            'any': any,
            'bool': bool,
            'dict': dict,
            'dir': dir,
            'enumerate': enumerate,
            'float': float,
            'format': format,
            'getattr': getattr,
            'hasattr': hasattr,
            'int': int,
            'isinstance': isinstance,
            'len': len,
            'list': list,
            'locals': locals,
            'map': map,
            'max': max,
            'min': min,
            'next': next,
            'print': print,
            'range': range,
            'repr': repr,
            'round': round,
            'setattr': setattr,
            'slice': slice,
            'str': str,
            'sum': sum,
            'tuple': tuple,
            'type': type,
            'zip': zip,
        }
    }
    
    try:
        # First execute the code to define the function
        exec(code, restricted_globals, local_dict)
        
        # Find the first function defined in the code
        func_name = next((name for name, obj in local_dict.items() 
                         if callable(obj) and not name.startswith('_')), None)
        
        if func_name is None:
            return {
                'status': 'error',
                'output': 'Could not find any function in the code'
            }
            
        func = local_dict[func_name]
        
        # Execute the function with test input and capture its return value
        with contextlib.redirect_stdout(output_buffer):
            # Default result in case no test input is provided or evaluated
            result = None 
            if test_input:
                seed_processed_input = test_input
                # Parse the test_input string to check for random seed
                if "np.random.seed" in test_input:
                    parts = test_input.split(",")
                    cleaned_parts = []
                    for part in parts:
                        part_stripped = part.strip()
                        if part_stripped.startswith("np.random.seed"):
                            try:
                                # Execute just the seed setting
                                exec(part_stripped, restricted_globals)
                            except Exception as e:
                                return {
                                    'status': 'error',
                                    'output': f'Error setting random seed: {e}'
                                }
                        else:
                            cleaned_parts.append(part)
                    # Reconstruct the input string without the seed part
                    seed_processed_input = ",".join(cleaned_parts).strip()
                
                # Construct the actual function call string
                call_string = f"{func_name}({seed_processed_input})"
                
                # Evaluate the function call string
                result = eval(call_string, restricted_globals, local_dict)

            else:
                # Default execution if no test input
                # This path might need adjustment depending on expected behavior without input
                # or we handle it based on specific function needs if this scenario occurs.
                # If a default like np.array([-1, 0, 1]) is always needed, it should be passed
                # as default test_input from the frontend or handled here explicitly.
                # result = func() # Assuming func() is valid if no test_input
                pass # Or maybe raise an error if test_input is mandatory? For now, do nothing.

            # Print the result if execution happened
            if result is not None:
                if isinstance(result, np.ndarray):
                    # Convert to list and then to string for cleaner output
                    print(str(result.tolist()))
                else:
                    print(str(result))
            elif not test_input:
                 print("No test input provided, function not called.")
            
        # Get the printed output
        output = output_buffer.getvalue()
        
        return {
            'status': 'success',
            'output': output if output else 'Code executed successfully with no output.'
        }
        
    except Exception as e:
        # Get the full traceback
        error_msg = traceback.format_exc()
        return {
            'status': 'error',
            'output': f'Error executing code:\n{error_msg}'
        }
        
    finally:
        output_buffer.close()

@app.route('/run_code', methods=['POST'])
def run_code():
    """Handle code execution requests."""
    data = request.get_json()
    
    if not data or 'code' not in data:
        return jsonify({
            'status': 'error',
            'output': 'No code provided'
        }), 400
    
    test_input = data.get('test_input')
    result = secure_exec(data['code'], test_input)
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 