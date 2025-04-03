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
    """
    output_buffer = io.StringIO()
    local_dict = {}
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
        exec(code, restricted_globals, local_dict)
        func_name = next((name for name, obj in local_dict.items() 
                         if callable(obj) and not name.startswith('_')), None)
        
        if func_name is None:
            return {
                'status': 'error',
                'output': 'Could not find any function in the code'
            }
            
        func = local_dict[func_name]
        
        with contextlib.redirect_stdout(output_buffer):
            result = None 
            if test_input:
                seed_processed_input = test_input
                if "np.random.seed" in test_input:
                    parts = test_input.split(",")
                    cleaned_parts = []
                    for part in parts:
                        part_stripped = part.strip()
                        if part_stripped.startswith("np.random.seed"):
                            try:
                                exec(part_stripped, restricted_globals)
                            except Exception as e:
                                return {
                                    'status': 'error',
                                    'output': f'Error setting random seed: {e}'
                                }
                        else:
                            cleaned_parts.append(part)
                    seed_processed_input = ",".join(cleaned_parts).strip()
                
                call_string = f"{func_name}({seed_processed_input})"
                result = eval(call_string, restricted_globals, local_dict)
            else:
                pass

            if result is not None:
                if isinstance(result, np.ndarray):
                    print(str(result.tolist()))
                else:
                    print(str(result))
            elif not test_input:
                 print("No test input provided, function not called.")
            
        output = output_buffer.getvalue()
        
        return {
            'status': 'success',
            'output': output if output else 'Code executed successfully with no output.'
        }
        
    except Exception as e:
        error_msg = traceback.format_exc()
        return {
            'status': 'error',
            'output': f'Error executing code:\n{error_msg}'
        }
        
    finally:
        output_buffer.close()

@app.route('/api/run_code', methods=['POST'])
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

# Health check endpoint
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200 