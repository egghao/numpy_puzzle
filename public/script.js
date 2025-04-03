// Initialize CodeMirror
let editor = CodeMirror.fromTextArea(document.getElementById("codeEditor"), {
    mode: "python",
    theme: "monokai",
    lineNumbers: true,
    autoCloseBrackets: true,
    matchBrackets: true,
    indentUnit: 4,
    tabSize: 4,
    indentWithTabs: false,
    lineWrapping: true,
    gutters: ["CodeMirror-linenumbers"],
    extraKeys: {
        "Tab": "indentMore",
        "Shift-Tab": "indentLess"
    }
});

// Questions data
const questions = [
    {
        id: 1,
        title: "ReLU Activation Function",
        description: "Implement the ReLU (Rectified Linear Unit) activation function using NumPy. The function should return max(0, x) for each element in the input array.",
        difficulty: "Easy",
        category: "Activation Functions",
        status: 'pending',
        template: `import numpy as np

def relu(x):
    """
    Implement the ReLU activation function.
    
    Args:
        x (np.ndarray): Input array
        
    Returns:
        np.ndarray: Output array with ReLU applied
    """
    # Your code here
    pass`,
        testCases: [
            {
                input: "np.array([-1, 0, 1, 2])",
                output: "array([0, 0, 1, 2])",
                description: "Basic test case with positive and negative numbers"
            },
            {
                input: "np.array([[-3.5, -2.1], [0.0, 4.2]])",
                output: "array([[0., 0.], [0., 4.2]])",
                description: "2D array with floating point values"
            },
            {
                input: "np.zeros((3, 3))",
                output: "array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])",
                description: "Edge case: all zeros"
            }
        ]
    },
    {
        id: 2,
        title: "Linear Layer Forward Pass",
        description: "Implement the forward pass of a linear (fully connected) layer using NumPy. The layer should perform the operation: output = input @ weights + bias",
        difficulty: "Medium",
        category: "Linear Layers",
        status: 'pending',
        template: `import numpy as np

def linear_forward(x, weights, bias):
    """
    Implement the forward pass of a linear layer.
    
    Args:
        x (np.ndarray): Input array of shape (batch_size, input_features)
        weights (np.ndarray): Weight matrix of shape (input_features, output_features)
        bias (np.ndarray): Bias vector of shape (output_features,)
        
    Returns:
        np.ndarray: Output array of shape (batch_size, output_features)
    """
    # Your code here
    pass`,
        testCases: [
            {
                input: "x = np.array([[1, 2], [3, 4]]), weights = np.array([[0.1, 0.2], [0.3, 0.4]]), bias = np.array([0.1, 0.2])",
                output: "array([[1.0, 1.6], [2.2, 3.2]])",
                description: "Basic test case with 2D input"
            },
            {
                input: "x = np.array([[0, 0], [1, 1]]), weights = np.array([[1, 2], [3, 4]]), bias = np.array([0, 0])",
                output: "array([[0, 0], [4, 6]])",
                description: "Test with zeros and ones"
            },
            {
                input: "x = np.ones((3, 2)), weights = np.ones((2, 4)), bias = np.zeros(4)",
                output: "array([[2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.]])",
                description: "Edge case: all ones with larger output dimensions"
            }
        ]
    },
    {
        id: 3,
        title: "Sigmoid Activation",
        description: "Implement the Sigmoid activation function using NumPy. The function should return 1/(1 + exp(-x)) for each element in the input array.",
        difficulty: "Easy",
        category: "Activation Functions",
        status: 'pending',
        template: `import numpy as np

def sigmoid(x):
    """
    Implement the Sigmoid activation function.
    
    Args:
        x (np.ndarray): Input array
        
    Returns:
        np.ndarray: Output array with Sigmoid applied
    """
    # Your code here
    pass`,
        testCases: [
            {
                input: "np.array([-1, 0, 1])",
                output: "array([0.26894142, 0.5, 0.73105858])",
                description: "Basic test case with positive and negative numbers"
            },
            {
                input: "np.array([[-10, 0], [10, 20]])",
                output: "array([[4.53978687e-05, 5.00000000e-01], [9.99954602e-01, 9.99999979e-01]])",
                description: "Test with extreme values"
            },
            {
                input: "np.zeros((2, 2))",
                output: "array([[0.5, 0.5], [0.5, 0.5]])",
                description: "Edge case: all zeros (sigmoid(0) = 0.5)"
            }
        ]
    },
    {
        id: 4,
        title: "2D Convolution",
        description: "Implement a 2D convolution operation using NumPy. The function should perform convolution between input and kernel with optional stride and padding.",
        difficulty: "Hard",
        category: "Convolution",
        status: 'pending',
        template: `import numpy as np

def conv2d(x, kernel, stride=1, padding=0):
    """
    Implement 2D convolution operation.
    
    Args:
        x (np.ndarray): Input array of shape (batch_size, in_channels, height, width)
        kernel (np.ndarray): Kernel array of shape (out_channels, in_channels, kernel_height, kernel_width)
        stride (int): Stride of the convolution
        padding (int): Padding size
        
    Returns:
        np.ndarray: Output array of shape (batch_size, out_channels, out_height, out_width)
    """
    # Your code here
    pass`,
        testCases: [
            {
                input: "x = np.array([[[[1, 1, 1], [1, 1, 1], [1, 1, 1]]]]), kernel = np.array([[[[1, 1], [1, 1]]]])",
                output: "array([[[[4, 4], [4, 4]]]])",
                description: "Basic test case with 3x3 input and 2x2 kernel"
            },
            {
                input: "x = np.array([[[[1, 2], [3, 4]]]]), kernel = np.array([[[[1, 1], [1, 1]]]])",
                output: "array([[[[10]]]])",
                description: "Small input with varied values"
            },
            {
                input: "x = np.ones((1, 1, 5, 5)), kernel = np.ones((1, 1, 3, 3)), stride=2, padding=1",
                output: "array([[[[9., 9., 9.], [9., 9., 9.], [9., 9., 9.]]]])",
                description: "Edge case: test with stride and padding"
            }
        ]
    },
    {
        id: 5,
        title: "Max Pooling 2D",
        description: "Implement 2D max pooling operation using NumPy. The function should perform max pooling over the input array with given kernel size and stride.",
        difficulty: "Medium",
        category: "Pooling",
        status: 'pending',
        template: `import numpy as np

def max_pool2d(x, kernel_size, stride=None):
    """
    Implement 2D max pooling operation.
    
    Args:
        x (np.ndarray): Input array of shape (batch_size, channels, height, width)
        kernel_size (int or tuple): Size of the pooling window
        stride (int or tuple, optional): Stride of the pooling operation
        
    Returns:
        np.ndarray: Output array after max pooling
    """
    # Your code here
    pass`,
        testCases: [
            {
                input: "x = np.array([[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]]]), kernel_size = 2, stride = 2",
                output: "array([[[[6, 8], [14, 16]]]])",
                description: "Basic test case with 4x4 input and 2x2 pooling"
            },
            {
                input: "x = np.array([[[[1, 2], [3, 4]]]]), kernel_size = 2, stride = 1",
                output: "array([[[[4]]]])",
                description: "Small input with stride 1"
            },
            {
                input: "x = np.zeros((1, 2, 3, 3)), kernel_size = 2, stride = 1",
                output: "array([[[[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]]]])",
                description: "Edge case: all zeros with multiple channels"
            }
        ]
    },
    {
        id: 6,
        title: "Softmax",
        description: "Implement the Softmax function using NumPy. The function should compute softmax values for each set of scores in x.",
        difficulty: "Easy",
        category: "Activation Functions",
        status: 'pending',
        template: `import numpy as np

def softmax(x):
    """
    Implement the Softmax function.
    
    Args:
        x (np.ndarray): Input array of shape (batch_size, num_classes)
        
    Returns:
        np.ndarray: Output array with softmax applied
    """
    # Your code here
    pass`,
        testCases: [
            {
                input: "np.array([[1, 2, 3]])",
                output: "array([[0.09003057, 0.24472847, 0.66524096]])",
                description: "Basic test case with 3 classes"
            },
            {
                input: "np.array([[1000, 2000, 3000]])",
                output: "array([[0., 0., 1.]])",
                description: "Edge case: very large values (numerical stability test)"
            },
            {
                input: "np.array([[1, 1, 1], [2, 2, 2]])",
                output: "array([[0.33333333, 0.33333333, 0.33333333], [0.33333333, 0.33333333, 0.33333333]])",
                description: "Multiple rows with equal values (uniform distribution)"
            }
        ]
    },
    {
        id: 7,
        title: "Cross Entropy Loss",
        description: "Implement the Cross Entropy Loss function using NumPy. The function should compute the cross entropy between predicted probabilities and true labels.",
        difficulty: "Medium",
        category: "Loss Functions",
        status: 'pending',
        template: `import numpy as np

def cross_entropy_loss(y_pred, y_true):
    """
    Implement Cross Entropy Loss.
    
    Args:
        y_pred (np.ndarray): Predicted probabilities of shape (batch_size, num_classes)
        y_true (np.ndarray): True labels of shape (batch_size,)
        
    Returns:
        float: Average cross entropy loss
    """
    # Your code here
    pass`,
        testCases: [
            {
                input: "y_pred = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]]), y_true = np.array([0, 1])",
                output: "0.35667494393873245",
                description: "Basic test case with 2 samples and 3 classes"
            },
            {
                input: "y_pred = np.array([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]]), y_true = np.array([0, 1, 2])",
                output: "0.10536051565782628",
                description: "Perfect predictions test case"
            },
            {
                input: "y_pred = np.array([[0.3, 0.3, 0.4], [0.1, 0.1, 0.8]]), y_true = np.array([2, 0])",
                output: "1.2039728043259361",
                description: "Edge case: incorrect predictions"
            }
        ]
    },
    {
        id: 8,
        title: "Tanh Activation",
        description: "Implement the Tanh activation function using NumPy. The function should return (exp(x) - exp(-x))/(exp(x) + exp(-x)) for each element in the input array.",
        difficulty: "Easy",
        category: "Activation Functions",
        status: 'pending',
        template: `import numpy as np

def tanh(x):
    """
    Implement the Tanh activation function.
    
    Args:
        x (np.ndarray): Input array
        
    Returns:
        np.ndarray: Output array with Tanh applied
    """
    # Your code here
    pass`,
        testCases: [
            {
                input: "np.array([-1, 0, 1])",
                output: "array([-0.76159416, 0., 0.76159416])",
                description: "Basic test case with positive and negative numbers"
            },
            {
                input: "np.array([[-10, 10], [100, -100]])",
                output: "array([[-1., 1.], [1., -1.]])",
                description: "Edge case: very large/small values (saturation test)"
            },
            {
                input: "np.zeros((2, 2))",
                output: "array([[0., 0.], [0., 0.]])",
                description: "Edge case: all zeros (tanh(0) = 0)"
            }
        ]
    },
    {
        id: 9,
        title: "Leaky ReLU",
        description: "Implement the Leaky ReLU activation function using NumPy. The function should return max(0.01x, x) for each element in the input array.",
        difficulty: "Easy",
        category: "Activation Functions",
        status: 'pending',
        template: `import numpy as np

def leaky_relu(x, negative_slope=0.01):
    """
    Implement the Leaky ReLU activation function.
    
    Args:
        x (np.ndarray): Input array
        negative_slope (float): Slope for negative values
        
    Returns:
        np.ndarray: Output array with Leaky ReLU applied
    """
    # Your code here
    pass`,
        testCases: [
            {
                input: "np.array([-1, 0, 1])",
                output: "array([-0.01, 0., 1.])",
                description: "Basic test case with positive and negative numbers"
            },
            {
                input: "np.array([[-2, 2], [-4, 4]]), negative_slope=0.2",
                output: "array([[-0.4, 2. ], [-0.8, 4. ]])",
                description: "Different negative slope value"
            },
            {
                input: "np.zeros((2, 2)), negative_slope=0.1",
                output: "array([[0., 0.], [0., 0.]])",
                description: "Edge case: all zeros"
            }
        ]
    },
    {
        id: 10,
        title: "1D Convolution",
        description: "Implement a 1D convolution operation using NumPy. The function should perform convolution between input and kernel with optional stride and padding.",
        difficulty: "Hard",
        category: "Convolution",
        status: 'pending',
        template: `import numpy as np

def conv1d(x, kernel, stride=1, padding=0):
    """
    Implement 1D convolution operation.
    
    Args:
        x (np.ndarray): Input array of shape (batch_size, in_channels, length)
        kernel (np.ndarray): Kernel array of shape (out_channels, in_channels, kernel_length)
        stride (int): Stride of the convolution
        padding (int): Padding size
        
    Returns:
        np.ndarray: Output array of shape (batch_size, out_channels, out_length)
    """
    # Your code here
    pass`,
        testCases: [
            {
                input: "x = np.array([[[1, 1, 1, 1]]]), kernel = np.array([[[1, 1]]])",
                output: "array([[[2, 2, 2]]])",
                description: "Basic test case with 4-length input and 2-length kernel"
            },
            {
                input: "x = np.array([[[1, 2, 3, 4, 5]]]), kernel = np.array([[[1, 2, 1]]]), stride=2",
                output: "array([[[4, 10]]])",
                description: "Test with stride"
            },
            {
                input: "x = np.array([[[1, 2, 3]]]), kernel = np.array([[[1, 1]]]), padding=1",
                output: "array([[[1, 3, 5, 3]]])",
                description: "Edge case: test with padding"
            }
        ]
    },
    {
        id: 11,
        title: "Average Pooling 2D",
        description: "Implement 2D average pooling operation using NumPy. The function should perform average pooling over the input array with given kernel size and stride.",
        difficulty: "Medium",
        category: "Pooling",
        status: 'pending',
        template: `import numpy as np

def avg_pool2d(x, kernel_size, stride=None):
    """
    Implement 2D average pooling operation.
    
    Args:
        x (np.ndarray): Input array of shape (batch_size, channels, height, width)
        kernel_size (int or tuple): Size of the pooling window
        stride (int or tuple, optional): Stride of the pooling operation
        
    Returns:
        np.ndarray: Output array after average pooling
    """
    # Your code here
    pass`,
        testCases: [
            {
                input: "x = np.array([[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]]]), kernel_size = 2, stride = 2",
                output: "array([[[[3.5, 5.5], [11.5, 13.5]]]])",
                description: "Basic test case with 4x4 input and 2x2 pooling"
            },
            {
                input: "x = np.ones((1, 1, 3, 3)), kernel_size = 2, stride = 1",
                output: "array([[[[1., 1.], [1., 1.]]]])",
                description: "All ones test with stride 1"
            },
            {
                input: "x = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]]), kernel_size = 2, stride = 1",
                output: "array([[[[2.5]], [[6.5]]]])",
                description: "Edge case: multiple channels"
            }
        ]
    },
    {
        id: 12,
        title: "Dropout",
        description: "Implement the Dropout operation using NumPy. The function should randomly zero some of the elements of the input tensor with probability p.",
        difficulty: "Medium",
        category: "Regularization",
        status: 'pending',
        template: `import numpy as np

def dropout(x, p=0.5, training=True):
    """
    Implement Dropout operation.
    
    Args:
        x (np.ndarray): Input array
        p (float): Probability of an element to be zeroed
        training (bool): If True, apply dropout. If False, return input as is
        
    Returns:
        np.ndarray: Output array after dropout
    """
    # Your code here
    pass`,
        testCases: [
            {
                input: "np.array([[1, 2, 3], [4, 5, 6]]), p=0.5, training=True, np.random.seed(42)",
                output: "array([[0., 4., 6.], [8., 0., 12.]])",
                description: "Test with random seed for reproducibility"
            },
            {
                input: "np.array([[1, 2, 3], [4, 5, 6]]), p=0, training=True",
                output: "array([[1., 2., 3.], [4., 5., 6.]])",
                description: "Edge case: p=0 (no dropout)"
            },
            {
                input: "np.array([[1, 2, 3], [4, 5, 6]]), p=0.5, training=False",
                output: "array([[1, 2, 3], [4, 5, 6]])",
                description: "Edge case: training=False (inference mode)"
            }
        ]
    },
    {
        id: 13,
        title: "Batch Normalization",
        description: "Implement Batch Normalization operation using NumPy. The function should normalize the input using running mean and variance.",
        difficulty: "Hard",
        category: "Normalization",
        status: 'pending',
        template: `import numpy as np

def batch_norm(x, running_mean, running_var, weight=None, bias=None, eps=1e-5, momentum=0.1):
    """
    Implement Batch Normalization operation.
    
    Args:
        x (np.ndarray): Input array of shape (batch_size, num_features, ...)
        running_mean (np.ndarray): Running mean of shape (num_features,)
        running_var (np.ndarray): Running variance of shape (num_features,)
        weight (np.ndarray, optional): Weight parameter of shape (num_features,)
        bias (np.ndarray, optional): Bias parameter of shape (num_features,)
        eps (float): A value added to the denominator for numerical stability
        momentum (float): The value used for the running_mean and running_var computation
        
    Returns:
        np.ndarray: Normalized output array
    """
    # Your code here
    pass`,
        testCases: [
            {
                input: "x = np.array([[1, 2], [3, 4]]), running_mean = np.array([0, 0]), running_var = np.array([1, 1]), weight = np.array([1, 1]), bias = np.array([0, 0])",
                output: "array([[1, 2], [3, 4]])",
                description: "Basic test case with 2x2 input"
            },
            {
                input: "x = np.array([[0, 0], [2, 2]]), running_mean = np.array([1, 1]), running_var = np.array([1, 1]), weight = np.array([2, 2]), bias = np.array([1, 1])",
                output: "array([[-1.,  -1.], [ 3.,  3.]])",
                description: "Test with non-zero mean and custom weight/bias"
            },
            {
                input: "x = np.ones((2, 3)), running_mean = np.zeros(3), running_var = np.ones(3), weight = None, bias = None",
                output: "array([[1., 1., 1.], [1., 1., 1.]])",
                description: "Edge case: no weight/bias parameters"
            }
        ]
    },
    {
        id: 14,
        title: "Layer Normalization",
        description: "Implement Layer Normalization operation using NumPy. The function should normalize the input across the last dimension.",
        difficulty: "Medium",
        category: "Normalization",
        status: 'pending',
        template: `import numpy as np

def layer_norm(x, weight=None, bias=None, eps=1e-5):
    """
    Implement Layer Normalization operation.
    
    Args:
        x (np.ndarray): Input array of shape (batch_size, ..., normalized_shape)
        weight (np.ndarray, optional): Weight parameter of shape (normalized_shape,)
        bias (np.ndarray, optional): Bias parameter of shape (normalized_shape,)
        eps (float): A value added to the denominator for numerical stability
        
    Returns:
        np.ndarray: Normalized output array
    """
    # Your code here
    pass`,
        testCases: [
            {
                input: "x = np.array([[1, 2], [3, 4]]), weight = np.array([1, 1]), bias = np.array([0, 0])",
                output: "array([[-0.70710678, 0.70710678], [-0.70710678, 0.70710678]])",
                description: "Basic test case with 2x2 input"
            },
            {
                input: "x = np.array([[1, 2, 3], [4, 5, 6]]), weight = np.array([2, 2, 2]), bias = np.array([1, 1, 1])",
                output: "array([[-1.22474487, 0., 1.22474487], [-1.22474487, 0., 1.22474487]])",
                description: "Test with custom weight and bias"
            },
            {
                input: "x = np.ones((2, 3)), weight = None, bias = None",
                output: "array([[0., 0., 0.], [0., 0., 0.]])",
                description: "Edge case: all ones input (zero output when normalized)"
            }
        ]
    },
    {
        id: 15,
        title: "MSE Loss",
        description: "Implement Mean Squared Error Loss function using NumPy. The function should compute the mean squared error between predictions and targets.",
        difficulty: "Easy",
        category: "Loss Functions",
        status: 'pending',
        template: `import numpy as np

def mse_loss(y_pred, y_true):
    """
    Implement Mean Squared Error Loss.
    
    Args:
        y_pred (np.ndarray): Predicted values
        y_true (np.ndarray): Target values
        
    Returns:
        float: Mean squared error loss
    """
    # Your code here
    pass`,
        testCases: [
            {
                input: "y_pred = np.array([1, 2, 3]), y_true = np.array([1, 2, 4])",
                output: "0.3333333333333333",
                description: "Basic test case with 3 samples"
            },
            {
                input: "y_pred = np.array([[1, 2], [3, 4]]), y_true = np.array([[1, 2], [3, 4]])",
                output: "0.0",
                description: "Edge case: perfect predictions (zero error)"
            },
            {
                input: "y_pred = np.array([10, 20, 30]), y_true = np.array([0, 0, 0])",
                output: "466.6666666666667",
                description: "Edge case: large error"
            }
        ]
    },
    {
        id: 16,
        title: "Nearest Neighbor Upsampling 2D",
        description: "Implement 2D nearest neighbor upsampling using NumPy. The function should increase the height and width of the input tensor by a given scale factor.",
        difficulty: "Medium",
        category: "Upsampling",
        status: 'pending',
        template: `import numpy as np

def upsample_nearest2d(x, scale_factor):
    \"\"\"
    Implement 2D nearest neighbor upsampling.

    Args:
        x (np.ndarray): Input array of shape (batch_size, channels, height, width)
        scale_factor (int or tuple): Multiplier for spatial dimensions (height, width)

    Returns:
        np.ndarray: Upsampled output array
    \"\"\"
    # Your code here
    pass`,
        testCases: [
            {
                input: "x = np.array([[[[1, 2], [3, 4]]]]), scale_factor=2",
                output: "array([[[[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]]]])",
                description: "Basic 2x2 input, scale factor 2"
            },
            {
                input: "x = np.array([[[[1]]]]), scale_factor=3",
                output: "array([[[[1, 1, 1], [1, 1, 1], [1, 1, 1]]]])",
                description: "1x1 input, scale factor 3"
            },
            {
                input: "x = np.array([[[[1, 2]]]]), scale_factor=(2, 1)",
                output: "array([[[[1, 2], [1, 2]]]])",
                description: "Different scale factors for height and width"
            }
        ]
    },
    {
        id: 17,
        title: "Negative Log Likelihood Loss (NLL Loss)",
        description: "Implement the Negative Log Likelihood (NLL) loss function using NumPy. Assumes the input contains log-probabilities. The function should compute the average NLL loss given predicted log-probabilities and true class indices.",
        difficulty: "Medium",
        category: "Loss Functions",
        status: 'pending',
        template: `import numpy as np

def nll_loss(log_probs, y_true):
    \"\"\"
    Implement Negative Log Likelihood Loss.

    Args:
        log_probs (np.ndarray): Log probabilities of shape (batch_size, num_classes)
        y_true (np.ndarray): True labels (indices) of shape (batch_size,)

    Returns:
        float: Average NLL loss
    \"\"\"
    # Your code here
    pass`,
        testCases: [
            {
                input: "log_probs = np.log(np.array([[0.1, 0.7, 0.2], [0.8, 0.1, 0.1]])), y_true = np.array([1, 0])",
                output: "0.2899092034016418", // (-np.log(0.7) - np.log(0.8)) / 2
                description: "Basic case with 2 samples, 3 classes"
            },
            {
                input: "log_probs = np.log(np.array([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])), y_true = np.array([0, 1, 2])",
                output: "0.10536051565782628", // (-np.log(0.9) * 3) / 3
                description: "High confidence predictions"
            },
            {
                input: "log_probs = np.log(np.array([[0.4, 0.3, 0.3], [0.3, 0.4, 0.3]])), y_true = np.array([0, 1])",
                output: "0.916290731874155", // (-np.log(0.4) - np.log(0.4)) / 2
                description: "Lower confidence probabilities"
            }
        ]
    },
    {
        id: 18,
        title: "Cosine Similarity",
        description: "Compute the pairwise cosine similarity between two sets of vectors using NumPy. The function should compute the cosine similarity matrix between rows of x1 and rows of x2.",
        difficulty: "Medium",
        category: "Distance Functions",
        status: 'pending',
        template: `import numpy as np

def cosine_similarity(x1, x2, eps=1e-8):
    \"\"\"
    Compute pairwise cosine similarity between row vectors in x1 and x2.

    Args:
        x1 (np.ndarray): First set of vectors, shape (N, D)
        x2 (np.ndarray): Second set of vectors, shape (M, D)
        eps (float): Small value to avoid division by zero.

    Returns:
        np.ndarray: Cosine similarity matrix of shape (N, M)
    \"\"\"
    # Your code here
    pass`,
        testCases: [
            {
                input: "x1 = np.array([[1., 1.], [0., 1.]]), x2 = np.array([[1., 0.], [1., 1.]])",
                output: "array([[0.70710678, 1.        ], [0.        , 1.        ]])",
                description: "Basic 2D vectors"
            },
            {
                input: "x1 = np.array([[1., 2., 3.]]), x2 = np.array([[1., 2., 3.], [-1., -2., -3.]])",
                output: "array([[ 1., -1.]])",
                description: "Identical and opposite vectors"
            },
            {
                input: "x1 = np.array([[1., 0., 0.]]), x2 = np.array([[0., 1., 0.], [0., 0., 1.]])",
                output: "array([[0., 0.]])",
                description: "Orthogonal vectors"
            }
        ]
    },
    {
        id: 19,
        title: "2D Transposed Convolution",
        description: "Implement a 2D transposed convolution (often called deconvolution) using NumPy. This operation is typically used to upsample feature maps.",
        difficulty: "Hard",
        category: "Convolution",
        status: 'pending',
        template: `import numpy as np

def conv_transpose2d(x, kernel, stride=1, padding=0):
    \"\"\"
    Implement 2D transposed convolution operation.

    Args:
        x (np.ndarray): Input array of shape (batch_size, in_channels, height, width)
        kernel (np.ndarray): Kernel array of shape (in_channels, out_channels, kernel_height, kernel_width)
        stride (int): Stride of the transposed convolution
        padding (int): Padding size (applied to the output conceptually)

    Returns:
        np.ndarray: Output array after transposed convolution
    \"\"\"
    # Your code here
    pass`,
        testCases: [
            {
                input: "x = np.array([[[[1]]]]), kernel = np.array([[[[1, 1], [1, 1]]]]), stride=1",
                output: "array([[[[1, 1], [1, 1]]]])",
                description: "Upsample 1x1 to 2x2"
            },
            {
                input: "x = np.array([[[[1, 2], [3, 4]]]]), kernel = np.array([[[[1]]]]), stride=2",
                output: "array([[[[1, 0, 2, 0], [0, 0, 0, 0], [3, 0, 4, 0], [0, 0, 0, 0]]]])",
                description: "2x2 input, 1x1 kernel, stride 2"
            },
            {
                input: "x = np.array([[[[1]]]]), kernel = np.array([[[[1]]]]), stride=1, padding=1", // Padding reduces output size
                output: "array([[[[1]]]])",
                description: "1x1 input/kernel, stride 1, padding 1 (no effective padding)"
            }
        ]
    },
    {
        id: 20,
        title: "Adaptive Average Pooling 2D",
        description: "Implement 2D adaptive average pooling using NumPy. The function should pool the input tensor to a fixed output size, regardless of the input size.",
        difficulty: "Medium",
        category: "Pooling",
        status: 'pending',
        template: `import numpy as np

def adaptive_avg_pool2d(x, output_size):
    \"\"\"
    Implement 2D adaptive average pooling.

    Args:
        x (np.ndarray): Input array of shape (batch_size, channels, height, width)
        output_size (int or tuple): The target output size (height, width).

    Returns:
        np.ndarray: Output array of shape (batch_size, channels, output_height, output_width)
    \"\"\"
    # Your code here
    pass`,
        testCases: [
            {
                input: "x = np.array([[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]]]), output_size=(2, 2)",
                output: "array([[[[ 3.5,  5.5], [11.5, 13.5]]]])",
                description: "Pool 4x4 input to 2x2 output"
            },
            {
                input: "x = np.arange(16).reshape(1, 1, 4, 4), output_size=1", // output_size=1 means (1, 1)
                output: "array([[[[ 7.5]]]])",
                description: "Pool 4x4 input to 1x1 output (global average pooling)"
            },
            {
                input: "x = np.ones((1, 2, 6, 6)), output_size=(3, 2)",
                output: "array([[[[1., 1.], [1., 1.], [1., 1.]], [[1., 1.], [1., 1.], [1., 1.]]]])",
                description: "Pool 6x6 input to 3x2 output, multiple channels"
            }
        ]
    },
    {
        id: 21,
        title: "Hardsigmoid Activation",
        description: "Implement the Hardsigmoid activation function using NumPy. It's a faster approximation of the sigmoid function.",
        difficulty: "Easy",
        category: "Activation Functions",
        status: 'pending',
        template: `import numpy as np

def hardsigmoid(x):
    \"\"\"
    Implement the Hardsigmoid activation function.
    Returns 0 if x <= -3, 1 if x >= 3, and (x/6 + 0.5) otherwise.

    Args:
        x (np.ndarray): Input array

    Returns:
        np.ndarray: Output array with Hardsigmoid applied
    \"\"\"
    # Your code here
    pass`,
        testCases: [
            {
                input: "np.array([-4, -3, 0, 3, 4])",
                output: "array([0., 0., 0.5, 1., 1.])",
                description: "Basic test covering all three regions"
            },
            {
                input: "np.array([-1.5, 1.5])",
                output: "array([0.25, 0.75])",
                description: "Values within the linear region"
            },
            {
                input: "np.zeros((2,2))",
                output: "array([[0.5, 0.5], [0.5, 0.5]])",
                description: "Edge case: all zeros"
            }
        ]
    },
    {
        id: 22,
        title: "Pad Tensor",
        description: "Implement padding for a tensor using NumPy. The function should pad the input tensor according to the specified padding amounts for each dimension.",
        difficulty: "Medium",
        category: "Utility",
        status: 'pending',
        template: `import numpy as np

def pad(x, pad_width, mode='constant', constant_values=0):
    \"\"\"
    Pad an N-dimensional tensor.

    Args:
        x (np.ndarray): Input tensor.
        pad_width (tuple): Tuple of tuples, ((before_dim1, after_dim1), ...). Specifies padding for each dimension.
        mode (str): Padding mode (e.g., 'constant', 'reflect', 'edge'). Defaults to 'constant'.
        constant_values (scalar): Value to use for constant padding. Defaults to 0.

    Returns:
        np.ndarray: Padded tensor.
    \"\"\"
    # Your code here
    # Hint: np.pad is very useful here!
    pass`,
        testCases: [
            {
                input: "x = np.array([[1, 2], [3, 4]]), pad_width=((1, 1), (1, 1))",
                output: "array([[0, 0, 0, 0], [0, 1, 2, 0], [0, 3, 4, 0], [0, 0, 0, 0]])",
                description: "Constant padding of 1 around a 2x2 array"
            },
            {
                input: "x = np.array([1, 2, 3]), pad_width=((2, 1))",
                output: "array([0, 0, 1, 2, 3, 0])",
                description: "1D array padding (2 before, 1 after)"
            },
            {
                input: "x = np.array([[1, 2]]), pad_width=((0, 0), (1, 3)), constant_values=-1",
                output: "array([[-1, 1, 2, -1, -1, -1]])",
                description: "Padding only the last dimension with a specific value"
            }
        ]
    },
    {
        id: 23,
        title: "Embedding Lookup",
        description: "Implement a simple embedding lookup using NumPy. Given an embedding matrix and indices, return the corresponding embedding vectors.",
        difficulty: "Medium",
        category: "Embedding",
        status: 'pending',
        template: `import numpy as np

def embedding(indices, embedding_matrix):
    \"\"\"
    Perform embedding lookup.

    Args:
        indices (np.ndarray): Array of indices to look up.
        embedding_matrix (np.ndarray): The embedding matrix (num_embeddings, embedding_dim).

    Returns:
        np.ndarray: The corresponding embedding vectors.
    \"\"\"
    # Your code here
    pass`,
        testCases: [
            {
                input: "indices = np.array([1, 0, 1]), embedding_matrix = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])",
                output: "array([[0.3, 0.4], [0.1, 0.2], [0.3, 0.4]])",
                description: "Basic lookup with 1D indices"
            },
            {
                input: "indices = np.array([[0, 2], [1, 0]]), embedding_matrix = np.arange(6).reshape(3, 2)", // Matrix: [[0, 1], [2, 3], [4, 5]]
                output: "array([[[0, 1], [4, 5]], [[2, 3], [0, 1]]])",
                description: "Lookup with 2D indices"
            },
            {
                input: "indices = np.array([0]), embedding_matrix = np.array([[10, 20]])",
                output: "array([[10, 20]])",
                description: "Single index lookup"
            }
        ]
    },
    {
        id: 24,
        title: "L1 Loss (Mean Absolute Error)",
        description: "Implement the L1 Loss function (Mean Absolute Error) using NumPy. It calculates the average absolute difference between predicted and true values.",
        difficulty: "Easy",
        category: "Loss Functions",
        status: 'pending',
        template: `import numpy as np

def l1_loss(y_pred, y_true):
    \"\"\"
    Implement L1 Loss (Mean Absolute Error).

    Args:
        y_pred (np.ndarray): Predicted values.
        y_true (np.ndarray): True values.

    Returns:
        float: The mean absolute error.
    \"\"\"
    # Your code here
    pass`,
        testCases: [
            {
                input: "y_pred = np.array([1, 2, 3]), y_true = np.array([1, 2, 4])",
                output: "0.6666666666666666", // (|1-1| + |2-3| + |3-2|) / 3 = (0 + 1 + 1) / 3
                description: "Basic 1D case"
            },
            {
                input: "y_pred = np.array([[1, 1], [1, 1]]), y_true = np.array([[1, 1], [1, 1]])",
                output: "0.0",
                description: "Edge case: perfect predictions (zero loss)"
            },
            {
                input: "y_pred = np.array([1, 2]), y_true = np.array([-1, -2])",
                output: "3.0", // (|1 - (-1)| + |2 - (-2)|) / 2 = (2 + 4) / 2
                description: "Positive predictions, negative targets"
            }
        ]
    }
];

// DOM Elements
const questionsPanel = document.getElementById('questionsPanel');
const toggleQuestionsBtn = document.getElementById('toggleQuestions');
const questionsGrid = document.getElementById('questionsGrid');
const questionTitle = document.getElementById('questionTitle');
const questionContent = document.getElementById('questionContent');
const testCasesContainer = document.getElementById('testCases');
const runCodeButton = document.getElementById('runCode');
const outputElements = [document.getElementById('output'), document.getElementById('output2'), document.getElementById('output3')];
const inputDataElements = [document.getElementById('inputData'), document.getElementById('inputData2'), document.getElementById('inputData3')];
const expectedResultElements = [document.getElementById('expectedResult'), document.getElementById('expectedResult2'), document.getElementById('expectedResult3')];
const tabs = document.querySelectorAll('.tab:not(.add-tab)');
const tabContents = document.querySelectorAll('.tab-content');
const searchBar = document.getElementById('searchBar');
const filterBtn = document.getElementById('filterBtn');
const filterDropdown = document.getElementById('filterDropdown');
const themeToggle = document.getElementById('themeToggle');
const codeView = document.querySelector('.code-view'); // Get reference to code-view

let currentQuestion = null;

// Filter state
let activeFilters = {
    difficulty: ['Easy', 'Medium', 'Hard'],
    category: [
        'Activation Functions',
        'Linear Layers',
        'Convolution',
        'Pooling',
        'Regularization',
        'Normalization',
        'Loss Functions',
        'Upsampling',
        'Distance Functions',
        'Embedding',
        'Utility'
    ]
};

// Resizable Panel Logic
let isResizing = false;
let startX, startWidth;

const savedPanelWidth = localStorage.getItem('panelWidth');
if (savedPanelWidth) {
    questionsPanel.style.width = savedPanelWidth + 'px';
}

questionsPanel.addEventListener('mousedown', (e) => {
    // Check if the mousedown is near the right edge (e.g., within 10px)
    // This provides a more specific drag handle than the whole panel
    const rect = questionsPanel.getBoundingClientRect();
    if (e.clientX >= rect.right - 10 && e.clientX <= rect.right) {
        isResizing = true;
        startX = e.clientX;
        startWidth = questionsPanel.offsetWidth;
        document.body.style.cursor = 'col-resize'; // Apply cursor to whole body during drag
        document.body.style.userSelect = 'none'; // Prevent text selection globally
        // Optional: Add a class for visual feedback during resize
        questionsPanel.classList.add('resizing'); 
    }
});

document.addEventListener('mousemove', (e) => {
    if (!isResizing) return;
    
    const currentX = e.clientX;
    const diffX = currentX - startX;
    let newWidth = startWidth + diffX;

    // Clamp width within min/max defined in CSS (or define here)
    const minWidth = parseInt(getComputedStyle(questionsPanel).minWidth, 10);
    const maxWidth = parseInt(getComputedStyle(questionsPanel).maxWidth, 10);
    if (newWidth < minWidth) newWidth = minWidth;
    if (maxWidth && newWidth > maxWidth) newWidth = maxWidth;

    questionsPanel.style.width = newWidth + 'px';
});

document.addEventListener('mouseup', () => {
    if (isResizing) {
        isResizing = false;
        document.body.style.cursor = 'default'; // Reset body cursor
        document.body.style.userSelect = 'auto'; // Re-enable selection
        questionsPanel.classList.remove('resizing'); // Remove visual feedback class
        
        // Persist the new width
        localStorage.setItem('panelWidth', questionsPanel.offsetWidth);

        // Important: Trigger a resize event so CodeMirror or other components can adjust
        window.dispatchEvent(new Event('resize'));
    }
});

// Toggle questions panel
toggleQuestionsBtn.addEventListener('click', () => {
    questionsPanel.classList.toggle('collapsed');
    toggleQuestionsBtn.textContent = questionsPanel.classList.contains('collapsed') ? '☰' : '×';
    
    // Allow the content to resize properly
    setTimeout(() => {
        window.dispatchEvent(new Event('resize'));
    }, 300);
});

// Toggle filter dropdown
filterBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    filterBtn.classList.toggle('active');
    filterDropdown.classList.toggle('show');
});

// Close filter dropdown when clicking outside
document.addEventListener('click', (e) => {
    if (!filterDropdown.contains(e.target) && !filterBtn.contains(e.target)) {
        filterBtn.classList.remove('active');
        filterDropdown.classList.remove('show');
    }
});

// Handle filter changes
filterDropdown.addEventListener('change', (e) => {
    const checkbox = e.target;
    const value = checkbox.value;
    const type = checkbox.closest('.filter-section').querySelector('h3').textContent.toLowerCase();
    
    if (checkbox.checked) {
        activeFilters[type].push(value);
    } else {
        activeFilters[type] = activeFilters[type].filter(item => item !== value);
    }
    
    populateQuestions();
});

// Populate questions list
function populateQuestions() {
    // --- Get Filter Values ---
    const selectedDifficulties = Array.from(filterDropdown.querySelectorAll('input[type="checkbox"][value^="Easy"], input[type="checkbox"][value^="Medium"], input[type="checkbox"][value^="Hard"]'))
                                    .filter(cb => cb.checked)
                                    .map(cb => cb.value);
    const selectedCategories = Array.from(filterDropdown.querySelectorAll('input[type="checkbox"]:not([value^="Easy"]):not([value^="Medium"]):not([value^="Hard"])'))
                                   .filter(cb => cb.checked)
                                   .map(cb => cb.value);
    // --- Get Search Term ---
    const searchTerm = searchBar.value.toLowerCase();

    questionsGrid.innerHTML = ''; // Clear existing questions

    // --- Filter Questions ---
    const filteredQuestions = questions.filter(q => {
        const difficultyMatch = selectedDifficulties.length === 0 || selectedDifficulties.includes(q.difficulty);
        const categoryMatch = selectedCategories.length === 0 || selectedCategories.includes(q.category);
        // Add search term filtering (title or description)
        const searchMatch = searchTerm === '' ||
                            q.title.toLowerCase().includes(searchTerm) ||
                            q.description.toLowerCase().includes(searchTerm);

        return difficultyMatch && categoryMatch && searchMatch; // Combine all filters
    });

    // --- Render Filtered Questions ---
    if (filteredQuestions.length === 0) {
        questionsGrid.innerHTML = '<p style="padding: 15px; color: var(--text-secondary);">No questions match your criteria.</p>';
    } else {
        filteredQuestions.forEach(question => {
            const card = document.createElement('div');
            card.className = 'question-card';
            card.dataset.questionId = question.id;
            card.innerHTML = `
                <div class="question-card-content">
                    <h3>${question.title}</h3>
                    <span class="difficulty ${question.difficulty.toLowerCase()}">${question.difficulty}</span>
                    <span class="status ${question.status}">${question.status}</span>
                </div>
            `;
            card.addEventListener('click', () => loadQuestion(question.id));
            questionsGrid.appendChild(card);
        });
    }
}

// Helper function to format array/value strings for display (removes array wrappers)
function formatValueDisplayString(str) {
    if (!str) return str;
    str = str.trim();
    // Remove 'np.array(' or 'array(' wrappers
    if (str.startsWith('np.array(') && str.endsWith(')')) {
        str = str.substring(9, str.length - 1);
    } else if (str.startsWith('array(') && str.endsWith(')')) {
         str = str.substring(6, str.length - 1);
    }
    // Note: Robustly removing dtype info is complex with regex, skipping for now.
    return str.trim();
}

// Helper function to format the multi-argument input display
function formatInputDisplay(inputString) {
    if (!inputString || typeof inputString !== 'string') return inputString;
    inputString = inputString.trim();

    // Handle single value inputs (no '=')
    if (!inputString.includes('=')) {
        return formatValueDisplayString(inputString);
    }

    // Split arguments carefully, respecting brackets/parentheses
    let parts = [];
    let balance = 0;
    let currentPartStart = 0;
    for (let i = 0; i < inputString.length; i++) {
        const char = inputString[i];
        if (char === '(' || char === '[' || char === '{') {
            balance++;
        } else if (char === ')' || char === ']' || char === '}') {
            balance--;
        } else if (char === ',' && balance === 0) {
            parts.push(inputString.substring(currentPartStart, i).trim());
            currentPartStart = i + 1;
        }
    }
    parts.push(inputString.substring(currentPartStart).trim()); // Add the last part

    // Format each part as "name: value"
    let result = [];
    for (const part of parts) {
        const eqIndex = part.indexOf('=');
        if (eqIndex > -1) {
            const name = part.substring(0, eqIndex).trim();
            const valueExpr = part.substring(eqIndex + 1).trim();
            result.push(`${name}: ${formatValueDisplayString(valueExpr)}`);
        } else {
             // Fallback for parts without '=', though shouldn't happen with the logic above
            result.push(formatValueDisplayString(part));
        }
    }
    return result.join('\n'); // Use newline characters for display in <pre>
}

// Load question into code view
function loadQuestion(questionId) {
    const question = questions.find(q => q.id === questionId);

    if (!question) {
        console.error("Question not found:", questionId);
        questionTitle.textContent = "Error: Question not found";
        questionContent.innerHTML = "";
        testCasesContainer.innerHTML = "";
        editor.setValue("# Question not found");
        if (codeView) codeView.classList.remove('question-loaded'); // Show welcome screen on error
        return;
    }

    // Question found, show the code container
    if (codeView) codeView.classList.add('question-loaded');

    currentQuestion = question;

    questionTitle.textContent = question.title;
    questionContent.innerHTML = `
        <p>${question.description}</p>
        <div class="template">
            <h3>Template</h3>
            <pre><code>${question.template}</code></pre>
        </div>
    `;
    
    // Populate test cases display (format expected output)
    testCasesContainer.innerHTML = question.testCases.map((testCase, index) => `
        <div class="test-case">
            <h4>Test Case ${index + 1}</h4>
            <p>${testCase.description}</p>
            <div class="test-case-content">
                <pre data-type="Input"><code>${testCase.input}</code></pre> // Keep raw input here
                <pre data-type="Expected Output"><code>${formatValueDisplayString(testCase.output)}</code></pre>
            </div>
        </div>
    `).join('');
    
    // Set editor content
    editor.setValue(question.template);
    
    // Set input data and expected result for all tabs (format both)
    const testCasesData = question.testCases;
    
    // Update first tab
    if (testCasesData[0]) {
        inputDataElements[0].textContent = formatInputDisplay(testCasesData[0].input) || 'No test input available';
        expectedResultElements[0].textContent = formatValueDisplayString(testCasesData[0].output) || 'No expected output available';
    } else {
        inputDataElements[0].textContent = 'No test input available';
        expectedResultElements[0].textContent = 'No expected output available';
    }
    
    // Update second tab
    if (testCasesData[1]) {
        inputDataElements[1].textContent = formatInputDisplay(testCasesData[1].input) || 'No test input available';
        expectedResultElements[1].textContent = formatValueDisplayString(testCasesData[1].output) || 'No expected output available';
    } else {
        inputDataElements[1].textContent = 'No test input available';
        expectedResultElements[1].textContent = 'No expected output available';
    }
    
    // Update third tab
    if (testCasesData[2]) {
        inputDataElements[2].textContent = formatInputDisplay(testCasesData[2].input) || 'No test input available';
        expectedResultElements[2].textContent = formatValueDisplayString(testCasesData[2].output) || 'No expected output available';
    } else {
        inputDataElements[2].textContent = 'No test input available';
        expectedResultElements[2].textContent = 'No expected output available';
    }
    
    // Reset output displays
    outputElements.forEach(out => {
        if (out) { // Check if element exists
            out.textContent = 'Run code to see output';
            out.style.color = '';
        }
    });
    
    // On mobile, collapse the questions panel after selection
    if (window.innerWidth <= 768) {
        questionsPanel.classList.add('collapsed');
        toggleQuestionsBtn.textContent = '☰';
    }
    
    // Update scroll indicators
    setTimeout(updateScrollIndicators, 100);
}

// Add scroll indicator functionality
function updateScrollIndicators() {
    const ioContents = document.querySelectorAll('.io-content');
    
    ioContents.forEach(content => {
        // Check if element is scrollable
        const isScrollable = content.scrollHeight > content.clientHeight + 5; // Add small buffer
        
        // Add or remove scrollable class
        if (isScrollable) {
            content.classList.add('scrollable');
        } else {
            content.classList.remove('scrollable');
        }
        
        // Add scroll event listener to update active state
        content.addEventListener('scroll', () => {
            // Check if scrolled to bottom (with a small tolerance)
            if (content.scrollTop + content.clientHeight >= content.scrollHeight - 15) {
                content.classList.remove('scrollable');
            } else {
                content.classList.add('scrollable');
            }
        });
    });
}

// Add event listeners for tabs
tabs.forEach((tab, index) => {
    tab.addEventListener('click', () => {
        // Remove active class from all tabs and contents
        tabs.forEach(t => t.classList.remove('active'));
        tabContents.forEach(c => c.classList.remove('active'));

        // Add active class to the clicked tab and corresponding content
        tab.classList.add('active');
        const tabId = tab.getAttribute('data-tab'); // e.g., 'case1', 'case2'
        const correspondingContent = document.getElementById(tabId);
        if (correspondingContent) {
            correspondingContent.classList.add('active');
        }

        // --- Reset the output area for the activated tab --- 
        if (outputElements[index]) { // Check if the output element exists
            outputElements[index].textContent = 'Run code to see output'; 
            outputElements[index].style.color = ''; // Reset color styling
            updateScrollIndicators(); // Call the correct function to refresh all indicators
        }
        // Also reset input/expected for consistency visually when just clicking tabs?
        // Maybe not, loadQuestion handles the placeholder text for those.
        // Focus on resetting the *result* of a previous run.
    });
});

// Run after window resize to recalculate scroll indicators
window.addEventListener('resize', () => {
    setTimeout(updateScrollIndicators, 200);
});

// Function to execute code via API
async function executeCode(code, testInput) {
    try {
        const response = await fetch('/api/run_code', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                code: code,
                test_input: testInput
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        return result;
    } catch (error) {
        console.error('Error:', error);
        return {
            status: 'error',
            output: `Error: ${error.message}`
        };
    }
}

// Theme toggle functionality
function applyTheme(theme) {
    if (theme === 'dark') {
        document.body.classList.add('dark-mode');
        themeToggle.textContent = '🌙'; // Moon icon for dark mode
    } else {
        document.body.classList.remove('dark-mode');
        themeToggle.textContent = '☀️'; // Sun icon for light mode
    }
    // Add a small delay before updating scroll indicators to allow CSS transitions
    setTimeout(updateScrollIndicators, 350);
}

themeToggle.addEventListener('click', () => {
    document.body.classList.toggle('dark-mode');
    const newTheme = document.body.classList.contains('dark-mode') ? 'dark' : 'light';
    localStorage.setItem('theme', newTheme);
    applyTheme(newTheme);
});

// Apply saved theme on initial load
const savedTheme = localStorage.getItem('theme') || 'light'; // Default to light
applyTheme(savedTheme);

// Helper function to set initial filter checkbox states
function updateFilterDropdownCheckboxes() {
    if (!filterDropdown) return; // Exit if filter dropdown doesn't exist

    const checkboxes = filterDropdown.querySelectorAll('input[type="checkbox"]');
    checkboxes.forEach(checkbox => {
        // Set initial state based on HTML 'checked' attribute or default to checked
        // This example assumes the HTML accurately reflects the desired default.
        // If you loaded saved preferences, you'd apply them here.
        if (checkbox.hasAttribute('checked')) {
            checkbox.checked = true;
        } else {
             // checkbox.checked = false; // Or keep default if not specified
        }
        // For simplicity now, let's ensure all are checked initially, matching HTML
        checkbox.checked = true; 
    });
}

// Initial setup on DOMContentLoaded
document.addEventListener('DOMContentLoaded', () => {
    // --- Theme Initialization ---
    const currentTheme = localStorage.getItem('theme');
    // Default to dark mode if no theme saved OR if saved theme is dark
    if (currentTheme === null || currentTheme === 'dark') {
        document.body.classList.add('dark-mode');
        localStorage.setItem('theme', 'dark'); // Ensure dark is saved
        if (themeToggle) themeToggle.textContent = '☀️'; // Sun icon = dark mode on (click for light)
    } else {
        // Explicitly set light mode if saved theme is light
        document.body.classList.remove('dark-mode');
        // No need to set localStorage here, it's already 'light'
        if (themeToggle) themeToggle.textContent = '🌙'; // Moon icon = light mode on (click for dark)
    }

    // Populate questions on initial load
    populateQuestions();

    // Select the first available question by default
    let questionLoadedSuccessfully = false;
    const firstAvailableQuestion = questions.find(q => {
         // Find first question matching default filters (if any apply)
         // This logic might need refinement if default filters are complex
         const initialDifficulties = Array.from(filterDropdown.querySelectorAll('input[type="checkbox"][value^="Easy"], input[type="checkbox"][value^="Medium"], input[type="checkbox"][value^="Hard"]')).filter(cb => cb.checked).map(cb => cb.value);
         const initialCategories = Array.from(filterDropdown.querySelectorAll('input[type="checkbox"]:not([value^="Easy"]):not([value^="Medium"]):not([value^="Hard"])')).filter(cb => cb.checked).map(cb => cb.value);
         const difficultyMatch = initialDifficulties.length === 0 || initialDifficulties.includes(q.difficulty);
         const categoryMatch = initialCategories.length === 0 || initialCategories.includes(q.category);
         return difficultyMatch && categoryMatch;
    });

    if (firstAvailableQuestion) {
        loadQuestion(firstAvailableQuestion.id);
        questionLoadedSuccessfully = true;
    } else if (questions.length > 0) {
        loadQuestion(questions[0].id);
        questionLoadedSuccessfully = true; // Assume success if questions exist
    }

    // Ensure welcome screen is shown if no question was loaded
    if (!questionLoadedSuccessfully && codeView) {
        codeView.classList.remove('question-loaded');
    }

    // Apply initial theme to CodeMirror
    if (editor) {
        editor.setOption("theme", document.body.classList.contains('dark-mode') ? "monokai" : "default");
    }

    // Initialize filter dropdown state
    updateFilterDropdownCheckboxes();
    if (filterBtn) filterBtn.classList.remove('active');

    // Default state (no class) shows welcome, loadQuestion adds class on success.
});

// Theme Toggle Event Listener
if (themeToggle) {
    themeToggle.addEventListener('click', () => {
        document.body.classList.toggle('dark-mode');
        const newTheme = document.body.classList.contains('dark-mode') ? 'dark' : 'light';
        localStorage.setItem('theme', newTheme);
        themeToggle.textContent = newTheme === 'dark' ? '☀️' : '🌙';

        // Update CodeMirror theme
        if (editor) {
            editor.setOption("theme", newTheme === 'dark' ? "monokai" : "default");
        }
    });
}

// Add event listener for the search bar input for live filtering
// This listener should already exist and call populateQuestions correctly.
// Ensure it is present and correctly attached if issues persist.
if (searchBar) { // Defensive check
    searchBar.addEventListener('input', populateQuestions);
}

// Call when content is updated
function refreshOutput() {
    setTimeout(updateScrollIndicators, 100);
}

// Add observer for changes to output
const outputObserver = new MutationObserver(refreshOutput);
const observerConfig = { childList: true, characterData: true, subtree: true };

// Observe each output element
document.querySelectorAll('.io-content').forEach(content => {
    outputObserver.observe(content, observerConfig);
});

// Initial update
updateScrollIndicators();

// Run after code execution
runCodeButton.addEventListener('click', async () => {
    const code = editor.getValue();
    
    // Get all output elements
    const outputs = [
        document.getElementById('output'),
        document.getElementById('output2'),
        document.getElementById('output3')
    ];
    
    // Set all to "Running..."
    outputs.forEach(out => {
        if (out) { // Check if element exists
            out.textContent = 'Running...';
            out.style.color = '#ddd';
        }
    });
    
    // Get the current question's test cases
    const currentTitle = questionTitle.textContent;
    const currentQuestion = questions.find(q => q.title === currentTitle);
    const testCasesData = currentQuestion?.testCases || [];
    
    let allPassed = true; // Flag to track if all tests passed
    let attempted = false; // Flag to track if any test was run

    // Run each test case
    for (let i = 0; i < 3; i++) {
        const outputElement = outputs[i];
        if (!outputElement) continue; // Skip if output element doesn't exist
        
        if (i >= testCasesData.length) {
            outputElement.textContent = 'No test case available';
            continue;
        }
        
        attempted = true; // Mark that at least one test case was available and attempted
        const testCase = testCasesData[i];
        const formattedExpectedOutput = formatValueDisplayString(testCase.output);
        
        try {
            // Call the renamed function executeCode
            const result = await executeCode(code, testCase.input); 
            
            if (result.status === 'error') {
                outputElement.style.color = '#ff4444'; // Red for error
                outputElement.textContent = result.output;
                allPassed = false; // Mark as failed if any test case errors
            } else {
                const actualOutput = result.output ? result.output.trim() : '';
                outputElement.textContent = actualOutput;
                
                // Check if output matches formatted expected result
                if (actualOutput === formattedExpectedOutput) {
                    outputElement.style.color = '#4caf50'; // Green for pass
                } else {
                    outputElement.style.color = '#ffcc00'; // Yellow/Orange for mismatch
                    allPassed = false; // Mark as failed if any test case mismatches
                }
            }
        } catch (error) {
            outputElement.style.color = '#ff4444'; // Red for error
            outputElement.textContent = `Error: ${error.message}`;
            allPassed = false; // Mark as failed on exception
        }
    }
    
    // Update question status
    if (currentQuestion && attempted) {
        if (allPassed) {
            currentQuestion.status = 'completed';
        } else {
            currentQuestion.status = 'ongoing';
        }
        populateQuestions(); // Refresh the question list to show updated status
    }

    // Update scroll indicators after all outputs are populated
    setTimeout(refreshOutput, 500); 
});

// Add event listener for the search bar
searchBar.addEventListener('input', populateQuestions); 