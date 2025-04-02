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
    }
];

// DOM Elements
const questionsPanel = document.getElementById('questionsPanel');
const toggleQuestionsBtn = document.getElementById('toggleQuestions');
const questionsGrid = document.getElementById('questionsGrid');
const questionTitle = document.getElementById('questionTitle');
const questionContent = document.getElementById('questionContent');
const testCases = document.getElementById('testCases');
const runCode = document.getElementById('runCode');
const output = document.getElementById('output');
const inputData = document.getElementById('inputData');
const expectedResult = document.getElementById('expectedResult');
const filterBtn = document.getElementById('filterBtn');
const filterDropdown = document.getElementById('filterDropdown');

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
        'Loss Functions'
    ]
};

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

// Populate questions grid
function populateQuestions() {
    questionsGrid.innerHTML = '';
    const filteredQuestions = questions.filter(question => {
        return activeFilters.difficulty.includes(question.difficulty) &&
               activeFilters.category.includes(question.category);
    });
    
    filteredQuestions.forEach(question => {
        const card = document.createElement('div');
        card.className = 'question-card';
        card.innerHTML = `
            <div class="question-card-content">
                <h3>${question.title}</h3>
                <span class="difficulty ${question.difficulty.toLowerCase()}">${question.difficulty}</span>
            </div>
        `;
        card.addEventListener('click', () => loadQuestion(question));
        questionsGrid.appendChild(card);
    });
}

// Load question into code view
function loadQuestion(question) {
    questionTitle.textContent = question.title;
    questionContent.innerHTML = `
        <p>${question.description}</p>
        <div class="template">
            <h3>Template</h3>
            <pre><code>${question.template}</code></pre>
        </div>
    `;
    
    // Populate test cases
    testCases.innerHTML = question.testCases.map((testCase, index) => `
        <div class="test-case">
            <h4>Test Case ${index + 1}</h4>
            <p>${testCase.description}</p>
            <div class="test-case-content">
                <pre data-type="Input"><code>${testCase.input}</code></pre>
                <pre data-type="Expected Output"><code>${testCase.output}</code></pre>
            </div>
        </div>
    `).join('');
    
    // Set editor content
    editor.setValue(question.template);
    
    // Set input data and expected result for all tabs
    const testCasesData = question.testCases;
    
    // Update first tab
    if (testCasesData[0]) {
        inputData.textContent = testCasesData[0].input || 'No test input available';
        expectedResult.textContent = testCasesData[0].output || 'No expected output available';
    }
    
    // Update second tab
    if (testCasesData[1]) {
        document.getElementById('inputData2').textContent = testCasesData[1].input || 'No test input available';
        document.getElementById('expectedResult2').textContent = testCasesData[1].output || 'No expected output available';
    }
    
    // Update third tab
    if (testCasesData[2]) {
        document.getElementById('inputData3').textContent = testCasesData[2].input || 'No test input available';
        document.getElementById('expectedResult3').textContent = testCasesData[2].output || 'No expected output available';
    }
    
    // Reset output displays
    output.textContent = 'Run code to see output';
    document.getElementById('output2').textContent = 'Run code to see output';
    document.getElementById('output3').textContent = 'Run code to see output';
    
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

// Make tab switching more robust
function initTabs() {
    const tabs = document.querySelectorAll('.tab');
    
    tabs.forEach(tab => {
        if (!tab.classList.contains('add-tab')) {
            tab.addEventListener('click', () => {
                // Remove active class from all tabs and tab contents
                document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
                
                // Add active class to clicked tab
                tab.classList.add('active');
                
                // Show corresponding tab content
                const tabId = tab.getAttribute('data-tab');
                document.getElementById(tabId).classList.add('active');
                
                // Update scroll indicators when tab changes
                setTimeout(updateScrollIndicators, 100);
            });
        }
    });
}

// Run after window resize to recalculate scroll indicators
window.addEventListener('resize', () => {
    setTimeout(updateScrollIndicators, 200);
});

// Run code with better error handling for output and run all test cases simultaneously
runCode.addEventListener('click', async () => {
    const code = editor.getValue();
    
    // Get all output elements
    const outputs = [
        document.getElementById('output'),
        document.getElementById('output2'),
        document.getElementById('output3')
    ];
    
    // Set all to "Running..."
    outputs.forEach(out => {
        out.textContent = 'Running...';
        out.style.color = '#ddd';
    });
    
    // Get the current question's test cases
    const currentTitle = questionTitle.textContent;
    const currentQuestion = questions.find(q => q.title === currentTitle);
    const testCases = currentQuestion?.testCases || [];
    
    // Run each test case
    for (let i = 0; i < 3; i++) {
        if (i >= testCases.length) {
            outputs[i].textContent = 'No test case available';
            continue;
        }
        
        const testCase = testCases[i];
        
        try {
            const response = await fetch('http://localhost:5000/run_code', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                mode: 'cors',
                credentials: 'omit',
                body: JSON.stringify({ 
                    code,
                    test_input: testCase.input
                })
            });
            
            const result = await response.json();
            
            if (result.status === 'error') {
                outputs[i].style.color = '#ff4444';
                outputs[i].textContent = result.output;
            } else {
                outputs[i].style.color = '#ddd';
                outputs[i].textContent = result.output;
                
                // Check if output matches expected result
                if (result.output.trim() === testCase.output.trim()) {
                    outputs[i].style.color = '#4caf50';
                }
            }
        } catch (error) {
            outputs[i].style.color = '#ff4444';
            outputs[i].textContent = `Error: Could not connect to the server. Make sure the Python server is running on port 5000.`;
        }
    }
    
    // Update scroll indicators after all outputs are populated
    setTimeout(updateScrollIndicators, 100);
});

// Initialize the application
populateQuestions();
initTabs(); // Initialize tabs

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

// Run when tab is changed
document.querySelectorAll('.tab').forEach(tab => {
    if (!tab.classList.contains('add-tab')) {
        tab.addEventListener('click', refreshOutput);
    }
});

// Run after code execution
runCode.addEventListener('click', function() {
    setTimeout(refreshOutput, 500);
}); 