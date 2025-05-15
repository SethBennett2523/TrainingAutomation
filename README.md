# Training Automation

<!-- 
Seth Bennett 
Thomas12.Bennett@live.uwe.ac.uk

UWE-AI Perception Team lead
-->
A series of programs to simplify the process of designing and training YOLO models for UWE AI

---

## Dataset

This project was built for the FSOCO `bounding boxes` dataset.
The FSOCO dataset can be downloaded directly [here](http://fsoco.cs.uni-freiburg.de/datasets/fsoco_bounding_boxes_train.zip) or torrented using this magnet link:

```magnet
magnet:?xt=urn:btih:aedcdd632f60b698ed9a94d2f5c4c145aa06aad7&dn=fsoco_bounding_boxes_train.zip&tr=udp%3a%2f%2ftracker.torrent.eu.org%3a451%2fannounce
```

### Links

- [**Website**](https://fsoco.github.io/fsoco-dataset/)
- **Datasets**
  - *Direct Download*
    - [**Bounding Boxes**](http://fsoco.cs.uni-freiburg.de/datasets/fsoco_bounding_boxes_train.zip)
    - [**Segmentation**](http://fsoco.cs.uni-freiburg.de/datasets/fsoco_segmentation_train.zip)
  - *Magnet*
    - **Bounding Boxes** `magnet:?xt=urn:btih:aedcdd632f60b698ed9a94d2f5c4c145aa06aad7&dn=fsoco_bounding_boxes_train.zip&tr=udp%3a%2f%2ftracker.torrent.eu.org%3a451%2fannounce`
    - **Segmentation** `magnet:?xt=urn:btih:aa9643c161f2db9056290af40ef8433e81fb5610&dn=fsoco_segmentation_train.zip&tr=udp%3A%2F%2Ftracker.torrent.eu.org%3A451%2Fannounce`

---

## Installation

This project supports multiple platforms including Windows and Linux with both NVIDIA and AMD GPUs.

### Prerequisites

- Python 3.8 or newer
- CUDA 11.8+ (for NVIDIA GPUs) or ROCm 5.6+ (for AMD GPUs on Linux)
- 8GB+ RAM

### Basic Installation

1. Clone the repository

   ```bash
   git clone https://github.com/SethBennett2523/TrainingAutomation.git
   cd TrainingAutomation
   ```

2. Install core dependencies

   ```bash
   pip install -r requirements.txt
   ```

3. Install platform-specific PyTorch:

   **Windows with NVIDIA GPU:**

   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

   **Windows with AMD GPU:**

   ```bash
   # Standard PyTorch (CPU version) - GPU support through DirectML if needed
   pip install torch torchvision
   # Optional: pip install torch-directml
   ```

   **Linux with NVIDIA GPU:**

   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

   **Linux with AMD GPU (ROCm):**

   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm5.6
   ```

   **CPU only (any platform):**

   ```bash
   pip install torch torchvision
   ```

## Usage

Basic usage instructions for training models:

```bash
python main.py --mode train --config config.yaml --data data/data.yaml
```

For hyperparameter optimisation:

```bash
python main.py --mode optimise --trials 25 --config config.yaml --data data/data.yaml
```

See all available options:

```bash
python main.py --help
```

---

## Contribution Guidelines

I welcome contributions to improve the Training Automation system. Please follow these guidelines:

### Development Environment Setup

1. Fork the repository and clone your fork
2. Install development dependencies:

   ```bash
   pip install -r requirements.txt
   pip install pytest pytest-cov mypy flake8
   ```

3. Create a feature branch from the `devmain` branch

### Pull Request Process

1. Update the documentation to reflect your changes
2. Add or update tests as appropriate
3. Ensure all tests pass with `pytest tests/`
4. Submit a pull request to the `devmain` branch
5. Address any feedback from the code review

### Issue Reporting

When reporting issues, please include:

1. A clear, descriptive title
2. Step-by-step instructions to reproduce the issue
3. Expected and actual results
4. Screenshots if applicable
5. Your environment details (OS, GPU, Python version)

### Commit Message Format

Follow the format:

```text
[type]: Short summary (50 chars or less)

More detailed explanation if necessary. Wrap at 72 characters.
```

Where `type` is one of:

- `feat`: **New** feature
- `fix`: Bug fix
- `docs`: Documentation only changes
- `style`: Formatting changes
- `refactor`: Code change that neither fixes a bug nor adds a feature
- `test`: Adding missing tests
- `chore`: Changes to build process or auxiliary tools

---

## Style Guide

This project follows specific coding standards to maintain consistency.

### Python Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guidelines with the following modifications
- Use 4 spaces for indentation (not tabs)
- Maximum line length: 100 characters
- Use double quotes for docstrings and single quotes for all other strings

### British English

For the sake of consistency, only one language standard should be used throughout the codebase and documentation. This project was developed for UWE-AI, UWE is in England, ergo British English.

- Always use British English spelling in comments, docstrings, and user-facing strings wherever possible
- Examples:
  - "optimisation" instead of "optimization"
  - "centre" instead of "center"
  - "visualise" instead of "visualize"

#### Function Over Form Exception

- When working with existing method names, class names, or libraries that use American or other non-english spellings, use the defined spelling in code:
  - If calling a method named `initialize_model()`, use that exact spelling in your code
  - If extending a class with `ColorManager`, maintain that spelling
  - Comments and documentation about such methods should still use British spelling
  - Example: `model.initialize_model()  # Initialises the model with optimised parameters`

Pull requests will be closed if arguments, standard output, methods, or the content of docstrings are spelt incorrectly according to these rules.

### Code Organization

- Organize imports in the following order:
  1. Standard library imports
  2. Related third party imports
  3. Local application imports
- Group imports with a blank line between each group
- Sort imports alphabetically within each group

### Documentation Approach

- **Docstrings are required** for all public modules, classes, functions, and methods
  - Follow Google style docstring format with parameter types
  - Ensure docstrings clearly explain purpose, parameters, and return values
  
- **Commit messages** should clearly explain what changes were made and why
  - Well-written commit messages reduce the need for extensive code comments
  
- **Code comments** are at your discretion
  - No strict requirements for commenting style or frequency
  - Use your best judgment to make complex sections understandable

### Testing

- Write unit tests for all new functionality
- Place tests in the `tests/` directory mirroring the module structure
- Name test files with `test_` prefix

---
