# trials-modern

**trials-modern** is a modified fork of the original [trials](https://github.com/The-FinAI/trials) project.
It implements the *Select and Trade* paper but updated to use **Gymnasium** and other modern libraries.

---

## Installation

This project uses [Poetry](https://python-poetry.org/) for dependency management.

```bash
# Install dependencies
poetry install
```

---

## Running the Project

```bash
# Run the main script
poetry run python ./Scripts/main.py
```

---

## Using CUDA with PyTorch (GPU Support)

If you need GPU acceleration and **Poetry cannot successfully install the correct CUDA-enabled PyTorch version**, you can manually install it like this:

```bash
poetry run pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

> **Important:**
>
> * Try installing via `poetry install` first — in most cases, that should handle PyTorch and CUDA automatically.
> * Use the manual `pip install` method **only if** Poetry fails to install the correct CUDA build.
> * This bypasses Poetry’s dependency management and installs the packages directly into the virtual environment.




