from pathlib import Path

# ModelMark pkg dir path
PACKAGE_DIR = Path(__file__).parent
# User cwd path
USER_DIR = Path.cwd()

# Create user config dir (if not already exists)
USER_CONFIG_DIR = USER_DIR / "modelmark_files"

# Pkg config file path
PACKAGE_CONFIG_PATH = PACKAGE_DIR / "config.py"
# User config file path
USER_CONFIG_PATH = USER_CONFIG_DIR / "config.py"

# Create user models dir (if not already created)
USER_MODELS_DIR = USER_DIR / "models"

# Pkg example model path
PACKAGE_MODEL_PATH = PACKAGE_DIR / "models" / "linear.py"
# User example mopdel path
USER_MODEL_PATH = USER_DIR / "models" / "linear.py"

PARSER_DESC = """MODELMARK

	Python tool to measure the performance of a custom neural network model
	and compare it to other popular architectures."""