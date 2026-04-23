import os
import sys

# Make the project root importable so tests can `import tax_engine, forecast`.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
