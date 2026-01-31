from fastapi.templating import Jinja2Templates
import os

# Initialize templates
template_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "tools", "forge-ui", "templates"))
templates = Jinja2Templates(directory=template_path)
