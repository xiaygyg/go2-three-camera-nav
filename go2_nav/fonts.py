"""Portable font lookup for diagnostics."""
from PIL import ImageFont

def font(size):
    for name in ['DejaVuSans.ttf', 'Arial.ttf', 'LiberationSans-Regular.ttf']:
        try: return ImageFont.truetype(name, size)
        except OSError: pass
    return ImageFont.load_default(size=size)
