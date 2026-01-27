
from Paradise import Paradise
from Tesla import Tesla

CABLE_MAP = {
    "0": (Paradise, 11),
    "1": (Paradise, 15),
    "3": (Tesla, 11),
    "4": (Tesla, 15),
}

def create_cable(serial_number: str):
    key = serial_number.strip().upper()[1]
    cls, length = CABLE_MAP[key]
    return cls(serial_number, length)
