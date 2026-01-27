
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

GOLDEN_MAP = {
    "11796-0312" : (Paradise, 12),
    "11796-0315": (Paradise, 15),
    "14131-034": (Paradise, 34),
    "11989-0315": (Tesla, 15),
    "11989-0312": (Tesla, 12)
}
def create_golden_cable(test_name):

    if not isinstance(test_name, str):
        raise TypeError("test_name must be a string")
    if("11796-0312" in test_name):
        return Paradise("golden", 12)
    elif("11796-0315" in test_name):
        return Paradise("golden", 15)
    elif("14131-034" in test_name):
        return Paradise("golden", 34)
    elif("11989-0315" in test_name):
        return Tesla("golden", 15)
    elif("11989-0312" in test_name):
        return Tesla("golden", 12)
    
