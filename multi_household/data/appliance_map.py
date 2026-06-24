"""Per-house appliance dictionary for REFIT, with deferable / non-controllable
classification.

The 20 REFIT houses each have 9 IAM (Individual Appliance Monitor) channels
named Appliance1..Appliance9 in the CSVs. The mapping from channel index to
physical appliance comes from REFIT_Readme.txt (which is not machine-readable),
so we encode it here.

Classification rules:
  • deferable      → whole cycle can be shifted to off-peak (washer, dryer,
                     dishwasher, washer-dryer).
  • semi_deferable → can be throttled but not shifted far (electric heater).
  • non_controllable → must run on demand (fridge, freezer, TV, lighting,
                     kettle, microwave, computer, etc.).
"""
from __future__ import annotations

# House → {channel index (1-9) → appliance name}
HOUSE_APPLIANCES: dict[int, dict[int, str]] = {
    1:  {1: "Fridge", 2: "Chest Freezer", 3: "Upright Freezer",
         4: "Tumble Dryer", 5: "Washing Machine", 6: "Dishwasher",
         7: "Computer Site", 8: "Television Site", 9: "Electric Heater"},
    2:  {1: "Fridge-Freezer", 2: "Washing Machine", 3: "Dishwasher",
         4: "Television", 5: "Microwave", 6: "Toaster",
         7: "Hi-Fi", 8: "Kettle", 9: "Oven Extractor Fan"},
    3:  {1: "Toaster", 2: "Fridge-Freezer", 3: "Freezer",
         4: "Tumble Dryer", 5: "Dishwasher", 6: "Washing Machine",
         7: "Television", 8: "Microwave", 9: "Kettle"},
    4:  {1: "Fridge", 2: "Freezer", 3: "Fridge-Freezer",
         4: "Washing Machine 1", 5: "Washing Machine 2",
         6: "Computer Site", 7: "Television Site",
         8: "Microwave", 9: "Kettle"},
    5:  {1: "Fridge-Freezer", 2: "Tumble Dryer", 3: "Washing Machine",
         4: "Dishwasher", 5: "Computer Site", 6: "Television Site",
         7: "Combination Microwave", 8: "Kettle", 9: "Toaster"},
    6:  {1: "Freezer", 2: "Washing Machine", 3: "Dishwasher",
         4: "MJY Computer", 5: "Television Site",
         6: "Microwave", 7: "Kettle", 8: "Toaster", 9: "PGM Computer"},
    7:  {1: "Fridge", 2: "Freezer (Garage)", 3: "Freezer",
         4: "Tumble Dryer", 5: "Washing Machine", 6: "Dishwasher",
         7: "Television Site", 8: "Toaster", 9: "Kettle"},
    8:  {1: "Fridge", 2: "Freezer", 3: "Dryer", 4: "Washing Machine",
         5: "Toaster", 6: "Computer", 7: "Television Site",
         8: "Microwave", 9: "Kettle"},
    9:  {1: "Fridge-Freezer", 2: "Washer Dryer", 3: "Washing Machine",
         4: "Dishwasher", 5: "Television Site",
         6: "Microwave", 7: "Kettle", 8: "Hi-Fi", 9: "Electric Heater"},
    10: {1: "Magimix", 2: "Freezer", 3: "Chest Freezer", 4: "Fridge-Freezer",
         5: "Washing Machine", 6: "Dishwasher", 7: "Television Site",
         8: "Microwave", 9: "Kenwood KMix"},
    11: {1: "Fridge", 2: "Fridge-Freezer", 3: "Washing Machine",
         4: "Dishwasher", 5: "Computer Site", 6: "Microwave",
         7: "Kettle", 8: "Router", 9: "Hi-Fi"},   # solar — excluded
    12: {1: "Fridge-Freezer", 2: "Television Site (Lounge)",
         3: "Microwave", 4: "Kettle", 5: "Toaster",
         6: "Television Site (Bedroom)",
         7: "Not Used", 8: "Not Used", 9: "Not Used"},
    13: {1: "Television Site", 2: "Unknown",
         3: "Washing Machine", 4: "Dishwasher", 5: "Tumble Dryer",
         6: "Television Site 2", 7: "Computer Site",
         8: "Microwave", 9: "Kettle"},
    15: {1: "Fridge-Freezer", 2: "Tumble Dryer", 3: "Washing Machine",
         4: "Dishwasher", 5: "Computer Site", 6: "Television Site",
         7: "Microwave", 8: "Kettle", 9: "Toaster"},
    16: {1: "Fridge-Freezer 1", 2: "Fridge-Freezer 2",
         3: "Electric Heater 1", 4: "Electric Heater 2",
         5: "Washing Machine", 6: "Dishwasher",
         7: "Computer Site", 8: "Television Site",
         9: "Dehumidifier/Heater"},
    17: {1: "Freezer (Garage)", 2: "Fridge-Freezer",
         3: "Tumble Dryer", 4: "Washing Machine",
         5: "Computer Site", 6: "Television Site",
         7: "Microwave", 8: "Kettle",
         9: "Plug Site (Bedroom)"},
    18: {1: "Fridge (Garage)", 2: "Freezer (Garage)",
         3: "Fridge-Freezer", 4: "Washer Dryer",
         5: "Washing Machine", 6: "Dishwasher",
         7: "Desktop Computer", 8: "Television Site", 9: "Microwave"},
    19: {1: "Fridge & Freezer", 2: "Washing Machine",
         3: "Television Site", 4: "Microwave", 5: "Kettle",
         6: "Toaster", 7: "Bread-maker", 8: "Lamp", 9: "Hi-Fi"},
    20: {1: "Fridge", 2: "Freezer", 3: "Tumble Dryer",
         4: "Washing Machine", 5: "Dishwasher",
         6: "Computer Site", 7: "Television Site",
         8: "Microwave", 9: "Kettle"},
    21: {1: "Fridge-Freezer", 2: "Tumble Dryer", 3: "Washing Machine",
         4: "Dishwasher", 5: "Food Mixer", 6: "Television",
         7: "Kettle/Toaster", 8: "Vivarium", 9: "Pond Pump"},  # solar — excluded
}

DEFERABLE_KEYWORDS       = ("tumble dryer", "washing machine",
                            "washer dryer", "dryer", "dishwasher",
                            "electric heater", "water heater",
                            "synthetic_ev")
SEMI_DEFERABLE_KEYWORDS  = ("dehumidifier",)


def classify(appliance_name: str) -> str:
    """Return 'deferable', 'semi_deferable', or 'non_controllable'.

    Electric heater / water heater are classified as deferable because users
    can pre-heat during off-peak and rely on thermal mass through peak hours
    without losing comfort. This matches UK Economy 7 ToU practice.
    """
    n = appliance_name.lower()
    if any(k in n for k in DEFERABLE_KEYWORDS):
        return "deferable"
    if any(k in n for k in SEMI_DEFERABLE_KEYWORDS):
        return "semi_deferable"
    return "non_controllable"


def appliance_channels(house_id: int, cls: str) -> list[int]:
    """Channel indices (1..9) for a given class in a given house.

    Example:
        >>> appliance_channels(1, "deferable")
        [4, 5, 6]   # Tumble Dryer, Washing Machine, Dishwasher
    """
    if house_id not in HOUSE_APPLIANCES:
        raise KeyError(f"unknown house {house_id}")
    return [ch for ch, name in HOUSE_APPLIANCES[house_id].items()
            if classify(name) == cls]


def appliance_summary(house_id: int) -> dict:
    """Counts + names per class for one house — useful for sanity-printing."""
    out = {"deferable": [], "semi_deferable": [], "non_controllable": []}
    for ch, name in HOUSE_APPLIANCES[house_id].items():
        out[classify(name)].append((ch, name))
    return out
