"""Official land-use codes of each city -> PlanArena's 16 classes (planarena/taxonomy.py).

Each function takes the source's own fields and returns a class key, or "unknown" when the source does not say.
The tables follow each source's published code list (cited in the docstrings); tests check that every code present in
the data maps to a class or is listed as unknown on purpose.
"""
from __future__ import annotations

import math
import re


def _num(x) -> int | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else int(v)


# ----------------------------------------------------------------------------- New York City (MapPLUTO)
def nyc(land_use, bldg_class, units_res) -> str:
    """MapPLUTO LandUse (01-11, NYC DCP) refined by the DOF building class (first letter) and residential units.

    Building classes: A one-family, B two-family, C walk-up, D elevator, E warehouse, F factory, G garage/gas
    station, H hotel, I health, K store, M religious, N asylum/home, O office, P assembly/cultural, W education,
    Y government (NYC Department of Finance building classification glossary).
    """
    lu = _num(land_use)
    bc = bldg_class.strip().upper()[:1] if isinstance(bldg_class, str) else ""
    units = _num(units_res) or 0
    if lu is None:
        return "unknown"
    if lu == 1:
        return "sf" if bc == "A" or units <= 1 else "mf_small"
    if lu == 2:
        return "mf_small" if units <= 4 else "mf"
    if lu == 3:
        return "mf"
    if lu == 4:
        return "mix"
    if lu == 5:
        return {"O": "off", "H": "hot", "G": "ret"}.get(bc, "ret")
    if lu == 6:
        return "whs" if bc == "E" else "ind"
    if lu == 7:
        return "tra"
    if lu == 8:
        return {"W": "edu", "I": "hlt"}.get(bc, "civ")
    return {9: "opn", 10: "prk", 11: "vac"}.get(lu, "unknown")


# ----------------------------------------------------------------------------- Milwaukee (MPROP)
def milwaukee(land_use, group, units_res) -> str:
    """MPROP land-use group (MPROP documentation, Appendix H) refined by the land-use code, which is a Standard
    Industrial Classification code for non-residential property (e.g. 5812 eating places, 4225 warehousing, 8211
    schools, 8062 hospitals, 7011 hotels, 8661 religious organisations) or a city code for homes and vacant land."""
    code, gp, units = _num(land_use) or 0, _num(group) or 0, _num(units_res) or 0
    sic2 = code // 100
    if code in (7521, 7523, 7525):
        return "prk"
    if gp == 1:
        return "sf"
    if gp == 2:
        return "mf_small"
    if gp == 3:
        return "mf" if units >= 5 else "mf_small"
    if gp == 4:
        return "mix"
    if gp == 5:
        return "whs" if sic2 in (50, 51) else "ret"
    if gp == 6:
        if 60 <= sic2 <= 67 or sic2 in (73, 81, 87, 74):
            return "off"
        if code == 7011:
            return "hot"
        if code == 7021:
            return "mf"  # rooming and boarding houses
        if sic2 == 80 or code == 8361:
            return "hlt"
        if sic2 == 82 or code == 8351:
            return "edu"  # schools, colleges; child day care
        if sic2 in (83, 84, 86):
            return "civ"
        return "ret"  # personal, repair, amusement and other consumer services
    if gp == 7:
        return "ret"  # commercial mixed use and commercial condominiums
    if gp == 8:
        return "whs" if code in range(4221, 4227) or sic2 in (50, 51) else "ind"
    if gp == 9:
        return "tra"
    if gp in (10, 12):
        return "opn"
    if gp == 11:
        return "edu" if sic2 == 82 else "hlt" if sic2 == 80 else "civ"
    if gp == 13 or code == 9850:
        return "vac"
    return "unknown"


# ----------------------------------------------------------------------------- Philadelphia (City Planning Commission)
def philadelphia(dig2, dig3, opa_category=None, opa_description=None) -> str:
    """Philadelphia land use (City Planning Commission, two- and three-digit codes). Where only the two-digit code is
    given, the parcel's OPA assessment record splits it: the category for homes (1 single family, 2 multi-family,
    14 apartments of 5+ units) and the building description for institutions, industry and hotels."""
    d2, d3, cat = _num(dig2), _num(dig3), _num(opa_category)
    desc = opa_description.upper() if isinstance(opa_description, str) else ""
    if d3 is not None:
        exact = {111: "sf", 112: "sf", 121: "sf", 119: "sf", 122: "mf_small", 123: "mf_small", 124: "mf_small",
                 125: "mf_small", 129: "mf_small", 131: "mf", 132: "mf", 133: "mf", 135: "hot", 136: "hlt", 137: "mf",
                 138: "civ", 139: "mf", 211: "ret", 212: "ret", 213: "ret", 219: "ret", 221: "off", 222: "off",
                 229: "off", 231: "mix", 232: "mix", 233: "mix", 239: "mix", 314: "tra", 316: "whs", 317: "whs",
                 411: "hlt", 412: "edu", 413: "edu", 514: "prk", 529: "opn", 613: "ret", 621: "opn", 921: "unknown"}
        if d3 in exact:
            return exact[d3]
        d2 = d2 or d3 // 10
    if d2 is None:
        return "unknown"
    if d2 in (11, 12, 13):
        if d2 == 13 or cat == 14:
            return "mf"
        return "mf_small" if cat == 2 else "sf"
    if d2 == 41:
        if any(w in desc for w in ("SCHOOL", "COLLEGE", "UNIVERSITY", "DAY CARE")):
            return "edu"
        if any(w in desc for w in ("HOSPITAL", "MEDICAL", "NURSING", "CLINIC")):
            return "hlt"
    if d2 in (21, 22) and ("HOTEL" in desc or "MOTEL" in desc):
        return "hot"
    if d2 == 31 and "WAREHOUSE" in desc:
        return "whs"
    return {21: "ret", 22: "off", 23: "mix", 31: "ind", 41: "civ", 51: "tra", 52: "opn", 61: "civ", 62: "opn",
            71: "opn", 72: "opn", 81: "opn", 91: "vac"}.get(d2, "unknown")


# ----------------------------------------------------------------------------- Southern California (SCAG)
def _codes(cls: str, *codes) -> dict:
    return {str(c): cls for c in codes}


# SCAG land-use codes (SCAG 2020 Annual Land Use dataset, code list of fields ALU20, GP19_SCAG and ZN19_SCAG)
SCAG = (_codes("sf", 1110, 1111, 1112, 1113, 1150, 1132)  # 1132 mobile home subdivisions: one home per lot
        | _codes("mf_small", 1122)  # duplexes, triplexes, 2- or 3-unit condominiums and townhouses
        | _codes("mf", 1123, 1124, 1125, 1130, 1131)  # apartments; mobile home and trailer parks (many homes per lot)
        | _codes("ret", 1200, 1220, 1221, 1222, 1223, 1230, 1232)  # 1232 commercial recreation
        | _codes("off", 1210, 1211, 1212, 1213)
        | _codes("hot", 1233)
        | _codes("whs", 1231, 1323, 1340, 1416)  # commercial and open storage, wholesaling, truck terminals
        | _codes("civ", 1240, 1241, 1242, 1243, 1245, 1246, 1250, 1251, 1253, 1270, 1271, 1274)
        | _codes("hlt", 1244, 1252)  # major medical; special care facilities
        | _codes("edu", 1260, 1261, 1262, 1263, 1264, 1265, 1266)
        | _codes("prk", 1247, 1414)  # public parking; park-and-ride lots
        | _codes("ind", 1300, 1310, 1311, 1312, 1313, 1314, 1320, 1321, 1322, 1324, 1325, 1330, 1331, 1332, 1500)
        | _codes("tra", 1273, 1276, 1400, 1410, 1411, 1412, 1413, 1415, 1417, 1418, 1420, 1430, 1431, 1432, 1433,
                 1434, 1435, 1436, 1437, 1438, 1440, 1441, 1442, 1450, 1460)
        | _codes("mix", 1600, 1610, 1620)
        | _codes("opn", 1800, 1810, 1820, 1830, 1840, 1850, 1860, 1870, 1880, 1890, 2000, 2100, 2110, 2120, 2200,
                 2300, 2400, 2500, 2600, 2700, 3400, 4000, 4100, 4200, 4300, 4400, 4500, 8888)
        | _codes("vac", 1272, 1275, 1700, 1900, 3000, 3100, 3200, 3300))
SCAG_UNKNOWN = ("7777", "9999")  # specific plan (its own code is used where given); unknown
SCAG_GENERIC_MF = ("1120", "1121", "1140")  # multi-family / mixed residential without a size


def scag(code, units=None, specific_plan=None) -> str:
    """SCAG land-use code -> class. The same code list describes existing land use (ALU20), the general plan
    (GP19_SCAG) and zoning (ZN19_SCAG). Code 7777 'Specific Plan' takes the specific plan's own code (SP19_SCAG) where
    given. Generic residential codes are split by the assessor's unit count where there is one (1100 residential,
    1120 multi-family, 1121 mixed multi-family, 1140 mixed residential); 1113 rural and 1150 rural residential are
    single-family, unless the assessor records five or more homes on the parcel. Under construction (1700) counts as
    vacant: nothing is in use yet."""
    c = str(code).strip() if code is not None else ""
    if c == "7777" and specific_plan is not None:
        return scag(specific_plan, units)
    n = _num(units)
    if c in ("1110", "1111", "1112", "1113") and n and n >= 5:
        return "mf"  # a condominium or townhouse complex the assessor records as one parcel of five or more homes
    if c == "1100":
        return "sf" if not n or n <= 1 else "mf_small" if n <= 4 else "mf"
    if c in SCAG_GENERIC_MF:
        return "mf_small" if n and n <= 4 else "mf"
    return SCAG.get(c, "unknown")


# ----------------------------------------------------------------------------- City of Los Angeles (plan and zones)
# SCAG's translation of the General Plan and zoning uses generic codes for housing (1110 single-family, 1120 or 1123
# multi-family), so a duplex designation reads as apartments. Residential designations are classed by the city's own
# density categories instead; their corresponding zones are those listed in the Community Plan land-use legends.
LA_PLAN_RESIDENTIAL = {
    **{d: "sf" for d in ("Minimum Residential", "Very Low Residential", "Very Low I Residential",
                         "Very Low II Residential", "Low Residential", "Low I Residential", "Low II Residential",
                         "Low III Residential")},  # A1, A2, RE, RS, R1, RU, RZ, RW1
    **{d: "mf_small" for d in ("Low Medium Residential", "Low Medium I Residential",
                               "Low Medium II Residential")},  # R2, RD, RW2
    **{d: "mf" for d in ("Medium Residential", "High Medium Residential", "High Density Residential",
                         "Very High Residential")},  # R3, R4, R5
}
LA_ZONE_RESIDENTIAL = [("RAS", "mf"), ("RMP", "mf"), ("RW2", "mf_small"), ("RW1", "sf"), ("RD", "mf_small"),
                       ("R2", "mf_small"), ("R3", "mf"), ("R4", "mf"), ("R5", "mf"), ("RZ", "sf"), ("RE", "sf"),
                       ("RS", "sf"), ("RU", "sf"), ("RA", "sf"), ("R1", "sf"), ("A1", "sf"), ("A2", "sf")]


def los_angeles_plan(designation, scag_code, specific_plan=None) -> str:
    """General Plan designation (GP19_CITY) -> class: residential by the city's density category, else SCAG's code."""
    d = designation.strip() if isinstance(designation, str) else ""
    return LA_PLAN_RESIDENTIAL.get(d) or scag(scag_code, specific_plan=specific_plan)


def los_angeles_zone(zone, scag_code, specific_plan=None) -> str:
    """Zone (ZN19_CITY, e.g. RD1.5, R3, C2) -> class: residential zones by family, else SCAG's code."""
    import re

    z = zone.strip().upper() if isinstance(zone, str) else ""
    for prefix, cls in LA_ZONE_RESIDENTIAL:
        if re.match(prefix + r"(?![A-Z])", z):
            return cls
    return scag(scag_code, specific_plan=specific_plan)


# ----------------------------------------------------------------------------- Chicago region (CMAP)
# CMAP Land Use Inventory codes (2020 and 2023 inventories share the code list; layer domain_LANDUSE)
CMAP = (_codes("sf", 1111, 1112)  # detached; attached (townhouses, rowhouses)
        | _codes("mf", 1130, 1140)  # multi-family; mobile home parks
        | _codes("ret", 1211, 1212, 1214, 1215, 1240)  # malls, regional, big box, urban mix, cultural/entertainment
        | _codes("mix", 1216)  # urban mix with residential
        | _codes("off", 1220)
        | _codes("hot", 1250)
        | _codes("hlt", 1310)
        | _codes("edu", 1321, 1322)
        | _codes("civ", 1330, 1340, 1350, 1370, 1380)  # government, prisons, religious, other, national labs
        | _codes("ind", 1410, 1420, 1431, 1433)  # mineral, general, manufacturing and flex over 100k sq ft
        | _codes("whs", 1432, 1450)  # warehousing over 100k sq ft; storage
        | _codes("tra", 1511, 1512, 1520, 1530, 1550, 1561, 1562, 1563, 1564, 1565, 1570)
        | _codes("prk", 1540)
        | _codes("opn", 1151, 1360, 2000, 3100, 3200, 3300, 3400, 3500, 5000)  # residential common areas, cemeteries,
                                                                              # agriculture, open space, water
        | _codes("vac", 4110, 4120, 4130, 4140, 4210, 4220, 4230, 4240))  # vacant; under construction
CMAP_UNKNOWN = ("6000", "9999")  # non-parcel; uncodeable
CMAP_RESIDENTIAL = ("1111", "1112", "1130")
CMAP_BUSINESS = ("1211", "1212", "1214", "1215", "1220", "1240", "1250", "1410", "1420", "1431", "1432", "1433", "1450")
# Cook County Assessor commercial valuation property types (first match wins)
COOK_PROPERTY_TYPE = [("MIXED USE", "mix"), ("HOTEL", "hot"), ("MOTEL", "hot"), ("MEDICAL", "hlt"),
                      ("NURSING", "hlt"), ("OFFICE", "off"), ("WAREHOUSE", "whs"), ("SELF STORAGE", "whs"),
                      ("DISTRIBUTION", "whs"), ("PARKING", "prk"), ("DAY CARE", "edu")]


def cmap(code, units=None, property_type=None) -> str:
    """CMAP land-use code -> class, refined by the Cook County Assessor where the inventory is coarse.

    Residential codes are split by the homes the assessor records on the parcel (2-4 small multi-family, 5+
    multi-family; a mixed-use apartment record makes it mixed use). Commercial and industrial codes are split by the
    commercial valuation's property type (office, medical, hotel, warehouse, parking, mixed use)."""
    c = str(code).strip() if code is not None else ""
    n = _num(units)
    pt = property_type.upper() if isinstance(property_type, str) else ""
    if c in CMAP_RESIDENTIAL:
        if "MIXED USE" in pt:
            return "mix"
        if n:
            return "sf" if n <= 1 else "mf_small" if n <= 4 else "mf"
    if c in CMAP_BUSINESS and pt:
        for word, cls in COOK_PROPERTY_TYPE:
            if word in pt:
                return cls
    return CMAP.get(c, "unknown")


# ----------------------------------------------------------------------------- Houston (City of Houston land use)
# Harris Central Appraisal District land-use codes, as published in the City of Houston "Land Use (Grouped)" layer
HOUSTON = (_codes("sf", 1001, 1003, 2001, 2003, 3100, 4108, 4301, 4444)  # houses; mobile homes; houses on other land
           | _codes("mf", 1006, 4209, 4211, 4212, 4213, 4214, 4221, 4222, 4318)  # condominiums, apartments (4+ units),
                                                                              # mobile home parks, boarding houses
           | _codes("mix", 4319)  # commercial building with residences
           | _codes("ret", 4321, 4323, 4324, 4325, 4326, 4327, 4328, 4329, 4330, 4331, 4332, 4333, 4334, 4336, 4337,
                    4340, 4341, 4342, 4343, 4344, 4345, 4346, 4347, 4348, 4350, 4351, 4356, 4361, 4362, 4363, 4364,
                    4365, 4370, 4373, 4374, 4375, 4376, 4381, 4382, 4383, 4384, 4385, 4386, 4388, 4390)
           | _codes("off", 4353, 4354, 4355, 4359, 4366)
           | _codes("hot", 4314, 4315, 4320)
           | _codes("hlt", 4316, 4317, 4349, 4352, 4357, 4640)  # nursing and retirement homes, medical, hospitals
           | _codes("edu", 4313, 4369, 4612, 4613)  # dormitories, day care, schools, colleges
           | _codes("civ", 4367, 4610, 4611, 4614, 4620, 4630, 4660, 4670, 4680, 4685)
           | _codes("prk", 4338, 4339, 4372)
           | _codes("whs", 4335, 4358, 4377, 4380, 4391, 4392, 4394, 4395, 4396, 4397, 4398, 4399, 4427, 4428, 4491)
           | _codes("ind", 3300, 4401, 4402, 4403, 4405, 4410, 4411, 4413, 4415, 4416, 4417, 4418, 4419, 4420, 4421,
                    4422, 4423, 4424, 4425, 4426, 4460, 4470, 4510, 4544, 4961)
           | _codes("tra", 1105, 2005, 4368, 4371, 4530, 4542, 4545, 4690, 4700, 4706, 4710, 4712, 4715, 4720, 4730,
                    4733, 4735, 4737, 4740, 4742, 4745, 4747, 4749, 4750, 4752, 4755, 4756, 4757, 4760, 4762, 4765,
                    4770, 4772, 4773, 4777)
           | _codes("opn", 1107, 2004, 4389, 4500, 4520, 4543, 9011, 9015, 9019, 9023, 9039, 9043, 9047, 9063, 9075,
                    9083, 9200, 9910, 9920, 9930)  # retention, wetlands, golf, subdivision common lots, agriculture
           | _codes("vac", 1000, 1002, 2000, 2002, 4200, 4299, 4300, 4310, 4400, 4499, 4540, 7000, 9999))
HOUSTON_GROUP = {"Single-Family Residential": "sf", "Multi-Family Residential": "mf", "Commercial": "ret",
                 "Office": "off", "Industrial": "ind", "Public & Institutional": "civ", "Park & Open Spaces": "opn",
                 "Transportation & Utility": "tra", "Undeveloped": "vac", "Agriculture Production": "opn"}
HOUSTON_RESIDENTIAL = {"B2": "mf_small", "B3": "mf_small", "B4": "mf_small", "B1": "mf"}  # duplex/tri/fourplex; apartments


def houston(code, group=None, state_class=None) -> str:
    """HCAD land-use code -> class. Houses are split by the Texas property class (B1 apartments, B2-B4 duplexes to
    fourplexes). Vacant exempt land (4600) and codes without a description take the City of Houston staff group
    (park, public, transportation or vacant)."""
    c = str(code).strip() if code is not None else ""
    sc = str(state_class).strip().upper() if isinstance(state_class, str) else ""
    cls = HOUSTON.get(c)
    if cls == "sf" and sc in HOUSTON_RESIDENTIAL:
        return HOUSTON_RESIDENTIAL[sc]
    if cls:
        return cls
    if c == "4600" and group not in ("Park & Open Spaces", "Public & Institutional", "Transportation & Utility"):
        return "vac"
    return HOUSTON_GROUP.get(group, "unknown") if isinstance(group, str) else "unknown"


# ----------------------------------------------------------------------------- Dallas (City of Dallas existing land use)
DALLAS = {"Single Family Detached": "sf", "Single Family Attached": "sf", "Multi-Family": "mf", "Mixed Use": "mix",
          "Commercial": "ret", "Office": "off", "Lodging": "hot", "Light Industrial": "ind",
          "Heavy Industrial": "ind", "Public or Institutional": "civ", "Open Space": "opn", "Agriculture": "opn",
          "Utility": "tra", "Transportation": "tra"}
# Dallas Central Appraisal District building classes that split the business and public categories (first match wins)
DALLAS_BUILDING = [("VETERINARY", "ret"), ("SENIOR CITIZEN HOUSING", "mf"), ("RETIREMENT COMMUNITY", "mf"),
                   ("ASSISTED LIVING", "hlt"), ("NURSING", "hlt"), ("CONVALESCENT", "hlt"), ("HOSPITAL", "hlt"),
                   ("MEDICAL", "hlt"), ("URGENT CARE", "hlt"), ("HOTEL", "hot"), ("MOTEL", "hot"),
                   ("WAREHOUSE", "whs"), ("COLD STORAGE", "whs"), ("TRUCK TERMINAL", "whs"),
                   ("PARKING", "prk"), ("SCHOOL", "edu"), ("DAY NURSERY", "edu"), ("CHURCH", "civ"),
                   ("GOVERNMENTAL", "civ"), ("FRATERNAL", "civ"), ("PUBLIC CHARITY", "civ"), ("AUDITORIUM", "civ"),
                   ("OFFICE/SHOWROOM", "ret"), ("OFFICE", "off")]
DALLAS_REFINED = ("Commercial", "Office", "Lodging", "Light Industrial", "Heavy Industrial", "Public or Institutional",
                  "Transportation")


def dallas(land_use, vacant=None, building_class=None, units=None) -> str:
    """City of Dallas existing land use -> class. The appraisal district's building class splits the business and
    public categories; residential parcels are split by the appraisal district's unit count (2-4 small multi-family,
    5+ multi-family); a vacant parcel in a built category is vacant land (open space, agriculture, utility and
    transportation land keeps its class)."""
    lu = land_use.strip() if isinstance(land_use, str) else ""
    cls = DALLAS.get(lu)
    if cls is None:
        return "unknown"
    if isinstance(vacant, str) and vacant.strip().lower() == "vacant" and cls not in ("opn", "tra"):
        return "vac"
    bc = building_class.upper() if isinstance(building_class, str) else ""
    if lu in DALLAS_REFINED and bc:
        for word, c in DALLAS_BUILDING:
            if word in bc:
                return c
    n = _num(units)
    if cls in ("sf", "mf") and n:
        return "sf" if n <= 1 else "mf_small" if n <= 4 else "mf"
    return cls

# forwardDallas! 2.0 (adopted 2024) placetypes: the plan's land-use designation of each parcel
DALLAS_PLACETYPE = {"Community Residential": "sf", "Small Town Residential": "sf", "City Residential": "mf",
                    "Neighborhood Mixed-Use": "mix", "Community Mixed-Use": "mix", "Regional Mixed-Use": "mix",
                    "City Center": "mix", "Flex Commercial": "ret", "Logistics/Industrial Park": "whs",
                    "Industrial Hub": "ind", "Institutional Campus": "civ", "Regional Open Space": "opn",
                    "Airport": "tra", "Utility": "tra"}


def dallas_plan(placetype) -> str:
    return DALLAS_PLACETYPE.get(placetype.strip(), "unknown") if isinstance(placetype, str) else "unknown"


# ----------------------------------------------------------------------------- San Antonio (BCAD; City of San Antonio)
# Texas property classes as used by the Bexar Appraisal District (State_cd)
SAN_ANTONIO_STATE = {"A1": "sf", "A2": "sf", "A3": "sf", "E1": "sf", "E2": "sf", "B1": "mf", "B2": "mf_small",
                     "B6": "mf_small", "C1": "vac", "C": "vac", "O1": "vac", "O2": "vac", "D1": "opn", "D2": "opn",
                     "F1": "ret", "F2": "ind", "F3": "ret", "G3": "ind", "J1": "tra", "J2": "tra", "J3": "tra",
                     "J4": "tra", "J5": "tra", "J6": "tra", "J7": "tra", "X": "civ"}
# City of San Antonio comprehensive plan land-use categories (Future_Land_Use, all adopted area plans)
SAN_ANTONIO_PLAN = [("ESTATE", "sf"), ("RURAL", "sf"), ("URBAN LOW DENSITY RESIDENTIAL", "mf_small"),
                    ("LOW DENSITY RESIDENTIAL", "sf"), ("MEDIUM DENSITY RESIDENTIAL", "mf"),
                    ("HIGH DENSITY RESIDENTIAL", "mf"), ("COMPACT MULTI FAMILY", "mf"), ("RESIDENTIAL", "sf"),
                    ("EMPLOYMENT/FLEX", "ind"), ("BUSINESS/INNOVATION", "off"), ("MIXED", "mix"), ("URBAN LIVING", "mix"),
                    ("TRANSIT ORIENTED", "mix"), ("COMMERCIAL", "ret"), ("OFFICE", "off"), ("BUSINESS PARK", "off"),
                    ("INDUSTRIAL", "ind"), ("PARK", "opn"), ("OPEN SPACE", "opn"), ("GREENSPACE", "opn"),
                    ("AGRICULTURAL", "opn"), ("GOVERNMENT", "civ"), ("PUBLIC INSTITUTIONAL", "civ"),
                    ("MILITARY", "civ"), ("VACANT", "vac")]
# San Antonio Unified Development Code base zoning districts -> the district's use class
SAN_ANTONIO_ZONE = [("RM-", "mf_small"), ("MF-", "mf"), ("MH", "mf"), ("R-", "sf"), ("RE", "sf"), ("RP", "sf"),
                    ("RD", "sf"), ("NC", "ret"), ("C-", "ret"), ("O-", "off"), ("MI-", "ind"), ("I-", "ind"),
                    ("L", "ind"), ("D", "mix"), ("UD", "mix"), ("FR", "opn"), ("UZROW", "tra")]


def _first_match(text, table) -> str:
    t = text.upper().strip() if isinstance(text, str) else ""
    for word, cls in table:
        if word in t:
            return cls
    return "unknown"


def san_antonio(state_class, units=None) -> str:
    """BCAD property class -> class; apartments (B1) of 2-4 units are small multi-family."""
    sc = state_class.strip().upper() if isinstance(state_class, str) else ""
    cls = SAN_ANTONIO_STATE.get(sc, "unknown")
    n = _num(units)
    if cls == "mf" and n and n <= 4:
        return "mf_small"
    return cls


def san_antonio_plan(category) -> str:
    return _first_match(category, SAN_ANTONIO_PLAN)


def san_antonio_zone(base) -> str:
    z = base.upper().strip() if isinstance(base, str) else ""
    for prefix, cls in SAN_ANTONIO_ZONE:
        if z == prefix or (z.startswith(prefix) and (prefix.endswith("-") or not z[len(prefix)].isalpha())):
            return cls
    return "unknown"


# ----------------------------------------------------------------------------- Phoenix (Maricopa County Assessor; MAG)
# Maricopa County Assessor property use code groups (first two digits of the property use code)
MARICOPA_GROUP = {"00": "vac", "01": "sf", "02": "opn", "03": "mf", "04": "hot", "05": "hot", "06": "hot",
                  "07": "mf", "08": "mf", "09": "ind", "10": "ret", "11": "ret", "12": "ret", "13": "ret",
                  "14": "ret", "15": "off", "16": "ret", "17": "ret", "18": "ret", "19": "hlt", "20": "ret",
                  "21": "hlt", "22": "ret", "23": "opn", "24": "opn", "25": "ret", "26": "prk", "27": "civ",
                  "28": "vac", "29": "edu", "30": "ind", "37": "whs", "40": "opn", "41": "opn", "43": "opn",
                  "45": "opn", "47": "opn", "48": "opn", "50": "tra", "51": "tra", "53": "tra", "54": "tra",
                  "55": "tra", "56": "tra", "57": "tra", "62": "tra", "85": "sf", "86": "mf_small"}
# groups 87-99 (miscellaneous, limited use, exempt and public ownership) describe ownership or tax status, not use:
# those parcels take MAG's existing land use instead
MAG_LAND_USE = {"Agriculture": "opn", "Airport": "tra", "Business Park": "off", "Cemetery": "opn",
                "Developing Employment": "vac", "Developing Residential": "vac", "Educational": "edu",
                "Golf Course": "opn", "Tourist Accomodations": "hot", "Industrial": "ind",
                "Medical/Nursing Home": "hlt", "Multi Family": "mf", "Mixed Use": "mix", "Office": "off",
                "Active Open Space": "opn", "Passive/Restricted Open Space": "opn", "Other Employment": "ind",
                "Public/Special Event/Military": "civ", "Religious/Institutional": "civ", "Commercial High": "ret",
                "Commercial Low": "ret", "Single Family High Density": "sf", "Single Family Low Density": "sf",
                "Single Family Medium Density": "sf", "Transportation": "tra", "Vacant": "vac", "Water": "opn",
                "Desert Parks and Preserves": "opn", "Wash": "opn", "Orchard": "opn", "Dairy or Feedlot": "opn",
                "Abandoned Agriculture": "vac", "Public Land": "civ", "Vacant State Trust": "vac",
                "Railroads": "tra", "Solar Generating Stations": "tra"}


def phoenix(use_group, description=None, mag=None) -> str:
    """Maricopa County Assessor property use group -> class (private roads in group 02 are transportation);
    ownership and exempt groups take MAG's existing land use (2026)."""
    g = str(use_group).strip().zfill(2) if use_group not in (None, "") else ""
    d = description.upper() if isinstance(description, str) else ""
    if g == "02" and "RDWY" in d:
        return "tra"
    cls = MARICOPA_GROUP.get(g)
    if cls:
        return cls
    return MAG_LAND_USE.get(mag.strip(), "unknown") if isinstance(mag, str) else "unknown"


def phoenix_plan(description) -> str:
    """City of Phoenix General Plan land-use designation (code table 'General Plan Codes') -> class. Residential
    designations by their highest density: up to 5 homes an acre single-family, up to 10 small multi-family, more
    multi-family. Preserves (hillside and desert preserves) are open space whatever the underlying designation."""
    import re

    d = description.upper().strip() if isinstance(description, str) else ""
    if not d or "UNDESIGNATED" in d and "PRESERVE" not in d:
        return "unknown"
    if "PRESERVE" in d:
        return "opn"
    if "PLANNED SCHOOL" in d:
        return "edu"
    homes = "RESIDENTIAL" in d or re.search(r"\bDU\b", d) is not None or d in ("MU", "MIXED USE")
    if (d.startswith(("MIXED USE", "MU")) or "TRANSITION" in d) and homes:
        return "mix"  # homes with commercial, business park or open space
    if "RESIDENTIAL" in d:
        if "15+" in d:
            return "mf"
        top = max((float(x) for x in re.findall(r"\d+(?:\.\d+)?", d)), default=0)
        return "sf" if top <= 5 else "mf_small" if top <= 10 else "mf"
    # mixed use without homes takes its leading non-residential use
    for word, cls in (("INDUSTRIAL", "ind"), ("COMMERCE", "off"), ("BUSINESS PARK", "off"), ("COMMERCIAL", "ret"),
                      ("PARK", "opn"), ("OPEN SPACE", "opn"), ("FLOODPLAIN", "opn"), ("AGRICULTUR", "opn"),
                      ("NURSERIES", "opn"), ("PUBLIC", "civ"), ("TRANSPORT", "tra"), ("FREEWAY", "tra"),
                      ("AIRPORT", "tra")):
        if word in d:
            return cls
    return "mix" if d.startswith(("MIXED USE", "MU")) else "unknown"


PHOENIX_ZONE = {"SF Residential": "sf", "MF Residential": "mf", "Commercial": "ret", "Industrial": "ind",
                "Downtown Code": "mix", "Non-Developable": "opn"}  # PUD and County: not screened


# ----------------------------------------------------------------------------- San Diego (SANDAG; City of San Diego)
# SANDAG Land Use 2025 (annual land use and housing inventory) codes
SANDAG = (_codes("sf", 1000, 1090, 1110, 1190)  # rural, single family (with or without units)
          | _codes("mf_small", 1120)  # single family multiple-units (duplexes and attached houses on one lot)
          | _codes("mf", 1200, 1280, 1290, 1300)  # multi-family, single-room occupancy, mobile home parks
          | _codes("civ", 1401, 1403, 1404, 1409, 6003, 6102, 6103, 6104, 6105, 6108, 6109, 6701, 6702, 6703, 6807)
          | _codes("edu", 1402, 6801, 6802, 6803, 6804, 6805, 6806, 6809)  # dormitories, colleges, schools
          | _codes("hot", 1501, 1502, 1503)
          | _codes("ind", 2001, 2101, 2103, 2201)
          | _codes("whs", 2104, 2105, 5001)  # warehousing, public storage, wholesale trade
          | _codes("tra", 2301, 4101, 4102, 4103, 4104, 4111, 4112, 4113, 4117, 4118, 4119, 4120)
          | _codes("prk", 4114, 4115, 4116)
          | _codes("ret", 5002, 5003, 5004, 5005, 5006, 5007, 5008, 5009, 7201, 7202, 7203, 7206, 7208, 7209, 7210)
          | _codes("off", 6001, 6002)
          | _codes("hlt", 6501, 6502, 6509)
          | _codes("opn", 6101, 7204, 7205, 7207, 7211, 7601, 7603, 7604, 7605, 7606, 7607, 7609, 8001, 8002, 8003,
                   9200, 9201, 9202)
          | _codes("vac", 9101, 9501, 9502, 9503, 9504, 9505, 9506, 9507)  # vacant; under construction
          | _codes("mix", 9700))


def sandag(code, units=None) -> str:
    """SANDAG land-use code -> class; multi-family of 2-4 units (assessor unit count) is small multi-family."""
    c = str(_num(code)) if _num(code) is not None else ""
    cls = SANDAG.get(c, "unknown")
    n = _num(units)
    if cls == "mf" and n and n <= 4 and c == "1200":
        return "mf_small"
    if cls == "sf" and n and n >= 2:
        return "mf_small" if n <= 4 else "mf"
    return cls


SAN_DIEGO_PLAN = {"Multiple Use": "mix", "Park, Open Space, & Recreation": "opn", "Industrial Employment": "ind",
                  "Roads / Freeways / Transportation": "tra", "Right-of-Way": "tra",
                  "Commercial Employment, Retail, & Services": "ret",
                  "Institutional & Public and Semi-Public Facilities": "civ", "Water Bodies": "opn",
                  "Military Use": "civ", "Agriculture": "opn"}


def san_diego_plan(category, density_hi=None) -> str:
    """City of San Diego General Plan land-use category -> class. Residential by the designation's highest
    density: up to 9 homes an acre single-family, up to 15 small multi-family, more multi-family."""
    c = category.strip() if isinstance(category, str) else ""
    if c == "Residential":
        d = _num(density_hi)
        if d is None:
            return "sf"
        return "sf" if d <= 9 else "mf_small" if d <= 15 else "mf"
    return SAN_DIEGO_PLAN.get(c, "unknown")


# City of San Diego Land Development Code base zone families -> the zone's use class (prefix match)
SAN_DIEGO_ZONE = [("RE-", "sf"), ("RS-", "sf"), ("RX-", "sf"), ("RT-", "mf_small"), ("RM-", "mf"), ("AR-", "sf"),
                  ("AG-", "opn"), ("OP-", "opn"), ("OC-", "opn"), ("OR-", "opn"), ("OF-", "opn"),
                  ("CN-", "ret"), ("CC-", "ret"), ("CR-", "ret"), ("CV-", "hot"), ("CO-", "off"), ("CP-", "prk"),
                  ("EMX-", "mix"), ("RMX-", "mix"), ("IP-", "ind"), ("IL-", "ind"), ("IH-", "ind"), ("IS-", "ind"),
                  ("IBT-", "ind")]


def san_diego_zone(zone) -> str:
    z = zone.upper().strip() if isinstance(zone, str) else ""
    for prefix, cls in SAN_DIEGO_ZONE:
        if z.startswith(prefix):
            return cls
    return "unknown"  # planned districts (CCPD, PD) and specific plans


# ----------------------------------------------------------------------------- zoning district -> use class (NYC,
# Philadelphia, Milwaukee): the use each district family is mapped for, as the "permitted" layer
def nyc_zone(zone) -> str:
    """NYC Zoning Resolution districts: R1-R2 single-family, R3-R5 low-rise (2-4 homes), R6-R10 apartments;
    C4 and C7-C8 commercial, C5-C6 central (offices); M manufacturing; paired M/R districts mixed use."""
    import re

    z = zone.upper().strip() if isinstance(zone, str) else ""
    if "/" in z or z.startswith("BPC"):
        return "mix"
    if z.startswith("PARK"):
        return "opn"
    m = re.match(r"([RCM])(\d+)", z)
    if not m:
        return "unknown"
    kind, n = m.group(1), int(m.group(2))
    if kind == "R":
        return "sf" if n <= 2 else "mf_small" if n <= 5 else "mf"
    if kind == "C":
        return "off" if n in (5, 6) else "ret"
    return "ind"


PHILADELPHIA_ZONE = [("SPPO", "opn"), ("IRMX", "mix"), ("ICMX", "ind"), ("RSD", "sf"), ("RSA", "sf"),
                     ("RTA", "mf_small"), ("RMX", "mix"), ("RM", "mf"), ("CMX", "mix"), ("CA", "ret"),
                     ("I1", "ind"), ("I2", "ind"), ("I3", "ind"), ("IP", "ind")]
MILWAUKEE_ZONE = {"RS": "sf", "RT": "mf_small", "RM": "mf", "RO": "mix", "NS": "ret", "LB": "ret", "RB": "ret",
                  "CS": "ret", "C9": "mix", "IO": "off", "IL": "ind", "IM": "ind", "IH": "ind", "PK": "opn"}


def philadelphia_zone(zone) -> str:
    z = zone.upper().replace("-", "").strip() if isinstance(zone, str) else ""
    return next((cls for prefix, cls in PHILADELPHIA_ZONE if z.startswith(prefix)), "unknown")


def milwaukee_zone(zone) -> str:
    z = zone.upper().strip() if isinstance(zone, str) else ""
    return MILWAUKEE_ZONE.get(z[:2], "unknown")


# ----------------------------------------------------------------------------- LA County Assessor use codes
# (UseCodeDescChar2 of the Parcel Data roll; a code ending in V is vacant land of that type)
LA_COUNTY_USE = [("SINGLE FAMILY", "sf"), ("MANUFACTURED HOME PARK", "mf"), ("MANUFACTURED HOME", "sf"),
                 ("DOUBLE, DUPLEX", "mf_small"), ("THREE UNITS", "mf_small"), ("FOUR UNITS", "mf_small"),
                 ("FIVE OR MORE", "mf"), ("ROOMING", "mf"), ("STORE COMBINATION", "mix"), ("HOTEL", "hot"),
                 ("OFFICE BUILDING", "off"), ("PROFESSIONAL BUILDING", "off"), ("BANK", "ret"), ("PARKING LOT", "prk"),
                 ("WAREHOUSING", "whs"), ("OPEN STORAGE", "whs"), ("WHOLESALE", "whs"), ("LUMBER YARD", "whs"),
                 ("MANUFACTURING", "ind"), ("FOOD PROCESSING", "ind"), ("MOTION PICTURE", "ind"),
                 ("MINERAL PROCESSING", "ind"), ("PETROLEUM", "ind"), ("INDUSTRIAL", "ind"), ("HOSPITAL", "hlt"),
                 ("HOMES FOR AGED", "hlt"), ("SCHOOL", "edu"), ("COLLEGE", "edu"), ("DAY CARE", "edu"),
                 ("CHURCH", "civ"), ("GOVERNMENT OWNED", "civ"), ("CLUB, LODGE", "civ"), ("CEMETERY", "opn"),
                 ("GOLF", "opn"), ("UTILITY", "tra"), ("RIGHTS OF WAY", "tra"), ("PIPELINE", "tra"), ("DUMP", "tra"),
                 ("RIVERS", "opn"), ("THEATER", "ret"), ("ATHLETIC", "ret"), ("RACE TRACK", "ret"), ("BOWLING", "ret"),
                 ("SKATING", "ret"), ("CAMP", "opn"), ("WATER RECREATION", "opn"), ("STORE", "ret"),
                 ("RESTAURANT", "ret"), ("SHOPPING CENTER", "ret"), ("SUPERMARKET", "ret"), ("DEPARTMENT", "ret"),
                 ("SERVICE STATION", "ret"), ("AUTO,", "ret"), ("REPAIR", "ret"), ("NURSERY", "ret"),
                 ("KENNEL", "ret"), ("COMMERCIAL", "ret")]
LA_COUNTY_GROUP = {"DRY FARM": "opn", "IRRIGATED FARM": "opn", "RECREATIONAL": "ret", "INSTITUTIONAL": "civ",
                   "RESIDENTIAL": "sf", "COMMERCIAL": "ret", "INDUSTRIAL": "ind"}


def la_county_use(use_code, desc1=None, desc2=None) -> str:
    """LA County Assessor use code and its descriptions -> class."""
    code = str(use_code).strip().upper() if use_code is not None else ""
    if code.endswith("V"):
        return "vac"
    d2 = desc2.upper() if isinstance(desc2, str) else ""
    for word, cls in LA_COUNTY_USE:
        if word in d2:
            return cls
    return LA_COUNTY_GROUP.get(desc1.upper().strip(), "unknown") if isinstance(desc1, str) else "unknown"


# ----------------------------------------------------------------------------- Dallas Central Appraisal District
DCAD_STATE = {"A11": "sf", "A12": "sf", "A13": "mf", "A20": "sf", "E11": "sf", "O11": "sf", "B12": "mf_small",
              "B11": "mf", "M31": "mf", "F10": "ret", "F20": "ind", "C11": "vac", "C12": "vac", "C13": "vac",
              "C14": "vac", "O10": "vac", "D10": "opn", "J30": "tra", "J40": "tra", "J51": "tra"}


def dcad(state_code, building_class=None, units=None) -> str:
    """DCAD state property tax division code (LAND.CSV SPTD_CD) -> class; commercial and industrial improvements
    are split by the commercial building class (as for the city's layer), residential by unit count."""
    sc = str(state_code).strip().upper() if state_code is not None else ""
    cls = DCAD_STATE.get(sc, "unknown")
    bc = building_class.upper() if isinstance(building_class, str) else ""
    if cls in ("ret", "ind") and bc:
        for word, c in DALLAS_BUILDING:
            if word in bc:
                return c
    n = _num(units)
    if cls in ("sf", "mf") and n and sc not in ("A13",):
        return "sf" if n <= 1 else "mf_small" if n <= 4 else "mf"
    return cls


# ----------------------------------------------------------------------------- Philadelphia2035 proposed land use
# (the City Planning Commission's land-use categories, as in the district plans' Proposed Land Use maps)
PHILADELPHIA_PLAN = {"Residential Low Density": "sf", "Residential Medium Density": "sf",  # rowhouses and twins
                     "Residential Two-Family Attached": "mf_small", "Residential High Density": "mf",
                     "Commercial Consumer": "ret", "Commercial Business / Professional": "off",
                     "Commercial Mixed Residential": "mix", "Industrial": "ind", "Civic/Institution": "civ",
                     "Transportation": "tra", "Greened Right-of-Way": "opn", "Culture / Amusement": "ret",
                     "Active Recreation": "opn", "Park / Open Space": "opn", "Cemetery": "opn", "Water": "opn",
                     "Vacant": "vac", "Vacant Land": "vac"}
# Some district plans drop the group word in the legend ("Low Density" under a "Residential" heading)
PHILADELPHIA_PLAN_SHORT = {"Low Density": "sf", "Medium Density": "sf", "High Density": "mf", "Consumer": "ret",
                           "Business / Professional": "off", "Mixed Residential": "mix"}


def philadelphia_plan(label) -> str:
    """Category of a Philadelphia2035 Proposed Land Use legend label; spacing and punctuation vary between plans."""
    key = lambda x: re.sub(r"[^a-z]", "", x.lower())
    k = key(label) if isinstance(label, str) else ""
    for table in (PHILADELPHIA_PLAN, PHILADELPHIA_PLAN_SHORT):
        for name, v in table.items():
            if k == key(name):
                return v
    return "unknown"
