import math

NUMBER_OF_TRADES = 1
ASQUARE = 10
X0 = 100
Y0 = 100

INVARIANT = X0 * Y0
XV0 = X0 * math.sqrt(ASQUARE)
YV0 = Y0 * math.sqrt(ASQUARE)

PLOW = (YV0 - Y0) / (INVARIANT * ASQUARE / (YV0 - Y0))
PHIGH = (INVARIANT * ASQUARE / (XV0 - X0)) / (XV0 - X0)

PLOW_POINT = (X0 * (math.sqrt(ASQUARE) - 1), ASQUARE * Y0 / (math.sqrt(ASQUARE) - 1))
PHIGH_POINT = (ASQUARE * X0 / (math.sqrt(ASQUARE) - 1), Y0 * (math.sqrt(ASQUARE) - 1))
