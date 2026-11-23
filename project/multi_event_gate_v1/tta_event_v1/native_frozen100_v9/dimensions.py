"""Predeclared grouping only, original100 scenes per task unchanged."""
WIDTHS=(4,)*7+(5,)*12+(4,)*3
assert len(WIDTHS)==22 and sum(WIDTHS)==100
assert WIDTHS[:7]==(4,)*7

def lanes(index):return 2*WIDTHS[index]
