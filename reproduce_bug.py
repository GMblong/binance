import math

def round_step_original(value, step):
    return round(math.floor(value / step) * step, 8)

def round_step_fixed(value, step):
    # Use a small epsilon to handle floating point precision issues
    # Or use decimal module for financial calculations
    precision = int(round(-math.log10(step), 0))
    return round(math.floor((value + 1e-10) / step) * step, precision)

# Test cases
test_cases = [
    (0.3, 0.1),
    (1.2, 0.2),
    (0.00015, 0.00001),
    (100.0, 0.01),
]

print(f"{'Value':<10} | {'Step':<10} | {'Original':<10} | {'Fixed':<10} | {'Expected'}")
print("-" * 60)
for val, step in test_cases:
    orig = round_step_original(val, step)
    fixed = round_step_fixed(val, step)
    expected = val # In these simple cases, they should match
    print(f"{val:<10} | {step:<10} | {orig:<10} | {fixed:<10} | {expected}")

# Specific failure case for floor(0.3 / 0.1)
val, step = 0.3, 0.1
orig = round_step_original(val, step)
print(f"\nFailure Case: round_step(0.3, 0.1)")
print(f"math.floor(0.3 / 0.1) = {math.floor(0.3 / 0.1)}")
print(f"Original Result: {orig}")
