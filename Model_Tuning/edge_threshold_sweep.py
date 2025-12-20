import subprocess
import itertools

EDGE_LEVELS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
SHIFTS = [0.0, -0.25, -0.5, -1.0]

print("edge,shift")

for edge, shift in itertools.product(EDGE_LEVELS, SHIFTS):
    print(f"\n=== edge >= {edge:.2f} | shift {shift:+.2f} ===")

    cmd = [
        "python",
        "execution_clv_report.py",
        "--edge", str(edge),
        "--shift", str(shift),
    ]

    subprocess.run(cmd, check=True)