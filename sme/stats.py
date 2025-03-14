"""
A simplified statistical reporting module for SME.
Provides a function to generate a basic stats table.
"""
def generate_stats_table(estimated_params, standard_errors):
    lines = ["Parameter Estimates and Standard Errors"]
    for i, (p, se) in enumerate(zip(estimated_params, standard_errors)):
        lines.append(f"Param {i}: {p:.4f} (SE: {se:.4f})")
    return "\n".join(lines)