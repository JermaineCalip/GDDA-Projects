import sys

# Skip header
header_skipped = False

for line in sys.stdin:
    if not header_skipped:
        header_skipped = True
        continue  # Skip the first line (header)

    # Strip leading/trailing whitespace and split by commas
    data = line.strip().split(",")
    
    if len(data) == 8:  # Ensure the line has 8 fields 
        country, price, quantity = data[7], data[5], data[3]

        try:
            # Calculate revenue
            revenue = float(price) * int(quantity)
            # Output: country and revenue 
            print(f"{country}\t{revenue}")
        except ValueError:
            # Skip lines with invalid price or quantity
            continue
    else:
        # If line doesn't have exactly 8 columns, print a debug message
        print(f"Skipping invalid line: {line}", file=sys.stderr)
