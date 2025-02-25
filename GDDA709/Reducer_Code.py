import sys
current_country = None
current_total = 0.0

for line in sys.stdin:
    # Strip any whitespace and split by tab to get the country and revenue
    line = line.strip()
    # Ensure the line is not empty
    if not line:
        continue
    try:
        country, revenue = line.split("\t")  
        revenue = float(revenue)  # Convert revenue to float for aggregation
    except ValueError:
        # If there is an error splitting or converting, print a debug message and skip this line
        print(f"Skipping invalid line: {line}", file=sys.stderr)
        continue

    # Debugging
    print(f"Processing country: {country}, revenue: {revenue}", file=sys.stderr)

    # If the current country matches the previous country, accumulate the revenue
    if country == current_country:
        current_total += revenue
    else:
        # If we have a previous country, print the result
        if current_country:
            print(f"{current_country}\t{current_total}")
        
        # Reset for the new country
        current_country = country
        current_total = revenue

# Output the final country and total revenue
if current_country:
    print(f"{current_country}\t{current_total}")
