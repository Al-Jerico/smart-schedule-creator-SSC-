import pandas as pd

#function that take a number of arrivals/departures and give a number of workers needed
def calculate_workers_needed(activity_num):
    if activity_num <= 125:
        return 2
    elif activity_num <= 175:
        return 3
    else:
        return 4
    
def highlight_short_nights(row):
    """
    Applies red background color to cells where there's a short night
    (PM shift followed by AM shift the next day)
    """
    return ['background-color: white' if pd.isna(cell) else 
            'background-color: #ffcccc' if i > 0 and i < len(row)-1 and 
            not pd.isna(row[i-1]) and '(PM)' in str(row[i-1]) and 
            '(AM)' in str(cell) else 'background-color: white' 
            for i, cell in enumerate(row)]

def update_schedule(existing_schedule, new_numbers):

    # Copy the existing schedule
    updated_schedule = existing_schedule.copy()

    # Extract arrivals and departures from existing schedule
    existing_arrivals = []
    existing_departures = []

    for index, row in existing_schedule.iterrows():
        if 'Arrivals' in row.values:
            existing_arrivals = row.tolist()[1:]
        if 'Departures' in row.values:
            existing_departures = row.tolist()[1:]


    # Extract arrivals and departures from new schedule
    new_arrivals = []
    new_departures = []

    for index, row in new_numbers.iterrows():
        if 'Arrivals' in row.values:
            new_arrivals = row.tolist()[1:]
        if 'Departures' in row.values:
            new_departures = row.tolist()[1:]


    # Update the arrivals and departures in the copied schedule
    for index, row in updated_schedule.iterrows():
        if 'Arrivals' in row.values:
            updated_schedule.loc[index] = ['Arrivals'] + new_arrivals
        if 'Departures' in row.values:
            updated_schedule.loc[index] = ['Departures'] + new_departures

    #print(existing_arrivals)
    #print(new_arrivals)

    #rint(existing_departures)
    #print(new_departures)

    # Create new rows for AM and PM changes
    am_changes = ['AM changes']
    pm_changes = ['PM changes']
    
    for day in range(len(existing_arrivals)):
        am_change_msg = "0"
        pm_change_msg = "0"

        existing_am_workers = calculate_workers_needed(existing_arrivals[day])
        new_am_workers = calculate_workers_needed(new_arrivals[day])
        am_workers_diff = new_am_workers - existing_am_workers
        if am_workers_diff != 0:
            am_change_msg = f"{'+' if am_workers_diff > 0 else ''}{am_workers_diff} workers"

        existing_pm_workers = calculate_workers_needed(existing_departures[day])
        new_pm_workers = calculate_workers_needed(new_departures[day])
        pm_workers_diff = new_pm_workers - existing_pm_workers
        if pm_workers_diff != 0:
            pm_change_msg = f"{'+' if pm_workers_diff > 0 else ''}{pm_workers_diff} workers"

        am_changes.append(am_change_msg)
        pm_changes.append(pm_change_msg)

    # Add new rows to the updated schedule
    am_changes_row = pd.DataFrame([am_changes], columns=updated_schedule.columns)
    pm_changes_row = pd.DataFrame([pm_changes], columns=updated_schedule.columns)
    updated_schedule = pd.concat([updated_schedule, am_changes_row, pm_changes_row], ignore_index=True)

    # Apply styling to highlight short nights
    styled_schedule = updated_schedule.style.apply(highlight_short_nights, axis=1)
    
    return styled_schedule, updated_schedule  # Return both styled and raw schedule