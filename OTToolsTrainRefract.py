import pandas as pd
import numpy as np
import datetime
import json
from datetime import timedelta
from ortools.sat.python import cp_model

class Worker:
    def __init__(self, name):
        self.name = name
        self.shifts = []
        self.current_shift = None

    def add_shift(self, shift):
        self.shifts.append(shift)
        self.current_shift = shift

    def switch_shift(self):
        if self.current_shift == 'AM':
            self.current_shift = 'PM'
        else:
            self.current_shift = 'AM'


class Schedule:
    def __init__(self):
        self.workers = []
        self.weeks = []

    def add_worker(self, worker):
        self.workers.append(worker)

    def start_new_week(self):
        for worker in self.workers:
            worker.switch_shift()
        self.weeks.append({worker.name: worker.current_shift for worker in self.workers})

    def get_current_shifts(self):
        return {worker.name: worker.current_shift for worker in self.workers}
    

def least_represented_shift(week_shifts):
        """Returns the shift type (AM or PM) that is least represented in the week."""
        am_count = 0
        pm_count = 0
        for shift in week_shifts:
            if shift == '(AM)':
                am_count += 1
            elif shift == '(PM)':
                pm_count += 1
        if am_count > pm_count:
            return '(PM)'
        else:
            return '(AM)'
    

class Scheduler:
    def __init__(self, config, departures, arrivals, start_date, requested_days_off=None, shift_preferences=None, prev_week_shifts=None):
        self.departures = departures
        self.arrivals = arrivals
        self.start_date = start_date
        self.requested_days_off = requested_days_off
        self.shift_preferences = shift_preferences
        self.prev_week_shifts = prev_week_shifts
        self.schedule = Schedule()

        """
        self.receptionist_names = {
            0: "Dario",
            1: "Steve",
            2: "Goeffrey",
            3: "Nury",
            4: "Jacqueline",
            5: "Eddie",
        }
        self.supervisor_names = {
            0: "Ana Rose",
            1: "Michelle",
            2: "Yousra",
        }
        self.on_call_receptionist_names = {
            0: "Zynk",
            1: "Spark",
            2: "Nova",
        }
        
        """

        self.receptionist_names = {i: name for i, name in enumerate(config['receptionists'])}
        self.supervisor_names = {i: name for i, name in enumerate(config['supervisors'])}
        self.on_call_receptionist_names = {i: name for i, name in enumerate(config['on_call_receptionists'])}

        self.worker_names = list(self.receptionist_names.values()) + list(self.on_call_receptionist_names.values())
        self.supervisor_names = list(self.supervisor_names.values())
        self.dates = [self.start_date + timedelta(days=i) for i in range(len(self.departures))] # create the attribute in the scheduler class


    def create_schedule(self):
        # Constants
        NUM_SUPERVISORS = 3
        NUM_RECEPTIONISTS = 6
        NUM_ONCALL_RECEPTIONISTS = 3
        NUM_SHIFTS = 2
        NUM_DAYS = len(self.departures)
        shift_names = ['AM', 'PM']

        # Create a list of dates
        dates = [self.start_date + timedelta(days=i) for i in range(NUM_DAYS)]

        # Create a dictionary that maps each employee to their desired number of days
        employee_days = {
            0: 5,  # Dario
            1: 5,  # Steve
            2: 5,  # Goeffrey
            3: 5,  # Nury
            4: 4,  # Jacqueline
            5: 5,  # Eddie
        }

        for worker_name in self.receptionist_names.values():
            worker = Worker(worker_name)
            if self.prev_week_shifts and worker_name in self.prev_week_shifts:
                worker.add_shift(self.prev_week_shifts[worker_name])
            self.schedule.add_worker(worker)

        # Determine the number of receptionists required for each shift
        num_receptionists_required = []
        for k in range(len(self.departures)):
            day_requirements = []
            for j in range(2):  # 2 shifts (AM and PM)
                if j == 0:  # AM shift
                    total = self.departures[k]
                else:  # PM shift
                    total = self.arrivals[k]
                if total <= 124:
                    day_requirements.append(2)
                elif total <= 174:
                    day_requirements.append(3)
                else:
                    day_requirements.append(4)
            num_receptionists_required.append(day_requirements)

        # Create the model
        model = cp_model.CpModel()

        # Variables
        supervisor_shifts = {}
        receptionist_shifts = {}
        on_call_receptionist_shifts = {}
        shift_consistency = {}
        consecutive_days_off = {}
        missing_worker = {}
        on_call_receptionists_per_day = {}
        days_off = {}

        
        for i in range(NUM_SUPERVISORS):
            for j in range(NUM_SHIFTS):
                for k in range(NUM_DAYS):
                    supervisor_shifts[(i, j, k)] = model.NewBoolVar(f'supervisor_{i}_shift_{j}_day_{k}')
                    if self.requested_days_off and self.supervisor_names[i] in self.requested_days_off and dates[k] in self.requested_days_off[self.supervisor_names[i]]:
                        model.Add(supervisor_shifts[(i, j, k)] == 0).OnlyEnforceIf(True)

       
        for i in range(NUM_RECEPTIONISTS):
            for j in range(NUM_SHIFTS):
                for k in range(NUM_DAYS):
                    receptionist_shifts[(i, j, k)] = model.NewBoolVar(f'receptionist_{i}_shift_{j}_day_{k}')
                    if self.requested_days_off and self.receptionist_names[i] in self.requested_days_off and dates[k] in self.requested_days_off[self.receptionist_names[i]]:
                        model.Add(receptionist_shifts[(i, j, k)] == 0).OnlyEnforceIf(True)

        
        for i in range(NUM_ONCALL_RECEPTIONISTS):  # 3 on-call receptionists
            for j in range(NUM_SHIFTS):
                for k in range(NUM_DAYS):
                    on_call_receptionist_shifts[(i, j, k)] = model.NewIntVar(0, 1, f'on_call_receptionist_{i}_{j}_{k}')

        
        for i in range(NUM_RECEPTIONISTS):
            for j in range(NUM_SHIFTS):
                shift_consistency[(i, j)] = model.NewBoolVar(f"shift_consistency_{i}_{j}")

        
        for i in range(NUM_RECEPTIONISTS + NUM_SUPERVISORS):
            for k in range(NUM_DAYS - 1):
                consecutive_days_off[(i, k)] = model.NewBoolVar(f"consecutive_days_off_{i}_{k}")

        
        for j in range(NUM_SHIFTS):
            for k in range(NUM_DAYS):
                missing_worker[(j, k)] = model.NewBoolVar(f"missing_worker_{j}_{k}")

        # Add a variable to track the number of on-call receptionists assigned to each day
        for k in range(NUM_DAYS):
            on_call_receptionists_per_day[k] = model.NewIntVar(0, NUM_ONCALL_RECEPTIONISTS, f'on_call_receptionists_day_{k}')

        # Introduce new variable to track days off
        for i in range(NUM_RECEPTIONISTS):
            days_off[i] = model.NewIntVar(0, NUM_DAYS, f'days_off_{i}')

        # Constraints

        for i in range(NUM_RECEPTIONISTS):
            for k in range(NUM_DAYS):
                if self.requested_days_off and self.receptionist_names[i] in self.requested_days_off and dates[k] in self.requested_days_off[self.receptionist_names[i]]:
                    model.Add(sum(receptionist_shifts[(i, j, k)] for j in range(NUM_SHIFTS)) == 0)

        for i in range(NUM_SUPERVISORS):
            for k in range(NUM_DAYS):
                if self.requested_days_off and self.supervisor_names[i] in self.requested_days_off and dates[k] in self.requested_days_off[self.supervisor_names[i]]:
                    model.Add(sum(supervisor_shifts[(i, j, k)] for j in range(NUM_SHIFTS)) == 0)

        # Add a constraint to ensure that the number of on-call receptionists assigned to each day is tracked correctly
        for k in range(NUM_DAYS):
            model.Add(on_call_receptionists_per_day[k] == sum(on_call_receptionist_shifts[(i, j, k)] for i in range(NUM_ONCALL_RECEPTIONISTS) for j in range(NUM_SHIFTS)))

        penalty_terms = []
        for i in range(NUM_RECEPTIONISTS):
            for k in range(NUM_DAYS - 1):
                penalty_term = model.NewIntVar(0, 1, f"penalty_term_{i}_{k}")
                model.Add(penalty_term == 1).OnlyEnforceIf(receptionist_shifts[(i, 1, k)], receptionist_shifts[(i, 0, k+1)])
                model.Add(penalty_term == 0).OnlyEnforceIf(receptionist_shifts[(i, 1, k)].Not(), receptionist_shifts[(i, 0, k+1)].Not())
                penalty_terms.append(penalty_term)

        for i in range(NUM_SUPERVISORS):
            for k in range(NUM_DAYS - 1):
                penalty_term = model.NewIntVar(0, 1, f"penalty_term_{i}_{k}")
                model.Add(penalty_term == 1).OnlyEnforceIf(supervisor_shifts[(i, 1, k)], supervisor_shifts[(i, 0, k+1)])
                model.Add(penalty_term == 0).OnlyEnforceIf(supervisor_shifts[(i, 1, k)].Not(), supervisor_shifts[(i, 0, k+1)].Not())
                penalty_terms.append(penalty_term)

        for i in range(NUM_RECEPTIONISTS):
            model.Add(sum(consecutive_days_off[(i, k)] for k in range(NUM_DAYS - 1)) >= 1)

        for i in range(NUM_SUPERVISORS):
            model.Add(sum(consecutive_days_off[(NUM_RECEPTIONISTS + i, k)] for k in range(NUM_DAYS - 1)) >= 1)

        for i in range(NUM_RECEPTIONISTS):
            for k in range(NUM_DAYS - 1):
                model.Add(sum(receptionist_shifts[(i, j, k)] for j in range(NUM_SHIFTS)) + 
                    sum(receptionist_shifts[(i, j, k+1)] for j in range(NUM_SHIFTS)) == 0).OnlyEnforceIf(consecutive_days_off[(i, k)])
                
        for i in range(NUM_SUPERVISORS):
            for k in range(NUM_DAYS - 1):
                model.Add(sum(supervisor_shifts[(i, j, k)] for j in range(NUM_SHIFTS)) + 
                    sum(supervisor_shifts[(i, j, k+1)] for j in range(NUM_SHIFTS)) == 0).OnlyEnforceIf(consecutive_days_off[(NUM_RECEPTIONISTS + i, k)])
                
        # Each receptionist can only work one shift per day
        for i in range(NUM_RECEPTIONISTS):
            for k in range(NUM_DAYS):
                model.Add(sum(receptionist_shifts[(i, j, k)] for j in range(NUM_SHIFTS)) <= 1)

        # Each supervisor can only work one shift per day
        for i in range(NUM_SUPERVISORS):
            for k in range(NUM_DAYS):
                model.Add(sum(supervisor_shifts[(i, j, k)] for j in range(NUM_SHIFTS)) <= 1)

        # Each supervisor works at most 5 days a week
        for i in range(NUM_SUPERVISORS):
            model.Add(sum(supervisor_shifts[(i, j, k)] for j in range(NUM_SHIFTS) for k in range(NUM_DAYS)) <= 5)

        # Add constraints for each employee to work their desired number of days
        for i in range(NUM_RECEPTIONISTS):
            employee_shifts = sum(receptionist_shifts[(i, j, k)] for j in range(NUM_SHIFTS) for k in range(NUM_DAYS))
            model.Add(employee_shifts <= employee_days[i])

        # Add constraint for at least 2 days off
        for i in range(NUM_RECEPTIONISTS):
            model.Add(days_off[i] >= 2)

        # Update constraint for days off
        for i in range(NUM_RECEPTIONISTS):
            for k in range(NUM_DAYS):
                model.Add(days_off[i] >= 1 - (receptionist_shifts[(i, 0, k)] + receptionist_shifts[(i, 1, k)]))
        
        # Introduce penalty variable
        penalty = {}
        for i in range(NUM_RECEPTIONISTS):
            penalty[i] = model.NewIntVar(0, 1, f'penalty_{i}')

        # Set penalty variable to 1 if worker does not get 2 consecutive days off
        for i in range(NUM_RECEPTIONISTS):
            has_two_days_off = model.NewIntVar(0, 1, f'has_two_days_off_{i}')
            model.Add(days_off[i] >= 2).OnlyEnforceIf(has_two_days_off)
            model.Add(days_off[i] < 2).OnlyEnforceIf(has_two_days_off.Not())
            model.Add(penalty[i] == 1 - has_two_days_off)

        # Each shift has exactly one supervisor
        for j in range(NUM_SHIFTS):
            for k in range(NUM_DAYS):
                model.Add(sum(supervisor_shifts[(i, j, k)] for i in range(NUM_SUPERVISORS)) == 1)
        
        # For each shift, ensure we have AT LEAST the minimum required number of workers
        for j in range(NUM_SHIFTS):
            for k in range(NUM_DAYS):
                model.Add(sum(receptionist_shifts[(i, j, k)] for i in range(NUM_RECEPTIONISTS)) + 
                    sum(on_call_receptionist_shifts[(i, j, k)] for i in range(NUM_ONCALL_RECEPTIONISTS)) == 
                    num_receptionists_required[k][j] - 1)
                
                # Also ensure we don't exceed the maximum number of workers needed
                model.Add(sum(receptionist_shifts[(i, j, k)] for i in range(NUM_RECEPTIONISTS)) + 
                    sum(on_call_receptionist_shifts[(i, j, k)] for i in range(NUM_ONCALL_RECEPTIONISTS)) <= 
                    num_receptionists_required[k][j] + 1)

        # Add constraints to ensure workers switch shifts from their previous week's shifts
        if self.prev_week_shifts:
            for worker_name, prev_shift in self.prev_week_shifts.items():
                # Find the worker's index based on their name
                worker_index = -1
                worker_type = None  # 'supervisor' or 'receptionist'
                
                # Check if worker is a supervisor
                if worker_name in self.supervisor_names:
                    worker_index = list(self.supervisor_names).index(worker_name)
                    worker_type = 'supervisor'
                # Check if worker is a receptionist
                elif worker_name in self.receptionist_names.values():
                    worker_index = list(self.receptionist_names.values()).index(worker_name)
                    worker_type = 'receptionist'
                
                if worker_index != -1:
                    # If they had AM shift last week, force PM shift this week
                    if prev_shift == 'AM':
                        for k in range(NUM_DAYS):
                            if worker_type == 'supervisor':
                                model.Add(supervisor_shifts[(worker_index, 0, k)] == 0)  # Forbid AM shift
                            else:
                                model.Add(receptionist_shifts[(worker_index, 0, k)] == 0)  # Forbid AM shift
                    # If they had PM shift last week, force AM shift this week
                    elif prev_shift == 'PM':
                        for k in range(NUM_DAYS):
                            if worker_type == 'supervisor':
                                model.Add(supervisor_shifts[(worker_index, 1, k)] == 0)  # Forbid PM shift
                            else:
                                model.Add(receptionist_shifts[(worker_index, 1, k)] == 0)  # Forbid PM shift

        # Add shift preference constraints
        if self.shift_preferences:
            for worker_name in self.shift_preferences:
                for date, preferred_shift in self.shift_preferences[worker_name]: # changed from items() to just iterate through the list
                    if date in dates:
                        day_index = dates.index(date)
                        shift_index = shift_names.index(preferred_shift)

                        # Find worker index
                        if worker_name in self.receptionist_names.values():
                            worker_index = list(self.receptionist_names.values()).index(worker_name)
                            # Add hard constraint for the shift
                            model.Add(receptionist_shifts[(worker_index, shift_index, day_index)] == 1)
                            # Ensure no other shift is assigned for this day
                            other_shift_index = 1 - shift_index  # if AM (0) then PM (1), and vice versa
                            model.Add(receptionist_shifts[(worker_index, other_shift_index, day_index)] == 0)
                        elif worker_name in self.supervisor_names:
                            worker_index = self.supervisor_names.index(worker_name)
                            # Add hard constraint for the shift
                            model.Add(supervisor_shifts[(worker_index, shift_index, day_index)] == 1)
                            # Ensure no other shift is assigned for this day
                            other_shift_index = 1 - shift_index  # if AM (0) then PM (1), and vice versa
                            model.Add(supervisor_shifts[(worker_index, other_shift_index, day_index)] == 0)

        # Objective function
        shift_vars = []
        for j in range(NUM_SHIFTS):
            for k in range(NUM_DAYS):
                shift_var = model.NewIntVar(0, 1, f'shift_{j}_{k}')
                shift_vars.append(shift_var)
                # Add constraint to ensure that the shift variable is 1 if any worker is assigned to the shift
                for i in range(NUM_RECEPTIONISTS):
                    model.Add(receptionist_shifts[(i, j, k)] <= shift_var)
                for i in range(NUM_SUPERVISORS):
                    model.Add(supervisor_shifts[(i, j, k)] <= shift_var)

        # variable that contain the objectives values
        objective_terms = [
            sum(shift_vars) + 
            50 * sum(penalty.values()) + 
            sum(receptionist_shifts[(i, j, k)] for i in range(NUM_RECEPTIONISTS) for j in range(NUM_SHIFTS) for k in range(NUM_DAYS)) + 
            sum(supervisor_shifts[(i, j, k)] for i in range(NUM_SUPERVISORS) for j in range(NUM_SHIFTS) for k in range(NUM_DAYS)) + 
            150 * sum(on_call_receptionist_shifts[(i, j, k)] for i in range(NUM_ONCALL_RECEPTIONISTS) for j in range(NUM_SHIFTS) for k in range(NUM_DAYS)) - 
            sum(receptionist_shifts[(i, j, k)] for i in range(NUM_RECEPTIONISTS) for j in range(NUM_SHIFTS) for k in range(NUM_DAYS)) +
            200 * sum(shift_consistency[(i, j)] for i in range(NUM_RECEPTIONISTS) for j in range(NUM_SHIFTS))
        ]

        # Objective function
        model.Minimize(sum(objective_terms))


        class SolutionCallback(cp_model.CpSolverSolutionCallback):
            def __init__(self, supervisor_names, receptionist_names, on_call_receptionist_names, supervisor_shifts, receptionist_shifts, on_call_receptionist_shifts, departures, arrivals, dates, shift_names):
                super().__init__()
                self.supervisor_names = supervisor_names
                self.receptionist_names = receptionist_names
                self.on_call_receptionist_names = on_call_receptionist_names
                self.supervisor_shifts = supervisor_shifts
                self.receptionist_shifts = receptionist_shifts
                self.on_call_receptionist_shifts = on_call_receptionist_shifts
                self.departures = departures
                self.arrivals = arrivals
                self.dates = dates
                self.shift_names = shift_names
                self.solutions = []

            def on_solution_callback(self):
                df = pd.DataFrame(index=range(NUM_SUPERVISORS + NUM_RECEPTIONISTS), columns=['Worker'] + [date.strftime('%Y-%m-%d') for date in dates])

                row = 0
                # Add a row for Boris, the manager
                df.loc[row, 'Worker'] = 'Boris'
                row += 1

                # Add supervisors
                for i in range(NUM_SUPERVISORS):
                    df.loc[row, 'Worker'] = f'{self.supervisor_names[i]}'
                    for k, date in enumerate(dates):
                        for j in range(NUM_SHIFTS):
                            if self.Value(supervisor_shifts[(i, j, k)]) == 1:
                                df.loc[row, date.strftime('%Y-%m-%d')] = f'({shift_names[j]})'
                    row += 1

                # Add a row for Valerie, the extra supervisor
                df.loc[row, 'Worker'] = 'Valérie'
                row += 1

                # Add receptionists
                for i in range(NUM_RECEPTIONISTS):
                    df.loc[row, 'Worker'] = f'{self.receptionist_names[i]}'
                    for k, date in enumerate(dates):
                        for j in range(NUM_SHIFTS):
                            if self.Value(receptionist_shifts[(i, j, k)]) == 1:
                                df.loc[row, date.strftime('%Y-%m-%d')] = f'({shift_names[j]})'
                    row += 1

                # Add on-call receptionists
                for i in range(NUM_ONCALL_RECEPTIONISTS):
                    df.loc[row, 'Worker'] = f'{self.on_call_receptionist_names[i]}'
                    for k, date in enumerate(dates):
                        for j in range(NUM_SHIFTS):
                            if self.Value(on_call_receptionist_shifts[(i, j, k)]) == 1:
                                df.loc[row, date.strftime('%Y-%m-%d')] = f'({shift_names[j]})'
                    row += 1

                # add the numbers of departures and arrivals for each day
                for k, date in enumerate(dates):
                    df.loc[row, 'Worker'] = 'Departures'
                    df.loc[row, date.strftime('%Y-%m-%d')] = self.departures[k]
                    df.loc[row+1, 'Worker'] = 'Arrivals'
                    df.loc[row+1, date.strftime('%Y-%m-%d')] = self.arrivals[k]
                
                df.loc[row+2, 'Worker'] = ''

                self.solutions.append(df)



        # Solve the model
        solver = cp_model.CpSolver()
        solution_callback = SolutionCallback(
            self.supervisor_names,
            self.receptionist_names,
            self.on_call_receptionist_names,
            supervisor_shifts,
            receptionist_shifts,
            on_call_receptionist_shifts,
            self.departures,
            self.arrivals,
            dates,
            shift_names
        )
        solver.parameters.enumerate_all_solutions = True
        solver.parameters.max_time_in_seconds = 60.0  # Set a time limit of 60 seconds
        status = solver.Solve(model, solution_callback)

        # Check if any solutions were found
        if not solution_callback.solutions:
            print("No solutions found. Status:", status)
            # Return an empty schedule if no solution is found
            df = pd.DataFrame(columns=['Worker'] + [date.strftime('%Y-%m-%d') for date in dates])
            df['Worker'] = ['Boris'] + list(self.supervisor_names) + [name for name in self.receptionist_names.values()] + list(self.on_call_receptionist_names)
            return df

        # Get the last solution (usually the best one)
        results = solution_callback.solutions[len(solution_callback.solutions)-1]
        
        return results
    

def post_process_schedule(df, worker_names, num_days=7):
        # Identify worker rows by filtering out non-worker rows
        locked_workers = set()

        def check_and_switch_shifts_weekly(df, worker, worker_names, locked_workers, week_start):
                        week_shifts = df.loc[df['Worker'] == worker, df.columns[week_start+1:week_start+num_days+1]].values.flatten()
                        if len(set(week_shifts)) > 1:  # Worker doesn't have the same shift for the whole week
                            minority_shift = least_represented_shift(week_shifts)
                            # we will iterate through each day of the week
                            #print(week_start, week_start+num_days)
                            for day in range(num_days):
                                #print(f"Checking day {day} for worker {worker}")
                                # we will check if the shift is a minority shift
                                if week_shifts[day] == minority_shift:
                                    #print("alul")
                                    # we will now go over every other worker for that day
                                    has_been_switched = False
                                    other_worker_index = 0
                                    while other_worker_index < len(worker_names) and not has_been_switched:
                                        if worker_names[other_worker_index] in locked_workers:
                                            other_worker_index += 1
                                            continue
                                        else:
                                            other_worker = worker_names[other_worker_index]
                                            opposite_paired_shift = '(AM)' if minority_shift == '(PM)' else '(PM)'
                                            if other_worker != worker and other_worker not in locked_workers and df.loc[df['Worker'] == other_worker, df.columns[week_start+1+day]].values[0] == opposite_paired_shift:
                                                # we will check if the other worker has the opposite shift for that day and if its a minority shift for him
                                                #print(f"Checking if worker {other_worker} has the opposite shift for day {day}")
                                                #rint(f"{minority_shift}")
                                                #print(f"{df.loc[df['Worker'] == other_worker, df.columns[week_start+1+day]].values[0]}")
                                                if df.loc[df['Worker'] == other_worker, df.columns[week_start+1+day]].values[0] != minority_shift:
                                                    #print the shift of the two worker
                                                    #print(f"Switching shifts for worker {worker} and {other_worker} at day {day} from {df.loc[df['Worker'] == worker, df.columns[week_start+1+day]].values[0]} to {df.loc[df['Worker'] == other_worker, df.columns[week_start+1+day]].values[0]}")
                                                    # we will switch the two shifts
                                                    worker_row_index = df.index[df['Worker'] == worker].tolist()[0]
                                                    other_worker_row_index = df.index[df['Worker'] == other_worker].tolist()[0]

                                                    df.iat[worker_row_index, week_start+1+day] = df.loc[df['Worker'] == other_worker, df.columns[week_start+1+day]].values[0]
                                                    df.iat[other_worker_row_index, week_start+1+day] = minority_shift

                                                    has_been_switched = True
                                        other_worker_index += 1


                        return True  # No switch needed

                    # Iterate over each week
            
        for week_start in range(0, len(df.columns) - 2, num_days):
                #reset the locked workers
                locked_workers = set()
                #print(f"Processing week starting on {df.columns[week_start+1]}")
                # Iterate over each worker and attempt to switch shifts if necessary for the current week
                for worker in worker_names:
                    while not check_and_switch_shifts_weekly(df, worker, worker_names, locked_workers, week_start):
                        pass  # Repeat until no more switches are needed
                    locked_workers.add(worker)  # Mark the worker as locked for the current week
                    #print(f"Locked workers: {locked_workers}")

        return df

"""
from collections import Counter

def consistent_shifts(df, worker_names, dates, requested_days_off, shift_preferences):
    '''
    this function go over every worker and checks if they have the same shift for the whole week.
    principe de fonctionnement:
    - on regarde chaque worker, on regarde les paquets de shift du travailleur:
        - si le paquet de shift est du meme type, alors on fait rien
        - si le paquet de shift est different, on regarde quel est le type de shift le plus présent, et on essaye de completer le paquet avec le bon type de shift. on recupere le type de shift de ce paquet et au prochain paquet de shift on mettra l'autre type de shift.
        - Si on ne peut rien changer car pas de possibilitée, on passe au prochain worker
        - a chaque worker fini, on le lock et aucun echange est disponible avec lui.
    '''
    for worker in worker_names:
        shifts = df.loc[worker].tolist()
        i = 0
        previous_shift_type = None

        while i < len(shifts):
            # Trouver le paquet de shifts consécutifs
            j = i
            while j < len(shifts) and shifts[j] == shifts[i]:
                j += 1
            
            current_shift_type = shifts[i]
            
            # Si le type de shift est différent du précédent, on essaie de le corriger
            if previous_shift_type is not None and current_shift_type != previous_shift_type:
                most_common_shift = Counter(shifts[i:j]).most_common(1)[0][0]
                
                # Essayer de compléter le paquet avec le shift le plus fréquent
                for k in range(i, j):
                    if shifts[k] != most_common_shift:
                        if can_change_shift(worker, dates[k], most_common_shift, requested_days_off, shift_preferences):
                            shifts[k] = most_common_shift
            
            # Mettre à jour le type de shift précédent
            previous_shift_type = most_common_shift if 'most_common_shift' in locals() else current_shift_type
            
            i = j
        
        # Mettre à jour le DataFrame avec les nouveaux shifts
        df.loc[worker] = shifts
        
        # Verrouiller le worker
        lock_worker(worker)
    
    return df

def can_change_shift(worker, date, new_shift, requested_days_off, shift_preferences):
    # Vérifier si le travailleur a demandé ce jour de congé
    if date in requested_days_off.get(worker, []):
        return False
    
    # Vérifier si le nouveau shift est dans les préférences du travailleur
    if new_shift not in shift_preferences.get(worker, []):
        return False
    
    # Ajoutez ici d'autres vérifications si nécessaire
    
    return True

def lock_worker(worker):
    # Implémentez ici la logique pour verrouiller un travailleur
    # Cela dépendra de votre système spécifique
    pass

"""

def plot_schedule(config, departures, arrivals, start_date, requested_days_off, shift_preferences=None, prev_week_shifts=None, mid_shifts=None):
    """
    Plot the schedule for multiple weeks.
    """
    # If departures and arrivals are empty, create a scheduler with empty lists for one week
    if not departures and not arrivals:
        scheduler = Scheduler([], [], start_date, requested_days_off, shift_preferences, prev_week_shifts)
        results = scheduler.create_schedule()
    else:
        weekly_departures = [departures[i:i+7] for i in range(0, len(departures), 7)]
        weekly_arrivals = [arrivals[i:i+7] for i in range(0, len(arrivals), 7)]

        results = pd.DataFrame()

        # Call the create_schedule function for each week
        for i, (week_departures, week_arrivals) in enumerate(zip(weekly_departures, weekly_arrivals)):
            week_start_date = start_date + timedelta(days=i*7)
            scheduler = Scheduler(config, week_departures, week_arrivals, week_start_date, requested_days_off, shift_preferences, prev_week_shifts)
            df = scheduler.create_schedule()
            print(f"the value of prev_week_shifts is: {prev_week_shifts}")
            dates = scheduler.dates

            if i == 0:  # First week, include worker names
                results = df
            else:
                df = df.iloc[:, 1:]  # Remove the first column (worker names)
                df.reset_index(drop=True, inplace=True)  # Reset the index
                results.reset_index(drop=True, inplace=True)  # Reset the index of the results DataFrame
                results = pd.concat([results, df], axis=1)  # Concatenate along columns

    # Write the DataFrame to the Excel file
    results.to_excel('solutions.xlsx', index=False)

    post_processed_results = post_process_schedule(results, scheduler.worker_names)
    post_processed_results = post_process_schedule(post_processed_results, scheduler.supervisor_names)

    """
    
    post_processed_results = enforce_weekly_consistency(results, scheduler.worker_names, dates, requested_days_off, shift_preferences)
    post_processed_results = enforce_weekly_consistency(post_processed_results, scheduler.supervisor_names, dates, requested_days_off, shift_preferences)

    """

    """
    # ADD THIS SECTION
    print("\nDario's Schedule Before Post-processing:")
    dario_shifts = results[results['Worker'] == 'Dario']
    if not dario_shifts.empty:
      for col in dario_shifts.columns[1:]:
          date_obj = datetime.datetime.strptime(col, '%Y-%m-%d').date()
          shift = dario_shifts[col].iloc[0]
          print(f"{date_obj.strftime('%Y-%m-%d')}: {shift}")
    else:
      print("Dario not found in the schedule.")


    post_processed_results = post_process_schedule(results, scheduler.worker_names, 
                                                   shift_preferences=shift_preferences, 
                                                   days_off=requested_days_off)
    
    # ADD THIS SECTION
    print("\nDario's Schedule After Post-processing:")
    dario_shifts = post_processed_results[post_processed_results['Worker'] == 'Dario']
    if not dario_shifts.empty:
      for col in dario_shifts.columns[1:]:
          date_obj = datetime.datetime.strptime(col, '%Y-%m-%d').date()
          shift = dario_shifts[col].iloc[0]
          print(f"{date_obj.strftime('%Y-%m-%d')}: {shift}")
    else:
      print("Dario not found in the schedule.")
    
    post_processed_results.to_excel('solutions_post_processed.xlsx', index=False)


    post_processed_results = post_process_schedule(post_processed_results, scheduler.supervisor_names,
                                                   shift_preferences=shift_preferences, 
                                                   days_off=requested_days_off)
    """

    # Write the DataFrame to the Excel file
    post_processed_results.to_excel('solutions_post_processed.xlsx', index=False)

    # Après avoir créé le schedule
    if mid_shifts:
        from appshcedule import MyApp  # Importez la classe MyApp
        resultsWithMidshift = MyApp.apply_mid_shifts(post_processed_results, mid_shifts)
        resultsWithMidshift.to_excel('solutions_with_mid_shifts.xlsx', index=False)

    
