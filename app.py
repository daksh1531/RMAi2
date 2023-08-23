from flask import Flask, render_template, request, redirect, url_for,flash
import csv
import random
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from tabulate import tabulate as tb
import csv
import random
import ast
import asyncio
import os
import threading
import concurrent.futures


app = Flask(__name__)
# Sample data to store project requirements, roles, and team size
projects = []
roles = []
team_size = 0
path = 'processed_data.csv'
global best_team_result
best_team_result=None
global hierarchy_weight,rating_weight,appraisal_weight,synergy_weight

def generate_synergy_graph(data):
    graph = nx.Graph()
    for i, employee1 in data.iterrows():
        for j, employee2 in data.iterrows():
            if i != j and employee1['Name'] != '' and employee2['Name'] != '':
                synergy_score = random.randint(0, 3)  # Randomly assign a synergy score between 0 and 3
                graph.add_edge(employee1['Name'], employee2['Name'], weight=synergy_score)
    return graph



def generate_team(size, required_roles, required_skills):
    data = pd.read_csv(path).to_dict(orient='records')  # Read data from CSV
    print(size, required_roles,required_skills)
    for d in data:
        d['Skills'] = ast.literal_eval(d['Skills'])
    # Filter employees based on availability, required roles, and required skills (case-insensitive)
    filtered_data = [
    d for d in data
    if (
        d['Availability'] == 1
        and d['Role'].lower() in [role.lower() for role in required_roles]
        and any(skill.lower() in [s.lower() for s in d['Skills']] for skill in required_skills)
    )]
    if len(filtered_data) < size:
        raise ValueError("Not enough employees available to form a team of the desired size.")

    team = random.sample(filtered_data, size)

    return [(m['Name'], m['Role'], m['Level'], m['Skills'], m['Rating'], m['Appraisals']) for m in team]


def expand_neighborhood(team):
    expanded_teams = []
    for i in range(4):
        new_team = team.copy()

        if not new_team:  # Skip expansion if the team is empty
            break

        original_member = random.choice(new_team)
        role = original_member[1]

        df = pd.read_csv(path)
        candidates = df[(df["Role"] == role) & (~df["Name"].isin([member[0] for member in team])) & (df["Availability"] == 1)].values.tolist()

        if not candidates:
            expanded_teams.append(new_team)
            continue

        while True:
            new_member = random.choice(candidates)
            if new_member not in new_team:
                new_team[new_team.index(original_member)] = new_member
                expanded_teams.append(new_team)
                break

    return expanded_teams




def evaluate_balance(teams, synergy_graph, hierarchy_weight, rating_weight, appraisal_weight, synergy_weight):
    results = {}

    for i, team in enumerate(teams):
        if not team:
            # Skip empty teams
            continue

        levels = {"Junior": 0, "Intermediate": 0, "Senior": 0}
        ratings = []
        appraisals_sum = 0
        synergy_score_sum = 0
        unique_pairs = 0

        for member1 in team:
            level = member1[2]
            levels[level] += 1
            ratings.append(float(member1[4]))
            appraisals_sum += float(member1[5])

            for member2 in team:
                if member1 != member2:
                    synergy_score = synergy_graph.get_edge_data(member1[0], member2[0], default={'weight': 0})['weight']
                    synergy_score_sum += synergy_score
                    unique_pairs += 1

        avg_rating = sum(ratings) / len(ratings) if ratings else 0
        appraisal_factor = appraisals_sum / (len(team) * 5) if team else 0
        synergy_factor = synergy_score_sum / unique_pairs if unique_pairs > 0 else 0

        hierarchy_balance = 10 - abs(levels["Junior"] - levels["Intermediate"]) - abs(levels["Junior"] - levels["Senior"]) - abs(levels["Intermediate"] - levels["Senior"])

        if levels["Junior"] != len(team):
            hierarchy_balance *= 0.5  # Reduce the hierarchy balance factor for teams with mixed hierarchy levels

        evaluation = (
            hierarchy_weight * hierarchy_balance +
            rating_weight * avg_rating +
            appraisal_weight * appraisal_factor +
            synergy_weight * synergy_factor
        )

        results[i] = evaluation, synergy_factor
    return results




def select_best_team(initial_team, expanded_teams, synergy_graph):
    # Passing additional parameters hierarchy_weight, rating_weight, appraisal_weight, and synergy_weight
    initial_results = evaluate_balance([initial_team], synergy_graph, hierarchy_weight, rating_weight, appraisal_weight, synergy_weight)
    if not initial_results:
        # Handle the case when the initial team is empty
        return initial_team

    best_team = initial_team
    best_index = initial_results[0][0]
    best_synergy = initial_results[0][1]

    for team in expanded_teams:
        team_results = evaluate_balance([team], synergy_graph, hierarchy_weight, rating_weight, appraisal_weight, synergy_weight)
        team_index = team_results[0][0]
        team_synergy = team_results[0][1]
        print(f"    Evaluated Neighbor: {tb(team, headers=['Name', 'Role', 'Level', 'Skills', 'Rating', 'Appraisals']).replace('  ', ' ')}")
        print(f"        Evaluation Value: {team_index}")
        print(f"        Synergy Score: {team_synergy}")
        if team_index > best_index:
            best_team = team
            best_index = team_index
            best_synergy = team_synergy
            print(f"        New Best Solution: {tb(best_team, headers=['Name', 'Role', 'Level', 'Skills', 'Rating', 'Appraisals']).replace('  ', ' ')}")
    return best_team


def find_best_team(roles, projects, team_size):
    # Load the processed data from CSV and generate a synergy graph
    data = pd.read_csv(path)
    synergy_graph = generate_synergy_graph(data)

    required_roles = roles
    required_skills = projects

    best_solution = generate_team(team_size, required_roles, required_skills)
    best_index, best_synergy = evaluate_balance([best_solution], synergy_graph).get(0, (None, None))

    for i in range(1, 1000):
        print(f"Iteration {i}")
        expanded_teams = expand_neighborhood(best_solution)
        best_team = select_best_team(best_solution, expanded_teams, synergy_graph)
        new_index, new_synergy = evaluate_balance([best_team], synergy_graph).get(0, (None, None))
        if new_index is not None and new_synergy is not None and new_index > best_index:
            best_solution = best_team
            best_index = new_index
            best_synergy = new_synergy
            print(f"Best Solution Found: {tb(best_solution, headers=['Name', 'Role', 'Level', 'Skills', 'Rating', 'Appraisals']).replace('  ', ' ')}")
            print(f"Evaluation Value of Best Solution: {best_index}")
            print(f"Synergy Score of Best Solution: {best_synergy}")
        else:
            print("No improvement found")

    return best_solution


def extract_years_and_months(exp):
    parts = exp.split()
    years = 0
    months = 0

    for part in parts:
        if 'y' in part:
            years = int(part.strip('y'))
        elif 'm' in part:
            months = int(part.strip('m'))

    total_experience = years + months / 12
    return total_experience

def get_level(experience):
    if experience >= 7:
        return "Senior"
    elif experience >= 5:
        return "Intermediate"
    elif experience >= 3:
        return "Junior"
    else:
        return "Junior"

def get_random_availability():
      return 1#random.randint(0, 1)

def extract_attributes(input_file, output_file):
    data = {}
    with open(input_file, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            name = row['Name'].strip()
            exp = extract_years_and_months(row['Exp.'].strip())
            core_technology = row['Core Technology'].strip()
            skill = row['Skills'].strip()
            appraisal = row['appraisal']  # Store behavioral appraisal
            level = get_level(exp)
            availability = get_random_availability()

            if name not in data:
                data[name] = {
                    'Name': name,
                    'Role': core_technology,
                    'Level': level,
                    'Skills': [skill],           # Store each skill in a list
                    'Rating': row['behavioural'],   # Store behavioral rating
                    'Appraisals': appraisal,   # Store appraisal value
                    'Availability': availability
                }
            else:
                data[name]['Skills'].append(skill)                 # Add additional skill to the list

    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['Name', 'Role', 'Level', 'Skills', 'Rating', 'Appraisals', 'Availability']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for employee_data in data.values():
            writer.writerow(employee_data)

@app.route('/', methods=['GET', 'POST'])
def home():
    global projects, roles, team_size
    global hierarchy_weight,rating_weight,appraisal_weight,synergy_weight

    if request.method == 'POST':
        projects = request.form.get('projects').split(',')
        roles = request.form.get('roles').split(',')
        team_size = int(request.form.get('team_size'))
        hierarchy_weight = float(request.form.get('hierarchy_weight'))
        rating_weight = float(request.form.get('rating_weight'))
        appraisal_weight = float(request.form.get('appraisal_weight'))
        synergy_weight = float(request.form.get('synergy_weight'))
        flash('Project details have been saved.', 'success')

        # Call the asynchronous function to generate and evaluate the best team in a separate thread
        # t = threading.Thread(target=generate_and_evaluate_team, args=(hierarchy_weight, rating_weight, appraisal_weight, synergy_weight))
        # t.start()
        # t.join()
        #best_team_result = t.result()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(generate_and_evaluate_team, hierarchy_weight, rating_weight, appraisal_weight, synergy_weight)
            best_team_result = future.result()
        # Get the result from the global variable after the processing is done
        #global best_team_result
        best_team, iterations = best_team_result

        return render_template('home.html', best_team=best_team, iterations=iterations,
                               projects=','.join(projects), roles=','.join(roles), team_size=team_size,
                               hierarchy_weight=hierarchy_weight, rating_weight=rating_weight,
                               appraisal_weight=appraisal_weight, synergy_weight=synergy_weight)

    # If it's a GET request, populate the form fields with the previously submitted values
    return render_template('home.html', projects=','.join(projects), roles=','.join(roles), team_size=team_size)


# Async function to generate and evaluate the best team
def generate_and_evaluate_team(hierarchy_weight, rating_weight, appraisal_weight, synergy_weight):
    global projects, roles, team_size,best_team_result
    data = pd.read_csv('processed_data.csv')  # Load the processed data

    # The team generation and evaluation process
    best_team = None
    iterations = 1000  # Number of iterations for generating the best team
    synergy_graph = generate_synergy_graph(data)

    initial_team = generate_team(team_size, projects, roles)

    # Evaluate the initial team
    initial_results = evaluate_balance([initial_team], synergy_graph, hierarchy_weight, rating_weight, appraisal_weight, synergy_weight)
    if not initial_results:
        # Handle the case when the initial team is empty
        return initial_team

    best_solution = initial_team
    best_index, best_synergy = initial_results[0]

    for i in range(1, 1000):
        print(f"Iteration {i}")
        expanded_teams = expand_neighborhood(best_solution)
        best_team = select_best_team(best_solution, expanded_teams, synergy_graph)
        new_results = evaluate_balance([best_team], synergy_graph, hierarchy_weight, rating_weight, appraisal_weight, synergy_weight)
        new_index, new_synergy = new_results[0]
        if new_index > best_index:
            best_solution = best_team
            best_index = new_index
            best_synergy = new_synergy

    # After generating the best team, update the global variable with the result
    return best_team, iterations



@app.route('/add', methods=['GET', 'POST'])
def add_data():
    if request.method == 'POST':
        # Process the uploaded CSV file
        if 'csv_file' not in request.files:
            flash('No selected file. Please choose a CSV file to upload.', 'error')
            return redirect(request.url)

        csv_file = request.files['csv_file']
        if csv_file.filename == '':
            flash('No selected file. Please choose a CSV file to upload.', 'error')
            return redirect(request.url)

        # Check if the file has a valid CSV extension
        if not csv_file.filename.endswith('.csv'):
            flash('Invalid file format. Please choose a CSV file to upload.', 'error')
            return redirect(request.url)

        # Adjust the file path for saving the uploaded CSV file
        csv_file.save('uploaded_data.csv')

        # Adjust the file paths for CSV processing and output
        input_file = 'uploaded_data.csv'
        output_file = 'processed_data.csv'
        extract_attributes(input_file, output_file)

        flash('CSV data has been processed successfully.', 'success')
        return redirect(url_for('read_data'))

    return render_template('add_data.html')

@app.route('/read')
def read_data():
    data = []
    with open('processed_data.csv', 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append(row)
    return render_template('read_data.html', data=data)

@app.route('/synergy')
def synergy_graph():
    return render_template('synergy.html')

if __name__ == '__main__':
    # Check if the processed data file exists, if not, extract attributes from the raw data file
    if not os.path.isfile(path):
        input_file = 'raw_data.csv'
        output_file = 'processed_data.csv'
        extract_attributes(input_file, output_file)

    app.secret_key = 'clarion'
    app.run(debug=True)
