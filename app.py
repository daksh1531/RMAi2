from flask import Flask, render_template, request, redirect, url_for, flash, session
import csv
import random
import pandas as pd
import networkx as nx
import ast
import os
from tabulate import tabulate as tb

os.environ['PYTHONUNBUFFERED'] = '1'

app = Flask(__name__)
app.secret_key = 'clarion'  # Set your secret key for session management
path = 'processed_data.csv'
projects = []
roles = []
team_size = 0
best_team_result = None
hierarchy_weight = 0.25
rating_weight = 0.25
appraisal_weight = 0.25
synergy_weight = 0.25


def generate_synergy_graph(data):
    graph = nx.Graph()
    for i, employee1 in data.iterrows():
        for j, employee2 in data.iterrows():
            if i != j and employee1['Name'] != '' and employee2['Name'] != '':
                synergy_score = random.randint(0, 3)  # Randomly assign a synergy score between 0 and 3
                graph.add_edge(employee1['Name'], employee2['Name'], weight=synergy_score)
    return graph



def generate_team(size, required_roles, required_skills, role_weight=0.25, skill_weight=0.25, rating_weight=0.25, appraisal_weight=0.25):
    data = pd.read_csv(path).to_dict(orient='records')  # Read data from CSV

    # Convert role names to lowercase for case-insensitive filtering
    required_roles_lower = [role.lower() for role in required_roles]

    # Initialize dictionaries to store members for each role and skill
    selected_roles = {}
    selected_skills = {}

    for d in data:
        d['Skills'] = ast.literal_eval(d['Skills'])
        role = d['Role'].lower()

        # Check if the role is required and not already selected
        if role in required_roles_lower and role not in selected_roles:
            selected_roles[role] = d

        # Check if any required skill is present and not already selected
        for skill in required_skills:
            if skill.lower() in [s.lower() for s in d['Skills']] and skill not in selected_skills:
                selected_skills[skill] = d

    # Combine selected roles and skills to create the initial team
    initial_team = list(selected_roles.values()) + list(selected_skills.values())

    if len(initial_team) < size:
        print("Not enough employees available with the required roles and skills to form a team of the desired size.")
        raise ValueError("Not enough employees available to form a team of the desired size.")

    return initial_team

def expand_neighborhood(team):
    expanded_teams = []
    for i in range(4):
        new_team = team.copy()

        if not new_team:  # Skip expansion if the team is empty
            break

        original_member = random.choice(new_team)

        # Check if the original_member list has the expected structure
        if len(original_member) < 2:
            print(f"Warning: Unexpected structure in original_member list: {original_member}")
            continue
        print(original_member)

        role = original_member['Role']

        df = pd.read_csv(path)
        candidates = df[(df["Role"] == role) & (~df["Name"].isin([member['Name'] for member in team])) & (df["Availability"] == 1)].to_dict(orient='records')


        if not candidates:
            expanded_teams.append(new_team)
            continue

        while True:
            new_member = random.choice(candidates)

            # Check if the new_member list has the expected structure
            if len(new_member) < 2:
                print(f"Warning: Unexpected structure in new_member list: {new_member}")
                continue

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
            print(f"Member1 Keys: {member1.keys()}")

            # Check if 'Rating' key is present in the dictionary
            if 'Rating' not in member1:
                print(f"Warning: 'Rating' key not found in member1 dictionary: {member1}")
                continue

            level = member1['Level']
            levels[level] += 1

            # Modify this part to handle the case where 'Rating' key might be missing
            rating = float(member1['Rating']) if 'Rating' in member1 else 0.0
            ratings.append(rating)

            # Modify this part to handle the case where 'Appraisals' key might be '-'
            try:
                appraisals_member1 = float(member1['Appraisals'])
            except ValueError:
                print(f"Warning: Could not convert 'Appraisals' to float: {member1['Appraisals']}. Setting it to 0.")
                appraisals_member1 = 0.0

            appraisals_sum += appraisals_member1

            for member2 in team:
                if member1 != member2:
                    # Modify this part to handle the case where 'Appraisals' key might be '-'
                    try:
                        appraisals_member2 = float(member2['Appraisals'])
                    except ValueError:
                        print(f"Warning: Could not convert 'Appraisals' to float: {member2['Appraisals']}. Setting it to 0.")
                        appraisals_member2 = 0.0

                    synergy_score = synergy_graph.get_edge_data(member1['Name'], member2['Name'], default={'weight': 0})['weight']
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

    print(f"Initial Team:\n{tb([initial_team], headers='keys', tablefmt='grid').replace('  ', ' ')}")
    print(f"Initial Evaluation Value: {best_index}")
    print(f"Initial Synergy Score: {best_synergy}")

    for team in expanded_teams:
        team_results = evaluate_balance([team], synergy_graph, hierarchy_weight, rating_weight, appraisal_weight, synergy_weight)
        team_index = team_results[0][0]
        team_synergy = team_results[0][1]
        print(f"Evaluated Team:\n{tb([team], headers='keys', tablefmt='grid').replace('  ', ' ')}")
        print(f"Evaluation Value for the Current Team: {team_index}")
        print(f"Synergy Score for the Current Team: {team_synergy}")

        if team_index > best_index:
            best_team = team
            best_index = team_index
            best_synergy = team_synergy
            print(f"New Best Team:\n{tb([best_team], headers='keys', tablefmt='grid').replace('  ', ' ')}")
            print(f"New Best Evaluation Value: {best_index}")
            print(f"New Best Synergy Score: {best_synergy}")

    print(f"Final Best Team:\n{tb([best_team], headers='keys', tablefmt='grid').replace('  ', ' ')}")
    print(f"Final Best Evaluation Value: {best_index}")
    print(f"Final Best Synergy Score: {best_synergy}")

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
            name = row.get('Name', '').strip()
            exp = extract_years_and_months(row.get('Exp.', '').strip())
            core_technology = row.get('Core Technology', '').strip()
            skill = row.get('Skills', '').strip()
            appraisal = row.get('appraisal', '')  # Store behavioral appraisal
            level = get_level(exp)

            # Check if 'availability' (case-insensitive) key exists in the CSV file
            availability = int(row.get('availability', 1))

            if name not in data:
                data[name] = {
                    'Name': name,
                    'Role': core_technology,
                    'Level': level,
                    'Skills': [skill],           # Store each skill in a list
                    'Rating': row.get('behavioural', ''),   # Store behavioral rating
                    'Appraisals': appraisal,   # Store appraisal value
                    'Availability': availability  # Use the extracted availability value
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
    try:
        global projects, roles, team_size, hierarchy_weight, rating_weight, appraisal_weight, synergy_weight, best_team_result

        if request.method == 'POST':
            projects = request.form.get('projects').split(',')
            roles = request.form.get('roles').split(',')
            team_size = int(request.form.get('team_size'))
            hierarchy_weight = float(request.form.get('hierarchy_weight'))
            rating_weight = float(request.form.get('rating_weight'))
            appraisal_weight = float(request.form.get('appraisal_weight'))
            synergy_weight = float(request.form.get('synergy_weight'))
            flash('Project details have been saved.', 'success')

            # Call the function to generate and evaluate the best team
            best_team_result = generate_and_evaluate_team(
                hierarchy_weight, rating_weight, appraisal_weight, synergy_weight
            )

            # Store the result in the session
            session['best_team_result'] = best_team_result

            # Redirect to the same route to avoid resubmission on page refresh
            return redirect(url_for('home'))

        # If it's a GET request, populate the form fields with the previously submitted values
        # Retrieve the result from the session
        best_team_result = session.get('best_team_result', (None, None))
        best_team, iterations = best_team_result

        print("Best Team Result:", best_team_result)  # Add this line for debugging

        return render_template(
            'home.html',
            best_team=best_team,
            iterations=iterations,
            projects=','.join(projects),
            roles=','.join(roles),
            team_size=team_size,
            hierarchy_weight=hierarchy_weight,
            rating_weight=rating_weight,
            appraisal_weight=appraisal_weight,
            synergy_weight=synergy_weight
        )

    except ValueError as ve:
        flash(str(ve), 'error')
        return redirect(url_for('home'))

    except Exception as e:
        flash('An unexpected error occurred.', 'error')
        print(f"Error: {e}")
        return redirect(url_for('home'))
# Async function to generate and evaluate the best team
def generate_and_evaluate_team(hierarchy_weight, rating_weight, appraisal_weight, synergy_weight, iterations=1000):
    try:
        global projects, roles, team_size
        data = pd.read_csv('processed_data.csv')  # Load the processed data

        # The team generation and evaluation process
        best_team = None
        synergy_graph = generate_synergy_graph(data)

        initial_team = generate_team(team_size, projects, roles)

        # Evaluate the initial team
        initial_results = evaluate_balance([initial_team], synergy_graph, hierarchy_weight, rating_weight, appraisal_weight, synergy_weight)
        if not initial_results:
            # Handle the case when the initial team is empty
            return initial_team, iterations

        best_solution = initial_team
        best_index, best_synergy = initial_results[0]

        for i in range(1, iterations + 1):
            print(f"Iteration {i}")
            expanded_teams = expand_neighborhood(best_solution)
            best_team = select_best_team(best_solution, expanded_teams, synergy_graph)
            new_results = evaluate_balance([best_team], synergy_graph, hierarchy_weight, rating_weight, appraisal_weight, synergy_weight)
            new_index, new_synergy = new_results[0]
            if new_index > best_index:
                best_solution = best_team
                best_index = new_index
                best_synergy = new_synergy

        print(f"Final Best Team (inside generate_and_evaluate_team):\n{tb([best_solution], headers='keys', tablefmt='grid').replace('  ', ' ')}")
        print(f"Final Best Evaluation Value (inside generate_and_evaluate_team): {best_index}")
        print(f"Final Best Synergy Score (inside generate_and_evaluate_team): {best_synergy}")

        # After generating the best team, update the global variable with the result
        return best_solution, iterations

    except Exception as e:
       import traceback
       print(f"Error in generate_and_evaluate_team: {e}")
       traceback.print_exc()
       raise


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

    app.run(debug=True)