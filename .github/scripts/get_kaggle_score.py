import subprocess
import pandas as pd
import sys
import os
import time

# get the competition name as a argument
competition_name = sys.argv[1]

# get the score from the kaggle leaderboard
def get_kaggle_score():
    command = f"kaggle competitions submissions -c {competition_name} --csv"
    result = subprocess.Popen(command,shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()[0]
    result = result.decode("utf-8") # convert bytes to string
    with open("result.csv", "w") as f:
        f.write(result)

    df = pd.read_csv("result.csv")
    return df["publicScore"].iloc[0]

def save_env_variable(var_name, var_value):
    env_file = os.getenv('GITHUB_ENV')

    with open(env_file, "a") as myfile:
        myfile.write(f"{var_name}={var_value}")


if __name__ == "__main__":
    time.sleep(60) # wait for the submission to be processed
    score = get_kaggle_score()

    print (f"Kaggle score: {score}")
    save_env_variable("KAGGLE_SCORE", str(score))