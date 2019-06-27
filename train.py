import yaml
import pprint
import random
import mlflow


def get_hyperparameters(job_id, hyperparams_fname="hyperparameters.yml"):
    # Update file name with correct path
    with open(hyperparams_fname, 'r') as stream:
        hyperparam_set = yaml.load(stream)

    print("\nHypermeter set for job_id: ", job_id)
    print("------------------------------------")
    pprint.pprint(hyperparam_set[job_id]["hyperparam_set"])
    print("------------------------------------\n")

    return hyperparam_set[job_id-1]["hyperparam_set"]


def calculate_value(ones, tens, hundreds):
    # log hyperparams for this run
    mlflow.log_param('ones', ones)
    mlflow.log_param('tens', tens)
    mlflow.log_param('hundreds', hundreds)

    # calculate value
    result = ones + 10*tens + 100*hundreds

    # assume ground truth is whether result was correct or not
    truth = int(str(hundreds) + str(tens) + str(ones))
    score = truth==result

    # how well did the model do this run
    mlflow.log_metric('is_correct', score)


if __name__ == "__main__":
    # get hyperparameters for this specific job
    currjob_hyperparams = get_hyperparameters(random.randint(1, 5))

    # run training
    calculate_value(currjob_hyperparams["ones"],
                          currjob_hyperparams["tens"],
                          currjob_hyperparams["hundreds"])
