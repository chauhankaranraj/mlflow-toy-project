import itertools
import yaml


# hyperparameters to tune
hyperparams = {
    'ones': [6, 1, 9],
    'tens': [4, 7],
    'hundreds': [3, 5],
}

# generate a list of dicts with specific values for each parameter
keys, values = zip(*hyperparams.items())
hyperparam_allsets = [dict(hyperparam_set=dict(zip(keys, v))) for v in itertools.product(*values)]
print("Total number of hyperparameter sets: "+str(len(hyperparam_allsets)))

# write to yaml file, each "entry" is a particular realization of hyperparameters
OUT_FNAME = "hyperparameters.yml"
with open(OUT_FNAME, 'w') as outfile:
    yaml.dump(hyperparam_allsets, outfile, default_flow_style=False)
print("Hyperparameter sets saved to: " + OUT_FNAME)
