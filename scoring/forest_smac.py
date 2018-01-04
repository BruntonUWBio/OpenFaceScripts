import joblib
import numpy as np
import os
import sys
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

# Import ConfigSpace and different types of parameters
from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter
from ConfigSpace.conditions import InCondition

# Import SMAC-utilities
from smac.tae.execute_func import ExecuteTAFuncDict
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC

sys.path.append('/home/gvelchuru/OpenFaceScripts')
from scoring.AUScorer import emotion_list
from scoring.EmotionPredictor import make_emotion_data

from pathos.multiprocessing import ProcessingPool as Pool

def make_cs():
    cs = ConfigurationSpace()
    cs.add_hyperparameter(UniformIntegerHyperparameter("n_estimators", 1, 30, default=10))

    max_features = CategoricalHyperparameter('max_features', ['auto', 'value'], default='auto')
    max_features_value = UniformFloatHyperparameter('max_features_value', .1, 1)
    cs.add_hyperparameters([max_features, max_features_value])
    cs.add_condition(InCondition(child=max_features_value, parent=max_features, values=['value']))

    max_depth = CategoricalHyperparameter('max_depth', [None, 'value'])
    max_depth_value = UniformIntegerHyperparameter("max_depth_value", 1, 10)
    cs.add_hyperparameters([max_depth, max_depth_value])
    cs.add_condition(InCondition(child=max_depth_value, parent=max_depth, values=['value']))

    min_samples_split = UniformFloatHyperparameter("min_samples_split", .1, 1)
    cs.add_hyperparameter(min_samples_split)

    min_samples_leaf = UniformFloatHyperparameter("min_samples_leaf", .1, .5)
    cs.add_hyperparameter(min_samples_leaf)

    min_weight_fraction_leaf = UniformFloatHyperparameter("min_weight_fraction_leaf", 0, .5)
    cs.add_hyperparameter(min_weight_fraction_leaf)

    max_leaf_nodes = CategoricalHyperparameter('max_leaf_nodes', [None, 'value'])
    max_leaf_nodes_value = UniformIntegerHyperparameter('max_leaf_nodes_value', 2, 100)
    cs.add_hyperparameters([max_leaf_nodes, max_leaf_nodes_value])
    cs.add_condition(InCondition(child=max_leaf_nodes_value, parent=max_leaf_nodes, values=['value']))

    min_impurity_split = UniformFloatHyperparameter('min_impurity_split', 0, 1)
    cs.add_hyperparameter(min_impurity_split)

    bootstrap = CategoricalHyperparameter('bootstrap', [True, False], default=True)
    cs.add_hyperparameter(bootstrap)


def forest_from_cfg(cfg, emotion):
    for string in ['max_features']:
        cfg[string] = cfg[string + '_value'] if cfg[string] == 'value' else 'auto'
        cfg.pop(string + '_value', None)

    for string in ['max_depth', 'max_leaf_nodes']:
        cfg[string] = cfg[string + '_value'] if cfg[string] == 'value' else None
        cfg.pop(string + '_value', None)

    clf = RandomForestClassifier(**cfg, random_state=42)
    au_data, target_data = make_emotion_data(emotion)
    scores = cross_val_score(clf, au_data, target_data, cv=5)
    return 1 - np.mean(scores)


def use_smac(emotion):
    scenario = Scenario({'run_obj': 'quality',
                         'runcount-limit': 200,
                         "cs": make_cs(),
                         "deterministic": "true",
                         "shared_model": True,
                         "input_psmac_dirs": "smac3-output*",
                         "seed": np.random.RandomState()
                         })
    smac = SMAC(scenario=scenario, rng=np.random.RandomState(42), tae_runner=forest_from_cfg)
    incumbent = smac.optimize()
    # joblib.dump(RandomForestClassifier(**incumbent), '{0}_smac_optimized_random_forest.pkl'.format(emotion))
    inc_value = forest_from_cfg(incumbent, emotion)
    out_writer.write("Optimized Value for {0}: {1}".format(emotion, inc_value))
    out_writer.write('\n' + '\n')
    out_writer.write(incumbent)


if __name__ == '__main__':
    OpenDir = sys.argv[sys.argv.index('-d') + 1]
    os.chdir(OpenDir)
    print("Optimizing")
    out_file = 'smac.txt'
    with open(out_file) as out_writer:
        Pool(len(emotion_list())).map(use_smac, emotion_list())
