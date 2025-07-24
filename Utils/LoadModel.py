import yaml
import warnings
from DeepLearning_Models.utils.general import join, plot_combined
from DeepLearning_Models.ActorCritic.policy_gradient import PolicyGradient
import torch

# Suppress all deprecation warnings. This is because python 3.8.19 is the only overlap between gym's ranges of compatability and pytorch's compatability
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)

#string joiner
yaml.add_constructor("!join", join)

def read_config(DOMAIN):
    config_file = open("config_yamls/config_envs/{}.yml".format(DOMAIN))
    config = yaml.load(config_file, Loader=yaml.FullLoader)
    config.update(yaml.load(open("config_yamls/config_explanations/{}.yml".format(DOMAIN), encoding="utf8"), Loader= yaml.FullLoader))
    return config
def load_weights(config, PATH = None):

    model = PolicyGradient(config, 100)
    if PATH == None:
        PATH_actor = config["output"]["actor_output"]
        PATH_critic = config["output"]["critic_output"]
    if torch.cuda.is_available():
        model.baseline_network.load_state_dict(torch.load(PATH_critic))
        model.policy.load_state_dict((torch.load(PATH_actor)))
    else:
        model.baseline_network.load_state_dict(torch.load(PATH_critic,map_location=torch.device('cpu')))
        model.policy.load_state_dict((torch.load(PATH_actor,map_location=torch.device('cpu'))))

    return model

