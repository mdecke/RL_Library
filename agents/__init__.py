from typing import Dict

from .DDPG_agent import DdpgAgent
from .TD_agent import TdAgent
# from .SAC_agent import SACAgent
from .Expert import MLEExpert

AGENTS = {
    # "sac": SACAgent,
    "ddpg": DdpgAgent,
    "tdn": TdAgent,
}

EXPERTS = {
    "mle": MLEExpert,  # Placeholder, actual class defined in expert.py
    # "cnf": "CNFExpert",  # Placeholder for future implementation
}

def create_agent(env, algo_type:str, cfg:Dict):
    agent_type = algo_type.strip().lower()
    if agent_type not in AGENTS:
        raise ValueError(f"Agent type '{agent_type}' is not recognized. Available types: {list(AGENTS.keys())}")
    agent_class = AGENTS[agent_type]
    return agent_class(env, cfg)

def create_expert(expert_type:str, cfg:Dict):
    expert_type_lower = expert_type.strip().lower()
    if expert_type_lower not in EXPERTS:
        raise ValueError(f"Expert type '{expert_type}' is not recognized. Available types: {list(EXPERTS.keys())}")
    expert_class = EXPERTS[expert_type_lower]
    return expert_class(cfg)