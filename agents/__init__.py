from typing import Dict

from .DDPG_agent import DdpgAgent
from .TD_agent import TdAgent
# from .SAC_agent import SACAgent

AGENTS = {
    # "sac": SACAgent,
    "ddpg": DdpgAgent,
    "tdn": TdAgent,
}

def create_agent(env, algo_type:str, cfg:Dict):
    agent_type = algo_type.strip().lower()
    if agent_type not in AGENTS:
        raise ValueError(f"Agent type '{agent_type}' is not recognized. Available types: {list(AGENTS.keys())}")
    agent_class = AGENTS[agent_type]
    return agent_class(env, cfg)