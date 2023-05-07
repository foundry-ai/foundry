import jax
import stanza.envs.pusht as pusht

data = pusht.expert_data()
from stanza.data.trajectory import chunk_trajectory
from stanza.data import PyTreeData
from functools import partial
chunked = data.map(partial(chunk_trajectory, 
                chunk_size=16, start_padding=1, end_padding=7))

traj0 = PyTreeData.from_data(data[0], chunk_size=2048)
traj0_chunked = PyTreeData.from_data(chunked[0], chunk_size=2048)
print(f"Got data {len(traj0)}, chunked {len(traj0_chunked)}")
print(jax.tree_util.tree_map(lambda x: x[:20], traj0.data.observation))
print(jax.tree_util.tree_map(lambda x: x[:20], traj0.data.action))
print("Chunked:")
print(jax.tree_util.tree_map(lambda x: x[:2], traj0_chunked.data.observation))
print(jax.tree_util.tree_map(lambda x: x[:2], traj0_chunked.data.action))

# data = PyTreeData.from_data(chunked.flatten(), chunk_size=2048)