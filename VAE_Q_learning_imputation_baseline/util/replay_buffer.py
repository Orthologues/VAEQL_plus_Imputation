#########################################################
# Author： Jiawei Zhao
# Email1: jiz@sdu.dk
# Email2: jwz.student.bmc.lu@gmail.com
# Date: 2026-02-19
# Desicrption: The shared replay buffer class for the VAE-Q learning imputation baseline.
#########################################################

from collections import deque, namedtuple
import random
import torch

ReplayTransition = namedtuple("ReplayTransition", ["state", "action", "reward", "next_state", "mask"])
ReplayBatch = namedtuple("ReplayBatch", ["states", "actions", "rewards", "next_states", "masks"])


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, mask) -> None:
        """
        state: Tensor (Current Patient State), z-score normalized or one-hot/unary-encoded as appropriate
        action: Int (0: Down, 1: Up)
        reward: Float (Quality of the nudge)
        next_state: Tensor (New Patient State), discretized & action-applied
        mask: Tensor (Which values are observed vs missing)
        """
        self.buffer.append(
            ReplayTransition(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                mask=mask,
            )
        )
    
    def sample(self, batch_size) -> ReplayBatch:
        batch = random.sample(self.buffer, batch_size)
        return ReplayBatch(
            states=torch.stack([transition.state for transition in batch]),
            actions=torch.tensor([transition.action for transition in batch]),
            rewards=torch.tensor([transition.reward for transition in batch]),
            next_states=torch.stack([transition.next_state for transition in batch]),
            masks=torch.stack([transition.mask for transition in batch]),
        )

    def __len__(self):
        return len(self.buffer)
