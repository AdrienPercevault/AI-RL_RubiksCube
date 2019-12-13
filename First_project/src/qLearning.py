import numpy as np


def _choose_action(env, state, epsilon, Q):
    action=0
    if np.random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[state, :])
    return action


def _qlearning_learn(state, state2, reward, action, action2, lr_rate, gamma, Q):
    old_value = Q[state, action]
    learned_value = reward + gamma * np.max(Q[state2, :])
    Q[state, action] = (1 - lr_rate) * old_value +  lr_rate * learned_value


def start(env, state, epsilon, Q, total_episodes, max_steps, lr_rate):
    for episode in range(total_episodes):
        state = env.reset()
        action = _choose_action(env, state, epsilon, Q)
        t = 0
        while t < max_steps:
            #env.render()  
            state2, reward, done, info = env.step(action)
            action2 = _choose_action(env, state2, epsilon, Q) 
            _qlearning_learn(state, state2, reward, action, action2, lr_rate, gamma, Q)
            state = state2
            action = action2
            t += 1
            if done:
                break
