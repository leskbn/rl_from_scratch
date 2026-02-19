import numpy as np


def generate_episode(env, policy, max_steps=100):
    state = env.reset()
    t = 0
    states = []
    rewards = []

    while t < max_steps:
        a = np.random.choice(env.n_actions, p=policy[state])
        next_state, reward, terminated, truncated, _ = env.step(a)

        t += 1

        states.append(state)
        rewards.append(reward)

        state = next_state

        if terminated or truncated:
            break

    return states, rewards


def mc_first_visit_prediction(env, policy, num_episodes, gamma, max_steps=100):
    V = np.zeros(env.S)
    N = np.zeros(env.S, dtype=np.int64)

    for ep in range(num_episodes):
        G = 0
        states, rewards = generate_episode(env, policy, max_steps)
        returns = np.zeros(len(states))

        for t in reversed(range(len(rewards))):
            G = rewards[t] + gamma * G
            returns[t] = G

        visited = set()

        for t in range(len(states)):
            G_t = returns[t]
            s = states[t]
            r, c = env.decode(s)
            if s == env.goal_id or ((r, c) in getattr(env, "obstacles", set())):
                continue
            if s not in visited:
                visited.add(s)
                N[s] += 1
                # incremental mean, 모든 에피소드를 다 끝내고 평균 내는 것이 아니라 에피소드마다 업데이트 하기 위해
                V[s] += (G_t - V[s]) / N[s]

    return V


def mc_every_visit_prediction(env, policy, num_episodes, gamma, max_steps=100):
    V = np.zeros(env.S)
    N = np.zeros(env.S, dtype=np.int64)

    for ep in range(num_episodes):
        G = 0
        states, rewards = generate_episode(env, policy, max_steps)

        for t in reversed(range(len(rewards))):
            G = rewards[t] + gamma * G
            s = states[t]
            r, c = env.decode(s)
            if s == env.goal_id or ((r, c) in getattr(env, "obstacles", set())):
                continue
            N[s] += 1
            # incremental mean, 모든 에피소드를 다 끝내고 평균 내는 것이 아니라 에피소드마다 업데이트 하기 위해
            V[s] += (G - V[s]) / N[s]

    return V


def mc_prediction(env, policy, num_episodes, gamma, max_steps=100, first_visit=True):
    if first_visit:
        return mc_first_visit_prediction(env, policy, num_episodes, gamma, max_steps)
    else:
        return mc_every_visit_prediction(env, policy, num_episodes, gamma, max_steps)


def select_action_eps_greedy(Q, s, eps, n_actions):
    u = np.random.rand()

    if u < eps:
        return np.random.randint(n_actions)
    else:
        return int(np.argmax(Q[s]))


def generate_episode_sar(env, Q, eps, max_steps=100):
    state = env.reset()
    t = 0
    states = []
    rewards = []
    actions = []

    while t < max_steps:
        a = select_action_eps_greedy(Q, state, eps, env.n_actions)
        next_state, reward, terminated, truncated, _ = env.step(a)

        t += 1

        states.append(state)
        rewards.append(reward)
        actions.append(a)

        state = next_state

        if terminated or truncated:
            break

    return states, actions, rewards


def mc_control_on_policy(env, num_episodes, gamma, eps, max_steps, first_visit=True):
    Q = np.zeros((env.S, env.n_actions))
    N = np.zeros((env.S, env.n_actions), dtype=np.int64)

    for ep in range(num_episodes):
        G = 0
        states, actions, rewards = generate_episode_sar(env, Q, eps, max_steps)

        if first_visit:
            returns = np.zeros(len(states))
            for t in reversed(range(len(rewards))):
                G = rewards[t] + gamma * G
                returns[t] = G

            visited = set()

            for t in range(len(states)):
                s = states[t]
                a = actions[t]
                G_t = returns[t]

                r, c = env.decode(s)
                if s == env.goal_id or ((r, c) in getattr(env, "obstacles", set())):
                    continue

                if (s, a) not in visited:
                    visited.add((s, a))

                    N[s, a] += 1
                    Q[s, a] += (G_t - Q[s, a]) / N[s, a]
        else:
            for t in reversed(range(len(rewards))):
                G = rewards[t] + gamma * G
                s = states[t]
                a = actions[t]

                r, c = env.decode(s)
                if s == env.goal_id or ((r, c) in getattr(env, "obstacles", set())):
                    continue
                N[s, a] += 1
                Q[s, a] += (G - Q[s, a]) / N[s, a]

    return Q
