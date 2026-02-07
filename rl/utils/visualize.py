import numpy as np

ARROWS = {0: "^", 1: ">", 2: "v", 3: "<"}


def print_policy_arrows(env, policy):
    grid = np.empty((env.H, env.W), dtype=object)
    print(" ")
    print("S: 시작, G: 목표, #: 장애물")
    for s in range(env.S):
        r, c = env.decode(s)

        if hasattr(env, "obstacles") and (r, c) in env.obstacles:
            grid[r, c] = "#"
        elif s == env.start_id:
            grid[r, c] = "S"
        elif s == env.goal_id:
            grid[r, c] = "G"
        else:
            a = int(np.argmax(policy[s]))
            grid[r, c] = ARROWS[a]

    for r in range(env.H):
        print(" ".join(grid[r]))
