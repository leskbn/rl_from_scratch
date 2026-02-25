import numpy as np


# TD(0) Prediction
# 에피소드 진행하면서 매 스텝마다 V를 업데이트 (bootstrap)
# V(s) <- V(s) + α * δ_t
# δ_t = r_{t+1} + γ V(s') - V(s)   (TD error)
#
# MC와 달리 에피소드 끝까지 기다리지 않고 바로 업데이트
# DP와 달리 transition 모델 없이 샘플로만 학습
# => model-free + online update
def td0(env, policy, num_episodes, alpha, gamma):
    """policy: shape (S,A), return V: shape (S,)"""
    V = np.zeros(env.S)

    for ep in range(num_episodes):
        done = False
        state = env.reset()

        while not done:
            a = np.random.choice(env.n_actions, p=policy[state])
            next_state, reward, terminated, truncated, _ = env.step(a)

            # delta_t = r + γV(s') - V(s)
            # terminated이면 terminal state이므로 미래 가치 0
            V[state] = V[state] + alpha * (
                reward + gamma * V[next_state] * (0.0 if terminated else 1.0) - V[state]
            )

            state = next_state

            if truncated or terminated:
                done = True

    return V


# TD(λ) Forward View (Prediction)
# n-step return G^(n)_t 를 λ로 가중평균한 λ-return G^λ_t 로 V를 업데이트
#
# G^(n)_t = r_{t+1} + γr_{t+2} + ... + γ^n V(s_{t+n})  (n-step return)
# G^λ_t  = (1-λ) Σ_{n=1}^∞ λ^{n-1} G^(n)_t            (λ-return): 1스텝 리턴부터 무한 스텝(실제로는 에피길이만큼의) 리턴들을 가중 평균낸 것
#                                                                  어떤 n이 최적인지 모르니까 다 써버리자..
# λ=0 이면 TD(0), λ=1 이면 MC와 동일
#
# [재귀식 유도]
# G^λ_t = (1-λ)G^(1)_t + λ(1-λ)G^(2)_t + λ²(1-λ)G^(3)_t + ...
#       = (1-λ)G^(1)_t + λ[(1-λ)G^(2)_t + λ(1-λ)G^(3)_t + ...]
#                              ↑ 이 괄호 안은 t+1에서 시작하는 λ-return = G^λ_{t+1}
#       = (1-λ)G^(1)_t + λ G^λ_{t+1}
#       = (1-λ)(r_{t+1} + γV(s_{t+1})) + λ G^λ_{t+1}   (G^(1)_t 대입)
#
# 재귀식으로 압축:
# G^λ_t = r_{t+1} + γ[(1-λ)V(s_{t+1}) + λ G^λ_{t+1}]
# => G^λ_t 계산에 G^λ_{t+1}만 필요 => 뒤에서부터 한 번만 훑으면 계산 가능
#
# 에피소드가 끝나야 업데이트 가능 (offline) => Backward view로 online 구현 가능
def td_lambda_forward(env, policy, num_episodes, alpha, gamma, lam):
    """policy: shape (S,A), lam: λ∈[0,1], return V: shape (S,)"""
    V = np.zeros(env.S)
    last_terminated = False

    for ep in range(num_episodes):
        state = env.reset()
        done = False
        states = []
        rewards = []
        last_terminated = False

        # 에피소드 끝까지 진행하며 states, rewards 수집
        while not done:
            a = np.random.choice(env.n_actions, p=policy[state])
            next_state, reward, terminated, truncated, _ = env.step(a)

            states.append(state)
            rewards.append(reward)

            state = next_state

            if truncated or terminated:
                states.append(state)  # 마지막 next_state 저장 (states[t+1] 접근용)
                last_terminated = terminated
                done = True

        # terminal이면 G^λ_T = 0, truncated이면 V(s_T)로 bootstrap
        G_lambda = 0.0 if last_terminated else V[states[-1]]

        # 뒤에서부터 G^λ_t 재귀 계산 및 V 업데이트
        # G^λ_t = r_{t+1} + γ[(1-λ)V(s_{t+1}) + λ G^λ_{t+1}]
        for t in reversed(range(len(rewards))):
            G_lambda = rewards[t] + gamma * (
                (1 - lam) * V[states[t + 1]] + lam * G_lambda
            )

            # V(s_t) <- V(s_t) + α(G^λ_t - V(s_t))
            V[states[t]] = V[states[t]] + alpha * (G_lambda - V[states[t]])

    return V


# TD(λ) Backward View / Eligibility Traces (Prediction)
# Forward view와 수학적으로 동일하지만 매 스텝 online 업데이트 가능
#
# TD error δ_t 를 최근에 방문한 상태들에게 소급 적용
# eligibility trace e(s): 각 상태가 얼마나 최근에/자주 방문됐는지 기록
#
# 매 스텝:
#   δ_t = r_{t+1} + γV(s') - V(s)      (TD error, TD(0)과 동일)
#   e(s_t) <- e(s_t) + 1                (방문한 상태 trace 증가)
#   V(s)   <- V(s) + α δ_t e(s)  ∀s    (모든 상태를 trace 비율로 업데이트)
#   e(s)   <- γλ · e(s)           ∀s    (시간이 지날수록 trace decay)
#
# λ=0 이면 e(s)가 즉시 decay => 현재 상태만 업데이트 => TD(0)
# λ=1 이면 오래된 상태도 오래 기억 => MC에 가까워짐
def td_lambda_backward(env, policy, num_episodes, alpha, gamma, lam):
    """policy: shape (S,A), lam: λ∈[0,1], return V: shape (S,)"""
    V = np.zeros(env.S)

    for ep in range(num_episodes):
        state = env.reset()
        e = np.zeros(env.S)  # eligibility trace, 에피소드마다 초기화
        done = False

        while not done:
            a = np.random.choice(env.n_actions, p=policy[state])
            next_state, reward, terminated, truncated, _ = env.step(a)

            # delta_t = r + γV(s') - V(s)
            delta_t = (
                reward + gamma * V[next_state] * (0.0 if terminated else 1.0) - V[state]
            )

            e[state] += 1  # 방문한 상태 trace 증가
            V += alpha * delta_t * e  # 모든 상태 업데이트 (trace 비율로)
            e *= gamma * lam  # 모든 상태 trace decay

            state = next_state

            if truncated or terminated:
                done = True

    return V
