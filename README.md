# Model-Based Offline Reinforcement Learning with GAN

- Formula of Key Idea
$${\underset{\pi} \max \ \mathbb{E}[\ \underset{t=0}{\overset{T-1}{\Sigma}} \gamma^t \widehat{R}(s_t, a_t, s_{t+1})+\gamma^TV^\pi(s_T) \ \vert \ s_{t+1}\ \sim \widehat{P}(\ \cdot \ \vert s_t,a_t),\ a_t\ \sim \pi(\ \cdot \ \vert s_t)\ ] \qquad s.t.\ \ \mathbb{E}_{\pi}[log(D(s_t))] \le log(d)}$$

    can be converted to (Largrange Dual Problem)

$${\underset{\lambda} \max \underset{\pi} \min \ \mathbb{E}[\ \underset{t=0}{\overset{T-1}{\Sigma}} -\gamma^t \widehat{R}(s_t, a_t, s_{t+1})-\gamma^TV^\pi(s_T) \ \vert \ s_{t+1}\ \sim \widehat{P}(\ \cdot \ \vert s_t,a_t),\ a_t\ \sim \pi(\ \cdot \ \vert s_t)\ ]\ +\ \lambda(\mathbb{E}_{\pi}[log(D(s_t))]-log(d)) \qquad s.t.\ \ \lambda \ge 0}$$

    where D is a Discriminator Network (Policy as a Generator), and d is a constant at about 0.4~0.7

- Result(Score) for D4RL halfcheetah-medium
!(./MB_OfflineRL_GAN.png)

- Description Video (Korean) Link : https://youtu.be/5AcFcyXfCNg
