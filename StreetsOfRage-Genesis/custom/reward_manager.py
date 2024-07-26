class HierarchicalReward:
    def __init__(self):
        self.wei_hea = 1  # Adjust these weights as needed
        self.wei_sco = 1
        self.wei_map = 1
        self.wei_dam = 1
        
        self.health = 0
        self.time = 0
        self.map = 0
        self.go = 0
        self.damage = 0
        self.score_temp = 0
        
        self.cumulative_reward = 0
        self.cumulative_health = 0
        self.cumulative_map = 0
        self.cumulative_damage = 0
        self.cumulative_score = 0

    def get_cumulative_reward(self):
        return self.cumulative_reward

    def get_cumulative_health(self):
        return self.cumulative_health

    def get_cumulative_map(self):
        return self.cumulative_map

    def get_cumulative_damage(self):
        return self.cumulative_damage

    def get_cumulative_score(self):
        return self.cumulative_score

    def calculate_reward(self, info, ac):
        state = self.process_state(info)
        policy = self.select_policy(state)
        
        if policy == "navigation":
            return self.navigation_reward(info, ac)
        else:  # combat
            return self.combat_reward(info, ac)

    def process_state(self, info):
        enemy_count = sum(1 for i in range(1, 8) if info[f'enemy{i}'] > 0)
        health_str = str(info['health0'])[:2]
        health_int = int(health_str)
        
        return {
            "enemy_count": enemy_count,
            "health": health_int,
            "map": info['map'],
            "go": info['go'],
            "damage": info['damage0'],
            "score": info['score'],
            "time": info['time']
        }

    def select_policy(self, state):
        if state["enemy_count"] == 0:
            return "navigation"
        else:
            return "combat"

    def navigation_reward(self, info, ac):
        rew = 0
        state = self.process_state(info)
        
        # MAP progress reward
        if self.map == 0:
            self.map = state["map"]
        if info['go'] == 0 and self.map > 2000:
            self.map = 800
        if 800 < state["map"] < 5000 and state["map"] > self.map:
            rew += self.wei_map
            self.cumulative_reward += self.wei_map
            self.cumulative_map += self.wei_map
            self.map = state["map"]
        
        # Time penalty
        if round(state["time"]) < round(self.time):
            rew -= self.wei_sco
            self.cumulative_reward -= self.wei_sco
        self.time = round(state["time"])
        
        return rew

    def combat_reward(self, info, ac):
        rew = 0
        state = self.process_state(info)
        
        # Health loss penalty
        health_diff = state["health"] - self.health
        if health_diff < 0:
            rew -= self.wei_hea
            self.cumulative_reward -= self.wei_hea
            self.cumulative_health -= self.wei_hea
        self.health = state["health"]
        
        # Damage dealt reward
        if self.damage != state["damage"] and state["damage"] > 0:
            rew += self.wei_dam
            self.cumulative_reward += self.wei_dam
            self.cumulative_damage += self.wei_dam
        self.damage = state["damage"]
        
        # Small penalty for enemies present
        rew -= 0.0001
        self.cumulative_reward -= 0.0001
        
        return rew

    def update_common_rewards(self, info):
        state = self.process_state(info)
        rew = 0
        
        # GO progress reward
        if state["go"] > self.go:
            rew += self.wei_map * 100
            self.cumulative_reward += self.wei_map * 100
            self.cumulative_map += self.wei_map * 100
            self.go = state["go"]
        
        # Score reward
        if state["score"] > self.score_temp:
            rew += self.wei_sco
            self.cumulative_reward += self.wei_sco
            self.cumulative_score += self.wei_sco
            self.score_temp = state["score"]

    def reset(self):
        self.health = 0
        self.time = 0
        self.map = 0
        self.go = 0
        self.damage = 0
        self.score_temp = 0

        self.cumulative_reward = 0
        self.cumulative_health = 0
        self.cumulative_map = 0
        self.cumulative_damage = 0
        self.cumulative_score = 0

'''def calculate_reward(self, info, ac):
    rew = 0

    enemy_count = sum(1 for i in range(1, 8) if info[f'enemy{i}'] > 0)
    health_str = str(info['health0'])[:2]
    health_int = int(health_str)
    if self.health == 0:
        self.health = health_int
    health_diff = health_int - self.health
    
    # Negative reward for losing health
    if health_diff < 0:
        rew -= self.wei_hea
        self.cumulative_reward -= self.wei_hea
        self.cumulative_health -= self.wei_hea
        self.health = health_int

    # State 1: No enemies
    if enemy_count == 0:
        if round(info['time']) < round(self.time):
            self.cumulative_reward -= self.wei_sco
        self.time = round(info['time'])
        
        # MAP
        if self.map == 0:
            self.map = info['map']
        if info['go'] == 0 and self.map > 2000:
            self.map = 800
        if 800 < info['map'] < 5000 and info['map'] > self.map:
            self.cumulative_reward += self.wei_map
            self.cumulative_map += self.wei_map
            self.map = info['map']

    # Progress GO
    if info['go'] > self.go:
        rew += self.wei_map * 100
        self.cumulative_reward += self.wei_map * 100
        self.cumulative_map += self.wei_map * 100
        self.go = info['go']

    # Combat
    if self.damage != info['damage0']:
        if info['damage0'] > 0 and info['damage'] != 0:
            rew += self.wei_dam
            self.cumulative_reward += self.wei_dam
            self.cumulative_damage += self.wei_dam
    self.damage = info['damage0']

    # Small penalty for enemies present
    rew -= 0.000001
    self.cumulative_reward -= 0.000001

    # Score
    if info['score'] > self.score_temp:
        self.cumulative_reward += self.wei_sco
        self.cumulative_score += self.wei_sco
        self.score_temp = info['score']

    return rew'''

