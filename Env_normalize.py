import numpy as np
import pandas as pd


PV = []
E_load = []
H_load = []
C_load = []

PV1 = pd.read_excel("数据\\智能体1\\1-data_PV.xlsx")
E_Load1 = pd.read_excel("数据\\智能体1\\1-data_E.xlsx")
H_Load1 = pd.read_excel("数据\\智能体1\\1-data_H.xlsx")
C_Load1 = pd.read_excel("数据\\智能体1\\1-data_C.xlsx")

PV2 = pd.read_excel("数据\\智能体2\\2-data_PV.xlsx")
E_Load2 = pd.read_excel("数据\\智能体2\\2-data_E.xlsx")
H_Load2 = pd.read_excel("数据\\智能体2\\2-data_H.xlsx")
C_Load2 = pd.read_excel("数据\\智能体2\\2-data_C.xlsx")

PV3 = pd.read_excel("数据\\智能体3\\3-data_PV.xlsx")
E_Load3 = pd.read_excel("数据\\智能体3\\3-data_E.xlsx")
H_Load3 = pd.read_excel("数据\\智能体3\\3-data_H.xlsx")
C_Load3 = pd.read_excel("数据\\智能体3\\3-data_C.xlsx")

PV4 = pd.read_excel("数据\\智能体4\\4-data_PV.xlsx")
E_Load4 = pd.read_excel("数据\\智能体4\\4-data_E.xlsx")
H_Load4 = pd.read_excel("数据\\智能体4\\4-data_H.xlsx")
C_Load4 = pd.read_excel("数据\\智能体4\\4-data_C.xlsx")


PV1  = PV1.T*1.000
E_Load1 = E_Load1.T*0.001
H_Load1 = H_Load1.T*0.001
C_Load1 = C_Load1.T*0.001

PV2    = PV2.T*1.000
E_Load2 = E_Load2.T*0.001
H_Load2 = H_Load2.T*0.001
C_Load2 = C_Load2.T*0.001

PV3    = PV3.T*1.000
E_Load3 = E_Load3.T*0.001
H_Load3 = H_Load3.T*0.001
C_Load3 = C_Load3.T*0.001

PV4    = PV4.T*1.000
E_Load4 = E_Load4.T*0.001
H_Load4 = H_Load4.T*0.001
C_Load4 = C_Load4.T*0.001


PV += [PV1, PV2, PV3, PV4]
E_load += [E_Load1, E_Load2, E_Load3, E_Load4]
H_load += [H_Load1, H_Load2, H_Load3, H_Load4]
C_load += [C_Load1, C_Load2, C_Load3, C_Load4]

invalid_periods_E = [{"start": 9, "end": 23, "filter": "random"}]
invalid_periods_G = [{"start": 0, "end": 9, "filter": "random"}]
action_bound = np.array(1.0)

class EnergySystemEnv:
    def __init__(self, num_agents, rounds, T, PV, E_load, H_load, C_load, params):
        self.num_agents = num_agents
        self.T = T
        self.rounds = rounds
        self.PV = PV
        self.E_load = E_load
        self.H_load = H_load
        self.C_load = C_load
        self.params = params
        self.current_step = 0
        self.round = 0
        self.P_e_sum = 0
        self.previous_states = None  # 存储前一步的状态
        self.record_rounds = 4000
        # self.P_trade= []
        # 确保参数中的所有限制都是 NumPy 数组
        self.params['P_EG_lower'] = np.array(self.params['P_EG_lower'])
        self.params['P_EG_upper'] = np.array(self.params['P_EG_upper'])
        self.params['P_CCHP_lower'] = np.array(self.params['P_CCHP_lower'])
        self.params['P_CCHP_upper'] = np.array(self.params['P_CCHP_upper'])
        self.params['P_EB_lower'] = np.array(self.params['P_EB_lower'])
        self.params['P_EB_upper'] = np.array(self.params['P_EB_upper'])
        self.params['ESS_scope'] = np.array(self.params['ESS_scope'])
        self.params['CCHP_EC_ratio'] = np.array(self.params['CCHP_EC_ratio'])
        self.params['CCHP_EH_ratio'] = np.array(self.params['CCHP_EH_ratio'])

        self.data = {agent_index: [] for agent_index in range(self.num_agents)}
        self.rewards_data = {agent_index: [] for agent_index in range(self.num_agents)}
        self.max_rewards = -np.inf
        self.all_rewards = 0
        self.current_rewards = np.zeros(self.num_agents)  # 当前回合的奖励总值
        self.power_data = {agent_index: [] for agent_index in range(self.num_agents)}  # 存储每个时间步的功率数据

        # 初始化设备的上下限约束、爬坡约束等
        self.initialize_system_state()
        # 初始化电价等级信息
        self.price_levels = [self.get_price_level(hour) for hour in range(T)] # Get price level based on current hour

    def initialize_system_state(self):
        self.SoC = np.full(self.num_agents, self.params['SoC_initial'])
        self.P_EG = np.array(self.params['P_EG_best'])
        self.P_CCHP = np.array([3.0, 3.0, 3.0, 3.0])
        self.P_EC = np.array([3.0, 3.0, 3.0, 3.0])
        self.P_EB = np.array([9.0, 9.0, 9.0, 9.0])
        self.ESS = np.zeros(self.num_agents)
        self.G_CCHP = np.zeros(self.num_agents)
        self.H_CCHP = np.zeros(self.num_agents)
        self.C_CCHP = np.zeros(self.num_agents)
        self.H_EB = np.zeros(self.num_agents)
        self.C_EC = np.zeros(self.num_agents)
        self.P_e = np.zeros(self.num_agents)




    def normalize_state(self, state, lower_bounds, upper_bounds):
        # Min-Max scaling to [0, 1]
        normalized_state = (state - lower_bounds) / (upper_bounds - lower_bounds)
        return normalized_state

    def get_states(self):
        # self.round = 5

        states = []
        for i in range(self.num_agents):
            # 获取当前时间步及未来4个时间步的电价等级信息
            future_prices = [self.price_levels[(self.current_step + j) % self.T] for j in range(5)]
            raw_state = [
                self.SoC[i],
                self.PV[i][self.round][self.current_step],
                self.E_load[i][self.round][self.current_step],
                self.H_load[i][self.round][self.current_step],
                self.C_load[i][self.round][self.current_step],
                self.P_EG[i],
                self.P_CCHP[i],
                self.P_EB[i],
                self.P_EC[i],
                self.current_step / self.T  # 将当前时间步归一化添加到状态中
            ]+ future_prices  # 添加未来电价信息
            # Define lower and upper bounds for each state component
            lower_bounds = [
                self.params['SoC_scope'][0],
                min(self.PV[i].min()),
                min(self.E_load[i].min()),
                min(self.H_load[i].min()),
                min(self.C_load[i].min()),
                self.params['P_EG_upper'][i],
                self.params['P_CCHP_upper'][i],
                self.params['P_EB_upper'][i],
                self.params['P_EC_upper'][i],
                0
            ] + [0] * 5  # 电价等级的最小值为0
            upper_bounds = [
                self.params['SoC_scope'][1],
                max(self.PV[i].max()),
                max(self.E_load[i].max()),
                max(self.H_load[i].max()),
                max(self.C_load[i].max()),
                self.params['P_EG_lower'][i],
                self.params['P_CCHP_lower'][i],
                self.params['P_EB_lower'][i],
                self.params['P_EC_lower'][i],
                1
            ]+ [2] * 5  # 电价等级的最大值为2
            normalized_state = self.normalize_state(np.array(raw_state), np.array(lower_bounds), np.array(upper_bounds))
            states.append(normalized_state)
        return states

    def get_price_level(self, hour):
        if hour >= 8 and hour < 12:
            return 0
        elif hour >= 0 and hour < 8:
            return 2
        elif hour >= 17 and hour < 21:
            return 0
        else:
            return 1

    def step(self, actions):
        actions = np.array(actions)
        actions = actions.reshape(4, 3)
        actual_actions = self.denormalize_actions(actions)

        rewards = np.zeros(self.num_agents)
        next_states = []
        penalties = np.zeros(self.num_agents)  # 约束违反的惩罚
        valid = np.ones(self.num_agents, dtype=bool)

        if self.current_step != 0:
            self.previous_states = {  # 存储当前步的状态以便下一步使用
                'P_EG': np.copy(self.P_EG),
                'P_CCHP': np.copy(self.P_CCHP),
                'P_EB': np.copy(self.P_EB),
                'P_EC': np.copy(self.P_EC),
                # 'H_AC': np.copy(self.H_AC),
            }
        else:
            self.previous_states = False

        next_states = self.update_system_state(actual_actions)
        # next_states.append(next_state)

        # 计算奖励与惩罚
        penalties, valid = self.check_constraints_and_calculate_penalties()
        rewards = self.calculate_rewards(penalties, valid)

        self.current_rewards += rewards  # 累积当前回合的奖励

        # 存储当前时间步的功率数据
        for agent_index in range(self.num_agents):
            self.power_data[agent_index].append({
                'Time step': self.current_step,
                'P_CCHP': self.P_CCHP[agent_index],
                'G_CCHP': self.G_CCHP[agent_index],
                'H_CCHP': self.H_CCHP[agent_index],
                'C_CCHP': self.C_CCHP[agent_index],
                'P_EG': self.P_EG[agent_index],
                'P_EB': self.P_EB[agent_index],
                'H_EB': self.H_EB[agent_index],
                'P_EC': self.P_EC[agent_index],
                'C_EC': self.C_EC[agent_index],
                'ESS': self.ESS[agent_index],
                'P_trade': self.P_e[agent_index],
                'P_IES_b': self.power_buy_IES[agent_index],
                'P_IES_s': self.power_sell_IES[agent_index],
                'P_grid_b': self.power_buy_grid[agent_index],
                'P_grid_s': self.power_sell_grid[agent_index],
                'SoC': self.SoC[agent_index],
                'E_load':self.E_load[agent_index][self.round][self.current_step],
                'H_load': self.H_load[agent_index][self.round][self.current_step],
                'C_load': self.C_load[agent_index][self.round][self.current_step],
                'PV': self.PV[agent_index][self.round][self.current_step],
            })

        self.current_step += 1
        # print('\n')
        done = self.current_step >= self.T
        if done:
            self.record_rewards_and_power()
            # 打印当前回合数和这一回合的奖励
            print(f"Round {self.round} completed. Rewards: {self.current_rewards}")
            self.round = (self.round+1) % 1000 + 1
            # self.round = 5
            self.current_rewards = np.zeros(self.num_agents)  # 当前回合的奖励总值
            next_states = self.reset()
        else:
            next_states = self.get_states()
        return next_states, rewards, done

    # 反正则化动作
    def denormalize_actions(self, actions):
        normalized_actions = actions
        actual_actions = np.zeros_like(normalized_actions)
        # 根据参数配置反正规化动作
        for i in range(self.num_agents):
            # actual_actions[i, 0] = (normalized_actions[i, 0] + 1) / 2 * (
            #             self.params['P_EG_upper'][i] - self.params['P_EG_lower'][i]) + self.params['P_EG_lower'][i]
            # actual_actions[i, 1] = (normalized_actions[i, 1] + 1) / 2 * (
            #             self.params['P_CCHP_upper'][i] - self.params['P_CCHP_lower'][i]) + self.params['P_CCHP_lower'][i]
            # # actual_actions[i, 2] = (normalized_actions[i, 2] + 1) / 2 * (
            # #             self.params['P_EB_upper'][i] - self.params['P_EB_lower'][i]) + self.params['P_EB_lower'][i]
            # actual_actions[i, 2] = (normalized_actions[i, 2] + 1) / 2 * (
            #             self.params['ESS_scope'][1] - self.params['ESS_scope'][0]) + self.params['ESS_scope'][0]

            actual_actions[i, 0] = (normalized_actions[i, 0] + 1) / 2 * (4) -2
            actual_actions[i, 1] = (normalized_actions[i, 1] + 1) / 2 * (20) -10
            # actual_actions[i, 2] = (normalized_actions[i, 2] + 1) / 2 * (
            #             self.params['P_EB_upper'][i] - self.params['P_EB_lower'][i]) + self.params['P_EB_lower'][i]
            actual_actions[i, 2] = (normalized_actions[i, 2] + 1) / 2 * (
                        self.params['ESS_scope'][1] - self.params['ESS_scope'][0]) + self.params['ESS_scope'][0]
        return actual_actions

    def update_system_state(self, actions):
        # 应用动作并使用 clip 以确保不超出范围
        for agent_index in range(self.num_agents):
            self.P_EG[agent_index] = np.clip(self.P_EG[agent_index] + actions[agent_index][0], self.params['P_EG_lower'][agent_index], self.params['P_EG_upper'][agent_index])
            self.P_CCHP[agent_index] = np.clip(self.P_CCHP[agent_index] + actions[agent_index][1], self.params['P_CCHP_lower'][agent_index], self.params['P_CCHP_upper'][agent_index])
            self.ESS[agent_index] = np.clip(actions[agent_index][2], -self.params['climb_constrain_ess'][agent_index], self.params['climb_constrain_ess'][agent_index])
            # 根据动作和等式约束更新系统状态
            self.H_CCHP[agent_index] = self.P_CCHP[agent_index] * self.params['CCHP_EH_ratio'][agent_index]
            self.C_CCHP[agent_index] = self.P_CCHP[agent_index] * self.params['CCHP_EC_ratio'][agent_index]
            self.G_CCHP[agent_index] = self.P_CCHP[agent_index] / 0.3
            self.H_EB[agent_index] = self.H_load[agent_index][self.round][self.current_step] - self.H_CCHP[agent_index]
            self.C_EC[agent_index] = self.C_load[agent_index][self.round][self.current_step] - self.C_CCHP[agent_index]
            self.P_EB[agent_index] = self.H_EB[agent_index] / 0.90
            self.P_EC[agent_index] = self.C_EC[agent_index] / 0.92

            # 根据储能设备的操作更新SoC
            if self.ESS[agent_index] < 0:
                self.SoC[agent_index] -= self.ESS[agent_index] * self.params['Epsilon_cha'] / self.params['ESS_cap'][agent_index]
            else:  # 充电
                self.SoC[agent_index] -= self.ESS[agent_index] / (self.params['Epsilon_dis'] * self.params['ESS_cap'][agent_index])

            # 计算电网购买或销售的电量
            self.P_e[agent_index] = (self.E_load[agent_index][self.round][self.current_step] - self.PV[agent_index][self.round][self.current_step]
                           - self.P_CCHP[agent_index] - self.P_EG[agent_index] - self.ESS[agent_index] + self.P_EB[agent_index] + self.P_EC[agent_index])
        # self.P_trade.append(self.P_trade)
        return []

    def Cluster_energy_coordination_center(self):
        C_grid = 0
        C_grid1 = 0
        C_grid2 = 0
        Sum_P_buy = 0
        Sum_P_sell = 0
        Price_IES_b = 0
        Price_IES_s = 0
        P_buy = {}
        P_sell = {}
        # 初始化每个区域的交易成本和收益
        self.power_buy_IES = {i: 0 for i in range(len(self.P_e))}
        self.power_buy_grid = {i: 0 for i in range(len(self.P_e))}
        self.power_sell_IES = {i: 0 for i in range(len(self.P_e))}
        self.power_sell_grid = {i: 0 for i in range(len(self.P_e))}
        cost_from_IES = {i: 0 for i in range(len(self.P_e))}
        cost_from_grid = {i: 0 for i in range(len(self.P_e))}
        revenue_to_IES = {i: 0 for i in range(len(self.P_e))}
        revenue_to_grid = {i: 0 for i in range(len(self.P_e))}

        price_level = self.get_price_level(self.current_step)
        for index, value in enumerate(self.P_e):
            if value >= 0:
                C_grid1 += self.params['Price_buy'][price_level] * abs(value) # Buying price
                P_buy[index] = value
            else:
                C_grid1 -= self.params['Price_sold'][price_level] * abs(value) # Selling price
                P_sell[index] = -value
        if sum(self.P_e)>=0:
            C_grid2 = self.params['Price_buy'][price_level] * sum(self.P_e)
        else:
            C_grid2 = self.params['Price_sold'][price_level] * (sum(self.P_e))
        C_grid = C_grid1 - C_grid2
        Sum_P_buy = sum(P_buy.values())
        Sum_P_sell = sum(P_sell.values())
        if Sum_P_buy != 0:
            Price_IES_b = (self.params['Price_buy'][price_level] * Sum_P_buy - 0.5 * C_grid)/Sum_P_buy
        else:
            Price_IES_b = 0
        if Sum_P_sell != 0:
            Price_IES_s = (self.params['Price_sold'][price_level] * Sum_P_sell - 0.5 * C_grid)/Sum_P_sell
        else:
            Price_IES_s = 0
        # 买方与卖方之间进行交易
        for buyer in list(P_buy.keys()):
            for seller in list(P_sell.keys()):
                if P_buy[buyer] == 0:
                    break
                if P_sell[seller] == 0:
                    continue

                # 买方购买的功率不能超过卖方能提供的功率
                transaction_power = min(P_buy[buyer], P_sell[seller])

                # 交易完成后，更新买方和卖方的需求
                P_buy[buyer] -= transaction_power
                P_sell[seller] -= transaction_power
                self.power_buy_IES[buyer] += transaction_power
                self.power_sell_IES[seller] += transaction_power

                # 计算交易成本和收益
                cost = transaction_power * Price_IES_b
                revenue = transaction_power * Price_IES_s
                cost_from_IES[buyer] += cost
                revenue_to_IES[seller] += revenue

        # 处理买方剩余的需求，从主电网购买
        for buyer, power_needed in P_buy.items():
            if power_needed > 0:
                cost = power_needed * self.params['Price_buy'][price_level]
                self.power_buy_grid[buyer] += power_needed
                cost_from_grid[buyer] += cost
                # print(f"区域 {buyer} 从主电网购买了 {power_needed} 单位的功率")

        # 处理卖方剩余的电量，卖给主电网
        for seller, power_surplus in P_sell.items():
            if power_surplus > 0:
                revenue = power_surplus * self.params['Price_sold'][price_level]
                self.power_sell_grid[seller] += power_surplus
                revenue_to_grid[seller] += revenue
                # print(f"区域 {seller} 向主电网出售了 {power_surplus} 单位的功率")
        return list(cost_from_IES.values()),list(revenue_to_IES.values()),list(cost_from_grid.values()),list(revenue_to_grid.values())

    def check_constraints_and_calculate_penalties(self):
        penalties = np.zeros(self.num_agents)
        valid = np.ones(self.num_agents, dtype=bool)
        total_violations = []  # 保存所有违规信息

        # Check constraints for CCHP
        for agent_index in range(self.num_agents):
            violations = []
            penalty_amount = 0

            # Function to calculate penalty for each violation
            def calculate_penalty(lower_bound, upper_bound, actual_value):
                if actual_value < lower_bound:
                    return lower_bound - actual_value
                elif actual_value > upper_bound:
                    return actual_value - upper_bound
                else:
                    return 0

            # Check CCHP constraints
            penalty = calculate_penalty(self.params['P_CCHP_lower'][agent_index],
                                        self.params['P_CCHP_upper'][agent_index], self.P_CCHP[agent_index])
            if penalty > 0:
                penalty_amount += penalty
                violations.append(f"Agent {agent_index} - CCHP generation out of bounds by {penalty}")

            # Check constraints for each of the three EG units
            penalty = calculate_penalty(self.params['P_EG_lower'][agent_index],
                                        self.params['P_EG_upper'][agent_index], self.P_EG[agent_index])
            if penalty > 0:
                penalty_amount += penalty
                violations.append(f"Agent {agent_index} - EG unit power out of bounds by {penalty}")

            # Check constraints for EB, EC, and AC
            penalty = calculate_penalty(self.params['P_EB_lower'][agent_index], self.params['P_EB_upper'][agent_index],
                                        self.P_EB[agent_index])
            if penalty > 0:
                penalty_amount += penalty
                violations.append(f"Agent {agent_index} - EB power out of bounds by {penalty}")

            penalty = calculate_penalty(self.params['P_EC_lower'][agent_index], self.params['P_EC_upper'][agent_index],
                                        self.P_EC[agent_index])
            if penalty > 0:
                penalty_amount += penalty
                violations.append(f"Agent {agent_index} - EC power out of bounds by {penalty}")

            # penalty = calculate_penalty(-self.params['P_e_max'], self.params['P_e_max'],
            #                             self.P_e[agent_index])
            # if penalty > 0:
            #     penalty_amount += penalty
            #     violations.append(f"Agent {agent_index} - P_e power out of bounds by {penalty}")

            # penalty = calculate_penalty(self.params['H_AC_lower'][agent_index], self.params['H_AC_upper'][agent_index],
            #                             self.H_AC[agent_index])
            # if penalty > 0:
            #     penalty_amount += penalty
            #     violations.append(f"Agent {agent_index} - AC heat output out of bounds by {penalty}")

            # Check SoC and ESS constraints
            penalty = calculate_penalty(self.params['SoC_scope'][0], self.params['SoC_scope'][1], self.SoC[agent_index])
            if penalty > 0:
                penalty_amount += penalty
                violations.append(f"Agent {agent_index} - SoC out of bounds by {penalty}")
                # self.SoC[agent_index] = np.clip(self.SoC[agent_index], self.params['SoC_scope'][0], self.params['SoC_scope'][1])

            penalty = calculate_penalty(-self.params['climb_constrain_ess'][agent_index],
                                        self.params['climb_constrain_ess'][agent_index], self.ESS[agent_index])
            if penalty > 0:
                penalty_amount += penalty
                violations.append(f"Agent {agent_index} - ESS out of bounds by {penalty}")

            # 检查爬坡约束，仅当存在前一步状态时进行
            if self.previous_states:
                # 例如，对于CCHP设备的爬坡约束
                delta_P_CCHP = self.P_CCHP[agent_index] - self.previous_states['P_CCHP'][agent_index]
                if not (self.params['delta_P_CCHP_min'] <= delta_P_CCHP <= self.params['delta_P_CCHP_max']):
                    penalty = self.params['ramping_penalty'] * abs(delta_P_CCHP)
                    penalties[agent_index] += penalty
                    violations.append(f"Agent {agent_index} - CCHP ramp violation by {delta_P_CCHP} exceeding limits")
                # EG爬坡约束
                delta_P_EG = self.P_EG[agent_index] - self.previous_states['P_EG'][agent_index]
                if not (self.params['delta_P_EG_min'][agent_index] <= delta_P_EG <= self.params['delta_P_EG_max'][agent_index] ):
                    penalty = self.params['ramping_penalty'] * abs(delta_P_EG)
                    penalties[agent_index] += penalty
                    violations.append(f"Agent {agent_index} - EG unit ramp violation by {delta_P_EG}")

                # EB爬坡约束
                delta_P_EB = self.P_EB[agent_index] - self.previous_states['P_EB'][agent_index]
                if not (self.params['delta_P_EB_min'] <= delta_P_EB <= self.params['delta_P_EB_max']):
                    penalty = self.params['ramping_penalty'] * abs(delta_P_EB)
                    penalties[agent_index] += penalty
                    violations.append(f"Agent {agent_index} - EB ramp violation by {delta_P_EB}")

                # EC爬坡约束
                delta_P_EC = self.P_EC[agent_index] - self.previous_states['P_EC'][agent_index]
                if not (self.params['delta_P_EC_min'] <= delta_P_EC <= self.params['delta_P_EC_max']):
                    penalty = self.params['ramping_penalty'] * abs(delta_P_EC)
                    penalties[agent_index] += penalty
                    violations.append(f"Agent {agent_index} - EC ramp violation by {delta_P_EC}")


            # Calculate a large penalty if any constraints are violated
            if penalty_amount > 1:
                valid[agent_index] = False
                penalties[agent_index] = -100000 * penalty_amount  # Large negative penalty 10w,20w
                total_violations.extend(violations)  # 将本智能体的违规信息加入总列表
        # if  valid.all() == True:
        #     print(f"{self.round}:Violations for agents at step {self.current_step}-No violation")
        # # 循环结束后输出
        # print(f"{self.round}:Violations for agents at step {self.current_step}:")
        # print(f"Total number of violations across all agents: {len(total_violations)}")
        # for violation in total_violations:
        #     print(violation)
        return penalties, valid

    def calculate_rewards(self, penalties, valid):
        cost_IES = []
        revenue_IES = []
        cost_grid = []
        revenue_grid = []
        cost_IES,revenue_IES,cost_grid,revenue_grid = self.Cluster_energy_coordination_center()
        # 根据成本计算奖励
        rewards = np.zeros(self.num_agents)
        reward1 = np.zeros(self.num_agents)
        reward2 = np.zeros(self.num_agents)
        reward3 = np.zeros(self.num_agents)
        reward4 = np.zeros(self.num_agents)
        # costs = np.zeros(self.num_agents)
        price_level = self.get_price_level(self.current_step)
        # penalties, valid = self.check_constraints_and_calculate_penalties(agent_index)

        for agent_index in range(self.num_agents):
            # Energy costs calculation based on updated states
            # costs[agent_index] -= 10000 * (abs(self.P_CCHP1[agent_index] - self.P_CCHP2[agent_index]))
            if self.P_e[agent_index] >= 0:
                reward1[agent_index] -= (cost_IES[agent_index] + cost_grid[agent_index])  # Buying price
            else:
                reward1[agent_index] += (revenue_IES[agent_index] + revenue_grid[agent_index]) # Selling price

            # Cost for using gas
            reward2[agent_index] -= self.params['Price_gas'] * self.G_CCHP[agent_index]

            # Cost for electricity generation

            reward3[agent_index] -= (self.params['P_params_alpha'][agent_index] * self.P_EG[agent_index] ** 2 +
                           self.params['P_params_beta'][agent_index] * self.P_EG[agent_index] +
                           self.params['P_params_gama'][agent_index])

            if valid[agent_index] == 0:
                rewards[agent_index] = reward1[agent_index] + reward2[agent_index] + reward3[agent_index] + penalties[agent_index]
            else:
                rewards[agent_index] = reward1[agent_index] + reward2[agent_index] + reward3[agent_index] 

        return rewards
        # 需要实现成本计算逻辑
        # return reward, penalty

    def reset(self):
        self.initialize_system_state()
        self.current_step = 0
        return self.get_states()

    def record_rewards_and_power(self):
        for agent_index in range(self.num_agents):
            # 记录奖励
            self.rewards_data[agent_index].append({
                'Round': self.round,
                'Reward': self.current_rewards[agent_index],
                'Sum_Reward': np.sum(self.current_rewards, axis=0)
            })
        self.all_rewards = sum(self.current_rewards)
            # 如果当前奖励是最大值，记录功率数据
            # if self.current_rewards[agent_index] > self.max_rewards[agent_index]:
            #     self.max_rewards[agent_index] = self.current_rewards[agent_index]
            #     self.data[agent_index] = self.power_data[agent_index]  # 更新为当前回合的功率数据
        if self.all_rewards > self.max_rewards:
            self.max_rewards = self.all_rewards
            for agent_index in range(self.num_agents):
                self.data[agent_index] = self.power_data[agent_index]

        power_writer = pd.ExcelWriter('power.xlsx')
        for agent_index in range(self.num_agents):
            power_df = pd.DataFrame(self.data[agent_index])
            power_df.to_excel(power_writer, sheet_name=f'Agent_{agent_index}_Max_Reward_Round', index=False)

        power_writer.save()

        reward_writer = pd.ExcelWriter('reward.xlsx')
        for agent_index in range(self.num_agents):
            reward_df = pd.DataFrame(self.rewards_data[agent_index])
            reward_df.to_excel(reward_writer, sheet_name=f'Agent_{agent_index}_Rewards', index=False)

        reward_writer.save()
        self.power_data = {agent_index: [] for agent_index in range(self.num_agents)}


# 参数初始化部分
params = {
    'SoC_initial': 0.5,
    'P_EG_best': [13, 12.5, 14, 12, 13],
    'P_EG_upper': [16, 16.5,17,15, 16],
    'P_EG_lower': [8, 7, 7.8, 8, 7.8],
    'P_CCHP_upper': [12, 11, 11.5, 11.4, 14],
    'P_CCHP_lower': [0, 0, 0, 0, 0],
    'CCHP_EC_ratio':[0.9, 0.85, 1, 1.1, 0.9],
    'CCHP_EH_ratio':[1.2, 1.1, 1, 0.86, 1.2],
    'P_EB_upper': [25, 25.5, 25.8, 24, 25.2],
    'P_EB_lower': [0, 0, 0, 0, 0],
    'P_EC_upper': [21,18,22,24,25],
    'P_EC_lower': [0, 0, 0, 0, 0],
    'H_AC_upper': [10,5,7.5,12.5,10.2],
    'H_AC_lower': [0, 0, 0, 0, 0],
    'P_e_max':28,
    'Price_buy': [149, 94, 43],
    'Price_sold': [85, 48, 28],
    'Price_gas': 100,
    'P_params_alpha': [0.0625, 0.025, 0.0175, 0.025, 0.0625],
    'P_params_beta': [1, 3, 1.75, 3, 1],
    'P_params_gama': [0, 0, 0, 0, 0],
    'Epsilon_dis': 0.95,
    'Epsilon_cha': 0.95,
    'ESS_cap': [10,10,10,10,10],
    'ESS_scope': [-0.5,0.5],
    'SoC_scope':[0.15,0.85],

    'climb_constrain_ess': [0.5, 0.5, 0.5, 0.5, 0.5],

    'delta_P_CCHP_min': -10, 'delta_P_CCHP_max': 10,  # 最小下降率， 最大爬坡率
    'delta_P_EG_min': [-2, -2, -2, -2, -2], 'delta_P_EG_max': [2, 2, 2, 2, 2],
    'delta_P_EB_min': -10, 'delta_P_EB_max': 10,
    'delta_P_EC_min': -10, 'delta_P_EC_max': 10,
    'delta_H_AC_min': -5, 'delta_H_AC_max': 5,
    'ramping_penalty': -100000,  # 违反爬坡约束的惩罚系数

    # 更多参数初始化
}
env = EnergySystemEnv(num_agents=4, rounds=3010, T=24, PV=PV, E_load=E_load, H_load=H_load,
                      C_load=C_load, params=params)
