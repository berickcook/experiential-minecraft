import copy
import logging
import math
import heapq
from time import sleep, time
from tagilmo.utils.vereya_wrapper import MCConnector, RobustObserver
import tagilmo.utils.mission_builder as mb
import numpy as np
import uuid

from copy import deepcopy
from tagilmo.utils.mathutils import normAngle, degree2rad

class State:
    def __init__(self, pos_input, grid_input, act, prev, step):
        self.pos_input = pos_input.copy()
        self.grid_input = grid_input.copy()
        self.action = act
        self.prev_state = prev
        self.change_pos = []
        self.change_grid = []
        self.step = step
        self.applied_rules_pos = dict()
        self.applied_rule_ids_pos = dict()
        self.applied_rules_grid = dict()
        self.applied_rule_ids_grid = dict()
        self.compare = None
        self.confidence = None
        self.confidence_count = None
        self.confidence_total = None

class Airis:

    def __init__(self):
        self.knowledge = dict()
        self.grid_input = []
        self.pos_input = []
        self.inventory_input = dict()
        self.hotbar_input = dict()
        self.mission_input = []
        self.states = None
        self.action_plan = []
        self.last_action = None
        self.last_change_pos = set()
        self.last_change_grid = set()
        self.prev_last_change_pos = set()
        self.prev_last_change_grid = set()
        self.given_goal = None #([0, 65, 0, None, None],[])
        self.applied_rules_pos = dict()
        self.applied_rules_grid = dict()
        self.bad_predictions_pos = None
        self.bad_predictions_grid = None
        self.prev_applied_rules_pos = None
        self.prev_applied_rules_grid = None
        self.time_step = 0
        self.actions = ['move', 'mine', 'jump', 'turn 0', 'turn 45', 'turn 90', 'turn 135', 'turn 180', 'turn 225', 'turn 270' 'turn 315', 'look up', 'look ceiling', 'look down', 'look floor', 'look straight']

    def capture_input(self, pos_input, grid_input, action, state, pre):
        if pre:
            self.pos_input = np.asarray([math.floor(pos_input[0]), math.floor(pos_input[1]), math.floor(pos_input[2]), round(pos_input[3]), round(pos_input[4] % 360)])
            self.grid_input = np.asarray(grid_input)

            if not self.action_plan:
                self.states = [State(self.pos_input, self.grid_input, None, 0, 0)]
                self.states[0].change_pos = self.last_change_pos
                self.states[0].change_grid = self.last_change_grid

            if self.given_goal:
                if not self.action_plan:
                    self.make_plan()
                    if self.action_plan:
                        return self.action_plan.pop()
                else:
                    return self.action_plan.pop()
            else:
                new_state = self.predict(action, 0)
                self.states.append(new_state)
                return action, 1

        else:
            new_pos_input = np.asarray([math.floor(pos_input[0]), math.floor(pos_input[1]), math.floor(pos_input[2]), round(pos_input[3]), round(pos_input[4] % 360)])
            new_grid_input = np.asarray(grid_input)
            clear_plan = False
            self.last_action = action
            self.prev_applied_rules_pos = deepcopy(self.applied_rules_pos)
            self.prev_applied_rules_grid = deepcopy(self.applied_rules_grid)
            self.bad_predictions_pos = dict()
            self.bad_predictions_grid = dict()

            self.last_change_pos = set()
            self.last_change_grid = set()

            for i, d, in enumerate(self.pos_input):
                self.last_change_pos.add((i, self.pos_input[i], new_pos_input[i], new_pos_input[i] - self.pos_input[i]))

            for i, d, in enumerate(self.grid_input):
                self.last_change_grid.add((i, self.grid_input[i], new_grid_input[i]))

            pos_mismatch = [i for i, v in enumerate(self.states[state].pos_input) if v != new_pos_input[i]]
            grid_mismatch = [i for i, v in enumerate(self.states[state].grid_input) if v != new_grid_input[i]]

            self.applied_rules_pos = copy.deepcopy(self.states[state].applied_rules_pos)
            self.applied_rules_grid = copy.deepcopy(self.states[state].applied_rules_grid)

            if pos_mismatch:
                clear_plan = True
                for index in pos_mismatch:
                    try:
                        self.bad_predictions_pos[index] = deepcopy(self.applied_rules_pos[index])
                        del self.applied_rules_pos[index]
                    except KeyError:
                        pass

                for index in pos_mismatch:
                    self.create_rule(action, 'Pos', index, self.pos_input[index], new_pos_input[index])

            if grid_mismatch:
                clear_plan = True
                for index in grid_mismatch:
                    try:
                        self.bad_predictions_grid[index] = deepcopy(self.applied_rules_grid[index])
                        del self.applied_rules_grid[index]
                    except KeyError:
                        pass

                for index in grid_mismatch:
                    self.create_rule(action, 'Grid', index, self.grid_input[index], new_grid_input[index])

            for index in self.applied_rules_pos.keys():
                self.update_good_rule(self.applied_rules_pos[index])

            for index in self.applied_rules_grid.keys():
                self.update_good_rule(self.applied_rules_grid[index])

            for index in self.bad_predictions_pos.keys():
                self.update_bad_rule(self.bad_predictions_pos[index])

            for index in self.bad_predictions_grid.keys():
                self.update_bad_rule(self.bad_predictions_grid[index])

            print('Prediction Confidence: ', self.states[1].confidence)
            if clear_plan:
                print('Prediction Incorrect...')
                self.action_plan = []
            else:
                print('Prediction Correct!')

            self.prev_last_change_pos = deepcopy(self.last_change_pos)
            self.prev_last_change_grid = deepcopy(self.last_change_grid)

    def make_plan(self):
        current_state = 0
        state_heap = [(self.compare(0), 0)]
        goal_reached = False
        if state_heap[0][0] == 0:
            goal_reached = True

        while not goal_reached:
            for act in self.actions:
                try:
                    check = self.knowledge['Action Rules'][act]
                except KeyError:
                    self.states.append(self.predict(act, 0))
                    self.action_plan.append((act, 0))
                    goal_reached = True
                    break



    def compare(self, state):
        compare_total = 0
        # Compare current position to goal position
        for i, c_val in enumerate(self.given_goal[0]):
            if c_val is not None:
                compare_total += abs(c_val - self.states[state].pos_input[i])

        return compare_total

    def predict(self, act, base_state):
        confidence_count = 0
        confidence_total = 0
        predict_heap = dict()

        predict_state = State(self.states[base_state].pos_input, self.states[base_state].grid_input, act, base_state, self.states[base_state].step + 1)

        # Predict pos conditions and heapify the output to guide more efficient grid checking. TO DO Check the speed of multiprocessing pooling vs current linear

        # Evaluate pos input
        for idx, val in enumerate(predict_state.pos_input):
            predict_heap['Pos' + str(idx)] = []
            try:
                rules_list = self.knowledge['Pos-' + str(idx) + '/Actions'][act]
            except KeyError:
                predict_heap['Pos' + str(idx)] = None
                continue

            idx_heap = self.predict_pos_conditions('Pos', idx, val, rules_list, predict_state)
            print('Size of Pos idx_heap', len(idx_heap))

            while idx_heap:
                pos_data = heapq.heappop(idx_heap)
                diff_count = pos_data[0]
                rule = pos_data[1]
                updates = self.knowledge['Pos-' + str(idx) + '/' + str(idx) + '/' + str(rule) + '/Updates']
                new_val = self.knowledge['Pos-' + str(idx) + '/' + str(idx) + '/' + str(rule) + '/New Value']
                age = self.knowledge['Pos-' + str(idx) + '/' + str(idx) + '/' + str(rule) + '/Age']

                grid_data = self.predict_grid_conditions('Pos', idx, val, rule, predict_state)
                diff_count += grid_data[0]
                heapq.heappush(predict_heap['Pos' + str(idx)],(diff_count, rule, idx, new_val, 'Pos', pos_data[2] + grid_data[1], updates, age))
                if diff_count == 0:
                    break

        # Evaluate grid input
        for idx, val in enumerate(predict_state.grid_input):
            predict_heap['Grid' + str(idx)] = []
            try:
                rules_list = self.knowledge['Grid-' + str(idx) + '/' + str(val) + '/Actions'][act]
            except KeyError:
                predict_heap['Grid' + str(idx)] = None
                continue

            idx_heap = self.predict_pos_conditions('Grid', idx, val, rules_list, predict_state)
            print('Size of Grid idx_heap', len(idx_heap))

            while idx_heap:
                pos_data = heapq.heappop(idx_heap)
                diff_count = pos_data[0]
                rule = pos_data[1]
                updates = self.knowledge['Grid-' + str(idx) + '/' + str(val) + '/' + str(rule) + '/Updates']
                new_val = self.knowledge['Grid-' + str(idx) + '/' + str(val) + '/' + str(rule) + '/New Value']
                age = self.knowledge['Grid-' + str(idx) + '/' + str(val) + '/' + str(rule) + '/Age']

                grid_data = self.predict_grid_conditions('Grid', idx, val, rule, predict_state)
                diff_count += grid_data[0]
                heapq.heappush(predict_heap['Grid'+str(idx)], (diff_count, rule, idx, new_val, 'Grid', pos_data[2] + grid_data[1], updates, age))
                if diff_count == 0:
                    break

        # Apply changes to predict state
        for idx_key in predict_heap.keys():
            if predict_heap[idx_key] is not None:
                if predict_heap[idx_key][0][4] == 'Pos':
                    predict_state.pos_input[predict_heap[idx_key][0][2]] += predict_heap[idx_key][0][3]
                    predict_state.applied_rules_pos[predict_heap[idx_key][0][2]] = predict_heap[idx_key][0]
                elif predict_heap[idx_key][0][4] == 'Grid':
                    predict_state.grid_input[predict_heap[idx_key][0][2]] = predict_heap[idx_key][0][3]
                    predict_state.applied_rules_grid[predict_heap[idx_key][0][2]] = predict_heap[idx_key][0]
                confidence_total += predict_heap[idx_key][0][5]
                confidence_count += predict_heap[idx_key][0][5] - predict_heap[idx_key][0][0]
            else:
                pass

        if confidence_total != 0:
            predict_state.confidence = confidence_count / confidence_total
            predict_state.confidence_count = confidence_count
            predict_state.confidence_total = confidence_total

        return predict_state

    def predict_pos_conditions(self, t, idx, val, rules_list, predict_state):
        idx_rule_heap = []
        o_val = None
        if t == 'Pos':
            o_val = idx
        elif t == 'Grid':
            o_val = val

        for rule in rules_list:
            diff_count = 0
            condition_set = self.knowledge['Pos-' + str(idx) + '/' + str(o_val) + '/' + str(rule) + '/Pos Conditions Set']
            for i in condition_set:
                if predict_state.pos_input[i] != self.knowledge['Pos-' + str(idx) + '/' + str(o_val) + '/' + str(rule) + '/Pos Conditions'][i]:
                    diff_count += 1

            heapq.heappush(idx_rule_heap, (diff_count, rule, len(condition_set)))
            if diff_count == 0:
                break

        return idx_rule_heap

    def predict_grid_conditions(self, t, idx, val, rule, predict_state):
        o_val = None
        if t == 'Pos':
            o_val = idx
        elif t == 'Grid':
            o_val = val
            
        diff_count = 0
        condition_set = self.knowledge['Grid-' + str(idx) + '/' + str(o_val) + '/' + str(rule) + '/Grid Conditions Set']
        for i in condition_set:
            if predict_state.grid_input[i] != self.knowledge['Grid-' + str(idx) + '/' + str(o_val) + '/' + str(rule) + '/Grid Conditions'][i]:
                diff_count += 1

        return diff_count, len(condition_set)
            
    def create_rule(self, act, input_type, index, pre_val, post_val):
        new_rule = str(uuid.uuid4())[:6]
        o_val = None
        n_val = None

        conditions_pos = self.pos_input
        conditions_pos_set = set(range(len(self.pos_input)))
        conditions_pos_freq = [1] * len(self.pos_input)

        conditions_grid = self.grid_input
        conditions_grid_set = set(range(len(self.grid_input)))
        conditions_grid_freq = [1] * len(self.grid_input)

        try:
            check = self.knowledge['Action Rules']
        except KeyError:
            self.knowledge['Action Rules'] = dict()

        try:
            self.knowledge['Action Rules'][act].append(new_rule)
        except KeyError:
            self.knowledge['Action Rules'][act] = [new_rule]

        if input_type == 'Pos':
            try:
                check = self.knowledge[input_type + '-' + str(index) + '/Actions']
            except KeyError:
                self.knowledge[input_type + '-' + str(index) + '/Actions'] = dict()
        elif input_type == 'Grid':
            try:
                check = self.knowledge[input_type + '-' + str(index) + '/' + str(pre_val) + '/Actions']
            except KeyError:
                self.knowledge[input_type + '-' + str(index) + '/' + str(pre_val) + '/Actions'] = dict()

        # TO DO Add duplicate checking?

        if input_type == 'Pos':
            try:
                self.knowledge[input_type + '-' + str(index) + '/Actions'][act].append(new_rule)
            except KeyError:
                self.knowledge[input_type + '-' + str(index) + '/Actions'][act] = [new_rule]
            o_val = index
            n_val = post_val - pre_val
        elif input_type == 'Grid':
            try:
                self.knowledge[input_type + '-' + str(index) + '/' + str(pre_val) + '/Actions'][act].append(new_rule)
            except KeyError:
                self.knowledge[input_type + '-' + str(index) + '/' + str(pre_val) + '/Actions'][act] = [new_rule]
            o_val = pre_val
            n_val = post_val

        self.knowledge[input_type + '-' + str(index) + '/' + str(o_val) + '/' + str(new_rule) + '/Action'] = act
        self.knowledge[input_type + '-' + str(index) + '/' + str(o_val) + '/' + str(new_rule) + '/Updates'] = 0
        self.knowledge[input_type + '-' + str(index) + '/' + str(o_val) + '/' + str(new_rule) + '/New Value'] = n_val
        self.knowledge[input_type + '-' + str(index) + '/' + str(o_val) + '/' + str(new_rule) + '/Age'] = self.time_step
        self.knowledge['Pos-' + str(index) + '/' + str(o_val) + '/' + str(new_rule) + '/Pos Conditions'] = conditions_pos
        self.knowledge['Pos-' + str(index) + '/' + str(o_val) + '/' + str(new_rule) + '/Pos Conditions Set'] = conditions_pos_set
        self.knowledge['Pos-' + str(index) + '/' + str(o_val) + '/' + str(new_rule) + '/Pos Conditions Freq'] = conditions_pos_freq
        self.knowledge['Grid-' + str(index) + '/' + str(o_val) + '/' + str(new_rule) + '/Grid Conditions'] = conditions_grid
        self.knowledge['Grid-' + str(index) + '/' + str(o_val) + '/' + str(new_rule) + '/Grid Conditions Set'] = conditions_grid_set
        self.knowledge['Grid-' + str(index) + '/' + str(o_val) + '/' + str(new_rule) + '/Grid Conditions Freq'] = conditions_grid_freq

    def update_good_rule(self, rule_data):
        # rule data format: (diff_count, rule, idx, new_val, 'Grid' / 'Pos', pos_data[2] + grid_data[1], updates, age)
        idx = rule_data[2]
        rule = rule_data[1]
        t = rule_data[4]
        o_val = None
        if t == 'Pos':
            o_val = idx
        elif t == 'Grid':
            o_val = self.grid_input[idx]

        pos_remove_list = []
        grid_remove_list = []

        for u_idx in self.knowledge['Pos-' + str(idx) + '/' + str(o_val) + '/' + str(rule) + '/Pos Conditions Set']:
            if self.pos_input[u_idx] != self.knowledge['Pos-' + str(idx) + '/' + str(o_val) + '/' + str(rule) + '/Pos Conditions'][u_idx]:
                self.knowledge['Pos-' + str(idx) + '/' + str(o_val) + '/' + str(rule) + '/Pos Conditions Freq'][u_idx] = 0
                pos_remove_list.append(u_idx)

        for u_idx in self.knowledge['Grid-' + str(idx) + '/' + str(o_val) + '/' + str(rule) + '/Grid Conditions Set']:
            if self.grid_input[u_idx] != self.knowledge['Grid-' + str(idx) + '/' + str(o_val) + '/' + str(rule) + '/Grid Conditions'][u_idx]:
                self.knowledge['Grid-' + str(idx) + '/' + str(o_val) + '/' + str(rule) + '/Grid Conditions Freq'][u_idx] = 0
                grid_remove_list.append(u_idx)

        for item in pos_remove_list:
            self.knowledge['Pos-' + str(idx) + '/' + str(o_val) + '/' + str(rule) + '/Pos Conditions Set'].remove(item)

        for item in grid_remove_list:
            self.knowledge['Grid-' + str(idx) + '/' + str(o_val) + '/' + str(rule) + '/Grid Conditions Set'].remove(item)

    def update_bad_rule(self, rule_data):
        # rule data format: (diff_count, rule, idx, new_val, 'Grid' / 'Pos', pos_data[2] + grid_data[1], updates, age)
        idx = rule_data[2]
        rule = rule_data[1]
        t = rule_data[4]
        o_val = None
        if t == 'Pos':
            o_val = idx
        elif t == 'Grid':
            o_val = self.grid_input[idx]

        pos_add_list = []
        grid_add_list = []

        for u_idx, val in enumerate(self.knowledge['Pos-' + str(idx) + '/' + str(o_val) + '/' + str(rule) + '/Pos Conditions']):
            if self.pos_input[u_idx] == val:
                self.knowledge['Pos-' + str(idx) + '/' + str(o_val) + '/' + str(rule) + '/Pos Conditions Freq'][u_idx] = 1
                pos_add_list.append(u_idx)

        for u_idx, val in enumerate(self.knowledge['Grid-' + str(idx) + '/' + str(o_val) + '/' + str(rule) + '/Grid Conditions Set']):
            if self.grid_input[u_idx] == val:
                self.knowledge['Grid-' + str(idx) + '/' + str(o_val) + '/' + str(rule) + '/Grid Conditions Freq'][u_idx] = 1
                grid_add_list.append(u_idx)

        for item in pos_add_list:
            self.knowledge['Pos-' + str(idx) + '/' + str(o_val) + '/' + str(rule) + '/Pos Conditions Set'].add(item)

        for item in grid_add_list:
            self.knowledge['Grid-' + str(idx) + '/' + str(o_val) + '/' + str(rule) + '/Grid Conditions Set'].add(item)

    def lookDir(self, rob, pitch, yaw):
        logging.info("\tinside lookDir")
        for t in range(3000):
            sleep(0.02)  # wait for action
            aPos = rob.waitNotNoneObserve('getAgentPos')
            dPitch = normAngle(degree2rad(pitch) - degree2rad(aPos[3]))
            dYaw = normAngle(degree2rad(yaw) - degree2rad(aPos[4]))
            if abs(dPitch) < 0.006 and abs(dYaw) < 0.006: break
            rob.sendCommand("turn " + str(dYaw * 0.4))
            rob.sendCommand("pitch " + str(dPitch * 0.4))
        rob.sendCommand("turn 0")
        rob.sendCommand("pitch 0")

    def center(self, rob, pos, o_pitch, o_yaw):
        dist = self.lookAt(rob, pos)
        timeout = time()
        if dist > 0.2:
            rob.sendCommand('move .2')
        while dist > 0.2:
            if time() > timeout + 1:
                break
            dist = self.lookAt(rob, pos)
        rob.sendCommand('move 0')
        self.lookDir(rob, o_pitch, o_yaw)

    def lookAt(self, rob, pos):
        pos = [pos[0] + .5, pos[1], pos[2] + .5]
        dist = 0
        for t in range(3000):
            sleep(0.02)
            aPos = rob.waitNotNoneObserve('getAgentPos')
            dist = math.sqrt((aPos[0] - pos[0]) * (aPos[0] - pos[0]) + (aPos[2] - pos[2]) * (aPos[2] - pos[2]))
            if dist < 0.5:
                break
            [pitch, yaw] = mc.dirToPos(aPos, pos)
            pitch = normAngle(pitch - degree2rad(aPos[3]))
            yaw = normAngle(yaw - degree2rad(aPos[4]))
            if abs(pitch) < 0.02 and abs(yaw) < 0.02: break
            rob.sendCommand("turn " + str(yaw * 0.4))
            rob.sendCommand("pitch " + str(pitch * 0.4))
        rob.sendCommand("turn 0")
        rob.sendCommand("pitch 0")
        return dist

    def jump_forward(self, rob):
        stats = [mc.getFullStat(key) for key in fullStatKeys]
        o_pitch = round(stats[3])
        o_yaw = round(stats[4]) % 360
        s_pos = [math.floor(stats[0]), math.floor(stats[1]), math.floor(stats[2])]
        e_pos = s_pos
        match o_yaw:
            case 0:
                e_pos = [s_pos[0], s_pos[1], s_pos[2] + 1]
            case 45:
                e_pos = [s_pos[0] - 1, s_pos[1], s_pos[2] + 1]
            case 90:
                e_pos = [s_pos[0] - 1, s_pos[1], s_pos[2]]
            case 135:
                e_pos = [s_pos[0] - 1, s_pos[1], s_pos[2] - 1]
            case 180:
                e_pos = [s_pos[0], s_pos[1], s_pos[2] - 1]
            case 225:
                e_pos = [s_pos[0] + 1, s_pos[1], s_pos[2] - 1]
            case 270:
                e_pos = [s_pos[0] + 1, s_pos[1], s_pos[2]]
            case 315:
                e_pos = [s_pos[0] + 1, s_pos[1], s_pos[2] + 1]
        self.lookAt(rob, e_pos)
        rob.sendCommand('move 1')
        rob.sendCommand('jump 1')
        timeout = time()
        timedout = False
        while [math.floor(stats[0]), math.floor(stats[1]), math.floor(stats[2])] != [e_pos[0], math.floor(stats[1]), e_pos[2]]:
            self.lookAt(rob, e_pos)
            if time() > timeout + 1:
                timedout = True
                break
            stats = [mc.getFullStat(key) for key in fullStatKeys]
        rob.sendCommand('move 0')
        rob.sendCommand('jump 0')
        sleep(.02)
        self.lookDir(rob, o_pitch, o_yaw)
        stats = [mc.getFullStat(key) for key in fullStatKeys]
        if [math.floor(stats[0]), math.floor(stats[1]), math.floor(stats[2])] != [e_pos[0], math.floor(stats[1]), e_pos[2]]:
            if not timedout:
                self.center(rob, [e_pos[0], math.floor(stats[1]), e_pos[2]], o_pitch, o_yaw)

    def move_forward(self, rob):
        stats = [mc.getFullStat(key) for key in fullStatKeys]
        o_pitch = round(stats[3])
        o_yaw = round(stats[4]) % 360
        s_pos = [math.floor(stats[0]), math.floor(stats[1]), math.floor(stats[2])]
        e_pos = s_pos
        match o_yaw:
            case 0:
                e_pos = [s_pos[0], s_pos[1], s_pos[2] + 1]
            case 45:
                e_pos = [s_pos[0] - 1, s_pos[1], s_pos[2] + 1]
            case 90:
                e_pos = [s_pos[0] - 1, s_pos[1], s_pos[2]]
            case 135:
                e_pos = [s_pos[0] - 1, s_pos[1], s_pos[2] - 1]
            case 180:
                e_pos = [s_pos[0], s_pos[1], s_pos[2] - 1]
            case 225:
                e_pos = [s_pos[0] + 1, s_pos[1], s_pos[2] - 1]
            case 270:
                e_pos = [s_pos[0] + 1, s_pos[1], s_pos[2]]
            case 315:
                e_pos = [s_pos[0] + 1, s_pos[1], s_pos[2] + 1]
        self.lookAt(rob, e_pos)
        rob.sendCommand('move 1')
        timeout = time()
        timedout = False
        while [math.floor(stats[0]), math.floor(stats[1]), math.floor(stats[2])] != e_pos:
            self.lookAt(rob, e_pos)
            if time() > timeout + 1:
                timedout = True
                break
            stats = [mc.getFullStat(key) for key in fullStatKeys]
            if math.floor(stats[1]) != math.floor(e_pos[1]):
                break
        rob.sendCommand('move 0')
        sleep(.02)
        self.lookDir(rob, o_pitch, o_yaw)
        stats = [mc.getFullStat(key) for key in fullStatKeys]
        if [math.floor(stats[0]), math.floor(stats[1]), math.floor(stats[2])] != [e_pos[0], math.floor(stats[1]), e_pos[2]]:
            if not timedout:
                self.center(rob, [e_pos[0], math.floor(stats[1]), e_pos[2]], o_pitch, o_yaw)

    def mine(self, rob):
        stats_old = [mc.getFullStat(key) for key in fullStatKeys]
        rob.sendCommand('attack 1')
        timeout = 0
        while np.all(self.grid_input == mc.getNearGrid()):
            sleep(0.2)
            timeout += 0.2
            if timeout > 30:
                break
        rob.sendCommand('attack 0')
        sleep(0.2)
        stats = [mc.getFullStat(key) for key in fullStatKeys]
        while stats != stats_old:
            stats_old = stats
            sleep(0.1)
            stats = [mc.getFullStat(key) for key in fullStatKeys]


    def save_knowledge(self, fname):
        np.save(fname, self.knowledge)

    def load_knowledge(self, fname):
        try:
            self.knowledge = np.load(fname, allow_pickle=True).item()
        except FileNotFoundError:
            self.knowledge = dict()

if __name__ == '__main__':

    miss = mb.MissionXML()
    # https://www.chunkbase.com/apps/superflat-generator
    # flat world not working currently
    # miss.setWorld(mb.flatworld("3;7,25*1,3*3,2;1;stronghold,biome_1,village,decoration,dungeon,lake,mineshaft,lava_lake"))
    # miss.addAgent(1)
    world = mb.defaultworld(
        seed='5',
        forceReset="false",
        forceReuse="true")
    miss.setWorld(world)

    airis = Airis()

    mc = MCConnector(miss)
    mc.safeStart()

    rob = RobustObserver(mc)

    fullStatKeys = ['XPos', 'YPos', 'ZPos', 'Pitch', 'Yaw']

    sleep(60)
    print('starting!')

    airis.lookDir(rob, 0, 0)

    stats = [mc.getFullStat(key) for key in fullStatKeys]
    stats = [math.floor(stats[0]), math.floor(stats[1]), math.floor(stats[2]), round(stats[3]), round(stats[4]) % 360]
    grid = mc.getNearGrid()
    stats_old = stats

    # Test routine 3 (dig straight down)
    action, state = airis.capture_input(stats, grid, 'look floor', None, True)
    airis.lookDir(rob, 90, 0)
    stats = [mc.getFullStat(key) for key in fullStatKeys]
    stats = [math.floor(stats[0]), math.floor(stats[1]), math.floor(stats[2]), round(stats[3]), round(stats[4]) % 360]
    grid = mc.getNearGrid()
    airis.capture_input(stats, grid, action, state, False)

    for test in range(20):
        action, state = airis.capture_input(stats, grid, 'mine', None, True)
        airis.mine(rob)
        stats = [mc.getFullStat(key) for key in fullStatKeys]
        stats = [math.floor(stats[0]), math.floor(stats[1]), math.floor(stats[2]), round(stats[3]), round(stats[4]) % 360]
        grid = mc.getNearGrid()
        airis.capture_input(stats, grid, action, state, False)

    # # Test routine 2 (walk forward until collision, then jump forward, then continue walking forward)
    # for test in range(10):
    #     action, state = airis.capture_input(stats, grid, 'move', None, True)
    #     airis.move_forward(rob, 0, 0)
    #     stats = [mc.getFullStat(key) for key in fullStatKeys]
    #     stats = [math.floor(stats[0]), math.floor(stats[1]), math.floor(stats[2]), round(stats[3]), round(stats[4]) % 360]
    #     grid = mc.getNearGrid()
    #     airis.capture_input(stats, grid, action, state, False)
    #
    #     while stats_old != stats:
    #         stats_old = stats
    #         action, state = airis.capture_input(stats, grid, 'move', None, True)
    #         airis.move_forward(rob, 0, 0)
    #         stats = [mc.getFullStat(key) for key in fullStatKeys]
    #         stats = [math.floor(stats[0]), math.floor(stats[1]), math.floor(stats[2]), round(stats[3]), round(stats[4]) % 360]
    #         grid = mc.getNearGrid()
    #         airis.capture_input(stats, grid, action, state, False)
    #
    #     stats_old = stats
    #     action, state = airis.capture_input(stats, grid, 'jump', None, True)
    #     airis.jump_forward(rob, 0, 0)
    #     stats = [mc.getFullStat(key) for key in fullStatKeys]
    #     stats = [math.floor(stats[0]), math.floor(stats[1]), math.floor(stats[2]), round(stats[3]), round(stats[4]) % 360]
    #     grid = mc.getNearGrid()
    #     airis.capture_input(stats, grid, action, state, False)

    # # Test routine 1 (walk forward 1, turn around, walk forward 1, turn around, ...)
    # for test in range(10):
    #     action, state = airis.capture_input(stats, grid, 'move', None, True)
    #     airis.move_forward(rob, 0, 0)
    #     stats = [mc.getFullStat(key) for key in fullStatKeys]
    #     stats = [math.floor(stats[0]), math.floor(stats[1]), math.floor(stats[2]), round(stats[3]), round(stats[4]) % 360]
    #     grid = mc.getNearGrid()
    #     airis.capture_input(stats, grid, action, state, False)
    #
    #     action, state = airis.capture_input(stats, grid, 'turn 45', None, True)
    #     airis.lookDir(rob, 0, 45)
    #     stats = [mc.getFullStat(key) for key in fullStatKeys]
    #     stats = [math.floor(stats[0]), math.floor(stats[1]), math.floor(stats[2]), round(stats[3]), round(stats[4]) % 360]
    #     grid = mc.getNearGrid()
    #     airis.capture_input(stats, grid, action, state, False)
    #
    #     action, state = airis.capture_input(stats, grid, 'turn 90', None, True)
    #     airis.lookDir(rob, 0, 90)
    #     stats = [mc.getFullStat(key) for key in fullStatKeys]
    #     stats = [math.floor(stats[0]), math.floor(stats[1]), math.floor(stats[2]), round(stats[3]), round(stats[4]) % 360]
    #     grid = mc.getNearGrid()
    #     airis.capture_input(stats, grid, action, state, False)
    #
    #     action, state = airis.capture_input(stats, grid, 'turn 135', None, True)
    #     airis.lookDir(rob, 0, 135)
    #     stats = [mc.getFullStat(key) for key in fullStatKeys]
    #     stats = [math.floor(stats[0]), math.floor(stats[1]), math.floor(stats[2]), round(stats[3]), round(stats[4]) % 360]
    #     grid = mc.getNearGrid()
    #     airis.capture_input(stats, grid, action, state, False)
    #
    #     action, state = airis.capture_input(stats, grid, 'turn 180', None, True)
    #     airis.lookDir(rob, 0, 180)
    #     stats = [mc.getFullStat(key) for key in fullStatKeys]
    #     stats = [math.floor(stats[0]), math.floor(stats[1]), math.floor(stats[2]), round(stats[3]), round(stats[4]) % 360]
    #     grid = mc.getNearGrid()
    #     airis.capture_input(stats, grid, action, state, False)
    #
    #     action, state = airis.capture_input(stats, grid, 'move', None, True)
    #     airis.move_forward(rob, 0, 180)
    #     stats = [mc.getFullStat(key) for key in fullStatKeys]
    #     stats = [math.floor(stats[0]), math.floor(stats[1]), math.floor(stats[2]), round(stats[3]), round(stats[4]) % 360]
    #     grid = mc.getNearGrid()
    #     airis.capture_input(stats, grid, action, state, False)
    #
    #     action, state = airis.capture_input(stats, grid, 'turn 135', None, True)
    #     airis.lookDir(rob, 0, 135)
    #     stats = [mc.getFullStat(key) for key in fullStatKeys]
    #     stats = [math.floor(stats[0]), math.floor(stats[1]), math.floor(stats[2]), round(stats[3]), round(stats[4]) % 360]
    #     grid = mc.getNearGrid()
    #     airis.capture_input(stats, grid, action, state, False)
    #
    #     action, state = airis.capture_input(stats, grid, 'turn 90', None, True)
    #     airis.lookDir(rob, 0, 90)
    #     stats = [mc.getFullStat(key) for key in fullStatKeys]
    #     stats = [math.floor(stats[0]), math.floor(stats[1]), math.floor(stats[2]), round(stats[3]), round(stats[4]) % 360]
    #     grid = mc.getNearGrid()
    #     airis.capture_input(stats, grid, action, state, False)
    #
    #     action, state = airis.capture_input(stats, grid, 'turn 45', None, True)
    #     airis.lookDir(rob, 0, 45)
    #     stats = [mc.getFullStat(key) for key in fullStatKeys]
    #     stats = [math.floor(stats[0]), math.floor(stats[1]), math.floor(stats[2]), round(stats[3]), round(stats[4]) % 360]
    #     grid = mc.getNearGrid()
    #     airis.capture_input(stats, grid, action, state, False)
    #
    #     action, state = airis.capture_input(stats, grid, 'turn 0', None, True)
    #     airis.lookDir(rob, 0, 0)
    #     stats = [mc.getFullStat(key) for key in fullStatKeys]
    #     stats = [math.floor(stats[0]), math.floor(stats[1]), math.floor(stats[2]), round(stats[3]), round(stats[4]) % 360]
    #     grid = mc.getNearGrid()
    #     airis.capture_input(stats, grid, action, state, False)

    airis.save_knowledge('Knowledge.npy')
    print('Test Routine Complete')