import logging
import math
import heapq
from time import sleep, time
from tagilmo.utils.vereya_wrapper import MCConnector, RobustObserver
import tagilmo.utils.mission_builder as mb
import numpy as np
import uuid

from copy import copy, deepcopy
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
        self.all_rules_pos = dict()
        self.applied_rules_grid = dict()
        self.all_rules_grid = dict()
        self.compare = None
        self.confidence = None
        self.confidence_count = None
        self.confidence_total = None
        self.debug_heap = None
        self.debug_dict = dict()


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
        # self.given_goal = None
        self.given_goal = ((0, 65, 0, None, None), ())
        self.goal_achieved = False
        self.applied_rules_pos = dict()
        self.applied_rules_grid = dict()
        self.bad_predictions_pos = None
        self.bad_predictions_grid = None
        self.prev_applied_rules_pos = None
        self.prev_applied_rules_grid = None
        self.time_step = 0
        self.state_history = set()
        self.prediction_flex = 1
        self.last_compare = None
        self.best_compare = None
        self.grid_map = np.empty((500, 128, 500), dtype=np.dtype('U42'))
        self.map_origin_x = 0
        self.map_origin_y = 0
        self.map_origin_z = 0
        self.actions = ['move 0', 'move 45', 'move 90', 'move 135', 'move 180', 'move 225', 'move 270', 'move 315',
                        'jump 0', 'jump 45', 'jump 90', 'jump 135', 'jump 180', 'jump 225', 'jump 270', 'jump 315']
        # 'mine up 0', 'mine up 45', 'mine up 90', 'mine up 135', 'mine up 180', 'mine up 225', 'mine up 270', 'mine up 315',
        # 'mine straight 0', 'mine straight 45', 'mine straight 90', 'mine straight 135', 'mine straight 180', 'mine straight 225', 'mine straight 270', 'mine straight 315']
        # 'mine down 0', 'mine down 45', 'mine down 90', 'mine down 135', 'mine down 180', 'mine down 225', 'mine down 270', 'mine down 315',

    def capture_input(self, pos_input, grid_input, action, state, pre):
        if pre:
            self.pos_input = np.asarray([math.floor(pos_input[0]), math.floor(pos_input[1]), math.floor(pos_input[2]), round(pos_input[3]), round(pos_input[4] % 360)])
            self.grid_input = np.asarray(grid_input, dtype=np.dtype('U42'))

            self.map_origin_x = 2
            self.map_origin_y = 2
            self.map_origin_z = 2

            grid_input_3d = self.grid_input.copy()
            grid_input_3d = grid_input_3d.reshape((5, 5, 5))

            self.grid_map[self.pos_input[0] - self.map_origin_x:self.pos_input[0] + self.map_origin_x + 1, self.pos_input[1] - self.map_origin_y:self.pos_input[1] + self.map_origin_y + 1, self.pos_input[2] - self.map_origin_z:self.pos_input[2] + self.map_origin_z + 1] = grid_input_3d

            if not self.action_plan:
                self.states = [State(self.pos_input, self.grid_input, None, 0, 0)]
                self.states[0].change_pos = self.last_change_pos
                self.states[0].change_grid = self.last_change_grid

            if self.given_goal:
                if (self.pos_input[0], self.pos_input[1], self.pos_input[2]) == (self.given_goal[0][0], self.given_goal[0][1], self.given_goal[0][2]):
                    self.goal_achieved = True
                    return 'turn 0', 0
                else:
                    if not self.action_plan:
                        self.make_plan()
                        if self.action_plan:
                            return self.action_plan.pop(0)
                    else:
                        return self.action_plan.pop(0)
            else:
                new_state = self.predict(action, 0)
                self.states.append(new_state)
                return action, 1

        else:
            new_pos_input = np.asarray([math.floor(pos_input[0]), math.floor(pos_input[1]), math.floor(pos_input[2]), round(pos_input[3]), round(pos_input[4] % 360)])
            new_grid_input = np.asarray(grid_input, dtype=np.dtype('U42'))

            new_grid_3d = new_grid_input.copy()
            new_grid_3d = new_grid_3d.reshape((5, 5, 5))

            self.grid_map[self.pos_input[0] - self.map_origin_x:self.pos_input[0] + self.map_origin_x + 1, self.pos_input[1] - self.map_origin_y:self.pos_input[1] + self.map_origin_y + 1, self.pos_input[2] - self.map_origin_z:self.pos_input[2] + self.map_origin_z + 1] = new_grid_3d

            clear_plan = False
            self.last_action = action
            self.prev_applied_rules_pos = deepcopy(self.applied_rules_pos)
            self.prev_applied_rules_grid = deepcopy(self.applied_rules_grid)
            self.bad_predictions_pos = dict()
            self.bad_predictions_grid = dict()

            self.last_change_pos = set()
            self.last_change_grid = set()

            if np.all(self.pos_input == new_pos_input) and np.all(self.grid_input == new_grid_input) and self.states[state].confidence == 0:
                # create "no change" rules
                for i, v in enumerate(self.pos_input):
                    self.create_rule(action, 'Pos', i, self.pos_input[i], new_pos_input[i])
                for i, v in enumerate(self.grid_input):
                    self.create_rule(action, 'Grid', i, self.grid_input[i], new_grid_input[i])

            pos_mismatch = [i for i, v in enumerate(self.states[state].pos_input) if v != new_pos_input[i]]
            grid_mismatch = [i for i, v in enumerate(self.states[state].grid_input) if v != new_grid_input[i]]

            self.applied_rules_pos = deepcopy(self.states[state].applied_rules_pos)
            self.applied_rules_grid = deepcopy(self.states[state].applied_rules_grid)

            if pos_mismatch:
                clear_plan = True
                for index in pos_mismatch:
                    print('POS mismatch - ', index, self.pos_input[index], new_pos_input[index], self.states[state].pos_input[index])
                    try:
                        print('POS Prediction: ', self.applied_rules_pos[index])
                        print('POS Predict Heap: ', self.states[state].debug_heap['Pos' + str(index)])
                        self.bad_predictions_pos[index] = deepcopy(self.applied_rules_pos[index])
                        del self.applied_rules_pos[index]
                    except KeyError:
                        pass

                for index in pos_mismatch:
                    found = False
                    try:
                        while self.states[state].all_rules_pos[index]:
                            data = heapq.heappop(self.states[state].all_rules_pos[index])
                            if data[3] == new_pos_input[index]:
                                found = True
                                self.applied_rules_pos[index] = data
                                break
                    except KeyError:
                        pass

                    if not found:
                        self.create_rule(action, 'Pos', index, self.pos_input[index], new_pos_input[index])

            elif grid_mismatch:
                clear_plan = True
                for index in grid_mismatch:
                    print('GRID mismatch - ', index, self.grid_input[index], new_grid_input[index], self.states[state].grid_input[index])
                    try:
                        print('GRID Prediction: ', self.applied_rules_grid[index])
                        print('GRID Predict Heap: ', self.states[state].debug_heap['Grid' + str(index)])
                        self.bad_predictions_grid[index] = deepcopy(self.applied_rules_grid[index])
                        del self.applied_rules_grid[index]
                    except KeyError:
                        pass

                for index in grid_mismatch:
                    found = False
                    try:
                        while self.states[state].all_rules_grid[index]:
                            data = heapq.heappop(self.states[state].all_rules_grid[index])
                            if data[3] == new_grid_input[index]:
                                found = True
                                self.applied_rules_pos[index] = data
                                break
                    except KeyError:
                        pass

                    if not found:
                        self.create_rule(action, 'Grid', index, self.grid_input[index], new_grid_input[index])

            # (diff_count, rule, idx, new_val, 'Pos', pos_data[2] + grid_data[1], updates, age, idx)
            # self.debug_dict['Pos' + str(idx) + str(rule) + state] = (idx, rule, i, predict_state.pos_input[i]

            for index in self.applied_rules_pos.keys():
                self.update_good_rule(self.applied_rules_pos[index])

            for index in self.applied_rules_grid.keys():
                self.update_good_rule(self.applied_rules_grid[index])

            # for key in self.states[state].debug_dict.keys():
            #     print('debug dict', key, self.states[state].debug_dict[key])

            for index in self.bad_predictions_pos.keys():
                if self.bad_predictions_pos[index][0] == 0:
                    self.update_bad_rule(self.bad_predictions_pos[index])

            for index in self.bad_predictions_grid.keys():
                if self.bad_predictions_grid[index][0] == 0:
                    self.update_bad_rule(self.bad_predictions_grid[index])

            # print('Prediction State: ', self.states[state])

            print('Prediction Confidence: ', self.states[state].confidence, self.states[state].confidence_count, '/', self.states[state].confidence_total)
            print('Prediction previous state', self.states[state].prev_state)
            if clear_plan:
                print('Prediction Incorrect...')
                self.action_plan = []
            else:
                print('Prediction Correct!')

            self.last_compare = self.states[state].compare
            print('State Goal Compare', self.states[state].compare)

            self.prev_last_change_pos = deepcopy(self.last_change_pos)
            self.prev_last_change_grid = deepcopy(self.last_change_grid)
            self.time_step += 1
            if not clear_plan:
                self.state_history.add(hash((tuple(self.pos_input), tuple(self.grid_input), action)))

    def make_plan(self):
        current_state = 0
        self.action_plan = []
        goal_compare = self.compare(0)
        # goal_heap: Compare, State Index, Confidence
        goal_heap = [(goal_compare, 0, 1, None, None)]
        compare_heap = []
        goal_state = 0
        # confidence_heap: Confidence, State Index, Compare
        most_confidence_heap = []
        least_confidence_heap = []
        state_hash_set = set()
        goal_reached = False
        if goal_compare == 0:
            goal_reached = True

        while not goal_reached:
            fresh_state = False
            for act in self.actions:
                try:
                    check = self.knowledge['Action Rules'][act]
                except KeyError:
                    self.states.append(self.predict(act, 0))
                    goal_reached = True
                    goal_state = len(self.states) - 1
                    break

                new_state = self.predict(act, current_state)
                state_hash = hash((tuple(self.states[current_state].pos_input), tuple(self.states[current_state].grid_input), act))
                if state_hash not in state_hash_set:
                    self.states.append(new_state)
                    state_idx = len(self.states) - 1
                    goal_compare = self.compare(state_idx)
                    self.states[state_idx].compare = goal_compare
                    state_confidence = self.states[state_idx].confidence
                    if state_confidence == 1:
                        heapq.heappush(goal_heap, (goal_compare, state_idx, state_confidence, act, state_hash))
                    heapq.heappush(compare_heap, (goal_compare, state_idx, state_confidence, act, state_hash))
                    heapq.heappush(most_confidence_heap, (-state_confidence, state_idx, goal_compare, act, state_hash))
                    heapq.heappush(least_confidence_heap, (state_confidence, state_idx, goal_compare, act, state_hash))
                    state_hash_set.add(state_hash)
                    fresh_state = True
                    print('Predicting that action', act, 'from state', current_state, 'will result in state', state_idx, 'with a confidence of', state_confidence, self.states[state_idx].confidence_count, '/', self.states[state_idx].confidence_total, 'and a compare of', goal_compare)

            if not fresh_state:
                print('No fresh predictions found from current state', current_state)

            if compare_heap and not goal_reached:
                if compare_heap[0][0] == 0:
                    goal_reached = True
                    goal_state = compare_heap[0][1]
                    print('Predicted state reaches goal! Trying...')
                    print('Goal State', goal_state)
                    break

                if most_confidence_heap[0][0] == -1:
                    current_state = most_confidence_heap[0][1]
                    heapq.heappop(most_confidence_heap)
                    print('Setting new current state of', current_state)
                else:
                    goal_reached = True
                    goal_found = False

                    print('Best compare is', goal_heap[0][0], 'from state', goal_heap[0][2])
                    best_compare = goal_heap[0][0]
                    print('least confidence pre popping', least_confidence_heap)
                    while goal_heap:
                        temp_heap = deepcopy(least_confidence_heap)
                        while temp_heap[0][2] != best_compare and temp_heap[0][2] == 1:
                            print('least confident compare of', temp_heap[0][2], 'does not match best compare. Popping.')
                            heapq.heappop(temp_heap)
                            if not temp_heap:
                                break
                        if temp_heap:
                            goal_state = temp_heap[0][1]
                            goal_found = True
                            print('goal state found in temp heap')
                            break
                        else:
                            print('no low confidence predictions from best compare of', best_compare)
                            heapq.heappop(goal_heap)
                            best_compare = goal_heap[0][0]

                    if not goal_found:
                        goal_state = least_confidence_heap[0][1]
                        print('goal state set to overall least confident prediction')

                    print('goal state is', goal_state, 'with a confidence of', self.states[goal_state].confidence, self.states[goal_state].confidence_count, '/', self.states[goal_state].confidence_total)

            # if len(self.states) > 550:
            #     goal_reached = True
            #     goal_state = goal_heap[0][1]

        plan_state = goal_state
        self.action_plan.insert(0, (self.states[plan_state].action, plan_state))
        plan_state = self.states[plan_state].prev_state
        while plan_state > 0:
            self.action_plan.insert(0, (self.states[plan_state].action, plan_state))
            plan_state = self.states[plan_state].prev_state
            pass

    def compare(self, state):
        compare_total = 0
        # Compare current position to goal position
        # self.given_goal = ((0, 65, 0, None, None),())
        for i, c_val in enumerate(self.given_goal[0]):
            if c_val is not None:
                if i == 1:
                    if self.states[state].pos_input[i] < c_val - 1:
                        compare_total += abs(c_val - self.states[state].pos_input[i]) * 10
                    else:
                        compare_total += abs(c_val - self.states[state].pos_input[i])
                else:
                    compare_total += abs(c_val - self.states[state].pos_input[i])

        return compare_total

    def predict(self, act, base_state):
        confidence_count = 0
        confidence_total = 0
        predict_heap = dict()
        self.debug_dict = dict()

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

            o_val = idx

            best_diff = None

            for rule in rules_list[::-1]:
                diff_count = 0
                total_len = 0
                # POS
                condition_set = self.knowledge['Pos-' + str(idx) + '/' + str(o_val) + '/' + str(rule) + '/Pos Conditions Set']
                total_len += len(condition_set)
                for i in condition_set:
                    if predict_state.pos_input[i] != self.knowledge['Pos-' + str(idx) + '/' + str(o_val) + '/' + str(rule) + '/Pos Conditions'][i]:
                        diff_count += 1

                if best_diff is not None:
                    if diff_count > best_diff:
                        continue

                # GRID
                condition_set = self.knowledge['Grid-' + str(idx) + '/' + str(o_val) + '/' + str(rule) + '/Grid Conditions Set']
                total_len += len(condition_set)
                for i in condition_set:
                    if predict_state.grid_input[i] != self.knowledge['Grid-' + str(idx) + '/' + str(o_val) + '/' + str(rule) + '/Grid Conditions'][i]:
                        diff_count += 1

                if best_diff is not None:
                    if diff_count > best_diff:
                        continue

                updates = self.knowledge['Pos-' + str(idx) + '/' + str(idx) + '/' + str(rule) + '/Updates']
                new_val = self.knowledge['Pos-' + str(idx) + '/' + str(idx) + '/' + str(rule) + '/New Value']
                age = self.knowledge['Pos-' + str(idx) + '/' + str(idx) + '/' + str(rule) + '/Age']

                best_diff = diff_count

                if predict_heap['Pos' + str(idx)]:
                    if predict_heap['Pos' + str(idx)][0][0] == diff_count:
                        if age > predict_heap['Pos' + str(idx)][0][7]:
                            heapq.heapreplace(predict_heap['Pos' + str(idx)], (diff_count, rule, idx, new_val, 'Pos', total_len, updates, age, idx))
                    else:
                        heapq.heappush(predict_heap['Pos' + str(idx)], (diff_count, rule, idx, new_val, 'Pos', total_len, updates, age, idx))
                else:
                    heapq.heappush(predict_heap['Pos' + str(idx)], (diff_count, rule, idx, new_val, 'Pos', total_len, updates, age, idx))

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

            o_val = val

            best_diff = None

            for rule in rules_list[::-1]:
                diff_count = 0
                total_len = 0
                # POS
                condition_set = self.knowledge['Pos-' + str(idx) + '/' + str(o_val) + '/' + str(rule) + '/Pos Conditions Set']
                total_len += len(condition_set)
                for i in condition_set:
                    if predict_state.pos_input[i] != self.knowledge['Pos-' + str(idx) + '/' + str(o_val) + '/' + str(rule) + '/Pos Conditions'][i]:
                        diff_count += 1

                if best_diff is not None:
                    if diff_count > best_diff:
                        continue

                # GRID
                condition_set = self.knowledge['Grid-' + str(idx) + '/' + str(o_val) + '/' + str(rule) + '/Grid Conditions Set']
                total_len += len(condition_set)
                for i in condition_set:
                    if predict_state.grid_input[i] != self.knowledge['Grid-' + str(idx) + '/' + str(o_val) + '/' + str(rule) + '/Grid Conditions'][i]:
                        diff_count += 1

                if best_diff is not None:
                    if diff_count > best_diff:
                        continue

                updates = self.knowledge['Grid-' + str(idx) + '/' + str(val) + '/' + str(rule) + '/Updates']
                new_val = self.knowledge['Grid-' + str(idx) + '/' + str(val) + '/' + str(rule) + '/New Value']
                age = self.knowledge['Grid-' + str(idx) + '/' + str(val) + '/' + str(rule) + '/Age']

                best_diff = diff_count

                if predict_heap['Grid' + str(idx)]:
                    if predict_heap['Grid' + str(idx)][0][0] == diff_count:
                        if age > predict_heap['Grid' + str(idx)][0][7]:
                            heapq.heapreplace(predict_heap['Grid' + str(idx)], (diff_count, rule, idx, new_val, 'Grid', total_len, updates, age, val))
                    else:
                        heapq.heappush(predict_heap['Grid' + str(idx)], (diff_count, rule, idx, new_val, 'Grid', total_len, updates, age, val))
                else:
                    heapq.heappush(predict_heap['Grid' + str(idx)], (diff_count, rule, idx, new_val, 'Grid', total_len, updates, age, val))

                if diff_count == 0:
                    break

        grid_changes = []

        # Apply changes to predict state
        for idx_key in predict_heap.keys():
            if predict_heap[idx_key] is not None:
                if predict_heap[idx_key][0][4] == 'Pos':
                    if predict_heap[idx_key][0][2] <= 2:
                        predict_state.pos_input[predict_heap[idx_key][0][2]] += predict_heap[idx_key][0][3]
                    else:
                        predict_state.pos_input[predict_heap[idx_key][0][2]] = predict_heap[idx_key][0][3]
                    predict_state.applied_rules_pos[predict_heap[idx_key][0][2]] = predict_heap[idx_key][0]
                    predict_state.all_rules_pos[predict_heap[idx_key][0][2]] = predict_heap[idx_key]
                    # if predict_heap[idx_key][0][0] != 0:
                    #     try:
                    #         print('Best Rule POS condition difference for state', predict_state, self.debug_dict['Pos'+str(predict_heap[idx_key][0][2])+str(predict_heap[idx_key][0][1])])
                    #     except KeyError:
                    #         pass
                elif predict_heap[idx_key][0][4] == 'Grid':
                    grid_changes.append((predict_heap[idx_key][0][2], predict_heap[idx_key][0][3]))
                    # predict_state.grid_input[predict_heap[idx_key][0][2]] = predict_heap[idx_key][0][3]
                    predict_state.applied_rules_grid[predict_heap[idx_key][0][2]] = predict_heap[idx_key][0]
                    predict_state.all_rules_grid[predict_heap[idx_key][0][2]] = predict_heap[idx_key]
                    # if predict_heap[idx_key][0][0] != 0:
                    #     try:
                    #         print('Best Rule GRID condition difference for state', predict_state, self.debug_dict['Grid'+str(predict_heap[idx_key][0][2])+str(predict_heap[idx_key][0][1])])
                    #     except KeyError:
                    #         pass
                confidence_total += predict_heap[idx_key][0][5]
                confidence_count += predict_heap[idx_key][0][5] - predict_heap[idx_key][0][0]
            else:
                pass

        predict_state.grid_input = self.grid_map[predict_state.pos_input[0] - self.map_origin_x:predict_state.pos_input[0] + self.map_origin_x + 1, predict_state.pos_input[1] - self.map_origin_y:predict_state.pos_input[1] + self.map_origin_y + 1, predict_state.pos_input[2] - self.map_origin_z:predict_state.pos_input[2] + self.map_origin_z + 1]

        predict_state.grid_input = predict_state.grid_input.flatten()

        for item in grid_changes:
            predict_state.grid_input[item[0]] = item[1]

        predict_state.debug_heap = deepcopy(predict_heap)

        if confidence_total != 0:
            predict_state.confidence = confidence_count / confidence_total
            predict_state.confidence_count = confidence_count
            predict_state.confidence_total = confidence_total
        else:
            predict_state.confidence = 0

        return predict_state

    def create_rule(self, act, input_type, index, pre_val, post_val):
        new_rule = str(uuid.uuid4())[:6]
        o_val = None
        n_val = None

        conditions_pos = self.pos_input
        conditions_pos_set = set(range(len(self.pos_input)))
        conditions_pos_freq = [1] * len(self.pos_input)

        for i, v in enumerate(self.pos_input):
            try:
                if self.applied_rules_pos[i]:
                    conditions_pos_freq[i] = 0
                    conditions_pos_set.remove(i)
            except KeyError:
                pass

        conditions_grid = self.grid_input
        conditions_grid_set = set(range(len(self.grid_input)))
        conditions_grid_freq = [1] * len(self.grid_input)

        for i, v in enumerate(self.grid_input):
            try:
                if self.applied_rules_grid[i]:
                    conditions_grid_freq[i] = 0
                    conditions_grid_set.remove(i)
            except KeyError:
                pass

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
            if index <= 2:
                n_val = post_val - pre_val
            else:
                n_val = post_val
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
        print('Rule', new_rule, 'created for', act, input_type, index, pre_val, post_val)

    def update_good_rule(self, rule_data):
        # rule data format: (diff_count, rule, idx, new_val, 'Grid' / 'Pos', pos_data[2] + grid_data[1], updates, age, o_val)
        idx = rule_data[2]
        rule = rule_data[1]
        t = rule_data[4]
        o_val = rule_data[8]

        pos_remove_list = []
        grid_remove_list = []

        for u_idx in self.knowledge['Pos-' + str(idx) + '/' + str(o_val) + '/' + str(rule) + '/Pos Conditions Set']:
            if self.pos_input[u_idx] != self.knowledge['Pos-' + str(idx) + '/' + str(o_val) + '/' + str(rule) + '/Pos Conditions'][u_idx]:
                # print('updating rule to remove POS set index', u_idx)
                self.knowledge['Pos-' + str(idx) + '/' + str(o_val) + '/' + str(rule) + '/Pos Conditions Freq'][u_idx] = 0
                pos_remove_list.append(u_idx)

        for u_idx in self.knowledge['Grid-' + str(idx) + '/' + str(o_val) + '/' + str(rule) + '/Grid Conditions Set']:
            if self.grid_input[u_idx] != self.knowledge['Grid-' + str(idx) + '/' + str(o_val) + '/' + str(rule) + '/Grid Conditions'][u_idx]:
                # print('updating rule to remove GRID set index', u_idx)
                self.knowledge['Grid-' + str(idx) + '/' + str(o_val) + '/' + str(rule) + '/Grid Conditions Freq'][u_idx] = 0
                grid_remove_list.append(u_idx)

        for item in pos_remove_list:
            self.knowledge['Pos-' + str(idx) + '/' + str(o_val) + '/' + str(rule) + '/Pos Conditions Set'].remove(item)

        for item in grid_remove_list:
            self.knowledge['Grid-' + str(idx) + '/' + str(o_val) + '/' + str(rule) + '/Grid Conditions Set'].remove(item)

        # print('updating complete')

    def update_bad_rule(self, rule_data):
        # rule data format: (diff_count, rule, idx, new_val, 'Grid' / 'Pos', pos_data[2] + grid_data[1], updates, age, o_val)
        idx = rule_data[2]
        rule = rule_data[1]
        t = rule_data[4]
        o_val = rule_data[8]

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
        timeout = time()
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
            if time() > timeout + 1:
                break
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
        sleep(0.2)
        while np.all(self.grid_input == mc.getNearGrid()):
            sleep(0.2)
            timeout += 0.2
            if timeout > 1:
                break
        rob.sendCommand('attack 0')
        sleep(0.2)
        stats = [mc.getFullStat(key) for key in fullStatKeys]
        while stats != stats_old:
            stats_old = stats
            timeout += .1
            sleep(0.1)
            stats = [mc.getFullStat(key) for key in fullStatKeys]
            if timeout > 10:
                break

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

    sleep(30)
    print('starting!')

    airis.lookDir(rob, 0, 0)

    stats = [mc.getFullStat(key) for key in fullStatKeys]
    stats = [math.floor(stats[0]), math.floor(stats[1]), math.floor(stats[2]), round(stats[3]), 0]  # round(stats[4]) % 360]
    grid = mc.getNearGrid()

    while not airis.goal_achieved:
        action, state = airis.capture_input(stats, grid, None, None, True)
        print('Action Plan: ', airis.action_plan)
        print('performing action', action, 'and predicting state', state)
        # self.actions = ['move 0', 'move 45', 'move 90', 'move 135', 'move 180', 'move 225', 'move 270', 'move 315',
        #                 'jump 0', 'jump 45', 'jump 90', 'jump 135', 'jump 180', 'jump 225', 'jump 270', 'jump 315',
        #                 'mine up 0', 'mine up 45', 'mine up 90', 'mine up 135', 'mine up 180', 'mine up 225', 'mine up 270', 'mine up 315',
        #                 'mine down 0', 'mine down 45', 'mine down 90', 'mine down 135', 'mine down 180', 'mine down 225', 'mine down 270', 'mine down 315',
        #                 'mine straight 0', 'mine straight 45', 'mine straight 90', 'mine straight 135', 'mine straight 180', 'mine straight 225', 'mine straight 270', 'mine straight 315']
        match action:
            case 'move 0':
                airis.lookDir(rob, 0, 0)
                airis.move_forward(rob)

            case 'move 45':
                airis.lookDir(rob, 0, 45)
                airis.move_forward(rob)

            case 'move 90':
                airis.lookDir(rob, 0, 90)
                airis.move_forward(rob)

            case 'move 135':
                airis.lookDir(rob, 0, 135)
                airis.move_forward(rob)

            case 'move 180':
                airis.lookDir(rob, 0, 180)
                airis.move_forward(rob)

            case 'move 225':
                airis.lookDir(rob, 0, 225)
                airis.move_forward(rob)

            case 'move 270':
                airis.lookDir(rob, 0, 270)
                airis.move_forward(rob)

            case 'move 315':
                airis.lookDir(rob, 0, 315)
                airis.move_forward(rob)

            case 'jump 0':
                airis.lookDir(rob, 0, 0)
                airis.jump_forward(rob)

            case 'jump 45':
                airis.lookDir(rob, 0, 45)
                airis.jump_forward(rob)

            case 'jump 90':
                airis.lookDir(rob, 0, 90)
                airis.jump_forward(rob)

            case 'jump 135':
                airis.lookDir(rob, 0, 135)
                airis.jump_forward(rob)

            case 'jump 180':
                airis.lookDir(rob, 0, 180)
                airis.jump_forward(rob)

            case 'jump 225':
                airis.lookDir(rob, 0, 225)
                airis.jump_forward(rob)

            case 'jump 270':
                airis.lookDir(rob, 0, 270)
                airis.jump_forward(rob)

            case 'jump 315':
                airis.lookDir(rob, 0, 315)
                airis.jump_forward(rob)

            case 'mine up 0':
                airis.lookDir(rob, -60, 0)
                airis.mine(rob)
                airis.lookDir(rob, 0, 0)

            case 'mine up 45':
                airis.lookDir(rob, -60, 45)
                airis.mine(rob)
                airis.lookDir(rob, 0, 45)

            case 'mine up 90':
                airis.lookDir(rob, -60, 90)
                airis.mine(rob)
                airis.lookDir(rob, 0, 90)

            case 'mine up 135':
                airis.lookDir(rob, -60, 135)
                airis.mine(rob)
                airis.lookDir(rob, 0, 135)

            case 'mine up 180':
                airis.lookDir(rob, -60, 180)
                airis.mine(rob)
                airis.lookDir(rob, 0, 180)

            case 'mine up 225':
                airis.lookDir(rob, -60, 225)
                airis.mine(rob)
                airis.lookDir(rob, 0, 225)

            case 'mine up 270':
                airis.lookDir(rob, -60, 270)
                airis.mine(rob)
                airis.lookDir(rob, 0, 270)

            case 'mine up 315':
                airis.lookDir(rob, -60, 315)
                airis.mine(rob)
                airis.lookDir(rob, 0, 315)

            case 'mine down 0':
                airis.lookDir(rob, 60, 0)
                airis.mine(rob)
                airis.lookDir(rob, 0, 0)

            case 'mine down 45':
                airis.lookDir(rob, 60, 45)
                airis.mine(rob)
                airis.lookDir(rob, 0, 45)

            case 'mine down 90':
                airis.lookDir(rob, 60, 90)
                airis.mine(rob)
                airis.lookDir(rob, 0, 90)

            case 'mine down 135':
                airis.lookDir(rob, 60, 135)
                airis.mine(rob)
                airis.lookDir(rob, 0, 135)

            case 'mine down 180':
                airis.lookDir(rob, 60, 180)
                airis.mine(rob)
                airis.lookDir(rob, 0, 180)

            case 'mine down 225':
                airis.lookDir(rob, 60, 225)
                airis.mine(rob)
                airis.lookDir(rob, 0, 225)

            case 'mine down 270':
                airis.lookDir(rob, 60, 270)
                airis.mine(rob)
                airis.lookDir(rob, 0, 270)

            case 'mine down 315':
                airis.lookDir(rob, 60, 315)
                airis.mine(rob)
                airis.lookDir(rob, 0, 315)

            case 'mine straight 0':
                airis.lookDir(rob, 0, 0)
                airis.mine(rob)
                airis.lookDir(rob, 0, 0)

            case 'mine straight 45':
                airis.lookDir(rob, 0, 45)
                airis.mine(rob)
                airis.lookDir(rob, 0, 45)

            case 'mine straight 90':
                airis.lookDir(rob, 0, 90)
                airis.mine(rob)
                airis.lookDir(rob, 0, 90)

            case 'mine straight 135':
                airis.lookDir(rob, 0, 135)
                airis.mine(rob)
                airis.lookDir(rob, 0, 135)

            case 'mine straight 180':
                airis.lookDir(rob, 0, 180)
                airis.mine(rob)
                airis.lookDir(rob, 0, 180)

            case 'mine straight 225':
                airis.lookDir(rob, 0, 225)
                airis.mine(rob)
                airis.lookDir(rob, 0, 225)

            case 'mine straight 270':
                airis.lookDir(rob, 0, 270)
                airis.mine(rob)
                airis.lookDir(rob, 0, 270)

            case 'mine straight 315':
                airis.lookDir(rob, 0, 315)
                airis.mine(rob)
                airis.lookDir(rob, 0, 315)

        stats = [mc.getFullStat(key) for key in fullStatKeys]
        stats = [math.floor(stats[0]), math.floor(stats[1]), math.floor(stats[2]), round(stats[3]), 0]  # round(stats[4]) % 360]
        grid = mc.getNearGrid()
        airis.capture_input(stats, grid, action, state, False)
        print('Current Stats', stats)
        airis.save_knowledge('Knowledge.npy')

    print('Test Routine Complete')
