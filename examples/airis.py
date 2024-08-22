import logging
import math
import heapq
from time import sleep, time
from tagilmo.utils.vereya_wrapper import MCConnector, RobustObserver
import tagilmo.utils.mission_builder as mb
import numpy as np
import uuid
import os, sys

from copy import copy, deepcopy
from tagilmo.utils.mathutils import normAngle, degree2rad


class State:
    def __init__(self, pos_input, grid_input):
        self.pos_input = pos_input
        self.grid_input = grid_input
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
        # self.incoming_edges = dict()
        self.outgoing_edges = dict()
        self.state_hash = hash((tuple(self.pos_input[0]), tuple(self.grid_input)))


class Airis:

    def __init__(self):
        self.knowledge = dict()
        self.grid_input = []
        self.pos_input = []
        self.inventory_input = dict()
        self.hotbar_input = dict()
        self.mission_input = []
        self.states = None
        self.state_graph = dict()
        self.prediction_state_graph = dict()
        self.action_plan = []
        self.last_action = None
        # self.given_goal = None
        self.given_goal = ((0, 65, 0, None, None), ())
        self.current_goal = self.given_goal
        self.goal_achieved = False
        self.applied_rules_pos = None
        self.applied_rules_grid = None
        self.all_rules_pos = None
        self.all_rules_grid = None
        self.bad_predictions_pos = None
        self.bad_predictions_grid = None
        self.time_step = 0
        self.state_history = set()
        self.prediction_flex = 1
        self.last_compare = None
        self.best_compare = None
        self.true_compare = None
        self.grid_map = np.empty((256, 500, 500), dtype=np.dtype('U42'))
        self.grid_origin_x = 2
        self.grid_origin_y = 2
        self.grid_origin_z = 2
        self.map_origin_x = 250
        self.map_origin_y = 128
        self.map_origin_z = 250
        self.debug_dict = dict()
        self.explored_states = set()
        self.grid_output = dict()
        self.state_output = dict()
        self.edges_output = dict()
        
        self.actions = ['move 0', 'move 45', 'move 90', 'move 135', 'move 180', 'move 225', 'move 270', 'move 315',
                        'jump 0', 'jump 45', 'jump 90', 'jump 135', 'jump 180', 'jump 225', 'jump 270', 'jump 315']
        # 'mine up 0', 'mine up 45', 'mine up 90', 'mine up 135', 'mine up 180', 'mine up 225', 'mine up 270', 'mine up 315',
        # 'mine straight 0', 'mine straight 45', 'mine straight 90', 'mine straight 135', 'mine straight 180', 'mine straight 225', 'mine straight 270', 'mine straight 315']
        # 'mine down 0', 'mine down 45', 'mine down 90', 'mine down 135', 'mine down 180', 'mine down 225', 'mine down 270', 'mine down 315',

    def capture_input(self, pos_input, grid_input, action, state, pre, confidence, applied_rules):
        if pre:
            self.pos_input = np.asarray([(math.floor(pos_input[0]), math.floor(pos_input[1]), math.floor(pos_input[2]))])
            self.grid_input = np.asarray(grid_input, dtype=np.dtype('U42'))

            grid_input_3d = self.grid_input.copy()
            grid_input_3d = grid_input_3d.reshape((5, 5, 5))

            # print('Position', self.pos_input)
            # print('Grid Calculation: x', self.pos_input[0][0] - self.grid_origin_x, self.pos_input[0][0] + self.grid_origin_x + 1)
            # print('Grid Calculation: y', self.pos_input[0][1] - self.grid_origin_y, self.pos_input[0][1] + self.grid_origin_y + 1)
            # print('Grid Calculation: z', self.pos_input[0][2] - self.grid_origin_z, self.pos_input[0][2] + self.grid_origin_z + 1)
            # print('Input', grid_input_3d)
            # print('Map pre', pre)

            hold = deepcopy(self.grid_map[self.map_origin_y + self.pos_input[0][1] - self.grid_origin_y:self.map_origin_y + self.pos_input[0][1] + self.grid_origin_y + 1, self.map_origin_z + self.pos_input[0][2] - self.grid_origin_z:self.map_origin_z + self.pos_input[0][2] + self.grid_origin_z + 1, self.map_origin_x + self.pos_input[0][0] - self.grid_origin_x:self.map_origin_x + self.pos_input[0][0] + self.grid_origin_x + 1])
            hold = hold.flatten()
            # map_mismatch = [i for i, v in enumerate(hold) if v != self.grid_input[i] and v != '']
            # if map_mismatch:
            #     for index in map_mismatch:
            #         print('Map Mismatch!')
            #         print('Index', index)
            #         print('Actual', self.grid_input[index])
            #         print('Map', hold[index])
            #     raise Exception

            self.grid_map[self.map_origin_y + self.pos_input[0][1] - self.grid_origin_y:self.map_origin_y + self.pos_input[0][1] + self.grid_origin_y + 1, self.map_origin_z + self.pos_input[0][2] - self.grid_origin_z:self.map_origin_z + self.pos_input[0][2] + self.grid_origin_z + 1, self.map_origin_x + self.pos_input[0][0] - self.grid_origin_x:self.map_origin_x + self.pos_input[0][0] + self.grid_origin_x + 1] = grid_input_3d

            # print('Map Post', post)

            # post = post.flatten()
            # pre = pre.flatten()
            #
            # for i, v in enumerate(pre):
            #     if v != '' and v != post[i]:
            #         print('Grid was', v, 'but now its', post[i])
            #         raise Exception

            if not self.action_plan:
                for key in self.state_output.keys():
                    self.state_output[key][3] = 3
                self.edges_output = dict()
                prior_state = State(self.pos_input, self.grid_input)
                self.states = []
                try:
                    prior_state = self.state_graph[(self.pos_input[0][0], self.pos_input[0][1], self.pos_input[0][2])]
                    prior_state.pos_input = self.pos_input
                    prior_state.grid_input = self.grid_input
                    if prior_state.pos_input[0][0] != self.pos_input[0][0] or prior_state.pos_input[0][1] != self.pos_input[0][1] or prior_state.pos_input[0][2] != self.pos_input[0][2]:
                        print('Break point 1', prior_state, prior_state.pos_input, self.pos_input)
                        raise Exception
                    heapq.heappush(self.states, (self.compare(self.pos_input, self.grid_input), 0, self.state_graph[(self.pos_input[0][0], self.pos_input[0][1], self.pos_input[0][2])], None, prior_state))
                except KeyError:
                    self.state_graph[(self.pos_input[0][0], self.pos_input[0][1], self.pos_input[0][2])] = prior_state
                    if prior_state.pos_input[0][0] != self.pos_input[0][0] or prior_state.pos_input[0][1] != self.pos_input[0][1] or prior_state.pos_input[0][2] != self.pos_input[0][2]:
                        print('Break point 2', prior_state, prior_state.pos_input, self.pos_input)
                        raise Exception
                    heapq.heappush(self.states, (self.compare(self.pos_input, self.grid_input), 0, self.state_graph[(self.pos_input[0][0], self.pos_input[0][1], self.pos_input[0][2])], None, prior_state))

                self.explored_states.add(tuple(self.pos_input[0]))
                # print('State Graph')
                # for key in self.state_graph.keys():
                #     print(key, self.state_graph[key], self.state_graph[key].pos_input, '--------------------------------------------------------')
                #     print('Edges')
                #     for edge in self.state_graph[key].outgoing_edges.keys():
                #         print('Action: ', edge)
                #         for item in self.state_graph[key].outgoing_edges[edge]:
                #             print(item)

            if self.current_goal:
                if (self.pos_input[0][0], self.pos_input[0][2]) == (self.current_goal[0][0], self.current_goal[0][2]) and self.pos_input[0][1] >= self.current_goal[0][1] - 1:
                    if np.all(self.current_goal == self.given_goal):
                        # self.goal_achieved = True
                        rob.sendCommand('chat /tp 206 64 119')
                        rob.sendCommand('chat Goal Reached! Resetting to initial position...')
                        self.last_compare = None
                        sleep(1)
                        self.state_history = set()
                    else:
                        self.current_goal = self.given_goal
                        self.last_compare = None

                print('Planning for goal', self.current_goal)
                if not self.goal_achieved:
                    if not self.action_plan:
                        self.make_plan(prior_state)
                        # print('Prediction State Graph')
                        # for key in self.prediction_state_graph.keys():
                        #     print(key, self.prediction_state_graph[key], self.prediction_state_graph[key].pos_input, '--------------------------------------------------------')
                        #     print('Edges')
                        #     for edge in self.prediction_state_graph[key].outgoing_edges.keys():
                        #         print('Action: ', edge)
                        #         for item in self.prediction_state_graph[key].outgoing_edges[edge]:
                        #             print(item)
                        if self.action_plan:
                            # print('Action Plan:', self.action_plan)
                            for yi, y in enumerate(grid_input_3d):
                                for zi, z in enumerate(grid_input_3d[yi]):
                                    for xi, x in enumerate(grid_input_3d[yi][zi]):
                                        if x != 'air':
                                            self.grid_output[(self.pos_input[0][1] + yi - self.grid_origin_y, self.pos_input[0][2] + zi - self.grid_origin_z, self.pos_input[0][0] + xi - self.grid_origin_x)] = (self.pos_input[0][0] + xi - self.grid_origin_x, self.pos_input[0][1] + yi - self.grid_origin_y, self.pos_input[0][2] + zi - self.grid_origin_z, x)

                            self.state_output[tuple(self.pos_input[0])] = [self.pos_input[0][0], self.pos_input[0][1], self.pos_input[0][2], 1]

                            np.save('output/state_output_temp.npy', self.state_output)
                            try:
                                os.replace('output/state_output_temp.npy', 'output/state_output.npy')
                            except PermissionError:
                                pass

                            np.save('output/edge_output_temp.npy', self.edges_output)
                            try:
                                os.replace('output/edge_output_temp.npy', 'output/edge_output.npy')
                            except PermissionError:
                                pass

                            np.save('output/grid_output_temp.npy', self.grid_output)
                            try:
                                os.replace('output/grid_output_temp.npy', 'output/grid_output.npy')
                            except PermissionError:
                                pass
                            print('Action Plan Length:', len(self.action_plan))
                            return self.action_plan.pop(0)
                    else:
                        # print('Action Plan:', self.action_plan)
                        for yi, y in enumerate(grid_input_3d):
                            for zi, z in enumerate(grid_input_3d[yi]):
                                for xi, x in enumerate(grid_input_3d[yi][zi]):
                                    if x != 'air':
                                        self.grid_output[(self.pos_input[0][1] + yi - self.grid_origin_y, self.pos_input[0][2] + zi - self.grid_origin_z, self.pos_input[0][0] + xi - self.grid_origin_x)] = (self.pos_input[0][0] + xi - self.grid_origin_x, self.pos_input[0][1] + yi - self.grid_origin_y, self.pos_input[0][2] + zi - self.grid_origin_z, x)

                        self.state_output[tuple(self.pos_input[0])] = [self.pos_input[0][0], self.pos_input[0][1], self.pos_input[0][2], 1]

                        np.save('output/state_output_temp.npy', self.state_output)
                        try:
                            os.replace('output/state_output_temp.npy', 'output/state_output.npy')
                        except PermissionError:
                            pass

                        np.save('output/edge_output_temp.npy', self.edges_output)
                        try:
                            os.replace('output/edge_output_temp.npy', 'output/edge_output.npy')
                        except PermissionError:
                            pass

                        np.save('output/grid_output_temp.npy', self.grid_output)
                        try:
                            os.replace('output/grid_output_temp.npy', 'output/grid_output.npy')
                        except PermissionError:
                            pass
                        print('Action Plan Length:', len(self.action_plan))
                        return self.action_plan.pop(0)


        else:
            new_pos_input = np.asarray([(math.floor(pos_input[0]), math.floor(pos_input[1]), math.floor(pos_input[2]))])
            new_grid_input = np.asarray(grid_input, dtype=np.dtype('U42'))

            new_grid_3d = new_grid_input.copy()
            new_grid_3d = new_grid_3d.reshape((5, 5, 5))
            # print('new pos', new_pos_input)

            hold = deepcopy(self.grid_map[self.map_origin_y + new_pos_input[0][1] - self.grid_origin_y:self.map_origin_y + new_pos_input[0][1] + self.grid_origin_y + 1, self.map_origin_z + new_pos_input[0][2] - self.grid_origin_z:self.map_origin_z + new_pos_input[0][2] + self.grid_origin_z + 1, self.map_origin_x + new_pos_input[0][0] - self.grid_origin_x:self.map_origin_x + new_pos_input[0][0] + self.grid_origin_x + 1])
            hold = hold.flatten()
            # map_mismatch = [i for i, v in enumerate(hold) if v != new_grid_input[i] and v != '']
            # if map_mismatch:
            #     for index in map_mismatch:
            #         print('Map Mismatch!')
            #         print('Index', index)
            #         print('Actual', self.grid_input[index])
            #         print('Map', hold[index])
            #         raise Exception

            try:
                self.state_output[tuple(self.pos_input[0])][3] = 5
            except KeyError:
                pass

            try:
                self.edges_output[(tuple(self.pos_input[0]), tuple(new_pos_input[0]))][6] = 5
            except KeyError:
                pass

            self.grid_map[self.map_origin_y + new_pos_input[0][1] - self.grid_origin_y:self.map_origin_y + new_pos_input[0][1] + self.grid_origin_y + 1, self.map_origin_z + new_pos_input[0][2] - self.grid_origin_z:self.map_origin_z + new_pos_input[0][2] + self.grid_origin_z + 1, self.map_origin_x + new_pos_input[0][0] - self.grid_origin_x:self.map_origin_x + new_pos_input[0][0] + self.grid_origin_x + 1] = new_grid_3d

            post_state = State(new_pos_input, new_grid_input)
            self.states = []
            try:
                post_state = self.state_graph[(new_pos_input[0][0], new_pos_input[0][1], new_pos_input[0][2])]
                post_state.pos_input = new_pos_input
                post_state.grid_input = new_grid_input
            except KeyError:
                self.state_graph[(new_pos_input[0][0], new_pos_input[0][1], new_pos_input[0][2])] = post_state

            self.explored_states.add(tuple(new_pos_input[0]))

            clear_plan = False
            self.last_action = action
            self.bad_predictions_pos = dict()
            self.bad_predictions_grid = dict()

            if confidence == 0:
                # create no confidence rules
                for i, v in enumerate(self.pos_input):
                    try:
                        print('deleting 0 confidence applied_rule')
                        print(applied_rules[0][i])
                        del applied_rules[0][i]
                    except KeyError:
                        pass
                    self.create_rule(action, 'Pos', i, self.pos_input[i], new_pos_input[i])
                # for i, v in enumerate(self.grid_input):
                #     self.create_rule(action, 'Grid', i, self.grid_input[i], new_grid_input[i])

            pos_mismatch = [i for i, v in enumerate(state.pos_input) if not np.all(v == new_pos_input[i])]
            grid_mismatch = [i for i, v in enumerate(state.grid_input) if v != new_grid_input[i]]

            self.applied_rules_pos = deepcopy(applied_rules[0])
            self.applied_rules_grid = deepcopy(applied_rules[1])
            self.all_rules_pos = deepcopy(applied_rules[2])
            self.all_rules_grid = deepcopy(applied_rules[3])

            if pos_mismatch:
                clear_plan = True
                for index in pos_mismatch:
                    print('POS mismatch - ', index, self.pos_input[index], new_pos_input[index], state.pos_input[index])
                    try:
                        print('POS Prediction: ', self.applied_rules_pos[index])
                        # print('POS Predict Heap: ', state.debug_heap['Pos' + str(index)])
                        # self.bad_predictions_pos[index] = deepcopy(self.applied_rules_pos[index])
                        del self.applied_rules_pos[index]
                    except KeyError:
                        pass

                for index in pos_mismatch:
                    found = False
                    try:
                        while self.all_rules_pos[index]:
                            data = heapq.heappop(self.all_rules_pos[index])
                            # (diff_count, -age, rule, idx, new_val, 'Pos', total_len, updates, idx, act, differences)
                            print('Checking other rules', tuple(map(lambda old, change: old + change, self.pos_input[index], data[4])), new_pos_input[index])
                            if np.all(tuple(map(lambda old, change: old + change, self.pos_input[index], data[4])) == new_pos_input[index]):
                                print('Applying incorrect rule to other rule', data)
                                self.applied_rules_pos[index] = data
                                found = True
                                break
                    except KeyError:
                        pass

                    if not found:
                        self.create_rule(action, 'Pos', index, self.pos_input[index], new_pos_input[index])

            # elif grid_mismatch:
            #     clear_plan = True
            #     for index in grid_mismatch:
            #         print('GRID mismatch - ', index, self.grid_input[index], new_grid_input[index], self.states[state].grid_input[index])
            #         try:
            #             print('GRID Prediction: ', self.applied_rules_grid[index])
            #             print('GRID Predict Heap: ', self.states[state].debug_heap['Grid' + str(index)])
            #             self.bad_predictions_grid[index] = deepcopy(self.applied_rules_grid[index])
            #             del self.applied_rules_grid[index]
            #         except KeyError:
            #             pass
            #
            #     for index in grid_mismatch:
            #         found = False
            #         try:
            #             while self.states[state].all_rules_grid[index]:
            #                 data = heapq.heappop(self.states[state].all_rules_grid[index])
            #                 if data[3] == new_grid_input[index]:
            #                     found = True
            #                     self.applied_rules_pos[index] = data
            #                     break
            #         except KeyError:
            #             pass
            #
            #         if not found:
            #             self.create_rule(action, 'Grid', index, self.grid_input[index], new_grid_input[index])

            # (diff_count, rule, idx, new_val, 'Pos', pos_data[2] + grid_data[1], updates, age, idx)
            # self.debug_dict['Pos' + str(idx) + str(rule) + state] = (idx, rule, i, predict_state.pos_input[i]

            for index in self.applied_rules_pos.keys():
                print('pre-checking differences')
                for item in self.applied_rules_pos[index][10]:
                    if item[0] == 'Grid':
                        print('Predicted value for index', item[2], 'is', state.grid_input[item[2]], 'actual is', self.grid_input[item[2]], 'state is', post_state.grid_input[item[2]], 'predicted state', state, 'graph state', post_state)
                self.update_good_rule(self.applied_rules_pos[index])

            for index in self.applied_rules_grid.keys():
                self.update_good_rule(self.applied_rules_grid[index])

            # for key in self.states[state].debug_dict.keys():
            #     print('debug dict', key, self.states[state].debug_dict[key])

            # for index in self.bad_predictions_pos.keys():
            #     self.update_bad_rule(self.bad_predictions_pos[index])
            #
            # for index in self.bad_predictions_grid.keys():
            #     self.update_bad_rule(self.bad_predictions_grid[index])

            # print('Prediction State: ', self.states[state])

            print('Prediction Confidence: ', confidence)
            if clear_plan:
                print('Prediction Incorrect...')
                self.action_plan = []
            else:
                print('Prediction Correct!')

            # if self.last_compare is not None:
            #     if abs(self.compare(new_pos_input, new_grid_input) - self.last_compare[0]) > 20 and np.all(self.current_goal == self.given_goal):
            #         self.current_goal = ((self.last_compare[1][0][0], self.last_compare[1][0][1], self.last_compare[1][0][2], None, None), ())

            self.last_compare = (self.compare(self.pos_input, self.grid_input), self.pos_input, self.grid_input)

            print('State Goal Compare', state.compare)
            print('Actual Compare', self.last_compare[0])
            print('Current Goal', self.current_goal)

            self.time_step += 1
            # if not clear_plan:
            self.state_history.add(hash((tuple(self.pos_input[0]), tuple(self.grid_input), action)))

    def make_plan(self, original_state):
        self.action_plan = []
        goal_compare = self.compare(self.states[0][2].pos_input, self.states[0][2].grid_input)
        goal_reached = False
        at_goal = False
        if goal_compare == 0:
            goal_reached = True
            at_goal = True
        step = 0
        goal_heap = []
        check_states = set()
        check_states.add(self.states[0][2])
        path_heap = dict()
        goal_state = None
        heap_iter = 1
        debug_connected = set()
        debug_checked = set()
        debug_connected.add(self.states[0][2])
        print('pre-adding state to debug_connected', self.states[0][2])
        debug_new = False

        while not goal_reached:
            # Check for missing outgoing edges in current state
            if self.states:
                current_state = heapq.heappop(self.states)[2]
                debug_checked.add(current_state)
                print('adding current state to debug_checked', current_state)
            else:
                break

            try:
                self.state_output[tuple(current_state.pos_input[0])][3] = 2
            except KeyError:
                pass
            # print('current state pos info', current_state.pos_input, 'grid info', current_state.grid_input)
            goal_compare = self.compare(current_state.pos_input, current_state.grid_input)
            current_state.compare = goal_compare
            if goal_compare == 0:
                goal_reached = True
                goal_state = current_state
                break

            for act in self.actions:
                real_state = False
                try:
                    act_updates = self.knowledge['Action Updates'][act]
                except:
                    print('Action not yet tried, trying...')
                    new_state = self.predict(act, current_state)
                    goal_state = new_state
                    path_heap[goal_state] = [(step + 1, heap_iter, act, current_state)]
                    goal_reached = True
                    current_state.outgoing_edges[act] = [deepcopy(new_state.applied_rules_pos), deepcopy(new_state.applied_rules_grid), new_state.confidence, 0, new_state, deepcopy(new_state.all_rules_pos), deepcopy(new_state.all_rules_grid)]
                    debug_new = True
                    break
                update = False
                try:
                    check = current_state.outgoing_edges[act]
                    check[4].compare = self.compare(check[4].pos_input, check[4].grid_input)
                    if check[3] < act_updates:
                        update = True
                    if not update:
                        self.edges_output[(tuple(current_state.pos_input[0]), tuple(check[4].pos_input[0]))] = [current_state.pos_input[0][0], current_state.pos_input[0][1], current_state.pos_input[0][2], check[4].pos_input[0][0], check[4].pos_input[0][1], check[4].pos_input[0][2], 3]
                        if check[2] != 1:
                            self.edges_output[(tuple(current_state.pos_input[0]), tuple(check[4].pos_input[0]))][6] = 0

                        if check[2] == 1:
                            debug_connected.add(check[4])
                            print('adding to debug_connected', check[4])
                            if check[4] not in check_states:
                                print('state not in check_states, adding to state heap', check[4])
                                heapq.heappush(self.states, (check[4].compare + step + 1, heap_iter, check[4], act, current_state))
                                check_states.add(check[4])
                                heap_iter += 1
                        else:
                            if check[4] not in check_states:
                                heapq.heappush(goal_heap, (check[4].compare, check[2], step + 1, heap_iter, check[4], act, current_state))
                                heap_iter += 1

                        try:
                            heapq.heappush(path_heap[check[4]], (step + 1, heap_iter, act, current_state))
                            heap_iter += 1
                        except KeyError:
                            path_heap[check[4]] = [(step + 1, heap_iter, act, current_state)]
                            heap_iter += 1

                        # print('Up-to-date Edge found:', current_state, act, check[4], check)
                except KeyError:
                    update = True

                if update:
                    new_state = self.predict(act, current_state)
                    target_state = new_state
                    if (new_state.pos_input[0][0], new_state.pos_input[0][1], new_state.pos_input[0][2]) not in self.state_graph.keys():
                        if (new_state.pos_input[0][0], new_state.pos_input[0][1], new_state.pos_input[0][2]) in self.explored_states:
                            print('Predicted state not in state graph, but it should be!')
                            print('pos', new_state.pos_input)
                            print('===============================')
                            print('Predicted Grid', new_state.grid_input)
                            print('+++++++++++++++++++++++++++++++')
                            for item in self.state_graph.keys():
                                if item[0] == new_state.pos_input[0][0] and item[1] == new_state.pos_input[0][1] and item[2] == new_state.pos_input[0][2]:
                                    print('State graph Hash', item[3])
                                    print('State graph Grid', self.state_graph[item].grid_input)

                            raise Exception
                        self.state_graph[(new_state.pos_input[0][0], new_state.pos_input[0][1], new_state.pos_input[0][2])] = new_state
                        # print('predicted state not in graph, adding', new_state, new_state.pos_input, 'to key', (new_state.pos_input[0][0], new_state.pos_input[0][1], new_state.pos_input[0][2]))
                    else:
                        target_state = self.state_graph[(new_state.pos_input[0][0], new_state.pos_input[0][1], new_state.pos_input[0][2])]
                        if target_state.pos_input[0][0] != new_state.pos_input[0][0] or target_state.pos_input[0][1] != new_state.pos_input[0][1] or target_state.pos_input[0][2] != new_state.pos_input[0][2]:
                            print('Break point 3', target_state, target_state.pos_input, new_state.pos_input)
                            print('Error in key', (new_state.pos_input[0][0], new_state.pos_input[0][1], new_state.pos_input[0][2]))
                            raise Exception
                        # print('predicted state in graph, replacing with existing', target_state, target_state.pos_input)

                    target_state.compare = self.compare(target_state.pos_input, target_state.grid_input)

                    current_state.outgoing_edges[act] = [deepcopy(new_state.applied_rules_pos), deepcopy(new_state.applied_rules_grid), new_state.confidence, act_updates, target_state, deepcopy(new_state.all_rules_pos), deepcopy(new_state.all_rules_grid)]

                    self.edges_output[(tuple(current_state.pos_input[0]), tuple(target_state.pos_input[0]))] = [current_state.pos_input[0][0], current_state.pos_input[0][1], current_state.pos_input[0][2], target_state.pos_input[0][0], target_state.pos_input[0][1], target_state.pos_input[0][2], 3]
                    if new_state.confidence != 1:
                        self.edges_output[(tuple(current_state.pos_input[0]), tuple(target_state.pos_input[0]))][6] = 0
                    # try:
                    #     target_state.incoming_edges[act].append(current_state)
                    # except KeyError:
                    #     target_state.incoming_edges[act] = [current_state]

                    if new_state.confidence == 1:
                        debug_connected.add(target_state)
                        print('adding to debug_connected', target_state)
                        if target_state not in check_states:
                            print('state not in check_states, adding to state heap', target_state)
                            heapq.heappush(self.states, (target_state.compare + step + 1, heap_iter, target_state, act, current_state))
                            check_states.add(target_state)
                            heap_iter += 1
                    else:
                        if target_state not in check_states:
                            heapq.heappush(goal_heap, (target_state.compare, new_state.confidence, step + 1, heap_iter, target_state, act, current_state))
                            heap_iter += 1

                    try:
                        heapq.heappush(path_heap[target_state], (step + 1, heap_iter, act, current_state))
                        heap_iter += 1
                    except KeyError:
                        path_heap[target_state] = [(step + 1, heap_iter, act, current_state)]
                        heap_iter += 1
                    # print('New Edge', current_state, act, target_state, current_state.outgoing_edges[act][2])


        # if (60, 62, 16) in self.explored_states or (60, 63 , 16) in self.explored_states:
        #     if self.pos_input[0][1] < 62:
        #         find = False
        #         find1 = 'Have not yet explored 60 62 16'
        #         find2 = 'Have not yet explored 60 63 16'
        #         if (60, 62, 16) in self.explored_states:
        #             find1 = 'Explored 60 62 16'
        #             for key in self.prediction_state_graph.keys():
        #                 if key[0] == 60 and key[1] == 62 and key[2] == 16:
        #                     error_state = self.prediction_state_graph[key]
        #                     if error_state not in check_states and len(check_states) > 5:
        #                         find1 = 'Cannot find 60 62 16'
        #                     else:
        #                         find = True
        #                         find1 = 'Found 60 62 16'
        #                         break
        #
        #         if (60, 63, 16) in self.explored_states:
        #             find2 = 'Explored 60 63 16'
        #             for key in self.prediction_state_graph.keys():
        #                 if key[0] == 60 and key[1] == 63 and key[2] == 16:
        #                     error_state = self.prediction_state_graph[key]
        #                     if error_state not in check_states and len(check_states) > 5:
        #                         find2 = 'Cannot find 60 63 16'
        #                     else:
        #                         find = True
        #                         find2 = 'Found 60 63 16'
        #                         break
        #
        #         if not find:
        #             for item in check_states:
        #                 print('State Pos', item.pos_input[0], 'state', item)
        #                 for edge_act in item.outgoing_edges.keys():
        #                     print('-', edge_act)
        #                     for i, data in enumerate(item.outgoing_edges[edge_act]):
        #                         print('--', i, '|', data)
        #                         if i == 4:
        #                             if (data.pos_input[0][0], data.pos_input[0][1], data.pos_input[0][2]) in self.explored_states:
        #                                 print('--- Target state in explored states', data.pos_input[0][0], data.pos_input[0][1], data.pos_input[0][2])
        #             print(find1)
        #             print(find2)
        #             raise Exception

        if not goal_reached:
            goal_state = goal_heap[0][4]
            while goal_state == original_state:
                heapq.heappop(goal_heap)
                goal_state = goal_heap[0][4]
            self.action_plan.insert(0, (goal_heap[0][5], goal_heap[0][4], goal_heap[0][1], (goal_heap[0][6].outgoing_edges[goal_heap[0][5]][0], goal_heap[0][6].outgoing_edges[goal_heap[0][5]][1], goal_heap[0][6].outgoing_edges[goal_heap[0][5]][5], goal_heap[0][6].outgoing_edges[goal_heap[0][5]][6])))
            try:
                self.state_output[tuple(goal_state.pos_input[0])][3] = 4
            except KeyError:
                pass
            self.edges_output[(tuple(goal_heap[0][6].pos_input[0]), tuple(goal_state.pos_input[0]))] = [goal_heap[0][6].pos_input[0][0], goal_heap[0][6].pos_input[0][1], goal_heap[0][6].pos_input[0][2], goal_state.pos_input[0][0], goal_state.pos_input[0][1], goal_state.pos_input[0][2], 1]
            goal_state = goal_heap[0][6]

        if not at_goal:
            while goal_state != original_state:
                #Action Plan (Action, resulting state, confidence, (applied_rules pos, grid, all pos, all grid) )
                print('Adding Goal State to action plan', goal_state)
                # print('Previous State', path_heap[goal_state][0][3])
                print('Original State', original_state)
                self.action_plan.insert(0, (path_heap[goal_state][0][2], goal_state, path_heap[goal_state][0][3].outgoing_edges[path_heap[goal_state][0][2]][2], (path_heap[goal_state][0][3].outgoing_edges[path_heap[goal_state][0][2]][0], path_heap[goal_state][0][3].outgoing_edges[path_heap[goal_state][0][2]][1], path_heap[goal_state][0][3].outgoing_edges[path_heap[goal_state][0][2]][5], path_heap[goal_state][0][3].outgoing_edges[path_heap[goal_state][0][2]][6])))
                try:
                    self.state_output[tuple(goal_state.pos_input[0])][3] = 0
                except KeyError:
                    pass
                try:
                    if path_heap[goal_state][0][3].confidence == 1:
                        self.edges_output[(tuple(path_heap[goal_state][0][3].pos_input[0]), tuple(goal_state.pos_input[0]))][6] = 4 #(path_heap[goal_state][0][3].pos_input[0][0], path_heap[goal_state][0][3].pos_input[0][1], path_heap[goal_state][0][3].pos_input[0][2], goal_state.pos_input[0][0], goal_state.pos_input[0][1], goal_state.pos_input[0][2], 4)
                    else:
                        self.edges_output[(tuple(path_heap[goal_state][0][3].pos_input[0]), tuple(goal_state.pos_input[0]))][6] = 1
                except KeyError:
                    pass
                goal_state = path_heap[goal_state][0][3]
                print('New goal state', goal_state)

        print('Number of persistent states', len(self.state_graph.keys()))

    # def make_plan(self):
    #     current_state = 0
    #     self.action_plan = []
    #     goal_compare = self.compare(self.states[0].pos_input, self.states[0].grid_input)
    #     # goal_heap: Compare, State Index, Confidence
    #     goal_heap = []
    #     compare_heap = []
    #     goal_state = 0
    #     # confidence_heap: Confidence, State Index, Compare
    #     most_confidence_heap = []
    #     least_confidence_heap = []
    #     state_hash_set = set()
    #     goal_reached = False
    #     if goal_compare == 0:
    #         goal_reached = True
    #     step = 0
    #
    #     while not goal_reached:
    #         fresh_state = False
    #         step += 1
    #         for act in self.actions:
    #             try:
    #                 check = self.knowledge['Action Rules'][act]
    #             except KeyError:
    #                 self.states.append(self.predict(act, 0))
    #                 goal_reached = True
    #                 goal_state = len(self.states) - 1
    #                 break
    #
    #             new_state = self.predict(act, current_state)
    #             state_hash = hash((tuple(self.states[current_state].pos_input[0]), tuple(self.states[current_state].grid_input), act))
    #             if state_hash not in state_hash_set:
    #                 self.states.append(new_state)
    #                 state_idx = len(self.states) - 1
    #                 goal_compare = self.compare(self.states[state_idx].pos_input, self.states[state_idx].grid_input)
    #                 self.states[state_idx].compare = goal_compare
    #                 state_confidence = self.states[state_idx].confidence
    #                 if state_hash not in self.state_history:
    #                     heapq.heappush(goal_heap, (goal_compare, state_idx, state_confidence, act, state_hash))
    #                 heapq.heappush(compare_heap, (goal_compare, state_idx, state_confidence, act, state_hash))
    #                 heapq.heappush(most_confidence_heap, (-state_confidence, state_idx, goal_compare, act, state_hash))
    #                 heapq.heappush(least_confidence_heap, (state_confidence, state_idx, goal_compare, act, state_hash))
    #                 state_hash_set.add(state_hash)
    #                 fresh_state = True
    #
    #         if compare_heap and not goal_reached:
    #             if compare_heap[0][0] == 0:
    #                 goal_reached = True
    #                 goal_state = compare_heap[0][1]
    #                 print('Predicted state reaches goal! Trying...')
    #                 print('Goal State', goal_state)
    #                 break
    #
    #             if most_confidence_heap[0][0] == -1:
    #                 current_state = most_confidence_heap[0][1]
    #                 heapq.heappop(most_confidence_heap)
    #             else:
    #                 goal_reached = True
    #                 goal_found = False
    #
    #                 if goal_heap:
    #                     while goal_heap[0][2] == 1:
    #                         heapq.heappop(goal_heap)
    #                     if goal_heap:
    #                         goal_state = goal_heap[0][1]
    #                         goal_found = True
    #
    #                 if not goal_found:
    #                     goal_state = least_confidence_heap[0][1]
    #                     print('goal state set to overall least confident prediction')
    #
    #                 print('goal state is', goal_state, 'with a confidence of', self.states[goal_state].confidence, self.states[goal_state].confidence_count, '/', self.states[goal_state].confidence_total)
    #
    #     plan_state = goal_state
    #     self.action_plan.insert(0, (self.states[plan_state].action, plan_state))
    #     plan_state = self.states[plan_state].prev_state
    #     while plan_state > 0:
    #         self.action_plan.insert(0, (self.states[plan_state].action, plan_state))
    #         plan_state = self.states[plan_state].prev_state
    #         pass

    def compare(self, pos_input, grid_input):
        compare_total = 0
        # Compare current position to goal position
        # self.given_goal = ((0, 65, 0, None, None),())
        for i, c_val in enumerate(self.current_goal[0]):
            if c_val is not None:
                if i == 1:
                    if pos_input[0][i] < c_val - 1:
                        compare_total += abs(c_val - pos_input[0][i]) * 10
                    else:
                        compare_total += abs(c_val - pos_input[0][i])
                else:
                    compare_total += abs(c_val - pos_input[0][i])

        return compare_total

    def predict(self, act, base_state):
        confidence_count = 0
        confidence_total = 0
        predict_heap = dict()
        self.debug_dict = dict()

        predict_state = State(copy(base_state.pos_input), copy(base_state.grid_input))

        # Predict pos conditions and heapify the output to guide more efficient grid checking. TO DO Check the speed of multiprocessing pooling vs current linear

        # print('predicting state', act, predict_state, 'from base state', base_state)
        # print('base state pos', base_state.pos_input)

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
                differences = []
                diff_count = 0
                total_len = 0
                # POS
                condition_set = self.knowledge['Pos-' + str(idx) + '/' + str(o_val) + '/' + str(rule) + '/Pos Conditions Set']
                total_len += len(condition_set)
                for i in condition_set:
                    if predict_state.pos_input[0][i] != self.knowledge['Pos-' + str(idx) + '/' + str(o_val) + '/' + str(rule) + '/Pos Conditions'][0][i]:
                        diff_count += 1
                        differences.append(('Pos', predict_state.pos_input[0][i], self.knowledge['Pos-' + str(idx) + '/' + str(o_val) + '/' + str(rule) + '/Pos Conditions'][0][i]))
                    # if not np.all(predict_state.pos_input[i] == self.knowledge['Pos-' + str(idx) + '/' + str(o_val) + '/' + str(rule) + '/Pos Conditions'][i]):
                    #     diff_count += 1

                if best_diff is not None:
                    if diff_count > best_diff:
                        continue

                # GRID
                # condition_set = self.knowledge['Grid-' + str(idx) + '/' + str(o_val) + '/' + str(rule) + '/Grid Conditions Set']
                # total_len += len(condition_set)
                # for i in condition_set:
                #     if predict_state.grid_input[i] != self.knowledge['Grid-' + str(idx) + '/' + str(o_val) + '/' + str(rule) + '/Grid Conditions'][i]:
                #         diff_count += 1

                total_len += len(predict_state.grid_input)
                for i, v in enumerate(predict_state.grid_input):
                    if v not in self.knowledge['Grid-' + str(idx) + '/' + str(o_val) + '/' + str(rule) + '/Positive Grid Conditions'][i]:
                        diff_count += 1
                        differences.append(('Grid', v, i, self.knowledge['Grid-' + str(idx) + '/' + str(o_val) + '/' + str(rule) + '/Positive Grid Conditions'][i]))

                if best_diff is not None:
                    if diff_count > best_diff:
                        continue

                updates = self.knowledge['Pos-' + str(idx) + '/' + str(idx) + '/' + str(rule) + '/Updates']
                new_val = self.knowledge['Pos-' + str(idx) + '/' + str(idx) + '/' + str(rule) + '/New Value']
                age = self.knowledge['Pos-' + str(idx) + '/' + str(idx) + '/' + str(rule) + '/Age']

                best_diff = diff_count

                if total_len == 0:
                    continue

                if diff_count == total_len:
                    continue

                # if predict_heap['Pos' + str(idx)]:
                #     if predict_heap['Pos' + str(idx)][0][0] == diff_count:
                #         if age > predict_heap['Pos' + str(idx)][0][7]:
                #             heapq.heapreplace(predict_heap['Pos' + str(idx)], (diff_count, -age, rule, idx, new_val, 'Pos', total_len, updates, idx, act, differences))
                #     else:
                #         heapq.heappush(predict_heap['Pos' + str(idx)], (diff_count, -age, rule, idx, new_val, 'Pos', total_len, updates, idx, act, differences))
                # else:
                heapq.heappush(predict_heap['Pos' + str(idx)], (diff_count, -age, rule, idx, new_val, 'Pos', total_len, updates, idx, act, differences))

                # if diff_count == 0:
                #     break

            if not predict_heap['Pos' + str(idx)]:
                predict_heap['Pos' + str(idx)] = None

        # Evaluate grid input
        # for idx, val in enumerate(predict_state.grid_input):
        #     predict_heap['Grid' + str(idx)] = []
        #     try:
        #         rules_list = self.knowledge['Grid-' + str(idx) + '/' + str(val) + '/Actions'][act]
        #     except KeyError:
        #         predict_heap['Grid' + str(idx)] = None
        #         continue
        #
        #     o_val = val
        #
        #     best_diff = None
        #
        #     for rule in rules_list[::-1]:
        #         diff_count = 0
        #         total_len = 0
        #         # POS
        #         condition_set = self.knowledge['Pos-' + str(idx) + '/' + str(o_val) + '/' + str(rule) + '/Pos Conditions Set']
        #         total_len += len(condition_set)
        #         for i in condition_set:
        #             if predict_state.pos_input[0][i] != self.knowledge['Pos-' + str(idx) + '/' + str(o_val) + '/' + str(rule) + '/Pos Conditions'][0][i]:
        #                 diff_count += 1
        #             # if not np.all(predict_state.pos_input[i] == self.knowledge['Pos-' + str(idx) + '/' + str(o_val) + '/' + str(rule) + '/Pos Conditions'][i]):
        #             #     diff_count += 1
        #
        #         if best_diff is not None:
        #             if diff_count > best_diff:
        #                 continue
        #
        #         # GRID
        #         # condition_set = self.knowledge['Grid-' + str(idx) + '/' + str(o_val) + '/' + str(rule) + '/Grid Conditions Set']
        #         # total_len += len(condition_set)
        #         # for i in condition_set:
        #         #     if predict_state.grid_input[i] != self.knowledge['Grid-' + str(idx) + '/' + str(o_val) + '/' + str(rule) + '/Grid Conditions'][i]:
        #         #         diff_count += 1
        #
        #         total_len += len(predict_state.grid_input)
        #         for i, v in enumerate(predict_state.grid_input):
        #             if v not in self.knowledge['Grid-' + str(idx) + '/' + str(o_val) + '/' + str(rule) + '/Positive Grid Conditions'][i]:
        #                 diff_count += 1
        #
        #         if best_diff is not None:
        #             if diff_count > best_diff:
        #                 continue
        #
        #         updates = self.knowledge['Grid-' + str(idx) + '/' + str(val) + '/' + str(rule) + '/Updates']
        #         new_val = self.knowledge['Grid-' + str(idx) + '/' + str(val) + '/' + str(rule) + '/New Value']
        #         age = self.knowledge['Grid-' + str(idx) + '/' + str(val) + '/' + str(rule) + '/Age']
        #
        #         best_diff = diff_count
        #
        #         if total_len == 0:
        #             continue
        #
        #         if diff_count == total_len:
        #             continue
        #
        #         if predict_heap['Grid' + str(idx)]:
        #             if predict_heap['Grid' + str(idx)][0][0] == diff_count:
        #                 if age > predict_heap['Grid' + str(idx)][0][7]:
        #                     heapq.heapreplace(predict_heap['Grid' + str(idx)], (diff_count, rule, idx, new_val, 'Grid', total_len, updates, age, val, act))
        #             else:
        #                 heapq.heappush(predict_heap['Grid' + str(idx)], (diff_count, rule, idx, new_val, 'Grid', total_len, updates, age, val, act))
        #         else:
        #             heapq.heappush(predict_heap['Grid' + str(idx)], (diff_count, rule, idx, new_val, 'Grid', total_len, updates, age, val, act))
        #
        #         # if diff_count == 0:
        #         #     break
        #
        #     if not predict_heap['Grid' + str(idx)]:
        #         predict_heap['Grid' + str(idx)] = None

        grid_changes = []

        # Apply changes to predict state
        # (diff_count, -age, rule, idx, new_val, 'Pos', total_len, updates, age, idx, act, differences)
        for idx_key in predict_heap.keys():
            if predict_heap[idx_key] is not None:
                print('Predict heap for ', idx_key, predict_heap[idx_key])
                if predict_heap[idx_key][0][5] == 'Pos':
                    predict_state.pos_input[predict_heap[idx_key][0][3]] = tuple(map(lambda old, change: old + change, predict_state.pos_input[predict_heap[idx_key][0][3]], predict_heap[idx_key][0][4]))
                    # if predict_heap[idx_key][0][2] <= 2:
                    #     predict_state.pos_input[predict_heap[idx_key][0][2]] += predict_heap[idx_key][0][3]
                    # else:
                    #     predict_state.pos_input[predict_heap[idx_key][0][2]] = predict_heap[idx_key][0][3]
                    predict_state.applied_rules_pos[predict_heap[idx_key][0][3]] = predict_heap[idx_key][0]
                    predict_state.all_rules_pos[predict_heap[idx_key][0][3]] = predict_heap[idx_key]
                    # if predict_heap[idx_key][0][0] != 0:
                    #     try:
                    #         print('Best Rule POS condition difference for state', predict_state, self.debug_dict['Pos'+str(predict_heap[idx_key][0][2])+str(predict_heap[idx_key][0][1])])
                    #     except KeyError:
                    #         pass
                elif predict_heap[idx_key][0][5] == 'Grid':
                    grid_changes.append((predict_heap[idx_key][0][3], predict_heap[idx_key][0][4]))
                    # predict_state.grid_input[predict_heap[idx_key][0][2]] = predict_heap[idx_key][0][3]
                    predict_state.applied_rules_grid[predict_heap[idx_key][0][3]] = predict_heap[idx_key][0]
                    predict_state.all_rules_grid[predict_heap[idx_key][0][3]] = predict_heap[idx_key]
                    # if predict_heap[idx_key][0][0] != 0:
                    #     try:
                    #         print('Best Rule GRID condition difference for state', predict_state, self.debug_dict['Grid'+str(predict_heap[idx_key][0][2])+str(predict_heap[idx_key][0][1])])
                    #     except KeyError:
                    #         pass
                confidence_total += predict_heap[idx_key][0][6]
                confidence_count += predict_heap[idx_key][0][6] - predict_heap[idx_key][0][0]
            else:
                pass

        predict_state.grid_input = self.grid_map[self.map_origin_y + predict_state.pos_input[0][1] - self.grid_origin_y:self.map_origin_y + predict_state.pos_input[0][1] + self.grid_origin_y + 1, self.map_origin_z + predict_state.pos_input[0][2] - self.grid_origin_z:self.map_origin_z + predict_state.pos_input[0][2] + self.grid_origin_z + 1, self.map_origin_x + predict_state.pos_input[0][0] - self.grid_origin_x:self.map_origin_x + predict_state.pos_input[0][0] + self.grid_origin_x + 1]

        predict_state.grid_input = predict_state.grid_input.flatten()

        for item in grid_changes:
            print('Changing Position', item)
            predict_state.grid_input[item[0]] = item[1]

        predict_state.debug_heap = deepcopy(predict_heap)

        if confidence_total != 0:
            predict_state.confidence = confidence_count / confidence_total
            predict_state.confidence_count = confidence_count
            predict_state.confidence_total = confidence_total
        else:
            predict_state.confidence = 0

        # Update state hash
        predict_state.state_hash = hash((tuple(predict_state.pos_input[0]), tuple(predict_state.grid_input)))

        # print('returning state', predict_state, predict_state.pos_input)

        return predict_state

    def create_rule(self, act, input_type, index, pre_val, post_val):
        new_rule = str(uuid.uuid4())[:6]
        o_val = None
        n_val = None
        positive_conditions_grid = []
        negative_conditions_grid = []

        conditions_pos = self.pos_input
        conditions_pos_set = set(range(len(self.pos_input[0])))
        conditions_pos_freq = [1] * len(self.pos_input[0])

        # for i, v in enumerate(self.pos_input):
        #     try:
        #         if self.applied_rules_pos[i]:
        #             conditions_pos_freq[i] = 0
        #             conditions_pos_set.remove(i)
        #     except KeyError:
        #         pass

        # conditions_grid = self.grid_input
        conditions_grid_set = set(range(len(self.grid_input)))
        conditions_grid_freq = [1] * len(self.grid_input)

        for i, v in enumerate(self.grid_input):
            add_set = set()
            add_set.add(v)
            positive_conditions_grid.append(add_set)
            negative_conditions_grid.append(set())

        # for i, v in enumerate(self.grid_input):
        #     try:
        #         if self.applied_rules_grid[i]:
        #             conditions_grid_freq[i] = 0
        #             conditions_grid_set.remove(i)
        #     except KeyError:
        #         pass

        try:
            check = self.knowledge['Action Rules']
        except KeyError:
            self.knowledge['Action Rules'] = dict()

        try:
            check = self.knowledge['Action Updates']
        except KeyError:
            self.knowledge['Action Updates'] = dict()

        try:
            self.knowledge['Action Rules'][act].append(new_rule)
        except KeyError:
            self.knowledge['Action Rules'][act] = [new_rule]

        try:
            self.knowledge['Action Updates'][act] += 1
        except KeyError:
            self.knowledge['Action Updates'][act] = 1

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
            n_val = tuple(map(lambda new, old: new - old, post_val, pre_val))
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
        self.knowledge['Grid-' + str(index) + '/' + str(o_val) + '/' + str(new_rule) + '/Positive Grid Conditions'] = positive_conditions_grid
        self.knowledge['Grid-' + str(index) + '/' + str(o_val) + '/' + str(new_rule) + '/Negative Grid Conditions'] = negative_conditions_grid
        self.knowledge['Grid-' + str(index) + '/' + str(o_val) + '/' + str(new_rule) + '/Grid Conditions Set'] = conditions_grid_set
        self.knowledge['Grid-' + str(index) + '/' + str(o_val) + '/' + str(new_rule) + '/Grid Conditions Freq'] = conditions_grid_freq
        print('Rule', new_rule, 'created for', act, input_type, index, pre_val, post_val)

    def update_good_rule(self, rule_data):
        # rule data format: (diff_count, -age, rule, idx, new_val, 'Grid' / 'Pos', pos_data[2] + grid_data[1], updates, o_val, act, differences)
        idx = rule_data[3]
        rule = rule_data[2]
        idx_t = rule_data[5]
        o_val = rule_data[8]

        pos_remove_list = []
        grid_remove_list = []
        updated = False

        print('Updating rule', rule, 'diff count was', rule_data[0])
        print('Differences', rule_data[10])
        #
        print('Rule Pos Condition Set', self.knowledge['Pos-' + str(idx) + '/' + str(o_val) + '/' + str(rule) + '/Pos Conditions Set'])
        # print('Rule Pos Condition Freq', self.knowledge['Pos-' + str(idx) + '/' + str(o_val) + '/' + str(rule) + '/Pos Conditions Freq'])
        for u_idx in self.knowledge['Pos-' + str(idx) + '/' + str(o_val) + '/' + str(rule) + '/Pos Conditions Set']:
            print('checking data', self.pos_input[0][u_idx], 'and', self.knowledge['Pos-' + str(idx) + '/' + str(o_val) + '/' + str(rule) + '/Pos Conditions'][0][u_idx])
            if self.pos_input[0][u_idx] != self.knowledge['Pos-' + str(idx) + '/' + str(o_val) + '/' + str(rule) + '/Pos Conditions'][0][u_idx]:
                print('updating rule to remove POS set index', u_idx)
                self.knowledge['Pos-' + str(idx) + '/' + str(o_val) + '/' + str(rule) + '/Pos Conditions Freq'][u_idx] = 0
                pos_remove_list.append(u_idx)
                updated = True

            # if not np.all(self.pos_input[u_idx] == self.knowledge['Pos-' + str(idx) + '/' + str(o_val) + '/' + str(rule) + '/Pos Conditions'][u_idx]):
            #     # print('updating rule to remove POS set index', u_idx)
            #     self.knowledge['Pos-' + str(idx) + '/' + str(o_val) + '/' + str(rule) + '/Pos Conditions Freq'][u_idx] = 0
            #     pos_remove_list.append(u_idx)
        print('Checking differences')
        for item in rule_data[10]:
            if item[0] == 'Grid':
                print('Index', item[2], 'actual is', self.grid_input[item[2]], 'and conditions set is', self.knowledge['Grid-' + str(idx) + '/' + str(o_val) + '/' + str(rule) + '/Positive Grid Conditions'][item[2]])
        for u_idx, val in enumerate(self.grid_input):
            if val not in self.knowledge['Grid-' + str(idx) + '/' + str(o_val) + '/' + str(rule) + '/Positive Grid Conditions'][u_idx]:
                print(val, 'not found for index', u_idx, 'adding...')
                self.knowledge['Grid-' + str(idx) + '/' + str(o_val) + '/' + str(rule) + '/Positive Grid Conditions'][u_idx].add(val)
                updated = True

        # for u_idx in self.knowledge['Grid-' + str(idx) + '/' + str(o_val) + '/' + str(rule) + '/Grid Conditions Set']:
        #     if self.grid_input[u_idx] != self.knowledge['Grid-' + str(idx) + '/' + str(o_val) + '/' + str(rule) + '/Grid Conditions'][u_idx]:
        #         # print('updating rule to remove GRID set index', u_idx)
        #         self.knowledge['Grid-' + str(idx) + '/' + str(o_val) + '/' + str(rule) + '/Grid Conditions Freq'][u_idx] = 0
        #         grid_remove_list.append(u_idx)

        for item in pos_remove_list:
            self.knowledge['Pos-' + str(idx) + '/' + str(o_val) + '/' + str(rule) + '/Pos Conditions Set'].remove(item)

        # for item in grid_remove_list:
        #     self.knowledge['Grid-' + str(idx) + '/' + str(o_val) + '/' + str(rule) + '/Grid Conditions Set'].remove(item)

        # [input_type + '-' + str(index) + '/' + str(o_val) + '/' + str(new_rule) + '/Age']
        self.knowledge[idx_t + '-' + str(idx) + '/' + str(o_val) + '/' + str(rule) + '/Age'] = self.time_step

        print('rule was updated')
        self.knowledge['Action Updates'][rule_data[9]] += 1

    # def update_bad_rule(self, rule_data):
    #     # rule data format: (diff_count, rule, idx, new_val, 'Grid' / 'Pos', pos_data[2] + grid_data[1], updates, age, o_val, act)
    #     idx = rule_data[2]
    #     rule = rule_data[1]
    #     t = rule_data[4]
    #     o_val = rule_data[8]
    #
    #     pos_add_list = []
    #     grid_add_list = []
    #
    #     # for u_idx, val in enumerate(self.knowledge['Pos-' + str(idx) + '/' + str(o_val) + '/' + str(rule) + '/Pos Conditions'][0]):
    #     #     if self.pos_input[0][u_idx] == val:
    #     #         self.knowledge['Pos-' + str(idx) + '/' + str(o_val) + '/' + str(rule) + '/Pos Conditions Freq'][0] = 1
    #     #         pos_add_list.append(u_idx)
    #     for u_idx, val in enumerate(self.knowledge['Pos-' + str(idx) + '/' + str(o_val) + '/' + str(rule) + '/Pos Conditions'][0]):
    #         if self.pos_input[0][u_idx] == val:
    #             self.knowledge['Pos-' + str(idx) + '/' + str(o_val) + '/' + str(rule) + '/Pos Conditions Freq'][u_idx] = 1
    #             pos_add_list.append(u_idx)
    #
    #     # if np.all(self.knowledge['Pos-' + str(idx) + '/' + str(o_val) + '/' + str(rule) + '/Pos Conditions'][0] == self.pos_input[0]):
    #     #     self.knowledge['Pos-' + str(idx) + '/' + str(o_val) + '/' + str(rule) + '/Pos Conditions Freq'][0] = 1
    #     #     pos_add_list.append(0)
    #
    #     for u_idx, val in enumerate(self.grid_input):
    #         self.knowledge['Grid-' + str(idx) + '/' + str(o_val) + '/' + str(rule) + '/Negative Grid Conditions'][u_idx].add(val)
    #
    #     # for u_idx, val in enumerate(self.knowledge['Grid-' + str(idx) + '/' + str(o_val) + '/' + str(rule) + '/Grid Conditions Set']):
    #     #     if self.grid_input[u_idx] == val:
    #     #         self.knowledge['Grid-' + str(idx) + '/' + str(o_val) + '/' + str(rule) + '/Grid Conditions Freq'][u_idx] = 1
    #     #         grid_add_list.append(u_idx)
    #
    #     for item in pos_add_list:
    #         self.knowledge['Pos-' + str(idx) + '/' + str(o_val) + '/' + str(rule) + '/Pos Conditions Set'].add(item)
    #
    #     # for item in grid_add_list:
    #     #     self.knowledge['Grid-' + str(idx) + '/' + str(o_val) + '/' + str(rule) + '/Grid Conditions Set'].add(item)

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

    def jump_forward(self, rob, stats):
        a_stats = [mc.getFullStat(key) for key in fullStatKeys]
        o_pitch = round(a_stats[3])
        o_yaw = round(a_stats[4]) % 360
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
        # if [math.floor(stats[0]), math.floor(stats[1]), math.floor(stats[2])] != [e_pos[0], math.floor(stats[1]), e_pos[2]]:
        #     if not timedout:
        #         self.center(rob, [e_pos[0], math.floor(stats[1]), e_pos[2]], o_pitch, o_yaw)
        old_stats = [mc.getFullStat(key) for key in fullStatKeys]
        sleep(.5)
        stats = [mc.getFullStat(key) for key in fullStatKeys]
        while old_stats[1] != stats[1]:
            old_stats = [mc.getFullStat(key) for key in fullStatKeys]
            sleep(.05)
            stats = [mc.getFullStat(key) for key in fullStatKeys]



    def move_forward(self, rob, stats):
        a_stats = [mc.getFullStat(key) for key in fullStatKeys]
        o_pitch = round(a_stats[3])
        o_yaw = round(a_stats[4]) % 360
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
        sleep(.05)
        self.lookDir(rob, o_pitch, o_yaw)
        stats = [mc.getFullStat(key) for key in fullStatKeys]
        # if [math.floor(stats[0]), math.floor(stats[1]), math.floor(stats[2])] != [e_pos[0], math.floor(stats[1]), e_pos[2]]:
        #     if not timedout:
        #         self.center(rob, [e_pos[0], math.floor(stats[1]), e_pos[2]], o_pitch, o_yaw)
        old_stats = [mc.getFullStat(key) for key in fullStatKeys]
        sleep(.5)
        stats = [mc.getFullStat(key) for key in fullStatKeys]
        while old_stats[1] != stats[1]:
            old_stats = [mc.getFullStat(key) for key in fullStatKeys]
            sleep(.05)
            stats = [mc.getFullStat(key) for key in fullStatKeys]


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

    logcount = len(os.listdir('./logs')) * 10
    sys.stdout = open('./logs/Console_Log' + str(logcount) + '.txt', 'w')

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

    rob.sendCommand('chat /gamemode creative')
    rob.sendCommand('chat /effect give @s minecraft:night_vision infinite 0 true')
    sleep(60)
    rob.sendCommand('chat /tp 206 64 119')
    rob.sendCommand('chat /difficulty peaceful')

    sleep(10)
    print('starting!')

    airis.lookDir(rob, 0, 0)

    stats = [mc.getFullStat(key) for key in fullStatKeys]
    stats = [math.floor(stats[0]), math.floor(stats[1]), math.floor(stats[2]), round(stats[3]), 0]  # round(stats[4]) % 360]
    grid = mc.getNearGrid()

    while not airis.goal_achieved:
        stats = [mc.getFullStat(key) for key in fullStatKeys]
        stats = [math.floor(stats[0]), math.floor(stats[1]), math.floor(stats[2]), round(stats[3]), 0]  # round(stats[4]) % 360]
        try:
            action, state, confidence, applied_rules = airis.capture_input(stats, grid, None, None, True, None, None)
            print('performing action', action, 'and predicting state', state)
            # self.actions = ['move 0', 'move 45', 'move 90', 'move 135', 'move 180', 'move 225', 'move 270', 'move 315',
            #                 'jump 0', 'jump 45', 'jump 90', 'jump 135', 'jump 180', 'jump 225', 'jump 270', 'jump 315',
            #                 'mine up 0', 'mine up 45', 'mine up 90', 'mine up 135', 'mine up 180', 'mine up 225', 'mine up 270', 'mine up 315',
            #                 'mine down 0', 'mine down 45', 'mine down 90', 'mine down 135', 'mine down 180', 'mine down 225', 'mine down 270', 'mine down 315',
            #                 'mine straight 0', 'mine straight 45', 'mine straight 90', 'mine straight 135', 'mine straight 180', 'mine straight 225', 'mine straight 270', 'mine straight 315']
            match action:
                case 'move 0':
                    airis.lookDir(rob, 0, 0)
                    airis.move_forward(rob, stats)

                case 'move 45':
                    airis.lookDir(rob, 0, 45)
                    airis.move_forward(rob, stats)

                case 'move 90':
                    airis.lookDir(rob, 0, 90)
                    airis.move_forward(rob, stats)

                case 'move 135':
                    airis.lookDir(rob, 0, 135)
                    airis.move_forward(rob, stats)

                case 'move 180':
                    airis.lookDir(rob, 0, 180)
                    airis.move_forward(rob, stats)

                case 'move 225':
                    airis.lookDir(rob, 0, 225)
                    airis.move_forward(rob, stats)

                case 'move 270':
                    airis.lookDir(rob, 0, 270)
                    airis.move_forward(rob, stats)

                case 'move 315':
                    airis.lookDir(rob, 0, 315)
                    airis.move_forward(rob, stats)

                case 'jump 0':
                    airis.lookDir(rob, 0, 0)
                    airis.jump_forward(rob, stats)

                case 'jump 45':
                    airis.lookDir(rob, 0, 45)
                    airis.jump_forward(rob, stats)

                case 'jump 90':
                    airis.lookDir(rob, 0, 90)
                    airis.jump_forward(rob, stats)

                case 'jump 135':
                    airis.lookDir(rob, 0, 135)
                    airis.jump_forward(rob, stats)

                case 'jump 180':
                    airis.lookDir(rob, 0, 180)
                    airis.jump_forward(rob, stats)

                case 'jump 225':
                    airis.lookDir(rob, 0, 225)
                    airis.jump_forward(rob, stats)

                case 'jump 270':
                    airis.lookDir(rob, 0, 270)
                    airis.jump_forward(rob, stats)

                case 'jump 315':
                    airis.lookDir(rob, 0, 315)
                    airis.jump_forward(rob, stats)

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
            airis.capture_input(stats, grid, action, state, False, confidence, applied_rules)
            print('Current Stats', stats)
            airis.save_knowledge('Knowledge.npy')

            if os.path.getsize('./logs/Console_Log'+str(logcount)+'.txt') > 100000000:
                logcount += 1
                sys.stdout = open('./logs/Console_Log' + str(logcount) + '.txt', 'w')
        except TypeError:
            pass

    print('Test Routine Complete')
