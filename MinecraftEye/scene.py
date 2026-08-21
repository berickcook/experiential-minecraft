from settings import *
import moderngl as mgl
from world import World
from world_objects.voxel_marker import VoxelMarker
from world_objects.water import Water
from world_objects.clouds import Clouds
from world_objects.state import State
from world_objects.line import Line
import time


class Scene:
    def __init__(self, app):
        self.app = app
        self.world = World(self.app)
        self.voxel_marker = VoxelMarker(self.world.voxel_handler)
        self.water = Water(app)
        self.clouds = Clouds(app)
        self.states = dict()
        # self.states[(480, 33, 480)] = State(app, glm.vec3(480.33, 33.33, 480.33))
        # self.states[(480, 33, 480)].mode = 0
        # self.states[(482, 37, 489)] = State(app, glm.vec3(482.33, 37.33, 489.33))
        # self.states[(482, 37, 489)].mode = 0
        # self.states[(481, 33, 480)] = State(app, glm.vec3(481.33, 33.33, 480.33))
        # self.states[(481, 33, 480)].mode = 1
        # self.states[(482, 33, 480)] = State(app, glm.vec3(482.33, 33.33, 480.33))
        # self.states[(482, 33, 480)].mode = 2
        # self.states[(483, 33, 480)] = State(app, glm.vec3(483.33, 33.33, 480.33))
        # self.states[(483, 33, 480)].mode = 3
        # self.states[(484, 33, 480)] = State(app, glm.vec3(484.33, 33.33, 480.33))
        # self.states[(484, 33, 480)].mode = 0
        # self.states[(485, 33, 480)] = State(app, glm.vec3(485.33, 33.33, 480.33))
        # self.states[(485, 33, 480)].mode = 0
        # self.states[(486, 33, 480)] = State(app, glm.vec3(486.33, 33.33, 480.33))
        # self.states[(486, 33, 480)].mode = 0
        # self.states[(487, 33, 480)] = State(app, glm.vec3(487.33, 33.33, 480.33))
        # self.states[(487, 33, 480)].mode = 0
        self.edges = dict()
        # self.edges[(480, 33, 480)] = Line(app, (480.45, 33.45, 480.45), (482.45, 37.45, 489.45))
        # self.edges[(480, 33, 480)].mode = 3
        # self.edges[(481, 33, 480)] = Line(app, (480.45, 33.45, 480.45), (481.45, 33.45, 480.45))
        # self.edges[(481, 33, 480)].mode = 3
        self.grid = dict()
        self.time = time.time()
        self.render_mode = 0
        self.render_mode_name = 'All Data'
        self.read_only = False
        self.toggle = False
        self.world.voxel_handler.add_voxel(WORLD_W * CHUNK_SIZE / 2, 128, WORLD_D * CHUNK_SIZE / 2, 'diamond_ore')
        self.view_box = []
        self.goal_box = []
        self.agent_pos = glm.vec3(0, 0, 0)
        self.goal_pos = glm.vec3(0, 0, 0)
    def update(self):
        self.world.update()
        self.voxel_marker.update()
        self.clouds.update()
        updated = False
        current_state = None

        if time.time() - self.time > 2 and not self.read_only:
            try:
                load_states = np.load('../output/state_output.npy', allow_pickle=True).item()
                load_edges = np.load('../output/edge_output.npy', allow_pickle=True).item()
                load_grid = np.load('../output/grid_output.npy', allow_pickle=True).item()

                self.time = time.time()

                for load_key in load_states.keys():
                    try:
                        self.states[load_key].mode = load_states[load_key][3]
                    except KeyError:
                        self.states[load_key] = State(self.app, glm.vec3(load_states[load_key][0] + (WORLD_W * CHUNK_SIZE / 2) + .33, load_states[load_key][1] + .33 + 64, load_states[load_key][2] + (WORLD_D * CHUNK_SIZE / 2) + .33))
                        self.states[load_key].mode = load_states[load_key][3]
                    if load_states[load_key][3] == 1:
                        self.app.player.focus_pos = glm.vec3(load_states[load_key][0] + (WORLD_W * CHUNK_SIZE / 2) + .33, load_states[load_key][1] + .33 + 64, load_states[load_key][2] + (WORLD_D * CHUNK_SIZE / 2) + .33)
                        self.agent_pos = self.app.player.focus_pos
                    updated = True

                for load_key in load_edges.keys():
                    try:
                        self.edges[load_key].mode = load_edges[load_key][6]
                    except KeyError:
                        self.edges[load_key] = Line(self.app, glm.vec3(load_edges[load_key][0] + (WORLD_W * CHUNK_SIZE / 2) + .45, load_edges[load_key][1] + .45 + 64, load_edges[load_key][2] + (WORLD_D * CHUNK_SIZE / 2) + .45), glm.vec3(load_edges[load_key][3] + (WORLD_W * CHUNK_SIZE / 2) + .45, load_edges[load_key][4] + .45 + 64, load_edges[load_key][5] + (WORLD_D * CHUNK_SIZE / 2) + .45))
                        self.edges[load_key].mode = load_edges[load_key][6]
                        if load_edges[load_key][7] == 1:
                            self.goal_pos = glm.vec3(load_edges[load_key][3] + (WORLD_W * CHUNK_SIZE / 2) + .33, load_edges[load_key][4] + .33 + 64, load_edges[load_key][5] + (WORLD_D * CHUNK_SIZE / 2) + .33)
                    updated = True

                del_list = []
                for check_key in self.edges.keys():
                    if check_key not in load_edges.keys():
                        del_list.append(check_key)

                for item in del_list:
                    del self.edges[item]

                for check_key in self.states.keys():
                    if check_key not in load_states.keys():
                        self.states[check_key].mode = 3

                for load_key in load_grid.keys():
                    if load_key not in self.grid.keys():
                        self.grid[load_key] = load_grid[load_key]
                        self.world.voxel_handler.add_voxel(load_grid[load_key][0] + (WORLD_W * CHUNK_SIZE / 2), load_grid[load_key][1] + 64, load_grid[load_key][2] + (WORLD_D * CHUNK_SIZE / 2), load_grid[load_key][3])
                        updated = True

                # Create view box
                self.view_box = []
                self.view_box.append(Line(self.app, glm.vec3(self.agent_pos[0] - 2 - .33, self.agent_pos[1] - 2 - .33, self.agent_pos[2] - 2 - .33), glm.vec3(self.agent_pos[0] - 2 - .33, self.agent_pos[1] + 2 + .66, self.agent_pos[2] - 2 - .33)))
                self.view_box[0].mode = 6
                self.view_box.append(Line(self.app, glm.vec3(self.agent_pos[0] + 2 + .66, self.agent_pos[1] - 2 - .33, self.agent_pos[2] - 2 - .33), glm.vec3(self.agent_pos[0] + 2 + .66, self.agent_pos[1] + 2 + .66, self.agent_pos[2] - 2 - .33)))
                self.view_box[1].mode = 6
                self.view_box.append(Line(self.app, glm.vec3(self.agent_pos[0] - 2 - .33, self.agent_pos[1] - 2 - .33, self.agent_pos[2] + 2 + .66), glm.vec3(self.agent_pos[0] - 2 - .33, self.agent_pos[1] + 2 + .66, self.agent_pos[2] + 2 + .66)))
                self.view_box[2].mode = 6
                self.view_box.append(Line(self.app, glm.vec3(self.agent_pos[0] + 2 + .66, self.agent_pos[1] - 2 - .33, self.agent_pos[2] + 2 + .66), glm.vec3(self.agent_pos[0] + 2 + .66, self.agent_pos[1] + 2 + .66, self.agent_pos[2] + 2 + .66)))
                self.view_box[3].mode = 6

                self.view_box.append(Line(self.app, glm.vec3(self.agent_pos[0] - 2 - .33, self.agent_pos[1] - 2 - .33, self.agent_pos[2] - 2 - .33), glm.vec3(self.agent_pos[0] - 2 - .33, self.agent_pos[1] - 2 - .33, self.agent_pos[2] + 2 + .66)))
                self.view_box[4].mode = 6
                self.view_box.append(Line(self.app, glm.vec3(self.agent_pos[0] + 2 + .66, self.agent_pos[1] - 2 - .33, self.agent_pos[2] - 2 - .33), glm.vec3(self.agent_pos[0] + 2 + .66, self.agent_pos[1] - 2 - .33, self.agent_pos[2] + 2 + .66)))
                self.view_box[5].mode = 6
                self.view_box.append(Line(self.app, glm.vec3(self.agent_pos[0] - 2 - .33, self.agent_pos[1] - 2 - .33, self.agent_pos[2] + 2 + .66), glm.vec3(self.agent_pos[0] + 2 + .66, self.agent_pos[1] - 2 - .33, self.agent_pos[2] + 2 + .66)))
                self.view_box[6].mode = 6
                self.view_box.append(Line(self.app, glm.vec3(self.agent_pos[0] - 2 - .33, self.agent_pos[1] - 2 - .33, self.agent_pos[2] - 2 - .33), glm.vec3(self.agent_pos[0] + 2 + .66, self.agent_pos[1] - 2 - .33, self.agent_pos[2] - 2 - .33)))
                self.view_box[7].mode = 6

                self.view_box.append(Line(self.app, glm.vec3(self.agent_pos[0] - 2 - .33, self.agent_pos[1] + 2 + .66, self.agent_pos[2] - 2 - .33), glm.vec3(self.agent_pos[0] - 2 - .33, self.agent_pos[1] + 2 + .66, self.agent_pos[2] + 2 + .66)))
                self.view_box[8].mode = 6
                self.view_box.append(Line(self.app, glm.vec3(self.agent_pos[0] + 2 + .66, self.agent_pos[1] + 2 + .66, self.agent_pos[2] - 2 - .33), glm.vec3(self.agent_pos[0] + 2 + .66, self.agent_pos[1] + 2 + .66, self.agent_pos[2] + 2 + .66)))
                self.view_box[9].mode = 6
                self.view_box.append(Line(self.app, glm.vec3(self.agent_pos[0] - 2 - .33, self.agent_pos[1] + 2 + .66, self.agent_pos[2] + 2 + .66), glm.vec3(self.agent_pos[0] + 2 + .66, self.agent_pos[1] + 2 + .66, self.agent_pos[2] + 2 + .66)))
                self.view_box[10].mode = 6
                self.view_box.append(Line(self.app, glm.vec3(self.agent_pos[0] - 2 - .33, self.agent_pos[1] + 2 + .66, self.agent_pos[2] - 2 - .33), glm.vec3(self.agent_pos[0] + 2 + .66, self.agent_pos[1] + 2 + .66, self.agent_pos[2] - 2 - .33)))
                self.view_box[11].mode = 6

                # Create Goal Box
                self.goal_box = []
                self.goal_box.append(Line(self.app, glm.vec3(self.goal_pos[0] - 2 - .33, self.goal_pos[1] - 2 - .33, self.goal_pos[2] - 2 - .33), glm.vec3(self.goal_pos[0] - 2 - .33, self.goal_pos[1] + 2 + .66, self.goal_pos[2] - 2 - .33)))
                self.goal_box[0].mode = 4
                self.goal_box.append(Line(self.app, glm.vec3(self.goal_pos[0] + 2 + .66, self.goal_pos[1] - 2 - .33, self.goal_pos[2] - 2 - .33), glm.vec3(self.goal_pos[0] + 2 + .66, self.goal_pos[1] + 2 + .66, self.goal_pos[2] - 2 - .33)))
                self.goal_box[1].mode = 4
                self.goal_box.append(Line(self.app, glm.vec3(self.goal_pos[0] - 2 - .33, self.goal_pos[1] - 2 - .33, self.goal_pos[2] + 2 + .66), glm.vec3(self.goal_pos[0] - 2 - .33, self.goal_pos[1] + 2 + .66, self.goal_pos[2] + 2 + .66)))
                self.goal_box[2].mode = 4
                self.goal_box.append(Line(self.app, glm.vec3(self.goal_pos[0] + 2 + .66, self.goal_pos[1] - 2 - .33, self.goal_pos[2] + 2 + .66), glm.vec3(self.goal_pos[0] + 2 + .66, self.goal_pos[1] + 2 + .66, self.goal_pos[2] + 2 + .66)))
                self.goal_box[3].mode = 4

                self.goal_box.append(Line(self.app, glm.vec3(self.goal_pos[0] - 2 - .33, self.goal_pos[1] - 2 - .33, self.goal_pos[2] - 2 - .33), glm.vec3(self.goal_pos[0] - 2 - .33, self.goal_pos[1] - 2 - .33, self.goal_pos[2] + 2 + .66)))
                self.goal_box[4].mode = 4
                self.goal_box.append(Line(self.app, glm.vec3(self.goal_pos[0] + 2 + .66, self.goal_pos[1] - 2 - .33, self.goal_pos[2] - 2 - .33), glm.vec3(self.goal_pos[0] + 2 + .66, self.goal_pos[1] - 2 - .33, self.goal_pos[2] + 2 + .66)))
                self.goal_box[5].mode = 4
                self.goal_box.append(Line(self.app, glm.vec3(self.goal_pos[0] - 2 - .33, self.goal_pos[1] - 2 - .33, self.goal_pos[2] + 2 + .66), glm.vec3(self.goal_pos[0] + 2 + .66, self.goal_pos[1] - 2 - .33, self.goal_pos[2] + 2 + .66)))
                self.goal_box[6].mode = 4
                self.goal_box.append(Line(self.app, glm.vec3(self.goal_pos[0] - 2 - .33, self.goal_pos[1] - 2 - .33, self.goal_pos[2] - 2 - .33), glm.vec3(self.goal_pos[0] + 2 + .66, self.goal_pos[1] - 2 - .33, self.goal_pos[2] - 2 - .33)))
                self.goal_box[7].mode = 4

                self.goal_box.append(Line(self.app, glm.vec3(self.goal_pos[0] - 2 - .33, self.goal_pos[1] + 2 + .66, self.goal_pos[2] - 2 - .33), glm.vec3(self.goal_pos[0] - 2 - .33, self.goal_pos[1] + 2 + .66, self.goal_pos[2] + 2 + .66)))
                self.goal_box[8].mode = 4
                self.goal_box.append(Line(self.app, glm.vec3(self.goal_pos[0] + 2 + .66, self.goal_pos[1] + 2 + .66, self.goal_pos[2] - 2 - .33), glm.vec3(self.goal_pos[0] + 2 + .66, self.goal_pos[1] + 2 + .66, self.goal_pos[2] + 2 + .66)))
                self.goal_box[9].mode = 4
                self.goal_box.append(Line(self.app, glm.vec3(self.goal_pos[0] - 2 - .33, self.goal_pos[1] + 2 + .66, self.goal_pos[2] + 2 + .66), glm.vec3(self.goal_pos[0] + 2 + .66, self.goal_pos[1] + 2 + .66, self.goal_pos[2] + 2 + .66)))
                self.goal_box[10].mode = 4
                self.goal_box.append(Line(self.app, glm.vec3(self.goal_pos[0] - 2 - .33, self.goal_pos[1] + 2 + .66, self.goal_pos[2] - 2 - .33), glm.vec3(self.goal_pos[0] + 2 + .66, self.goal_pos[1] + 2 + .66, self.goal_pos[2] - 2 - .33)))
                self.goal_box[11].mode = 4

            except FileNotFoundError:
                pass
            except PermissionError:
                pass

            # if updated:
            #     for state in self.states.keys():
            #         if self.states[state] != current_state:
            #             self.states[state].mode = 3




    def render(self):
        # chunks rendering
        self.world.render()

        # rendering without cull face
        self.app.ctx.disable(mgl.CULL_FACE)
        if not self.app.player.xray_mode:
            self.clouds.render()
            self.water.render()
        self.app.ctx.enable(mgl.CULL_FACE)

        # voxel selection
        self.voxel_marker.render()

        # view box render
        for item in self.view_box:
            item.render()

        for item in self.goal_box:
            item.render()

        # States / Edges render

        for key in self.states.keys():
            if self.app.player.show_all_states_mode:
                self.states[key].render()
            else:
                if self.states[key].mode != 3:
                    self.states[key].render()

        for key in self.edges.keys():
            match self.render_mode:
                case 0:
                    # 0 - All
                    self.edges[key].render()
                case 1:
                    # 1 - Only Plan
                    match self.edges[key].mode:
                        case 1:
                            self.edges[key].render()
                        case 4:
                            self.edges[key].render()
                        case 5:
                            self.edges[key].render()
                case 2:
                    # 2 - Plan + Perfect Match Edges
                    match self.edges[key].mode:
                        case 3:
                            self.edges[key].render()
                        case 1:
                            self.edges[key].render()
                        case 4:
                            self.edges[key].render()
                        case 5:
                            self.edges[key].render()
                case 3:
                    # 3 - Plan + Imperfect Match Edges
                    match self.edges[key].mode:
                        case 0:
                            self.edges[key].render()
                        case 1:
                            self.edges[key].render()
                        case 4:
                            self.edges[key].render()
                        case 5:
                            self.edges[key].render()
                case 4:
                    # 4 - Only Perfect Match Edges
                    if self.edges[key].mode == 3:
                        self.edges[key].render()
                case 5:
                    # 5 - Only Imperfect Match Edges
                    if self.edges[key].mode == 0:
                        self.edges[key].render()