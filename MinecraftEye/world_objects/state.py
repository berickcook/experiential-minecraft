from settings import *
from meshes.state_mesh import StateMesh
from meshes.cube_mesh import CubeMesh


class State:
    def __init__(self, app, position):
        self.app = app
        self.position = position
        self.m_model = self.get_model_matrix()
        self.mesh = StateMesh(self.app)
        self.mode = 2

    def set_uniform(self):
        self.mesh.program['mode_id'] = self.mode
        self.mesh.program['m_model'].write(self.get_model_matrix())

    def get_model_matrix(self):
        m_model = glm.translate(glm.mat4(), glm.vec3(self.position))
        return m_model

    def render(self):
        self.set_uniform()
        self.mesh.render()
