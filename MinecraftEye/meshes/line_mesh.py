from settings import *
from meshes.base_mesh import BaseMesh


class LineMesh(BaseMesh):
    def __init__(self, app, start_pos, end_pos):
        super().__init__()
        self.start = start_pos
        self.end = end_pos
        self.app = app
        self.ctx = self.app.ctx
        self.program = self.app.shader_program.line

        self.vbo_format = '2f2 3f2'
        self.attrs = ('in_tex_coord_0', 'in_position',)
        self.vao = self.get_vao()


    @staticmethod
    def get_data(vertices, indices):
        data = [vertices[ind] for triangle in indices for ind in triangle]
        return np.array(data, dtype='float16')

    def get_vertex_data(self):
        x1 = self.end[0] - self.start[0] + .05
        x2 = x1

        y1 = self.end[1] - self.start[1] + .05
        y2 = y1

        z1 = self.end[2] - self.start[2] + .05
        z2 = z1

        vertices = [
            (0, 0, .1), (x2, y1, z2), (.1, .1, .1), (x1, y2, z2),
            (0, .1, 0), (x1, y1, z1), (.1, 0, 0), (x2, y2, z1),
            (x1, y1, z2), (.1, 0, .1), (x2, y2, z2), (0, .1, .1),
            (x1, y2, z1), (0, 0, 0), (x2, y1, z1), (.1, .1, 0,)
        ]

        indices = [
            (0, 2, 3), (0, 1, 2),
            (1, 7, 2), (1, 6, 7),
            (6, 5, 4), (4, 7, 6),
            (3, 4, 5), (3, 5, 0),
            (3, 7, 4), (3, 2, 7),
            (0, 6, 1), (0, 5, 6),
            (8, 10, 11), (8, 9, 10),
            (9, 15, 10), (9, 14, 15),
            (14, 13, 12), (12, 15, 14),
            (11, 12, 13), (11, 13, 8),
            (11, 15, 12), (11, 10, 15),
            (8, 14, 9), (8, 13, 14)
        ]
        vertex_data = self.get_data(vertices, indices)

        tex_coord_vertices = [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0), (1, 0), (1, 1), (0, 1)]
        tex_coord_indices = [
            (0, 2, 3), (0, 1, 2),
            (0, 2, 3), (0, 1, 2),
            (0, 1, 2), (2, 3, 0),
            (2, 3, 0), (2, 0, 1),
            (0, 2, 3), (0, 1, 2),
            (3, 1, 2), (3, 0, 1),
            (4, 6, 7), (4, 5, 6),
            (4, 6, 7), (4, 5, 6),
            (4, 5, 6), (6, 7, 4),
            (6, 7, 4), (6, 4, 5),
            (4, 6, 7), (4, 5, 6),
            (7, 5, 6), (7, 4, 5)
        ]

        tex_coord_data = self.get_data(tex_coord_vertices, tex_coord_indices)
        vertex_data = np.hstack([tex_coord_data, vertex_data])
        return vertex_data
