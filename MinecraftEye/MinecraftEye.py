from settings import *
import moderngl as mgl
import pygame as pg
import sys
from shader_program import ShaderProgram
from scene import Scene
from player import Player
from textures import Textures


class VoxelEngine:
    def __init__(self):
        pg.init()
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, MAJOR_VER)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, MINOR_VER)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_PROFILE_MASK, pg.GL_CONTEXT_PROFILE_CORE)
        pg.display.gl_set_attribute(pg.GL_DEPTH_SIZE, DEPTH_SIZE)
        pg.display.gl_set_attribute(pg.GL_MULTISAMPLESAMPLES, NUM_SAMPLES)

        pg.display.set_mode(WIN_RES, flags=pg.OPENGL | pg.DOUBLEBUF)
        self.ctx = mgl.create_context()

        self.ctx.enable(flags=mgl.DEPTH_TEST | mgl.CULL_FACE | mgl.BLEND)
        self.ctx.gc_mode = 'auto'

        self.clock = pg.time.Clock()
        self.delta_time = 0
        self.time = 0

        pg.event.set_grab(True)
        pg.mouse.set_visible(False)

        self.is_running = True
        self.on_init()
        self.focus = True

    def on_init(self):
        self.textures = Textures(self)
        self.player = Player(self)
        self.shader_program = ShaderProgram(self)
        self.scene = Scene(self)

    def update(self):
        if self.focus:
            self.player.update()
        self.shader_program.update()
        self.scene.update()

        self.delta_time = self.clock.tick()
        self.time = pg.time.get_ticks() * 0.001

        if not self.scene.read_only:
            pg.display.set_caption(self.scene.render_mode_name + ' - Agent Pos: [ ' + str(math.floor(self.player.focus_pos[0] - (WORLD_W * CHUNK_SIZE / 2))) + ' , ' + str(math.floor(self.player.focus_pos[1]) - 64) + ' , ' + str(math.floor(self.player.focus_pos[2] - (WORLD_D * CHUNK_SIZE / 2))) + ' ] - ' + f'{self.clock.get_fps() :.0f}')
        else:
            pg.display.set_caption(self.scene.render_mode_name + ' - ' + f'{self.clock.get_fps() :.0f}' + ' Read Only Mode')

    def render(self):
        self.ctx.clear(color=BG_COLOR)
        self.scene.render()
        pg.display.flip()

    def handle_events(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.is_running = False
            if event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE:
                self.focus = False
                pg.event.set_grab(False)
                pg.mouse.set_visible(True)
            if event.type == pg.KEYDOWN and (event.key == pg.K_KP_ENTER or event.key == pg.K_RETURN):
                self.focus = True
                pg.event.set_grab(True)
                pg.mouse.set_visible(False)
            self.player.handle_event(event=event)

    def run(self):
        while self.is_running:
            self.handle_events()
            self.update()
            self.render()
        pg.quit()
        sys.exit()


if __name__ == '__main__':
    app = VoxelEngine()
    app.run()
