import pygame as pg
from camera import Camera
import moderngl as mgl
from settings import *


class Player(Camera):
    def __init__(self, app, position=PLAYER_POS, yaw=-90, pitch=0):
        self.app = app
        self.cam = Camera
        self.mode_pressed = False
        self.read_pressed = False
        self.follow_pressed = False
        self.xray_pressed = False
        self.xray_mode = False
        self.show_all_states_pressed = False
        self.show_all_states_mode = False
        self.angle = 0
        self.height = 0
        self.height_switch = False
        self.distance = 40
        self.follow_mode = False
        self.distance_mode = 8
        self.distance_switch = 0
        super().__init__(position, yaw, pitch)

    def update(self):
        self.keyboard_control()
        self.mouse_control()

        if self.follow_mode:
            if self.distance_mode == 0:
                self.app.ctx.enable(flags=mgl.DEPTH_TEST | mgl.CULL_FACE | mgl.BLEND)
                self.distance = 100
                self.xray_mode = False
            elif self.distance_mode == 1:
                self.app.ctx.disable(flags=mgl.DEPTH_TEST | mgl.CULL_FACE | mgl.BLEND)
                self.distance = 100
                self.xray_mode = True
            elif self.distance_mode == 2:
                self.app.ctx.enable(flags=mgl.DEPTH_TEST | mgl.CULL_FACE | mgl.BLEND)
                self.distance = 50
                self.xray_mode = False
            elif self.distance_mode == 3:
                self.app.ctx.disable(flags=mgl.DEPTH_TEST | mgl.CULL_FACE | mgl.BLEND)
                self.distance = 50
                self.xray_mode = True
            elif self.distance_mode == 4:
                self.app.ctx.enable(flags=mgl.DEPTH_TEST | mgl.CULL_FACE | mgl.BLEND)
                self.distance = 25
                self.xray_mode = False
            elif self.distance_mode == 5:
                self.app.ctx.disable(flags=mgl.DEPTH_TEST | mgl.CULL_FACE | mgl.BLEND)
                self.distance = 25
                self.xray_mode = True
            elif self.distance_mode == 6:
                self.app.ctx.enable(flags=mgl.DEPTH_TEST | mgl.CULL_FACE | mgl.BLEND)
                self.distance = 10
                self.xray_mode = False
            elif self.distance_mode == 7:
                self.app.ctx.disable(flags=mgl.DEPTH_TEST | mgl.CULL_FACE | mgl.BLEND)
                self.distance = 10
                self.xray_mode = True
            elif self.distance_mode == 8:
                self.app.ctx.enable(flags=mgl.DEPTH_TEST | mgl.CULL_FACE | mgl.BLEND)
                self.distance = 5
                self.xray_mode = False
            elif self.distance_mode == 9:
                self.app.ctx.disable(flags=mgl.DEPTH_TEST | mgl.CULL_FACE | mgl.BLEND)
                self.distance = 5
                self.xray_mode = True

            camX = self.focus_pos[0] + self.distance * -math.sin(self.angle*(math.pi/180))
            camY = self.focus_pos[1]
            camZ = self.focus_pos[2] + -self.distance * math.cos(self.angle*(math.pi/180))

            self.angle += .01 * self.app.delta_time
            if not self.height_switch:
                if self.height < self.distance / 2:
                    self.height += (self.distance / 10000) * self.app.delta_time
                else:
                    self.height_switch = not self.height_switch
            else:
                if self.height > -self.distance / 2:
                    self.height -= (self.distance / 10000) * self.app.delta_time
                else:
                    self.height_switch = not self.height_switch

            if self.distance_switch < 360:
                self.distance_switch += .01 * self.app.delta_time
            else:
                if self.distance_mode < 9:
                    self.distance_mode += 1
                else:
                    self.distance_mode = 0

                self.distance_switch = 0

            self.position = glm.vec3(camX, camY + self.height, camZ)

        super().update()

    def handle_event(self, event):
        # adding and removing voxels with clicks
        # if event.type == pg.MOUSEBUTTONDOWN:
        #     voxel_handler = self.app.scene.world.voxel_handler
        #     if event.button == 1:
        #         voxel_handler.set_voxel()
        #     if event.button == 3:
        #         voxel_handler.switch_mode()
        pass

    def mouse_control(self):
        if not self.follow_mode:
            mouse_dx, mouse_dy = pg.mouse.get_rel()
            if mouse_dx:
                self.rotate_yaw(delta_x=mouse_dx * MOUSE_SENSITIVITY)
            if mouse_dy:
                self.rotate_pitch(delta_y=mouse_dy * MOUSE_SENSITIVITY)

    def keyboard_control(self):
        key_state = pg.key.get_pressed()
        vel = PLAYER_SPEED * self.app.delta_time
        if key_state[pg.K_w]:
            if key_state[pg.K_LSHIFT]:
                self.move_forward(vel*2)
            else:
                self.move_forward(vel)
        if key_state[pg.K_s]:
            self.move_back(vel)
        if key_state[pg.K_d]:
            self.move_right(vel)
        if key_state[pg.K_a]:
            self.move_left(vel)
        if key_state[pg.K_e]:
            self.move_up(vel)
        if key_state[pg.K_q]:
            self.move_down(vel)
        if key_state[pg.K_m]:
            if not self.mode_pressed:
                if self.app.scene.render_mode < 5:
                    self.app.scene.render_mode += 1
                else:
                    self.app.scene.render_mode = 0
                match self.app.scene.render_mode:
                    case 0:
                        self.app.scene.render_mode_name = 'All Data'
                    case 1:
                        self.app.scene.render_mode_name = 'Only Plan'
                    case 2:
                        self.app.scene.render_mode_name = 'Plan + Perfect Match Edges'
                    case 3:
                        self.app.scene.render_mode_name = 'Plan + Imperfect Match Edges'
                    case 4:
                        self.app.scene.render_mode_name = 'Only Perfect Match Edges'
                    case 5:
                        self.app.scene.render_mode_name = 'Only Imperfect Match Edges'
                self.mode_pressed = True
        if not key_state[pg.K_m]:
            self.mode_pressed = False

        if key_state[pg.K_r]:
            if not self.read_pressed:
                self.app.scene.read_only = not self.app.scene.read_only
                self.read_pressed = True

        if not key_state[pg.K_r]:
            self.read_pressed = False

        if key_state[pg.K_f]:
            if not self.follow_pressed:
                self.follow_mode = not self.follow_mode
                self.cam.follow_mode = self.follow_mode
                if self.follow_mode:
                    self.rotate_yaw(self.yaw)
                    self.rotate_pitch(self.pitch)
                    # print('pitch', self.pitch)
                else:
                    self.app.ctx.enable(flags=mgl.DEPTH_TEST | mgl.CULL_FACE | mgl.BLEND)
                    self.xray_mode = False
                self.follow_pressed = True

        if not key_state[pg.K_f]:
            self.follow_pressed = False

        if key_state[pg.K_x]:
            if not self.xray_pressed:
                self.xray_mode = not self.xray_mode

                if self.xray_mode:
                    self.app.ctx.disable(flags=mgl.DEPTH_TEST | mgl.CULL_FACE | mgl.BLEND)
                else:
                    self.app.ctx.enable(flags=mgl.DEPTH_TEST | mgl.CULL_FACE | mgl.BLEND)
                self.xray_pressed = True

        if not key_state[pg.K_x]:
            self.xray_pressed = False

        if key_state[pg.K_b]:
            if not self.show_all_states_pressed:
                self.show_all_states_mode = not self.show_all_states_mode
                self.show_all_states_pressed = True

        if not key_state[pg.K_b]:
            self.show_all_states_pressed = False