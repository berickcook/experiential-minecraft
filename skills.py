from tagilmo.utils.mathutils import normAngle, degree2rad
from time import sleep, time
import math

class Skills:
    def __init__(self, mc, fullStatKeys):
        print('mc', mc)
        print('fullStatKeys', fullStatKeys)
        self.mc = mc
        self.fullStatKeys = fullStatKeys
    
    def lookDir(self, rob, pitch, yaw):
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
            [pitch, yaw] = self.mc.dirToPos(aPos, pos)
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
        a_stats = [self.mc.getFullStat(key) for key in self.fullStatKeys]
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
            stats = [self.mc.getFullStat(key) for key in self.fullStatKeys]
        rob.sendCommand('move 0')
        rob.sendCommand('jump 0')
        sleep(.02)
        self.lookDir(rob, o_pitch, o_yaw)
        # stats = [self.mc.getFullStat(key) for key in self.fullStatKeys]
        # if [math.floor(stats[0]), math.floor(stats[1]), math.floor(stats[2])] != [e_pos[0], math.floor(stats[1]), e_pos[2]]:
        #     if not timedout:
        #         self.center(rob, [e_pos[0], math.floor(stats[1]), e_pos[2]], o_pitch, o_yaw)
        # old_stats = [self.mc.getFullStat(key) for key in self.fullStatKeys]
        # sleep(.5)
        # stats = [self.mc.getFullStat(key) for key in self.fullStatKeys]
        # while old_stats[1] != stats[1]:
        #     old_stats = [self.mc.getFullStat(key) for key in self.fullStatKeys]
        #     sleep(.02)
        #     stats = [self.mc.getFullStat(key) for key in self.fullStatKeys]
        #     if old_stats[1] == stats[1]:
        #         old_stats = [self.mc.getFullStat(key) for key in self.fullStatKeys]
        #         sleep(.02)
        #         stats = [self.mc.getFullStat(key) for key in self.fullStatKeys]


    def move_forward(self, rob, stats):
        a_stats = [self.mc.getFullStat(key) for key in self.fullStatKeys]
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
            stats = [self.mc.getFullStat(key) for key in self.fullStatKeys]
            if math.floor(stats[1]) != math.floor(e_pos[1]):
                break
        rob.sendCommand('move 0')
        sleep(.02)
        self.lookDir(rob, o_pitch, o_yaw)
        # stats = [self.mc.getFullStat(key) for key in self.fullStatKeys]
        # # if [math.floor(stats[0]), math.floor(stats[1]), math.floor(stats[2])] != [e_pos[0], math.floor(stats[1]), e_pos[2]]:
        # #     if not timedout:
        # #         self.center(rob, [e_pos[0], math.floor(stats[1]), e_pos[2]], o_pitch, o_yaw)
        # old_stats = [self.mc.getFullStat(key) for key in self.fullStatKeys]
        # sleep(.5)
        # stats = [self.mc.getFullStat(key) for key in self.fullStatKeys]
        # while old_stats[1] != stats[1]:
        #     old_stats = [self.mc.getFullStat(key) for key in self.fullStatKeys]
        #     sleep(.02)
        #     stats = [self.mc.getFullStat(key) for key in self.fullStatKeys]
        #     if old_stats[1] == stats[1]:
        #         old_stats = [self.mc.getFullStat(key) for key in self.fullStatKeys]
        #         sleep(.02)
        #         stats = [self.mc.getFullStat(key) for key in self.fullStatKeys]


    def mine(self, rob):
        stats_old = [self.mc.getFullStat(key) for key in self.fullStatKeys]
        rob.sendCommand('attack 1')
        timeout = 0
        sleep(0.2)
        rob.sendCommand('attack 0')
        sleep(0.2)
        stats = [self.mc.getFullStat(key) for key in self.fullStatKeys]
        while stats != stats_old:
            stats_old = stats
            timeout += .1
            sleep(0.1)
            stats = [self.mc.getFullStat(key) for key in self.fullStatKeys]
            if timeout > 10:
                break