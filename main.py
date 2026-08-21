import math
from airis import Airis
import os, sys
from tagilmo.utils.vereya_wrapper import MCConnector, RobustObserver
import tagilmo.utils.mission_builder as mb
from time import sleep
from skills import Skills

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

    skills = Skills(mc, fullStatKeys)

    rob.sendCommand('chat /gamemode creative')
    rob.sendCommand('chat /effect give @s minecraft:night_vision infinite 0 true')
    rob.sendCommand('chat Waiting 5 seconds for world to load')
    rob.sendCommand('chat /difficulty peaceful')
    sleep(5)
    # rob.sendCommand('chat /tp 206 64 119')

    print('starting!')

    skills.lookDir(rob, 0, 0)

    stats = [mc.getFullStat(key) for key in fullStatKeys]
    stats = [math.floor(stats[0]), math.floor(stats[1]), math.floor(stats[2]), round(stats[3]), 0]  # round(stats[4]) % 360]
    grid = mc.getNearGrid()

    while not airis.goal_achieved:
        stats = [mc.getFullStat(key) for key in fullStatKeys]
        stats = [math.floor(stats[0]), math.floor(stats[1]), math.floor(stats[2]), round(stats[3]), 0]  # round(stats[4]) % 360]
        grid = mc.getNearGrid()
        print('Pre Action Stats', stats)
        if math.floor(stats[0]) > 240 or math.floor(stats[0]) < -240 or math.floor(stats[2]) > 240 or math.floor(stats[2]) < -240:
            rob.sendCommand('chat /tp 20 100 20')
            rob.sendCommand('chat Reached boundary of test area, teleporting...')
            sleep(5)
            stats = [mc.getFullStat(key) for key in fullStatKeys]
            stats = [math.floor(stats[0]), math.floor(stats[1]), math.floor(stats[2]), round(stats[3]), 0]  # round(stats[4]) % 360]
            grid = mc.getNearGrid()
        # try:
        action, state, confidence, applied_rules = airis.capture_input(stats, grid, None, None, True, None, None)
        print('performing action', action, 'and predicting state', state)
        # self.actions = ['move 0', 'move 45', 'move 90', 'move 135', 'move 180', 'move 225', 'move 270', 'move 315',
        #                 'jump 0', 'jump 45', 'jump 90', 'jump 135', 'jump 180', 'jump 225', 'jump 270', 'jump 315',
        #                 'mine up 0', 'mine up 45', 'mine up 90', 'mine up 135', 'mine up 180', 'mine up 225', 'mine up 270', 'mine up 315',
        #                 'mine down 0', 'mine down 45', 'mine down 90', 'mine down 135', 'mine down 180', 'mine down 225', 'mine down 270', 'mine down 315',
        #                 'mine straight 0', 'mine straight 45', 'mine straight 90', 'mine straight 135', 'mine straight 180', 'mine straight 225', 'mine straight 270', 'mine straight 315']
        match action:
            case 'move 0':
                skills.lookDir(rob, 0, 0)
                skills.move_forward(rob, stats)

            case 'move 45':
                skills.lookDir(rob, 0, 45)
                skills.move_forward(rob, stats)

            case 'move 90':
                skills.lookDir(rob, 0, 90)
                skills.move_forward(rob, stats)

            case 'move 135':
                skills.lookDir(rob, 0, 135)
                skills.move_forward(rob, stats)

            case 'move 180':
                skills.lookDir(rob, 0, 180)
                skills.move_forward(rob, stats)

            case 'move 225':
                skills.lookDir(rob, 0, 225)
                skills.move_forward(rob, stats)

            case 'move 270':
                skills.lookDir(rob, 0, 270)
                skills.move_forward(rob, stats)

            case 'move 315':
                skills.lookDir(rob, 0, 315)
                skills.move_forward(rob, stats)

            case 'jump 0':
                skills.lookDir(rob, 0, 0)
                skills.jump_forward(rob, stats)

            case 'jump 45':
                skills.lookDir(rob, 0, 45)
                skills.jump_forward(rob, stats)

            case 'jump 90':
                skills.lookDir(rob, 0, 90)
                skills.jump_forward(rob, stats)

            case 'jump 135':
                skills.lookDir(rob, 0, 135)
                skills.jump_forward(rob, stats)

            case 'jump 180':
                skills.lookDir(rob, 0, 180)
                skills.jump_forward(rob, stats)

            case 'jump 225':
                skills.lookDir(rob, 0, 225)
                skills.jump_forward(rob, stats)

            case 'jump 270':
                skills.lookDir(rob, 0, 270)
                skills.jump_forward(rob, stats)

            case 'jump 315':
                skills.lookDir(rob, 0, 315)
                skills.jump_forward(rob, stats)

            case 'mine up 0':
                skills.lookDir(rob, -60, 0)
                skills.mine(rob)
                skills.lookDir(rob, 0, 0)

            case 'mine up 45':
                skills.lookDir(rob, -60, 45)
                skills.mine(rob)
                skills.lookDir(rob, 0, 45)

            case 'mine up 90':
                skills.lookDir(rob, -60, 90)
                skills.mine(rob)
                skills.lookDir(rob, 0, 90)

            case 'mine up 135':
                skills.lookDir(rob, -60, 135)
                skills.mine(rob)
                skills.lookDir(rob, 0, 135)

            case 'mine up 180':
                skills.lookDir(rob, -60, 180)
                skills.mine(rob)
                skills.lookDir(rob, 0, 180)

            case 'mine up 225':
                skills.lookDir(rob, -60, 225)
                skills.mine(rob)
                skills.lookDir(rob, 0, 225)

            case 'mine up 270':
                skills.lookDir(rob, -60, 270)
                skills.mine(rob)
                skills.lookDir(rob, 0, 270)

            case 'mine up 315':
                skills.lookDir(rob, -60, 315)
                skills.mine(rob)
                skills.lookDir(rob, 0, 315)

            case 'mine down 0':
                skills.lookDir(rob, 60, 0)
                skills.mine(rob)
                skills.lookDir(rob, 0, 0)

            case 'mine down 45':
                skills.lookDir(rob, 60, 45)
                skills.mine(rob)
                skills.lookDir(rob, 0, 45)

            case 'mine down 90':
                skills.lookDir(rob, 60, 90)
                skills.mine(rob)
                skills.lookDir(rob, 0, 90)

            case 'mine down 135':
                skills.lookDir(rob, 60, 135)
                skills.mine(rob)
                skills.lookDir(rob, 0, 135)

            case 'mine down 180':
                skills.lookDir(rob, 60, 180)
                skills.mine(rob)
                skills.lookDir(rob, 0, 180)

            case 'mine down 225':
                skills.lookDir(rob, 60, 225)
                skills.mine(rob)
                skills.lookDir(rob, 0, 225)

            case 'mine down 270':
                skills.lookDir(rob, 60, 270)
                skills.mine(rob)
                skills.lookDir(rob, 0, 270)

            case 'mine down 315':
                skills.lookDir(rob, 60, 315)
                skills.mine(rob)
                skills.lookDir(rob, 0, 315)

            case 'mine straight 0':
                skills.lookDir(rob, 0, 0)
                skills.mine(rob)
                skills.lookDir(rob, 0, 0)

            case 'mine straight 45':
                skills.lookDir(rob, 0, 45)
                skills.mine(rob)
                skills.lookDir(rob, 0, 45)

            case 'mine straight 90':
                skills.lookDir(rob, 0, 90)
                skills.mine(rob)
                skills.lookDir(rob, 0, 90)

            case 'mine straight 135':
                skills.lookDir(rob, 0, 135)
                skills.mine(rob)
                skills.lookDir(rob, 0, 135)

            case 'mine straight 180':
                skills.lookDir(rob, 0, 180)
                skills.mine(rob)
                skills.lookDir(rob, 0, 180)

            case 'mine straight 225':
                skills.lookDir(rob, 0, 225)
                skills.mine(rob)
                skills.lookDir(rob, 0, 225)

            case 'mine straight 270':
                skills.lookDir(rob, 0, 270)
                skills.mine(rob)
                skills.lookDir(rob, 0, 270)

            case 'mine straight 315':
                skills.lookDir(rob, 0, 315)
                skills.mine(rob)
                skills.lookDir(rob, 0, 315)

        stats = [mc.getFullStat(key) for key in fullStatKeys]
        stats = [math.floor(stats[0]), math.floor(stats[1]), math.floor(stats[2]), round(stats[3]), 0]  # round(stats[4]) % 360]
        grid = mc.getNearGrid()
        airis.capture_input(stats, grid, action, state, False, confidence, applied_rules)
        print('Current Stats', stats)
        try:
            total = 0
            for item in airis.knowledge['Action Rules']:
                total += len(airis.knowledge['Action Rules'][item])
            print('Total number of rules: ', total)
        except KeyError:
            pass
        airis.save_knowledge('Knowledge.npy')

        if os.path.getsize('./logs/Console_Log'+str(logcount)+'.txt') > 100000000:
            logcount += 1
            sys.stdout = open('./logs/Console_Log' + str(logcount) + '.txt', 'w')
        # except TypeError:
        #     if math.floor(stats[0]) > 240 or math.floor(stats[0]) < -240 or math.floor(stats[2]) > 240 or math.floor(stats[2]) < -240:
        #         rob.sendCommand('chat /tp 20 100 20')
        #         rob.sendCommand('chat Reached boundary of test area, teleporting...')
        #         sleep(5)
        #     else:
        #         rob.sendCommand('chat /tp 20 100 20')
        #         rob.sendCommand('chat I think I am stuck. Maybe I will try exploring here later.')