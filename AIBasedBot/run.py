from sc2 import run_game, maps, Race, Difficulty
from sc2.player import Bot, Computer

import AIBasedBot

run_game(maps.get("AbyssalReefLE"), [
    Bot(Race.Protoss, AIBasedBot()),
    Computer(Race.Terran, Difficulty.Easy)
], realtime=False)