from sc2 import run_game, maps, Race, Difficulty
from sc2.player import Bot, Computer

import RuleBasedBot

run_game(maps.get("AbyssalReefLE"), [
    Bot(Race.Protoss, RuleBasedBot()),
    Computer(Race.Terran, Difficulty.Easy)
], realtime=False)