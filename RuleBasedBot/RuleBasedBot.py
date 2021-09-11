import sc2
from sc2.constants import NEXUS, PROBE, PYLON, ASSIMILATOR, GATEWAY, CYBERNETISCORE, STALKER, STARGATE, VOIDRAY
import random

class RuleBasedBot(sc2.BotAI):
    def __init__(self):
        self.iterationsPerMinute = 165
        self.maxWorkers = 50

    async def on_step(self, iteration):
        self.iteration = iteration
        
        await self.distribute_workers()
        await self.buildWorkers()
        await self.buildPylons()
        await self.buildAssimilator()
        await self.expand()
        await self.offensiveForceBuildings()
        await self.buildOffensiveForce()
        await self.attack()
    
    async def buildWorkers(self):
        if (len(self.units(NEXUS)) * 16) > len(self.units(PROBE)) and \
            len(self.units(PROBE)) < self.MAX_WORKERS:
            for nexus in self.units(NEXUS).ready.noqueue:
                if self.can_afford():
                    await self.do(nexus.train(PROBE))
    
    async def buildPylons(self):
        if self.supply_left < 5 and not self.already_pending(PYLON):
            nexuses = self.units(NEXUS).ready
            if nexuses.exists and self.can_afford(PYLON):
                await self.build(PYLON, near=nexuses.first)

    async def buildAssimilator(self):
        for nexus in self.units(NEXUS).ready:
            vaspenes = self.state.vespene_geyser.closer_than(25, nexus)
            for vaspene in vaspenes:
                if not self.can_afford(ASSIMILATOR):
                    break

                worker = self.select_build_worker(vaspene.position)
                if worker is None:
                    break

                if not self.units(ASSIMILATOR).closer_than(1, vaspene).exists:
                    await self.do(worker.build(ASSIMILATOR, vaspene))
    
    async def expand(self):
        if self.units(NEXUS).amount < 3 and self.can_afford(NEXUS):
            await self.expand_now()
    
    async def offensiveForceBuildings(self):
        if self.units(PYLON).ready.exists:
            pylon = self.units(PYLON).ready.random

            if self.units(GATEWAY).ready.exists and \
               not self.units(CYBERNETISCORE) and \
               self.can_afford(CYBERNETISCORE) and \
               not self.already_pending(CYBERNETISCORE):
                await self.build(CYBERNETISCORE, near=pylon)  
            elif len(self.units(GATEWAY)) < ((self.iteration / self.ITERATIONS_PER_MINUTE)/2) and \
                self.can_afford(GATEWAY) and \
                not self.already_pending(GATEWAY):
                    await self.build(GATEWAY, near=pylon)

            if self.units(CYBERNETISCORE).ready.exists and \
               len(self.units(STARGATE)) < ((self.iteration / self.ITERATIONS_PER_MINUTE)/2) and \
               self.can_afford(STARGATE) and \
               not self.already_pending(STARGATE):
                await self.build(STARGATE, near=pylon)
    
    async def buildOffensiveForce(self):
        for gw in self.units(GATEWAY).ready.noqueue:
            if not self.units(STALKER).amount > self.units(VOIDRAY).amount and \
               self.can_afford(STALKER) and \
               self.supply_left > 0:
                await self.do(gw.train(STALKER))
        
        for sg in self.units(STARGATE).ready.noqueue:
            if self.can_afford(VOIDRAY) and self.supply_left > 0:
                await self.do(sg.train(VOIDRAY))

    def findTarget(self, state):
        if len(self.known_enemy_units) > 0:
            return random.choice(self.known_enemy_units)
        elif len(self.known_enemy_structures) > 0:
            return random.choice(self.known_enemy_structures)
        else:
            return self.enemy_start_locations[0]

    async def attack(self):
        aggressiveUnits = {STALKER: [15, 5], VOIDRAY: [8, 3]}
        for UNIT in aggressiveUnits:
            if self.units(UNIT).amount > aggressiveUnits[UNIT][0] and self.units(UNIT).amount > aggressiveUnits[UNIT][1]:
                for s in self.units(UNIT).idle:
                    await self.do(s.attack(self.findTarget(self.state)))

            elif self.units(UNIT).amount > aggressiveUnits[UNIT][1]:
                if len(self.known_enemy_units) > 0:
                    for s in self.units(UNIT).idle:
                        await self.do(s.attack(random.choice(self.known_enemy_units)))