from typing import List, Optional
from scipy.optimize import root_scalar, fsolve
from pyfluids import Fluid, FluidsList, Input
from thermo.chemical import Chemical
import numpy as np
from os import system 
import time


class FeedSystemCriticalPath:
    def __init__(self, dt = 1, totalPipeLength=1.0,fluid: Fluid=None, N=2, extraComponents= [[None,None],[None,None]], mdot: float = None, inletPressure: float = None, outletPressure: float = None):
        self.N = N
        self.dL = totalPipeLength / N
        self.totalPipeLength = totalPipeLength
        self.inletPressure = inletPressure
        self.outletPressure = outletPressure
        self.extraComponents = extraComponents
        self.fluid = fluid
        self.mdot = mdot
        self.dt = dt
        fluid.update(Input.temperature(-180), Input.pressure(inletPressure))  # Initial pressure in Pascals
        self.initTemp = fluid.temperature
        self.populate()
    def discretise(self):
        self.discretisedFeed = []
        # scale and round extra component positions to the discretisation index
        check = True
        for comp in self.extraComponents:
            if comp[0] is not None:
                comp[1] = round(comp[1] / self.dL)
                check = True
            else:
                check = False
            # sort extraComponents by the first column
        if check:
            self.extraComponents.sort(key=lambda comp: comp[0])

        count = 0
        for i in range(self.N):
            pos = count * self.dL
            if i in [comp[1] for comp in self.extraComponents if comp[0] is not None]:
                #add in shit
                pass
            else:
                count += 1
                pipe = Pipe(fluid=self.fluid, length=self.dL, mdot=self.mdot, pos=pos, location=i+1,  temp=self.initTemp, diameter=0.01)
            self.discretisedFeed.append(pipe)

    def populate(self):
        component: systemComponent
        currentPressure = self.inletPressure
        prevComponent = None
        self.discretise()
        for i, component in enumerate(self.discretisedFeed):
            if component.type == "p":
                dp = component.dp(currentPressure, self.mdot)
                
                currentPressure -= dp
            elif component.type == "o":
                # add in shit
                component.pressureIn = self.inletPressure
                component.pressureOut = self.outletPressure
            else:
                raise ValueError(f"Unknown component type: {component.type}")

        self.discretisedFeed.insert(0, ghostCell(location=0, u=self.discretisedFeed[0].getVelocity(), pos=0, pressureIn=self.inletPressure, pressureOut=self.outletPressure, mdot=self.mdot, temp=self.initTemp))
        self.discretisedFeed.append(ghostCell(location=self.N, prevComp=self.discretisedFeed[-1], u=0, pos=self.totalPipeLength, pressureIn=self.inletPressure, pressureOut=self.outletPressure, mdot=self.mdot))
        self.boundaryPopulation()


    def boundaryPopulation(self):
        component: systemComponent
        for i, component in enumerate(self.discretisedFeed):
            if component.type == "g":
                continue
            cellV = component.getVelocity()
            
            if cellV<0:
                component.setuIn(cellV)
                component.setuOut(self.discretisedFeed[i+1].getVelocity())
            else:
                component.setuIn(self.discretisedFeed[i-1].getVelocity())
                component.setuOut(cellV)

            component.record()
            #print(self.discretisedFeed[i].getID(), self.discretisedFeed[i].type)
        




    def solve(self):
        for i, component in enumerate(self.discretisedFeed):
            
            if component.type == "g":
                continue
            component: systemComponent
            component.solve(self.discretisedFeed[i-1].get_u_iterate(), self.discretisedFeed[i+1].getPressureIn(), self.dt)
            self.upWinding(component, self.discretisedFeed[i-1], self.discretisedFeed[i+1])
            #print(f"Component {component.getID()}, {component.get_u_iterate()} m/s, {component.getPressureIn()/1e5} bar")


    def upWinding(self, current, previous, next):
        current: systemComponent
        previous: systemComponent
        next: systemComponent
        cellV = current.get_u_iterate()
        if cellV >= 0:
            current.setuIn(previous.get_u_iterate())
            current.setuOut(cellV)
        else:
            current.setuIn(cellV)
            current.setuOut(next.get_u_iterate())




class systemComponent:
    def __init__(self, type, location=None, pressureIn=None, pressureOut=None, length=None, pos=None, temp = None, fluid=None, rho=None, mdot=None, initP: np.array = None, initT: np.array = None, initMdot: np.array = None):
        self.id = location
        self.type = type
        self.pos = pos
        self.pressureIn = pressureIn
        self.pressureOut = pressureOut
        self.length = length
        self.temp = temp
        self.fluid: Fluid = fluid
        self.rho = rho
        
        self.mdot = mdot
        self.pressureThroughTime = [initP]
        self.tempThroughTime = [initT]
        self.mdotThroughTime = [initMdot]
        self.uIn = None
        self.uOut = None
        self.u_iterate = None

    def getID(self):
        return self.id
    def getuIn(self):
        return self.uIn
    def setuIn(self, uIn):
        self.uIn = uIn
    def getPos(self):
        return self.pos
    def getuOut(self):
        return self.uOut
    def setuOut(self, uOut):
        self.uOut = uOut
    def setMdot(self, mdot):
        self.mdot = mdot
    def getMdot(self):
        return self.mdot
    def get_u_iterate(self):
        return self.u_iterate
    def set_u_iterate(self, u_iterate):
        self.u_iterate = u_iterate

    def getPressureIn(self):
        return self.pressureIn

    def dp(self):
        raise NotImplementedError("This method should be overridden by subclasses")
    
    def getVelocity(self):
        raise NotImplementedError("This method should be overridden by subclasses")
    
    def solve(self, prevVelocity, prevPressure):
        """
        Solve the component's state based on previous velocity and pressure.
        This method should be overridden by subclasses.
        """
        raise NotImplementedError("This method should be overridden by subclasses")

class ghostCell(systemComponent):  
    def __init__(self, u=None, pos=None, pressureIn=None, pressureOut=None, temp=None, rho=None, mdot=None, prevComp=None, location=None):
        super().__init__(type="g", pos=pos, pressureIn=pressureIn, pressureOut=pressureOut, temp=temp, rho=rho, mdot=mdot, location=location)

        self.length = 0.0
        self.uIn = u
        self.uOut = self.uIn
        self.u_iterate = u
        if prevComp is not None:
            prevComp: systemComponent
            #prevComp.setuOut(u)
            prevComp.setMdot(0.01)
    
    def getVelocity(self):
        """
        Return the velocity of the ghost cell, which is set to the incoming velocity.
        """
        return self.u_iterate

    def dp(self):
        return 0.0  # No pressure drop in ghost cells
    def setU(self, uIn):
        super().setuIn(uIn)
        super().setuOut(uIn)
    

class Pipe(systemComponent):
    def __init__(self, fluid: Fluid=None, length=1.0, diameter=0.1, roughness=0.0001, location=0, pos=None, pressureIn=0, pressureOut=0, temp=None, rho=None, mdot=None): 
        super().__init__(type="p", location=location, pressureIn=pressureIn, pressureOut=pressureOut, length=length, pos=pos, temp=temp, fluid=fluid, rho=rho, mdot=mdot)
        self.diameter = diameter
        self.roughness = roughness

    def getVelocity(self):
        if self.rho is None or self.mdot is None:
            raise ValueError("Density and mass flow rate must be set before calculating velocity.")
        area = np.pi * (self.diameter / 2) ** 2
        self.u_iterate = self.mdot / (self.rho * area)
        return self.u_iterate

    def reCalc(self):


       # print(self.pressureIn, self.temp)
        self.fluid.update(Input.temperature(self.temp), Input.pressure(self.pressureIn)) if self.fluid else None
        self.rho = self.fluid.density if self.fluid else self.rho

        mue = self.fluid.dynamic_viscosity
        v = self.getVelocity()
        Re = (self.rho * v * self.diameter) / mue
        return Re
    
    def reCalcDirectional(self):

        #print(self.pressureIn, self.temp)
        self.fluid.update(Input.temperature(self.temp), Input.pressure(self.pressureIn)) if self.fluid else None
        self.rho = self.fluid.density if self.fluid else self.rho

        mue = self.fluid.dynamic_viscosity
        v = self.u_iterate
        Re = (self.rho * v * self.diameter) / mue
        #print(f"Reynolds number: {Re}")
        return Re

    def frictionFactor(self):
        Re = self.reCalc()
        if Re < 3000:
            return 64 / Re
        #using lambert equations for exact solutions, rather than colebrooks equation. Gives accuracy within 
        else:
            return root_scalar(self.colebrookFF, args = (self.roughness, self.diameter, Re), bracket=[1e-5, 1], method='bisect').root

    def frictionFactorDirectional(self):
        Re = self.reCalcDirectional()
        if Re < 3000:
            return 64 / Re
        #using lambert equations for exact solutions, rather than colebrooks equation. Gives accuracy within 
        else:
            return root_scalar(self.colebrookFF, args = (self.roughness, self.diameter, Re), bracket=[1e-5, 1], method='bisect').root

    
    def colebrookFF(self,f, surfaceRoughness, innerD, Re):
        return 1 / np.sqrt(f) + 2 * np.log10(surfaceRoughness / (3.7 * innerD) + 2.51 / (Re * np.sqrt(f)))
    
    def darcyWeisbach(self):
        f = self.frictionFactor()
        v = self.getVelocity()
        return f * (self.length / self.diameter) * (self.rho * v ** 2) / 2
    
    def darcyWeisbachDirectional(self):
        f = self.frictionFactorDirectional()
        v = self.u_iterate
        return f *(self.length/self.diameter) *(self.rho*v* abs(v))/2
    
    
    def dp(self, pressureIn=None, mdot=None):
        if pressureIn is not None:
            self.pressureIn = pressureIn
        self.fluid.update(Input.pressure(self.pressureIn), Input.temperature(self.temp)) 
        if mdot is not None:
            self.mdot = mdot
        dp = self.darcyWeisbach()
        
        return dp

    def update(self):
        self.fluid.update(Input.temperature(self.temp), Input.pressure(self.pressureIn))
        self.rho = self.fluid.density
        a = self.fluid.sound_speed
        self.mdot = self.u_iterate * self.rho * np.pi * (self.diameter / 2) ** 2

    def solve(self, prevVelocity, nextPressure,dt):
        self.update()
        
        self.fluid.update(Input.temperature(self.temp), Input.pressure(self.pressureIn))
        self.rho = self.fluid.density
        a = self.fluid.sound_speed
        dpdt = self.rho*a**2 *(prevVelocity - self.u_iterate)/self.length
        self.pressureIn+= dpdt * dt
        dudt = ((1/self.length)*(self.pressureIn +self.rho*(self.uIn)**2 - nextPressure - self.rho*(self.uOut)**2) - self.darcyWeisbachDirectional())/self.rho
        self.u_iterate += dudt * dt
        #self.fluid = self.fluid.isentropic_compression_to_pressure(self.pressureIn)
        self.temp = self.fluid.temperature
        self.record()



    def record(self):
        self.pressureThroughTime.append(self.pressureIn)
        self.tempThroughTime.append(self.temp)
        self.mdotThroughTime.append(self.mdot)


def barA(pa):
    """
    Convert pressure from bar to Pascals.
    """
    return pa * 1e5

if __name__ == "__main__":
    # Example usage
    fluid = Fluid(FluidsList.Oxygen)
    fluid.update(Input.temperature(-180), Input.pressure(barA(100)))
    simTime = 10 #s
    timeIterations= 1000
    dt = simTime / timeIterations
    feed_system = FeedSystemCriticalPath(dt = dt,totalPipeLength=1.0, fluid=fluid, N=10, mdot=1, inletPressure=barA(100), outletPressure=barA(50))

    dL = feed_system.dL
    initial_velocity = feed_system.discretisedFeed[1].getVelocity()
    sound_speed = fluid.sound_speed
    cfl_dt = dL / (abs(initial_velocity) + sound_speed)
    # Use a safety factor (e.g., 0.5) and choose the smaller of user-defined or CFL dt
    dt = min(simTime / timeIterations, 0.5 * cfl_dt)
    feed_system.dt = dt / 10
    timeIterations = round(simTime / feed_system.dt)
    print(f"Sim time: {feed_system.dt * timeIterations} s, dt: {feed_system.dt} s, dL: {dL} m, CFL dt: {cfl_dt} s")
    print(timeIterations, "iterations")

    for i in range(timeIterations):
        feed_system.solve()
        #print(f"Iteration {i+1}/{timeIterations} completed.")
        # print progress every 1000 iterations
        if i > 0 and i % 1000 == 0:
            print(f"Progress: {i / timeIterations * 100:.0f}%")
    import matplotlib.pyplot as plt

    # gather only the real cells (exclude ghost cells)
    real_cells = [c for c in feed_system.discretisedFeed if c.type != 'g']

    # pick the start, middle and end indices
    idxs = [0, len(real_cells) // 2, len(real_cells) - 1]
    positions = ['start', 'middle', 'end']

    # # gather final pressures and mass‐flow rates, filtering out None
    # positions = []
    # pressures = []
    # mdots = []
    # for cell in real_cells:
    #     cell: systemComponent
    #     p_last = cell.pressureThroughTime[-1]
    #     m_last = cell.mdotThroughTime[-1]
    #     if p_last is not None and m_last is not None:
    #         positions.append(cell.getPos())#m
    #         pressures.append(p_last / 1e5)   # convert to bar
    #         mdots.append(m_last)
    # print(dt)
    # # plot on shared x‐axis, twin y‐axes
    # fig, ax1 = plt.subplots(figsize=(8, 5))
    # ax1.plot(positions, pressures, 'b-o', label='Pressure (bar)')
    # ax1.set_xlabel('Position (m)')
    # ax1.set_ylabel('Pressure (bar)', color='b')
    # ax1.tick_params(axis='y', labelcolor='b')

    # ax2 = ax1.twinx()
    # ax2.plot(positions, mdots, 'r--s', label='Mass flow (kg/s)')
    # ax2.set_ylabel('Mass flow (kg/s)', color='r')
    # ax2.tick_params(axis='y', labelcolor='r')

    # plt.title('Final Timestep: Pressure and Mass Flow Along Feed System')
    # ax1.grid(True)
    # fig.tight_layout()
    # plt.show()
    fig, axes = plt.subplots(3, 1, figsize=(8, 12), sharex=True)

    for ax, idx, pos in zip(axes, idxs, positions):
        cell: systemComponent
        cell = real_cells[idx]
        # build a time vector assuming constant dt
        
        # build full time array
        raw_time = np.arange(len(cell.pressureThroughTime)) * dt

        # mask out any entries where pressure or mdot is None
        mask = [
            (p is not None and m is not None)
            for p, m in zip(cell.pressureThroughTime, cell.mdotThroughTime)
        ]

        # apply mask
        t = np.array([t for t, ok in zip(raw_time, mask) if ok])
        p_bar = np.array([p for p, ok in zip(cell.pressureThroughTime, mask) if ok]) / 1e5
        mdot = np.array([m for m, ok in zip(cell.mdotThroughTime, mask) if ok])

        # plot pressure
        ax.plot(t, p_bar, 'b-', label='Pressure (bar)')
        ax.set_ylabel('Pressure (bar)', color='b')
        ax.tick_params(axis='y', labelcolor='b')
        ax.grid(True)

        # plot mass flow rate on twin axis
        ax2 = ax.twinx()
        ax2.plot(t, mdot, 'r--', label='Mass flow (kg/s)')
        ax2.set_ylabel('Mass flow (kg/s)', color='r')
        ax2.tick_params(axis='y', labelcolor='r')

        ax.set_title(f'Cell at {pos} of feed system')

    # common x‐label
    axes[-1].set_xlabel('Time (s)')

    plt.tight_layout()
    plt.show()
    print(dt)