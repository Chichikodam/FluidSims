from typing import List, Optional
from scipy.optimize import root_scalar, fsolve
from pyfluids import Fluid, FluidsList, Input
from thermo.chemical import Chemical
import numpy as np
from os import system 
import time
import os
import csv

class FeedSystemCriticalPath:
    def __init__(self, dt = 1, totalPipeLength=1.0,fluid: Fluid=None, N=2, extraComponents= [[None,None],[None,None]], mdot: float = None, inletPressure: float = None, outletPressure: float = None, max_iterations=100000):
        self.N = N
        self.dL = totalPipeLength / N
        self.totalPipeLength = totalPipeLength
        self.inletPressure = inletPressure
        self.outletPressure = outletPressure
        self.extraComponents = extraComponents
        self.fluid = fluid
        self.mdot = mdot
        self.dt = dt
        self.max_iterations = max_iterations
        self.current_iteration = 0
        
        # Pre-allocate numpy arrays for all time history data
        self.pressure_history = np.zeros((N + 2, max_iterations))  # +2 for ghost cells
        self.temp_history = np.zeros((N + 2, max_iterations))
        self.mdot_history = np.zeros((N + 2, max_iterations))
        self.velocity_history = np.zeros((N + 2, max_iterations))
        
        fluid.update(Input.temperature(-180), Input.pressure(inletPressure))  # Initial pressure in Pascals
        self.initTemp = fluid.temperature
        self.populate()

    def setMaxIterations(self, max_iterations):
        self.max_iterations = max_iterations
        self.pressure_history = np.zeros((self.N + 2, max_iterations))  # +2 for ghost cells
        self.temp_history = np.zeros((self.N + 2, max_iterations))
        self.mdot_history = np.zeros((self.N + 2, max_iterations))
        self.velocity_history = np.zeros((self.N + 2, max_iterations))

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
                pipe = Pipe(fluid=self.fluid, length=self.dL,pos=pos, location=i+1,  temp=self.initTemp, diameter=0.01)
            self.discretisedFeed.append(pipe)

    def populate(self):
        component: systemComponent
        currentPressure = self.inletPressure
        prevComponent = None
        self.discretise()
        self.solveMdot(self.inletPressure, self.outletPressure)
        for i, component in enumerate(self.discretisedFeed):
            if component.type == "p":
                component.setMdot(self.mdot)
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
    def getSystemDP(self, mdot):
        comp: systemComponent
        # Implement the function to calculate the system pressure drop based on mdot
        return sum(comp.dp(mdot=mdot, pressureIn=self.inletPressure) for comp in self.discretisedFeed if comp.getType() != "g")


    def solveMdot(self, inletPressure: Optional[float] = None, outletPressure: Optional[float] = None):
        dpTarget = inletPressure - outletPressure
        
        def dpFunc(mdot):
            # Implement the function to calculate dp based on mdot
            return self.getSystemDP(mdot) - dpTarget
        
        # Find a suitable bracket
        brackets_to_try = [
            [0.001, 10],
            [0.0001, 1], 
            [0.01, 100],
            [-1, 1],
            [-10, 10]
        ]
        
        result = None
        for bracket in brackets_to_try:
            try:
                # Check if the function values have opposite signs
                f_a = dpFunc(bracket[0])
                f_b = dpFunc(bracket[1])
                
                if f_a * f_b < 0:  # Opposite signs
                    result = root_scalar(dpFunc, bracket=bracket, method='brentq')
                    self.mdot = result.root
                    break
            except Exception as e:
                continue
        
        if result is None:
            # Fallback method
            try:
                from scipy.optimize import fsolve
                initial_guess = dpTarget / 10000 if abs(dpTarget) > 0 else 0.01
                solution = fsolve(dpFunc, initial_guess)
                self.mdot = solution[0]
            except:
                print(f"Warning: Could not solve for system mdot. Using estimate.")
                # Simple estimate based on pressure drop
                self.mdot = max(0.001, dpTarget / 100000)

        

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
                component: ghostCell
                if i==0:
                    component.setuIn(self.discretisedFeed[i+1].getVelocity())
                    component.setuOut(self.discretisedFeed[i+1].getVelocity())
                    component.setPressureIn(self.inletPressure)
                    component.setPressureOut(self.inletPressure)
                elif i == len(self.discretisedFeed) - 1:
                    component.setuIn(self.discretisedFeed[i-1].getVelocity())
                    component.setuOut(self.discretisedFeed[i-1].getVelocity())
                    component.setPressureIn(self.outletPressure)
                    component.setPressureOut(self.outletPressure)
                    component.update(self.discretisedFeed[i-1])
                    component.solveMdot(self.discretisedFeed[i-1])
                continue
            component: systemComponent
            print(component.getPressureIn())
            component.solve(self.discretisedFeed[i-1].get_u_iterate(), self.discretisedFeed[i+1].getPressureIn(), self.dt)
            self.upWinding(component, self.discretisedFeed[i-1], self.discretisedFeed[i+1])
            # Record data efficiently using preallocation
            component.record_to_arrays(self.pressure_history, self.temp_history, 
                                     self.mdot_history, self.velocity_history, 
                                     self.current_iteration)
            #print(f"Component {component.getID()}, {component.get_u_iterate()} m/s, {component.getPressureIn()/1e5} bar")
        component: systemComponent
        for i, component in enumerate(self.discretisedFeed):
            if component.type == "g":
                component.solveMdot(self.discretisedFeed[i+1] if i==0 else self.discretisedFeed[i-1])
                continue
            component.solveMdot()
            # Record data efficiently using preallocation
            component.record_to_arrays(self.pressure_history, self.temp_history, 
                                     self.mdot_history, self.velocity_history, 
                                     self.current_iteration)
        self.current_iteration += 1

    def write_to_csv(self, pressure_filename="pressure_data.csv", massflow_filename="massflow_data.csv", temperature_filename="temperature_results.csv", output_dir="simulation_results"):
        """
        Write pressure and mass flow data to separate CSV files in a specified directory.
        
        Args:
            pressure_filename (str): Name of the pressure CSV file
            massflow_filename (str): Name of the mass flow CSV file
            output_dir (str): Directory to save the files in
        """
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created directory: {output_dir}")
        
        # Full file paths
        pressure_path = os.path.join(output_dir, pressure_filename)
        massflow_path = os.path.join(output_dir, massflow_filename)
        temperature_path = os.path.join(output_dir, temperature_filename)
        metadata_path = os.path.join(output_dir, "simulation_metadata.csv")
        
        # Get only real cells (exclude ghost cells)
        real_cells = [c for c in self.discretisedFeed if c.type != 'g']
        real_cell_ids = [c.getID() for c in real_cells]
        
        # Create time array
        time_array = np.arange(self.current_iteration) * self.dt
        
        # Write metadata file
        with open(metadata_path, 'w', newline='') as metadata_file:
            writer = csv.writer(metadata_file)
            writer.writerow(['Parameter', 'Value', 'Unit'])
            writer.writerow(['Total Iterations', self.current_iteration, 'iterations'])
            writer.writerow(['Time Step (dt)', self.dt, 's'])
            writer.writerow(['Total Simulation Time', time_array[-1] if len(time_array) > 0 else 0, 's'])
            writer.writerow(['Number of Nodes', len(real_cells), 'nodes'])
            writer.writerow(['Pipe Length', self.totalPipeLength, 'm'])
            writer.writerow(['Node Spacing (dL)', self.dL, 'm'])
            writer.writerow(['Inlet Pressure', self.inletPressure, 'Pa'])
            writer.writerow(['Outlet Pressure', self.outletPressure, 'Pa'])
            writer.writerow(['Initial Mass Flow', self.mdot, 'kg/s'])
        
        #write temp data with metadata comments
        with open(temperature_path, 'w', newline='') as temp_file:
            writer = csv.writer(temp_file)
            
            # Write metadata as comments
            writer.writerow(['# Simulation Metadata'])
            writer.writerow([f'# Total Iterations: {self.current_iteration}'])
            writer.writerow([f'# Time Step (dt): {self.dt} s'])
            writer.writerow([f'# Total Simulation Time: {time_array[-1] if len(time_array) > 0 else 0} s'])
            writer.writerow([f'# Number of Nodes: {len(real_cells)}'])
            writer.writerow([f'# Pipe Length: {self.totalPipeLength} m'])
            writer.writerow([f'# Node Spacing (dL): {self.dL} m'])
            writer.writerow(['# Data begins below'])
            
            # Write header
            header = ['Time (s)'] + [f'{cell.getType()}_{cell.getID()}_Temperature (K)' for cell in real_cells]
            writer.writerow(header)
            
            # Write data row by row
            for i in range(self.current_iteration):
                row = [time_array[i]]
                for cell_id in real_cell_ids:
                    row.append(self.temp_history[cell_id, i])
                writer.writerow(row)
        # Write pressure data with metadata comments
        with open(pressure_path, 'w', newline='') as pressure_file:
            writer = csv.writer(pressure_file)
            
            # Write metadata as comments
            writer.writerow(['# Simulation Metadata'])
            writer.writerow([f'# Total Iterations: {self.current_iteration}'])
            writer.writerow([f'# Time Step (dt): {self.dt} s'])
            writer.writerow([f'# Total Simulation Time: {time_array[-1] if len(time_array) > 0 else 0} s'])
            writer.writerow([f'# Number of Nodes: {len(real_cells)}'])
            writer.writerow([f'# Pipe Length: {self.totalPipeLength} m'])
            writer.writerow([f'# Node Spacing (dL): {self.dL} m'])
            writer.writerow(['# Data begins below'])
            
            cell: systemComponent
            # Write header
            header = ['Time (s)'] + [f'{cell.getType()}_{cell.getID()}_Pressure (Pa)' for cell in real_cells]
            writer.writerow(header)
            
            # Write data row by row
            for i in range(self.current_iteration):
                row = [time_array[i]]
                for cell_id in real_cell_ids:
                    row.append(self.pressure_history[cell_id, i])
                writer.writerow(row)
        
        # Write mass flow data with metadata comments
        with open(massflow_path, 'w', newline='') as massflow_file:
            writer = csv.writer(massflow_file)
            
            # Write metadata as comments
            writer.writerow(['# Simulation Metadata'])
            writer.writerow([f'# Total Iterations: {self.current_iteration}'])
            writer.writerow([f'# Time Step (dt): {self.dt} s'])
            writer.writerow([f'# Total Simulation Time: {time_array[-1] if len(time_array) > 0 else 0} s'])
            writer.writerow([f'# Number of Nodes: {len(real_cells)}'])
            writer.writerow([f'# Pipe Length: {self.totalPipeLength} m'])
            writer.writerow([f'# Node Spacing (dL): {self.dL} m'])
            writer.writerow(['# Data begins below'])
            
            # Write header
            header = ['Time (s)'] + [f'{cell.getType()}_{cell.getID()}_Massflow (kg/s)' for cell in real_cells]
            writer.writerow(header)
            
            # Write data row by row
            for i in range(self.current_iteration):
                row = [time_array[i]]
                for cell_id in real_cell_ids:
                    row.append(self.mdot_history[cell_id, i])
                writer.writerow(row)
        
        print(f"Data written to:")
        print(f"  Pressure: {pressure_path}")
        print(f"  Mass flow: {massflow_path}")
        print(f"  Metadata: {metadata_path}")
        print(f"Data shape: {self.current_iteration} time steps x {len(real_cell_ids)} cells")


    def get_cell_data(self, cell_id, up_to_iteration=None):
        """Get time series data for a specific cell from pre-allocated arrays"""
        if up_to_iteration is None:
            up_to_iteration = self.current_iteration
        
        return {
            'pressure': self.pressure_history[cell_id, :up_to_iteration],
            'temperature': self.temp_history[cell_id, :up_to_iteration],
            'mdot': self.mdot_history[cell_id, :up_to_iteration],
            'velocity': self.velocity_history[cell_id, :up_to_iteration]
        }

    def get_final_snapshot(self):
        """Get final state of all cells"""
        final_iter = self.current_iteration - 1
        return {
            'pressure': self.pressure_history[:, final_iter],
            'temperature': self.temp_history[:, final_iter],
            'mdot': self.mdot_history[:, final_iter],
            'velocity': self.velocity_history[:, final_iter]
        }


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
        # Keep legacy lists for backward compatibility, but use numpy arrays for performance
        self.pressureThroughTime = [initP] if initP is not None else []
        self.tempThroughTime = [initT] if initT is not None else []
        self.mdotThroughTime = [initMdot] if initMdot is not None else []
        self.uIn = None
        self.uOut = None
        self.u_iterate = None

    def record_to_arrays(self, pressure_history, temp_history, mdot_history, velocity_history, iteration):
        """Efficiently record dat to pre-allocated numpy arrays"""
        if self.id is not None and iteration < pressure_history.shape[1]:
            pressure_history[self.id, iteration] = self.pressureIn if self.pressureIn is not None else 0
            temp_history[self.id, iteration] = self.temp if self.temp is not None else 0
            mdot_history[self.id, iteration] = self.mdot if self.mdot is not None else 0
            velocity_history[self.id, iteration] = self.u_iterate if self.u_iterate is not None else 0

    def update(self, nextPressure: Optional[float] = None):
        self.fluid.update(Input.temperature(self.temp), Input.pressure(self.pressureIn))
        self.rho = self.fluid.density
        a = self.fluid.sound_speed
        self.pressureOut = nextPressure



    

    def dpIterate(self, mdot=None, pressureIn=None):
        """
        Calculate the pressure drop across the component based on the mass flow rate.
        This method should be overridden by subclasses.
        """
        if mdot is None:
            mdot = self.mdot
        if pressureIn is None:
            pressureIn = self.pressureIn
        raise NotImplementedError("This method should be overridden by subclasses")
        
#fix mdot
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
    def getType(self):  
        return self.type
    def getPressureIn(self):
        return self.pressureIn
    def setPressureIn(self, pressureIn):
        self.pressureIn = pressureIn
        self.fluid.update(Input.pressure(self.pressureIn), Input.temperature(self.temp)) if self.fluid else None

    def getPressureOut(self):
        return self.pressureOut
    def setPressureOut(self, pressureOut):
        self.pressureOut = pressureOut
        self.fluid.update(Input.pressure(self.pressureOut), Input.temperature(self.temp)) if self.fluid else None

    def dp(self):
        raise NotImplementedError("This method should be overridden by subclasses")
    
    def getVelocity(self):
        raise NotImplementedError("This method should be overridden by subclasses")
    
    def solveMdot(self):
        """
        Solve the mass flow rate based on the component's state.
        This method should be overridden by subclasses.
        """
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
        
        # if prevComp is not None:
        #     prevComp: systemComponent
        #     prevComp.setuOut(0)
        #     prevComp.setMdot(0.01)
    
    def getVelocity(self):
        """
        Return the velocity of the ghost cell, which is set to the incoming velocity.
        """
        return self.u_iterate

    def dp(self,cell):
        return cell.getPressureIn() - cell.getPressureOut()
    def setU(self, uIn):
        super().setuIn(uIn)
        super().setuOut(uIn)
    
    def update(self, cell = None):
        self.pressureIn = cell.getPressureOut()
        self.pressureOut =self.pressureIn- self.dp(cell)


    def solveMdot(self, cell):
        cell: systemComponent
        self.mdot = cell.getMdot()
        self.pressureIn = cell.getPressureOut()
        self.pressureOut = self.pressureIn # Default to a small value if no cell is provided



class Pipe(systemComponent):
    def __init__(self, fluid: Fluid=None, length=1.0, diameter=0.1, roughness=0.0001, location=0, pos=None, pressureIn=0, pressureOut=0, temp=None, rho=None, mdot=None): 
        super().__init__(type="p", location=location, pressureIn=pressureIn, pressureOut=pressureOut, length=length, pos=pos, temp=temp, fluid=fluid, rho=rho, mdot=mdot)
        self.diameter = diameter
        self.roughness = roughness
        self.Re: float  # Reynolds number, will be calculated later

        
    def getVelocity(self):
        if self.rho is None or self.mdot is None:
            raise ValueError("Density and mass flow rate must be set before calculating velocity.")
        area = np.pi * (self.diameter / 2) ** 2
        self.u_iterate = self.mdot / (self.rho * area)
        return self.u_iterate

    def getVelocity(self, mdot=None):
        """
        Calculate the velocity based on the mass flow rate and density.
        """
        if mdot is None:
            if self.rho is None or self.mdot is None:
                raise ValueError("Density and mass flow rate must be set before calculating velocity.")
            area = np.pi * (self.diameter / 2) ** 2
            self.u_iterate = self.mdot / (self.rho * area)
            return self.u_iterate
        else:
            if self.rho is None:
                raise ValueError("Density must be set before calculating velocity.")
            area = np.pi * (self.diameter / 2) ** 2
            u_iterate = mdot / (self.rho * area)
            return u_iterate
    def reCalc(self, mdot=None):


       # print(self.pressureIn, self.temp)
        self.fluid.update(Input.temperature(self.temp), Input.pressure(self.pressureIn)) if self.fluid else None
        self.rho = self.fluid.density if self.fluid else self.rho

        mue = self.fluid.dynamic_viscosity
        v = self.getVelocity(mdot=mdot) 
        Re = (self.rho * v * self.diameter) / mue
        return Re

    def reCalcDirectional(self, mdot=None):

        #print(self.pressureIn, self.temp)
        self.fluid.update(Input.temperature(self.temp), Input.pressure(self.pressureIn)) if self.fluid else None
        self.rho = self.fluid.density if self.fluid else self.rho

        mue = self.fluid.dynamic_viscosity
        if mdot is not None:
            v = self.getVelocity(mdot)
        else:
            v = self.u_iterate
        Re = (self.rho * v * self.diameter) / mue
        #print(f"Reynolds number: {Re}")
        return Re

    def frictionFactor(self,mdot=None):
        Re = self.reCalc(mdot=mdot)
        self.Re = Re  # Store the Reynolds number for later use
        def colebrookFF(f):
            return 1 / np.sqrt(f) + 2 * np.log10(self.roughness / (3.7 * self.diameter) + 2.51 / (self.Re * np.sqrt(f)))
    
        Re = abs(Re)  # Ensure Re is non-negative for the Colebrook equation
        if abs(Re) < 3000:
            return 64 / Re
        #using lambert equations for exact solutions, rather than colebrooks equation. Gives accuracy within 
        else:
            return root_scalar(colebrookFF, bracket=[1e-5, 1], method='bisect').root

    def frictionFactorDirectional(self, mdot=None):
        Re = self.reCalcDirectional(mdot=mdot)
        def colebrookFF(f):
            return 1 / np.sqrt(f) + 2 * np.log10(self.roughness / (3.7 * self.diameter) + 2.51 / (self.Re * np.sqrt(f)))
    
        self.Re = abs(Re)  # Ensure Re is non-negative for the Colebrook equation
        if abs(Re) < 3000:
            return 64 / Re
        #using lambert equations for exact solutions, rather than colebrooks equation. Gives accuracy within 
        else:
            return root_scalar(colebrookFF, bracket=[1e-5, 1], method='bisect').root

    
        def colebrookFF(f):
            return 1 / np.sqrt(f) + 2 * np.log10(self.roughness / (3.7 * self.diameter) + 2.51 / (self.Re * np.sqrt(f)))
    
    def darcyWeisbach(self, mdot=None):
        f = self.frictionFactor(mdot=mdot)
        v = self.getVelocity(mdot=mdot)
        return f * (self.length / self.diameter) * (self.rho * v ** 2) / 2
    
    def darcyWeisbachDirectional(self, mdot=None):
        f = self.frictionFactorDirectional(mdot)
        v = self.u_iterate
        return f *(self.length/self.diameter) *(self.rho*v* abs(v))/2
    
    #pressureOut and pressureIn shud be changed 
    def dp(self, pressureIn=None, mdot=None):
        if pressureIn is not None:
            self.pressureIn = pressureIn
            self.fluid.update(Input.pressure(self.pressureIn), Input.temperature(self.temp)) 
        if mdot is not None:
            
            dp=self.darcyWeisbach(mdot=mdot)
            return dp
        else:
            
            dp = self.darcyWeisbach()
            self.pressureOut = self.pressureIn - dp
            return dp

    def dpIterate(self, mdot=None, pressureIn=None):
        if pressureIn is not None:
            self.pressureIn = pressureIn
            self.fluid.update(Input.pressure(self.pressureIn), Input.temperature(self.temp)) 
        if mdot is not None:
            self.mdot = mdot
        dp = self.darcyWeisbachDirectional(mdot=self.mdot)

        return dp
    
    def solve(self, prevVelocity, nextPressure,dt):
        self.update(nextPressure=nextPressure)
        self.solveMdot(outletPressure=nextPressure)
        entropy = self.fluid.entropy
        self.rho = self.fluid.density
        a = self.fluid.sound_speed
        dpdt = self.rho*a**2 *(prevVelocity - self.u_iterate)/self.length
        self.pressureIn+= dpdt * dt
        dudt = ((1/self.length)*(self.pressureIn +self.rho*(self.uIn)**2 - nextPressure - self.rho*(self.uOut)**2) - self.darcyWeisbachDirectional())/self.rho
        self.u_iterate += dudt * dt
        self.fluid.update(Input.pressure(self.pressureIn), Input.entropy(entropy))
        self.temp = self.fluid.temperature
        #self.record()



    
    def update(self, nextPressure: Optional[float] = None):
        self.fluid.update(Input.temperature(self.temp), Input.pressure(self.pressureIn))
        self.rho = self.fluid.density
        a = self.fluid.sound_speed
        self.outletPressure = nextPressure

    def solveMdot(self, inletPressure= None, outletPressure= None):
        if inletPressure is None:
            inletPressure = self.pressureIn
        if outletPressure is None:#change inlet outlet pressures, and update them when calculating pressures
            outletPressure = self.pressureOut
        self.pressureIn = inletPressure  
        self.pressureOut = outletPressure
        self.mdot = None  # Reset mdot before solving
        dpTarget = self.pressureIn - self.pressureOut

        def dpFunc(mdot):
            # Implement the function to calculate dp based on mdot
            accdP = self.dpIterate(mdot=mdot) - dpTarget
            return accdP
        
        # Find a suitable bracket by testing different ranges
        # Start with small values and expand if needed
        brackets_to_try = [
            [-1, 1],
            [-10, 10], 
            
            
            [-0.1, 0.1],
            [0.001, 10],
            [-10, 0.001]
        ]
        
        result = None
        for bracket in brackets_to_try:
            try:
                # Check if the function values have opposite signs
                f_a = dpFunc(bracket[0])
                f_b = dpFunc(bracket[1])
                print(f"Trying bracket {bracket}: f_a={f_a}, f_b={f_b}")
                if f_a * f_b < 0:  # Opposite signs
                    result = root_scalar(dpFunc, bracket=bracket, method='brentq')
                    self.mdot = result.root
                    print(f"Solved mdot: {self.mdot} for bracket {bracket}")
                    break
            except Exception as e:
                print(f"Error with bracket {bracket}: {e}")
                # Try next bracket if this one fails
                continue
        
        if result is None:
            # If no bracket works with brentq, try a different method
            try:
                # Try with fsolve as fallback
                from scipy.optimize import fsolve
                initial_guess = dpTarget / 1000 if abs(dpTarget) > 0 else 0.01  # Simple heuristic
                solution = fsolve(dpFunc, initial_guess)
                self.mdot = solution[0]
            except:
                # Last resort: set a small default value and warn
                print(f"Warning: Could not solve for mdot in pipe {self.id}. Using default value.")
                self.mdot = 0.001 if dpTarget > 0 else -0.001

# def solveMdot(self, inletPressure= None, outletPressure= None):
    #     if inletPressure is None:
    #         inletPressure = self.pressureIn
    #     if outletPressure is None:#change inlet outlet pressures, and update them when calculating pressures
    #         outletPressure = self.pressureOut
    #     self.inletPressure = inletPressure  
    #     self.outletPressure = outletPressure
    #     self.mdot = None  # Reset mdot before solving
    #     dpTarget = self.inletPressure - self.outletPressure
    #     def dpFunc(mdot):
    #         # Implement the function to calculate dp based on mdot
    #         accdP = self.dpIterate(mdot=mdot) - dpTarget
    #         return accdP
    #     # Use root_scalar to find the mdot that gives the desired dp
    #     result = root_scalar(dpFunc, bracket=[-100, 100], method='brentq')
    #     self.mdot = result.root
    def record(self):
        # Keep backward compatibility with legacy lists
        self.pressureThroughTime.append(self.pressureIn)
        self.tempThroughTime.append(self.temp)
        self.mdotThroughTime.append(self.mdot)
        

class Orifice(systemComponent):
    def __init__(self, fluid: Fluid, OD=0.05, ID=0.04, location=0, pos=None, pressureIn=0, pressureOut=0, temp=None, rho=None, mdot=None, l=0.01, CdA=0.01):
        super().__init__(type='o', location=location, pos=pos, pressureIn=pressureIn, pressureOut=pressureOut, temp=temp, rho=rho, mdot=mdot, fluid=fluid)
        self.outerD = OD
        self.innerD = ID
        self.length = l  # Length of the orifice
        self.pressureOut = pressureOut  # Pressure at the outlet of the orifice
        self.CdA = CdA
        

    def dp(self, pressureIn=None, mdot=None, timestep=None):
        if pressureIn is not None:
            self.pressureIn = pressureIn
        if mdot is not None:
            self.mdot = mdot
        self.fluid.update(Input.pressure(self.pressureIn), Input.temperature(self.temp))
        self.rho = self.fluid.density
        area = np.pi * (self.innerD / 2) ** 2
        return (self.mdot /(self.CdA*np.sqrt(2/self.rho)))
    

    def solveMdot(self):
        targetDp = self.pressureIn - self.pressureOut
        def dpFunc(mdot):
            # Calculate the pressure drop based on the current mass flow rate
            dp = self.dp(mdot=mdot)
            return dp - targetDp

        # Use root_scalar to find the mdot that gives the desired dp
        result = root_scalar(dpFunc, bracket=[-100, 10], method='brentq')
        self.mdot = result.root

        



def barA(pa):
    """
    Convert pressure from bar to Pascals.
    """
    return pa * 1e5

if __name__ == "__main__":
    # Example usage
    fluid = Fluid(FluidsList.Oxygen)
    fluid.update(Input.temperature(-180), Input.pressure(barA(100)))
    simTime = 3 #s
    timeIterations= 1000
    dt = simTime / timeIterations
    feed_system = FeedSystemCriticalPath(dt = dt,totalPipeLength=1.0, fluid=fluid, N=5, mdot=1, inletPressure=barA(100), outletPressure=barA(50))

    dL = feed_system.dL
    initial_velocity = feed_system.discretisedFeed[1].getVelocity()
    sound_speed = fluid.sound_speed
    cfl_dt = dL / (abs(initial_velocity) + sound_speed)
    # Use a safety factor (e.g., 0.5) and choose the smaller of user-defined or CFL dt
    dt = min(simTime / timeIterations, 0.5 * cfl_dt)
    feed_system.dt = dt / 100
    timeIterations = round(simTime / feed_system.dt)
    feed_system.setMaxIterations(timeIterations+100)
    print(f"Sim time: {feed_system.dt * timeIterations} s, dt: {feed_system.dt} s, dL: {dL} m, CFL dt: {cfl_dt} s")
    print(timeIterations, "iterations")

    for i in range(timeIterations):
        feed_system.solve()
        if i > 0 and i % 100 == 0:
            print(f"Progress: {i / timeIterations * 100:.5f}%", end='\r')

    print("Progress: 100.00000%", end='\r')

    output_folder = f"simulation_results_{dt}_{simTime}"
    
    # Write CSV files to the new folder
    feed_system.write_to_csv(
        pressure_filename="pressure_results.csv", 
        massflow_filename="massflow_results.csv",
        temperature_filename="temperature_results.csv",
        output_dir=output_folder        
    )        
    

    print(f"\nSimulation completed. Data written to {output_folder}")