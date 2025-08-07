from typing import List, Optional
from scipy.optimize import root_scalar, fsolve
from pyfluids import Fluid, FluidsList, Input
from thermo.chemical import Chemical
from os import system 
import numpy as np
import csv
import os
from numba import jit, njit, prange
from numba.types import float64

# JIT-compiled utility functions for maximum performance
@njit
def friction_factor_laminar(re):
    """Laminar flow friction factor - JIT compiled"""
    return 64.0 / abs(re) if abs(re) > 1e-10 else 0.0

@njit
def friction_factor_turbulent_approx(re, roughness_ratio):
    """Approximate turbulent friction factor (Swamee-Jain) - JIT compiled"""
    re_abs = abs(re)
    if re_abs < 1e-10:
        return 0.0
    return 1.325 / (np.log((roughness_ratio / 3.7) + (5.74 / re_abs ** 0.9))) ** 2

@njit
def calculate_friction_factor(re, roughness, diameter):
    """Fast friction factor calculation - JIT compiled"""
    roughness_ratio = roughness / diameter
    if abs(re) < 3000:
        return friction_factor_laminar(re)
    else:
        return friction_factor_turbulent_approx(re, roughness_ratio)

@njit
def darcy_weisbach_jit(friction_factor, length, diameter, rho, velocity):
    """JIT-compiled Darcy-Weisbach equation"""
    return friction_factor * (length / diameter) * (rho * velocity * abs(velocity)) / 2

@njit
def reynolds_number_jit(rho, velocity, diameter, viscosity):
    """JIT-compiled Reynolds number calculation"""
    if abs(viscosity) < 1e-10:
        return 0.0
    return (rho * abs(velocity) * diameter) / viscosity

@njit
def pressure_wave_update(pressure_in, rho, sound_speed, du_dx, dt, damping=0.5):
    """JIT-compiled pressure wave equation"""
    dpdt = -rho * sound_speed**2 * du_dx
    return pressure_in + damping * dpdt * dt

@njit
def momentum_update(velocity, rho, dp_dx, friction_term, dt, damping=0.5):
    """JIT-compiled momentum equation"""
    if abs(rho) < 1e-10:
        return velocity
    dudt = -(1.0 / rho) * (dp_dx + friction_term)
    return velocity + damping * dudt * dt

@njit
def orifice_dp_jit(mdot, cd_a, rho):
    """JIT-compiled orifice pressure drop calculation"""
    if abs(rho) < 1e-10 or abs(cd_a) < 1e-10:
        return 0.0
    return (mdot / cd_a)**2 / (2 * rho)

@njit
def clamp_value(value, min_val, max_val):
    """JIT-compiled value clamping"""
    return max(min_val, min(max_val, value))

@njit
def update_history_arrays_jit(pressure_history, temp_history, mdot_history, velocity_history,
                             cell_id, pressure, temperature, mdot, velocity, iteration):
    """JIT-compiled efficient array update"""
    if cell_id >= 0 and iteration < pressure_history.shape[1] and cell_id < pressure_history.shape[0]:
        pressure_history[cell_id, iteration] = pressure
        temp_history[cell_id, iteration] = temperature
        mdot_history[cell_id, iteration] = mdot
        velocity_history[cell_id, iteration] = velocity


class FeedSystemCriticalPath:
    def __init__(self, dt=1, totalPipeLength=1.0, fluid: Fluid=None, N=2, extraComponents=[[None,None],[None,None]], 
                 mdot: float = None, inletPressure: float = None, outletPressure: float = None, max_iterations=100000):
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
        self.rampStartCDA = 194e-6  # Initial CdA value
        # Pre-allocate numpy arrays for all time history data
        self.pressure_history = np.zeros((N + 3, max_iterations))  # +3 for ghost cells and exit orifice outlet
        self.temp_history = np.zeros((N + 3, max_iterations))
        self.mdot_history = np.zeros((N + 3, max_iterations))
        self.velocity_history = np.zeros((N + 3, max_iterations))
        fluid.update(Input.temperature(-180), Input.pressure(inletPressure))
        self.initTemp = fluid.temperature
        self.populate()

    def initCont(self):
        #end orifice ramping:
        rampStartTime = 0.05  # seconds
        rampTime = 0.1  # seconds
        rampStartCDA = self.rampStartCDA  # Initial CdA value
        rampEndCDA = 50e-6  # Final CdA value
        rampNum = np.linspace(rampStartCDA, rampEndCDA, int(rampTime / self.dt))
        self.CdAList = []
        for i in range(len(rampNum)):
            self.CdAList.append((rampNum[i], i + int(rampStartTime/self.dt)))
        self.discretisedFeed[-1].setCdAList(self.CdAList)  # Set CdA list for the exit orifice

        


        

    def setMaxIterations(self, max_iterations):
        self.max_iterations = max_iterations
        self.pressure_history = np.zeros((self.N + 3, max_iterations))
        self.temp_history = np.zeros((self.N + 3, max_iterations))
        self.mdot_history = np.zeros((self.N + 3, max_iterations))
        self.velocity_history = np.zeros((self.N + 3, max_iterations))

    def discretise(self):
        self.discretisedFeed = []
        check = True
        for comp in self.extraComponents:
            if comp[0] is not None:
                comp[1] = round(comp[1] / self.dL)
                check = True
            else:
                check = False
        
        if check:
            self.extraComponents.sort(key=lambda comp: comp[0])

        count = 0
        for i in range(self.N):
            pos = count * self.dL
            if i in [comp[1] for comp in self.extraComponents if comp[0] is not None]:
                pass
            else:
                count += 1
                pipe = PipeJIT(fluid=self.fluid, length=self.dL, mdot=self.mdot, pos=pos, 
                              location=i+1, temp=self.initTemp, diameter=0.035)
            self.discretisedFeed.append(pipe)

    def populate(self):
        currentPressure = self.inletPressure
        self.discretise()
        self.discretisedFeed.append(OrificeJIT(location=self.N+1, CdA=self.rampStartCDA,fluid=self.fluid, ID=0.035, pos=self.totalPipeLength, 
                                              pressureIn=self.discretisedFeed[-1].getPressureOut(), 
                                              pressureOut=self.outletPressure, temp=self.initTemp, mdot=self.mdot, type="oe"))
        self.solveMdot(self.inletPressure, self.outletPressure)
        print(self.mdot)
        # Initialize pressures properly with steady-state pressure drop
        for i, component in enumerate(self.discretisedFeed):
            if component.type == "p":
                component.pressureIn = currentPressure
                dp = component.dp(currentPressure, self.mdot)
                currentPressure -= dp
                component.pressureOut = currentPressure
                print(f"Pipe {i}: Pin={component.pressureIn/1e5:.2f} bar, Pout={component.pressureOut/1e5:.2f} bar, dp={dp/1e5:.4f} bar")
            elif component.type == "o":
                component.pressureIn = self.inletPressure
                component.pressureOut = self.outletPressure
            elif component.type == "oe":
                component.pressureIn = currentPressure
                component.pressureOut = self.outletPressure
                dp = component.dp(currentPressure, self.mdot)
                print(f"Orifice: Pin={component.pressureIn/1e5:.2f} bar, Pout={component.pressureOut/1e5:.2f} bar, dp={dp/1e5:.4f} bar")
            else:
                raise ValueError(f"Unknown component type: {component.type}")

        self.discretisedFeed.insert(0, ghostCellJIT(location=0, u=self.discretisedFeed[0].getVelocity(), pos=0, 
                                                   pressureIn=self.inletPressure, pressureOut=self.inletPressure, 
                                                   mdot=self.mdot, temp=self.initTemp))
        self.boundaryPopulation()

    def getSystemDP(self, mdot):
        return sum(comp.dp(mdot=mdot, pressureIn=self.inletPressure) for comp in self.discretisedFeed if comp.getType() != "g")

    def solveMdot(self, inletPressure: Optional[float] = None, outletPressure: Optional[float] = None):
        dpTarget = inletPressure - outletPressure
        
        def dpFunc(mdot):
            return self.getSystemDP(mdot) - dpTarget
    
        brackets_to_try = [[0.001, 10], [0.0001, 1], [0.01, 100], [-1, 1], [-10, 10]]
        
        result = None
        for bracket in brackets_to_try:
            try:
                result = root_scalar(dpFunc, bracket=bracket, method='brentq')
                self.mdot = result.root
                break
            except Exception as e:
                continue
        
        if result is None:
            try:
                result = fsolve(dpFunc, 1.0)[0]

                self.mdot = result
            except:
                raise ValueError("Failed to solve for mass flow rate with any method.")

    def boundaryPopulation(self):
        for i, component in enumerate(self.discretisedFeed):
            if component.type == "g" or component.type == "oe":
                continue
            cellV = component.getVelocity()
            
            if cellV < 0:
                component.setuIn(cellV)
                component.setuOut(self.discretisedFeed[i+1].getVelocity())
            else:
                component.setuIn(self.discretisedFeed[i-1].getVelocity())
                component.setuOut(cellV)

            component.record()

    def solve(self):
        """Optimized solve method using JIT where possible"""
        # Enforce boundary conditions first
        self.discretisedFeed[0].setPressureIn(self.inletPressure)  # Inlet ghost cell
        #self.discretisedFeed[-1].setPressureIn(self.outletPressure)  # Outlet ghost cell/orifice
        
        # First pass: Update internal cells
        for i, component in enumerate(self.discretisedFeed):
            if component.type == "g":
                component: ghostCellJIT
                if i == 0:  # Inlet ghost cell
                    component.setPressureIn(self.inletPressure)
                    component.setPressureOut(self.inletPressure)
                    component.setuIn(self.discretisedFeed[i+1].getuIn())
                    component.setuOut(self.discretisedFeed[i+1].getuIn())
                    component.set_u_iterate(self.discretisedFeed[i+1].get_u_iterate())
                elif i == len(self.discretisedFeed) - 1:  # Outlet ghost cell
                    component.setPressureIn(self.outletPressure)
                    component.setPressureOut(self.outletPressure)
                    component.setuIn(self.discretisedFeed[i-1].getuOut())
                    component.setuOut(self.discretisedFeed[i-1].getuOut())
                    component.set_u_iterate(self.discretisedFeed[i-1].get_u_iterate())
                continue
                
            if component.type == "oe":
                component: OrificeJIT
                component.solve(self.discretisedFeed[i-1].getPressureOut())
                # Enforce outlet pressure for exit orifice
                component.pressureOut = self.outletPressure
            elif component.type == "o":
                component: OrificeJIT
                component.solve(self.discretisedFeed[i-1].getPressureOut(), self.discretisedFeed[i+1])
                self.upWinding(component, self.discretisedFeed[i-1], self.discretisedFeed[i+1])
            else:
                component: PipeJIT
                prev_pressure = self.discretisedFeed[i-1].getPressureIn() if i > 0 else self.inletPressure
                next_pressure = self.discretisedFeed[i+1].getPressureIn() if i < len(self.discretisedFeed)-1 else self.outletPressure
                prev_velocity = self.discretisedFeed[i-1].get_u_iterate() if i > 0 else 0
                
                component.solve_jit(prev_velocity, next_pressure, self.dt)
                self.upWinding(component, self.discretisedFeed[i-1], self.discretisedFeed[i+1])
        
        # Second pass: Update mass flows and record data
        for i, component in enumerate(self.discretisedFeed):
            if component.type == "g":
                continue
                
            if component.type == "oe":
                component.solveMdot(inletPressure=self.discretisedFeed[i-1].getPressureOut())
            else:
                next_pressure = self.discretisedFeed[i+1].getPressureIn()
                component.solveMdot(outletPressure=next_pressure)
            # Efficient array recording using JIT
            component.record_to_arrays_jit(self.pressure_history, self.temp_history, 
                                          self.mdot_history, self.velocity_history, 
                                          self.current_iteration)
        
        self.current_iteration += 1

    def upWinding(self, current, previous, next):
        cellV = current.get_u_iterate()
        if cellV >= 0:
            current.setuIn(previous.get_u_iterate())
            current.setuOut(cellV)
        else:
            current.setuIn(cellV)
            current.setuOut(next.get_u_iterate())

    def get_cell_data(self, cell_id, up_to_iteration=None):
        if up_to_iteration is None:
            up_to_iteration = self.current_iteration
        
        return {
            'pressure': self.pressure_history[cell_id, :up_to_iteration],
            'temperature': self.temp_history[cell_id, :up_to_iteration],
            'mdot': self.mdot_history[cell_id, :up_to_iteration],
            'velocity': self.velocity_history[cell_id, :up_to_iteration]
        }

    def get_final_snapshot(self):
        final_iter = self.current_iteration - 1
        return {
            'pressure': self.pressure_history[:, final_iter],
            'temperature': self.temp_history[:, final_iter],
            'mdot': self.mdot_history[:, final_iter],
            'velocity': self.velocity_history[:, final_iter]
        }

    def write_to_csv(self, pressure_filename="pressure_data.csv", massflow_filename="massflow_data.csv", 
                     temperature_filename="temperature_results.csv", output_dir="simulation_results"):
        """Write data to CSV files with metadata"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created directory: {output_dir}")
        
        pressure_path = os.path.join(output_dir, pressure_filename)
        massflow_path = os.path.join(output_dir, massflow_filename)
        temperature_path = os.path.join(output_dir, temperature_filename)
        metadata_path = os.path.join(output_dir, "simulation_metadata.csv")
        
        real_cells = [c for c in self.discretisedFeed if c.type != 'g']
        real_cell_ids = [c.getID() for c in real_cells]
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
        
        # Write pressure data
        with open(pressure_path, 'w', newline='') as pressure_file:
            writer = csv.writer(pressure_file)
            writer.writerow(['# Simulation Metadata'])
            writer.writerow([f'# Total Iterations: {self.current_iteration}'])
            writer.writerow([f'# Time Step (dt): {self.dt} s'])
            writer.writerow(['# Data begins below'])
            
            # Create headers with separate inlet/outlet for orifices
            header = ['Time (s)']
            data_indices = []  # Track which array indices to use for each column
            
            for cell in real_cells:
                if cell.getType() in ['o', 'oe']:  # Orifices
                    header.append(f'{cell.getType()}_{cell.getID()}_Pressure_In (Pa)')
                    header.append(f'{cell.getType()}_{cell.getID()}_Pressure_Out (Pa)')
                    data_indices.append(cell.getID())      # Inlet pressure
                    data_indices.append(cell.getID() + 1)  # Outlet pressure
                else:  # Pipes and other components
                    header.append(f'{cell.getType()}_{cell.getID()}_Pressure (Pa)')
                    data_indices.append(cell.getID())
            
            writer.writerow(header)
            
            for i in range(self.current_iteration):
                row = [time_array[i]]
                for data_idx in data_indices:
                    row.append(self.pressure_history[data_idx, i])
                writer.writerow(row)
        
        # Write mass flow data
        with open(massflow_path, 'w', newline='') as massflow_file:
            writer = csv.writer(massflow_file)
            writer.writerow(['# Simulation Metadata'])
            writer.writerow([f'# Total Iterations: {self.current_iteration}'])
            writer.writerow(['# Data begins below'])
            
            # Create headers with separate inlet/outlet for orifices
            header = ['Time (s)']
            data_indices = []  # Track which array indices to use for each column
            
            for cell in real_cells:
                if cell.getType() in ['o', 'oe']:  # Orifices
                    header.append(f'{cell.getType()}_{cell.getID()}_Massflow_In (kg/s)')
                    header.append(f'{cell.getType()}_{cell.getID()}_Massflow_Out (kg/s)')
                    data_indices.append(cell.getID())      # Inlet massflow
                    data_indices.append(cell.getID() + 1)  # Outlet massflow
                else:  # Pipes and other components
                    header.append(f'{cell.getType()}_{cell.getID()}_Massflow (kg/s)')
                    data_indices.append(cell.getID())
            
            writer.writerow(header)
            
            for i in range(self.current_iteration):
                row = [time_array[i]]
                for data_idx in data_indices:
                    row.append(self.mdot_history[data_idx, i])
                writer.writerow(row)
        
        # Write temperature data
        with open(temperature_path, 'w', newline='') as temperature_file:
            writer = csv.writer(temperature_file)
            writer.writerow(['# Simulation Metadata'])
            writer.writerow([f'# Total Iterations: {self.current_iteration}'])
            writer.writerow(['# Data begins below'])
            
            # Create headers with separate inlet/outlet for orifices
            header = ['Time (s)']
            data_indices = []  # Track which array indices to use for each column
            
            for cell in real_cells:
                if cell.getType() in ['o', 'oe']:  # Orifices
                    header.append(f'{cell.getType()}_{cell.getID()}_Temperature_In (K)')
                    header.append(f'{cell.getType()}_{cell.getID()}_Temperature_Out (K)')
                    data_indices.append(cell.getID())      # Inlet temperature
                    data_indices.append(cell.getID() + 1)  # Outlet temperature
                else:  # Pipes and other components
                    header.append(f'{cell.getType()}_{cell.getID()}_Temperature (K)')
                    data_indices.append(cell.getID())
            
            writer.writerow(header)
            
            for i in range(self.current_iteration):
                row = [time_array[i]]
                for data_idx in data_indices:
                    row.append(self.temp_history[data_idx, i])
                writer.writerow(row)
        # Print summary
        print(f"Data written to:")
        print(f"  Pressure: {pressure_path}")
        print(f"  Mass flow: {massflow_path}")
        print(f"  Metadata: {metadata_path}")
        print(f"Data shape: {self.current_iteration} time steps x {len(real_cell_ids)} cells")


class systemComponentJIT:
    def __init__(self, type, location=None, pressureIn=None, pressureOut=None, length=None, pos=None, 
                 temp=None, fluid=None, rho=None, mdot=None, initP=None, initT=None, initMdot=None):
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
        
        # Legacy lists for backward compatibility
        self.pressureThroughTime = [initP] if initP is not None else []
        self.tempThroughTime = [initT] if initT is not None else []
        self.mdotThroughTime = [initMdot] if initMdot is not None else []
        self.uIn = None
        self.uOut = None
        self.u_iterate = None

    def record_to_arrays_jit(self, pressure_history, temp_history, mdot_history, velocity_history, iteration):
        
        if self.id is not None:
            update_history_arrays_jit(pressure_history, temp_history, mdot_history, velocity_history,
                                     self.id, 
                                     self.pressureIn if self.pressureIn is not None else 0,
                                     self.temp if self.temp is not None else 0,
                                     self.mdot if self.mdot is not None else 0,
                                     self.u_iterate if self.u_iterate is not None else 0,
                                     iteration)

    def record_to_arrays(self, pressure_history, temp_history, mdot_history, velocity_history, iteration):
        """Legacy method for compatibility"""
        self.record_to_arrays_jit(pressure_history, temp_history, mdot_history, velocity_history, iteration)

    # Standard getters/setters
    def getID(self): return self.id
    def getuIn(self): return self.uIn
    def setuIn(self, uIn): self.uIn = uIn
    def getPos(self): return self.pos
    def getuOut(self): return self.uOut
    def setuOut(self, uOut): self.uOut = uOut
    def setMdot(self, mdot): self.mdot = mdot
    def getMdot(self): return self.mdot
    def get_u_iterate(self): return self.u_iterate
    def set_u_iterate(self, u_iterate): self.u_iterate = u_iterate
    def getType(self): return self.type
    def getPressureIn(self): return self.pressureIn
    def setPressureIn(self, pressureIn): 
        self.pressureIn = pressureIn
        if self.fluid:
            self.fluid.update(Input.pressure(self.pressureIn), Input.temperature(self.temp))
    def getPressureOut(self): return self.pressureOut
    def setPressureOut(self, pressureOut): 
        self.pressureOut = pressureOut
        if self.fluid:
            self.fluid.update(Input.pressure(self.pressureOut), Input.temperature(self.temp))

    def update(self, nextPressure = None):
        
        self.fluid.update(Input.temperature(self.temp), Input.pressure(self.pressureIn))
        self.rho = self.fluid.density
        self.viscosity = self.fluid.dynamic_viscosity

        
        if nextPressure:
            self.pressureOut = nextPressure

    def record(self):
        self.pressureThroughTime.append(self.pressureIn)
        self.tempThroughTime.append(self.temp)
        self.mdotThroughTime.append(self.mdot)


class ghostCellJIT(systemComponentJIT):  
    def __init__(self, u=None, pos=None, pressureIn=None, pressureOut=None, temp=None, rho=None, mdot=None, prevComp=None, location=None):
        super().__init__(type="g", pos=pos, pressureIn=pressureIn, pressureOut=pressureOut, temp=temp, rho=rho, mdot=mdot, location=location)
        self.length = 0.0
        self.uIn = u
        self.uOut = self.uIn
        self.u_iterate = u
    
    def getVelocity(self):
        return self.u_iterate

    def dp(self, cell):
        return cell.getPressureIn() - cell.getPressureOut()
    
    def setU(self, uIn):
        self.setuIn(uIn)
        self.setuOut(uIn)
    
    def update(self, cell=None):
        if cell:
            self.pressureIn = cell.getPressureOut()
            self.pressureOut = self.pressureIn - self.dp(cell)

    def solveMdot(self, cell=None):
        if cell:
            self.mdot = cell.getMdot()
            self.pressureIn = cell.getPressureOut()
            self.pressureOut = self.pressureIn


class PipeJIT(systemComponentJIT):
    def __init__(self, fluid: Fluid=None, length=1.0, diameter=0.1, roughness=0.0001, location=0, pos=None, 
                 pressureIn=0, pressureOut=0, temp=None, rho=None, mdot=None): 
        super().__init__(type="p", location=location, pressureIn=pressureIn, pressureOut=pressureOut, 
                        length=length, pos=pos, temp=temp, fluid=fluid, rho=rho, mdot=mdot)
        self.diameter = diameter
        self.roughness = roughness
        self.Re: float = 0.0

    def getVelocity(self, mdot=None):
        if mdot is None:
            mdot = self.mdot
        if self.rho is None or mdot is None:
            raise ValueError("Density and mass flow rate must be set before calculating velocity.")
        area = np.pi * (self.diameter / 2) ** 2
        self.u_iterate = mdot / (self.rho * area)
        return self.u_iterate

    def solve_jit(self, prevVelocity, nextPressure, dt):
        """JIT-optimized solve method"""
        # Input validation
        if nextPressure <= 0:
            nextPressure = max(1000, self.pressureOut)
        if abs(prevVelocity) > 500:
            prevVelocity = np.sign(prevVelocity) * 500
            
        # Update fluid properties (can't JIT this part due to fluid library)
        if self.fluid:
            self.fluid.update(Input.temperature(self.temp), Input.pressure(self.pressureIn))
            rho = self.fluid.density
            sound_speed = self.fluid.sound_speed
            viscosity = self.fluid.dynamic_viscosity
        else:
            rho = self.rho if self.rho else 1000
            sound_speed = 343
            viscosity = 1e-3
        
        # JIT-compiled numerical computation
        new_pressure, new_velocity, new_mdot = self._solve_numerical_jit(
            self.pressureIn, self.u_iterate, nextPressure, 
            rho, sound_speed, viscosity, self.uOut if self.uOut else 0, self.uIn if self.uIn else 0,
            self.length, self.diameter, self.roughness, dt
        )
        
        # Update state
        self.pressureIn = new_pressure
        self.u_iterate = new_velocity
        self.mdot = new_mdot
        self.rho = rho
        
        # Update fluid state
        if self.fluid:
            try:
                self.fluid.update(Input.pressure(self.pressureIn), Input.temperature(self.temp))
                self.temp = self.fluid.temperature
                self.rho = self.fluid.density
                self.viscosity = self.fluid.dynamic_viscosity   

            except:
                pass

    @staticmethod
    @njit
    def _solve_numerical_jit(pressure_in, velocity, next_pressure, rho, sound_speed, 
                           viscosity, u_out, u_in, length, diameter, roughness, dt):
        """JIT-compiled numerical solver for pipe"""
        damping_factor = 1
        
        # Calculate derivatives
        du_dx = (u_out - u_in) / length
        dp_dx = (next_pressure - pressure_in) / length
        
        # Calculate friction using JIT functions
        re = reynolds_number_jit(rho, velocity, diameter, viscosity)
        f = calculate_friction_factor(re, roughness, diameter)
        
        # Calculate friction pressure loss per unit length
        friction_dp_per_length = f * (1.0 / diameter) * (rho * velocity * abs(velocity)) / 2
        
        # Pressure wave equation: dp/dt = -rho * a^2 * (du/dx) - friction_loss_rate
        # The friction term reduces pressure over time
        dpdt = -rho * sound_speed * sound_speed * du_dx - friction_dp_per_length
        
        # Momentum equation: du/dt = -(1/rho) * (dp/dx + friction_force_per_length)
        friction_force_per_length = friction_dp_per_length
        dudt = -(1.0 / rho) * (dp_dx + friction_force_per_length)
        
        # Update with damping for stability
        new_pressure = pressure_in + damping_factor * dpdt * dt
        new_velocity = velocity + damping_factor * dudt * dt
        
        # Clamp values using JIT function
        #new_pressure = clamp_value(new_pressure, 1000.0, 1e8)
        #new_velocity = clamp_value(new_velocity, -500.0, 500.0)
        
        # Calculate mass flow
        area = np.pi * (diameter / 2)**2
        new_mdot = rho * new_velocity * area
        new_mdot = clamp_value(new_mdot, -1000.0, 1000.0)
        
        return new_pressure, new_velocity, new_mdot

    def dp(self, pressureIn=None, mdot=None):
        if pressureIn is None:
            pressureIn = self.pressureIn
      
        if mdot is None:
            mdot= self.mdot

        self.mdot = mdot    
        if self.fluid:
            self.fluid.update(Input.pressure(pressureIn), Input.temperature(self.temp))
            self.rho = self.fluid.density
            viscosity = self.fluid.dynamic_viscosity
        else:
            viscosity = 1e-3
            
        # Use JIT-compiled calculation
        velocity = mdot / (self.rho * np.pi * (self.diameter / 2) ** 2)
        re = reynolds_number_jit(self.rho, velocity, self.diameter, viscosity)
        f = calculate_friction_factor(re, self.roughness, self.diameter)
        return darcy_weisbach_jit(f, self.length, self.diameter, self.rho, velocity)

    def solveMdot(self, inletPressure=None, outletPressure=None):
        if inletPressure is None:
            inletPressure = self.pressureIn
        if outletPressure is None:
            outletPressure = self.pressureOut

        dpTarget = inletPressure - outletPressure

        def dpFunc(mdot):
            return self.dp(mdot=mdot) - dpTarget
        
        brackets_to_try = [[-1, 1], [-10, 10], [-100, 100], [-0.1, 0.1], [0.001, 10], [-10, 0.001]]
        
        result = None
        for bracket in brackets_to_try:
            try:
                result = root_scalar(dpFunc, bracket=bracket, method='brentq')
                self.mdot = result.root
                break
            except Exception:
                continue
        
        if result is None:
            self.mdot = 1.0

    def solve(self, prevVelocity, nextPressure, dt):
        """Fallback to JIT version"""
        self.solve_jit(prevVelocity, nextPressure, dt)


class OrificeJIT(systemComponentJIT):
    def __init__(self, fluid: Fluid, OD=0.05, ID=0.04, location=0, pos=None, pressureIn=0, pressureOut=0, 
                 temp=None, rho=None, mdot=None, l=0.01, CdA=None, type='o'):
        super().__init__(type=type, location=location, pos=pos, pressureIn=pressureIn, pressureOut=pressureOut, 
                        temp=temp, rho=rho, mdot=mdot, fluid=fluid)
        self.outerD = OD
        self.innerD = ID
        self.length = l
        self.pressureOut = pressureOut
        self.dynamic = False
        self.CdAList = [194e-6, 0] if CdA is None else CdA
        self.CdA = CdA
        self.iteration = 0
        self.constPOut = pressureOut
        self.e = 1/861
        if self.dynamic:
            self.CdA_dict = {iter_index: cdA_value for cdA_value, iter_index in self.CdAList}
        else:
            self.CdA_dict = {}
    
    def setCdAList(self, CdAList):
        """Set the CdA list for dynamic orifices"""
        self.CdAList = CdAList
        self.dynamic = True
        self.CdA_dict = {iter_index: cdA_value for cdA_value, iter_index in CdAList}
        self.CdA = CdAList[0][0]
    def getVelocity(self):
        if self.rho is None or self.mdot is None:
            raise ValueError("Density and mass flow rate must be set before calculating velocity.")
        area = np.pi * (self.innerD / 2) ** 2
        self.u_iterate = self.mdot / (self.rho * area)
        return self.u_iterate

    def dp(self, pressureIn=None, mdot=None):
        if pressureIn is None:
            pressureIn = self.pressureIn
        if mdot is not None:
            self.mdot = mdot
            
        if self.fluid:
            self.fluid.update(Input.pressure(pressureIn), Input.temperature(self.temp))
            self.rho = self.fluid.density
        
        # Use JIT-compiled orifice calculation
        return orifice_dp_jit(self.mdot, self.CdA, self.rho)

    def solveMdot(self, inletPressure=None, outletPressure=None):
        if inletPressure is None:
            inletPressure = self.pressureIn
        if outletPressure is None:
            outletPressure = self.pressureOut

        targetDp = inletPressure - outletPressure

        def dpFunc(mdot):
            return self.dp(mdot=mdot) - targetDp

        try:
            result = root_scalar(dpFunc, bracket=[-10, 100], method='brentq')
            self.mdot = result.root
        except:
            self.mdot = 1.0

    def solve(self, prevPressure, nextCell=None):
        self.pressureIn = prevPressure
        self.update()
        self.pressureOut = self.pressureIn - self.dp()
        self.getVelocity()
        if self.dynamic:
            self.iteration += 1
            # O(1) dictionary lookup instead of O(n) list search
            if self.iteration in self.CdA_dict:
                self.CdA = self.CdA_dict[self.iteration]

        if nextCell is not None:
            nextCell.setPressureIn(self.pressureOut)
    
    def record_to_arrays_jit(self, pressure_history, temp_history, mdot_history, velocity_history, iteration):
        """Record both inlet and outlet pressures for orifices"""
        if self.id is not None:
            # Record inlet pressure at self.id
            update_history_arrays_jit(pressure_history, temp_history, mdot_history, velocity_history,
                                     self.id, 
                                     self.pressureIn if self.pressureIn is not None else 0,
                                     self.temp if self.temp is not None else 0,
                                     self.mdot if self.mdot is not None else 0,
                                     self.u_iterate if self.u_iterate is not None else 0,
                                     iteration)
            # Record outlet pressure at self.id + 1
            update_history_arrays_jit(pressure_history, temp_history, mdot_history, velocity_history,
                                     self.id + 1, 
                                     self.pressureOut if self.pressureOut is not None else 0,
                                     self.temp if self.temp is not None else 0,
                                     self.mdot if self.mdot is not None else 0,
                                     self.u_iterate if self.u_iterate is not None else 0,
                                     iteration)

def barA(pa):
    """Convert pressure from bar to Pascals."""
    return pa * 1e5


if __name__ == "__main__":
    print("Starting JIT-optimized simulation...")
    print("Note: First iteration may be slow due to JIT compilation")
    
    # Example usage with JIT optimization
    fluid = Fluid(FluidsList.Oxygen)
    fluid.update(Input.temperature(-180), Input.pressure(barA(100)))
    simTime = 0.2# s
    timeIterations = 1000
    
    feed_system = FeedSystemCriticalPath(dt=0.001, totalPipeLength=9.0, fluid=fluid, N=10, mdot=1, 
                                        inletPressure=barA(100), outletPressure=barA(80))

    dL = feed_system.dL
    initial_velocity = feed_system.discretisedFeed[1].getVelocity()
    sound_speed = fluid.sound_speed
    
    # Proper CFL condition: dt < dx / (|u| + a)
    cfl_dt = dL / (abs(initial_velocity) + sound_speed)
    
    # Use a much more conservative safety factor
    dt = 0.1 * cfl_dt  # Very conservative
    dt = dt/200
    feed_system.dt = dt
    timeIterations = int(simTime / dt)
    feed_system.setMaxIterations(timeIterations + 100)
    feed_system.initCont()
    
    print(f"Sim time: {simTime} s, dt: {dt:.2e} s, dL: {dL} m, CFL dt: {cfl_dt:.2e} s")
    print(f"Sound speed: {sound_speed:.1f} m/s, Initial velocity: {initial_velocity:.3f} m/s")
    print(f"{timeIterations} iterations")
    
    # Print initial pressure distribution
    print("\nInitial pressure distribution:")
    for i, comp in enumerate(feed_system.discretisedFeed):
        if comp.type != 'g':
            print(f"  {comp.type}_{comp.getID()}: {comp.getPressureIn()/1e5:.3f} bar")
    
    import time as time_module
    start_time = time_module.time()

    for i in range(timeIterations):
        feed_system.solve()
        
        # Check for instability every 10 iterations
        if i % 10 == 0 and i > 0:
            max_pressure = np.max(feed_system.pressure_history[:, i])
            min_pressure = np.min(feed_system.pressure_history[:, i])
            
            if max_pressure > 1e8 or min_pressure < 0:
                print(f"Instability detected at iteration {i}")
                print(f"Max pressure: {max_pressure:.2e} Pa, Min pressure: {min_pressure:.2e} Pa")
                break
        
        if i > 0 and i % 100 == 0:
            elapsed = time_module.time() - start_time
            progress = i / timeIterations * 100
            estimated_total = elapsed / (i / timeIterations)
            remaining = estimated_total - elapsed
            print(f"Progress: {progress:.1f}% - Elapsed: {elapsed:.1f}s - Remaining: {remaining:.1f}s", end ='\r')
            
        # Print pressure distribution every 1000 iterations for debugging
        # if i > 0 and i % 1000 == 0:
        #     print(f"\nIteration {i} pressure distribution:")
        #     for j, comp in enumerate(feed_system.discretisedFeed):
        #         if comp.type != 'g':
        #             print(f"  {comp.type}_{comp.getID()}: {comp.getPressureIn()/1e5:.3f} bar")

    end_time = time_module.time()
    total_time = end_time - start_time
    
    print("Progress: 100.00000%")
    print(f"JIT-optimized simulation completed in {total_time:.2f} seconds")
    print(f"Performance: {timeIterations/total_time:.0f} iterations/second")

    output_folder = f"simulation_results_jit_{dt:.2e}_{simTime}"
    
    # Write CSV files to the new folder
    feed_system.write_to_csv(
        pressure_filename="pressure_results.csv", 
        massflow_filename="massflow_results.csv",
        temperature_filename="temperature_results.csv",
        output_dir=output_folder        
    )        

    print(f"\nJIT-optimized simulation completed. Data written to {output_folder}")
