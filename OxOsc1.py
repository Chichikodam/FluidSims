from pyfluids import FluidsList, Fluid,Input
import numpy as np
import matplotlib.pyplot as plt
from numpy import linspace
import numpy as np
ox = Fluid(FluidsList.Oxygen)
# Define the input parameters
tempA = linspace(-200,40, 200)   # Convert to Celsius
press = linspace(100000, 200000, 10)  # Pressure in Pascals
feedPressure = 50 # Feed pressure in bar
feedPressure = feedPressure * 1e5  # Convert to Pascals
K = 100.0 # bu
E = 200.0 # Young's modulus GPa
E = E * 1e9  # Convert to Pascals
# Calculate the properties for a range of temperatures and pressures
D = 1.25  # inner Diameter of the pipe in inches
D = D * 0.0254  # Convert to meters
v = 1.0 # Poisson's ratio
e = 0.1  # Thickness of the pipe in m
L = []
K = []
Lcombined = []
total_sound_speed  = []
fluids_sound_speed = []
drho = 1.0
rhoA = []  # Density array
KC = []
 # Length of the pipe in meters
pipeLength = linspace(0.1, 30, 10)  # Length of the pipe in meters
freq=  12.5 # Frequency in Hz

for i, temp in enumerate(tempA):
    ox.update(Input.temperature(temp),Input.pressure(feedPressure))
    rho = ox.density  # Density of the fluid
    rhoA.append(rho)  # Append density to the density array
    fluids_sound_speed.append(ox.sound_speed)  # Speed of sound in the fluid
    L.append(ox.sound_speed/(4*freq))  # Length of the pipe based on sound speed and frequency

    ox.update(Input.entropy(ox.entropy),Input.density(rho))
    Ki = ox.sound_speed**2 * (rho)  # Bulk modulus calculation
    K.append(Ki/1e8)
    
    c1 = (2*e/D)*(1+v)+ D*(1-v**2)/(D+e)
    total_sound_speedI = np.sqrt((Ki/ rho)/(1+((Ki/E)*(D/e))*c1))
    total_sound_speed.append(total_sound_speedI)  # Speed of sound
    KiC = total_sound_speedI**2 * (rho)  # Combined bulk modulus
    KC.append(KiC/1e8)  # Append combined bulk modulus to K array
    Lcombined.append(total_sound_speedI/(4*freq))


# Print th÷÷

# Analyze K array for constant region
K_arr = np.array(K)
grad = np.diff(K_arr)
sign_grad = np.sign(grad)
flip_idxs = np.where((sign_grad[:-1] < 0) & (sign_grad[1:] > 0))[0]

if flip_idxs.size > 0:
    flip_idx = flip_idxs[0] + 1
    print(f"Fluid boils at: {flip_idx}, \n"
          f"Temperature = {tempA[flip_idx]} °C,\n "
          f"K = {K_arr[flip_idx]:.3e} Pa")
else:
    print("No negative-to-positive gradient flip found.")

# plot results in subplots in one figure
fig, axes = plt.subplots(4, 1, figsize=(8, 8))

# Bulk modulus
axes[0].plot(tempA, K, label="Bulk Modulus")
axes[0].plot(tempA, KC, label="Combined Bulk Modulus", linestyle='--')
axes[0].set_xlabel("Temperature (°C)")
axes[0].set_ylabel("Bulk Modulus (GPa)")
axes[0].set_title("Bulk Modulus vs Temperature")
axes[0].legend(loc="best")

# Sound speeds
axes[1].plot(tempA, fluids_sound_speed, label="Fluid Sound Speed")
axes[1].plot(tempA, total_sound_speed, label="Total Sound Speed")
axes[1].set_xlabel("Temperature (°C)")
axes[1].set_ylabel("Sound Speed (m/s)")
axes[1].set_title("Sound Speeds vs Temperature")
axes[1].legend(loc="best")

# Pipe lengths
axes[2].plot(tempA, L, label="Pipe Length L")
axes[2].plot(tempA, Lcombined, label="Combined Pipe Length")
axes[2].set_xlabel("Temperature (°C)")
axes[2].set_ylabel("L (m)")
axes[2].set_title("Pipe Length vs Temperature")
axes[2].legend(loc="best")

# Density vs temperature
axes[3].plot(tempA, rhoA, label="Density", color="tab:green")
axes[3].set_xlabel("Temperature (°C)")
axes[3].set_ylabel("Density (kg/m³)")
axes[3].set_title("Density vs Temperature")
axes[3].legend(loc="best")

# Add a dotted vertical line at the boil-off temperature on all subplots
if flip_idxs.size > 0:
    boil_temp = tempA[flip_idx]
    for ax in axes:
        ax.axvline(x=boil_temp, color='gray', linestyle='--')
    # Display the boil-off temperature as text in the figure
    fig.text(0.5, 0.94, f"At {feedPressure/1e5} Bar:\nFluid boils off at {boil_temp:.2f} °C", 
                ha='center', va='center', fontsize=12)

plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.show()
