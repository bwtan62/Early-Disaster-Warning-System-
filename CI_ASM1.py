import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# Input variables
rainfall_intensity = ctrl.Antecedent(np.arange(0, 60, 0.1), 'rainfall_intensity')
river_water_level = ctrl.Antecedent(np.arange(0, 27, 0.1), 'river_water_level')
no_of_trees_planted = ctrl.Antecedent(np.arange(0, 500, 1), 'no_of_trees_planted')

# Output variable
flood_warning_level = ctrl.Consequent(np.arange(0, 100, 1), 'flood_warning_level')

# Membership function
rainfall_intensity['light'] = fuzz.trimf(rainfall_intensity.universe, [0, 0, 15])
rainfall_intensity['moderate'] = fuzz.trimf(rainfall_intensity.universe, [10, 20, 35])
rainfall_intensity['heavy'] = fuzz.trimf(rainfall_intensity.universe,[25, 45, 60])
rainfall_intensity['very heavy'] = fuzz.trapmf(rainfall_intensity.universe, [40, 55, 60, 60])

river_water_level['normal'] = fuzz.trimf(river_water_level.universe, [0, 0, 24])
river_water_level['alert'] = fuzz.trimf(river_water_level.universe, [23, 24, 25])
river_water_level['warning'] = fuzz.trimf(river_water_level.universe, [24, 25, 26])
river_water_level['danger'] = fuzz.trapmf(river_water_level.universe, [25, 26, 27, 27])

no_of_trees_planted['few'] = fuzz.trimf(no_of_trees_planted.universe, [0, 1, 50])
no_of_trees_planted['moderate'] = fuzz.trimf(no_of_trees_planted.universe, [51, 125, 250])
no_of_trees_planted['many'] = fuzz.trapmf(no_of_trees_planted.universe, [251, 375, 500, 500])

flood_warning_level['low'] = fuzz.trimf(flood_warning_level.universe, [0, 25, 50])
flood_warning_level['moderate'] = fuzz.trimf(flood_warning_level.universe, [20, 50, 80])
flood_warning_level['high'] = fuzz.trimf(flood_warning_level.universe, [50, 100, 100])

# Graph output
rainfall_intensity.view()
river_water_level.view()
no_of_trees_planted.view()
flood_warning_level.view()

# Rain Intensity & River Water Level & Number of Trees Planted (Few)
rule1 = ctrl.Rule(rainfall_intensity['light'] & river_water_level['normal'] & no_of_trees_planted['few'], flood_warning_level['low'])
rule2 = ctrl.Rule(rainfall_intensity['light'] & river_water_level['alert'] & no_of_trees_planted['few'], flood_warning_level['moderate'])
rule3 = ctrl.Rule(rainfall_intensity['moderate'] & river_water_level['normal'] & no_of_trees_planted['few'], flood_warning_level['moderate'])
rule4 = ctrl.Rule(rainfall_intensity['moderate'] & river_water_level['alert'] & no_of_trees_planted['few'], flood_warning_level['moderate'])
rule5 = ctrl.Rule(rainfall_intensity['heavy'] & river_water_level['alert'] & no_of_trees_planted['few'], flood_warning_level['moderate'])
rule6 = ctrl.Rule(rainfall_intensity['heavy'] & river_water_level['warning'] & no_of_trees_planted['few'], flood_warning_level['high'])
rule7 = ctrl.Rule(rainfall_intensity['heavy'] & river_water_level['danger'] & no_of_trees_planted['few'], flood_warning_level['high'])
rule8 = ctrl.Rule(rainfall_intensity['very heavy'] & river_water_level['alert'] & no_of_trees_planted['few'], flood_warning_level['high'])
rule9 = ctrl.Rule(rainfall_intensity['very heavy'] & river_water_level['warning'] & no_of_trees_planted['few'], flood_warning_level['high'])
rule10 = ctrl.Rule(rainfall_intensity['very heavy'] & river_water_level['danger'] & no_of_trees_planted['few'], flood_warning_level['high'])

# Rain Intensity & River Water Level & Number of Trees Planted (Moderate)
rule11 = ctrl.Rule(rainfall_intensity['light'] & river_water_level['normal'] & no_of_trees_planted['moderate'], flood_warning_level['low'])
rule12 = ctrl.Rule(rainfall_intensity['light'] & river_water_level['alert'] & no_of_trees_planted['moderate'], flood_warning_level['low'])
rule13 = ctrl.Rule(rainfall_intensity['moderate'] & river_water_level['normal'] & no_of_trees_planted['moderate'], flood_warning_level['low'])
rule14 = ctrl.Rule(rainfall_intensity['moderate'] & river_water_level['alert'] & no_of_trees_planted['moderate'], flood_warning_level['low'])
rule15 = ctrl.Rule(rainfall_intensity['heavy'] & river_water_level['alert'] & no_of_trees_planted['moderate'], flood_warning_level['moderate'])
rule16 = ctrl.Rule(rainfall_intensity['heavy'] & river_water_level['warning'] & no_of_trees_planted['moderate'], flood_warning_level['moderate'])
rule17 = ctrl.Rule(rainfall_intensity['heavy'] & river_water_level['danger'] & no_of_trees_planted['moderate'], flood_warning_level['high'])
rule18 = ctrl.Rule(rainfall_intensity['very heavy'] & river_water_level['alert'] & no_of_trees_planted['moderate'], flood_warning_level['moderate'])
rule19 = ctrl.Rule(rainfall_intensity['very heavy'] & river_water_level['warning'] & no_of_trees_planted['moderate'], flood_warning_level['high'])
rule20 = ctrl.Rule(rainfall_intensity['very heavy'] & river_water_level['danger'] & no_of_trees_planted['moderate'], flood_warning_level['high'])

# Rain Intensity & River Water Level & Number of Trees Planted (Many)
rule21 = ctrl.Rule(rainfall_intensity['light'] & river_water_level['normal'] & no_of_trees_planted['many'], flood_warning_level['low'])
rule22 = ctrl.Rule(rainfall_intensity['light'] & river_water_level['alert'] & no_of_trees_planted['many'], flood_warning_level['low'])
rule23 = ctrl.Rule(rainfall_intensity['moderate'] & river_water_level['normal'] & no_of_trees_planted['many'], flood_warning_level['low'])
rule24 = ctrl.Rule(rainfall_intensity['moderate'] & river_water_level['alert'] & no_of_trees_planted['many'], flood_warning_level['low'])
rule25 = ctrl.Rule(rainfall_intensity['heavy'] & river_water_level['alert'] & no_of_trees_planted['many'], flood_warning_level['low'])
rule26 = ctrl.Rule(rainfall_intensity['heavy'] & river_water_level['warning'] & no_of_trees_planted['many'], flood_warning_level['low'])
rule27 = ctrl.Rule(rainfall_intensity['heavy'] & river_water_level['danger'] & no_of_trees_planted['many'], flood_warning_level['moderate'])
rule28 = ctrl.Rule(rainfall_intensity['very heavy'] & river_water_level['alert'] & no_of_trees_planted['many'], flood_warning_level['low'])
rule29 = ctrl.Rule(rainfall_intensity['very heavy'] & river_water_level['warning'] & no_of_trees_planted['many'], flood_warning_level['moderate'])
rule30 = ctrl.Rule(rainfall_intensity['very heavy'] & river_water_level['danger'] & no_of_trees_planted['many'], flood_warning_level['high'])

rules = [rule1, rule2, rule3, rule4, rule5,rule6, rule7, rule8, rule9, rule10, rule11, rule12, rule13, rule14, rule15, rule16, rule17, rule18, rule19, rule20, rule21, rule22, rule23, rule24, rule25, rule26, rule27, rule28, rule29, rule30]

# Construct the fuzzy control system
flood_ctrl = ctrl.ControlSystem(rules=rules)
flood = ctrl.ControlSystemSimulation(control_system=flood_ctrl)

# Calculate output
def flood_out(rainfall_intensity, river_water_level, no_of_trees_planted):
    # define the values for the inputs
    flood.input['rainfall_intensity'] = rainfall_intensity
    flood.input['river_water_level'] = river_water_level
    flood.input['no_of_trees_planted'] = no_of_trees_planted
    
    # compute the outputs
    flood.compute()

    # print the output values
    print(flood.output)

    # to extract one of the outputs
    print("Flood Warning Level: ", flood.output['flood_warning_level'])
    print()
    
    flood_warning_level.view(sim=flood)

# 3D visualisation
def generate3D(input1, input2, flood, title):
    x, y = np.meshgrid(
        np.linspace(input1.universe.min(), input1.universe.max(), 100),
        np.linspace(input2.universe.min(), input2.universe.max(), 100)
    )
    z = np.zeros_like(x, dtype=float)
    
    for i,r in enumerate(x):
        for j,c in enumerate(r):
          flood.input[input1.label] = x[i,j]
          flood.input[input2.label] = y[i,j]
          try:
            flood.compute()
          except:
            z[i,j] = float('inf')
          z[i,j] = flood.output['flood_warning_level']
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='viridis', linewidth=0.4, antialiased=True)
    ax.contourf(x, y, z, zdir='z', offset=-2.5, cmap='viridis', alpha=0.5)
    ax.contourf(x, y, z, zdir='x', offset=x.max()*1.5, cmap='viridis', alpha=0.5)
    ax.contourf(x, y, z, zdir='y', offset=y.max()*1.5, cmap='viridis', alpha=0.5)
    
    ax.set_xlabel(input1.label)
    ax.set_ylabel(input2.label)
    ax.set_zlabel('flood_warning_level')
  
    ax.view_init(30, 200)
    ax.set_title(title)
    plt.show()
    
# Scenario 1
print("Scenario 1:\nrain intensity = 6mm/h, river water level = 23 meter, number of tree planted = 10k")
flood_out(6, 23, 10)

# Scenario 2
print("Scenario 2:\nrain intensity = 21mm/h, river water level = 24.6 meter, number of tree planted = 10k")
flood_out(21, 24.6, 10)

# Scenario 3
print("Scenario 3:\nrain intensity = 45mm/h, river water level = 26 meter, number of tree planted = 260k")
flood_out(45, 26, 260)

# Scenario 4
print("Scenario 4:\nrain intensity = 65mm/h, river water level = 28 meter, number of tree planted = 510k")
flood_out(65, 28, 510)

# User input
# ri = int(input("Please enter rain intensity from 0 to 60mm (0 >= 60): "))
# rwl = int(input("Please enter river water level from 0 to 27m (0 >= 27): "))
# num_tp = int(input("Please enter number of tree planted from 0 to 500k (0 >= 500): "))
# flood_out(ri, rwl, num_tp)

# Generate and display the 3D plots
generate3D(river_water_level, rainfall_intensity, flood, "Rain Intensity & River Water Level")
generate3D(no_of_trees_planted, rainfall_intensity, flood, "Rain Intensity & Number of Trees Planted")
generate3D(no_of_trees_planted, river_water_level, flood, "River Water Level & Number of Trees Planted")









