import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt


rainfall_intensity = ctrl.Antecedent(np.arange(0, 60, 0.1), 'rainfall_intensity')
river_water_level = ctrl.Antecedent(np.arange(0, 27, 0.1), 'river_water_level')
no_of_trees_planted = ctrl.Antecedent(np.arange(0, 500, 1), 'no_of_trees_planted')

flood_warning_level = ctrl.Consequent(np.arange(0, 100, 1), 'flood_warning_level')

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

flood_warning_level['low'] = fuzz.trimf(flood_warning_level.universe, [1, 25, 50])
flood_warning_level['moderate'] = fuzz.trimf(flood_warning_level.universe, [20, 50, 80])
flood_warning_level['high'] = fuzz.trimf(flood_warning_level.universe, [50, 100, 100])

rainfall_intensity.view()
river_water_level.view()
no_of_trees_planted.view()
flood_warning_level.view()

# Rain Intensity & River Water Level
rule1 = ctrl.Rule(rainfall_intensity['light'] & river_water_level['normal'], flood_warning_level['low'])
rule2 = ctrl.Rule(rainfall_intensity['light'] & river_water_level['alert'], flood_warning_level['moderate'])
rule3 = ctrl.Rule(rainfall_intensity['moderate'] & river_water_level['normal'], flood_warning_level['low'])
rule4 = ctrl.Rule(rainfall_intensity['moderate'] & river_water_level['alert'], flood_warning_level['moderate'])
rule5 = ctrl.Rule(rainfall_intensity['heavy'] & river_water_level['warning'], flood_warning_level['moderate'])
rule6 = ctrl.Rule(rainfall_intensity['heavy'] & river_water_level['danger'], flood_warning_level['high'])
rule7 = ctrl.Rule(rainfall_intensity['very heavy'] & river_water_level['warning'], flood_warning_level['moderate'])
rule8 = ctrl.Rule(rainfall_intensity['very heavy'] & river_water_level['danger'], flood_warning_level['high'])

# Rain Intensity & Number of Trees Planted
rule9 = ctrl.Rule(rainfall_intensity['light'] & no_of_trees_planted['few'], flood_warning_level['moderate'])
rule10 = ctrl.Rule(rainfall_intensity['light'] & no_of_trees_planted['moderate'], flood_warning_level['low'])
rule11 = ctrl.Rule(rainfall_intensity['light'] & no_of_trees_planted['many'], flood_warning_level['low'])
rule12 = ctrl.Rule(rainfall_intensity['moderate'] & no_of_trees_planted['few'], flood_warning_level['moderate'])
rule13 = ctrl.Rule(rainfall_intensity['moderate'] & no_of_trees_planted['moderate'], flood_warning_level['low'])
rule14 = ctrl.Rule(rainfall_intensity['moderate'] & no_of_trees_planted['many'], flood_warning_level['low'])
rule15 = ctrl.Rule(rainfall_intensity['heavy'] & no_of_trees_planted['few'], flood_warning_level['high'])
rule16 = ctrl.Rule(rainfall_intensity['heavy'] & no_of_trees_planted['moderate'], flood_warning_level['moderate'])
rule17 = ctrl.Rule(rainfall_intensity['heavy'] & no_of_trees_planted['many'], flood_warning_level['low'])
rule18 = ctrl.Rule(rainfall_intensity['very heavy'] & no_of_trees_planted['few'], flood_warning_level['high'])
rule19 = ctrl.Rule(rainfall_intensity['very heavy'] & no_of_trees_planted['moderate'], flood_warning_level['moderate'])
rule20 = ctrl.Rule(rainfall_intensity['very heavy'] & no_of_trees_planted['many'], flood_warning_level['moderate'])

# River Water Level & Number of Trees Planted
rule21 = ctrl.Rule(river_water_level['normal'] & no_of_trees_planted['few'], flood_warning_level['low'])
rule22 = ctrl.Rule(river_water_level['normal'] & no_of_trees_planted['moderate'], flood_warning_level['low'])
rule23 = ctrl.Rule(river_water_level['normal'] & no_of_trees_planted['many'], flood_warning_level['low'])
rule24 = ctrl.Rule(river_water_level['alert'] & no_of_trees_planted['few'], flood_warning_level['moderate'])
rule25 = ctrl.Rule(river_water_level['alert'] & no_of_trees_planted['moderate'], flood_warning_level['moderate'])
rule26 = ctrl.Rule(river_water_level['alert'] & no_of_trees_planted['many'], flood_warning_level['moderate'])
rule27 = ctrl.Rule(river_water_level['warning'] & no_of_trees_planted['few'], flood_warning_level['high'])
rule28 = ctrl.Rule(river_water_level['warning'] & no_of_trees_planted['moderate'], flood_warning_level['high'])
rule29 = ctrl.Rule(river_water_level['warning'] & no_of_trees_planted['many'], flood_warning_level['high'])
rule30 = ctrl.Rule(river_water_level['danger'] & no_of_trees_planted['few'], flood_warning_level['high'])
rule31 = ctrl.Rule(river_water_level['danger'] & no_of_trees_planted['moderate'], flood_warning_level['high'])
rule32 = ctrl.Rule(river_water_level['danger'] & no_of_trees_planted['many'], flood_warning_level['high'])

rules = [rule1, rule2, rule3, rule4, rule5,rule6, rule7, rule8, rule9, rule10, rule11, rule12, rule13, rule14, rule15, rule16, rule17, rule18, rule19, rule20,
         rule21, rule22, rule23, rule24, rule25, rule26, rule27, rule28, rule29, rule30, rule31, rule32]

# Construct the fuzzy control system
train_ctrl = ctrl.ControlSystem(rules=rules)

train = ctrl.ControlSystemSimulation(control_system=train_ctrl)

# define the values for the inputs
train.input['rainfall_intensity'] = 60
train.input['river_water_level'] = 27
train.input['no_of_trees_planted'] = 500

# compute the outputs
train.compute()

# print the output values
print(train.output)

# to extract one of the outputs
print("Flood Warning Level: ", train.output['flood_warning_level'])

flood_warning_level.view(sim=train)

# Rain Intensity & River Water Level
# ==========================================================================
# x, y = np.meshgrid(np.linspace(river_water_level.universe.min(), river_water_level.universe.max(), 100),
#                     np.linspace(rainfall_intensity.universe.min(), rainfall_intensity.universe.max(), 100))
# z = np.zeros_like(x, dtype=float)


# for i,r in enumerate(x):
#   for j,c in enumerate(r):
#     train.input['river_water_level'] = x[i,j]
#     train.input['rainfall_intensity'] = y[i,j]
#     try:
#       train.compute()
#     except:
#       z[i,j] = float('inf')
#     z[i,j] = train.output['flood_warning_level']

# Rain Intensity & Number of Trees Planted
# ==========================================================================
# x, y = np.meshgrid(np.linspace(no_of_trees_planted.universe.min(), no_of_trees_planted.universe.max(), 100),
#                     np.linspace(rainfall_intensity.universe.min(), rainfall_intensity.universe.max(), 100))
# z = np.zeros_like(x, dtype=float)


# for i,r in enumerate(x):
#   for j,c in enumerate(r):
#     train.input['no_of_trees_planted'] = x[i,j]
#     train.input['rainfall_intensity'] = y[i,j]
#     try:
#       train.compute()
#     except:
#       z[i,j] = float('inf')
#     z[i,j] = train.output['flood_warning_level']

# River Water Level & Number of Trees Planted
# ==========================================================================
# x, y = np.meshgrid(np.linspace(no_of_trees_planted.universe.min(), no_of_trees_planted.universe.max(), 100),
#                     np.linspace(river_water_level.universe.min(), river_water_level.universe.max(), 100))
# z = np.zeros_like(x, dtype=float)


# for i,r in enumerate(x):
#   for j,c in enumerate(r):
#     train.input['no_of_trees_planted'] = x[i,j]
#     train.input['river_water_level'] = y[i,j]
#     try:
#       train.compute()
#     except:
#       z[i,j] = float('inf')
#     z[i,j] = train.output['flood_warning_level']
   

# def plot3d(x,y,z):
#   fig = plt.figure()
#   ax = fig.add_subplot(111, projection='3d')

#   ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='viridis', linewidth=0.4, antialiased=True)

#   ax.contourf(x, y, z, zdir='z', offset=-2.5, cmap='viridis', alpha=0.5)
#   ax.contourf(x, y, z, zdir='x', offset=x.max()*1.5, cmap='viridis', alpha=0.5)
#   ax.contourf(x, y, z, zdir='y', offset=y.max()*1.5, cmap='viridis', alpha=0.5)

#   ax.view_init(30, 200)

# plot3d(x, y, z)


# def generate3D(input1, input2, train):
#     x, y = np.meshgrid(
#         np.linspace(input1.universe.min(), input1.universe.max(), 100),
#         np.linspace(input2.universe.min(), input2.universe.max(), 100)
#     )
#     z = np.zeros_like(x, dtype=float)
    
#     for i,r in enumerate(x):
#       for j,c in enumerate(r):
#         train.input[input1.label] = x[i,j]
#         train.input[input2.label] = y[i,j]
#         try:
#           train.compute()
#         except:
#           z[i,j] = float('inf')
#         z[i,j] = train.output['flood_warning_level']
    
#     return x, y, z

# x1, y1, z1 = generate3D(river_water_level, rainfall_intensity, train)
# x2, y2, z2 = generate3D(no_of_trees_planted, rainfall_intensity, train)
# x3, y3, z3 = generate3D(no_of_trees_planted, river_water_level, train)

# def plot3d(x,y,z, title):
#   fig = plt.figure()
#   ax = fig.add_subplot(111, projection='3d')

#   ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='viridis', linewidth=0.4, antialiased=True)

#   ax.contourf(x, y, z, zdir='z', offset=-2.5, cmap='viridis', alpha=0.5)
#   ax.contourf(x, y, z, zdir='x', offset=x.max()*1.5, cmap='viridis', alpha=0.5)
#   ax.contourf(x, y, z, zdir='y', offset=y.max()*1.5, cmap='viridis', alpha=0.5)

<<<<<<< HEAD
plot3d(x, y, z)
=======
#   ax.view_init(30, 200)
#   ax.set_title(title)

# plot3d(x1, y1, z1, "Rain Intensity & River Water Level")
# plot3d(x2, y2, z2, "Rain Intensity & Number of Trees Planted")
# plot3d(x3, y3, z3, "River Water Level & Number of Trees Planted")

# plt.show()


def generate3D(input1, input2, train, title):
    x, y = np.meshgrid(
        np.linspace(input1.universe.min(), input1.universe.max(), 100),
        np.linspace(input2.universe.min(), input2.universe.max(), 100)
    )
    z = np.zeros_like(x, dtype=float)
    
    for i,r in enumerate(x):
        for j,c in enumerate(r):
          train.input[input1.label] = x[i,j]
          train.input[input2.label] = y[i,j]
          try:
            train.compute()
          except:
            z[i,j] = float('inf')
          z[i,j] = train.output['flood_warning_level']
    
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

# Generate and display the 3D plots
generate3D(river_water_level, rainfall_intensity, train, "Rain Intensity & River Water Level")
generate3D(no_of_trees_planted, rainfall_intensity, train, "Rain Intensity & Number of Trees Planted")
generate3D(no_of_trees_planted, river_water_level, train, "River Water Level & Number of Trees Planted")

>>>>>>> 16b774120247ac1eb619fd07362ac0d34e195dab
