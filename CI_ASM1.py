import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


rainfall_intensity = ctrl.Antecedent(np.arange(0, 60, 0.1), 'rainfall_intensity')
river_water_level = ctrl.Antecedent(np.arange(22, 27, 0.1), 'river_water_level')
no_of_trees_planted = ctrl.Antecedent(np.arange(0, 500, 1), 'no_of_trees_planted')

flood_warning_level = ctrl.Consequent(np.arange(0, 100, 1), 'flood_warning_level')

rainfall_intensity['light'] = fuzz.trimf(rainfall_intensity.universe, [0, 0, 15])
rainfall_intensity['moderate'] = fuzz.trimf(rainfall_intensity.universe, [10, 20, 35])
rainfall_intensity['heavy'] = fuzz.trimf(rainfall_intensity.universe,[25, 45, 60])
rainfall_intensity['very heavy'] = fuzz.trapmf(rainfall_intensity.universe, [40, 55, 60, 60])

river_water_level['normal'] = fuzz.trimf(river_water_level.universe, [0, 22, 24])
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

# rule1 = ctrl.Rule(rainfall_intensity['low'] & river_water_level['low'] & no_of_trees['dry'], flood_warning_level['low'])
# rule2 = ctrl.Rule(rainfall_intensity['moderate'] & river_water_level['moderate'] & no_of_trees['moist'], flood_warning_level['moderate'])
# rule3 = ctrl.Rule(rainfall_intensity['high'] | river_water_level['high'] | no_of_trees['wet'], flood_warning_level['high'])

# rules = [rule1, rule2, rule3]
# print (rules)

# train_ctrl = ctrl.ControlSystem(rules = rules)

# train = ctrl.ControlSystemSimulation(control_system = train_ctrl)

# train.input['rainfall_intensity'] = 30
# train.input['river_water_level'] = 50
# train.input['soil_moisture'] = 70

# train.compute()

# print(train.output)

# print(train.output['flood_warning_level'])

# flood_warning_level.view(sim=train)
    