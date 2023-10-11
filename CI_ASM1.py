import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


rainfall_intensity = ctrl.Antecedent(np.arange(0, 101, 1), 'rainfall_intensity')
river_water_level = ctrl.Antecedent(np.arange(0, 101, 1), 'river_water_level')
soil_moisture = ctrl.Antecedent(np.arange(0, 101, 1), 'soil_moisture')

flood_warning_level = ctrl.Consequent(np.arange(0, 101, 1), 'flood_warning_level')


rainfall_intensity['low'] = fuzz.trimf(rainfall_intensity.universe, [0, 0, 30])
rainfall_intensity['moderate'] = fuzz.trimf(rainfall_intensity.universe, [20, 50, 80])
rainfall_intensity['high'] = fuzz.trimf(rainfall_intensity.universe, [70, 100, 100])

river_water_level['low'] = fuzz.trimf(river_water_level.universe, [0, 0, 30])
river_water_level['moderate'] = fuzz.trimf(river_water_level.universe, [20, 50, 80])
river_water_level['high'] = fuzz.trimf(river_water_level.universe, [70, 100, 100])

soil_moisture['dry'] = fuzz.trimf(soil_moisture.universe, [0, 0, 30])
soil_moisture['moist'] = fuzz.trimf(soil_moisture.universe, [20, 50, 80])
soil_moisture['wet'] = fuzz.trimf(soil_moisture.universe, [70, 100, 100])

flood_warning_level['low'] = fuzz.trimf(flood_warning_level.universe, [0, 0, 50])
flood_warning_level['moderate'] = fuzz.trimf(flood_warning_level.universe, [20, 50, 80])
flood_warning_level['high'] = fuzz.trimf(flood_warning_level.universe, [50, 100, 100])

rainfall_intensity.view()
river_water_level.view()
soil_moisture.view()
flood_warning_level.view()

rule1 = ctrl.Rule(rainfall_intensity['low'] & river_water_level['low'] & soil_moisture['dry'], flood_warning_level['low'])
rule2 = ctrl.Rule(rainfall_intensity['moderate'] & river_water_level['moderate'] & soil_moisture['moist'], flood_warning_level['moderate'])
rule3 = ctrl.Rule(rainfall_intensity['high'] | river_water_level['high'] | soil_moisture['wet'], flood_warning_level['high'])

rules = [rule1, rule2, rule3]
print (rules)

train_ctrl = ctrl.ControlSystem(rules = rules)

train = ctrl.ControlSystemSimulation(control_system = train_ctrl)

train.input['rainfall_intensity'] = 30
train.input['river_water_level'] = 50
train.input['soil_moisture'] = 70

train.compute()

print(train.output)

print(train.output['flood_warning_level'])

flood_warning_level.view(sim=train)
    