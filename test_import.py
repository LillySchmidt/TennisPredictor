import sys
sys.path.insert(0, '/home/stijn/lilly/elo-system')
sys.path.insert(0, '/home/stijn/lilly/ML')
print(sys.path[:3])
from ML import data_prep as ml_dp
print('ml_dp OK')
from tennis_elo import EloConfig
print('EloConfig OK')
