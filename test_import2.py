import sys
import importlib.util
sys.path.insert(0, '/home/stijn/lilly/elo-system')
sys.path.insert(0, '/home/stijn/lilly/ML')
print(sys.path[:3])
print('find_spec ML:', importlib.util.find_spec('ML'))
spec = importlib.util.spec_from_file_location('ML', '/home/stijn/lilly/ML/__init__.py', submodule_search_locations=['/home/stijn/lilly/ML'])
print('spec:', spec)
ml = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ml)
print('ML module loaded')
# now load data_prep submodule
dp_spec = importlib.util.spec_from_file_location('ML.data_prep', '/home/stijn/lilly/ML/data_prep.py')
dp = importlib.util.module_from_spec(dp_spec)
dp_spec.loader.exec_module(dp)
print('data_prep loaded')
