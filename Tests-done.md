# Tests done until now:

* __5 oct 2018__: basic curriculum script ad models trained uploaded
* __15 oct 2018__: added 10x10, 12x12 and 14x14 environments and tensorboard results
* __25 oct 2018__: stratified-curriculum script added, max frames control, accurancy control over N savelog window
* __26 oct 2018__: gicar (stratified int chi and random) added. Function controlled by delta_strat
* __29 oct 2018__: gigar (stratified int gaussian and random), gidb (stratified int direct big), gid (stratified int direct) added. Function controlled by delta_strat and sigma. Tests with sigmas between 0.0 and 
* __30 oct 2018__: tests with gigar, gicar, gidb, gid added using different sigmas (from 0.1 to 1.0)
* __13 nov 2018__: added MEAN/MIN accurancy control
* __15 nov 2018__: added model saving each x frames and added tests for all
* __16 nov 2018__: added discount control and tests with different discounts (0.90 default, 0.90, 0.95, 0.99)
* __21 nov 2018__: added reward_multiplier control in formula [1-rm*(current_step/max_steps)] and different reward multiplier tested ( 0.9 default, 0.8, 0.7, 0.6, 0.5)
* __06 dic 2018__: added test method choise in train and stratified_curriculum script's menu (also max-steps control added)
* __11 dic 2018__: added trainings with bigger max steps (from 10*size*size as 1560 to 5000 and more)
* __23 dic 2018__: added noise control to insert random walls on right and left side of main wall with door and enjoy2.py uploaded
