# SpatialSeg
A project skeleton for spatial semantic segmentation.
## Todo List
+ The optimizer/warmup does not work properly after loading the checkpoint file (it has no effect on the training without interruption from beginning to end).
+ The hyperparameters of the warmup section can be considered to be added to the config file.
+ At present, the project train is based on iters. If possible, it will be based on iters or epoch through parameter control in the future, but it feels a bit difficult for the project ?
+ Lr_scheduler, warmup, etc. can expand different types in the future, and select the type through config file.
