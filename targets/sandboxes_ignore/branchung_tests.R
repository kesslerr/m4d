
# emulation of branching

tar_pattern(pattern, 1,2,3, seed = 0L)

# toy

list(
  tar_target(x, seq_len(2)),
  tar_target(y, head(letters, 2)),
  tar_target(dynamic, c(x, y), pattern = map(x, y)) # 2 branches
)

tar_pattern(
  cross(x, map(y, z)),
  x = 2,
  y = 3,
  z = 3
)

# mine
list(
  tar_target(experiment, c("ERN", "LRP", "MMN")), # TODO: so kann ich das eigentlich auch in meinen targets umsetzeb
  tar_target(models, c("model1","model2","model3"))
  #tar_target(dynamic, c(x, y), pattern = map(x, y)) # 2 branches
)

tar_pattern(
  map(experiment, models),
  experiment=3,
  models=3
)

