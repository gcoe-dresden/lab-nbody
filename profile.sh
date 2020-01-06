NR_BODIES=16000

# run profiling on nbody
nvprof -f -o nbody-100k-timeline.nvvp ./nbody $NR_BODIES
# obtain metrics for nvvp
nvprof --analysis-metrics -f -o nbody-100k.nvvp ./nbody $NR_BODIES

# same for solution binary
nvprof -f -o nbody-solution-100k-timeline.nvvp ./nbody-solution $NR_BODIES
nvprof --analysis-metrics -f -o nbody-solution-100k.nvvp ./nbody-solution $NR_BODIES
