# run profiling on nbody
nvprof -f -o nbody-100k-timeline.nvvp ./nbody 100000
# obtain metrics for nvvp
nvprof --analysis-metrics -f -o nbody-100k.nvvp ./nbody 100000

# same for solution binary
nvprof -f -o nbody-solution-100k-timeline.nvvp ./nbody-solution 100000
nvprof --analysis-metrics -f -o nbody-solution-100k.nvvp ./nbody-solution 100000
