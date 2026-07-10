# Changelog

## crace 1.0



### crace 1.1.0

> Based on tag: v1.0.1
> 
> - src:
>   - options: add integer option testParallel
>     - value 0: disable parallelization in testing phase
>     - value > 1: enable parallelization in testing phase (must be <= parallel)
> 
>   - main.py
>     - add `print_affinity_info()` when parallel > 1 (mpi disabled)
> 
> - pkg:
>   - add changelog.md
>
> (Yunshuang Xiao)



### crace 1.0.1

> Based on tag: v1.0.0.post1
> 
> - src:
>   - main.py, mpi.py, utils.py:
>     - improve error handling in `start_cmdline()`, `start_mpi()` and `_enforce_single_thread_binding()`
>     - support single cpu binding in `_enforce_single_thread_binding()`
>
> (Yunshuang Xiao)



### crace 1.0.0.post1

> Based on commit 0686d5
> 
> - src:
>   - Experiments, ProbabilisticModel:
>     - add 'debug_level' as argument to initialize mentioned class
> 
>   - scripts.mpi:
>     - add try-except for mpi execution to ensure exiting
>     - add `_check_mpi_env()` to check mpi environment
>       - installed mpi4py or not
>       - installed MPI implementation or not
>       - mpi.py called by MPI launcher or not
>     - improve the output of cpu affinity for mpi executions
>
> (Yunshuang Xiao)




### No tag (**commit**: 0686d51)

> modified gh-pages.yml
>
> (Yunshuang Xiao)




### crace1.0.0.post0 

> Based on v1.0.0
> 
> - src:
>   - execution: improve pool management and add support for different python versions
> 
>   - race:
>       - bug fix for log_state: support for 'n_instances'
>       - bug fix for slice budget: min -> max
>
> 
> - package:
>   - makefile: support pyproject, build, tag
>       - pyptoject: generate pyproject.toml from generate_pyproject.py
>       - build: build package
>       - tag: add tag for the current version based on (provided) base tag
> 
>   - improve generate_pyproject.py
>       - add urls
>       - improve classfiers
> 
>   - delete setup.py
> 
>   - base information:
>     - modify package description
>
> (Yunshuang Xiao)




### crace 1.0.0

> The first public version: v1.0.0
> 
> (Yunshuang Xiao)




