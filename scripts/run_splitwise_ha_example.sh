SCHEDULER=mixed_pool
NUM_A100=1
NUM_H100=1
START_STATE=splitwise
TRACE=test_trace

python run.py \
    applications.0.scheduler=$SCHEDULER \
    cluster=half_half \
    cluster.servers.0.count=$NUM_A100 \
    cluster.servers.1.count=$NUM_H100 \
    start_state=$START_STATE \
    start_state.split_type=heterogeneous \
    performance_model=db \
    trace.filename=$TRACE \
    debug=True \
    seed=0
