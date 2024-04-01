NUM_DGX_A100=0
NUM_DGX_H100=1
SCHEDULER=token_jsq
START_STATE=baseline
TRACE=test_trace

python run.py \
    cluster=half_half \
    cluster.servers.0.count=$NUM_DGX_A100 \
    cluster.servers.1.count=$NUM_DGX_H100 \
    applications.0.scheduler=$SCHEDULER \
    start_state=$START_STATE \
    performance_model=db \
    trace.filename=$TRACE \
    debug=True \
    seed=0
