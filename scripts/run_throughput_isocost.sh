TRACE=rr_conversation_180
SEED=0
echo "Running throughput experiments for trace $TRACE"

# Baseline-A100
python run.py \
    applications.0.scheduler=token_jsq \
    cluster=half_half \
    cluster.servers.0.count=86 \
    cluster.servers.1.count=0 \
    start_state=baseline \
    performance_model=db \
    trace.filename=$TRACE \
    seed=$SEED


## Baseline-H100
python run.py \
    applications.0.scheduler=token_jsq \
    cluster=half_half \
    cluster.servers.0.count=0 \
    cluster.servers.1.count=40 \
    start_state=baseline \
    performance_model=db \
    trace.filename=$TRACE \
    seed=$SEED


# Splitwise-AA
python run.py \
    applications.0.scheduler=mixed_pool \
    cluster=half_half \
    cluster.servers.0.count=86 \
    cluster.servers.1.count=0 \
    start_state=splitwise \
    start_state.prompt.num_instances=51 \
    start_state.token.num_instances=35 \
    performance_model=db \
    trace.filename=$TRACE \
    seed=$SEED


# Splitwise-HH
python run.py \
    applications.0.scheduler=mixed_pool \
    cluster=half_half \
    cluster.servers.0.count=0 \
    cluster.servers.1.count=40 \
    start_state=splitwise \
    start_state.prompt.num_instances=25 \
    start_state.token.num_instances=15 \
    performance_model=db \
    trace.filename=$TRACE \
    seed=$SEED


# Splitwise-HA
python run.py \
    applications.0.scheduler=mixed_pool \
    cluster=half_half \
    cluster.servers.0.count=21 \
    cluster.servers.1.count=30 \
    start_state=splitwise \
    start_state.split_type=heterogeneous \
    performance_model=db \
    trace.filename=$TRACE \
    seed=$SEED


# Splitwise-HHcap
python run.py \
    applications.0.scheduler=mixed_pool \
    cluster=hhcap_half_half \
    cluster.servers.0.count=10 \
    cluster.servers.1.count=30 \
    start_state=splitwise_hhcap \
    start_state.split_type=heterogeneous \
    performance_model=db \
    trace.filename=$TRACE \
    seed=$SEED
