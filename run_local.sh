#!/bin/bash
cd ~/pipecat
source venv312/bin/activate
SP=$(python -c 'import site; print(site.getsitepackages()[0])')
export LD_LIBRARY_PATH=$SP/nvidia/cublas/lib:$SP/nvidia/cudnn/lib:$LD_LIBRARY_PATH
exec python examples/getting-started/${AGENT:-07-local-agent.py}
