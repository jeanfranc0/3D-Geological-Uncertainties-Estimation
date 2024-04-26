##################
#python libraries#
##################
import os, sys
import gpustat
# GPU picking
# http://stackoverflow.com/a/41638727/419116
# https://github.com/bamos/setGPU/blob/master/setGPU.py
# Status of GPU

def gpu_memory_map():
    """Returns map of GPU id to memory allocated on that GPU."""
    
    stats = gpustat.GPUStatCollection.new_query()
    ids = map(lambda gpu: int(gpu.entry['index']), stats)
    ratios = map(lambda gpu: float(gpu.entry['memory.used'])/float(gpu.entry['memory.total']), stats)
    pairs = list(zip(ratios, ids))
    print(pairs)
    
    return pairs

def pick_gpu_lowest_memory(n=1):
    """Returns GPU with the least allocated memory"""

    memory_gpu_map = gpu_memory_map() #[(memory, gpu_id) for (gpu_id, memory) in gpu_memory_map().items()]
    best_gpu = ",".join(str(y) for x, y in sorted(memory_gpu_map)[:n])
    #best_memory, best_gpu = sorted(memory_gpu_map)[0]
    return best_gpu

def setup_one_gpu():
    assert not 'tensorflow' in sys.modules, "GPU setup must happen before importing TensorFlow"
    gpu_id = pick_gpu_lowest_memory()
    print("Picking GPU "+str(gpu_id), flush=True)
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

def setup_no_gpu():
    if 'tensorflow' in sys.modules:
        print("Warning, GPU setup must happen before importing TensorFlow")
    os.environ["CUDA_VISIBLE_DEVICES"] = ''

def setup_more_gpu(n):
    """
    Setting up more GPUs to use
    """
    assert not 'tensorflow' in sys.modules, "GPU setup must happen before importing TensorFlow"
    gpu_id = pick_gpu_lowest_memory(n)
    print("Picking GPU "+str(gpu_id))
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
def setup_gpu(gpu_id="0"):
    """
    Setup specific GPU
    """
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
