import dgl
import psutil
import os
import pickle
import time

path = "/opt/li_dataset/coreutil"

def test_pickle():
    with open(path + "/all_data.pkl", "rb") as f:
        data = pickle.load(f)
        f.close()
        
def test_dgl():
    all_data, _ = dgl.load_graphs(
        os.path.join(path, "dgl_graphs.dgl")
    )


start_time = time.time()
test_pickle()
end_time = time.time()


print(
    "当前进程的内存使用：%.4f GB, 加载时间: %.4f s"
    % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024, end_time - start_time) 
)