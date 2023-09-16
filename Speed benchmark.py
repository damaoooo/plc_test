from model import MyModel
import dgl
import time
import torch
from torch.profiler import profile, record_function, ProfilerActivity

max_node = 10000

dummy_graph = dgl.graph(([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]), num_nodes=max_node)
dummy_graph = dgl.add_self_loop(dummy_graph)
dummy_graph.ndata['feat'] = torch.randn(max_node, 151)
dummy_graph = dummy_graph.to('cuda')

print("CUDA FP32 testing... ")
model = MyModel(151, 128, 64, 6, 0.3, 0.2, max_node)
model = model.cuda()
model.eval()


with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True, record_shapes=True) as prof:
    with torch.no_grad():
        start_time = time.time()
        for i in range(100):
            s = model(dummy_graph)
        end_time = time.time()
        print("CUDA FP32 time: ", (end_time - start_time) / 10)

print(prof.key_averages(group_by_stack_n=5).table(sort_by="cpu_time_total",row_limit=10))
print(prof.key_averages( group_by_stack_n=5).table(sort_by="cuda_time_total", row_limit=10))
prof.export_stacks("./profiler_stacks.txt", "self_cuda_time_total")


# print("CUDA FP16 testing... ")
# model = MyModel(151, 128, 64, 6, 0.3, 0.2, 500)
# model = model.cuda()
# model.eval()

# dummy_graph = dgl.graph(([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]), num_nodes=500)
# dummy_graph = dgl.add_self_loop(dummy_graph)
# dummy_graph.ndata['feat'] = torch.randn(500, 151)
# dummy_graph = dummy_graph.to('cuda')

# scaler = torch.cuda.amp.GradScaler()
# with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True, record_shapes=True) as prof:
#     with torch.no_grad():  
#         with torch.cuda.amp.autocast():
#             start_time = time.time()
#             for i in range(100):
#                 s = model(dummy_graph)
#             end_time = time.time()
#             print("CUDA FP16 time: ", (end_time - start_time) / 10)
            
# print(prof.key_averages(group_by_stack_n=5).table(sort_by="cpu_time_total",row_limit=10))
# print(prof.key_averages( group_by_stack_n=5).table(sort_by="cuda_time_total", row_limit=10))
# prof.export_stacks("./profiler_stacks.txt", "self_cuda_time_total")

# dummy_graph = dgl.graph(([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]), num_nodes=500)
# dummy_graph = dgl.add_self_loop(dummy_graph)
# dummy_graph.ndata['feat'] = torch.randn(500, 151)
# print("CPU FP32 testing... ")
# model = MyModel(151, 128, 64, 6, 0.3, 0.2, 500)
# model.eval()

# with torch.no_grad():
#     start_time = time.time()
#     for i in range(1000):
#         s = model(dummy_graph)
#     end_time = time.time()
#     print("CPU FP32 time: ", (end_time - start_time) / 10)
    
# print("CPU BF16 testing... ")
# dummy_graph = dgl.graph(([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]), num_nodes=500)
# dummy_graph = dgl.add_self_loop(dummy_graph)
# dummy_graph.ndata['feat'] = torch.randn(500, 151, dtype=torch.bfloat16)
# dummy_graph = dgl.to_bfloat16(dummy_graph)


# model = MyModel(151, 128, 64, 6, 0.3, 0.2, 500)
# model = model.to(dtype=torch.bfloat16)
# model.eval()

# with torch.no_grad():
#     start_time = time.time()
#     for i in range(1000):
#         s = model(dummy_graph)
#     end_time = time.time()
#     print("CPU FP16 time: ", (end_time - start_time) / 10)
