import os, runpy, sys, torch
torch.cuda.reset_peak_memory_stats()
sys.argv = ['-u', '/home/al.mikhalev/Code/nntile/torch_nntile/examples/train_gpt2_hf.py', 'train', '--seed', '--device', 'cuda', '--disable-tf32', '42', '--no-shuffle', '--config', '/home/al.mikhalev/Code/nntile/torch_nntile/examples/overhead_gpt2/gpt2_xs.json', '--seq-len', '768', '--batch-size', '1', '--max-sequences', '10', '--epochs', '1', '--output-dir', '/tmp/gpt2_overhead_rerun_20260825/xs_cuda_overlap']
runpy.run_path('/home/al.mikhalev/Code/nntile/torch_nntile/examples/train_gpt2_hf.py', run_name='__main__')
peak = torch.cuda.max_memory_allocated() / (1024**3)
print(f'peak_vram_gib={peak:.2f}', flush=True)
