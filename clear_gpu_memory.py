#!/usr/bin/env python3
"""清理 GPU 显存的工具脚本"""

import subprocess
import sys

def get_gpu_processes():
    """获取占用 GPU 的进程列表"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True
        )
        processes = []
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                parts = line.split(', ')
                if len(parts) >= 3:
                    pid = parts[0].strip()
                    name = parts[1].strip()
                    memory = parts[2].strip()
                    processes.append((pid, name, memory))
        return processes
    except Exception as e:
        print(f"Error getting GPU processes: {e}")
        return []

def kill_process(pid):
    """安全地终止进程"""
    try:
        import os
        import signal
        os.kill(int(pid), signal.SIGTERM)
        print(f"Sent SIGTERM to process {pid}")
        return True
    except ProcessLookupError:
        print(f"Process {pid} not found (may have already terminated)")
        return False
    except PermissionError:
        print(f"Permission denied: Cannot kill process {pid}")
        return False
    except Exception as e:
        print(f"Error killing process {pid}: {e}")
        return False

def clear_cuda_cache():
    """清理 CUDA 缓存（如果在 Python 环境中）"""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print("✓ Cleared PyTorch CUDA cache")
            return True
    except ImportError:
        pass
    except Exception as e:
        print(f"Error clearing CUDA cache: {e}")
    return False

def main():
    print("=" * 60)
    print("GPU Memory Cleanup Tool")
    print("=" * 60)
    
    # 显示当前 GPU 使用情况
    processes = get_gpu_processes()
    if not processes:
        print("\n✓ No compute processes found using GPU")
        print("  (Display processes like Xorg may still show memory usage)")
    else:
        print(f"\nFound {len(processes)} compute process(es) using GPU:")
        total_memory = 0
        for pid, name, memory in processes:
            mem_mib = memory.replace(' MiB', '')
            try:
                total_memory += int(mem_mib)
            except:
                pass
            print(f"  PID {pid:>6}: {memory:>10} - {name}")
        print(f"\nTotal GPU memory used: ~{total_memory} MiB")
        
        # 询问是否终止进程
        if len(sys.argv) > 1 and sys.argv[1] == '--kill':
            print("\nAttempting to terminate processes...")
            for pid, name, memory in processes:
                kill_process(pid)
        else:
            print("\nTo kill these processes, run:")
            print(f"  python {sys.argv[0]} --kill")
            print("\nOr manually kill with:")
            for pid, name, memory in processes:
                print(f"  kill {pid}")
    
    # 尝试清理 CUDA 缓存
    print("\nAttempting to clear CUDA cache...")
    clear_cuda_cache()
    
    print("\n" + "=" * 60)
    print("Note: If processes are already terminated but memory persists,")
    print("      you may need to restart the application or reboot.")
    print("=" * 60)

if __name__ == "__main__":
    main()




