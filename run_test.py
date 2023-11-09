from omegaconf import OmegaConf
from torch import multiprocessing as mp
from torch import distributed as dist
import os
from tools import Tester, log
import sys
import socket
from contextlib import closing


def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        return str(s.getsockname()[1])


if __name__ == "__main__":
    args = OmegaConf.load(
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     f"config/test.yaml")
    )

    #os.environ["MASTER_ADDR"] = args.distributed_addr
    os.environ["MASTER_PORT"] = find_free_port()
    world_size = args.world_size  # torch.cuda.device_count()
    log(f"start test on {world_size} GPU")
    if world_size == 1:
        tester = Tester(0, 1, args=args)
    else:
        try:
            log("STARTING SPAWN")
            # mp.set_start_method('spawn', force=True)
            mp.spawn(
                Tester,
                nprocs=world_size,
                args=(
                    world_size,
                    args,
                ),
                join=True,
            )
        except KeyboardInterrupt:
            log("Interrupted")
            try:
                dist.destroy_process_group()
            except KeyboardInterrupt:
                os.system(
                    "kill $(ps aux | grep multiprocessing.spawn | grep -v grep | awk '{print $2}') "
                )
