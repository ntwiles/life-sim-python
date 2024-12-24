from multiprocessing import Pipe, Process

from simulator.main import simulator_worker
from viewer.main import viewer_worker

def main():
    recv_conn, send_conn = Pipe()
    
    viewer = Process(target=viewer_worker, daemon=True, args=(recv_conn,))
    viewer.start()

    simulator = Process(target=simulator_worker, daemon=True, args=(send_conn,))
    simulator.start()

    simulator.join()
    send_conn.close()
    recv_conn.close()

if __name__ == "__main__":
    main()