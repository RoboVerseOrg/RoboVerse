import zmq
import torch
import time
class FrankaRobotClient():
    def __init__(self, socket_number = 5556):
        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)
        self.socket.connect(f"tcp://127.0.0.1:{socket_number}")
        print(f"Franka Robot Client connected to tcp://127.0.0.1:{socket_number}")

    def goto(self, goal):
        message = {"command": "goto"}
        goal = [i.item() if isinstance(i, torch.Tensor) else i for i in goal]
        message['goal'] = goal
        self.socket.send_json(message)
        response = self.socket.recv_json()
        if response['status'] == 'error':
            raise RuntimeError(f"Error in goto command: {response['message']}")
        return

    def do_homing(self):
        message = {"command": "homing"}
        self.socket.send_json(message)
        response = self.socket.recv_json()
        if response['status'] == 'error':
            raise RuntimeError(f"Error in homing command: {response['message']}")
        return

    def get_state(self, return_ee=False):
        message = {"command": "get_state", "return_ee": return_ee}
        self.socket.send_json(message)
        response = self.socket.recv_json()
        if response['status'] == 'error':
            raise RuntimeError(f"Error in get_state command: {response['message']}")
        return response['state']


    def goto_ee_state(self, state8):
        message = {"command": "goto_ee_state", "goal": state8}
        self.socket.send_json(message)
        response = self.socket.recv_json()
        if response['status'] == 'error':
            raise RuntimeError(f"Error in goto command: {response['message']}")
        return
