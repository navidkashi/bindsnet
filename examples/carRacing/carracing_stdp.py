from bindsnet.encoding import bernoulli
from bindsnet.encoding.encoders import Encoder
from bindsnet.environment import GymEnvironment
from bindsnet.learning import MSTDP
from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection
from bindsnet.pipeline import EnvironmentPipeline
from bindsnet.pipeline.action import select_softmax
import torch
from math import pi
import torch.nn.functional as F
import matplotlib.pyplot as plt


class GaborEncoder():
    def __init__(self):
        num_filters = 4
        input_size = 80
        self.stride = 2
        self.th = 0.6
        self.kernel = torch.zeros(size=(num_filters, 1, 11, 11))
        kernel_size = self.kernel.shape[2]
        middle_size = int((input_size - kernel_size) / self.stride + 1)
        for i in range(0, num_filters):
            self.kernel[i, 0] = self.gaborFilter(kernel_size, 1, 20, (i / float(num_filters)) * pi, 0.1)


    def gaborFilter(self, size, sigma, lmbda, theta, gamma, on_center=True):
        sigm = torch.tensor(sigma)
        lmbd = torch.tensor(lmbda)
        thet = torch.tensor(theta)
        gamm = torch.tensor(gamma)
        line_x = torch.linspace(-int(size / 2), int(size / 2), size)
        line_y = torch.linspace(-int(size / 2), int(size / 2), size)
        x, y = torch.meshgrid(line_x, line_y)
        rotated_x = x * torch.cos(thet) + y * torch.sin(thet)
        rotated_y = -x * torch.sin(thet) + y * torch.cos(thet)
        gabor = torch.exp(-(torch.pow(rotated_x, 2) + gamm ** 2 * torch.pow(rotated_y, 2)) / (2 * sigm ** 2)) * \
                torch.cos(2 * pi * rotated_x / lmbd)
        if not on_center:
            gabor = -1 * gabor

        gabor = gabor - gabor.mean()
        return -gabor

    def __call__(self, datum, time, dt: float = 1.0, device="cpu"):

        output = F.conv2d(
            datum,
            self.kernel,
            stride=(self.stride, self.stride),
            padding=0,
        ).squeeze()

        if (output.max()):
            output = output / output.max()

        output = output - self.th
        output[output < 0] = 0
        data = output
        time = int(time / dt)
        shape, size = data.shape, data.numel()
        _data = data.flatten()
        spikes = torch.zeros((time + 1, size))
        times = torch.zeros_like(_data).long() + time
        if data.max():
            times = time - (_data * (time / (0.3 * data.max()))).long()
        times[times < 0] = 0
        spikes[times, torch.arange(size)] = 1
        spikes = spikes[:-1]
        out = spikes.view(time, *shape).bool()

        return out.unsqueeze(1)


# Build network.
network = Network(dt=1.0)

# Layers of neurons.
inpt = Input(n=4 * 35 * 35, shape=[1, 4, 35, 35], traces=True)
middle = LIFNodes(n=100, traces=True)
out = LIFNodes(n=5, refrac=0, traces=True)

# Connections between layers.
inpt_middle = Connection(source=inpt, target=middle, wmin=0, wmax=1e-1)
middle_out = Connection(
    source=middle,
    target=out,
    wmin=0,
    wmax=1,
    update_rule=MSTDP,
    nu=1e-1,
    norm=0.5 * middle.n,
)

# Add all layers and connections to the network.
network.add_layer(inpt, name="Input Layer")
network.add_layer(middle, name="Hidden Layer")
network.add_layer(out, name="Output Layer")
network.add_connection(inpt_middle, source="Input Layer", target="Hidden Layer")
network.add_connection(middle_out, source="Hidden Layer", target="Output Layer")

# Load the Breakout environment.
environment = GymEnvironment("CarRacing-v2", add_channel_dim=False)
environment.reset()

# Build pipeline from specified components.
environment_pipeline = EnvironmentPipeline(
    network,
    environment,
    encoding=GaborEncoder(),
    action_function=select_softmax,
    output="Output Layer",
    time=100,
    history_length=1,
    plot_interval=1,
    plot_config={"data_step": True, "data_length": 100},
    delta=1,
    render_interval=1,
)


def run_pipeline(pipeline, episode_count):
    for i in range(episode_count):
        total_reward = 0
        pipeline.reset_state_variables()
        is_done = False
        while not is_done:
            result = pipeline.env_step()
            pipeline.step(result)

            reward = result[1]
            total_reward += reward

            is_done = result[2]
        print(f"Episode {i} total reward:{total_reward}")


print("Training: ")
run_pipeline(environment_pipeline, episode_count=100)

# stop MSTDP
environment_pipeline.network.learning = False

print("Testing: ")
run_pipeline(environment_pipeline, episode_count=100)