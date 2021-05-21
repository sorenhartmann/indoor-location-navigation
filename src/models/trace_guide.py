import torch
from torch.nn import Module, Parameter

class TraceGuide(Module):

    def __init__(self, n=None, loc_bias=None, time_min_max=None):

        super().__init__()

        if n is None:
            n = 10
        
        # Setup basis coefficients
        self.register_parameter("coeffs", Parameter(torch.empty(n, 2)))
        torch.nn.init.xavier_normal_(self.coeffs)

        # Setup location bias
        if loc_bias is not None:
            self.register_parameter("location_bias", Parameter(loc_bias))
        else:
            self.register_parameter("location_bias", Parameter(torch.zeros(2)))

        # Setup time offsets (gamma)
        if time_min_max is not None:
            time_offset = torch.linspace(*time_min_max, n).view(-1, 1).tile(1, 2)
        else:
            time_offset = torch.linspace(0., 10., n).view(-1, 1).tile(1, 2)
        self.register_parameter("time_offset", Parameter(time_offset))

        self.register_parameter("log_scale", torch.nn.Parameter(torch.tensor(0.)))
        
    def forward(self, x):
        """ Shape should be (some number of batch dims, 1) """

        x = x.unsqueeze(1) - self.time_offset
        x = torch.nn.functional.softplus(x)
        x = x * self.coeffs
        x = x.sum(1) + self.location_bias

        location = x
        scale = self.log_scale.exp()

        return location, scale

        in_size = 1
        for out_size in layer_sizes:
            layers_x.append(Linear(in_size, out_size))
            layers_y.append(Linear(in_size, out_size))
            layers_x.append(Softplus(beta=1.))
            layers_y.append(Softplus(beta=1.))
            in_size = out_size

        layers_x.append(Linear(in_size, 1))
        layers_y.append(Linear(in_size, 1))

        if loc_bias is not None:
            layers_x[-1].bias = torch.nn.Parameter(loc_bias[0])
            layers_y[-1].bias = torch.nn.Parameter(loc_bias[1])

        if time_min_max is not None:
            linspace = torch.linspace(time_min_max[0], time_min_max[1], layer_sizes[0])
            layers_x[0].bias = torch.nn.Parameter(-linspace)
            layers_y[0].bias = torch.nn.Parameter(-linspace)
            layers_x[0].weight = torch.nn.Parameter(torch.ones_like(layers_x[0].weight))
            layers_y[0].weight = torch.nn.Parameter(torch.ones_like(layers_y[0].weight))
            layers_x[0].weight.requires_grad = False
            layers_y[0].weight.requires_grad = False

        self._forward_x = Sequential(*layers_x)
        self._forward_y = Sequential(*layers_y)

        self.register_parameter("log_scale", torch.nn.Parameter(torch.tensor(0.)))

    # def forward(self, x):

    #     location = torch.cat([self._forward_x(x), self._forward_y(x)], dim=1)
    #     scale = self.log_scale.exp()

    #     return location, scale
