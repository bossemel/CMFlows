from DDSF import build_model as build_model_ddsf
from NSF import build_model as build_model_nsf


class CMFlow:
    def __init__(self, args):
        super(CMFlow, self).__init__()

        self.args = args
        if args.cop_flow == 'NSF':
            self.cop_flow = build_model_nsf(args, flow_type='cop_flow')

        if args.marg_flow == 'NSF':
            self.marg_flow_1 = build_model_nsf(args, flow_type='marg_flow')
            self.marg_flow_2 = build_model_nsf(args, flow_type='marg_flow')
        elif args.marg_flow == 'DDSF':
            self.marg_flow_1 = build_model_ddsf(args)
            self.marg_flow_2 = build_model_ddsf(args)

    def init_marg_flow(self):
        self.marg_flow = build_model_nsf(self.args, flow_type='marg_flow')

    def train(self):
        self.cop_flow.train()
        self.marg_flow_1.train()
        self.marg_flow_2.train()

    # def eval(self):
    #     self.cop_flow.eval()
    #     self.marg_flow_1.eval()
    #     self.marg_flow_2.eval()

    def to(self, device):
        self.cop_flow.to(device)
        self.marg_flow_1.to(device)
        self.marg_flow_2.to(device)
