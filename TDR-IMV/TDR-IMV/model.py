import torch.nn as nn
import torch

class TDR_IMV(nn.Module):
    def __init__(self, num_views, dims, num_classes, device,annealing_epochs=1):
        super(TDR_IMV, self).__init__()
        self.num_views = num_views
        self.num_classes = num_classes
        self.EvidenceCollectors = nn.ModuleList([Net(dims[i], self.num_classes) for i in range(self.num_views)])
        self.device = device
        self.annealing_epochs = annealing_epochs

    def Evidence_DC(self, alpha, beta):
        E = dict()
        for v in range(len(alpha)):
            E[v] = alpha[v]-1
            E[v] = torch.nan_to_num(E[v], 0)

        for v in range(len(alpha)):
            E[v] = torch.nan_to_num(E[v], 0)

        E_con = E[0]

        for i in range(len(E) - 1):
            for j in range(i + 1, len(E)):
                E_con = torch.min(torch.max(E[i], E[j]), E_con)

        for v in range(len(alpha)):
            E[v] = torch.abs(E[v] - E_con)
        alpha_con = E_con + 1

        E_div = E[0]
        for v in range(1,len(alpha)):
            E_div = torch.add(E_div, E[v])

        E_div = torch.div(E_div, len(alpha))

        S_con = torch.sum(alpha_con, dim=1, keepdim=True)

        b_con = torch.div(E_con, S_con)
        S_b = torch.sum(b_con, dim=1, keepdim=True)

        b_con2 = torch.pow(b_con, beta)
        S_b2 = torch.sum(b_con2,dim=1, keepdim=True)

        b_cona = torch.mul(b_con2, torch.div(S_b, S_b2))

        E_con = torch.mul(b_cona, S_con)

        E_con = torch.mul(E_con, len(alpha))
        E_a = torch.add(E_con, E_div)

        alpha_a = E_a + 1
        alpha_con = E_con + 1
        alpha_div = E_div+1

        alpha_a = torch.nan_to_num(alpha_a, 1)
        alpha_con = torch.nan_to_num(alpha_con, 1)
        alpha_div = torch.nan_to_num(alpha_div, 1)

        Sum = torch.sum(alpha_a, dim=1, keepdim=True)
        return alpha_a, alpha_con, alpha_div

    def forward(self, X, y, global_step, beta):
        evidences = self.collect(X)
        alpha = dict()
        for v_num in range(len(X)):
            alpha[v_num] = evidences[v_num] + 1
        alpha_a, alpha_con, alpha_div = self.Evidence_DC(alpha, beta)
        evidence_a = alpha_a - 1
        evidence_con = alpha_con - 1
        evidence_div = alpha_div - 1
        return evidences, evidence_a, evidence_con, evidence_div

    def collect(self, input):
        """
        :param input: Multi-view data
        :return: evidence of every view
        """
        evidence = dict()
        for v_num in range(self.num_views):
            evidence[v_num] = self.EvidenceCollectors[v_num](input[v_num])
        return evidence

class Net(nn.Module):
    def __init__(self, classifier_dims, classes):
        super(Net, self).__init__()
        self.classes = classes
        self.fc = nn.ModuleList()
        self.fc.append(nn.Linear(classifier_dims[0], 64))
        self.fc.append(nn.Sigmoid())

        self.evidence = nn.ModuleList()
        self.evidence.append(nn.Linear(64, classes))
        self.evidence.append(nn.Softplus())


    def forward(self, x):

        out = self.fc[0](x)
        for i in range(1, len(self.fc)):
            out = self.fc[i](out)

        evidence = self.evidence[0](out)
        evidence = self.evidence[1](evidence)

        return evidence*evidence
