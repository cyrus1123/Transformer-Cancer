import torch

class FCMeans:
    def __init__(self, k=3, max_iters=5, m=2):
        self.k = k
        self.max_iters = max_iters
        self.m = m

    def initialize_centers(self, data):
        min_values, _ = torch.min(data, dim=0)
        max_values, _ = torch.max(data, dim=0)
        centers = torch.rand(self.k, data.shape[1]).cuda()
        for i in range(self.k):
            centers[i] = min_values + torch.rand(1, data.shape[1]).cuda() * (max_values - min_values)
        return centers

    def temp(self, data):
        centers = self.initialize_centers(data)
        for _ in range(self.max_iters):
            distances = torch.cdist(data, centers)
            memberships = 1.0 / (distances ** (2 / (self.m - 1)))
            memberships = memberships / memberships.sum(dim=1, keepdim=True)
            new_centers = torch.matmul(memberships.t(), data)
            new_centers = new_centers / memberships.sum(dim=0, keepdim=True).t()
            if torch.all(torch.isclose(new_centers, centers, atol=1e-5)):
                break
            centers = new_centers
        cluster_assignments = torch.argmax(memberships, dim=1).cpu()
        return cluster_assignments.numpy()

    def fcmean(self, X, Y):
        data = torch.cat((X.view(-1, 1), Y.view(-1, 1)), dim=1).cuda()
        self.initialize_centers(data)
        return self.temp(data)

    def buffer(self, *args):
        max_rows = max(arg.size(0) for arg in args)
        padded_tensors = [torch.cat((arg, torch.zeros(max_rows - arg.size(0), arg.size(1))), dim=0) for arg in args]
        concatenated_tensor = torch.hstack(padded_tensors)
        avg_value = torch.mean(concatenated_tensor)
        return concatenated_tensor

