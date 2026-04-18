import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# 1. PRUNABLE LINEAR LAYER
class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # learnable gates
        self.gate_scores = nn.Parameter(torch.randn(out_features, in_features)-1)

    def forward(self, x):
        gates = torch.sigmoid(self.gate_scores)     # 0 → 1
        pruned_weight = self.weight * gates          # prune here
        return F.linear(x, pruned_weight, self.bias)


# 2. MODEL
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.fc1 = PrunableLinear(32*32*3, 512)
        self.fc2 = PrunableLinear(512, 256)
        self.fc3 = PrunableLinear(256, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x


# 3. SPARSITY LOSS (L1)

def sparsity_loss(model):
    loss = 0
    total = 0
    
    for m in model.modules():
        if isinstance(m, PrunableLinear):
            gates = torch.sigmoid(2*m.gate_scores)
            loss += torch.sum(gates)
            total += gates.numel()
    
    return loss / total   

# 4. DATA

transform = transforms.ToTensor()

train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = DataLoader(test_data, batch_size=128)

# 5. TRAIN
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

lambda_sparse = 1e-4 # CHANGE THIS (important)

EPOCHS = 8

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        
        out = model(x)
        
        cls_loss = criterion(out, y)
        sp_loss = sparsity_loss(model)
        
        loss = cls_loss + lambda_sparse * sp_loss
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print("Avg Gate",torch.mean(torch.sigmoid(model.fc1.gate_scores)).item())


# 6. EVALUATION
def evaluate(model):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            
            out = model(x)
            _, pred = torch.max(out, 1)
            
            total += y.size(0)
            correct += (pred == y).sum().item()
    
    return 100 * correct / total


# 7. SPARSITY %
def sparsity(model, threshold=0.2):
    total = 0
    pruned = 0
    
    for m in model.modules():
        if isinstance(m, PrunableLinear):
            gates = torch.sigmoid(m.gate_scores)
            
            total += gates.numel()
            pruned += (gates < threshold).sum().item()
    
    return 100 * pruned / total


# 8. RESULTS
acc = evaluate(model)
sp = sparsity(model)

print(f"\nTest Accuracy: {acc:.2f}%")
print(f"Sparsity: {sp:.2f}%")



# 9. PLOT GATES

all_gates = []

for m in model.modules():
    if isinstance(m, PrunableLinear):
        gates = torch.sigmoid(m.gate_scores).detach().cpu().numpy()
        all_gates.extend(gates.flatten())

plt.hist(all_gates, bins=50)
plt.title("Gate Distribution")
plt.show()