import torch
from torch_geometric.nn import GAE
from torch.utils.data import DataLoader
import torch.nn.functional as F

from models import RGCNEncoder, DistMultDecoder
from ogb.linkproppred import PygLinkPropPredDataset
from tqdm import tqdm


def main():
    if torch.cuda.is_available():
        print('All good, a GPU is available')
        device = torch.device("cuda")
    else:
        print('No GPU available, using CPU')
        device = 'cpu'

    dataset = PygLinkPropPredDataset(name="ogbl-biokg", root='datasets/')
    num_nodes = sum([num_nod for _, num_nod in dataset.data.node_stores[0]._mapping['num_nodes_dict'].items()])
    num_relations = len(dataset.data.node_stores[0]._mapping['edge_reltype'])

    model = GAE(
        RGCNEncoder(num_nodes, hidden_channels=500,
                    num_relations=num_relations),
        DistMultDecoder(num_relations, hidden_channels=500),
    )

    split_edge = dataset.get_edge_split()
    train_triples, valid_triples, test_triples = split_edge["train"], split_edge["valid"], split_edge["test"]
    dataset_head = torch.cat((train_triples['head'], valid_triples['head'], test_triples['head']))
    dataset_tail = torch.cat((train_triples['tail'], valid_triples['tail'], test_triples['tail']))
    edge_index = torch.stack((dataset_head, dataset_tail))
    edge_type = torch.cat((train_triples['relation'], valid_triples['relation'], test_triples['relation']))

    for epoch in range(1, 10001):
        loss = train(model, train_triples, edge_index, edge_type, num_nodes, device)
        print(f'Epoch: {epoch:05d}, Loss: {loss:.4f}')
        if (epoch % 500) == 0:
            valid_mrr = test(model, edge_index, edge_type, valid_triples)
            print(f'Val MRR: {valid_mrr:.4f}')

    test_mrr = test(model, edge_index, edge_type, test_triples)
    print(f'Test MRR: {test_mrr:.4f}')


def negative_sampling(edge_index, num_nodes):
    # Sample edges by corrupting either the subject or the object of each edge.
    mask_1 = torch.rand(edge_index.size(1)) < 0.5
    mask_2 = ~mask_1

    neg_edge_index = edge_index.clone()
    neg_edge_index[0, mask_1] = torch.randint(num_nodes, (mask_1.sum(), ))
    neg_edge_index[1, mask_2] = torch.randint(num_nodes, (mask_2.sum(), ))
    return neg_edge_index


def train(model, train_triples, edge_index, edge_type, num_nodes, device):
    batch_size = 64 * 1024
    num_train_edges = len(train_triples['head_type'])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    optimizer.zero_grad()
    train_edge_index = torch.stack((train_triples['head'], train_triples['tail']))
    train_edge_type = train_triples['relation']

    for edge_id in DataLoader(range(num_train_edges), batch_size, shuffle=True):

        z = model.encode(edge_index, edge_type)
        pos_out = model.decode(z, train_edge_index[:, edge_id], train_edge_type[edge_id])

        neg_edge_index = negative_sampling(train_edge_index[:, edge_id], num_nodes)
        neg_out = model.decode(z, neg_edge_index, train_edge_type[edge_id])

        out = torch.cat([pos_out, neg_out])
        gt = torch.cat([torch.ones_like(pos_out), torch.zeros_like(neg_out)])
        cross_entropy_loss = F.binary_cross_entropy_with_logits(out, gt)
        reg_loss = z.pow(2).mean() + model.decoder.rel_emb.pow(2).mean()
        loss = cross_entropy_loss + 1e-2 * reg_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()

    return loss


@torch.no_grad()
def test(model, edge_index, edge_type, test_triples):
    model.eval()
    test_edge_index = torch.stack((test_triples['head'], test_triples['tail']))
    test_edge_type = test_triples['relation']
    z = model.encode(edge_index, edge_type)

    test_mrr = compute_mrr(z, test_edge_index, test_edge_type)

    return test_mrr


@torch.no_grad()
def compute_rank(ranks):
    # fair ranking prediction as the average
    # of optimistic and pessimistic ranking
    true = ranks[0]
    optimistic = (ranks > true).sum() + 1
    pessimistic = (ranks >= true).sum()
    return (optimistic + pessimistic).float() * 0.5


@torch.no_grad()
def compute_mrr(z, edge_index, edge_type):
    ranks = []
    for i in tqdm(range(edge_type.numel())):
        (src, dst), rel = edge_index[:, i], edge_type[i]

        # Try all nodes as tails, but delete true triplets:
        tail_mask = torch.ones(data.num_nodes, dtype=torch.bool)
        for (heads, tails), types in [
            (data.train_edge_index, data.train_edge_type),
            (data.valid_edge_index, data.valid_edge_type),
            (data.test_edge_index, data.test_edge_type),
        ]:
            tail_mask[tails[(heads == src) & (types == rel)]] = False

        tail = torch.arange(data.num_nodes)[tail_mask]
        tail = torch.cat([torch.tensor([dst]), tail])
        head = torch.full_like(tail, fill_value=src)
        eval_edge_index = torch.stack([head, tail], dim=0)
        eval_edge_type = torch.full_like(tail, fill_value=rel)

        out = model.decode(z, eval_edge_index, eval_edge_type)
        rank = compute_rank(out)
        ranks.append(rank)

        # Try all nodes as heads, but delete true triplets:
        head_mask = torch.ones(data.num_nodes, dtype=torch.bool)
        for (heads, tails), types in [
            (data.train_edge_index, data.train_edge_type),
            (data.valid_edge_index, data.valid_edge_type),
            (data.test_edge_index, data.test_edge_type),
        ]:
            head_mask[heads[(tails == dst) & (types == rel)]] = False

        head = torch.arange(data.num_nodes)[head_mask]
        head = torch.cat([torch.tensor([src]), head])
        tail = torch.full_like(head, fill_value=dst)
        eval_edge_index = torch.stack([head, tail], dim=0)
        eval_edge_type = torch.full_like(head, fill_value=rel)

        out = model.decode(z, eval_edge_index, eval_edge_type)
        rank = compute_rank(out)
        ranks.append(rank)

    return (1. / torch.tensor(ranks, dtype=torch.float)).mean()

