import torch
from torch_geometric.nn import GAE, GAT
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau


import os
import argparse
import wandb
from os.path import join, exists
from codecarbon import EmissionsTracker
from models import DistMultDecoder, GNNEncoder
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from tqdm import tqdm

INPUT_CHANNELS = 150
HIDDEN_CHANNELS = 150
OUTPUT_CHANNELS = 150


def main():
    if torch.cuda.is_available():
        print('All good, a GPU is available')
        device = torch.device("cuda")
    else:
        print('No GPU available, using CPU')
        device = 'cpu'
    
    if not args.wandb_log:
        os.environ["WANDB_DISABLED"] = "true"
    
    wandb.init(project="GraphLearning")

    dataset = PygLinkPropPredDataset(name="ogbl-biokg", root='datasets/')
    num_nodes = sum([num_nod for _, num_nod in dataset.data.node_stores[0]._mapping['num_nodes_dict'].items()])
    num_relations = len(dataset.data.node_stores[0]._mapping['edge_reltype'])

    model = GAE(
        GNNEncoder(num_nodes, input_channels=INPUT_CHANNELS, hidden_channels=HIDDEN_CHANNELS, 
                   output_channels=OUTPUT_CHANNELS, num_relations=num_relations, gnn_model=args.encoder),
        DistMultDecoder(num_relations, input_channels=INPUT_CHANNELS),
    )
    if torch.cuda.is_available():
        model.cuda()

    split_edge = dataset.get_edge_split()
    train_triples, val_triples, test_triples = split_edge["train"], split_edge["valid"], split_edge["test"]
    dataset_head = torch.cat((train_triples['head'], val_triples['head'], test_triples['head']))
    dataset_tail = torch.cat((train_triples['tail'], val_triples['tail'], test_triples['tail']))
    edge_index = torch.stack((dataset_head, dataset_tail)).to(device)
    edge_type = torch.cat((train_triples['relation'], val_triples['relation'], test_triples['relation'])).to(device)

    # define edge_index and edge_type for each split
    train_edge_index = torch.stack((train_triples['head'], train_triples['tail'])).to(device)
    train_edge_type = train_triples['relation'].to(device)
    
    tracker = EmissionsTracker(measure_power_secs=100000, save_to_file=False)
    tracker.start()
    wandb.watch(model)

    train(model, edge_index, edge_type, train_edge_index, 
          train_edge_type, val_triples, num_nodes, len(train_triples['head_type']), device)

    emissions = tracker.stop()
    wandb.log({'Total CO2 emission (in Kg)': emissions})
    test_mrr_list = test(model, edge_index, edge_type, test_triples, device)
    test_mrr_value = torch.mean(torch.stack(test_mrr_list))

    print(f'Test MRR: {test_mrr_value:.4f}')



def negative_sampling(edge_index, num_nodes, device):
    # Sample edges by corrupting either the subject or the object of each edge.
    mask_1 = torch.rand(edge_index.size(1)) < 0.5
    mask_2 = ~mask_1

    neg_edge_index = edge_index.clone()
    neg_edge_index[0, mask_1] = torch.randint(num_nodes, (mask_1.sum(), )).to(device)
    neg_edge_index[1, mask_2] = torch.randint(num_nodes, (mask_2.sum(), )).to(device)
    return neg_edge_index


def train(model, edge_index, edge_type, train_edge_index, train_edge_type, val_triples, num_nodes, num_train_edges, device):
    ckpt_dir = 'checkpoints/rgcn'
    if not exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    batch_size = 64 * 1024
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, 'max', verbose=True,
                                  factor=0.5, min_lr=0,
                                  patience=5)
    model.train()
    
    for epoch in range(1, 90):

        pgb = tqdm(DataLoader(range(num_train_edges), batch_size, shuffle=True), leave=False)

        for edge_id in pgb:

            z = model.encode(edge_index, edge_type)
            pos_out = model.decode(z, train_edge_index[:, edge_id], train_edge_type[edge_id])

            neg_edge_index = negative_sampling(train_edge_index[:, edge_id], num_nodes, device)
            neg_out = model.decode(z, neg_edge_index, train_edge_type[edge_id])

            positive_score = F.logsigmoid(pos_out)
            negative_score = F.logsigmoid(-neg_out)

            positive_sample_loss = - positive_score.mean()
            negative_sample_loss = - negative_score.mean()
            sample_loss = (positive_sample_loss + negative_sample_loss)/2

            reg_loss = z.pow(2).mean() + model.decoder.rel_emb.pow(2).mean()
            loss = sample_loss + 1e-2 * reg_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            model.zero_grad()
        
        val_mrr_list = test(model, edge_index, edge_type, val_triples, device)
        val_mrr_value = torch.mean(torch.stack(val_mrr_list))

        print(f'Epoch: {epoch:05d}, Loss: {loss:.4f}, Val MRR: {val_mrr_value:.4f}')

        if (epoch % 100) == 0:
            save_dict = {}
            name = 'ckpt2-{}'.format(epoch)
            save_dict['state_dict'] = model.state_dict()
            save_dict['optimizer_state_dict'] = optimizer.state_dict()
            torch.save(save_dict, join(ckpt_dir, name))
     
        wandb.log({
            'training_loss': loss,
            'epoch': epoch,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'val_mrr': val_mrr_value
        })

        scheduler.step(val_mrr_value)


@torch.no_grad()
def test(model, edge_index, edge_type, test_triples, device):
    
    batch_size = 1024
    evaluator = Evaluator(name = "ogbl-biokg")
    model.eval()

    z = model.encode(edge_index, edge_type)

    # Prepare negative test edges for evaluation
    test_edge_index = torch.stack((test_triples['head'], test_triples['tail'])).to(device)
    test_edge_type = test_triples['relation'].to(device)

    pgb = tqdm(DataLoader(range(len(test_triples['head_type'])), batch_size, shuffle=True), leave=False)
    
    mrr_list = []
    for edge_id in pgb:

        pos_out = model.decode(z, test_edge_index[:, edge_id], test_edge_type[edge_id])
        neg_head_edge_index = torch.stack((test_triples['head_neg'][edge_id, :].flatten().to(device), test_edge_index[1][edge_id].repeat_interleave(500)))
        neg_tail_edge_index = torch.stack((test_edge_index[0][edge_id].repeat_interleave(500), test_triples['tail_neg'][edge_id, :].flatten().to(device)))
        neg_test_edge_index = torch.cat([neg_head_edge_index, neg_tail_edge_index], 1)
        neg_edge_type = test_triples['relation'][edge_id].repeat_interleave(1000).to(device)

        neg_out = model.decode(z, neg_test_edge_index, neg_edge_type).view(pos_out.size(0), 1000)

        batch_results = evaluator.eval({'y_pred_pos': pos_out, 'y_pred_neg': neg_out})
        mrr_list.extend(batch_results['mrr_list'])
    
    return mrr_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='training of the abstractor'
    )
    parser.add_argument('--wandb_log',  action='store_true', default=False,
                        help='login to wandb')
    parser.add_argument('--encoder',  action='store', default='rgcn',
                        help='GNN model')
    args = parser.parse_args()

    main()