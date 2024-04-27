import torch
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from Generator import Generator, Predictor
import copy

def similarity_loss(adjs_dense, perturbation_adjs, masking_matrix):
    diff = adjs_dense-perturbation_adjs
    diff_norm = torch.linalg.matrix_norm(diff, ord=1)/torch.ones_like(diff).sum()
    masking_matrix_norm = torch.linalg.matrix_norm(masking_matrix, ord=1)/torch.ones_like(masking_matrix).sum()
    return diff_norm-masking_matrix_norm

def kl_div(predicted_results, perturbation_predicted_results, masking_predicted_results):
    predicted_results = predicted_results.softmax(dim=1)
    perturbation_predicted_results = perturbation_predicted_results.log_softmax(dim=1)
    masking_predicted_results = masking_predicted_results.log_softmax(dim=1)
    loss_func = torch.nn.KLDivLoss(reduction='batchmean')
    kl_loss_1 = loss_func(perturbation_predicted_results, predicted_results)
    kl_loss_2 = loss_func(masking_predicted_results, predicted_results)

    return kl_loss_1+kl_loss_2

def generation(args, graphs_num, nodes_num_list, attrs_dim, data):
    dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=True)
    dataloader = [batch for batch in dataloader]
    if args.cuda:
        dataloader = [batch.cuda() for batch in dataloader]

    generator = Generator(args, graphs_num, nodes_num_list, attrs_dim)
    if args.cuda:
        generator.cuda()

    optimizer = torch.optim.Adam(
        params=generator.parameters(),
        lr=args.generation_lr
    )

    data_perturbation_list = []
    data_masking_list = []
    both_list = []

    pbar = tqdm(range(args.generation_epochs))
    for epoch in pbar:
        pbar.set_description('Hard Negative Samples Generation Epoch %d...' % epoch)

        for batch in dataloader:
            optimizer.zero_grad()

            adjs_dense, perturbation_adjs, masking_matrix, predicted_results, perturbation_predicted_results, masking_predicted_results, data_perturbation, data_masking = generator(
                batch)
            sim_loss = similarity_loss(adjs_dense, perturbation_adjs, masking_matrix)
            kl_loss = kl_div(predicted_results, perturbation_predicted_results, masking_predicted_results)
            l = sim_loss - kl_loss

            l.backward()
            optimizer.step()

            pbar.set_postfix(sim_loss=sim_loss.item(), kl_loss=kl_loss.item())

    pbar = tqdm(range(len(data)))
    pbar.set_description('Augmented Graphs Generation...')
    for i in pbar:
        each = data[i]
        if args.cuda:
            each = each.cuda()


        each_both = copy.deepcopy(each)

        p_matrix = generator.perturbation_matrices[each.id]
        p_bias = generator.perturbation_biases[each.id]
        m_matrix = generator.masking_matrices[each.id]

        values = torch.Tensor([1 for i in range(each.edge_index.size()[1])])
        if args.cuda:
            values = values.cuda()
        adjs = torch.sparse_coo_tensor(each.edge_index, values, (each.num_nodes, each.num_nodes), dtype=torch.float)
        adjs_dense = adjs.to_dense()
        perturbation_adjs = torch.mm(p_matrix, adjs_dense) + p_bias
        perturbation_adjs = torch.sigmoid(perturbation_adjs)
        perturbation_adjs = torch.where(perturbation_adjs <= args.gamma, torch.zeros_like(perturbation_adjs),
                                        torch.ones_like(perturbation_adjs))
        perturbation_adjs_sparse = perturbation_adjs.to_sparse()
        each_both.edge_index = perturbation_adjs_sparse.indices()

        masking_matrices = torch.sigmoid(m_matrix)
        masking_matrices = torch.where(masking_matrices <= args.gamma, torch.zeros_like(masking_matrices),
                                       torch.ones_like(masking_matrices))
        masked_attrs = torch.mul(masking_matrices, each.x)
        each_both.x = masked_attrs
        both_list.append(each_both)


    return both_list