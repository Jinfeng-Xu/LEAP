from typing import List
import copy

from torch_geometric.data import Batch
from torch_scatter import scatter
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score

from model import PromptedGNN, GNNBasedNet
from agent import H_PPO
from reward import compute_adv_ret, reshape_reward, reward_reshape, Normalization

criterion = nn.BCEWithLogitsLoss(reduction="none")

def attach_prompt(args, policy: H_PPO, data, data_type, gnn=None, tasknet=None, compute_reward=0, compute_prr=0):
    device = data.x.device

    graph_num = data.num_graphs
    nodes_per_graph = scatter(torch.ones_like(data.batch), data.batch, reduce='add')
    policy.prompt_slices = [torch.zeros(nodes, args.emb_dim).to(device) for nodes in nodes_per_graph]
    policy.step_state = torch.zeros(graph_num, policy.max_num_nodes, args.emb_dim).to(device)
    base_mask = torch.zeros(graph_num, policy.max_num_nodes).to(device)
    base_idx = torch.arange(0, policy.max_num_nodes).to(device)
    policy.batch_mask = torch.where(base_idx < nodes_per_graph.unsqueeze(-1), base_mask, torch.ones_like(base_mask))

    if data_type != "train":
        total_prompt_step_num = 0
    if compute_reward:
        init_r = generate_reward(gnn, tasknet, data)
        r_list = [copy.deepcopy([]) for _ in range(graph_num)]
        for rl, ir in zip(r_list, init_r):
            rl.append(ir)
    
    truncate_flag = torch.zeros(graph_num)
    batch_max_step = nodes_per_graph.max().item()
    for step in range(batch_max_step):
        if truncate_flag.sum() == graph_num:
            break
        else:
            policy.eval_prompt(data, nodes_per_graph, step, truncate_flag)

        if compute_reward:
            rs = generate_reward(gnn, tasknet, data, torch.cat(policy.prompt_slices))
            for i in range(graph_num):
                if step < nodes_per_graph[i] and truncate_flag[i] == 0:
                    r_list[i].append(rs[i])
        
        if data_type != "train":
            total_prompt_step_num += (~truncate_flag.bool() & (step<nodes_per_graph).cpu()).sum()

    reward_list, pr_ratio_list = None, None
    if compute_reward:
        reward_list = [reward_reshape(torch.stack(r).cpu().numpy(), reward_clip=args.reward_clip) for r in r_list]
    if compute_prr:
        pr_ratio_list = [torch.any(p!=0, dim=-1).sum().item()/p.size(0) for p in policy.prompt_slices]
    
    return torch.cat(policy.prompt_slices), reward_list, pr_ratio_list


def compute_auc(score, y, info="", first=False):
    if score.size(0) == 0:
        return -1
     
    auc_list = []
    # for each task
    for i in range(y.shape[1]):
        is_valid = y[:, i]**2 > 0
        label = ((y[is_valid, i] + 1) / 2).long()
        # AUC is only defined when there is at least one positive data.
        if torch.sum(label==0) > 0 and torch.sum(label==1) > 0:
            auc_list.append(roc_auc_score(label, score[is_valid, i]))
        else:
            # print("\t({}) targets of task {} are all {}!".format(info, i, 0 if torch.sum(label==0) == 0 else 1))
            pass
    
    if info == "Total_AUC" and len(auc_list) < y.shape[1] and first:
        print("\tSome targets are missing! Missing ratio:{:.6f}".format(1 - float(len(auc_list))/y.shape[1]))

    return sum(auc_list)/len(auc_list) * 100 if len(auc_list)>0 else -1


def tasknet_loss(gnn: PromptedGNN, tasknet: GNNBasedNet, data, prompt, require_grad=True, keep_loss_dim=False, policy_gnn_update=False, basic_prompt=None):
    if policy_gnn_update:
        gnn.train()
    
    with torch.no_grad():
        if basic_prompt is not None:
            node_emb = gnn(data.x, data.edge_index, data.edge_attr, prompt, basic_prompt)
        else:
            node_emb = gnn(data.x, data.edge_index, data.edge_attr, prompt)
    if require_grad:
        logit = tasknet(node_emb, data.batch)  # (batchsize, task_num)
    else:
        with torch.no_grad():
            logit = tasknet(node_emb, data.batch)
    # y = data.y.view(logit.shape).type(torch.float64)
    y = data.y.view(logit.shape).type(data.x.dtype)
    is_valid = y**2 > 0
    if torch.sum(is_valid) == 0:
        return None

    task_loss = criterion(logit, (y+1)/2)  # {-1,1} -> {0,1}
    task_loss = torch.where(is_valid, task_loss, torch.zeros(task_loss.shape).to(task_loss.dtype).to(task_loss.device))
    if keep_loss_dim:
        task_loss = torch.sum(task_loss, dim=-1) / torch.sum(is_valid, dim=-1)
    else:
        task_loss = torch.sum(task_loss) / torch.sum(is_valid)

    if policy_gnn_update:
        gnn.eval()

    return task_loss, logit.detach().cpu(), y.cpu()


def generate_reward(gnn: PromptedGNN, tasknet: GNNBasedNet, data, prompt=None, basic_prompt=None, weight_converge=0.0):
    r"""
    根据promptedGNN和tasknet的预测结果，生成强化学习的奖励值
    """
    with torch.no_grad():
        # pretrained gnn特征提取
        if basic_prompt is not None:
            node_emb = gnn(data.x, data.edge_index, data.edge_attr, prompt, basic_prompt)
        else:
            node_emb = gnn(data.x, data.edge_index, data.edge_attr, prompt)
        # 任务预测
        logit = tasknet(node_emb, data.batch)
    # 对齐真实标签和logits
    y = data.y.view(logit.shape).type(node_emb.dtype)
    # 过滤无效标签
    is_valid = y**2 > 0
    if torch.sum(is_valid) == 0:
        return None
    # 计算奖励 BCEWithLogitsLoss
    """
    TODO: 奖励可以考虑节点覆盖率 
    """
    reward = criterion(logit, (y+1)/2)
    reward = torch.sum(reward, dim=-1) / torch.sum(is_valid, dim=-1)
    # 覆盖率奖励
    if prompt is not None:
        non_zero_mask = (torch.abs(prompt).sum(dim=1) > 1e-6)  # 避免浮点误差
        num_non_zero = non_zero_mask.sum().item()
        total_nodes = prompt.size(0)
        pcr = num_non_zero / total_nodes
        converge_reward = weight_converge * pcr
    else:
        converge_reward = 0.0
    total_reward = reward - converge_reward
    return total_reward
    

def train_policy(args, epoch, gnn: PromptedGNN, tasknet: GNNBasedNet, tasknet_optim, policy: H_PPO, policy_optims: List[optim.Adam],
                 ens_loaders, train_loader, val_loader, test_loader, aucs, best_infos, device, basic_prompt=None):
    r"""
        联合训练策略网络（强化学习）和任务网络（监督学习）
        实现策略优化与下游任务的协同训练
        policy: H_PPO
        tasknet: 基于GNN的多任务分类网络
        gnn: pretrained gnn
    """
    old_train_auc, old_val_auc, old_test_auc = aucs
    if best_infos is None:
        best_epoch, best_train_auc, best_val_auc, best_test_auc = (-1, -1, -1, -1)
    else:
        best_epoch, best_train_auc, best_val_auc, best_test_auc = best_infos
    
    # ====== Annealing the learning rate ====== #
    # 线性衰减策略 学习率退火
    descend_decay_frac = 1.0 - (epoch - 1) / (args.total_epochs)
    policy_decay_frac, tasknet_decay_frac = 1.0, 1.0

    if args.policy_decay == "down":
        policy_decay_frac = descend_decay_frac
    for i in range(args.ensemble_num):
        # 策略网络退火
        for param_group in policy_optims[i].param_groups:
            if param_group["name"] == "actor_d":
                param_group["lr"] = policy_decay_frac * args.actor_d_lr
            elif param_group["name"] == "actor_c":
                param_group["lr"] = policy_decay_frac * args.actor_c_lr
            elif param_group["name"] == "critic":
                param_group["lr"] = policy_decay_frac * args.critic_lr

    if args.tasknet_decay == "down":
        tasknet_decay_frac = descend_decay_frac
    # 任务网络退火
    for param_group in tasknet_optim.param_groups:
        param_group["lr"] = tasknet_decay_frac * args.tasknet_lr

    # decreasing the entropy of discrete actors for more stable descrete actions sampling
    # 探索策略调整，减少离散动作的随机性，缩小连续动作的探索范围
    # 1. 降低离散动作熵权重 （鼓励利用已知策略）
    policy.coeff_ent_d = args.coeff_entropy_d * descend_decay_frac

    # decreasing the log_std for more stable continuous actions sampling
    # 2. 降低连续动作标准差 （逐步稳定动作采样）
    for i in range(args.ensemble_num):
        new_log_std = args.init_log_std + (-5 - args.init_log_std) * (epoch - 1) / args.total_epochs
        policy.actor_cs[i].log_std = (torch.ones(args.emb_dim) * new_log_std).to(device)

    if not args.sh_mode:
        print("\n===== EPOCH {}/{} ===== policy lr*{:.2f} tasknet lr*{:.2f}".format(
              epoch, args.total_epochs, policy_decay_frac, tasknet_decay_frac))


    # ====== Agent sampling to collect transitions ====== #
    tasknet.eval(); policy.train_or_eval(mode='train')
    # 初始化奖励归一器
    reward_transform = Normalization(shape=1)
    
    # train each ensemble sub-actors (sub-policies)
    for a_idx in range(args.ensemble_num):
        gnn.eval(); basic_prompt.eval()

        batch_data_list = []
        for batch_idx, batch_data in enumerate(ens_loaders[a_idx]):
            batch_data_list.extend(batch_data.to_data_list())
            # guarantee enough amount of graphs or it's the last batch of data
            if len(batch_data_list) < args.batch_size and batch_idx < len(ens_loaders[a_idx])-1:
                continue
            batch_data = Batch.from_data_list(batch_data_list).to(device)
            batch_data_list = []
            # 统计每个图的节点数
            nodes_per_graph = scatter(torch.ones_like(batch_data.batch), batch_data.batch, reduce='add')
            graph_num = batch_data.num_graphs
            batch_max_step = nodes_per_graph.max().item()
            # 初始化状态state
            state = torch.zeros([graph_num, batch_max_step, policy.max_num_nodes, args.emb_dim]).to(device)
            # 采样得到的动作 初始化
            action_d = torch.zeros([graph_num, batch_max_step]).long().to(device)
            action_c = torch.zeros([graph_num, batch_max_step, args.emb_dim]).to(device)
            # 动作的对数概率 初始化
            logprob_d = torch.zeros([graph_num, batch_max_step]).to(device)
            logprob_c = torch.zeros([graph_num, batch_max_step]).to(device)
            # 初始化奖励
            reward = torch.zeros([graph_num, batch_max_step+1]).to(device)
            with torch.no_grad():
                init_r = generate_reward(gnn, tasknet, batch_data, basic_prompt=basic_prompt, weight_converge=args.weight_converge)
            # t = 0 时刻reward
            reward[:, 0] = init_r
            done = torch.zeros([graph_num, batch_max_step]).to(device)
            # mask invalid step for each graph
            # 过滤无效时间步，例如填充部分
            valid_mask = (torch.arange(batch_max_step).expand(graph_num,-1).to(device)) < nodes_per_graph.unsqueeze(1)  # (num_graphs, batch_max_step)

            # collect a batch of episodes
            # 初始化提示片段
            policy.prompt_slices = [torch.zeros(nodes, args.emb_dim).to(device) for nodes in nodes_per_graph]
            # 初始化状态
            policy.step_state = torch.zeros(graph_num, policy.max_num_nodes, args.emb_dim).to(device)
            # mask nodes corresponding to `zero padding`
            base_mask = torch.zeros(graph_num, policy.max_num_nodes).to(device)
            base_idx = torch.arange(0, policy.max_num_nodes).to(device)
            policy.batch_mask = torch.where(base_idx < nodes_per_graph.unsqueeze(-1), base_mask, torch.ones_like(base_mask))
            for step in range(batch_max_step):
                # 标记完成状态
                done[nodes_per_graph == step + 1, step] = 1.0
                # attach a new feature prompt on a specific node for each graph
                # 生成动作并更新状态
                s, a_d, a_c, lp_d, lp_c = policy.train_prompt(a_idx, batch_data, nodes_per_graph, step)
                state[:, step] = s
                action_d[:, step] = a_d
                action_c[:, step] = a_c
                logprob_d[:, step] = lp_d
                logprob_c[:, step] = lp_c
                # execute downstream task with prompted graph to recieve reward
                # 计算奖励
                r = generate_reward(gnn, tasknet, batch_data, prompt=torch.cat(policy.prompt_slices), basic_prompt=basic_prompt, weight_converge=args.weight_converge)
                reward[:, step+1] = r

            # mask invalid step and compute scaled rewards, advantages, returns
            # 计算下一个状态
            next_state = torch.zeros_like(state)
            for i in range(graph_num):
                if nodes_per_graph[i] > 1:
                    # 提取有效状态
                    valid_s = state[i, :nodes_per_graph[i]]
                    # 填充下一个状态的前N-1步
                    next_state[i, :nodes_per_graph[i]-1] = valid_s[1:]
                    # 处理最后一个时间步
                    next_state[i, nodes_per_graph[i]-1] = valid_s[-1]
                else:
                    # 只有一个时间步，则保持状态
                    next_state[i, 0] = state[i, 0]
            # 计算优势函数和回报
            reward = reshape_reward(reward, nodes_per_graph)
            # 利用critic计算advantage和return
            state = state[valid_mask]; next_state = next_state[valid_mask]; done = done[valid_mask]
            advantage, approx_return, approx_return0, scaled_reward = \
                compute_adv_ret(args, policy.critic, state, reward, next_state, done, nodes_per_graph.tolist(), reward_transform)
            action_d = action_d[valid_mask]; action_c = action_c[valid_mask]
            logprob_d = logprob_d[valid_mask]; logprob_c = logprob_c[valid_mask]

            # 策略网络更新
            """
                如果有效数据量足够，则将经验数据传入train_policy进行梯度更新
            """
            if state.size(0) > args.minibatch_size:
                experience = (state, action_d, logprob_d, action_c, logprob_c, advantage, approx_return)
                policy.train_policy(args, a_idx, experience, nodes_per_graph, policy_optims[a_idx], approx_return0, scaled_reward, batch_idx+1, len(ens_loaders[a_idx]))
            else:
                if not args.sh_mode:
                    print("No enough trainsitions!")
            
    
    # ====== Update tasknet according to joint policy ====== #
    gnn.eval(); policy.train_or_eval('eval'); tasknet.train()
    for task_epoch in range(1, args.tasknet_epochs+1):
        epoch_loss = 0
        for batch_data in train_loader:
            batch_data = batch_data.to(device)
            # 当前策略生成的prompt
            prompt, _, _ = attach_prompt(args, policy, batch_data, "train", gnn, tasknet, compute_reward=0, compute_prr=0)

            if args.tasknet_train_mode:
                pr_loss, _, _ = tasknet_loss(gnn, tasknet, batch_data, prompt, policy_gnn_update=True, basic_prompt=basic_prompt)
            else:
                pr_loss, _, _ = tasknet_loss(gnn, tasknet, batch_data, prompt, basic_prompt=basic_prompt)
            # 模型优化
            tasknet_optim.zero_grad()
            pr_loss.backward()
            tasknet_optim.step()
            epoch_loss += pr_loss.detach().item()
        if not args.sh_mode:
            print("[{}/{}] Prompted Tasknet Loss with joint policy: {:.5f}".format(
                task_epoch, args.tasknet_epochs, epoch_loss/len(train_loader)))


    # ====== Evaluation ====== #
    if epoch % args.check_freq == 0 or epoch == 1 or epoch == args.total_epochs:

        if epoch < args.skip_epoch and epoch != 1:
            # warm-up
            train_auc, val_auc, test_auc = old_train_auc, old_val_auc, old_test_auc
        else:
            # eval
            train_auc, train_loss = evaluate_policy(args, gnn, tasknet, policy, train_loader, "train", device, basic_prompt=basic_prompt)
            val_auc, val_loss = evaluate_policy(args, gnn, tasknet, policy, val_loader, "val", device, basic_prompt=basic_prompt)
            test_auc, test_info = evaluate_policy(args, gnn, tasknet, policy, test_loader, "test", device, basic_prompt=basic_prompt)
            print("==== Epoch{:02d} AUC {:.2f} {:.2f} {:.2f} | Loss {:.5f} {:.5f} {:.5f} | "
                    "TEST P_min {:.3f} P_max {:.3f} P_norm {:.6f} P_Ratio {:.3f}".format(
                epoch,
                train_auc, val_auc, test_auc, train_loss, val_loss, test_info[0],
                test_info[1], test_info[2], test_info[3], test_info[4]
            ))
            # update results
            if val_auc > best_val_auc and epoch >= 20:
                best_epoch = epoch; best_train_auc = train_auc; best_val_auc = val_auc; best_test_auc = test_auc
            
    else:
        train_auc, val_auc, test_auc = old_train_auc, old_val_auc, old_test_auc

    return policy, tasknet, (train_auc, val_auc, test_auc), (best_epoch, best_train_auc, best_val_auc, best_test_auc)


def evaluate_policy(args, gnn: PromptedGNN, tasknet: GNNBasedNet, policy: H_PPO, loader, data_type, device, basic_prompt=None):
    gnn.eval(); tasknet.eval(); policy.train_or_eval(mode='eval')

    pr_ys, pr_logits, pr_losses, pr_ratio_list = [], [], [], []
    p_min, p_max, p_norm = float("inf"), -float("inf"), []

    for batch_data in loader:
        batch_data = batch_data.to(device)

        if data_type == "test":
            prompt, _, pr_ratio = attach_prompt(args, policy, batch_data, data_type, gnn, tasknet, compute_reward=0, compute_prr=1)
            pr_ratio_list.extend(pr_ratio)
        else:
            prompt, _, _, = attach_prompt(args, policy, batch_data, data_type, gnn, tasknet, compute_reward=0, compute_prr=0)

        pr_loss, pr_logit, pr_y = tasknet_loss(gnn, tasknet, batch_data, prompt, require_grad=False, keep_loss_dim=True, basic_prompt=basic_prompt)
        pr_losses.append(pr_loss.cpu()); pr_logits.append(pr_logit); pr_ys.append(pr_y)
        if data_type == "test":
            valid_prompt = prompt[prompt.any(dim=1)!=0]
            if valid_prompt.min() < p_min:
                p_min = valid_prompt.min()
            if valid_prompt.max() > p_max:
                p_max = valid_prompt.max()
            p_norm.extend(torch.mean(torch.abs(valid_prompt), dim=-1).tolist())

    ys = torch.cat(pr_ys); logits = torch.cat(pr_logits); losses = torch.cat(pr_losses).mean().item()
    auc = compute_auc(logits, ys, "Total_AUC")

    if data_type == "test":
        test_info = [losses, p_min, p_max, sum(p_norm)/len(p_norm), sum(pr_ratio_list)/len(pr_ratio_list)]
        return auc, test_info
    else:
        return auc, losses
