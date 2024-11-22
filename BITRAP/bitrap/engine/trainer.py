import os
import re
import copy
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from bitrap.utils.visualization import Visualizer
from bitrap.utils.box_utils import cxcywh_to_x1y1x2y2
from bitrap.utils.dataset_utils import restore
from bitrap.modeling.gmm2d import GMM2D
from bitrap.modeling.gmm4d import GMM4D
from .evaluate import evaluate_multimodal, compute_kde_nll
from .utils import print_info, viz_results, post_process

from tqdm import tqdm
import pickle as pkl
import pdb

def generate_filenames(initial_string, obs_length=15, pred_length=45):
    # Generates filenames given the fact that the 'initial_string' is the last frame fo the observation sequence.
    # Find the number in the string using regular expression
    match = re.search(r'(\d+)(\.png)$', initial_string)
    if not match:
        raise ValueError("The initial string must contain a number followed by .png")
    
    base = initial_string[:match.start(1)]
    number = int(match.group(1))
    extension = match.group(2)
    
    filenames = []
    
    # Generate 14 frames before the initial frame (counting backward)
    for i in range(obs_length - 1, 0, -1):
        filenames.append(f"{base}{number - i}{extension}")
    
    # Add the initial frame
    filenames.append(f"{base}{number}{extension}")
    
    # Generate 45 frames after the initial frame (counting forward)
    for i in range(1, pred_length + 1):
        filenames.append(f"{base}{number + i}{extension}")
    
    return filenames

def generate_vis_results(index,all_img_paths,all_X_globals,all_pred_trajs,all_gt_trajs,eval_results):
    """
    Generate the vis results given an image
    """
    # print("File names")
    # print(generate_filenames(all_img_paths[index]))
    print("Generating best BITRAP prediction")
    np.save("BITRAP_best.npy",postprocess_boxes(all_pred_trajs[index][:,eval_results['best_indices'][index],:]))
    # W,H = Image.open(all_img_paths[1]).size
    # print(np.array2string(postprocess_boxes(all_pred_trajs[index][:,eval_results['best_indices'][index],:]),separator=","))
    print("Generating worst BITRAP prediction")
    np.save("BITRAP_worst.npy",postprocess_boxes(all_pred_trajs[index][:,eval_results['worst_indices'][index],:]))
    # print(np.array2string(postprocess_boxes(all_pred_trajs[index][:,eval_results['worst_indices'][index],:]),separator=","))
    # print("ground truth sequence")
    # gt_45 = all_gt_trajs[index]
    # ob_15 = all_X_globals[index]
    # final_gt = np.concatenate((ob_15,gt_45),axis=0)
    # print(np.array2string(postprocess_boxes(final_gt),separator=","))

def do_train(cfg, epoch, model, optimizer, dataloader, device, logger=None, lr_scheduler=None):
    model.train()
    max_iters = len(dataloader)
    if cfg.DATASET.NAME in ['eth', 'hotel', 'univ', 'zara1', 'zara2']:
        viz = Visualizer(mode='plot')
    else:
        viz = Visualizer(mode='image')


    with torch.set_grad_enabled(True):
        for iters, batch in enumerate(tqdm(dataloader), start=1):
            X_global = batch['input_x'].to(device)
            y_global = batch['target_y'].to(device)
            img_path = batch['cur_image_file']
            # resolution = batch['pred_resolution'].numpy()

            # For ETH_UCY dataset only
            if cfg.DATASET.NAME in ['eth', 'hotel', 'univ', 'zara1', 'zara2']:
                input_x = batch['input_x_st'].to(device)
                neighbors_st = restore(batch['neighbors_x_st'])
                adjacency = restore(batch['neighbors_adjacency'])
                first_history_indices = batch['first_history_index']
            else:
                input_x = X_global
                neighbors_st, adjacency, first_history_indices = None, None, None

            pred_goal, pred_traj, loss_dict, dist_goal, dist_traj = model(input_x, 
                                                                    y_global, 
                                                                    neighbors_st=neighbors_st, 
                                                                    adjacency=adjacency, 
                                                                    cur_pos=X_global[:, -1, :cfg.MODEL.DEC_OUTPUT_DIM],
                                                                    first_history_indices=first_history_indices)
            if cfg.MODEL.LATENT_DIST == 'categorical':
                loss = loss_dict['loss_goal'] + \
                       loss_dict['loss_traj'] + \
                       model.param_scheduler.kld_weight * loss_dict['loss_kld'] - \
                       1. * loss_dict['mutual_info_p']
            else:
                loss = loss_dict['loss_goal'] + \
                       loss_dict['loss_traj'] + \
                       model.param_scheduler.kld_weight * loss_dict['loss_kld']
            model.param_scheduler.step()
            loss_dict = {k:v.item() for k, v in loss_dict.items()}
            loss_dict['lr'] = optimizer.param_groups[0]['lr']
            # optimize
            optimizer.zero_grad() # avoid gradient accumulate from loss.backward()
            loss.backward()
            
            # loss_dict['grad_norm'] = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
            optimizer.step()

            if cfg.SOLVER.scheduler == 'exp':
                lr_scheduler.step()
            if iters % cfg.PRINT_INTERVAL == 0:
                print_info(epoch, model, optimizer, loss_dict, logger)

            if cfg.VISUALIZE and iters % max(int(len(dataloader)/5), 1) == 0:
                ret = post_process(cfg, X_global, y_global, pred_traj, pred_goal=pred_goal, dist_goal=dist_goal)
                X_global, y_global, pred_goal, pred_traj, dist_traj, dist_goal = ret
                viz_results(viz, X_global, y_global, pred_traj, img_path, dist_goal, dist_traj,
                            bbox_type=cfg.DATASET.BBOX_TYPE, normalized=False, logger=logger, name='pred_train')
                
def do_val(cfg, epoch, model, dataloader, device, logger=None):
    model.eval()
    loss_goal_val = 0.0
    loss_traj_val = 0.0
    loss_KLD_val = 0.0
    with torch.set_grad_enabled(False):
        for iters, batch in enumerate(tqdm(dataloader), start=1):
            X_global = batch['input_x'].to(device)
            y_global = batch['target_y'].to(device)
            img_path = batch['cur_image_file']
            # For ETH_UCY dataset only
            if cfg.DATASET.NAME in ['eth', 'hotel', 'univ', 'zara1', 'zara2']:
                input_x = batch['input_x_st'].to(device)
                neighbors_st = restore(batch['neighbors_x_st'])
                adjacency = restore(batch['neighbors_adjacency'])
                first_history_indices = batch['first_history_index']
            else:
                input_x = X_global
                neighbors_st, adjacency, first_history_indices = None, None, None
            
            pred_goal, pred_traj, loss_dict, _, _ = model(input_x, 
                                                            y_global, 
                                                            neighbors_st=neighbors_st,
                                                            adjacency=adjacency,
                                                            cur_pos=X_global[:, -1, :cfg.MODEL.DEC_OUTPUT_DIM],
                                                            first_history_indices=first_history_indices)

            # compute loss
            loss = loss_dict['loss_goal'] + loss_dict['loss_traj'] + loss_dict['loss_kld']
            loss_goal_val += loss_dict['loss_goal'].item()
            loss_traj_val += loss_dict['loss_traj'].item()
            loss_KLD_val += loss_dict['loss_kld'].item()
    loss_goal_val /= (iters + 1)
    loss_traj_val /= (iters + 1)
    loss_KLD_val /= (iters + 1)
    loss_val = loss_goal_val + loss_traj_val + loss_KLD_val
    
    info = "loss_val:{:.4f}, \
            loss_goal_val:{:.4f}, \
            loss_traj_val:{:.4f}, \
            loss_kld_val:{:.4f}".format(loss_val, loss_goal_val, loss_traj_val, loss_KLD_val)
        
    if hasattr(logger, 'log_values'):
        logger.info(info)
        logger.log_values({'loss_val':loss_val, 
                           'loss_goal_val':loss_goal_val,
                           'loss_traj_val':loss_traj_val, 
                           'loss_kld_val':loss_KLD_val})#, step=epoch)
    else:
        print(info)
    return loss_val


# def cxcywh_to_x1y1x2y2(bboxes):
#     """
#     Convert bounding boxes from (cx, cy, w, h) format to (x1, y1, x2, y2) format.
    
#     Args:
#     bboxes (numpy.ndarray): Array of shape either (N, 4) where each row is [cx, cy, w, h]. i think the below method accounts for
#     more though. check it out later.
    
#     Returns:
#     numpy.ndarray: Array of shape (N, 4) where each row is [x1, y1, x2, y2].
#     """
#     bboxes = copy.deepcopy(bboxes)
#     bboxes[..., [0,1]] = bboxes[..., [0, 1]] - bboxes[..., [2, 3]]/2
#     bboxes[..., [2,3]] = bboxes[..., [0,1]] + bboxes[..., [2, 3]]
#     return bboxes

# def cxcywh_to_x1y1x2y2(bboxes):
    
#     cx, cy, w, h = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    
#     x1 = cx - w / 2
#     y1 = cy - h / 2
#     x2 = cx + w / 2
#     y2 = cy + h / 2
    
#     return np.stack([x1, y1, x2, y2], axis=1)

def postprocess_boxes(bboxes):
    bboxes = cxcywh_to_x1y1x2y2(bboxes)
    return bboxes

def inference(cfg, epoch, model, dataloader, device, logger=None, eval_kde_nll=False, test_mode=False): # reemoved video_num as first arg
    model.eval()
    all_img_paths = []
    all_X_globals = []
    all_pred_goals = []
    all_gt_goals = []
    all_pred_trajs = []
    all_gt_trajs = []
    all_distributions = []
    all_timesteps = []
    if cfg.DATASET.NAME in ['eth', 'hotel', 'univ', 'zara1', 'zara2']:
        viz = Visualizer(mode='plot')
    else:
        viz = Visualizer(mode='image')
    
    with torch.set_grad_enabled(False):
        for iters, batch in enumerate(tqdm(dataloader), start=1):
            X_global = batch['input_x'].to(device)
            y_global = batch['target_y']
            img_path = batch['cur_image_file']
            # resolution = batch['pred_resolution'].numpy()
            
            # For ETH_UCY dataset only
            if cfg.DATASET.NAME in ['eth', 'hotel', 'univ', 'zara1', 'zara2']:
                input_x = batch['input_x_st'].to(device)
                neighbors_st = restore(batch['neighbors_x_st'])
                adjacency = restore(batch['neighbors_adjacency'])
                first_history_indices = batch['first_history_index']
            else:
                input_x = X_global
                neighbors_st, adjacency, first_history_indices = None, None, None

            pred_goal, pred_traj, _, dist_goal, dist_traj = model(input_x, 
                                                                neighbors_st=neighbors_st,
                                                                adjacency=adjacency,
                                                                z_mode=False, 
                                                                cur_pos=X_global[:, -1, :cfg.MODEL.DEC_OUTPUT_DIM],
                                                                first_history_indices=first_history_indices)
            # transfer back to global coordinates
            ret = post_process(cfg, X_global, y_global, pred_traj, pred_goal=pred_goal, dist_traj=dist_traj, dist_goal=dist_goal)
            X_global, y_global, pred_goal, pred_traj, dist_traj, dist_goal = ret
            all_img_paths.extend(img_path)
            all_X_globals.append(X_global)
            all_pred_goals.append(pred_goal)
            all_pred_trajs.append(pred_traj)
            all_gt_goals.append(y_global[:, -1])
            all_gt_trajs.append(y_global)
            all_timesteps.append(batch['timestep'].numpy())
            # if iters == 1:
            #     print(img_path)
            if dist_traj is not None:
                all_distributions.append(dist_traj)
            else:
                all_distributions.append(dist_goal)
            if cfg.VISUALIZE and iters % max(int(len(dataloader)/5), 1) == 0:
                # print("Entering the visualization phase") # entered
                if iters == 1:
                    viz_results(viz, X_global, y_global, pred_traj, img_path, dist_goal, dist_traj, 
                                bbox_type=cfg.DATASET.BBOX_TYPE, normalized=False, logger=logger, name='pred_test')
                # commenting out the vis part for now. you can uncomment it if you want to save the visualizations.
                    pass
        
        # Evaluate
        all_X_globals = np.concatenate(all_X_globals, axis=0)
        all_pred_goals = np.concatenate(all_pred_goals, axis=0)
        all_pred_trajs = np.concatenate(all_pred_trajs, axis=0)
        all_gt_goals = np.concatenate(all_gt_goals, axis=0)
        all_gt_trajs = np.concatenate(all_gt_trajs, axis=0)
        all_timesteps = np.concatenate(all_timesteps, axis=0)
        if hasattr(all_distributions[0], 'mus'):
            distribution = model.GMM(torch.cat([d.input_log_pis for d in all_distributions], axis=0),
                                    torch.cat([d.mus for d in all_distributions], axis=0),
                                    torch.cat([d.log_sigmas for d in all_distributions], axis=0),
                                    torch.cat([d.corrs for d in all_distributions], axis=0))
        else:
            distribution = None 
        # eval_pred_results = evaluate(all_pred_goals, all_gt_goals)
        mode = 'bbox' if all_gt_trajs.shape[-1] == 4 else 'point'
        eval_results = evaluate_multimodal(all_pred_trajs, all_gt_trajs, mode=mode, distribution=distribution, bbox_type=cfg.DATASET.BBOX_TYPE)
        for key, value in eval_results.items():
            info = "Testing prediction {}:{}".format(key, str(np.around(value, decimals=3)))
            if hasattr(logger, 'log_values'):
                logger.info(info)
            else:
                print(info)
        
        if hasattr(logger, 'log_values'):
            logger.log_values(eval_results)

        if test_mode:
            # save inputs, redictions and targets for test mode
            outputs = {'img_path': all_img_paths, 'X_global': all_X_globals, 'timestep': all_timesteps,
                       'pred_trajs': all_pred_trajs, 'gt_trajs':all_gt_trajs,'distributions':distribution}
            # print("HOPE IS A GOOD THING")
            # print(len(eval_results['best_indices'])) # 6223, and not 6211, since min_track_size is now 60 and not 61.
            # print(len(eval_results['worst_indices']))
            # print("THE BEST OF THINGS")
        # len(all_img_paths) is 6211 - each element is the last frame of the observation sequence.
        # all_pred_trajs has the same length and all_pred_trajs[0].shape is (45,20,4)
        # all_gt_trajs[0].shape is (45,4). That's why it's tiled when evaluating multimodal results above in evaluate_multimodal()
        # all_X_globals[0].shape is (15,4) and seems to be the observation sequence.
        # all_timesteps is array([17011, 17018, 17025, ..., 13988, 13995, 14002]), first image of each




            # generate_vis_results(video_num,all_img_paths,all_X_globals,all_pred_trajs,all_gt_trajs,eval_results) 




            # print("Best index",eval_results['best_indices'][10])
            # print("Worst index",eval_results['worst_indices'][10])
            # if not os.path.exists(cfg.OUT_DIR):
            #     os.makedirs(cfg.OUT_DIR)
            # output_file = os.path.join(cfg.OUT_DIR, '{}_{}.pkl'.format(cfg.MODEL.LATENT_DIST, cfg.DATASET.NAME))
            # print("Writing outputs to: ", output_file)
            # pkl.dump(outputs, open(output_file,'wb'))

    # Mevaluate KDE NLL, since we sample 2000, need to use a smaller batchsize
    # Commenting out the below at the time of plotting qualitative results
    # if eval_kde_nll:
    #     dataloader_params ={
    #         "batch_size": cfg.TEST.KDE_BATCH_SIZE,
    #         "shuffle": False,
    #         "num_workers": cfg.DATALOADER.NUM_WORKERS,
    #         "collate_fn": dataloader.collate_fn,
    #         }
    #     kde_nll_dataloader = DataLoader(dataloader.dataset, **dataloader_params)
    #     inference_kde_nll(cfg, epoch, model, kde_nll_dataloader, device, logger)

def inference_kde_nll(cfg, epoch, model, dataloader, device, logger=None):
    model.eval()
    all_pred_goals = []
    all_gt_goals = []
    all_pred_trajs = []
    all_gt_trajs = []
    all_kde_nll = []
    all_per_step_kde_nll = []
    num_samples = model.K
    model.K = 2000
    with torch.set_grad_enabled(False):
        for iters, batch in enumerate(tqdm(dataloader), start=1):
            X_global = batch['input_x'].to(device)
            y_global = batch['target_y']
            img_path = batch['cur_image_file']
            resolution = batch['pred_resolution'].numpy()
            # For ETH_UCY dataset only
            if cfg.DATASET.NAME in ['eth', 'hotel', 'univ', 'zara1', 'zara2']:
                input_x = batch['input_x_st'].to(device)
                neighbors_st = restore(batch['neighbors_x_st'])
                adjacency = restore(batch['neighbors_adjacency'])
                first_history_indices = batch['first_history_index']
            else:
                input_x = X_global
                neighbors_st, adjacency, first_history_indices = None, None, None
            
            pred_goal, pred_traj, _, _, _ = model(input_x, 
                                                    neighbors_st=neighbors_st,
                                                    adjacency=adjacency,
                                                    z_mode=False, 
                                                    cur_pos=X_global[:, -1, :cfg.MODEL.DEC_OUTPUT_DIM],
                                                    first_history_indices=first_history_indices)
            
            # transfer back to global coordinates
            ret = post_process(cfg, X_global, y_global, pred_traj, pred_goal=pred_goal, dist_traj=None, dist_goal=None)
            X_global, y_global, pred_goal, pred_traj, dist_traj, dist_goal = ret
            for i in range(len(pred_traj)):
                KDE_NLL, KDE_NLL_PER_STEP = compute_kde_nll(pred_traj[i:i+1], y_global[i:i+1])
                all_kde_nll.append(KDE_NLL)
                all_per_step_kde_nll.append(KDE_NLL_PER_STEP)
        KDE_NLL = np.array(all_kde_nll).mean()
        KDE_NLL_PER_STEP = np.stack(all_per_step_kde_nll, axis=0).mean(axis=0)
        # Evaluate
        Goal_NLL = KDE_NLL_PER_STEP[-1]
        nll_dict = {'KDE_NLL': KDE_NLL} if cfg.MODEL.LATENT_DIST == 'categorical' else {'KDE_NLL': KDE_NLL, 'Goal_NLL': Goal_NLL}
        info = "Testing prediction KDE_NLL:{:.4f}, per step NLL:{}".format(KDE_NLL, KDE_NLL_PER_STEP)
        if hasattr(logger, 'log_values'):
            logger.info(info)
        else:
            print(info)
        if hasattr(logger, 'log_values'):
            logger.log_values(nll_dict)

    # reset model.K back to 20
    model.K = num_samples
    return KDE_NLL