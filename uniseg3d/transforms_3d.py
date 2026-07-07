import numpy as np
import ast
import scipy
import torch
import random
from torch_scatter import scatter_mean
from mmcv.transforms import BaseTransform
from mmdet3d.registry import TRANSFORMS

import MinkowskiEngine as ME


@TRANSFORMS.register_module()
class TextPromptGeneration(BaseTransform):
    def __init__(self, num_ins = 3, random_select=False, seq_length=126, embedding_dim=300):
        self.num_ins = num_ins
        self.seq_length = seq_length
        self.embedding_dim = embedding_dim
        self.random_select = random_select
    
    def transform(self, input_dict):
        if len(input_dict['text_info'])==0:
            input_dict['label_text'] = np.array([])
            input_dict['gt_text_prompt']  = np.array([])
            input_dict['text_object_id'] = np.array([])
            input_dict['text_token'] = np.array([])
            
            return input_dict
        else:
            text_infos = input_dict['text_info']
            text_infos_id = list(text_infos)
            if self.random_select:
                num_ins = np.random.randint(len(text_infos_id), size=1)[0] + 1
            else:
                num_ins = len(text_infos_id)
                
            text_token = []
            label_text = np.zeros((num_ins,))
            text_object_id = np.zeros((num_ins,))
            select_texts_infos_id = random.sample(text_infos_id, num_ins)
            gt_sp_masks = torch.as_tensor(input_dict['gt_sp_masks'])
            gt_text_prompt = torch.zeros((num_ins, gt_sp_masks.shape[1]))
            pts_instance_objextId = input_dict['pts_instance_objextId']
            for i, select_texts_info_id in enumerate(select_texts_infos_id):
                id = random.sample(range(len(text_infos[select_texts_info_id]['text'])), 1)[0]
                token = text_infos[select_texts_info_id]['text_token'][id]
                text_token.append(token)
                label_id = text_infos[select_texts_info_id]['label_id']
                label_text[i] = label_id
                text_object_id[i] = eval(select_texts_info_id)
                object_id = torch.tensor(eval(select_texts_info_id))
                id = torch.where(pts_instance_objextId == object_id)[0]
                assert len(id)>0
                gt_text_prompt[i] = gt_sp_masks[id[0]]
                
            input_dict['label_text'] = label_text
            input_dict['gt_text_prompt']  = gt_text_prompt
            input_dict['text_object_id'] = text_object_id
            input_dict['text_token'] = np.array(text_token)
            
            return input_dict





# @TRANSFORMS.register_module()
# class QA_Generation(BaseTransform):
#     def __init__(self, num_qa = 10, random_select=False, task_level=None, task_type=None):
#         self.num_qa = num_qa
#         self.random_select = random_select
#         self.task_level = task_level #['part_qa_data', 'object_qa_data', 'multi_turn_qa_data']
#         self.task_type = task_type 

#         # part_pure_qa_list = ['functional_part_grounding', 'functional_object_grounding', 'scale_comparison_size', 'distance_estimation', 'relative_position_forward_reasoning', 'relative_depth_forward', 'existence_verification', 'scale_estimation']
#         # part_qa_seg_list = ['functional_part_grounding', 'functional_object_grounding', 'scale_comparison_size', 'distance_estimation', 'relative_position_forward_reasoning', 'relative_depth_forward', 'grounded_dimension_reasoning']
    
#     def flatten_qa_data(self, qa_data):
#         qa_list = []
#         for qa_level in qa_data.keys():
#             if qa_level != 'multi_turn_qa_data':
#                 for i_key in qa_data[qa_level].keys():
#                     for i_task_type in qa_data[qa_level][i_key].keys():
#                         qa_list.extend(qa_data[qa_level][i_key][i_task_type])   
#             else:
#                 qa_list.extend(qa_data[qa_level])
#         return qa_list
    

#     def transform(self, input_dict):
#         flat_qa_list = self.flatten_qa_data(input_dict['qa_data'])

#         if len(flat_qa_list) < self.num_qa:
#             select_qa_list = flat_qa_list
#         else:
#             select_qa_list = random.sample(flat_qa_list, self.num_qa)

#         # here we need to align the object_id and the part_id

#         gt_sp_part_masks = torch.as_tensor(input_dict['gt_sp_part_masks'])
#         gt_sp_object_masks = torch.as_tensor(input_dict['gt_sp_object_masks'])
        
#         # compute num_superpoints: use part_masks first, fall back to object_masks if empty, get from sp_pts_mask if both empty
#         if gt_sp_part_masks.shape[0] > 0:
#             num_superpoints = gt_sp_part_masks.shape[1]
#         elif gt_sp_object_masks.shape[0] > 0:
#             num_superpoints = gt_sp_object_masks.shape[1]
#         else:
#             # if both masks are empty, get superpoint count from sp_pts_mask
#             sp_pts_mask = torch.as_tensor(input_dict['sp_pts_mask'])
#             num_superpoints = len(torch.unique(sp_pts_mask))


#         def count_p_tags(text):
#             """Count <p>...</p> tag occurrences in text"""
#             if not isinstance(text, str):
#                 return 0
#             # count <p> tags only (skip </p> since they are paired)
#             count = text.count('<p>')
#             if count>2:
#                 count = 2
#             return count

#         def get_seg_from_single_qa(single_qa, num_p_tags):
#             """Get the corresponding mask from a single QA dict

#             Args:
#                 single_qa: QA dict
#                 num_p_tags: number of <p>...</p> tags in the answer of this QA (counted from text)

#             Note: the returned seg has shape (num_p_tags, num_superpoints),
#             where num_p_tags is determined by the number of <p>...</p> tags actually present in the text.
#             Each <p>...</p> tag corresponds to a query token, so each row of mask corresponds to a query token.

#             Important: if num_p_tags == 0, return a (1, num_superpoints) NaN mask,
#             to match the behavior in query_decoder.py (when no <p>...</p> tag is found, a (1, hidden_dim) NaN token is returned).
#             """
#             # if num_p_tags == 0, directly return a (1, num_superpoints) NaN mask
#             # this corresponds to the case in query_decoder.py where no <p>...</p> tag is found
#             if num_p_tags == 0:
#                 return torch.full((1, num_superpoints), float('nan'), dtype=torch.float32)
            
#             target_ids = []
#             mask_tensors = []
#             for i_key in single_qa.keys():
#                 if i_key == 'object_id':
#                     target_ids.append(single_qa['object_id'])
#                     mask_tensors.append(gt_sp_object_masks)
#                 elif i_key == 'part_id':
#                     target_ids.append(single_qa['part_id'])
#                     mask_tensors.append(gt_sp_part_masks)


#             if len(target_ids) == 0:
#                 return torch.full((num_p_tags, num_superpoints), float('nan'), dtype=torch.float32)
            

#             segs = []
#             for target_id, mask_tensor in zip(target_ids, mask_tensors):
#                 # check whether mask_tensor is empty
#                 if mask_tensor.shape[0] == 0:
#                     # if mask_tensor is empty, return NaN mask
#                     if isinstance(target_id, (list, tuple, torch.Tensor, np.ndarray)):
#                         seg = torch.full((len(target_id), num_superpoints), float('nan'), dtype=torch.float32)
#                     else:
#                         seg = torch.full((1, num_superpoints), float('nan'), dtype=torch.float32)
#                     segs.append(seg)
#                     continue
                
#                 # check whether the index is within valid range
#                 max_valid_index = mask_tensor.shape[0] - 1
#                 if isinstance(target_id, (list, tuple, torch.Tensor, np.ndarray)):
#                     target_id_list = list(target_id) if not isinstance(target_id, torch.Tensor) else target_id.tolist()
                    
#                     # Check for invalid types (str) or out of bound indices
#                     is_invalid = False
#                     for idx in target_id_list:
#                         if not isinstance(idx, (int, float, np.integer, np.floating)):
#                             is_invalid = True
#                             break
#                         if idx < 0 or idx > max_valid_index:
#                             is_invalid = True
#                             break
                            
#                     if is_invalid:
#                         seg = torch.full((len(target_id), num_superpoints), float('nan'), dtype=torch.float32)
#                         segs.append(seg)
#                         continue
#                 else:
#                     if not isinstance(target_id, (int, float, np.integer, np.floating)) or target_id < 0 or target_id > max_valid_index:
#                         seg = torch.full((1, num_superpoints), float('nan'), dtype=torch.float32)
#                         segs.append(seg)
#                         continue

#                 # get mask
#                 try:
#                     seg = mask_tensor[target_id]  # shape: (len(target_id), num_superpoints)
#                     if seg.dim() == 1:
#                         seg = seg.unsqueeze(0)
#                 except:
#                     if isinstance(target_id, (list, tuple, torch.Tensor, np.ndarray)):
#                         seg = torch.full((len(target_id), num_superpoints), float('nan'), dtype=torch.float32)
#                     else:
#                         seg = torch.full((1, num_superpoints), float('nan'), dtype=torch.float32)
                
#                 # if all masks are empty, fill with NaN
#                 if seg.sum() == 0:
#                     seg = torch.full((seg.shape[0], num_superpoints), float('nan'), dtype=torch.float32)
                
#                 segs.append(seg)
                
#             # adjust seg shape based on the number of <p>...</p> tags in the text
#             # if the count of target_id is inconsistent with num_p_tags, adjust
#             segs = torch.cat(segs, dim=0)
#             if segs.shape[0] != num_p_tags:
#                 # target_id count is less than tag count: fill with NaN
#                 # padding = torch.full((num_p_tags - len(target_id), num_superpoints), float('nan'), dtype=seg.dtype)
#                 # seg = torch.cat([seg, padding], dim=0)
#                 # print('segs.shape: ', segs.shape)
#                 # print('num_p_tags: ', num_p_tags)
#                 # print('target_ids: ', target_ids)
#                 # print('single_qa: ', single_qa)
#                 # print('-------------------------------------------')
#                 segs = torch.full((num_p_tags, num_superpoints), float('nan'), dtype=torch.float32)


            
#             return segs  # shape: (num_p_tags, num_superpoints)
        
#         sp_gt_seg = []
#         valid_qa_list = []  # store valid QA

#         for i, qa in enumerate(select_qa_list):
#             # determine whether qa is a single QA (dict) or multi-turn dialogue (list)
#             if isinstance(qa, dict):
#                 # single QA
#                 # count the number of <p>...</p> tags from the answer text
#                 answer_text = qa.get('answer', '')
#                 num_p_tags = count_p_tags(answer_text)

#                 # i_qa_seg has shape (num_p_tags, num_superpoints),
#                 # where num_p_tags is determined by the number of <p>...</p> tags actually present in the text
#                 i_qa_seg = get_seg_from_single_qa(qa, num_p_tags)
#                 sp_gt_seg.append(i_qa_seg)
#                 valid_qa_list.append(qa)
#             elif isinstance(qa, list):
#                 # multi-turn dialogue: get corresponding mask for each round
#                 # count total <p>...</p> tags across all rounds
#                 total_p_tags = 0
#                 i_qa_seg_list = []
#                 for round_qa in qa:
#                     # count the number of <p>...</p> tags from each round's answer text
#                     round_answer = round_qa.get('answer', '') if isinstance(round_qa, dict) else ''
#                     round_p_tags = count_p_tags(round_answer)
#                     total_p_tags += round_p_tags

#                     round_seg = get_seg_from_single_qa(round_qa, round_p_tags)
#                     i_qa_seg_list.append(round_seg)

#                 # concatenate seg across all rounds, shape is (total_p_tags, num_superpoints)
#                 i_qa_seg_list = torch.cat(i_qa_seg_list, dim=0)
#                 sp_gt_seg.append(i_qa_seg_list)
#                 valid_qa_list.append(qa)

#         # save processed mask, qa_data stays as-is (already in input_dict)
#         input_dict['sp_gt_seg'] = sp_gt_seg
#         input_dict['selected_qa_data'] = valid_qa_list
        
#         return input_dict


def get_seg_from_single_qa(single_qa, num_p_tags, num_superpoints, gt_sp_object_masks, gt_sp_part_masks, input_dict):
    if num_p_tags == 0:
        return torch.full((1, num_superpoints), float('nan'), dtype=torch.float32)
    
    # target_ids = []
    # mask_tensors = torch.full((0, num_superpoints), float('nan'), dtype=torch.float32)

    qa_type = single_qa['qa_keys']

    target_ids_list = []
    mask_tensors_list = []
    reference_id_list = []
    if qa_type == 'object':
        target_ids_list.append(single_qa['object_id'])
        mask_tensors_list.append(gt_sp_object_masks)
        reference_id_list.append(input_dict['pts_object_objextId'])
    elif qa_type == 'part':
        target_ids_list.append(single_qa['object_id']) # here is not a bug, the orignial data only contain object_id. Only in multi_turn_qa_data, part_id is used.
        mask_tensors_list.append(gt_sp_part_masks)
        reference_id_list.append(input_dict['pts_part_objextId'])
    elif qa_type == 'multi_turn':
        for keys in single_qa.keys():
            if keys == 'object_id':
                target_ids_list.append(single_qa['object_id'])
                mask_tensors_list.append(gt_sp_object_masks)
                reference_id_list.append(input_dict['pts_object_objextId'])
            elif keys == 'part_id':
                target_ids_list.append(single_qa['part_id'])
                mask_tensors_list.append(gt_sp_part_masks)
                reference_id_list.append(input_dict['pts_part_objextId'])
        # if 'object_id' in single_qa.keys():
        #     target_ids = single_qa['object_id']
        #     mask_tensors = gt_sp_object_masks
        #     reference_id = input_dict['pts_object_objextId']
        # elif 'part_id' in single_qa.keys():
        #     target_ids = single_qa['part_id']
        #     mask_tensors = gt_sp_part_masks
        #     reference_id = input_dict['pts_part_objextId']
    else:
        return torch.full((num_p_tags, num_superpoints), float('nan'), dtype=torch.float32)

    
    segs = []
    # convert all elements in target_ids to int to avoid type mismatch causing 'in' comparison failure or subsequent indexing errors
    for target_ids, mask_tensors, reference_id in zip(target_ids_list, mask_tensors_list, reference_id_list):
        normalized_target_ids = []
        for tid in target_ids:
            if isinstance(tid, (int, np.integer)):
                normalized_target_ids.append(int(tid))
            elif isinstance(tid, (float, np.floating)):
                normalized_target_ids.append(int(tid))
            else:
                try:
                    normalized_target_ids.append(int(tid))
                except (ValueError, TypeError):
                    # keep original value if conversion fails; it will be marked invalid later
                    normalized_target_ids.append(tid)
        target_ids = normalized_target_ids
        
        # check whether target_ids are in reference_id
        is_in_reference_id = [target_id in reference_id for target_id in target_ids]

        # # if any id is not in reference, drop into pdb
        # if False in is_in_reference_id:
        #     print('-------------------------------------------')
        #     print('input_dict[scene_name]', input_dict['scene_name'])
        #     print('single_qa: ', single_qa)
        #     print('ids in qa json: ', target_ids)
        #     print('ids in the label file: ', reference_id)
        #     print('is_in_reference_id: ', is_in_reference_id)
        #     print('torch.unique(mask_tensors): ', torch.unique(mask_tensors))
        #     print('----------------------------------------------------------')
        #     import time
        #     time.sleep(1)


        if len(target_ids) == 0 or mask_tensors.shape[0] == 0:
            return torch.full((num_p_tags, num_superpoints), float('nan'), dtype=torch.float32)
            
        
        for i, target_id in enumerate(target_ids):

            valid_id = is_in_reference_id[i]
            if not valid_id:
                seg = torch.full((1, num_superpoints), float('nan'), dtype=torch.float32)
                segs.append(seg)
                continue
            
            max_valid_index = mask_tensors.shape[0] - 1

            if not isinstance(target_id, (int, float, np.integer, np.floating)):
                # print('target_id: ', target_id)
                # print('max_valid_index: ', max_valid_index)
                # print('single_qa: ', single_qa)
                # print('----------------------------------------------------------')
                try:
                    target_id = int(target_id)
                except:
                    print('target_id: ', target_id)
                    print('max_valid_index: ', max_valid_index)
                    print('single_qa: ', single_qa)
                    print('----------------------------------------------------------')

                    seg = torch.full((1, num_superpoints), float('nan'), dtype=torch.float32)
                    segs.append(seg)
                    continue
                
            if target_id>max_valid_index:
                seg = torch.full((1, num_superpoints), float('nan'), dtype=torch.float32)
            else:
                seg = mask_tensors[target_id]  # shape: (len(target_id), num_superpoints)
                if seg.dim() == 1:
                    seg = seg.unsqueeze(0)
            
            # some empty masks are already handled here
            if seg.sum() == 0:
                seg = torch.full((seg.shape[0], num_superpoints), float('nan'), dtype=torch.float32)
            segs.append(seg)
        
    # adjust seg shape based on the number of <p>...</p> tags in the text
    # if the count of target_id is inconsistent with num_p_tags, adjust
    segs = torch.cat(segs, dim=0)
    if segs.shape[0] != num_p_tags:
        # target_id count is less than tag count: fill with NaN
        # padding = torch.full((num_p_tags - len(target_id), num_superpoints), float('nan'), dtype=seg.dtype)
        # seg = torch.cat([seg, padding], dim=0)
        print('segs.shape: ', segs.shape)
        print('num_p_tags: ', num_p_tags)
        print('target_ids: ', target_ids)
        print('single_qa: ', single_qa)
        print('-------------------------------------------')
        segs = torch.full((num_p_tags, num_superpoints), float('nan'), dtype=torch.float32)
    return segs  # shape: (num_p_tags, num_superpoints)

def count_p_tags(text):
    """Count <p>...</p> tag occurrences in text"""
    if not isinstance(text, str):
        return 0
    # count <p> tags only (skip </p> since they are paired)
    count = text.count('<p>')
    if count>2:
        count = 2
    return count

@TRANSFORMS.register_module()
class QA_Generation_v2(BaseTransform):
    def __init__(self, num_qa = 10, random_select=False, task_level=None, task_type=None):
        self.num_qa = num_qa
        self.random_select = random_select
        self.task_level = task_level #['part_qa_data', 'object_qa_data', 'multi_turn_qa_data']
        self.task_type = task_type 

        # part_pure_qa_list = ['functional_part_grounding', 'functional_object_grounding', 'scale_comparison_size', 'distance_estimation', 'relative_position_forward_reasoning', 'relative_depth_forward', 'existence_verification', 'scale_estimation']
        # part_qa_seg_list = ['functional_part_grounding', 'functional_object_grounding', 'scale_comparison_size', 'distance_estimation', 'relative_position_forward_reasoning', 'relative_depth_forward', 'grounded_dimension_reasoning']
    
    def flatten_qa_data(self, qa_data):
        qa_list = []
        for qa_level in qa_data.keys():
            if qa_data[qa_level] is None:
                continue
            if qa_level != 'multi_turn_qa_data':
                if qa_level == 'part_qa_data':
                    qa_keys = 'part'
                elif qa_level == 'object_qa_data':
                    qa_keys = 'object'
                else:
                    continue
                for i_key in qa_data[qa_level].keys():
                    if qa_data[qa_level][i_key] is None:
                        continue
                    for i_task_type in qa_data[qa_level][i_key].keys():
                        i_qa=qa_data[qa_level][i_key][i_task_type].copy()
                        if 'question' in i_qa[0] and 'answer' in i_qa[0]:
                            for ii_qa in i_qa:
                                ii_qa['qa_keys'] = qa_keys
                                ii_qa['qa_type'] = qa_level + '_' + i_key + '_' + i_task_type
                            qa_list.extend(i_qa) 
                        else:
                            continue  
            else:
                qa_keys = 'multi_turn'
                i_qa=qa_data[qa_level].copy()
                # print('len(i_qa): ', len(i_qa))
                # print('len(i_qa)[0]: ', len(i_qa[0]))
                # print('----------------------------------------------------------')
                # import time
                # time.sleep(1)
                for i_qa_round in i_qa:
                    valid_qas = []
                    for ii_qa in i_qa_round:
                        if 'question' in ii_qa and 'answer' in ii_qa:
                            ii_qa['qa_keys'] = qa_keys  
                            ii_qa['qa_type'] = qa_level
                            valid_qas.append(ii_qa)
                    # skip rounds where LLM output parsing failed (e.g. entries containing only raw_model_output)
                    if len(valid_qas) == 0:
                        continue
                    # print('valid_qas: ', valid_qas)
                    # print('-------------------------------------------')
                    qa_list.extend(valid_qas)
        return qa_list
    

    def transform(self, input_dict):
        flat_qa_list = self.flatten_qa_data(input_dict['qa_data'])

        if len(flat_qa_list) < self.num_qa:
            select_qa_list = flat_qa_list
        else:
            select_qa_list = random.sample(flat_qa_list, self.num_qa)


        # here we need to align the object_id and the part_id

        gt_sp_part_masks = torch.as_tensor(input_dict['gt_sp_part_masks'])
        gt_sp_object_masks = torch.as_tensor(input_dict['gt_sp_object_masks'])
        
        # compute num_superpoints: use part_masks first, fall back to object_masks if empty, get from sp_pts_mask if both empty
        if gt_sp_part_masks.shape[0] > 0:
            num_superpoints = gt_sp_part_masks.shape[1]
        elif gt_sp_object_masks.shape[0] > 0:
            num_superpoints = gt_sp_object_masks.shape[1]
        else:
            # if both masks are empty, get superpoint count from sp_pts_mask
            sp_pts_mask = torch.as_tensor(input_dict['sp_pts_mask'])
            num_superpoints = len(torch.unique(sp_pts_mask))
        
        sp_gt_seg = []
        valid_qa_list = []  # store valid QA

        for i, qa in enumerate(select_qa_list):
            # determine whether qa is a single QA (dict) or multi-turn dialogue (list)
            if isinstance(qa, dict):
                # single QA
                # count the number of <p>...</p> tags from the answer text
                answer_text = qa.get('answer', '')
                num_p_tags = count_p_tags(answer_text)
                
                # i_qa_seg has shape (num_p_tags, num_superpoints),
                # where num_p_tags is determined by the number of <p>...</p> tags actually present in the text
                i_qa_seg = get_seg_from_single_qa(qa, num_p_tags, num_superpoints, gt_sp_object_masks, gt_sp_part_masks, input_dict)
                sp_gt_seg.append(i_qa_seg)
                valid_qa_list.append(qa)
            elif isinstance(qa, list):
                # multi-turn dialogue: get corresponding mask for each round
                # count total <p>...</p> tags across all rounds
                total_p_tags = 0
                i_qa_seg_list = []
                for round_qa in qa:
                    # count the number of <p>...</p> tags from each round's answer text
                    round_answer = round_qa.get('answer', '') if isinstance(round_qa, dict) else ''
                    round_p_tags = count_p_tags(round_answer)
                    total_p_tags += round_p_tags
                    
                    round_seg = get_seg_from_single_qa(round_qa, round_p_tags, num_superpoints, gt_sp_object_masks, gt_sp_part_masks, input_dict)

                    print('round_qa: ', round_qa, '\nround_seg.shape: ', round_seg.shape)
                    print('-------------------------------------------')
                    import time; time.sleep(1)
                    i_qa_seg_list.append(round_seg)
                
                # concatenate seg across all rounds, shape is (total_p_tags, num_superpoints)
                i_qa_seg_list = torch.cat(i_qa_seg_list, dim=0)
                sp_gt_seg.append(i_qa_seg_list)
                valid_qa_list.append(qa)
                

        # save processed mask, qa_data stays as-is (already in input_dict)
        input_dict['sp_gt_seg'] = sp_gt_seg
        input_dict['selected_qa_data'] = valid_qa_list
        
        return input_dict





@TRANSFORMS.register_module()
class QA_Generation_Test(BaseTransform):
    def __init__(self, num_qa = 10, random_select=False, task_level=None, task_type=None, sub_task=None):
        self.num_qa = num_qa
        self.random_select = random_select
        self.task_level = task_level #['part_qa_data', 'object_qa_data', 'multi_turn_qa_data']
        self.task_type = task_type #['qa_data', 'qa_data_seg']
        self.sub_task = sub_task 
        # part_pure_qa_list = ['functional_part_grounding', 'functional_object_grounding', 'scale_comparison_size', 'distance_estimation', 'relative_position_forward_reasoning', 'relative_depth_forward', 'existence_verification', 'scale_estimation']
        # part_qa_seg_list = ['functional_part_grounding', 'functional_object_grounding', 'scale_comparison_size', 'distance_estimation', 'relative_position_forward_reasoning', 'relative_depth_forward', 'grounded_dimension_reasoning']
    
    def flatten_qa_data(self, qa_data):
        qa_list = []

        # determine which task_level to iterate over
        # if self.task_level is not None, only look up that level; if it doesn't exist, target_levels is empty
        if self.task_level is not None:
            if self.task_level in qa_data:
                target_levels = [self.task_level]
            else:
                target_levels = []
        else:
            # if self.task_level is None, iterate over all levels
            target_levels = list(qa_data.keys())

        for qa_level in target_levels:
            if qa_level == 'multi_turn_qa_data':
                # Each item is a CONVERSATION (list of turn dicts). Tag the turns and keep the
                # conversation as a list so the eval decoder dispatches it to the multi-turn path
                # (single-turn items are dicts; conversations are lists).
                for conv in qa_data[qa_level]:
                    if not isinstance(conv, list):
                        continue
                    for turn in conv:
                        turn['qa_keys'] = 'multi_turn'
                        turn['qa_type'] = qa_level
                    qa_list.append(conv)
                continue
            elif qa_level == 'part_qa_data':
                    qa_keys = 'part'
            elif qa_level == 'object_qa_data':
                qa_keys = 'object'

            # determine which i_key to iterate over (corresponds to task_type parameter)
            current_keys = qa_data[qa_level].keys()
            if self.task_type is not None:
                # if task_type is specified, filter strictly
                if isinstance(self.task_type, list):
                    target_keys = [k for k in self.task_type if k in current_keys]
                elif self.task_type in current_keys:
                    target_keys = [self.task_type]
                else:
                    target_keys = []
            else:
                # use all if not specified
                target_keys = list(current_keys)

            for i_key in target_keys:
                # determine which i_task_type to iterate over (corresponds to sub_task parameter)
                current_sub_tasks = qa_data[qa_level][i_key].keys()
                if self.sub_task is not None:
                    # if sub_task is specified, filter strictly
                    if isinstance(self.sub_task, list):
                        target_sub_tasks = [t for t in self.sub_task if t in current_sub_tasks]
                    elif self.sub_task in current_sub_tasks:
                        target_sub_tasks = [self.sub_task]
                    else:
                        target_sub_tasks = []
                else:
                    # use all if not specified
                    target_sub_tasks = list(current_sub_tasks)
                for i_task_type in target_sub_tasks:
                    i_qa=qa_data[qa_level][i_key][i_task_type].copy()
                    for ii_qa in i_qa:
                        ii_qa['qa_keys'] = qa_keys
                        ii_qa['qa_key'] = i_key
                        ii_qa['qa_task_type'] = i_task_type
                        ii_qa['qa_level'] = qa_level
                        ii_qa['qa_type'] = qa_level + '_' + i_key + '_' + i_task_type
                    qa_list.extend(i_qa)   
        return qa_list
    

    def transform(self, input_dict):

        flat_qa_list = self.flatten_qa_data(input_dict['qa_data'])

        select_qa_list = flat_qa_list

        # here we need to align the object_id and the part_id
        gt_sp_part_masks = torch.as_tensor(input_dict['gt_sp_part_masks'])
        gt_sp_object_masks = torch.as_tensor(input_dict['gt_sp_object_masks'])
        

        sp_pts_mask = torch.as_tensor(input_dict['sp_pts_mask'])
        num_superpoints = len(torch.unique(sp_pts_mask))
        
        
        
        sp_gt_seg = []
        valid_qa_list = []  # store valid QA

        for i, qa in enumerate(select_qa_list):
            # determine whether qa is a single QA (dict) or multi-turn dialogue (list)
            if isinstance(qa, dict):
                # single QA
                # count the number of <p>...</p> tags from the answer text
                answer_text = qa.get('answer', '')
                num_p_tags = count_p_tags(answer_text)
                
                # i_qa_seg has shape (num_p_tags, num_superpoints),
                # where num_p_tags is determined by the number of <p>...</p> tags actually present in the text
                i_qa_seg = get_seg_from_single_qa(qa, num_p_tags, num_superpoints, gt_sp_object_masks, gt_sp_part_masks, input_dict)
                sp_gt_seg.append(i_qa_seg)
                valid_qa_list.append(qa)
            elif isinstance(qa, list):
                # multi-turn dialogue: get corresponding mask for each round
                # count total <p>...</p> tags across all rounds
                total_p_tags = 0
                i_qa_seg_list = []
                for round_qa in qa:
                    # count the number of <p>...</p> tags from each round's answer text
                    round_answer = round_qa.get('answer', '') if isinstance(round_qa, dict) else ''
                    round_p_tags = count_p_tags(round_answer)
                    total_p_tags += round_p_tags
                    
                    round_seg = get_seg_from_single_qa(round_qa, round_p_tags, num_superpoints, gt_sp_object_masks, gt_sp_part_masks, input_dict)
                    # one GT mask PER ROUND, appended separately so it aligns 1:1 with the per-turn
                    # predictions the decoder produces (do NOT concatenate rounds into one entry).
                    sp_gt_seg.append(round_seg)
                valid_qa_list.append(qa)
                

        # save processed mask, qa_data stays as-is (already in input_dict)
        input_dict['sp_gt_seg'] = sp_gt_seg
        input_dict['selected_qa_data'] = valid_qa_list

        return input_dict




@TRANSFORMS.register_module()
class AddSuperPointAnnotations_Reason3D(BaseTransform):
    """Prepare ground truth markup for training.
    
    Required Keys:
    - pts_semantic_mask (np.float32)
    
    Added Keys:
    - gt_sp_masks (np.int64)
    
    Args:
        num_classes (int): Number of classes.
    """
    
    def __init__(self,
                 num_classes,
                 stuff_classes,
                 merge_non_stuff_cls=True):
        self.num_classes = num_classes
        self.stuff_classes = stuff_classes
        self.merge_non_stuff_cls = merge_non_stuff_cls

 
    def transform(self, input_dict):
        """Private function for preparation ground truth 
        markup for training.
        
        Args:
            input_dict (dict): Result dict from loading pipeline.
        
        Returns:
            dict: results, 'gt_sp_masks' is added.
        """
        # create class mapping
        # because pts_instance_mask contains instances from non-instaces classes

        pts_instance_mask = torch.tensor(input_dict['pts_instance_mask'])
        pts_semantic_mask = torch.tensor(input_dict['pts_semantic_mask'])
        
        pts_instance_mask[pts_semantic_mask == self.num_classes] = -1
        for stuff_cls in self.stuff_classes:
            pts_instance_mask[pts_semantic_mask == stuff_cls] = -1
        
        idxs = torch.unique(pts_instance_mask)
        # assert idxs[0] == -1

        mapping = torch.zeros(torch.max(idxs) + 2, dtype=torch.long)
        new_idxs = torch.arange(len(idxs), device=idxs.device)
        mapping[idxs] = new_idxs - 1
        pts_instance_mask = mapping[pts_instance_mask]
        input_dict['pts_instance_mask'] = pts_instance_mask.numpy()


        # create gt instance markup     
        insts_mask = pts_instance_mask.clone()
        
        if torch.sum(insts_mask == -1) != 0:
            insts_mask[insts_mask == -1] = torch.max(insts_mask) + 1
            insts_mask = torch.nn.functional.one_hot(insts_mask)[:, :-1]
        else:
            insts_mask = torch.nn.functional.one_hot(insts_mask)

        if insts_mask.shape[1] != 0:
            insts_mask = insts_mask.T
            sp_pts_mask = torch.tensor(input_dict['sp_pts_mask'])
            sp_masks_inst = scatter_mean(
                insts_mask.float(), sp_pts_mask, dim=-1)
            sp_masks_inst = sp_masks_inst > 0.5
        else:
            sp_masks_inst = insts_mask.new_zeros(
                (0, input_dict['sp_pts_mask'].max() + 1), dtype=torch.bool)

        num_stuff_cls = len(self.stuff_classes)
        insts = new_idxs[1:] - 1
        # length of gt_labels_3d must equal the row count of gt_sp_masks (i.e. sp_masks_inst)
        gt_labels = insts.new_zeros(len(insts))

        for i, inst in enumerate(insts):
            index = pts_semantic_mask[pts_instance_mask == inst][0]
            gt_labels[i] = index - num_stuff_cls
        
        input_dict['gt_labels_3d'] = gt_labels.numpy()

        # create gt semantic markup
        sem_mask = torch.tensor(input_dict['pts_semantic_mask'])
        # print(sem_mask.shape)
        # print(torch.unique(sem_mask))
        # print('--------------------------------')


        if torch.sum(sem_mask < 0) != 0:
            sem_mask[sem_mask < 0] = torch.max(sem_mask) + 1
            sem_mask = torch.nn.functional.one_hot(sem_mask)[:, :-1]
        else:
            sem_mask = torch.nn.functional.one_hot(sem_mask)


       
        sem_mask = sem_mask.T
        sp_pts_mask = torch.tensor(input_dict['sp_pts_mask'])
        sp_masks_seg = scatter_mean(sem_mask.float(), sp_pts_mask, dim=-1)
        sp_masks_seg = sp_masks_seg > 0.5

        sp_masks_seg[-1, sp_masks_seg.sum(axis=0) == 0] = True

        assert sp_masks_seg.sum(axis=0).max().item()
        
        if self.merge_non_stuff_cls:
            sp_masks_seg = torch.vstack((
                sp_masks_seg[:num_stuff_cls, :], 
                sp_masks_seg[num_stuff_cls:, :].sum(axis=0).unsqueeze(0)))

        
        input_dict['gt_sp_masks_ins'] = sp_masks_inst
        input_dict['gt_sp_masks_sem'] = sp_masks_seg

        # sp_masks_all = torch.vstack((sp_masks_inst))
        sp_masks_all = sp_masks_inst

        input_dict['gt_sp_masks'] = sp_masks_all.numpy()

        # create eval markup
        if 'eval_ann_info' in input_dict.keys(): 
            pts_instance_mask[pts_instance_mask != -1] += num_stuff_cls
            for idx, stuff_cls in enumerate(self.stuff_classes):
                pts_instance_mask[pts_semantic_mask == stuff_cls] = idx

            input_dict['eval_ann_info']['pts_instance_mask'] = \
                pts_instance_mask.numpy()

        return input_dict



import json
@TRANSFORMS.register_module()
class QA_Generation_Reason3D_Test(BaseTransform):
    def __init__(self, num_qa = 10, reason3d_file=None):
        self.num_qa = num_qa
        self.reason3d_data = json.load(open(reason3d_file, 'r'))

    

    def transform(self, input_dict):

        scene_id = input_dict['scene_name']

        flat_qa_list = input_dict['qa_data']


        # here we need to align the object_id and the part_id
        gt_sp_sem_masks = torch.as_tensor(input_dict['gt_sp_masks_sem'])

        sp_pts_mask = torch.as_tensor(input_dict['sp_pts_mask'])
        num_superpoints = len(torch.unique(sp_pts_mask))
        
        
        sp_gt_seg = []
        valid_qa_list = []  # store valid QA

        for i, qa in enumerate(flat_qa_list):
            # determine whether qa is a single QA (dict) or multi-turn dialogue (list)
            qa['question'] = qa['question']+'<SEG>'+'<name>'
            qa['answer'] = '<p>'+qa['object_name']+'</p><SEG>'
            qa['qa_keys'] = 'reason3d'
            qa['qa_level'] = 'reason3d'
            qa['qa_key'] = 'reason3d'
            qa['qa_task_type'] = 'reason3d'
            qa['scene_name'] = scene_id
            object_id = int(qa['object_id'])

            i_qa_seg = gt_sp_sem_masks[object_id].unsqueeze(0)
            if i_qa_seg.sum() == 0:
                print(f'object_id: {object_id} is not in the scene {scene_id}')
                i_qa_seg = torch.full((i_qa_seg.shape[0], num_superpoints), float('nan'), dtype=torch.float32)
            
            sp_gt_seg.append(i_qa_seg)
            valid_qa_list.append(qa)

        # save processed mask, qa_data stays as-is (already in input_dict)
        input_dict['sp_gt_seg'] = sp_gt_seg
        input_dict['selected_qa_data'] = valid_qa_list
        return input_dict



@TRANSFORMS.register_module()
class TextPromptTest(BaseTransform):
    def __init__(self, num_ins = 3, seq_length=126, embedding_dim=300):
        self.num_ins = num_ins
        self.seq_length = seq_length
        self.embedding_dim = embedding_dim

    def transform(self, input_dict):
        if len(input_dict['text_info'])==0:
            input_dict['label_text'] = np.array([])
            input_dict['gt_text_prompt']  = np.array([])
            input_dict['text_object_id'] = np.array([])
            input_dict['text_token'] = np.array([])
            
            return input_dict
        else:
            text_infos = input_dict['text_info']
            text_infos_id = list(text_infos)

            text_token = []
            label_text = []
            text_object_id = []
            select_texts_infos_id = text_infos_id
            sp_pts_mask = torch.as_tensor(input_dict['sp_pts_mask'])
            gt_text_prompt = []
            for i, select_texts_info_id in enumerate(select_texts_infos_id):
                for id in range(len(text_infos[select_texts_info_id]['text'])):
                    token = text_infos[select_texts_info_id]['text_token'][id]
                    text_token.append(token)
                    label_id = text_infos[select_texts_info_id]['label_id']
                    pts_id = text_infos[select_texts_info_id]['pts_id']
                    label_text.append(label_id)
                    text_object_id.append(eval(select_texts_info_id))
                    gt = torch.zeros(sp_pts_mask.shape[0])
                    gt[pts_id] = 1.
                    gt_text_prompt.append(gt)
                    
            gt_text_prompt = torch.stack(gt_text_prompt, dim=0)
            gt_text_prompt = scatter_mean(gt_text_prompt, sp_pts_mask, dim=-1)
            gt_text_prompt = gt_text_prompt > 0.5
                    
            input_dict['label_text'] = np.array(label_text)
            input_dict['gt_text_prompt'] = gt_text_prompt
            input_dict['text_object_id'] = np.array(text_object_id)
            input_dict['text_token'] = np.array(text_token)
            
            return input_dict


@TRANSFORMS.register_module()
class PointPromptGeneration(BaseTransform):
    def __init__(self, samplePoint=False):
        self.samplePoint = samplePoint

    def transform(self, input_dict):
        sp_pts_mask = np.array(input_dict['sp_pts_mask'])
        
        pts_instance_mask = np.array(input_dict['pts_instance_mask']).copy()
        pts_instance_id = np.unique(pts_instance_mask)
        if 'eval_ann_info' in input_dict.keys():
            pts_instance_mask = pts_instance_mask-2
            pts_instance_id = pts_instance_id.copy()
            pts_instance_id = pts_instance_id-2
            pts_instance_id = np.delete(pts_instance_id, np.where(pts_instance_id<0))
        else:
            pts_instance_id = np.delete(pts_instance_id, np.where(pts_instance_id==-1))
        
        point_prompts = np.array([])
        select_ids = np.array([])

        if len(pts_instance_id) > 0:
            select_ids = pts_instance_id.copy()
            np.random.shuffle(select_ids)   
            for select_id in select_ids:
                pts_select_id = np.where(pts_instance_mask == select_id)[0]
                if self.samplePoint:
                    instance_points = input_dict['points'][pts_select_id][:,:3]
                    if len(instance_points)>500:
                        sample_ratio = 500.0/len(instance_points)
                    else:
                        sample_ratio = 1.0

                    sample_size = int(len(instance_points) * sample_ratio)

                    sample_indices = np.random.choice(len(instance_points), sample_size, replace=False)
                    sampled_points = instance_points[sample_indices]
                    centroid = np.mean(sampled_points, axis=0)
                    distances = np.linalg.norm(sampled_points - centroid, axis=1)
                    distance_order = np.argsort(distances) 
                    ordered_pts_select_id = pts_select_id[sample_indices][distance_order[0]]
                    point_prompts = np.append(point_prompts, ordered_pts_select_id)
                else:
                    random_pts_select_id = np.random.choice(pts_select_id, size=1, replace=False)
                    point_prompts = np.append(point_prompts, random_pts_select_id)
        else:
            pass

        input_dict['pts_instance_objextId_shuffle'] = input_dict['pts_instance_objextId'][select_ids]    
        input_dict['point_prompts'] = point_prompts
        input_dict['point_prompt_instance_ids'] = select_ids
        input_dict['point_prompt_sp_ids'] = sp_pts_mask[list(point_prompts.astype(int))]

        return input_dict

@TRANSFORMS.register_module()
class PointPromptTest(BaseTransform):
    def __init__(self, mode='agile', max_num_point=1, samplePoint=False, is_distance=False, size_file=None):
        self.mode=mode # 'agile'
        self.max_num_point = max_num_point
        self.is_distance = is_distance
        self.samplePoint = samplePoint
        if size_file is not None:
            import pickle
            with open(size_file, 'rb') as file:
                self.sizes = pickle.load(file)
    
    def transform(self, input_dict):
        raw_coords = input_dict['points'].coord.contiguous().clone()
        coords_qv, unique_map, inverse_map = ME.utils.sparse_quantize(
                                                coordinates=raw_coords,
                                                quantization_size=0.05,
                                                return_index=True,
                                                return_inverse=True)
        sp_pts_mask = np.array(input_dict['sp_pts_mask'])
        
        pts_instance_mask = np.array(input_dict['pts_instance_mask']).copy()
        pts_instance_id = np.unique(pts_instance_mask)
        if 'eval_ann_info' in input_dict.keys():
            pts_instance_mask = pts_instance_mask-2
            pts_instance_id = pts_instance_id.copy()
            pts_instance_id = pts_instance_id-2
            pts_instance_id = np.delete(pts_instance_id, np.where(pts_instance_id<0))
        else:
            pts_instance_id = np.delete(pts_instance_id, np.where(pts_instance_id==-1))

        point_prompts = np.array([])
        select_ids = np.array([])
        point_prompt_distance_norms = np.array([])
        select_ids_filter = np.array([])

        if len(pts_instance_id) > 0:
            if self.mode=='agile':
                select_ids = pts_instance_id.copy()
                for select_id in select_ids:
                    gt_inst = (pts_instance_mask == select_id).astype(np.int32)
                    me_coord = raw_coords[unique_map]
                    me_gt_inst = gt_inst[unique_map]
                    valid_index = torch.where(torch.as_tensor(me_gt_inst))[0]
                    zero_indices = (me_gt_inst == 0)  # background
                    one_indices = (me_gt_inst == 1)  # foreground
                    if one_indices.sum() == 0:
                        continue
                    pairwise_distances = torch.cdist(me_coord[zero_indices, :], me_coord[one_indices, :])
                    pairwise_distances, _ = torch.min(pairwise_distances, dim=0)
                    me_index = valid_index[torch.argmax(pairwise_distances)]
                    global_index = unique_map[me_index].numpy().astype(np.int32)
                    assert gt_inst[global_index] == 1
                    point_prompts = np.append(point_prompts, global_index)
                    select_ids_filter = np.append(select_ids_filter, select_id)
        else:
            pass
        if len(select_ids_filter)==0:
            pass
        elif len(select_ids_filter)==1:
            pass
        input_dict['pts_instance_objextId_shuffle'] = input_dict['pts_instance_objextId'][select_ids_filter.astype(int)]    
        input_dict['point_prompts'] = point_prompts
        input_dict['point_prompt_instance_ids'] = select_ids_filter.astype(int)
        input_dict['point_prompt_sp_ids'] = sp_pts_mask[list(point_prompts.astype(int))]
        if self.is_distance:
            input_dict['point_prompt_distance_norms'] = point_prompt_distance_norms

        return input_dict

@TRANSFORMS.register_module()
class ElasticTransfrom(BaseTransform):
    """Apply elastic augmentation to a 3D scene. Required Keys:

    Args:
        gran (List[float]): Size of the noise grid (in same scale[m/cm]
            as the voxel grid).
        mag (List[float]): Noise multiplier.
        voxel_size (float): Voxel size.
        p (float): probability of applying this transform.
    """

    def __init__(self, gran, mag, voxel_size, p=1.0):
        self.gran = gran
        self.mag = mag
        self.voxel_size = voxel_size
        self.p = p

    def transform(self, input_dict):
        """Private function-wrapper for elastic transform.

        Args:
            input_dict (dict): Result dict from loading pipeline.
        
        Returns:
            dict: Results after elastic, 'points' is updated
            in the result dict.
        """
        coords = input_dict['points'].tensor[:, :3].numpy() / self.voxel_size
        if np.random.rand() < self.p:
            coords = self.elastic(coords, self.gran[0], self.mag[0])
            coords = self.elastic(coords, self.gran[1], self.mag[1])
        input_dict['elastic_coords'] = coords
        return input_dict

    def elastic(self, x, gran, mag):
        """Private function for elastic transform to a points.

        Args:
            x (ndarray): Point cloud.
            gran (List[float]): Size of the noise grid (in same scale[m/cm]
                as the voxel grid).
            mag: (List[float]): Noise multiplier.
        
        Returns:
            dict: Results after elastic, 'points' is updated
                in the result dict.
        """
        blur0 = np.ones((3, 1, 1)).astype('float32') / 3
        blur1 = np.ones((1, 3, 1)).astype('float32') / 3
        blur2 = np.ones((1, 1, 3)).astype('float32') / 3

        noise_dim = np.abs(x).max(0).astype(np.int32) // gran + 3
        noise = [
            np.random.randn(noise_dim[0], noise_dim[1],
                            noise_dim[2]).astype('float32') for _ in range(3)
        ]

        for blur in [blur0, blur1, blur2, blur0, blur1, blur2]:
            noise = [
                scipy.ndimage.filters.convolve(
                    n, blur, mode='constant', cval=0) for n in noise
            ]

        ax = [
            np.linspace(-(b - 1) * gran, (b - 1) * gran, b) for b in noise_dim
        ]
        interp = [
            scipy.interpolate.RegularGridInterpolator(
                ax, n, bounds_error=0, fill_value=0) for n in noise
        ]

        return x + np.hstack([i(x)[:, None] for i in interp]) * mag


@TRANSFORMS.register_module()
class AddSuperPointAnnotations(BaseTransform):
    """Prepare ground truth markup for training.
    
    Required Keys:
    - pts_semantic_mask (np.float32)
    
    Added Keys:
    - gt_sp_masks (np.int64)
    
    Args:
        num_classes (int): Number of classes.
    """
    
    def __init__(self,
                 num_classes,
                 stuff_classes,
                 merge_non_stuff_cls=True,
                 merge_ov=False):
        self.num_classes = num_classes
        self.stuff_classes = stuff_classes
        self.merge_non_stuff_cls = merge_non_stuff_cls
        self.merge_ov = merge_ov
 
    def transform(self, input_dict):
        """Private function for preparation ground truth 
        markup for training.
        
        Args:
            input_dict (dict): Result dict from loading pipeline.
        
        Returns:
            dict: results, 'gt_sp_masks' is added.
        """
        # create class mapping
        # because pts_instance_mask contains instances from non-instaces classes
        pts_instance_mask = torch.tensor(input_dict['pts_instance_mask'])
        pts_semantic_mask = torch.tensor(input_dict['pts_semantic_mask'])
        
        pts_instance_mask[pts_semantic_mask == self.num_classes] = -1
        for stuff_cls in self.stuff_classes:
            pts_instance_mask[pts_semantic_mask == stuff_cls] = -1
        
        idxs = torch.unique(pts_instance_mask)
        assert idxs[0] == -1
        input_dict['pts_instance_objextId'] = idxs[1:] - 1
        
        mapping = torch.zeros(torch.max(idxs) + 2, dtype=torch.long)
        new_idxs = torch.arange(len(idxs), device=idxs.device)
        mapping[idxs] = new_idxs - 1
        pts_instance_mask = mapping[pts_instance_mask]
        input_dict['pts_instance_mask'] = pts_instance_mask.numpy()

        # create gt instance markup     
        insts_mask = pts_instance_mask.clone()
        
        if torch.sum(insts_mask == -1) != 0:
            insts_mask[insts_mask == -1] = torch.max(insts_mask) + 1
            insts_mask = torch.nn.functional.one_hot(insts_mask)[:, :-1]
        else:
            insts_mask = torch.nn.functional.one_hot(insts_mask)

        if insts_mask.shape[1] != 0:
            insts_mask = insts_mask.T
            sp_pts_mask = torch.tensor(input_dict['sp_pts_mask'])
            sp_masks_inst = scatter_mean(
                insts_mask.float(), sp_pts_mask, dim=-1)
            sp_masks_inst = sp_masks_inst > 0.5
        else:
            sp_masks_inst = insts_mask.new_zeros(
                (0, input_dict['sp_pts_mask'].max() + 1), dtype=torch.bool)

        insts = new_idxs[1:] - 1
        if self.merge_ov:
            novel_sp_masks = input_dict['sam3d_pseudo_sp_masks']
            is_novel = torch.zeros(len(insts) + len(novel_sp_masks) + self.num_classes + 1)
            is_novel[len(insts):len(insts)+len(novel_sp_masks)] = 1
            input_dict['is_novel'] = is_novel.bool()
            sp_masks_inst = torch.cat((sp_masks_inst, novel_sp_masks), dim=0)
        
        num_stuff_cls = len(self.stuff_classes)
        
        if self.merge_non_stuff_cls:
            if self.merge_ov:
                gt_labels = insts.new_zeros(len(insts) + len(novel_sp_masks) + num_stuff_cls + 1)
            else:
                gt_labels = insts.new_zeros(len(insts) + num_stuff_cls + 1)
        else:
            if self.merge_ov:
                gt_labels = insts.new_zeros(len(insts) + len(novel_sp_masks) + self.num_classes + 1)
            else:
                gt_labels = insts.new_zeros(len(insts) + self.num_classes + 1)

        for inst in insts:
            index = pts_semantic_mask[pts_instance_mask == inst][0]
            gt_labels[inst] = index - num_stuff_cls
        
        input_dict['gt_labels_3d'] = gt_labels.numpy()

        # create gt semantic markup
        sem_mask = torch.tensor(input_dict['pts_semantic_mask'])
        sem_mask = torch.nn.functional.one_hot(sem_mask, 
                                    num_classes=self.num_classes + 1)
       
        sem_mask = sem_mask.T
        sp_pts_mask = torch.tensor(input_dict['sp_pts_mask'])
        sp_masks_seg = scatter_mean(sem_mask.float(), sp_pts_mask, dim=-1)
        sp_masks_seg = sp_masks_seg > 0.5

        sp_masks_seg[-1, sp_masks_seg.sum(axis=0) == 0] = True

        assert sp_masks_seg.sum(axis=0).max().item()
        
        if self.merge_non_stuff_cls:
            sp_masks_seg = torch.vstack((
                sp_masks_seg[:num_stuff_cls, :], 
                sp_masks_seg[num_stuff_cls:, :].sum(axis=0).unsqueeze(0)))
        
        sp_masks_all = torch.vstack((sp_masks_inst, sp_masks_seg))

        input_dict['gt_sp_masks'] = sp_masks_all.numpy()

        # create eval markup
        if 'eval_ann_info' in input_dict.keys(): 
            pts_instance_mask[pts_instance_mask != -1] += num_stuff_cls
            for idx, stuff_cls in enumerate(self.stuff_classes):
                pts_instance_mask[pts_semantic_mask == stuff_cls] = idx

            input_dict['eval_ann_info']['pts_instance_mask'] = \
                pts_instance_mask.numpy()

        return input_dict






@TRANSFORMS.register_module()
class AddSuperPointAnnotations_Part3DGlamm(BaseTransform):
    """Prepare ground truth markup for training.
    
    Required Keys:
    - pts_semantic_mask (np.float32)
    
    Added Keys:
    - gt_sp_masks (np.int64)
    
    Args:
        num_classes (int): Number of classes.
    """
    
    def __init__(self,
                 num_classes,
                 stuff_classes,
                 merge_non_stuff_cls=True,
                 merge_ov=False):
        self.num_classes = num_classes
        self.stuff_classes = stuff_classes
        self.merge_non_stuff_cls = merge_non_stuff_cls
        self.merge_ov = merge_ov

    def create_sp_masks(self, mask, sp_pts_mask):

        if torch.sum(mask == -1) != 0:
            mask[mask == -1] = torch.max(mask) + 1
            mask = torch.nn.functional.one_hot(mask)[:, :-1]
        else:
            mask = torch.nn.functional.one_hot(mask)

        if mask.shape[1] != 0:
            mask = mask.T
            sp_masks = scatter_mean(
                mask.float(), sp_pts_mask, dim=-1)
            sp_masks = sp_masks > 0.5
        else:
            sp_masks = mask.new_zeros(
                (0, sp_pts_mask.max() + 1), dtype=torch.bool)

        return sp_masks
 
    def transform(self, input_dict):
        """Private function for preparation ground truth 
        markup for training.
        
        Args:
            input_dict (dict): Result dict from loading pipeline.
        
        Returns:
            dict: results, 'gt_sp_masks' is added.
        """
        # create class mapping
        # because pts_instance_mask contains instances from non-instaces classes
        
        sp_pts_mask = torch.tensor(input_dict['sp_pts_mask'])
        num_superpoints = len(torch.unique(sp_pts_mask))
        
        # handle pts_part_mask: if None, create empty superpoint mask
        if input_dict.get('pts_part_mask') is not None:
            pts_part_mask = torch.tensor(input_dict['pts_part_mask'])
            part_idxs = torch.unique(pts_part_mask)
            assert part_idxs[0] == -1
            input_dict['pts_part_objextId'] = part_idxs[1:]
            
            part_mask = pts_part_mask.clone()
            sp_masks_part = self.create_sp_masks(part_mask, sp_pts_mask)
        else:
            # if there is no part mask, create empty superpoint mask
            sp_masks_part = torch.zeros((0, num_superpoints), dtype=torch.int64)
            input_dict['pts_part_objextId'] = torch.tensor([], dtype=torch.int64)
        
        # handle pts_object_mask: if None, create empty superpoint mask
        if input_dict.get('pts_object_mask') is not None:
            pts_object_mask = torch.tensor(input_dict['pts_object_mask'])
            object_idxs = torch.unique(pts_object_mask)
            assert object_idxs[0] == -1
            input_dict['pts_object_objextId'] = object_idxs[1:]
            
            object_mask = pts_object_mask.clone()
            sp_masks_object = self.create_sp_masks(object_mask, sp_pts_mask)
        else:
            # if there is no object mask, create empty superpoint mask
            sp_masks_object = torch.zeros((0, num_superpoints), dtype=torch.int64)
            input_dict['pts_object_objextId'] = torch.tensor([], dtype=torch.int64)

        # check whether only -1 remains after superpoint aggregation (invalid data)
        # check whether only -1 remains after superpoint aggregation (invalid data)
        if sp_masks_part.shape[0] > 0:
            part_only_invalid = sp_masks_part.sum() == 0
        else:
            part_only_invalid = True
            
        if sp_masks_object.shape[0] > 0:
            object_only_invalid = sp_masks_object.sum() == 0
        else:
            object_only_invalid = True
        
        # print('input_dict.keys: ', input_dict.keys())
        # if part_only_invalid or object_only_invalid:
        #     print(f"Warning: Invalid masks after superpoint aggregation! "
        #           f"part_only_invalid={part_only_invalid}, object_only_invalid={object_only_invalid} for frame {input_dict['scene_name']}")
        
        input_dict['gt_sp_part_masks'] = sp_masks_part.numpy()
        input_dict['gt_sp_object_masks'] = sp_masks_object.numpy()

        # merge part and object masks as gt_sp_masks
        if sp_masks_part.shape[0] > 0 and sp_masks_object.shape[0] > 0:
            gt_sp_masks = torch.vstack((sp_masks_part, sp_masks_object))
        elif sp_masks_part.shape[0] > 0:
            gt_sp_masks = sp_masks_part
        elif sp_masks_object.shape[0] > 0:
            gt_sp_masks = sp_masks_object
        else:
            # if both are empty, create an empty mask
            gt_sp_masks = torch.zeros((0, num_superpoints), dtype=torch.int64)
        input_dict['gt_sp_masks'] = gt_sp_masks.numpy()
        
        # gt_labels_3d: label per instance (0 for part, 1 for object, or distinguish another way)
        num_parts = sp_masks_part.shape[0]
        num_objects = sp_masks_object.shape[0]
        if num_parts > 0 and num_objects > 0:
            gt_labels_3d = np.concatenate([
                np.zeros(num_parts, dtype=np.int64),      # part label is 0
                np.ones(num_objects, dtype=np.int64)      # object label is 1
            ])
        elif num_parts > 0:
            gt_labels_3d = np.zeros(num_parts, dtype=np.int64)
        elif num_objects > 0:
            gt_labels_3d = np.ones(num_objects, dtype=np.int64)
        else:
            gt_labels_3d = np.array([], dtype=np.int64)
        input_dict['gt_labels_3d'] = gt_labels_3d
        
        return input_dict




@TRANSFORMS.register_module()
class SwapChairAndFloor(BaseTransform):
    """Swap two categories for ScanNet200 dataset. It is convenient for
    panoptic evaluation. After this swap first two categories are
    `stuff` and other 198 are `thing`.
    """
    def transform(self, input_dict):
        """Private function-wrapper for swap transform.

        Args:
            input_dict (dict): Result dict from loading pipeline.
        
        Returns:
            dict: Results after swap, 'pts_semantic_mask' is updated
                in the result dict.
        """
        mask = input_dict['pts_semantic_mask'].copy()
        mask[input_dict['pts_semantic_mask'] == 2] = 3
        mask[input_dict['pts_semantic_mask'] == 3] = 2
        input_dict['pts_semantic_mask'] = mask
        if 'eval_ann_info' in input_dict:
            input_dict['eval_ann_info']['pts_semantic_mask'] = mask
        return input_dict









@TRANSFORMS.register_module()
class AddSuperPointAnnotations_ScanRefer(BaseTransform):
    """Prepare ground truth markup for training.
    
    Required Keys:
    - pts_semantic_mask (np.float32)
    
    Added Keys:
    - gt_sp_masks (np.int64)
    
    Args:
        num_classes (int): Number of classes.
    """
    
    def __init__(self,
                 num_classes,
                 stuff_classes,
                 merge_non_stuff_cls=True):
        self.num_classes = num_classes
        self.stuff_classes = stuff_classes
        self.merge_non_stuff_cls = merge_non_stuff_cls

 
    def transform(self, input_dict):
        """Private function for preparation ground truth 
        markup for training.
        
        Args:
            input_dict (dict): Result dict from loading pipeline.
        
        Returns:
            dict: results, 'gt_sp_masks' is added.
        """
        # create class mapping
        # because pts_instance_mask contains instances from non-instaces classes

        pts_instance_mask = torch.tensor(input_dict['pts_instance_mask'])
        pts_semantic_mask = torch.tensor(input_dict['pts_semantic_mask'])
        
        # pts_instance_mask[pts_semantic_mask == self.num_classes] = -1
        # for stuff_cls in self.stuff_classes:
        #     pts_instance_mask[pts_semantic_mask == stuff_cls] = -1
        
        idxs = torch.unique(pts_instance_mask)
        # assert idxs[0] == -1

        # mapping = torch.zeros(torch.max(idxs) + 2, dtype=torch.long)
        # new_idxs = torch.arange(len(idxs), device=idxs.device)
        # mapping[idxs] = new_idxs - 1
        # pts_instance_mask = mapping[pts_instance_mask]
        input_dict['pts_instance_mask'] = pts_instance_mask.numpy()


        # create gt instance markup     
        insts_mask = pts_instance_mask.clone()

        # print('insts_mask: ', insts_mask)
        # print('unique_instance_ids: ', torch.unique(insts_mask))
        # print('--------------------------------')

        
        if torch.sum(insts_mask == -1) != 0:
            insts_mask[insts_mask == -1] = torch.max(insts_mask) + 1
            insts_mask = torch.nn.functional.one_hot(insts_mask)[:, :-1]
        else:
            insts_mask = torch.nn.functional.one_hot(insts_mask)

        if insts_mask.shape[1] != 0:
            insts_mask = insts_mask.T
            sp_pts_mask = torch.tensor(input_dict['sp_pts_mask'])
            sp_masks_inst = scatter_mean(
                insts_mask.float(), sp_pts_mask, dim=-1)
            sp_masks_inst = sp_masks_inst > 0.5
        else:
            sp_masks_inst = insts_mask.new_zeros(
                (0, input_dict['sp_pts_mask'].max() + 1), dtype=torch.bool)

        num_stuff_cls = len(self.stuff_classes)
        # insts = new_idxs[1:] - 1
        # # length of gt_labels_3d must equal the row count of gt_sp_masks (i.e. sp_masks_inst)
        # gt_labels = insts.new_zeros(len(insts))

        # for i, inst in enumerate(insts):
        #     index = pts_semantic_mask[pts_instance_mask == inst][0]
        #     gt_labels[i] = index - num_stuff_cls
        
        # input_dict['gt_labels_3d'] = gt_labels.numpy()

        # create gt semantic markup
        sem_mask = torch.tensor(input_dict['pts_semantic_mask'])
        # print(sem_mask.shape)
        # print(torch.unique(sem_mask))
        # print('--------------------------------')


        if torch.sum(sem_mask < 0) != 0:
            sem_mask[sem_mask < 0] = torch.max(sem_mask) + 1
            sem_mask = torch.nn.functional.one_hot(sem_mask)[:, :-1]
        else:
            sem_mask = torch.nn.functional.one_hot(sem_mask)


       
        sem_mask = sem_mask.T
        sp_pts_mask = torch.tensor(input_dict['sp_pts_mask'])
        sp_masks_seg = scatter_mean(sem_mask.float(), sp_pts_mask, dim=-1)
        sp_masks_seg = sp_masks_seg > 0.5

        sp_masks_seg[-1, sp_masks_seg.sum(axis=0) == 0] = True

        assert sp_masks_seg.sum(axis=0).max().item()
        
        if self.merge_non_stuff_cls:
            sp_masks_seg = torch.vstack((
                sp_masks_seg[:num_stuff_cls, :], 
                sp_masks_seg[num_stuff_cls:, :].sum(axis=0).unsqueeze(0)))

        
        # print('sp_masks_inst shapes: ', sp_masks_inst.shape)
        # print('--------------------------------')

        input_dict['gt_sp_masks_ins'] = sp_masks_inst
        input_dict['gt_sp_masks_sem'] = sp_masks_seg

        # sp_masks_all = torch.vstack((sp_masks_inst))
        sp_masks_all = sp_masks_inst

        input_dict['gt_sp_masks'] = sp_masks_all.numpy()

        # create eval markup
        if 'eval_ann_info' in input_dict.keys(): 
            pts_instance_mask[pts_instance_mask != -1] += num_stuff_cls
            for idx, stuff_cls in enumerate(self.stuff_classes):
                pts_instance_mask[pts_semantic_mask == stuff_cls] = idx

            input_dict['eval_ann_info']['pts_instance_mask'] = \
                pts_instance_mask.numpy()

        return input_dict




import json
@TRANSFORMS.register_module()
class QA_Generation_ScanRefer(BaseTransform):
    def __init__(self, num_qa = 10, reason3d_file=None):
        self.num_qa = num_qa
        self.reason3d_data = json.load(open(reason3d_file, 'r'))

    

    def transform(self, input_dict):

        scene_id = input_dict['scene_name']

        flat_qa_list = input_dict['qa_data']

        if len(flat_qa_list) < self.num_qa:
            flat_qa_list = flat_qa_list
        else:
            flat_qa_list = random.sample(flat_qa_list, self.num_qa)


        # here we need to align the object_id and the part_id
        gt_sp_instance_masks = torch.as_tensor(input_dict['gt_sp_masks_ins'])

        sp_pts_mask = torch.as_tensor(input_dict['sp_pts_mask'])
        num_superpoints = len(torch.unique(sp_pts_mask))
        
        
        sp_gt_seg = []
        valid_qa_list = []  # store valid QA

        for i, qa in enumerate(flat_qa_list):

            # print('qa: ', qa)
            # print('qa.keys: ', qa.keys())
            # print('--------------------------------')

            # determine whether qa is a single QA (dict) or multi-turn dialogue (list)
            qa['question'] = qa['description']+'<sentence>'
            qa['answer'] = '<p>'+qa['description']+'</p><SEG>'
            qa['qa_keys'] = 'scanrefer'
            qa['qa_level'] = 'scanrefer'
            qa['qa_key'] = 'scanrefer'
            qa['qa_task_type'] = 'scanrefer'
            qa['scene_name'] = scene_id
            object_id = int(qa['object_id'])

            # print('gt_sp_instance_masks shape: ', gt_sp_instance_masks.shape)
            # print('object_id: ', object_id)
            # print('--------------------------------')

            i_qa_seg = gt_sp_instance_masks[object_id].unsqueeze(0)

           
            if i_qa_seg.sum() == 0:
                print(f'object_id: {object_id} is not in the scene {scene_id}')
                print('object_name: ', qa['object_name'])
                print('--------------------------------')
                i_qa_seg = torch.full((i_qa_seg.shape[0], num_superpoints), float('nan'), dtype=torch.float32)
            
            sp_gt_seg.append(i_qa_seg)
            valid_qa_list.append(qa)

        # save processed mask, qa_data stays as-is (already in input_dict)
        input_dict['sp_gt_seg'] = sp_gt_seg
        input_dict['selected_qa_data'] = valid_qa_list
        return input_dict





import json
@TRANSFORMS.register_module()
class QA_Generation_ScanRefer_Test(BaseTransform):
    def __init__(self, num_qa = 10, reason3d_file=None):
        self.num_qa = num_qa
        self.reason3d_data = json.load(open(reason3d_file, 'r'))

    

    def transform(self, input_dict):

        scene_id = input_dict['scene_name']

        flat_qa_list = input_dict['qa_data']


        # here we need to align the object_id and the part_id
        gt_sp_instance_masks = torch.as_tensor(input_dict['gt_sp_masks_ins'])

        sp_pts_mask = torch.as_tensor(input_dict['sp_pts_mask'])
        num_superpoints = len(torch.unique(sp_pts_mask))
        
        
        sp_gt_seg = []
        valid_qa_list = []  # store valid QA

        for i, qa in enumerate(flat_qa_list):

            # print('qa: ', qa)
            # print('qa.keys: ', qa.keys())
            # print('--------------------------------')

            # determine whether qa is a single QA (dict) or multi-turn dialogue (list)
            qa['question'] = qa['description']+'<sentence>'
            qa['answer'] = '<p>'+qa['description']+'</p><SEG>'
            qa['qa_keys'] = 'scanrefer'
            qa['qa_level'] = 'scanrefer'
            qa['qa_key'] = 'scanrefer'
            qa['qa_task_type'] = 'scanrefer'
            qa['scene_name'] = scene_id
            object_id = int(qa['object_id'])

            # print('gt_sp_instance_masks shape: ', gt_sp_instance_masks.shape)
            # print('object_id: ', object_id)
            # print('--------------------------------')

            i_qa_seg = gt_sp_instance_masks[object_id].unsqueeze(0)

           
            if i_qa_seg.sum() == 0:
                print(f'object_id: {object_id} is not in the scene {scene_id}')
                print('object_name: ', qa['object_name'])
                print('--------------------------------')
                i_qa_seg = torch.full((i_qa_seg.shape[0], num_superpoints), float('nan'), dtype=torch.float32)
            
            sp_gt_seg.append(i_qa_seg)
            valid_qa_list.append(qa)

        # save processed mask, qa_data stays as-is (already in input_dict)
        input_dict['sp_gt_seg'] = sp_gt_seg
        input_dict['selected_qa_data'] = valid_qa_list
        return input_dict
        







@TRANSFORMS.register_module()
class AddSuperPointAnnotations_Instruct3D(BaseTransform):
    """Prepare ground truth markup for training.
    
    Required Keys:
    - pts_semantic_mask (np.float32)
    
    Added Keys:
    - gt_sp_masks (np.int64)
    
    Args:
        num_classes (int): Number of classes.
    """
    
    def __init__(self,
                 num_classes,
                 stuff_classes,
                 merge_non_stuff_cls=True):
        self.num_classes = num_classes
        self.stuff_classes = stuff_classes
        self.merge_non_stuff_cls = merge_non_stuff_cls

 
    def transform(self, input_dict):
        """Private function for preparation ground truth 
        markup for training.
        
        Args:
            input_dict (dict): Result dict from loading pipeline.
        
        Returns:
            dict: results, 'gt_sp_masks' is added.
        """
        # create class mapping
        # because pts_instance_mask contains instances from non-instaces classes

        pts_instance_mask = torch.tensor(input_dict['pts_instance_mask'])
        pts_semantic_mask = torch.tensor(input_dict['pts_semantic_mask'])
        
        pts_instance_mask[pts_semantic_mask == self.num_classes] = -1
        for stuff_cls in self.stuff_classes:
            pts_instance_mask[pts_semantic_mask == stuff_cls] = -1
        
        # assert idxs[0] == -1
        input_dict['pts_instance_mask'] = pts_instance_mask.numpy()


        # create gt instance markup     
        insts_mask = pts_instance_mask.clone()
        
        if torch.sum(insts_mask == -1) != 0:
            insts_mask[insts_mask == -1] = torch.max(insts_mask) + 1
            insts_mask = torch.nn.functional.one_hot(insts_mask)[:, :-1]
        else:
            insts_mask = torch.nn.functional.one_hot(insts_mask)

        if insts_mask.shape[1] != 0:
            insts_mask = insts_mask.T
            sp_pts_mask = torch.tensor(input_dict['sp_pts_mask'])
            sp_masks_inst = scatter_mean(
                insts_mask.float(), sp_pts_mask, dim=-1)
            sp_masks_inst = sp_masks_inst > 0.5
        else:
            sp_masks_inst = insts_mask.new_zeros(
                (0, input_dict['sp_pts_mask'].max() + 1), dtype=torch.bool)

        num_stuff_cls = len(self.stuff_classes)

        # create gt semantic markup
        sem_mask = torch.tensor(input_dict['pts_semantic_mask'])
        # print(sem_mask.shape)
        # print(torch.unique(sem_mask))
        # print('--------------------------------')


        if torch.sum(sem_mask < 0) != 0:
            sem_mask[sem_mask < 0] = torch.max(sem_mask) + 1
            sem_mask = torch.nn.functional.one_hot(sem_mask)[:, :-1]
        else:
            sem_mask = torch.nn.functional.one_hot(sem_mask)


       
        sem_mask = sem_mask.T
        sp_pts_mask = torch.tensor(input_dict['sp_pts_mask'])
        sp_masks_seg = scatter_mean(sem_mask.float(), sp_pts_mask, dim=-1)
        sp_masks_seg = sp_masks_seg > 0.5

        sp_masks_seg[-1, sp_masks_seg.sum(axis=0) == 0] = True

        assert sp_masks_seg.sum(axis=0).max().item()
    
        
        input_dict['gt_sp_masks_ins'] = sp_masks_inst
        input_dict['gt_sp_masks_sem'] = sp_masks_seg

        # sp_masks_all = torch.vstack((sp_masks_inst))
        sp_masks_all = sp_masks_inst

        input_dict['gt_sp_masks'] = sp_masks_all.numpy()

        # create eval markup
        if 'eval_ann_info' in input_dict.keys(): 
            pts_instance_mask[pts_instance_mask != -1] += num_stuff_cls
            for idx, stuff_cls in enumerate(self.stuff_classes):
                pts_instance_mask[pts_semantic_mask == stuff_cls] = idx

            input_dict['eval_ann_info']['pts_instance_mask'] = \
                pts_instance_mask.numpy()

        return input_dict






import json
@TRANSFORMS.register_module()
class QA_Generation_Instruct3D(BaseTransform):
    def __init__(self, num_qa = None, reason3d_file=None):
        self.num_qa = num_qa

    

    def transform(self, input_dict):
        scene_id = input_dict['scene_name']
        flat_qa_list = input_dict['qa_data']
        if self.num_qa is None or len(flat_qa_list) < self.num_qa:
            flat_qa_list = flat_qa_list
        else:
            flat_qa_list = random.sample(flat_qa_list, self.num_qa)

        # here we need to align the object_id and the part_id
        gt_sp_masks_ins = torch.as_tensor(input_dict['gt_sp_masks_ins'])
        num_inst_masks = int(gt_sp_masks_ins.shape[0])

        pts_instance_mask = torch.as_tensor(input_dict['pts_instance_mask'])
        instance_ids = torch.unique(pts_instance_mask)
        instance_ids_set = set(int(x) for x in instance_ids.tolist())

        sp_pts_mask = torch.as_tensor(input_dict['sp_pts_mask'])
        num_superpoints = len(torch.unique(sp_pts_mask))
        
        
        sp_gt_seg = []
        valid_qa_list = []  # store valid QA

        for i, qa in enumerate(flat_qa_list):
            # determine whether qa is a single QA (dict) or multi-turn dialogue (list)
            qa['question'] = qa['description']+'<SEG>'+'<name>'
            qa['answer'] = '<p>'+qa['object_name']+'</p><SEG>'
            qa['qa_keys'] = 'instruct3d'
            qa['qa_level'] = 'instruct3d'
            qa['qa_key'] = 'instruct3d'
            qa['qa_task_type'] = 'instruct3d'
            qa['scene_name'] = scene_id
            raw_object_ids = qa['object_id']
            if isinstance(raw_object_ids, str):
                raw_object_ids = raw_object_ids.strip()
                if raw_object_ids.startswith('['):
                    raw_object_ids = ast.literal_eval(raw_object_ids)
                else:
                    raw_object_ids = [raw_object_ids]
            elif not isinstance(raw_object_ids, (list, tuple)):
                raw_object_ids = [raw_object_ids]

            object_ids = []
            for obj in raw_object_ids:
                if isinstance(obj, (list, tuple)):
                    object_ids.extend(int(x) for x in obj)
                else:
                    object_ids.append(int(obj))
            object_ids = list(dict.fromkeys(object_ids))


            i_qa_segs = []
            for object_id in object_ids:
                if object_id not in instance_ids_set:
                    continue
                if object_id < 0 or object_id >= num_inst_masks:
                    continue
                i_qa_segs.append(gt_sp_masks_ins[object_id].unsqueeze(0))
            

            # semantic segmentation: union of multiple object instance masks gives one semantic foreground mask
            if len(i_qa_segs) == 0:
                # print(
                #     f"object_ids: {object_ids} are not in the scene {scene_id}, "
                #     f"object_name: {qa['object_name']}"
                # )
                i_qa_seg = torch.zeros((1, num_superpoints), dtype=torch.float32)
            else:
                i_qa_seg = torch.cat(i_qa_segs, dim=0).sum(dim=0, keepdim=True)

            i_qa_seg = (i_qa_seg > 0).to(torch.float32)

            sp_gt_seg.append(i_qa_seg)
            valid_qa_list.append(qa)


        # save processed mask, qa_data stays as-is (already in input_dict)
        input_dict['sp_gt_seg'] = sp_gt_seg
        input_dict['selected_qa_data'] = valid_qa_list
        return input_dict