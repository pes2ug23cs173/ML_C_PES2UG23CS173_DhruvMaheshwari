import torch

def get_entropy_of_dataset(tensor: torch.Tensor):
    """
    Calculate the entropy of the entire dataset.
    Formula: Entropy = -Σ(p_i * log2(p_i)) where p_i is the probability of class i

    Args:
        tensor (torch.Tensor): Input dataset as a tensor, where the last column is the target.

    Returns:
        float: Entropy of the dataset.
    """
    if tensor.shape[0] == 0:
        return 0.0

    target = tensor[:, -1]
    unique_classes, counts = torch.unique(target, return_counts=True)
    total_samples = target.shape[0]
    
    probabilities = counts.float() / total_samples

    probabilities = probabilities[probabilities > 0]
    
    entropy = -torch.sum(probabilities * torch.log2(probabilities))
    
    return entropy.item()

def get_avg_info_of_attribute(tensor: torch.Tensor, attribute: int):
    """
    Calculate the average information (weighted entropy) of an attribute.
    Formula: Avg_Info = Σ((|S_v|/|S|) * Entropy(S_v)) where S_v is subset with attribute value v.

    Args:
        tensor (torch.Tensor): Input dataset as a tensor.
        attribute (int): Index of the attribute column.

    Returns:
        float: Average information of the attribute.
    """
    if tensor.shape[0] == 0:
        return 0.0

    unique_values = torch.unique(tensor[:, attribute])
    avg_info = 0.0
    total_samples = tensor.shape[0]

    for value in unique_values:
        subset = tensor[tensor[:, attribute] == value]
        subset_size = subset.shape[0]
        
        subset_entropy = get_entropy_of_dataset(subset)
        
        avg_info += (subset_size / total_samples) * subset_entropy
        
    return avg_info

def get_information_gain(tensor: torch.Tensor, attribute: int):
    """
    Calculate Information Gain for an attribute.
    Formula: Information_Gain = Entropy(S) - Avg_Info(attribute)

    Args:
        tensor (torch.Tensor): Input dataset as a tensor.
        attribute (int): Index of the attribute column.

    Returns:
        float: Information gain for the attribute (rounded to 4 decimals).
    """
    if tensor.shape[0] == 0:
        return 0.0

    entropy_s = get_entropy_of_dataset(tensor)
    avg_info_attribute = get_avg_info_of_attribute(tensor, attribute)
    
    information_gain = entropy_s - avg_info_attribute
    
    return round(information_gain, 4)

def get_selected_attribute(tensor: torch.Tensor):
    """
    Select the best attribute based on highest information gain.

    Returns a tuple with:
    1. Dictionary mapping attribute indices to their information gains
    2. Index of the attribute with highest information gain
    
    Example: ({0: 0.123, 1: 0.768, 2: 1.23}, 2)

    Args:
        tensor (torch.Tensor): Input dataset as a tensor.

    Returns:
        tuple: (dict of attribute:index -> information gain, index of best attribute)
    """
    if tensor.shape[0] == 0:
        return ({}, -1)
    
    num_attributes = tensor.shape[1] - 1
    information_gains = {}
    
    best_attribute = -1
    max_gain = -1.0
    
    for attribute_idx in range(num_attributes):
        gain = get_information_gain(tensor, attribute_idx)
        information_gains[attribute_idx] = gain
        
        if gain > max_gain:
            max_gain = gain
            best_attribute = attribute_idx
            
    return (information_gains, best_attribute)